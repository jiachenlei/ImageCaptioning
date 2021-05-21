import glob
import io
import ntpath
import os

import nltk
nltk.data.path.append("/mnt/traffic/leijiachen/data/nltk_data/")

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils_torch import split_data


class Flickr8kDataset(Dataset):
    """
    create a dataset for Flickr8k
    :param dataset_base_path: path of directory that contains Flicker8k_Dataset and Flickr8k_text
    :param vocab_set: a tuple type vocabulary set that contains vocabulary, word2idx, idx2word, max_len
    :param dist: indicates either it's dataset for training or validation or testing
    :param startseq: word that indicates start of a sentence
    :param endseq: word that indicates end of a sentence
    :param unkseq: word that indicates unknown word in a sentence
    :param padseq: word that indicates padding symbol in a sentence
    :param transformations: transformations to be implemented on dataset
    :param return_raw: 
    :param load_img_to_memory: whether load image to memory ahead of time or not
    :param return_type: return tensor type for training or corpus type for validation and test
    :param device: cuda or cpu
    :return: a torch.utils.data.Dataset object
    """

    def __init__(self, dataset_base_path='../../data/flickr8k/',
                 vocab_set=None, dist='val',
                 startseq="<start>", endseq="<end>", unkseq="<unk>", padseq="<pad>",
                 transformations=None,
                 return_raw=False,
                 load_img_to_memory=False,
                 return_type='tensor',
                 device=torch.device('cpu')):
        # e.g. 
        # 1305564994_00513f9a5b.jpg#0 A man in street racer armor be examine the tire of another racer 's motorbike .
        # 1000268201_693b08cb0e.jpg#1 A girl going into a wooden building .
        # ...
        self.token = dataset_base_path + 'Flickr8k_text/Flickr8k.token.txt'
        # path of the image files
        self.images_path = dataset_base_path + 'Flicker8k_Dataset/'
        # path of trainning set/validation set/test set
        self.dist_list = {
            'train': dataset_base_path + 'Flickr8k_text/Flickr_8k.trainImages.txt',
            'val': dataset_base_path + 'Flickr8k_text/Flickr_8k.devImages.txt',
            'test': dataset_base_path + 'Flickr8k_text/Flickr_8k.testImages.txt'
        }
        # whether load image to memory ahead of time or not (before __getitem__ is called)
        # pil_d is a dict stores imagename -> image bit file
        self.load_img_to_memory = load_img_to_memory
        self.pil_d = None

        self.device = torch.device(device)
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch

        self.return_raw = return_raw
        self.return_type = return_type
        # used in __getitem__
        self.__get_item__fn = self.__getitem__corpus if return_type == 'corpus' else self.__getitem__tensor
        # list all the paths for all images in directory image_path
        self.imgpath_list = glob.glob(self.images_path + '*.jpg')
        # get a dict object that maps image names in dist to corresponding captions
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgname_to_caplist = self.__get_imgname_to_caplist_dict(self.__get_imgpath_list(dist=dist))

        self.transformations = transformations if transformations is not None else transforms.Compose([
            transforms.ToTensor()
        ])

        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()

        if vocab_set is None:
            self.vocab, self.word2idx, self.idx2word, self.max_len = self.__construct_vocab()
        else:
            self.vocab, self.word2idx, self.idx2word, self.max_len = vocab_set
        self.db = self.get_db()

    def __all_imgname_to_caplist_dict(self):
        """
        get a dict object that maps all image names (including training set/validation set/test set) 
        to corresponding captions
        :param
        :return: a dict object
        """
        captions = open(self.token, 'r').read().strip().split('\n')
        imgname_to_caplist = {}
        for i, row in enumerate(captions):
            row = row.split('\t')
            row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
            if row[0] in imgname_to_caplist:
                imgname_to_caplist[row[0]].append(row[1])
            else:
                imgname_to_caplist[row[0]] = [row[1]]
        return imgname_to_caplist

    def __get_imgname_to_caplist_dict(self, img_path_list):
        """
        get a dict object that maps all image names in img_path_list
        to corresponding captions. This function clean image files and tokens that do not match
        :param: img_path_list: list object that contains path of images in trainning set or validation set or test set
        :return: a dict object
        """
        d = {}
        for i in img_path_list:
            if i[len(self.images_path):] in self.all_imgname_to_caplist:
                d[ntpath.basename(i)] = self.all_imgname_to_caplist[i[len(self.images_path):]]
        return d

    def __get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, img=self.imgpath_list, images=self.images_path)
        return dist_imgpathlist

    def __construct_vocab(self):
        """
        construct a vocabulary with the dataset
        :param: 
        :return: a tuple object that contains vocabulary(list), word2idx(dict), idx2word(dict), max_len(int)
        """
        words = [self.startseq, self.endseq, self.unkseq, self.padseq]
        max_len = 0
        for _, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                cap_words = nltk.word_tokenize(cap.lower())
                words.extend(cap_words)
                max_len = max(max_len, len(cap_words) + 2)  # determine the maximum length of caption
        vocab = sorted(list(set(words)))

        word2idx = {word: index for index, word in enumerate(vocab)}
        idx2word = {index: word for index, word in enumerate(vocab)}

        return vocab, word2idx, idx2word, max_len

    def get_vocab(self):
        return self.vocab, self.word2idx, self.idx2word, self.max_len

    def get_db(self):
        """
        construct a database with the dataset
        :param: 
        :return: a list object that contains list of imagename, tokenized caption list, length of each caption in caption list
                    or a numpy.ndarray object that formats like token file: imagefile, caption, length of caption
        """
        # load images to memory ahead of time (before __getitem__ is called)
        if self.load_img_to_memory:
            self.pil_d = {}
            for imgname in self.imgname_to_caplist.keys():
                self.pil_d[imgname] = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')

        if self.return_type == 'corpus':
            df = []
            for imgname, caplist in self.imgname_to_caplist.items():
                cap_wordlist = []
                cap_lenlist = []
                for caption in caplist:
                    toks = nltk.word_tokenize(caption.lower())
                    cap_wordlist.append(toks)
                    cap_lenlist.append(len(toks))
                df.append([imgname, cap_wordlist, cap_lenlist])
            return df

        # ----- Forming a df to sample from ------
        l = ["image_id\tcaption\tcaption_length\n"]

        for imgname, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                l.append(
                    f"{imgname}\t"
                    f"{cap.lower()}\t"
                    f"{len(nltk.word_tokenize(cap.lower()))}\n")
        img_id_cap_str = ''.join(l)

        df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t')
        return df.to_numpy()

    @property
    def pad_value(self):
        # return 0
        return self.word2idx[self.padseq]

    def __getitem__(self, index: int):
        return self.__get_item__fn(index)

    def __len__(self):
        return len(self.db)

    def get_image_captions(self, index: int):
        """
        :param index: [] index
        :returns: image_path, list_of_captions
        """
        imgname = self.db[index][0]
        return os.path.join(self.images_path, imgname), self.imgname_to_caplist[imgname]

    def __getitem__tensor(self, index: int):
        imgname = self.db[index][0]
        caption = self.db[index][1]
        capt_ln = self.db[index][2]
        cap_toks = [self.startseq] + nltk.word_tokenize(caption) + [self.endseq]
        img_tens = self.pil_d[imgname] if self.load_img_to_memory else Image.open(
            os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens).to(self.device)
        cap_tens = self.torch.LongTensor(self.max_len).fill_(self.pad_value)
        cap_tens[:len(cap_toks)] = self.torch.LongTensor([self.word2idx[word] for word in cap_toks])
        return img_tens, cap_tens, len(cap_toks)

    def __getitem__corpus(self, index: int):
        imgname = self.db[index][0]
        cap_wordlist = self.db[index][1]
        cap_lenlist = self.db[index][2]
        img_tens = self.pil_d[imgname] if self.load_img_to_memory else Image.open(
            os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens).to(self.device)
        return img_tens, cap_wordlist, cap_lenlist
