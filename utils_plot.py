import itertools

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, cm as cm
import skimage
import skimage.transform
from utils import initGlobalConfig
from utils import parseArgument

from datasets.flickr8k import Flickr8kDataset
from models import torch as _torch
import torch
from torchvision import transforms

args = parseArgument(mode = 'test')
# Set global value according to configuration and process command-line arguments
config = initGlobalConfig()

DATASET_BASE_PATH = config["DATASET_BASE_PATH"]
EMBEDDING_DIM = config["EMBEDDING_DIM"]
EMBEDDING = f"{EMBEDDING_DIM}"
ATTENTION_DIM = config["ATTENTION_DIM"]
DECODER_SIZE = config["DECODER_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]

MODEL_NAME = f'saved_models/{args.name}_m{args.model}_b{BATCH_SIZE}_emd{EMBEDDING}'


def parepareDataset(device):

    print("Prepare dataset for testing")
    train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                                return_type='tensor',
                                load_img_to_memory=False)
    vocab, word2idx, idx2word, max_len = vocab_set = train_set.get_vocab()

    test_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', vocab_set=vocab_set, device=device,
                            return_type='corpus',
                            load_img_to_memory=False)
    
    # Dataset Transformation
#     eval_transformations = transforms.Compose([
#         transforms.Resize(256),  # smaller edge of image resized to 256
#         transforms.CenterCrop(256),  # get 256x256 crop from random location
#         transforms.ToTensor(),  # convert the PIL Image to a tensor
#         transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
#                             (0.229, 0.224, 0.225))
#     ])

#     test_set.transformations = eval_transformations

    vocab_size = len(vocab)
    print("Dataset Preparation Complete")

    return test_set, word2idx, idx2word, vocab_size


def visualize_att(image, seq, alphas, idx2word, save_path, endseq='<end>', smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param idx2word: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = transforms.ToPILImage()(image)

    # words = [idx2word[ind] for ind in seq]
    words = list(itertools.takewhile(lambda word: word != endseq,
                                     map(lambda idx: idx2word[idx], iter(seq))))
    plt.figure()
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().detach().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().detach().numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig(save_path)

    
def _main():
    # Dataset and DataLoader
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_set, word2idx, idx2word, vocab_size = parepareDataset(device)
    embedding_matrix = None
    # Model
    Captioner = _torch.__dict__[args.model]
    model = Captioner(encoded_image_size=14, encoder_dim=2048,
                            attention_dim=ATTENTION_DIM, embed_dim=EMBEDDING_DIM, decoder_dim=DECODER_SIZE,
                            vocab_size=vocab_size,
                            pretrained = False,
                            embedding_matrix=embedding_matrix, train_embd=False).to(device)
    model.load_state_dict(torch.load(f"{MODEL_NAME}_best_val.pt")["state_dict"])
    model.eval()
    # Metrics
    print("Start generating plots")
    _path = f"./visualization/{args.name}_{args.model}"
    for idx, batch in enumerate(test_set):
        image, caption, length = batch
        image = image.unsqueeze(0)
    
        # [b, max_len] [b, max_len, patch_size, patch_size]
        sampled_ids, alphas = model.sample(image, word2idx["<start>"], return_alpha = True)
        # [max_len]
        sampled_ids = sampled_ids.squeeze(0)
        # [max_len, patch_size, patch_size]
        alphas = alphas.squeeze(0)
        cap_idx = []
        cap_idx = [c for c in sampled_ids.tolist() if c != word2idx["<end>"] and c != word2idx["."]]
        
        image = image.squeeze(0)
        visualize_att(image, cap_idx, alphas, idx2word, _path + f"{idx}.png")
        
        

if __name__ == "__main__":
    _main()
