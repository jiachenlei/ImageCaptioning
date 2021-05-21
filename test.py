# %%
import pickle
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.flickr8k import Flickr8kDataset
from glove import embedding_matrix_creator
from metrics import *
from utils_torch import *

from utils import initLogging, initGlobalConfig
from utils import parseArgument
from metrics import meteor_score_fn as _meteor_score_fn

import models.torch as _torch

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
    eval_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(256),  # get 256x256 crop from random location
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))
    ])

    test_set.transformations = eval_transformations

    vocab_size = len(vocab)
    print("Dataset Preparation Complete")

    return test_set, word2idx, idx2word, vocab_size


def prepareLoader(test_set, vocab_size):

    print("Prepare dataloader for training")

    # collate corresponding captions list of the image
    eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])
    
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                            collate_fn=eval_collate_fn)
    print("Dataloader Preparation Complete")

    return test_loader


def evaluate_model(data_loader, model, loss_fn, word2idx, vocab_size, bleu_score_fn, meteor_score_fn, tensor_to_word_fn, desc=''):

    running_bleu = [0.0] * 5
    running_meteor = 0.0
    running_loss = 0.0
    running_acc = 0.0

    model.eval()
    t = tqdm(iter(data_loader), desc=f'{desc}')
    with torch.no_grad():
        for batch_idx, batch in enumerate(t):
            images, captions, lengths = batch

            outputs = tensor_to_word_fn(model.sample(images, startseq_idx=word2idx['<start>']).cpu().numpy())

            for i in (1, 2, 3, 4):
                running_bleu[i] += bleu_score_fn(reference_corpus=captions, candidate_corpus=outputs, n=i)

            running_meteor += meteor_score_fn(captions, outputs)

            t.set_postfix({
                'bleu1': running_bleu[1] / (batch_idx + 1),
                'bleu2': running_bleu[2] / (batch_idx + 1),
                'bleu3': running_bleu[3] / (batch_idx + 1),
                'bleu4': running_bleu[4] / (batch_idx + 1),
                'meteor': running_meteor / (batch_idx + 1),
            }, refresh=True)
    
    for i in (1, 2, 3, 4):
        running_bleu[i] /= len(data_loader)

    running_meteor /= len(data_loader)
    running_acc /= len(_data_loader)
    running_loss /= len(_data_loader)

    return running_bleu, running_meteor


def test():
    # Dataset and DataLoader
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_set, word2idx, idx2word, vocab_size = parepareDataset(device)
    test_loader = prepareLoader(test_set, vocab_size)
    # Word embedding
    embedding_matrix = None
    # embedding_matrix = embedding_matrix_creator(embedding_dim=EMBEDDING_DIM, word2idx=word2idx)
    # Model
    Captioner = _torch.__dict__[args.model]
    model = Captioner(encoded_image_size=14, encoder_dim=2048,
                            attention_dim=ATTENTION_DIM, embed_dim=EMBEDDING_DIM, decoder_dim=DECODER_SIZE,
                            vocab_size=vocab_size,
                            embedding_matrix=embedding_matrix, train_embd=False).to(device)
    model.load_state_dict(torch.load(f"{MODEL_NAME}_{epoch}")["state_dict"])
    # Metrics
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = _test_set.pad_value).to(device)
    acc_fn = accuracy_fn(ignore_value = _test_set.pad_value)

    corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
    meteor_score_fn = _meteor_score_fn()
    tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

    print("Start testing")
    test_loss, test_acc, test_bleu, test_meteor = evaluate_model(
                            desc = f'\tTest Bleu Score: ',
                            model = model,
                            loss_fn = loss_fn,
                            word2idx = word2idx,
                            bleu_score_fn = corpus_bleu_score_fn,
                            meteor_score_fn = meteor_score_fn,
                            tensor_to_word_fn = tensor_to_word_fn,
                            data_loader = test_loader,
                            vocab_size = vocab_size) 
        
    print(f'test_loss: {test_loss} ' + \
        f'test_acc: {test_acc} ' + \
        ''.join([f'test_bleu{i}: {test_bleu[i]:.4f} ' for i in range(1, 5)]) + \
        f'test_meteor: {test_meteor}'
        )



def main():

    test()


if __name__ == "__main__":
    main()