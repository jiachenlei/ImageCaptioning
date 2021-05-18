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

import logging
import parser
from utils import initLogging, initGlobalConfig
from utils import parseArgument
from metrics import meteor_score_fn as _meteor_score_fn

import models.torch as _torch

args = parseArgument()
# Set logger
initLogging(args.name, args.model, args.resume_e)
logger = logging.getLogger(__name__)
logger.info("Start Initialization")
# Set global value according to configuration and process command-line arguments
config = initGlobalConfig()

DATASET_BASE_PATH = config["DATASET_BASE_PATH"]
EMBEDDING_DIM = config["EMBEDDING_DIM"]
EMBEDDING = f"{EMBEDDING_DIM}"
ATTENTION_DIM = config["ATTENTION_DIM"]
DECODER_SIZE = config["DECODER_SIZE"]
BATCH_SIZE = config["BATCH_SIZE"]
LOG_INTERVAL = 25 * (256 // BATCH_SIZE)
LR = config["LR"]
NUM_EPOCHS = config["NUM_EPOCHS"]

if args.resume_e != -1:
    logger.info(f"Resume training from epoch {args.resume_e}, " + \
                f"Overall number of epochs is set to {NUM_EPOCHS - args.resume_e}, " + \
                f"Start from {START_EPOCH}th epoch")
    START_EPOCH = NUM_EPOCHS - args.resume_e
else:
    START_EPOCH = 0

SAVE_FREQ = config["SAVE_FREQ"]
MODEL_NAME = f'saved_models/{args.name}_m{args.model}_b{BATCH_SIZE}_emd{EMBEDDING}'


def parepareDataset(device):

    logger.info("Prepare dataset for training")

    train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                                return_type='tensor',
                                load_img_to_memory=False)
    vocab, word2idx, idx2word, max_len = vocab_set = train_set.get_vocab()

    val_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', vocab_set=vocab_set, device=device,
                            return_type='corpus',
                            load_img_to_memory=False)
    test_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', vocab_set=vocab_set, device=device,
                            return_type='corpus',
                            load_img_to_memory=False)
    train_eval_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', vocab_set=vocab_set, device=device,
                                    return_type='corpus',
                                    load_img_to_memory=False)

    # Dataset Transformation
    train_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(256),  # get 256x256 crop from random location
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))
    ])
    eval_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(256),  # get 256x256 crop from random location
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                            (0.229, 0.224, 0.225))
    ])

    train_set.transformations = train_transformations
    val_set.transformations = eval_transformations
    test_set.transformations = eval_transformations
    train_eval_set.transformations = eval_transformations

    vocab_size = len(vocab)
    logger.info("Dataset Preparation Complete")

    return train_set, val_set, test_set, train_eval_set, word2idx, idx2word, vocab_size


def prepareLoader(train_set, val_set, test_set, train_eval_set, vocab_size):

    logger.info("Prepare dataloader for training")
    # collate corresponding captions list of the image
    eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                            collate_fn=eval_collate_fn)
    
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                            collate_fn=eval_collate_fn)
    train_eval_loader = DataLoader(train_eval_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                                collate_fn=eval_collate_fn)

    logger.info("Dataloader Preparation Complete")
    return train_loader, val_loader, test_loader, train_eval_loader


def train_step(train_loader, model, loss_fn, optimizer, vocab_size, acc_fn, desc=''):
    running_acc = 0.0
    running_loss = 0.0
    model.train()
    t = tqdm(iter(train_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch

        optimizer.zero_grad()

        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, lengths)

        # Since decoding starts with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        running_acc += (torch.argmax(scores, dim=1) == targets).sum().float().item() / targets.size(0)
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1),
                       }, refresh=True)
    
    # record the final loss and accuracy
    logger.info(f'{desc} ' + \
                f'train_loss: {running_loss / len(train_loader):.4f} ' + \
                f'train_acc: {running_acc / len(train_loader):.4f}')

    return running_loss / len(train_loader)


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
                'bleu4': running_bleu[4] / (batch_idx + 1),
                'meteor': running_meteor / (batch_idx + 1),
            }, refresh=True)
    
    for i in (1, 2, 3, 4):
        running_bleu[i] /= len(data_loader)

    running_meteor /= len(data_loader)

    return running_bleu, running_meteor


def train():
    # Dataset and DataLoader
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, train_eval_set, word2idx, idx2word, vocab_size = parepareDataset(device)
    train_loader, val_loader, test_loader, train_eval_loader = prepareLoader(train_set, val_set, test_set, train_eval_set, vocab_size)
    # Word embedding
    # we do not use pretrained embedding in this experiment
    embedding_matrix = None
    # embedding_matrix = embedding_matrix_creator(embedding_dim=EMBEDDING_DIM, word2idx=word2idx)
    # Model
    Captioner = _torch.__dict__[args.model]
    final_model = Captioner(encoded_image_size=14, encoder_dim=2048,
                            attention_dim=ATTENTION_DIM, embed_dim=EMBEDDING_DIM, decoder_dim=DECODER_SIZE,
                            vocab_size=vocab_size,
                            embedding_matrix=embedding_matrix, train_embd=False).to(device)

    if args.resume_e != -1:
        logger.info(f"Loading model checkpoint file {MODEL_NAME}_{args.resume_e}.pt")
        try:
            final_model.load_state_dict(torch.load(f"{MODEL_NAME}_{args.resume_e}.pt")["state_dict"])
        except RuntimeError as e:
            print(e + \
                  "\nThis may caused by the corrupted model file"  )

    # Metrics
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_set.pad_value).to(device)
    acc_fn = accuracy_fn(ignore_value=train_set.pad_value)
    sentence_bleu_score_fn = bleu_score_fn(4, 'sentence')
    corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
    meteor_score_fn = _meteor_score_fn()
    tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)
    # Optimizer
    optimizer = torch.optim.RMSprop(params=final_model.parameters(), lr=LR)

    logger.info("Start training")

    train_loss_min = 100
    val_bleu4_max = 0.0
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        train_loss = train_step(desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}] ', model=final_model,
                                optimizer=optimizer, loss_fn=loss_fn, acc_fn=acc_fn,
                                train_loader=train_loader, vocab_size=vocab_size)
        with torch.no_grad():
            train_bleu, train_meteor = evaluate_model(desc=f'\tTrain Bleu Score: ', model=final_model,
                                        loss_fn=loss_fn, word2idx = word2idx,
                                        bleu_score_fn=corpus_bleu_score_fn,
                                        meteor_score_fn = meteor_score_fn,
                                        tensor_to_word_fn=tensor_to_word_fn,
                                        data_loader=train_eval_loader,
                                        vocab_size=vocab_size)
            val_bleu, val_meteor = evaluate_model(desc=f'\tValidation Bleu Score: ', model=final_model,
                                    loss_fn=loss_fn, word2idx = word2idx,
                                    bleu_score_fn=corpus_bleu_score_fn,
                                    meteor_score_fn = meteor_score_fn,
                                    tensor_to_word_fn=tensor_to_word_fn,
                                    data_loader=val_loader,
                                    vocab_size=vocab_size) 

            logger.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] ' + \
                        ''.join([f'train_bleu{i}: {train_bleu[i]:.4f} ' for i in range(1, 5)]) + \
                        f'train_meteor: {train_meteor}' + \
                        ''.join([f'val_bleu{i}: {val_bleu[i]:.4f} ' for i in range(1, 5)]) + \
                        f'val_meteor: {val_meteor}'
            )

            state = {
                'epoch': epoch + 1,
                'state_dict': final_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss_latest': train_loss,
                'val_bleu4_latest': val_bleu[4],
                'train_loss_min': min(train_loss, train_loss_min),
                'val_bleu4_max': max(val_bleu[4], val_bleu4_max),
                'train_meteor': train_meteor,
                'val_meteor': val_meteor,
                'train_bleus': train_bleu,
                'val_bleus': val_bleu,
            }
            torch.save(state, f'{MODEL_NAME}_{epoch}.pt')
            if train_loss < train_loss_min:
                train_loss_min = train_loss
                torch.save(state, f'{MODEL_NAME}''_best_train.pt')
            if val_bleu[4] > val_bleu4_max:
                val_bleu4_max = val_bleu[4]
                torch.save(state, f'{MODEL_NAME}''_best_val.pt')

    torch.save(state, f'{MODEL_NAME}_ep{NUM_EPOCHS:02d}_final.pt')

    return final_model


def test(model):

    logger.info("Start testing specified model")
    with torch.no_grad():
        model.eval()
        test_bleu, test_meteor = evaluate_model(desc=f'Test: ', model=final_model,
                                loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                                tensor_to_word_fn=tensor_to_word_fn,
                                data_loader=test_loader, _data_loader=None,
                                vocab_size=vocab_size)

    for setname, result in test_bleu:
        print(setname, end=' ')
        for ngram in (1, 2, 3, 4):
            logger.info(f'Bleu-{ngram}: {result[ngram]}')

    logger.info(f'Meteor: {test_meteor}')


def main():

    model = train()

    # test(model)


if __name__ == "__main__":
    main()

# # %%
# t_i = 1003
# dset = train_set
# im, cp, _ = dset[t_i]
# print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0), word2idx['<start>'])[0]]))
# print(dset.get_image_captions(t_i)[1])

# plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
# t_i = 500
# dset = val_set
# im, cp, _ = dset[t_i]
# print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0), word2idx['<start>'])[0]]))
# print(cp)

# plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# # %%
# t_i = 500
# dset = test_set
# im, cp, _ = dset[t_i]
# print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0), word2idx['<start>'])[0]]))
# print(cp)

# plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

#