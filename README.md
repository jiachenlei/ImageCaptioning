# Image-Captioning-PyTorch
This repo contains codes to preprocess, train and evaluate sequence models on Flickr8k Image dataset in pytorch. This repo is a fork of  "https://github.com/Subangkar/Image-Captioning-Attention-PyTorch".  
  
Pretrained Resnet50 Resnext50 Res2net50 Inception-v3 and Res2next & LSTM with attention were added. And part of the code arcitecture was modified

**Pre-requisites**:
 - Datasets:
    - Flickr8k Dataset: [images](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and [annotations](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
 - Pre-trained word embeddings:
    - [Glove Embeddings of 6B words](http://nlp.stanford.edu/data/glove.6B.zip)

**Data Folder Structure for training using [`train_torch.py`](train_torch.py) or [`train_attntn.py`](train_attntn.py):**
**Train model with pretrained backbone by:**
```
python train_attntn.py --name 0503 --model resnet101_attention
```
where name represents part of the name of the model (other parts includes: model,e.g. resnet101, hidden dimension, e.g. 300 ,etc)
```
data/
    flickr8k/
        Flicker8k_Dataset/
            *.jpg
        Flickr8k_text/
            Flickr8k.token.txt
            Flickr_8k.devImages.txt
            Flickr_8k.testImages.txt
            Flickr_8k.trainImages.txt
    glove.6B/
        glove.6B.50d.txt
        glove.6B.100d.txt
        glove.6B.200d.txt
        glove.6B.300d.txt
```
