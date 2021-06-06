# Embedding Imputation with Grounded Language Information (KG2Vec)

The official implementations for the ACL 2019 paper
[Embedding Imputation with Grounded Language Information](https://www.aclweb.org/anthology/P19-1326/).

[__Ziyi Yang__](https://web.stanford.edu/~zy99/), __Chenguang Zhu__, __Vin Sachidananda__, __Eric Darve__

-------------------------------------------------------------------------------------
In this paper, we propose the model **KG2Vec** (Konlwledge Graph To Vector) to solve the Out-Of-Vocabulary (OOV) problem. KG2Vec models with **Graph Convolutional Networks** and leverages grounded information in the form of a knowledge graph. This is in contrast to existing approaches which typically make use of vector space properties or subword information. We propose an online method to construct a graph from grounded information and design an algorithm to map from the resulting graphical structure to the space of the pre-trained embeddings. Finally, we evaluate our approach on a range of rare and unseen word tasks across various domains and show that our model can learn better representations. For example, on the Card-660 task our method improves Pearson’s and Spearman’s correlation coefficients upon the state-of-the-art by 11% and 17.8% respectively using GloVe embeddings.

## Dependencies

* Python 3.7
* PyTorch 1.7.1
* Numpy 1.17.0
* Scipy 1.4.1

## Instructions for Run KG2Vec on the Stanford Rare Word dataset and the Cambridge Card-660 dataset

1. First download training data, including pretrained word vectors and preprocessed word definitions (features), from [here](https://drive.google.com/file/d/1a_Chiuvt4phYuyD7hSU3cD-C4N-3saDo/view?usp=sharing). Put all the files to the folder ```train_data```.

2. Results reproduction.
For example, to reproduce KG2Vec's performance on the Cambridge Card-660 dataset using ConceptNet embeddings, run:
```zsh
python3 train_wiki.py --epochs 250 --lr 0.00075 --hidden 400 --dataset card --batch_size 400 --wv con
```
To reproduce KG2Vec's performance on the Cambridge Card-660 dataset using GloVe embeddings, run:
```zsh
python3 train_wiki.py --epochs 300 --lr 0.001 --hidden 400 --dataset card --batch_size 256 --wv con
```

To test on Stanford Rare Word dataset, change the argument ```dataset``` to ```rw``` (tuning with hyperparameters e.g., hidden size, learning rate and number of epochs, may be needed).

## Cite KG2Vec
If you find KG2Vec useful for you research, please cite our paper:
```bib
@inproceedings{yang-etal-2019-embedding,
    title = "Embedding Imputation with Grounded Language Information",
    author = "Yang, Ziyi  and
      Zhu, Chenguang  and
      Sachidananda, Vin  and
      Darve, Eric",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1326",
    doi = "10.18653/v1/P19-1326",
    pages = "3356--3361",
}
```
