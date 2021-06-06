import time
import argparse
import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn

from utils_wiki import build_graph
from models import GCN
from numpy import linalg as LA
from scipy.stats import pearsonr, spearmanr
import json

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=200,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='card',
                    help='which dataset to test')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--wv', type=str, default='glove',
                    help='Word vectors type, ConceptNet (con) or GloVe (glove)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def train(model, optimizer, epoch, idx_train, adj, batch_sz = 32):
    t = time.time()
    model.train()
    mse_loss = nn.MSELoss(reduction = "mean")
    permutation = torch.randperm(idx_train.size()[0])
    loss = 0
    for i in range(0, idx_train.size()[0], batch_sz):
        indices = permutation[i: (i + batch_sz)]
        output = model(features, adj)
        optimizer.zero_grad()
        loss_train = mse_loss(output[indices, :], word_vector[indices, :])
        loss_train.backward()
        optimizer.step()
        loss += loss_train.item()

    print('Epoch: {:04d}'.format(epoch+1),
          'Training Loss: {:.4f}'.format(loss),
          'Time: {:.4f}s'.format(time.time() - t))

def test(model, adj, features, word2node, word2id, emb, dataset):
    model.eval()
    output = model(features, adj).detach().cpu().numpy()
    if dataset == "rw":
        rw_path = "test_data/rw/rw.txt"
        print ("Testing on SRW.")
    else:
        rw_path = "test_data/card-660/dataset.tsv"
        print ("Testing on card-660.")

    predict = []
    gold = []
    cnt = 0
    covered_cnt = 0
    with open(rw_path, "rb") as f:
        for line in f:
            line_t = line.decode().split('\t')
            w1 = line_t[0]
            w2 = line_t[1]
            gold.append(float(line_t[2]))
            if w1 in word2id:
                v1 = emb[word2id[w1], :]
                covered_cnt += 1
            elif w1 in word2node:
                cnt += 1
                v1 = output[word2node[w1], :]
            else:
                # if still an OOV, use the zero vector as the embedding
                v1 = np.zeros(300, )

            if w2 in word2id:
                v2 = emb[word2id[w2], :]
                covered_cnt += 1
            elif w2 in word2node:
                cnt += 1
                v2 = output[word2node[w2], :]
            else:
                v2 = np.zeros(300, )

            if LA.norm(v1) >= 1e-5:
                v1 /= LA.norm(v1, 2)
            if LA.norm(v2) >= 1e-5:
                v2 /= LA.norm(v2, 2)
            predict.append(v1.dot(v2))
    print ("kb2vec covers ", cnt)
    print ("glove missed ", 660*2 - covered_cnt)

    p = 100 * pearsonr(predict, gold)[0]
    s = 100 * spearmanr(predict, gold)[0]
    print ("Pearson: ", p)
    print ("spearman: ", s)
    return p, s


def show_params(args):
    print("lr: ", args.lr)
    print("epochs: ", args.epochs)
    print("hidden size: ", args.hidden)
    print("batch size: ", args.batch_size)
    if args.wv == "con":
        wv_type = "ConceptNet"
    elif args.wv == "glove":
        wv_type = "GloVe"
    else:
        raise Exception(f"Unsupported word vector type: {args.wv}")
    print("wv type: ", wv_type)

if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print("Loading word vectors...")
    show_params(args)
    if args.wv == "glove":
        word2id = np.load("train_data/word2id_glove.npy", encoding = 'latin1', allow_pickle=True).item()
        emb = np.load("train_data/emb_glove.npy", encoding = 'latin1', allow_pickle=True)

    if args.wv == "con":
        word2id = np.load("train_data/word2id_ConNet.npy", encoding = 'latin1', allow_pickle=True).item()
        emb = np.load("train_data/emb_ConNet.npy", encoding = 'latin1', allow_pickle=True)
    print ("Word vectors loaded.")

    ds_test = args.dataset

    json_oov = os.path.join("train_data", f"wiki_{ds_test}_{args.wv}")
    json_folder = os.path.join("train_data", f"wiki_saved_{args.wv}")

    # Build a word graph using train + OOV json
    adj, features, word_vector, word2node, idx_train = build_graph(
        word2id, emb, json_folder, json_oov)

    # Set up the GCN Model and the optimizer
    model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        word_vector = word_vector.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, idx_train, adj, args.batch_size)

    print(f"Training Finished! Training time : {time.time() - t_total}s")

    test(model, adj, features, word2node, word2id, emb, ds_test)
