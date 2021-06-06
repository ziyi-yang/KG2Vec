import numpy as np
import os
from tqdm import tqdm
import scipy.sparse as sp
import torch
import json
import nltk
import urllib
import urllib.parse

# stop_words list
sw_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def build_graph(word2id, emb, json_folder, json_oov):
    """Build the adjacency matrix, node features and ids dictionary."""
    content_list = []
    word2node = {}
    feats_all = []
    wv_all = []
    for num, filename in enumerate(os.listdir(json_folder)):
        if filename.endswith(".json"):
            pretrained_word = filename[: -5]
            if pretrained_word in word2id:
                wv = emb[word2id[pretrained_word], :]
            elif pretrained_word.lower() in word2id:
                wv = emb[word2id[pretrained_word.lower()], :]
            else:
                continue
            content, feats = \
            wiki2graph(os.path.join(json_folder, filename), word2node, word2id, emb, pretrained_word)
            content_list.append(content)
            feats_all.append(feats)
            wv_all.append(wv)

    train_num = len(content_list)

    for filename in os.listdir(json_oov):
        if filename.endswith(".json"):
            oov_name = filename[: -5]
            content, feats = \
            wiki2graph(os.path.join(json_oov, filename), word2node, word2id, emb, oov_name)
            content_list.append(content)
            feats_all.append(feats)

    num_nodes = len(content_list)
    print ("Generating graph edges.")
    edges, edges_val = generate_edges(content_list)
    edges = np.asarray(edges)
    edges_val = np.asarray(edges_val)

    adj = sp.coo_matrix((edges_val, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(feats_all))
    wv_all = torch.FloatTensor(np.array(wv_all))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(range(train_num))
    return adj, features, wv_all, word2node, idx_train

def generate_edges(content_list):
    edges = []
    edges_val = []
    for index_1, content_1 in enumerate(content_list[:-1]):
        for index_2, content_2 in enumerate(content_list[index_1 + 1:]):
            if not set(content_1).isdisjoint(content_2):
                edges.append([index_1, index_1 + index_2 + 1])
                s1 = set(content_1)
                s2 = set(content_2)
                intes = len(s1.intersection(s2)) + 0.0
                union = len(s1.union(s2))
                if intes/union < 0.5:
                    val = 0
                else:
                    val = intes/union
                edges_val.append(val)
    return edges, edges_val

def wiki2graph(json_file, word2node, word2id, emb, oov_name):
    with open(json_file) as data_file:
        data = json.load(data_file)
    if oov_name not in word2node:
        word2node[oov_name] = len(word2node)

    descp = ""
    if "wikipedia" in data:
        descp = descp + " " + data["wikipedia"]
    if "wiki_def" in data:
        descp = descp + " " + data["wiki_def"]

    wv_id = []
    content = []
    for word in nltk.word_tokenize(descp):
        if word.isalpha() and word.lower() not in sw_en:
            content.append(word.lower())
            if word.lower() in word2id:
                wv_id.append(word2id[word.lower()])
    if not wv_id:
        feats = np.zeros(300,)
    else:
        feats = np.mean(emb[wv_id, :], axis = 0)
    return (content, feats)
