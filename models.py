import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nfeat)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = torch.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        return x
