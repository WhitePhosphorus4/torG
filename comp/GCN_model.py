import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from comp.layers import GraphConvolution
# from torch_geometric.nn import GCNConv



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        '''对比实验'''
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)