import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, alpha=0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)

        self.relu = nn.ReLU(alpha)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GCBNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha):
        '''GCBNet'''
        super(GCBNet, self).__init__()

        self.baseGCN_module = GraphConvolution(nfeat, nhid, bias=True, alpha=alpha)

        self.net_module = nn.ModuleList()
        for _ in range(K):
            self.net_module.append(MLP(nfeat, nhid, nclass, dropout, alpha))

        self.generate_input = nhid + nclass * K
        self.generate_module = nn.Linear(self.generate_input, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        g = F.relu(self.baseGCN_module(x, adj))
        g = F.dropout(g, self.dropout, training=self.training)
        m = torch.cat([net(x) for net in self.net_module], dim=1)

        z = torch.cat([g, m], dim=1)

        x = self.generate_module(z)
        return F.log_softmax(x, dim=1)