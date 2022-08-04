import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NetGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, alpha=0.6, pow=1):
        super(NetGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pow = pow
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
        K_adj = adj ** self.pow
        support = torch.mm(input, self.weight)
        output = torch.spmm(K_adj, support)
        if self.bias is not None:
            output = output + self.bias

        return F.elu(output)
        # return self.leakyrelu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha):
        '''NGCN'''
        super(NGCN, self).__init__()

        self.net_module = nn.ModuleList()
        for i in range(1, K+1):
            self.net_module.append(NetGraphConvolution(nfeat, nhid, bias=True, alpha=alpha, pow=i))

        self.generate_module = nn.Linear(K*nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.cat([net(x, adj) for net in self.net_module], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.generate_module(x))
        return F.log_softmax(x, dim=1)