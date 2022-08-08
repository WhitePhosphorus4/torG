import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GMKEALayer(nn.Module):
    '''Meta-kernel attention layer'''
    '''注意，meta版本开始，不同参数完全视为不同的核注意力层'''
    def __init__(self, in_features, out_features, num_node, dropout, alpha, concat, ci_l=[], pow=[], ci_s=[], beta=[], gamma=[]):
        super(GMKEALayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.alpha = alpha  # leakyrelu的激活斜率
        self.concat = concat    # 是否multihead

        self.ci_l = ci_l    # 线性核的ci
        self.pow = pow    # 线性核的pow
        self.ci_s = ci_s   # sigmoid核的ci
        self.beta = beta    # sigmoid的beta
        self.gamma = gamma    # 高斯核的gamma
        self.n_kernel_parameter = self.get_kernel_parameter_num()
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))    # 建立一个权重，用于对特征数F进行线性变换
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if not self.concat:
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))    # 计算α，输入是上一层两个输出的拼接，输出是eij
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.Mui = nn.Parameter(torch.empty(size=(sum(self.n_kernel_parameter), 1)))    # 计算μ，输入是上一层输出，输出是eij
        nn.init.xavier_uniform_(self.Mui.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)   # 激活

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if self.concat:
            e = self.generate_kernel_weight(Wh)
        else:
            e = self._self_attentional_mechanism(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)   # 对于邻接矩阵中的元素，如果大于0，则说明有新的邻接边出现，那么则使用新计算出的权值，否则表示没有变连接，使用一个默认值来表示
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)   # 做一次激活
        else:
            return h_prime

    def set_kernel_parameter(self, ci_l, pow, ci_s, beta, gamma):
        # TODO: 设置核参数
        self.ci_l = ci_l    # 线性核的ci
        self.pow = pow    # 线性核的pow
        
        self.ci_s = ci_s   # sigmoid核的ci
        self.beta = beta    # sigmoid的beta

        self.gamma = gamma    # 高斯核的gamma

    def get_kernel_parameter_num(self):
        '''获取核函数的个数'''
        return [len(self.ci_l) * len(self.pow), len(self.ci_s) * len(self.beta), len(self.gamma)]

    def generate_kernel_weight(self, Wh):
        '''生成注意力权重'''
        self.Mui.data.copy_(F.softmax(self.Mui.data, dim=0))
        # Mui = self.Mui.data
        # Mui = F.softmax(Mui, dim=0)
        # print(Mui)
        Mui = self.Mui.data
        e = torch.zeros_like(Wh[:, 1].repeat(Wh.shape[0], 1))
        index = 0
        for ci_l in self.ci_l:
            for pow in self.pow:
                t = self._polynomial_kernel_mechanism(Wh, ci_l, pow)
                kw = Mui[index, :]
                e += kw * t
                index += 1
        for ci_s in self.ci_s:
            for beta in self.beta:
                t = self._sigmoid_kernel_mechanism(Wh, ci_s, beta)
                kw = Mui[index, :]
                e += kw * t
                index += 1
        for gamma in self.gamma:
            t = self._gaussian_kernel_mechanism(Wh, gamma)
            kw = Mui[index, :]
            e += kw * t
            index += 1
        
        return e
        
    def _self_attentional_mechanism(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        return e

    def _polynomial_kernel_mechanism(self, Wh, ci_l, pow):
        '''线性核计算单元'''
        e = torch.matmul(Wh, Wh.T)
        c = torch.ones_like(e) * ci_l
        e = e + c
        e.pow(pow)
        return e

    def _signmoid_kernel_mechanism(self, Wh, ci_s, beta):
        '''sigmoid核计算单元'''
        e = torch.matmul(Wh, Wh.T)
        c = torch.ones_like(e) * ci_s
        e = beta * e + c
        e = torch.sigmoid(e)
        return e
    
    def _gaussian_kernel_mechanism(self, Wh, gamma):
        '''高斯核计算单元'''
        row_size = Wh.shape[0]
        col_size = Wh.shape[1]
        # e = torch.zeros((row_size, row_size))
        e = torch.zeros_like(Wh[:, 1].repeat(row_size, 1))
        for i in range(col_size):
            col = Wh[:,1]
            t1 = col ** 2
            A1 = t1.repeat(row_size, 1)
            B = col*col.T
            A2 = A1.T
            C = A1 + A2 - 2*B
            # e += torch.exp(-gama*C/(2*sigma**2))
            e += torch.exp(-gamma*C)
        return e
        
    def __repr__(self):
        ret = self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ') ' + 'kernels: '
        if self.n_kernel_parameter[0] != 0:
            ret += 'polynomial <- {}(c_l:{}, pow:{})'.format(self.n_kernel_parameter[0], self.ci_l, self.pow)
        if self.n_kernel_parameter[1] != 0:
            ret += 'sigmoid <- {}(c_s:{}, beta:{})'.format(self.n_kernel_parameter[1], self.ci_s, self.beta)
        if self.n_kernel_parameter[2] != 0:
            ret += 'gaussian <- {}(gamma:{})'.format(self.n_kernel_parameter[2], self.gamma)
        return ret


class GMKEA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nnode, dropout, alpha, nheads, ci_l, pow, ci_s, beta, gamma):
        """Dense version of GAT."""
        super(GMKEA, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList()
        for _ in range(nheads):
            self.attentions.append(GMKEALayer(nfeat, nhid, nnode, dropout=dropout, alpha=alpha, concat=True, ci_l=ci_l, pow=pow, ci_s=ci_s, beta=beta, gamma=gamma))
        print(self.attentions)

        self.out_att = GMKEALayer(nhid * nheads, nclass, nnode, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)