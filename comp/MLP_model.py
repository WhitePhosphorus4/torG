import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(nfeat, 128)
        # self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, nclass)
        self.relu = nn.LeakyReLU(alpha)
        # self.relu = nn.ReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout(self.relu(x))
        # x = self.fc2(x)
        x = self.dropout(self.relu(x))
        x = self.fc3(x)
        x = self.dropout(self.relu(x))
        x = self.fc4(x)
        x = self.dropout(self.relu(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)
        # return x
