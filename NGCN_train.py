from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_loader import *
from utils import loss_polt
from comp.NGCN_model import NGCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=1000, help='Patience')
parser.add_argument('--model', type=str, default="NGCN", help='choose the version of GAT.')
parser.add_argument('--dataset', type=str, default='LiDAR', help='choose the dataset, name should be <setname>-<datasetname>-<loadtype>')

parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.5e-3')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
# parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--K', type=int, default=5, help='K.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed) 
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_cite_data()
elif args.dataset.startswith('PLT'):
    adj, features, labels, idx_train, idx_val, idx_test = load_Planetoid_data(args.dataset)
elif args.dataset == 'LiDAR':
    adj, features, labels, idx_train, idx_val, idx_test = load_txt_data(data_name="UHnoL")   
elif args.dataset == 'test':
    adj, features, labels, idx_train, idx_val, idx_test = load_citation()
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_ogb_data(args.dataset)


if args.model == "NGCN":
    model = NGCN(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout,
                alpha=args.alpha,
                K = args.K)

# optimizer
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


x_index=[]
train_loss_list=[]
val_loss_list=[]
train_acc_list=[]
val_acc_list=[]
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    train_loss_list.append(loss_train.item())
    val_loss_list.append(loss_val.item())
    train_acc_list.append(acc_train.item())
    val_acc_list.append(acc_val.item())
    x_index.append(epoch)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), acc_val.data.item()


def compute_test(model, features, adj, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        unique, count = np.unique(output.max(1)[1].type_as(labels).cpu().numpy(), return_counts=True)
        print('TEST RESULT : The number of each class is {}'.format(dict(zip(unique,count))))
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


if __name__ == "__main__":
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    best = -10086
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc = train(epoch)
        loss_values.append(loss)
        acc_values.append(acc)

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        # if loss_values[-1] < best:
        #     best = loss_values[-1]
        if acc_values[-1] > best:
            best = acc_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # plot loss
    loss_polt(x_index, train_loss_list, val_loss_list, train_acc_list, val_acc_list)

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    # Testing
    compute_test(model, features, adj, idx_test)

    # if args.model == "EnhanceMultiMetaKGAT":
    #     print("enhance_weight : {}".format(model.enhance_weight.data))
    