import numpy as np
import scipy.sparse as sp
import torch
from utils import *
from torch_geometric.datasets import Planetoid
import random
import sys
import pickle as pkl
import networkx as nx

def load_cite_data(path="./data/cora/", dataset="cora"):
    """cite dataset loader"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    unique, count = np.unique(labels, return_counts=True)
    print('The number of each class is {}'.format(dict(zip(unique,count))))

    # 建图
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_Zscore(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_citeseer_data():
    """citeseer dataset loader"""
    # TODO
    print('Loading citeseer dataset...')

    


def load_Planetoid_data(dataname="PLT-citeseer-public"):
    """The citation network datasets "Cora", "CiteSeer" and "PubMed" """
    print('Loading {} dataset...'.format(dataname))
    dataset = Planetoid(root='./data', name=dataname.split('-')[1], split=dataname.split('-')[2])

    dataset = dataset[0]
    features = dataset.x
    features = sp.csr_matrix(features[:, :], dtype=np.float32)
    labels = dataset.y.reshape((dataset.y.shape[0], 1))

    edges = dataset.edge_index.T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    idx_train = dataset.train_mask
    idx_val = dataset.val_mask
    idx_test = dataset.test_mask
    # idx_train = range(120)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(labels.reshape((labels.shape[0])))

    idx_train = torch.LongTensor(torch.where(idx_train == True)[0])
    idx_val = torch.LongTensor(torch.where(idx_val == True)[0])
    idx_test = torch.LongTensor(torch.where(idx_test == True)[0])

    unique, count = np.unique(labels[idx_train], return_counts=True)
    print('The number of each class is {}'.format(dict(zip(unique,count))))

    print('Loading {} dataset Done.'.format(dataname))
    return adj, features, labels, idx_train, idx_val, idx_test


def load_ogb_data(dataname="ogbn-arxiv"):
    """ogb data loader"""
    print('Loading {} dataset...'.format(dataname))

    dataset = PygNodePropPredDataset(name = "ogbn-arxiv", transform=T.ToSparseTensor()) 

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0] # pyg graph object
    
    features = graph.x  
    labels = graph.y
    adj = graph.adj_t

    print('Loading {} dataset Done.'.format(dataname))
    return adj, features, labels, train_idx, valid_idx, test_idx


def load_LiDAR_data(txtname='./Points/SPP.txt', num_points=3000, Spatialwight=0.5, sigmalist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
    '''load points data with constructed diagram'''
    print('Loading LiDAR')
    OriginData = pointloader(txtname)
    OriginData = np.random.permutation(OriginData)
    A = OriginData[:num_points, :]
    featureadj = MultikernelMatrix(A, Spatialwight, sigmalist)
    # featureadj += np.eye(featureadj.shape[0])
    featureadj = torch.FloatTensor(featureadj)

    featuresxyz=A[:, :3]
    featuresxyz=normalize_maxmin(featuresxyz)
    featurelamda=A[:, 3:6]
    featurelamda=normalize_maxmin(featurelamda)

    features = torch.FloatTensor(np.concatenate([featuresxyz, featurelamda, featureadj], 1))
    print('Loading LiDAR Done')

    labels = torch.LongTensor(A[:, -1])
    unique, count = np.unique(labels, return_counts=True)
    print('The number of each class is {}'.format(dict(zip(unique,count))))

    idx_train = range(int(0.6*num_points))
    idx_val = range(int(0.2*num_points), int(0.6*num_points))
    idx_test = range(int(0.6*num_points), int(1*num_points))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return featureadj, features, labels, idx_train, idx_val, idx_test


def load_txt_data(path='./Points/', data_name='SPP', num_points=None, each_class_num=50, val_num=1000, test_num=5000):
    '''load points data'''
    print('Loading {} dataset....'.format(data_name))
    all = np.load(path+data_name+'_adj.npy')[:, :]
    if num_points is None:
        num_points = all.shape[0]
    ADJ = sp.coo_matrix(all[:num_points, :num_points], dtype=np.float32)
    Feature = np.load(path+data_name+'_features.npy')[:num_points, :]

    # ADJ = torch.FloatTensor(ADJ)
    ADJ = sparse_mx_to_torch_sparse_tensor(ADJ)
    features = torch.FloatTensor(Feature[:, :6])
    # labels = torch.LongTensor(np.where(encode_onehot(Feature[:, -1]))[1])
    labels = torch.LongTensor(Feature[:, -1])
    unique, count = np.unique(labels, return_counts=True)
    print('The number of total each class is {}'.format(dict(zip(unique,count))))

    # fix each class num
    cou = [0 for i in range(len(unique))]
    idx_train = []
    for i in range(num_points):
        cou[labels[i]-1] += 1
        if cou[labels[i]-1] <= each_class_num:
            idx_train.append(i)
    r = [i for i in range(num_points) if i not in idx_train]
    idx_val = random.sample(list(r), int(val_num))
    r = [i for i in r if i not in idx_val]
    idx_test = random.sample(list(r), int(test_num))
    # idx_val = r[:int(0.5*len(r))]
    # idx_test = r[int(0.5*len(r)):]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    unique, count = np.unique(labels[idx_train], return_counts=True)
    print('Train Dataset : The number of train each class is {}'.format(dict(zip(unique, count))))
    unique, count = np.unique(labels[idx_val], return_counts=True)
    print('Val Dataset : The number of train each class is {}'.format(dict(zip(unique, count))))
    unique, count = np.unique(labels[idx_test], return_counts=True)
    print('Test Dataset : The number of train each class is {}'.format(dict(zip(unique, count))))

    print('Loading {} dataset Done.'.format(data_name))
    return ADJ, features, labels, idx_train, idx_val, idx_test


def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datax/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datax/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


if __name__ == "__main__":
    # load_txt_data()
    # load_LiDAR_data()
    # adj, features, labels, idx_train, idx_val, idx_test = load_Planetoid_data("PLT-citeseer-public")
    # adj, features, labels, idx_train, idx_val, idx_test = load_cite_data('./data/citeseer/', 'citeseer')
    # adj, features, labels, idx_train, idx_val, idx_test = load_citation()
    adj, features, labels, idx_train, idx_val, idx_test = load_txt_data(data_name='UH')
    print(adj.shape, features.shape, labels.shape, idx_train.shape, idx_val.shape, idx_test.shape)
