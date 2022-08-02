import numpy as np
import scipy.sparse as sp
import torch
import os
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import matplotlib.pyplot as plt


def create_features_from_LiDAR(path='./Points/', file_name='SPP.txt', clc_points=10000, Spatialwight=0.5, sigmalist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]):
    """create the features and adj from LiDAR data"""
    print('Loading LiDAR {} dataset...'.format(file_name))
    OriginData = pointloader(os.path.join(path, file_name))
    OriginData = np.random.permutation(OriginData)
    A = OriginData[:clc_points, :]
    featureadj = MultikernelMatrix(A, Spatialwight, sigmalist)
    # featureadj += np.eye(featureadj.shape[0])

    featuresxyz=A[:, :3]
    featuresxyz=normalize_maxmin(featuresxyz)
    featurelamda=A[:, 3:6]
    featurelamda=normalize_maxmin(featurelamda)

    features = np.concatenate([featuresxyz, featurelamda, featureadj], 1)
    labels = A[:, -1]
    features = np.concatenate([features, labels.reshape(labels.shape[0], 1)], 1)

    print('Loading LiDAR Done, start saving')
    data_name = file_name.split('.')[0]
    np.save(os.path.join(path, data_name+'_features'), features)
    np.save(os.path.join(path, data_name+'_adj'), featureadj)
    print('already save LiDAR features and adj')

    unique, count = np.unique(labels, return_counts=True)
    print(' The number of each class is {}'.format(dict(zip(unique, count))))



def MultikernelMatrix(Mx,Spatialwight,sigmalist):
    '''
    Multikernel Matrix
    '''
    print('Building Multikernel Matrix')
    Spatial = Mx[:, :3]
    Spectral = Mx[:, 3:6]
    SpatialDistance = MatrixDistance(Spatial)
    SpectralDistance = MatrixDistance(Spectral)
    MultikernelMatrix = np.zeros((Mx.shape[0],Mx.shape[0]))
    for i in range (len(sigmalist)):
        SpatialGaussAdj = np.exp(-SpatialDistance/sigmalist[i]) - np.eye(SpatialDistance.shape[0])
        SpectralGaussAdj = np.exp(-SpectralDistance/sigmalist[i]) - np.eye(SpectralDistance.shape[0])
        SpatialGaussAdj = normalize_maxmin(SpatialGaussAdj)
        SpectralGaussAdj = normalize_maxmin(SpectralGaussAdj)
        ADJMatrix = SpatialGaussAdj*Spatialwight + SpectralGaussAdj*(1-Spatialwight)
        MultikernelMatrix += ADJMatrix
    MultikernelMatrix += np.eye(MultikernelMatrix.shape[0])
    print('Multikernel Matrix Done')
    return MultikernelMatrix


def MatrixDistance(Mx):
    '''Calculate the distance matrix.'''
    DisMatrix=np.zeros((Mx.shape[0],Mx.shape[0]))
    for i in range(Mx.shape[1]):
        col=Mx[:,[i]]
        len=Mx.shape[0]
        a=col**2
        A = a.repeat(len,axis=-1)
        B=col*col.T
        c=a.T
        C=c.repeat(len,axis=0)
        D=A+C-2*B
        DisMatrix+=D
    return DisMatrix


def pointloader(txtname):
    '''
    Read a numpy array from a text file.
    '''
    f = open(txtname, mode='r')
    a = []
    for line in f:
        a.append(line[:-1].split(' '))
    return np.array(a, dtype=np.double)


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_softmax(adj):
    '''softmax normalization'''
    sumrow=np.sum(adj,axis=1)
    normalizedadj=adj/sumrow[:,np.newaxis]
    return normalizedadj

def normalize_Zscore(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx.dot(r_mat_inv_sqrt)
    mx.transpose()
    mx.dot(r_mat_inv_sqrt)
    return mx

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # 求逆，然后折叠
    r_inv[np.isinf(r_inv)] = 0.     # 如果是无穷大，就改为0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_maxmin(Mx, axis=2):
    '''
    Normalize the matrix Mx by max-min normalization.
    axis=0: normalize each row
    axis=1: normalize each column
    axis=2: normalize the whole matrix
    '''
    if axis == 1:
        M_min = np.min(Mx, axis=1)
        M_max = np.max(Mx, axis=1)
        for i in range(Mx.shape[1]):
            Mx[:, i] = (Mx[:, i] - M_min) / (M_max - M_min)
    elif axis == 0:
        M_min = np.min(Mx, axis=0)
        M_max = np.max(Mx, axis=0)
        for i in range(Mx.shape[0]):
            Mx[i, :] = (Mx[i, :] - M_min) / (M_max - M_min)
    elif axis == 2:
        M_min = np.amin(Mx)
        M_max = np.amax(Mx)
        Mx = (Mx - M_min) / (M_max - M_min)
    else:
        print('Error')
        return None
    return Mx



def loss_polt(x, train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    '''
    Plot the loss curve and save it to the path.
    '''
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, label='train_loss')
    plt.plot(x, val_loss_list, label='val_loss')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, train_acc_list, label='train_acc')
    plt.plot(x, val_acc_list, label='val_acc')
    plt.title("acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    # plt.show()
    plt.savefig('losspic.jpg', bbox_inches='tight', dpi=450)


if __name__ == "__main__":
    file_name = 'sptest.txt'
    create_features_from_LiDAR(file_name=file_name)
    # print(features.shape)