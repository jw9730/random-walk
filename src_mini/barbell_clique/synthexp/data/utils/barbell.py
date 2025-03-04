import torch
import numpy as np
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from numpy.linalg import eigh
from torch_geometric.data import Data


def gen_barbell(n):
    """ generates a barbell graph of size 2n, together with its eigenvectiors and eigenvalues,
    also returns the uniform features according to the description """
    edge_index1, edge_index2 = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index1 += [[i, j]]  # first clique
                edge_index2 += [[i+n, j+n]] # second clique
    edge_index = edge_index1 + edge_index2 + [[0, n], [n, 0]]  # both clique with bridge
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

    # compute spectral decomp
    num_nodes = 2*n
    edge_index, edge_weight = get_laplacian(
        edge_index,
        edge_weight=None,
        normalization='sym',
        num_nodes=num_nodes,
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    L = L.todense()
    eig_vals, eig_vecs = eigh(L)
    eig_vals, eig_vecs = torch.from_numpy(eig_vals), torch.from_numpy(eig_vecs)

    # signal generation
    x, y = gen_barbell_signal(n)
    return x, edge_index, y, eig_vals, eig_vecs

def gen_barbell_signal(n):
    mu1, mu2 = -np.sqrt(3*n), np.sqrt(3*n)
    std1, std2 = np.sqrt(n), np.sqrt(n)
    x1 = torch.empty([n, 1]).uniform_(mu1 - np.sqrt(3) * std1, mu1 + np.sqrt(3) * std1)
    x2 = torch.empty([n, 1]).uniform_(mu2 - np.sqrt(3) * std2, mu2 + np.sqrt(3) * std2)
    y1, y2 = x2.mean(dim=0).repeat(n, 1), x1.mean(dim=0).repeat(n, 1)
    x = torch.cat((x1, x2), dim=0)
    y = torch.cat((y1, y2), dim=0)
    return x, y

def gen_many_barbells(num_samples, n):
    data_ls = []
    _, edge_index, _, eig_vals, eig_vecs = gen_barbell(n)
    for _ in range(num_samples):
        x, y = gen_barbell_signal(n)
        data = Data(x=x, edge_index=edge_index, y=y, eig_vals=eig_vals, eig_vecs=eig_vecs)
        data_ls += [data]
    return data_ls