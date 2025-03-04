import torch
from torch.nn import Module
import torch.nn as nn
from torch.nn import Linear, ReLU, GELU

from itertools import chain

from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_torch_coo_tensor
)
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import numpy as np

from .layers import GCNConv, SAGEConv, GINConv, SumConv, TaylorBuNNConv, LinConv, SpectralBuNNConv, LimBuNNConv, SheafBuNNConv
from torch_geometric.nn.conv import ChebConv, GATConv
from .orthogonal import Orthogonal


MODULES = {
    'GCN': GCNConv,
    'SAGE': SAGEConv,
    'GIN': GINConv,
    'TaylorBuNN': TaylorBuNNConv,
    'SpectralBuNN': SpectralBuNNConv,
    'Sum': SumConv,
    'MLP': LinConv,
    'LimBuNN': LimBuNNConv,
    'Cheb': ChebConv,
    'GAT': GATConv,
    'SheafBuNN': SheafBuNNConv
}

ACT = {
    'relu': ReLU,
    'gelu': GELU
}

class ModelNode(Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, layer_type,
                 normalize='sym', bias=True, eps=True, act='relu', k=1, heads=1,
                 **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layer_type = layer_type
        self.num_layers = num_layer
        self.normalize = normalize

        self.act = ACT[act]()


        self.enc = Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()

        for _ in range(num_layer):
            module = MODULES[layer_type]
            conv = module(hidden_dim, hidden_dim, normalize=self.normalize, bias=bias, eps=eps,
                          K=k, heads=heads)
            self.convs.append(conv)

        self.dec = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None, **kwargs):
        h = self.enc(x)
        for conv in self.convs:
            h = conv(h, edge_index=edge_index)
            h = self.act(h)
        h = self.dec(h)
        return h


class BuNNNode(Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, layer_type,
                 normalize='sym', bias=True, eps=True, act='relu',
                 bundle_dim=2, num_bundles=1, tau=1, max_deg=4,
                 num_gnn_layer=0,  gnn_type="GCN", learn_tau=False, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layer_type = layer_type
        self.num_layers = num_layer
        self.normalize = normalize
        self.bias = bias
        self.eps = eps

        self.bundle_dim = bundle_dim
        self.num_bundles = num_bundles
        self.tau = tau
        self.learn_tau = learn_tau
        if learn_tau:
            assert layer_type == 'SpectralBuNN'
        self.max_deg = max_deg
        self.num_gnn_layer = num_gnn_layer
        self.gnn_type = gnn_type
        self.num_bun_params = int((self.bundle_dim-1) * self.bundle_dim / 2 * self.num_bundles)  # TODO: check that this is right!!

        self.act_st = act
        self.act = ACT[act]()

        self.enc = Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.node_encs = nn.ModuleList()

        for _ in range(num_layer):
            module = MODULES[layer_type]
            conv = module(hidden_dim, hidden_dim, normalize=self.normalize, bias=bias,
                          eps=eps, num_bundles=self.num_bundles, learn_tau=self.learn_tau, tau=self.tau,
                          max_deg=self.max_deg)
            self.convs.append(conv)

            enc = ModelNode(self.hidden_dim, self.num_bun_params, self.hidden_dim,
                            num_layer=self.num_gnn_layer,
                            layer_type=self.gnn_type, normalize=self.normalize,
                            bias=self.bias, eps=self.eps, act=self.act_st)
            self.node_encs.append(enc)

        self.orthogonal = Orthogonal(self.bundle_dim, "euler")

        self.dec = Linear(hidden_dim, out_dim)


    def forward(self, x, edge_index, batch=None, **kwargs):
        num_nodes = x.size(0)
        if self.layer_type == "SpectralBuNN":
            if batch is None or not("eig_vecs" in batch.keys()) or not("eig_vals" in  batch.keys()):
                assert self.normalize == 'sym'
                edge_index_lap, edge_weight = get_laplacian(
                    edge_index,
                    edge_weight=None,
                    normalization=self.normalize,
                    num_nodes=num_nodes,
                )
                L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
                L = L.todense()
                eig_vals, eig_vecs = eigh(L)
                eig_vals, eig_vecs = torch.from_numpy(eig_vals).to(x.device), torch.from_numpy(eig_vecs).to(x.device)
            else:
                eig_vals, eig_vecs = batch.eig_vals, batch.eig_vecs
        else:
            eig_vals, eig_vecs = None, None
        h = self.enc(x)
        for i, conv in enumerate(self.convs):
            node_rep = self.node_encs[i](h, edge_index)
            node_rep = node_rep.reshape([num_nodes*self.num_bundles, -1])
            node_rep = self.orthogonal(node_rep)
            node_rep = node_rep.reshape([num_nodes, self.num_bundles, self.bundle_dim, -1])
            h = conv(h, edge_index, node_rep, eig_vecs=eig_vecs, eig_vals=eig_vals)
            h = self.act(h)
        h = self.dec(h)
        return h

    def grouped_parameters(self):
        w_params, bunn_params, tau_params = [], [], []
        for name, param in self.named_parameters():
            if "tau" in name:
                tau_params.append(param)
            elif "node_enc" in name:
                bunn_params.append(param)
            else:
                w_params.append(param)
        return w_params, bunn_params, tau_params


class ConstantHalf100(Module):
    def __init__(self, n):
        super().__init__()

        self.cnst1 = np.sqrt(3*n)
        self.cnst2 = -np.sqrt(3*n)
        self.mod = torch.nn.Linear(1, 1)  # not used, just there to run

    def forward(self, x, edge_index, batch=None, **kwargs):
        num_nodes = x.shape[0]
        y1 = self.cnst1 * torch.ones([int(num_nodes / 2), 1], dtype=torch.float, device=x.device, requires_grad=True)
        y2 = self.cnst2 * torch.ones([int(num_nodes / 2), 1], dtype=torch.float, device=x.device, requires_grad=True)
        y = torch.cat((y1, y2), dim=0)
        return y


class ConstantFull(Module):
    def __init__(self, n):
        super().__init__()

        self.mod = torch.nn.Linear(1, 1)  # not used, just there to run

    def forward(self, x, edge_index, batch=None, **kwargs):
        num_nodes = x.shape[0]
        return torch.zeros([num_nodes, 1], dtype=torch.float, device=x.device, requires_grad=True)