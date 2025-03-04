from typing import Any

import torch
from torch import Tensor
import warnings
import math

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch.nn as nn
from torch.nn import Parameter, ReLU, Module


class GCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, normalize='sym', bias=True, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.bias_bool = bias

        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, **kwargs):
        edge_index, _ = add_self_loops(edge_index)

        if self.normalize == 'sym':
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight=None, num_nodes=x.size(0),
                add_self_loops=False, dtype=x.dtype)
        elif self.normalize == 'rw':
            deg = degree(edge_index, num_nodes=x.shape(0))
            deg_inv = 1 / deg
            edge_weight = None
        else:
            raise NotImplementedError


        h = self.linear(x)
        h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        if self.normalize == 'rw':
            torch.einsum('n, nd -> nd', deg_inv, h)

        if self.bias_bool:
            h = h + self.bias

        return h

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class LinConv(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, **kwargs):
        return self.linear(x)

class SAGEConv(MessagePassing):

    def __init__(self, in_dim, out_dim, bias=True, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias_bool = bias


        self.linear_self = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.linear_ngb = torch.nn.Linear(in_dim, out_dim, bias=False)

        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)


    def forward(self, x, edge_index, **kwargs):
        deg = degree(edge_index[0], num_nodes=x.shape[0])
        deg_inv = 1 / deg

        agg = self.propagate(edge_index, x=x, edge_weight=None)
        agg = torch.einsum('n, nd -> nd', deg_inv, agg)

        h = self.linear_self(x) + self.linear_ngb(agg)
        if self.bias_bool:
            h = h + self.bias

        return h

class SumConv(MessagePassing):

    def __init__(self, in_dim, out_dim, bias=True, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias_bool = bias


        self.linear_self = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.linear_ngb = torch.nn.Linear(in_dim, out_dim, bias=False)

        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)


    def forward(self, x, edge_index, **kwargs):
        agg = self.propagate(edge_index, x=x, edge_weight=None)

        h = self.linear_self(x) + self.linear_ngb(agg)
        if self.bias_bool:
            h = h + self.bias
        return h


class GINConv(MessagePassing):
    def __init__(self, in_dim, out_dim, bias=True, eps=True, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.bias_bool = bias
        self.eps_bool = eps

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim, bias=self.bias_bool),
            ReLU(),
            nn.Linear(self.out_dim, self.out_dim, bias=self.bias_bool)
        )

        if self.eps_bool:
            self.eps = torch.nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('eps', None)

    def forward(self, x, edge_index, **kwargs):
        agg = self.propagate(edge_index, x=x)
        h = self.mlp(agg + (1 + self.eps) * x)

        return h


class TaylorBuNNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, normalize='sym', bias=True,
                 max_deg=4, tau=1, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.bias_bool = bias
        self.max_deg = max_deg
        self.tau = tau

        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, node_rep, **kwargs):
        bundle_dim = node_rep.shape[-1]
        num_nodes = x.shape[0]
        dim = x.shape[1]
        num_bundles = node_rep.shape[1]

        if self.normalize == 'sym':
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight=None, num_nodes=x.size(0),
                add_self_loops=False, dtype=x.dtype)
        elif self.normalize == 'rw':
            deg = degree(edge_index[0], num_nodes=x.shape[0])
            deg_inv = 1 / deg
            edge_weight = None
        else:
            raise NotImplementedError

        vector_field = x.reshape(num_nodes, num_bundles, bundle_dim,
                                 -1)  # works since dim divisible by bundle dim
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, dim)

        h = self.linear(h)

        curr_pow = h  # -tau^p*Delta^p/(p!)
        for k in range(1, self.max_deg + 1):
            agg = self.propagate(edge_index, x=curr_pow, edge_weight=edge_weight)  # A*( -tau^p*Delta^p/(p!))
            if self.normalize == 'rw':
                agg = torch.einsum('n, nd -> nd', deg_inv, agg)
            curr_pow = -self.tau * 1 / k * (curr_pow - agg)  # -tau/(p+1) * ((-tau)^p*Delta^p/(p!) - A*( (-tau^p)*Delta^p/(p!)))
            h = h + curr_pow

        if self.bias_bool:
            h = h + self.bias

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, num_bundles, bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, dim)
        return h

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SpectralBuNNConv(Module):
    def __init__(self, in_dim, out_dim, normalize='sym', bias=True,
                 learn_tau=True, tau=1, num_bundles=1, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.bias_bool = bias
        self.num_bundles = num_bundles

        self.relu = ReLU()   # to keep the taus positive

        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

        if learn_tau:
            self.tau = Parameter((torch.rand([self.num_bundles]) + 0.5) * tau)
        else:
            big_tau = tau * torch.ones([self.num_bundles], dtype=torch.float)
            self.register_buffer('tau', big_tau)
            self.tau = big_tau

    def forward(self, x, edge_index, node_rep, eig_vecs, eig_vals, **kwargs):
        # TODO: this works for a single graph, need to generalize to batches
        # eig_vect is of shape: [n, k] where k<=n where [i, q] is the value at node i of eigenvector k
        # eig_vals is of shape [k]
        bundle_dim = node_rep.shape[-1]
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]
        num_eigs = eig_vecs.shape[1]

        if self.normalize == 'sym':
            _, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight=None, num_nodes=x.size(0),
                add_self_loops=False, dtype=x.dtype)
        elif self.normalize == 'rw':
            deg = degree(edge_index[0], num_nodes=x.shape[0])
            deg_inv = 1 / deg
            edge_weight = None
        else:
            raise NotImplementedError
        vector_field = x.reshape(num_nodes, num_bundles, bundle_dim,
                                 -1)  # works since dim divisible by bundle dim
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.in_dim)

        h = self.linear(h)

        # compute projections
        proj = torch.einsum('kn, nd -> kd', eig_vecs.transpose(0, 1), h)  # phi^T*X
        proj = proj.reshape(num_eigs, num_bundles, bundle_dim, -1)
        # compute exponentials
        rep_eigvals = eig_vals.unsqueeze(1).repeat(1, self.num_bundles)
        exponent = torch.einsum('kb, b -> kb', rep_eigvals, self.relu(self.tau))  # -lambda*tau
        expo = torch.exp(-exponent)
        # evolve the projections
        evol = torch.einsum('kbdc, kb -> kbdc', proj, expo)
        # unproject
        unproj = torch.einsum('nk, kbdc -> nbdc', eig_vecs, evol)
        h = unproj.reshape(num_nodes, self.out_dim)

        if self.bias_bool:
            h = h + self.bias

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, num_bundles, bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.out_dim)
        return h


class LimBuNNConv(Module):
    def __init__(self, in_dim, out_dim, normalize='sym', bias=True,
                 num_bundles=1, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.bias_bool = bias
        self.num_bundles = num_bundles

        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, node_rep, **kwargs):
        # TODO: this works for a single graph, need to generalize to batches...
        # eig_vect is of shape: [n, k] where k<=n where [i, q] is the value at node i of eigenvector k
        # eig_vals is of shape [k]
        bundle_dim = node_rep.shape[-1]
        num_nodes = x.shape[0]
        num_bundles = node_rep.shape[1]

        if self.normalize == 'sym':
            deg = degree(edge_index[0], num_nodes=x.shape[0])
            deg_sqrt = deg.pow(0.5)
        elif self.normalize == 'rw':
            deg = degree(edge_index[0], num_nodes=x.shape[0])
        else:
            raise NotImplementedError
        vector_field = x.reshape(num_nodes, num_bundles, bundle_dim,
                                 -1)  # works since dim divisible by bundle dim
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, self.in_dim)

        h = self.linear(h)

        if self.normalize == 'sym':
            h = torch.einsum('n, nd -> nd', deg_sqrt, h)
        else:
            h = torch.einsum('n, nd -> nd', deg, h)
        proj = h.sum(dim=0) / (edge_index.size(1))
        h = proj.unsqueeze(0).repeat(num_nodes, 1)
        if self.normalize == 'sym':
            h = torch.einsum('n, nd -> nd', deg_sqrt, h)

        if self.bias_bool:
            h = h + self.bias

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, num_bundles, bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, self.out_dim)
        return h


class SheafBuNNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, normalize='sym', bias=True, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.bias_bool = bias

        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        if self.bias_bool:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, node_rep, **kwargs):
        bundle_dim = node_rep.shape[-1]
        num_nodes = x.shape[0]
        dim = x.shape[1]
        num_bundles = node_rep.shape[1]

        if self.normalize == 'sym':
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight=None, num_nodes=x.size(0),
                add_self_loops=False, dtype=x.dtype)
        elif self.normalize == 'rw':
            deg = degree(edge_index[0], num_nodes=x.shape[0])
            deg_inv = 1 / deg
            edge_weight = None
        else:
            raise NotImplementedError

        vector_field = x.reshape(num_nodes, num_bundles, bundle_dim,
                                 -1)  # works since dim divisible by bundle dim
        # first transform into the 'edge space'
        vector_field = torch.einsum('abcd, abde -> abce', node_rep, vector_field)
        h = vector_field.reshape(num_nodes, dim)

        agg = self.propagate(edge_index, x=h, edge_weight=edge_weight)  # A*x
        if self.normalize == 'rw':
            agg = torch.einsum('n, nd -> nd', deg_inv, agg)
        h = h - agg  # Lap * thingy

        # transform back to 'node space'
        vector_field = h.reshape(num_nodes, num_bundles, bundle_dim, -1)
        vector_field = torch.einsum('abcd, abde -> abce', node_rep.transpose(2, 3),
                                    vector_field)  # inverse is transpose
        h = vector_field.reshape(num_nodes, dim)

        h = self.linear(h)

        if self.bias_bool:
            h = h + self.bias
        return h

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
