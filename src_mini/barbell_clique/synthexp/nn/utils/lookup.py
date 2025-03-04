# from https://github.com/twitter-research/cwn/blob/387afe8974fb043addff6fdddc4c502fea272a5c/mp/nn.py#L7

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import BatchNorm1d as BN, LayerNorm as LN, Identity


def get_nonlinearity(nonlinearity, return_module=True):
    if nonlinearity == 'relu':
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == 'elu':
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == 'id':
        module = torch.nn.Identity
        function = lambda x: x
    elif nonlinearity == 'sigmoid':
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == 'tanh':
        module = torch.nn.Tanh
        function = torch.tanh
    elif nonlinearity == 'gelu':
        module = torch.nn.GELU
        function = F.elu
    else:
        raise NotImplementedError('Nonlinearity {} is not currently supported.'.format(nonlinearity))
    if return_module:
        return module
    return function


def get_pooling_fn(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    else:
        raise NotImplementedError('Readout {} is not currently supported.'.format(readout))


def get_norm(norm):
    if norm == "batch":
        return BN
    elif norm == "layer":
        return LN
    elif norm == "identity" or norm is None:
        return Identity
