# pylint: disable=protected-access
import math
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.utils import remove_self_loops

import graph_walker


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _, seqlen, _ = x.shape
        return self.dropout(self.pe[:, :seqlen])


class RWNN(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        min_degree,
        no_backtrack,
        walk_len,
        train_n_walks,
        eval_n_walks,
        test_n_walks,
        num_nodes,
    ):
        super().__init__()
        print(f'Using {num_layers} layers and {dim} dimensions')
        print(f'Min degree: {min_degree}, No backtrack: {no_backtrack}, Walk len: {walk_len}, Train n walks: {train_n_walks}, Eval n walks: {eval_n_walks}, Num nodes: {num_nodes}')
        assert num_nodes == 10
        self.min_degree = min_degree
        self.no_backtrack = no_backtrack
        self.walk_len = walk_len
        self.train_n_walks = train_n_walks
        self.eval_n_walks = eval_n_walks
        self.test_n_walks = test_n_walks
        self.walk_embed = nn.Embedding(2 * num_nodes, dim)
        self.attr_embed = nn.Linear(1, dim)
        self.start_embed = nn.Sequential(
            nn.Linear(1, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, dim, bias=False)
        )
        for weight in self.start_embed.parameters():
            weight.data.mul_(nn.init.calculate_gain('tanh'))
        self.pos_encoder = PositionalEncoding(d_model=dim, dropout=0.)
        self.pos_embed = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.pos_embed.weight)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, 1)

    @torch.compile
    def _encoder(self, x):
        return self.encoder(x)

    def forward(self, x, edge_index, batch=None, is_test=False, **kwargs):
        # remove self-loops
        assert (edge_index == batch.edge_index).all()
        edge_index, _ = remove_self_loops(batch.edge_index)
        batch.edge_index = edge_index

        G = nx.to_undirected(torch_geometric.utils.to_networkx(batch))
        n_walks = self.train_n_walks if self.training else (self.test_n_walks if is_test else self.eval_n_walks)
        walks, _ = graph_walker.random_walks(
            G=G,
            n_walks=n_walks,
            walk_len=self.walk_len,
            min_degree=self.min_degree,
            no_backtrack=self.no_backtrack,
            verbose=False
        )
        named_walks = torch.tensor(graph_walker._anonymize(walks).astype(np.int32), dtype=torch.long, device=x.device)
        walks = walks.astype(np.int32)
        attr = batch.x[walks]
        start_attr = attr[:, 0:1]
        targets = batch.y[walks[:, 0]]

        x = self.walk_embed(named_walks - 1) + self.attr_embed(attr) + self.start_embed(start_attr)
        x = x + self.pos_embed(self.pos_encoder(x))
        x = self._encoder(F.relu(x))
        outputs = self.head(x.mean(dim=1))

        if not self.training:
            # Monte Carlo averaging
            outputs = outputs.view(n_walks, batch.num_nodes, 1).mean(dim=0)
            targets = targets.view(n_walks, batch.num_nodes, 1).mean(dim=0)
        return outputs, targets


class RWNNMLP(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        min_degree,
        no_backtrack,
        walk_len,
        train_n_walks,
        eval_n_walks,
        test_n_walks,
        num_nodes,
    ):
        super().__init__()
        print(f'Using {num_layers} layers and {dim} dimensions')
        print(f'Min degree: {min_degree}, No backtrack: {no_backtrack}, Walk len: {walk_len}, Train n walks: {train_n_walks}, Eval n walks: {eval_n_walks}, Num nodes: {num_nodes}')
        assert num_nodes == 10
        self.min_degree = min_degree
        self.no_backtrack = no_backtrack
        self.walk_len = walk_len
        self.train_n_walks = train_n_walks
        self.eval_n_walks = eval_n_walks
        self.test_n_walks = test_n_walks
        self.dim = dim
        self.walk_embed = nn.Embedding(2 * num_nodes, dim)
        self.attr_embed = nn.Linear(1, dim)
        self.start_embed = nn.Sequential(
            nn.Linear(1, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, dim, bias=False)
        )
        for weight in self.start_embed.parameters():
            weight.data.mul_(nn.init.calculate_gain('tanh'))
        self.pos_encoder = PositionalEncoding(d_model=dim, dropout=0.)
        self.pos_embed = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.pos_embed.weight)
        assert num_layers == 1
        self.encoder = nn.Sequential(
            nn.Linear(walk_len * dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, x, edge_index, batch=None, is_test=False, **kwargs):
        # remove self-loops
        assert (edge_index == batch.edge_index).all()
        edge_index, _ = remove_self_loops(batch.edge_index)
        batch.edge_index = edge_index

        G = nx.to_undirected(torch_geometric.utils.to_networkx(batch))
        n_walks = self.train_n_walks if self.training else (self.test_n_walks if is_test else self.eval_n_walks)
        walks, _ = graph_walker.random_walks(
            G=G,
            n_walks=n_walks,
            walk_len=self.walk_len,
            min_degree=self.min_degree,
            no_backtrack=self.no_backtrack,
            verbose=False
        )
        named_walks = torch.tensor(graph_walker._anonymize(walks).astype(np.int32), dtype=torch.long, device=x.device)
        walks = walks.astype(np.int32)
        attr = batch.x[walks]
        start_attr = attr[:, 0:1]
        targets = batch.y[walks[:, 0]]

        x = self.walk_embed(named_walks - 1) + self.attr_embed(attr) + self.start_embed(start_attr)
        x = x + self.pos_embed(self.pos_encoder(x))
        x = self.encoder(F.relu(x).view(x.shape[0], self.walk_len * self.dim))
        outputs = self.head(x)

        if not self.training:
            # Monte Carlo averaging
            outputs = outputs.view(n_walks, batch.num_nodes, 1).mean(dim=0)
            targets = targets.view(n_walks, batch.num_nodes, 1).mean(dim=0)
        return outputs, targets


class WalkLM(nn.Module):
    def __init__(
        self,
        num_layers,
        dim,
        walk_len,
        train_n_walks,
        eval_n_walks,
        test_n_walks,
        num_nodes,
    ):
        super().__init__()
        print(f'Using {num_layers} layers and {dim} dimensions')
        print(f'Walk len: {walk_len}, Train n walks: {train_n_walks}, Eval n walks: {eval_n_walks}, Num nodes: {num_nodes}')
        assert num_nodes == 10
        self.walk_len = walk_len
        self.train_n_walks = train_n_walks
        self.eval_n_walks = eval_n_walks
        self.test_n_walks = test_n_walks
        self.attr_embed = nn.Linear(1, dim)
        self.start_embed = nn.Sequential(
            nn.Linear(1, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, dim, bias=False)
        )
        for weight in self.start_embed.parameters():
            weight.data.mul_(nn.init.calculate_gain('tanh'))
        self.pos_encoder = PositionalEncoding(d_model=dim, dropout=0.)
        self.pos_embed = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.pos_embed.weight)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(dim, 1)

    @torch.compile
    def _encoder(self, x):
        return self.encoder(x)

    def forward(self, x, edge_index, batch=None, is_test=False, **kwargs):
        # remove self-loops
        assert (edge_index == batch.edge_index).all()
        edge_index, _ = remove_self_loops(batch.edge_index)
        batch.edge_index = edge_index

        G = nx.to_undirected(torch_geometric.utils.to_networkx(batch))
        n_walks = self.train_n_walks if self.training else (self.test_n_walks if is_test else self.eval_n_walks)
        walks, _ = graph_walker.random_walks(
            G=G,
            n_walks=n_walks,
            walk_len=self.walk_len,
            min_degree=False,
            no_backtrack=False,
            verbose=False
        )
        walks = walks.astype(np.int32)
        attr = batch.x[walks]
        start_attr = attr[:, 0:1]
        targets = batch.y[walks[:, 0]]

        x = self.attr_embed(attr) + self.start_embed(start_attr)
        x = x + self.pos_embed(self.pos_encoder(x))
        x = self._encoder(F.relu(x))
        outputs = self.head(x.mean(dim=1))
    
        if not self.training:
            # Monte Carlo averaging
            outputs = outputs.view(n_walks, batch.num_nodes, 1).mean(dim=0)
            targets = targets.view(n_walks, batch.num_nodes, 1).mean(dim=0)
        return outputs, targets
