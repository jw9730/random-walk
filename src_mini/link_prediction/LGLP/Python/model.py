#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:31:24 2019

@author: lei.cai
"""
import math
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch_geometric import nn as gnn
from torch_geometric.utils import to_networkx

import graph_walker


class BaseNet(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False):
        super().__init__()
        conv = gnn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], cached=False))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i-1], latent_dim[i], cached=False))

        latent_dim = sum(latent_dim)

        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

        # print the number of trainable parameters
        print(f'input dim: {input_dim}, hidden size: {hidden_size}, latent dim: {latent_dim}')
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, data):
        data.to(torch.device("cuda"))
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        cur_message_layer = x
        cat_message_layers = []
        lv = 0
        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)  
            cur_message_layer = torch.tanh(cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch==i).nonzero()[0].cpu().numpy()[0])

        cur_message_layer = cur_message_layer[idx,:]

        hidden = self.linear1(cur_message_layer)
        self.feature = hidden
        hidden = F.relu(hidden)

        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training)

        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc, self.feature
        else:
            return logits


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


class Model(nn.Module):
    def __init__(self, input_dim, num_layers, dim=32):
        super().__init__()
        print(f'Using {num_layers} layers and {dim} dimensions and input dimension {input_dim}')
        self.pos_encoder = PositionalEncoding(d_model=128, dropout=0.)
        self.walk_embed = nn.Linear(128, dim)
        self.pos_embed = nn.Linear(128, dim, bias=False)
        self.restart_embed = nn.Embedding(2, dim)
        self.attr_embed = nn.Linear(input_dim, dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=1, dim_feedforward=128, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, named_walks, restarts, attr):
        _, seqlen = named_walks.shape
        walk_enc = self.pos_encoder.pe[:, named_walks - 1].squeeze(0)
        pos_enc = self.pos_encoder.pe[:, :seqlen]
        x = self.walk_embed(walk_enc) + self.pos_embed(pos_enc) + self.restart_embed(restarts) + self.attr_embed(attr)
        x = self.encoder(F.relu(x))
        return F.relu(x).mean(dim=1)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False):
        super().__init__()
        assert latent_dim == [32, 32, 32, 1]
        self.latent_dim = latent_dim
        self.train_n_walks = 32
        self.eval_n_walks = 32
        self.walk_len = 200
        self.model = Model(input_dim=input_dim, num_layers=3, dim=32)

        self.linear1 = nn.Linear(32, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout
        self.dropout_rate = 0.0

        # print the number of trainable parameters
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def forward(self, data):
        data.to(torch.device("cuda"))
        x, _, batch, y = data.x, data.edge_index, data.batch, data.y
        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch==i).nonzero()[0].cpu().numpy()[0])

        G = nx.to_undirected(to_networkx(data))
        n_walks = self.train_n_walks if self.training else self.eval_n_walks
        walks, restarts = graph_walker.random_walks(
            G=G,
            n_walks=n_walks,
            walk_len=self.walk_len,
            k=5,
            min_degree=True,
            no_backtrack=True,
            start_nodes=idx,
            verbose=False
        )
        named_walks = torch.tensor(graph_walker._anonymize(walks).astype(np.int32), dtype=torch.long, device=torch.device("cuda"))
        restarts = torch.tensor(restarts, dtype=torch.long, device=torch.device("cuda"))
        walks = walks.astype(np.int32)
        attr = x[walks]
        y_loss = y.repeat(n_walks)

        outputs = self.model.forward(named_walks, restarts, attr)

        hidden = self.linear1(outputs)
        self.feature = hidden
        hidden = F.relu(hidden)

        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training, p=self.dropout_rate)

        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)

        avg_logits = logits.view(n_walks, -1, logits.shape[-1]).mean(dim=0)

        if y is not None:
            y = Variable(y)
            if self.training:
                loss = F.nll_loss(logits, y_loss)
            else:
                loss = F.nll_loss(avg_logits, y)

            pred = avg_logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return avg_logits, loss, acc, self.feature
        else:
            return avg_logits
