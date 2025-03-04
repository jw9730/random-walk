import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, is_undirected, degree

MAXINT = np.iinfo(np.int64).max


def preproc(data):
    data.coalesce()
    data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float32)
    edge_idx = data.edge_index
    edge_feat = data.edge_attr
    node_feat = data.x
    # Enforce undirected graphs
    if edge_idx.shape[1] > 0 and not is_undirected(edge_idx):
        x = edge_feat.detach().numpy()
        e = edge_idx.detach().numpy()
        x_map = {(e[0,i], e[1,i]): x[i] for i in range(e.shape[1])}
        edge_idx = is_undirected(edge_idx)
        e = edge_idx.detach().numpy()
        x = [x_map[(e[0,i], e[1,i])] if (e[0,i], e[1,i]) in x_map.keys() else x_map[(e[1,i], e[0,i])] for i in range(e.shape[1])]
        edge_feat = torch.tensor(x)
    data.edge_index = edge_idx
    data.edge_attr = edge_feat
    data.x = node_feat
    order = node_feat.shape[0]
    # create bitwise encoding of adjacency matrix using 64-bit integers
    data.node_id = torch.arange(0, order)
    bit_id = torch.zeros((order, order // 63 + 1), dtype=torch.int64)
    bit_id[data.node_id, data.node_id // 63] = torch.tensor(1) << data.node_id % 63
    data.adj_bits = scatter_sum(bit_id[edge_idx[0]], edge_idx[1], dim=0, dim_size=data.num_nodes)
    # compute node offsets in the adjacency list
    data.degrees = degree(edge_idx[0], dtype=torch.int64, num_nodes=data.num_nodes)
    adj_offset = torch.zeros((order,), dtype=torch.int64)
    adj_offset[1:] = torch.cumsum(data.degrees, dim=0)[:-1]
    data.adj_offset = adj_offset
    if not torch.is_tensor(data.y):
        data.y = torch.tensor(data.y)
    return data


def merge_batch(graph_data):
    adj_offset = [d.adj_offset for d in graph_data]
    degrees = [d.degrees for d in graph_data]
    edge_idx = [d.edge_index for d in graph_data]
    num_nodes = torch.tensor([d.shape[0] for d in degrees])
    num_edges = torch.tensor([e.shape[1] for e in edge_idx])
    num_graphs = len(graph_data)
    x_node = torch.cat([d.x for d in graph_data], dim=0)
    x_edge = torch.cat([d.edge_attr for d in graph_data], dim=0)
    x_edge = x_edge.view(x_edge.shape[0], -1)
    adj_offset = torch.cat(adj_offset)
    degrees = torch.cat(degrees)
    edge_idx = torch.cat(edge_idx, dim=1)
    node_graph_idx = torch.cat([i * torch.ones(x, dtype=torch.int64) for i, x in enumerate(num_nodes)])
    edge_graph_idx = torch.cat([i * torch.ones(x, dtype=torch.int64) for i, x in enumerate(num_edges)])
    node_shift = torch.zeros((len(graph_data),), dtype=torch.int64)
    edge_shift = torch.zeros((len(graph_data),), dtype=torch.int64)
    node_shift[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
    edge_shift[1:] = torch.cumsum(num_edges, dim=0)[:-1]
    adj_offset += edge_shift[node_graph_idx]
    edge_idx += node_shift[edge_graph_idx].view(1, -1)
    graph_offset = node_shift
    adj_bits = [d.adj_bits for d in graph_data]
    max_enc_length = np.max([p.shape[1] for p in adj_bits])
    adj_bits = torch.cat([F.pad(b, (0,max_enc_length-b.shape[1],0,0), 'constant', 0) for b in adj_bits], dim=0)
    node_id = torch.cat([d.node_id for d in graph_data], dim=0)
    y = torch.cat([d.y for d in graph_data], dim=0)
    data = Data(x=x_node, edge_index=edge_idx, edge_attr=x_edge, y=y)
    data.batch = node_graph_idx
    data.edge_batch = edge_graph_idx
    data.adj_offset = adj_offset
    data.degrees = degrees
    data.graph_offset = graph_offset
    data.order = num_nodes
    data.num_graphs = num_graphs
    data.node_id = node_id
    data.adj_bits = adj_bits
    return data


class Walker(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.win_size = 8
        self.compute_id = True
        self.compute_adj = True
        self.non_backtracking = True
        self.struc_feat_dim = 2 * self.win_size - 1

    def unweighted_choice(self, i, walks, adj_nodes, adj_offset, degrees, nb_degrees, choices):
        # do uniform step
        cur_nodes = walks[i]
        edge_idx = choices[i] % degrees[cur_nodes]
        chosen_edges = adj_offset[cur_nodes] + edge_idx
        if self.non_backtracking and i > 0:
            old_nodes = walks[i - 1]
            new_nodes = adj_nodes[chosen_edges]
            # correct backtracking
            bt = new_nodes == old_nodes
            if bt.max():
                bt_nodes = walks[i][bt]
                chosen_edges[bt] = adj_offset[bt_nodes] + (edge_idx[bt] + 1 + (choices[i][bt] % nb_degrees[bt_nodes])) % degrees[bt_nodes]
        return chosen_edges

    def sample_walks(self, data):
        device = data.x.device
        # get adjacency data
        adj_nodes = data.edge_index[1]
        adj_offset = data.adj_offset
        degrees = data.degrees
        node_id = data.node_id
        adj_bits = data.adj_bits
        # use default_old number of steps if not specified
        steps = self.steps
        # set dimensions
        s = self.win_size
        n = degrees.shape[0]
        l = steps + 1
        # starting nodes
        start = torch.arange(0, n, dtype=torch.int64).view(-1).to(device)
        start = start[degrees[start] > 0]
        # init tensor to hold walk indices
        w = start.shape[0]
        walks = torch.zeros((l, w), dtype=torch.int64, device=device)
        walks[0] = start
        walk_edges = torch.zeros((l-1, w), dtype=torch.int64, device=device)
        # get all random decisions at once (faster then individual calls)
        choices = torch.randint(0, MAXINT, (steps, w), device=device)
        if self.compute_id:
            id_enc = torch.zeros((l, s, w), dtype=torch.bool, device=device)
        if self.compute_adj:
            edges = torch.zeros((l, s, w), dtype=torch.bool, device=device)
        # remove one choice of each node with deg > 1 for no_backtrack walks
        nb_degree_mask = (degrees == 1)
        nb_degrees = nb_degree_mask * degrees + (~nb_degree_mask) * (degrees - 1)
        for i in range(steps):
            chosen_edges = self.unweighted_choice(i, walks, adj_nodes, adj_offset, degrees, nb_degrees, choices)
            # update nodes
            walks[i+1] = adj_nodes[chosen_edges]
            # update edge features
            walk_edges[i] = chosen_edges
            o = min(s, i+1)
            prev = walks[i+1-o:i+1]
            if self.compute_id:
                # get local identity relation
                id_enc[i+1, s-o:] = torch.eq(walks[i+1].view(1, w), prev)
            if self.compute_adj:
                # look up edges in the bit-wise adjacency encoding
                cur_id = node_id[walks[i+1]]
                cur_int = (cur_id // 63).view(1, -1, 1).repeat(o, 1, 1)
                edges[i + 1, s - o:] = (torch.gather(adj_bits[prev], 2, cur_int).view(o,-1) >> (cur_id % 63).view(1,-1)) % 2 == 1
        # permute walks into the correct shapes
        data.walk_nodes = walks.permute(1, 0)
        data.walk_edges = walk_edges.permute(1, 0)
        # combine id, adj and edge features
        feat = []
        if self.compute_id:
            feat.append(torch._cast_Float(id_enc.permute(2, 1, 0)))
        if self.compute_adj:
            feat.append(torch._cast_Float(edges.permute(2, 1, 0))[:, :-1, :])
        data.walk_x = torch.cat(feat, dim=1) if len(feat) > 0 else None
        return data


class VNUpdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            # nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(dim, dim, bias=False)
        )

    def forward(self, data):
        x = scatter_sum(data.h, data.batch, dim=0)
        if 'vn_h' in data:
            x += data.vn_h
        data.vn_h = self.mlp(x)
        data.h += data.vn_h[data.batch]
        return data


class ConvModule(nn.Module):
    def __init__(self, conv_dim, node_dim_in, edge_dim_in, w_feat_dim, dim_out, kernel_size):
        super().__init__()
        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.kernel_size = kernel_size
        # pool into center node
        self.pool_node = kernel_size // 2
        # rescale for residual connection
        self.node_rescale = nn.Linear(node_dim_in, dim_out, bias=False) if node_dim_in != dim_out else nn.Identity()
        # lost nodes due to lack of padding:
        self.border = kernel_size - 1
        self.convs = nn.Sequential(
            nn.Conv1d(node_dim_in + edge_dim_in + w_feat_dim, conv_dim, 1, padding=0, bias=False),
            nn.Conv1d(conv_dim, conv_dim, kernel_size, groups=conv_dim, padding=0, bias=False),
            # nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, 1, padding=0, bias=False),
            nn.ReLU()
        )
        self.node_out = nn.Sequential(
            nn.Linear(conv_dim, 2*dim_out, bias=False),
            # nn.BatchNorm1d(2*dim_out),
            nn.ReLU(),
            nn.Linear(2*dim_out, dim_out, bias=False)
        )

    def forward(self, data):
        walk_nodes = data.walk_nodes
        # build walk feature tensor
        walk_node_h = data.h[walk_nodes].transpose(2, 1)
        if 'walk_edge_h' not in data:
            padding = torch.zeros((walk_node_h.shape[0], self.edge_dim_in, 1), dtype=torch.float32, device=walk_node_h.device)
            data.walk_edge_h = torch.cat([padding, data.edge_h[data.walk_edges].transpose(2, 1)], dim=2)
        if 'walk_x' in data:
            x = torch.cat([walk_node_h, data.walk_edge_h, data.walk_x], dim=1)
        else:
            x = torch.cat([walk_node_h, data.walk_edge_h], dim=1)
        # apply the cnn
        y = self.convs(x)
        # pool in walklet embeddings into nodes
        flatt_dim = y.shape[0] * y.shape[2]
        y_flatt = y.transpose(2, 1).reshape(flatt_dim, -1)
        # get center indices
        if 'walk_nodes_flatt' not in data:
            data.walk_nodes_flatt = walk_nodes[:, self.pool_node:-(self.kernel_size - 1 - self.pool_node)].reshape(-1)
        # pool graphlet embeddings into nodes
        p_node = scatter_mean(y_flatt, data.walk_nodes_flatt, dim=0, dim_size=data.num_nodes)
        # rescale for the residual connection
        data.h_orig = self.node_rescale(data.h) + self.node_out(p_node)
        # alternative walk-wise pooling
        data.h_alt = self.node_rescale(data.h) + self.node_out(y.mean(dim=2))
        return data


class CRaWl(nn.Module):
    def __init__(self, num_layers, walk_len, dim, walk_pool):
        """
        :param model_dir: Directory to store model in
        :param node_feat_dim: Dimension of the node features
        :param edge_feat_dim: Dimension of the edge features
        """
        super().__init__()
        self.out_dim = 1
        self.layers = num_layers
        node_features = 1
        edge_features = 1
        self.hidden = dim
        self.kernel_size = 9
        self.dropout = 0.0
        self.vn = 1
        self.walker = Walker(walk_len)
        self.walk_dim = self.walker.struc_feat_dim
        self.conv_dim = dim
        self.walk_pool = walk_pool
        self.walk_embed = nn.Embedding(20, dim)
        self.attr_embed = nn.Linear(1, dim)
        modules = []
        for i in range(self.layers):
            modules.append(ConvModule(conv_dim=self.conv_dim,
                                      node_dim_in=node_features if i == 0 else self.hidden,
                                      edge_dim_in=edge_features,
                                      w_feat_dim=self.walk_dim,
                                      dim_out=self.hidden,
                                      kernel_size=self.kernel_size))
            if self.vn and i < self.layers - 1:
                modules.append(VNUpdate(self.hidden))
        self.convs = nn.Sequential(*modules)
        self.node_out = nn.Sequential(
            # nn.BatchNorm1d(self.hidden),
            nn.ReLU()
        )
        self.graph_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.out_dim)
        )

    def forward(self, x, edge_index, batch=None, **kwargs):
        # remove self-loops
        assert (edge_index == batch.edge_index).all()
        edge_index, _ = remove_self_loops(batch.edge_index)
        batch.edge_index = edge_index
        # proprocess
        batch = preproc(batch.to('cpu'))
        batch = merge_batch([batch])
        assert batch.is_undirected()
        # prepare batch
        batch.h = batch.x
        batch.edge_h = batch.edge_attr
        batch.walk_edge_h = None
        batch.walk_nodes_flatt = None
        # compute walks
        batch = batch.to(x.device)
        batch = self.walker.sample_walks(batch)
        if self.vn:
            batch.vn_h = None
        # apply convolutions
        self.convs(batch)
        # pool node embeddings
        batch.h = self.node_out(batch.h_orig) if not self.walk_pool else self.node_out(batch.h_alt)
        batch.y_pred = self.graph_out(batch.h)
        return batch.y_pred, batch.y
