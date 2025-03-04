import tqdm
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected

import graph_walker  # pylint: disable=import-error


shrikhande_graph = nx.Graph()
vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 4), (3, 8), (3, 11), (3, 12), (4, 6), (4, 11), (4, 13), (4, 14), (5, 6), (5, 7), (5, 12), (5, 15), (5, 16),
        (6, 9), (6, 13), (6, 15), (7, 10), (7, 14), (7, 16), (8, 9), (8, 12), (8, 13), (8, 16),
        (9, 10), (9, 13), (9, 15), (10, 11), (10, 14), (10, 15), (11, 12), (11, 14), (11, 15),
        (12, 15), (12, 16), (13, 14), (13, 16), (14, 16)]
shrikhande_graph.add_nodes_from(vertices)
shrikhande_graph.add_edges_from(edges)
shrikhande_graph = nx.to_undirected(shrikhande_graph)
rook_4x4_graph = nx.Graph()
vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 8), (2, 9), (2, 10),
        (3, 4), (3, 11), (3, 12), (3, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 11), (5, 14),
        (6, 7), (6, 9), (6, 12), (6, 15), (7, 10), (7, 13), (7, 16), (8, 9), (8, 10), (8, 11), (8, 14),
        (9, 10), (9, 12), (9, 15), (10, 13), (10, 16), (11, 12), (11, 13), (11, 14), (12, 13), (12, 15),
        (13, 16), (14, 15), (14, 16), (15, 16)]
rook_4x4_graph.add_nodes_from(vertices)
rook_4x4_graph.add_edges_from(edges)
rook_4x4_graph = nx.to_undirected(rook_4x4_graph)
# load graph data
data_list = []
for i, data in enumerate([shrikhande_graph, rook_4x4_graph]):
    x = torch.ones(data.number_of_nodes(), 1, dtype=torch.long)
    y = torch.tensor([i], dtype=torch.long)
    edge_index = torch.tensor(list(data.edges())).transpose(1, 0) - 1
    edge_index = to_undirected(edge_index)
    data_list.append(Data(edge_index=edge_index, x=x, y=y))
G1 = nx.to_undirected(to_networkx(data_list[0]))
G2 = nx.to_undirected(to_networkx(data_list[1]))


def compute_cover_times(G, walks, restarts):
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    G_nodes = list(G.nodes)
    for walk_idx, (walk, restart) in tqdm.tqdm(list(enumerate(zip(walks, restarts)))):
        walk_start_node = walk[0]
        visited = np.zeros(len(G_nodes), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != walk_start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == walk_start_node, "Restart node must be start node"
                continue
            if i in G_nodes:
                visited[G_nodes.index(i)] = True
            if j in G_nodes:
                visited[G_nodes.index(j)] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
            assert t < len(walk) - 2, f"Walk did not cover all nodes, start node: {walk_start_node}, walk idx: {walk_idx}"
    assert np.all(cover_times != -1), "All walks must cover all nodes"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(G_nodes), dtype=np.float32) - 1
    walk_start_nodes = walks[:, 0]
    for walk_start_node in np.unique(walk_start_nodes):
        idxs = np.where(walk_start_nodes == walk_start_node)[0]
        cover_times_per_start_node[list(G_nodes).index(walk_start_node)] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times_per_start_node


def compute_undirected_edge_cover_times(G, walks, restarts):
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    G_nodes = list(G.nodes)
    G_edges = list(G.edges)
    for walk_idx, (walk, restart) in tqdm.tqdm(list(enumerate(zip(walks, restarts)))):
        walk_start_node = walk[0]
        visited = np.zeros(len(G_edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != walk_start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == walk_start_node, "Restart node must be start node"
                continue
            if (i, j) in G_edges:
                visited[G_edges.index((i, j))] = True
            if (j, i) in G_edges:
                visited[G_edges.index((j, i))] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
            assert t < len(walk) - 2, f"Walk did not cover all edges, start node: {walk_start_node}, walk idx: {walk_idx}"
    assert np.all(cover_times != -1), "All walks must cover all edges"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(G_nodes), dtype=np.float32) - 1
    walk_start_nodes = walks[:, 0]
    for walk_start_node in np.unique(walk_start_nodes):
        idxs = np.where(walk_start_nodes == walk_start_node)[0]
        cover_times_per_start_node[list(G_nodes).index(walk_start_node)] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times


n_walks = 100
walk_len = 10000
for G in [G1, G2]:
    walks, restarts = graph_walker.random_walks(
        G,
        n_walks=n_walks,
        walk_len=walk_len,
        min_degree=True,
        sub_sampling=0.,
        p=1, q=1, alpha=0, k=None,
        no_backtrack=True,
        verbose=True
    )

    cover_times = compute_cover_times(G, walks, restarts).astype(float)
    print(f"Cover time: {np.max(cover_times):.2f}, across starting vertices: {np.mean(cover_times):.2f} ± {np.std(cover_times):.2f}")

    undirected_edge_cover_times = compute_undirected_edge_cover_times(G, walks, restarts).astype(float)
    print(f"Undirected edge cover time: {np.max(undirected_edge_cover_times):.2f}, across starting vertices: {np.mean(undirected_edge_cover_times):.2f} ± {np.std(undirected_edge_cover_times):.2f}")
