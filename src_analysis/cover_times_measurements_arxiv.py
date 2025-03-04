from functools import partial
import multiprocessing as mp
from pathlib import Path
import tqdm
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset

import graph_walker  # pylint: disable=import-error


dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="../experiments/data")
split_idx = dataset.get_idx_split()
train_idx = split_idx["train"]
val_idx = split_idx["valid"]
test_idx = split_idx["test"]
# setup data for walk
batch = dataset._data
G_directed = to_networkx(batch)
G_undirected = nx.to_undirected(G_directed)
file_dir = Path("../src_llama3/data/ogbn_arxiv/processed_transitin_probs")
file_path = file_dir / 'processed_min_deg_True_sub_sample_0.0.pt'
if not file_path.exists():
    print("Precomputing transition probabilities and adjacency matrices...")
    file_dir.mkdir(parents=True, exist_ok=True)
    (
        indptr_undirected,
        indices_undirected,
        data_undirected
    ) = graph_walker.transition_probs(
        G=G_undirected,
        min_degree=True,
        sub_sampling=0.0
    )
    A_directed = nx.adjacency_matrix(G_directed)
    indptr_directed = A_directed.indptr.astype(np.uint32)
    indices_directed = A_directed.indices.astype(np.uint32)
    data_directed = A_directed.data.astype(np.float32)
    processed_data = (
        indptr_undirected,
        indices_undirected,
        data_undirected,
        indptr_directed,
        indices_directed,
        data_directed
    )
    torch.save(processed_data, file_path)
(
    batch.indptr_undirected,
    batch.indices_undirected,
    batch.data_undirected,
    batch.indptr_directed,
    batch.indices_directed,
    batch.data_directed
) = torch.load(file_path)


def compute_cover_times(B_r_dict, start_nodes, walks, restarts):
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(list(enumerate(zip(walks, restarts)))):
        walk_start_node = walk[0]
        B_r_nodes = list(B_r_dict[walk_start_node].nodes)
        visited = np.zeros(len(B_r_nodes), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != walk_start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == walk_start_node, "Restart node must be start node"
                continue
            if i in B_r_nodes:
                visited[B_r_nodes.index(i)] = True
            if j in B_r_nodes:
                visited[B_r_nodes.index(j)] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
            assert t < len(walk) - 2, f"Walk did not cover all nodes, start node: {walk_start_node}, walk idx: {walk_idx}"
    assert np.all(cover_times != -1), "All walks must cover all nodes"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(start_nodes), dtype=np.float32) - 1
    walk_start_nodes = walks[:, 0]
    for walk_start_node in np.unique(walk_start_nodes):
        idxs = np.where(walk_start_nodes == walk_start_node)[0]
        cover_times_per_start_node[list(start_nodes).index(walk_start_node)] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times_per_start_node


def compute_undirected_edge_cover_times(B_r_dict, start_nodes, walks, restarts):
    """Compute the undirected edge cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(list(enumerate(zip(walks, restarts)))):
        walk_start_node = walk[0]
        B_r_edges = list(B_r_dict[walk_start_node].edges)
        visited = np.zeros(len(B_r_edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != walk_start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == walk_start_node, "Restart node must be start node"
                continue
            if (i, j) in B_r_edges:
                visited[B_r_edges.index((i, j))] = True
            if (j, i) in B_r_edges:
                visited[B_r_edges.index((j, i))] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
            assert t < len(walk) - 2, f"Walk did not cover all edges, start node: {walk_start_node}, walk idx: {walk_idx}"
    assert np.all(cover_times != -1), "All walks must cover all edges"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(start_nodes), dtype=np.float32) - 1
    walk_start_nodes = walks[:, 0]
    for walk_start_node in np.unique(walk_start_nodes):
        idxs = np.where(walk_start_nodes == walk_start_node)[0]
        cover_times_per_start_node[list(start_nodes).index(walk_start_node)] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times


np.random.seed(42)
n_tests = 100
start_nodes = np.random.choice(split_idx["test"].tolist(), (n_tests,), replace=False)

radius = 1
work_func = partial(nx.ego_graph, G_undirected, radius=radius, center=True, undirected=True)
with mp.Pool(mp.cpu_count()) as pool:
    results = list(tqdm.tqdm(pool.imap(work_func, start_nodes), total=len(start_nodes)))
    pool.close()
    pool.join()
ego_graphs = dict(zip(start_nodes, results))


n_walks = 100
walk_len = 10000


alpha = 0.3
walks, restarts = graph_walker.random_walks_with_precomputed_probs(
    batch.indptr_undirected,
    batch.indices_undirected,
    batch.data_undirected,
    n_walks=n_walks,
    walk_len=walk_len,
    p=1, q=1, alpha=alpha, k=None,
    no_backtrack=True,
    start_nodes=start_nodes,
    verbose=True
)

cover_times = compute_cover_times(ego_graphs, start_nodes, walks, restarts).astype(float)
print(f"Cover time: {np.mean(cover_times):.2f} ± {np.std(cover_times):.2f}")


alpha = 0.7
walks, restarts = graph_walker.random_walks_with_precomputed_probs(
    batch.indptr_undirected,
    batch.indices_undirected,
    batch.data_undirected,
    n_walks=n_walks,
    walk_len=walk_len,
    p=1, q=1, alpha=alpha, k=None,
    no_backtrack=True,
    start_nodes=start_nodes,
    verbose=True
)

cover_times = compute_cover_times(ego_graphs, start_nodes, walks, restarts).astype(float)
print(f"Cover time: {np.mean(cover_times):.2f} ± {np.std(cover_times):.2f}")
