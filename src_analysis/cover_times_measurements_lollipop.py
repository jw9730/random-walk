import time
import numpy as np
import networkx as nx
import tqdm

import graph_walker  # pylint: disable=import-error

import fire


class RandomWalkConfig:
    def __init__(
            self,
            n_walks=20,
            walk_len=10000,
            min_degree=False,
            sub_sampling=0.,
            p=1, q=1, alpha=0, k=None,
            no_backtrack=False,
            start_nodes=None,
            seed=None,
            verbose=True
        ):
        self.n_walks = n_walks
        self.walk_len = walk_len
        self.min_degree = min_degree
        self.sub_sampling = sub_sampling
        self.p = p
        self.q = q
        self.alpha = alpha
        self.k = k
        self.no_backtrack = no_backtrack
        self.start_nodes = start_nodes
        self.seed = seed
        self.verbose = verbose


def compute_random_walks(G, config):
    walks, restarts = graph_walker.random_walks(
        G,
        n_walks=config.n_walks,
        walk_len=config.walk_len,
        min_degree=config.min_degree,
        sub_sampling=config.sub_sampling,
        p=config.p,
        q=config.q,
        alpha=config.alpha,
        k=config.k,
        no_backtrack=config.no_backtrack,
        start_nodes=config.start_nodes,
        seed=config.seed,
        verbose=config.verbose
    )
    return walks, restarts


def compute_stationary_distribution(G, config):
    """Compute the stationary distribution of the random walk."""
    probs = graph_walker.stationary_distribution(
        G,
        min_degree=config.min_degree,
        sub_sampling=config.sub_sampling
    )
    return probs


def compute_cover_times(G, walks, restarts):
    """Compute the cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        visited = np.zeros(len(G.nodes), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert i in G.nodes and j in G.nodes, "Node must exist in graph"
            visited[i] = True
            visited[j] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    assert np.all(cover_times != -1), "All walks must cover all edges"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(G.nodes), dtype=np.float32) - 1
    start_nodes = walks[:, 0]
    for start_node in np.unique(start_nodes):
        idxs = np.where(start_nodes == start_node)[0]
        cover_times_per_start_node[start_node] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times_per_start_node


def compute_undirected_edge_cover_times(G, walks, restarts):
    """Compute the undirected edge cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    edges = list(G.edges)
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        visited = np.zeros(len(edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert (i, j) in edges or (j, i) in edges, "Edge must exist in graph"
            edge_idx = edges.index((i, j)) if (i, j) in edges else edges.index((j, i))
            visited[edge_idx] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    assert np.all(cover_times != -1), "All walks must cover all edges"
    # compute average per start node
    cover_times_per_start_node = np.zeros(len(G.nodes), dtype=np.float32) - 1
    start_nodes = walks[:, 0]
    for start_node in np.unique(start_nodes):
        idxs = np.where(start_nodes == start_node)[0]
        cover_times_per_start_node[start_node] = np.mean(cover_times[idxs])
    assert np.all(cover_times_per_start_node != -1), "All nodes must be used as start nodes"
    return cover_times


def compute_empirical_stationary_distribution(G, walks):
    """Compute the empirical long-term distribution of the random walk."""
    unique, counts = np.unique(walks, return_counts=True)
    assert np.all(unique == np.arange(len(G.nodes)))
    return counts / np.sum(counts)


def run_tests(G, config):
    # generate random walk
    time_start = time.time()
    walks, restarts = compute_random_walks(G, config)
    time_end = time.time()
    seconds = time_end - time_start

    # estimate stationary distribution
    if config.p == 1 and config.q == 1 and config.alpha == 0 and config.k is None and not config.no_backtrack:
        sample_probs = compute_empirical_stationary_distribution(G, walks)
        true_probs = compute_stationary_distribution(G, config)
        error = np.linalg.norm(sample_probs - true_probs)
        print(f"Stationary distribution error: {error:.2e}")

    # compute cover times
    cover_time = np.max(compute_cover_times(G, walks, restarts))

    # compute undirected edge cover times
    undirected_edge_cover_time = np.max(compute_undirected_edge_cover_times(G, walks, restarts))

    # return results
    return (
        seconds,
        cover_time,
        undirected_edge_cover_time,
    )


def get_labels(G, start_nodes, cover_times):
    labels = {node: 0. for node in G.nodes}
    count = {node: 0. for node in G.nodes}
    for start_node, cover_time in zip(start_nodes, cover_times):
        if cover_time == -1:
            continue
        labels[start_node] += cover_time
        count[start_node] += 1
    for node in G.nodes:
        labels[node] = int(labels[node] / count[node]) if count[node] > 0 else -1
    return labels


def run_all_tests(G, name, n_seeds=5):
    print(f"Running tests on {name}")

    # unbiased random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"Unbiased (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    print("Unbiased done\n")

    # unbiased random walks with no backtracking
    for seed in range(n_seeds):
        config = RandomWalkConfig(no_backtrack=True, seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"Unbiased + no backtracking (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    print("Unbiased + no backtracking done\n")

    # MDLR random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(min_degree=True, seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"MDLR (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    print("MDLR done\n")

    # MDLR random walks with no backtracking
    for seed in range(n_seeds):
        config = RandomWalkConfig(min_degree=True, no_backtrack=True, seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"MDLR + no backtracking (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    print("MDLR + no backtracking done\n")

    # node2vec random walks
    for seed in range(n_seeds):
        config = RandomWalkConfig(p=0.25, q=0.25, seed=seed * 100)
        seconds, cover_time, undirected_edge_cover_time = run_tests(G, config)
        print(f"node2vec (seed={seed}): {seconds} sec, cover time= {cover_time}, undirected edge cover time= {undirected_edge_cover_time}")
    print("node2vec done\n")


def main(N=3):
    # lollipop
    clique = nx.complete_graph(2 * N)
    chain = nx.path_graph(N)
    G = nx.disjoint_union(clique, chain)
    G.add_edge(2 * N - 1, 2 * N)
    G = nx.to_undirected(G)
    run_all_tests(G, f"lollipop_(n={G.number_of_nodes()},m={G.number_of_edges()})", n_seeds=5)


if __name__ == '__main__':
    fire.Fire(main)
