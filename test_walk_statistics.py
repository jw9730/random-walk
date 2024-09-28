import pathlib
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import tqdm

import graph_walker  # pylint: disable=import-error


class RandomWalkConfig:
    def __init__(
            self,
            n_walks=10,
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
    return cover_times


def compute_edge_cover_times(G, walks, restarts):
    """Compute the edge cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    edges = set(G.edges)
    edges.update([(j, i) for (i, j) in edges])
    edges = list(edges)
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        visited = np.zeros(len(edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert (i, j) in edges, "Edge must exist in graph"
            edge_idx = edges.index((i, j))
            visited[edge_idx] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    return cover_times


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
    return cover_times


def compute_local_cover_times(G, walks, restarts, radius):
    """Compute the local cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        B = nx.ego_graph(G, start_node, radius)
        nodes = list(B.nodes)
        visited = np.zeros(len(B.nodes), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert i in G.nodes and j in G.nodes, "Node must exist in graph"
            if i in nodes:
                i_node_idx = nodes.index(i)
                visited[i_node_idx] = True
            if j in nodes:
                j_node_idx = nodes.index(j)
                visited[j_node_idx] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    return cover_times


def compute_local_edge_cover_times(G, walks, restarts, radius):
    """Compute the local edge cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        B = nx.ego_graph(G, start_node, radius)
        edges = set(B.edges)
        edges.update([(j, i) for (i, j) in edges])
        edges = list(edges)
        visited = np.zeros(len(edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert (i, j) in G.edges, "Edge must exist in graph"
            if (i, j) in edges:
                edge_idx = edges.index((i, j))
                visited[edge_idx] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    return cover_times


def compute_local_undirected_edge_cover_times(G, walks, restarts, radius):
    """Compute the local undirected edge cover time of a graph."""
    cover_times = np.zeros(walks.shape[0], dtype=np.int32) - 1
    for walk_idx, (walk, restart) in tqdm.tqdm(enumerate(zip(walks, restarts))):
        start_node = walk[0]
        B = nx.ego_graph(G, start_node, radius)
        edges = list(B.edges)
        visited = np.zeros(len(edges), dtype=bool)
        for t, (i, j, rs) in enumerate(zip(walk[:-1], walk[1:], restart[1:])):
            if j != start_node:
                assert not rs, "Restart node must be start node"
            if rs:
                assert j == start_node, "Restart node must be start node"
                continue
            assert (i, j) in G.edges or (j, i) in G.edges, "Edge must exist in graph"
            if (i, j) in edges or (j, i) in edges:
                edge_idx = edges.index((i, j)) if (i, j) in edges else edges.index((j, i))
                visited[edge_idx] = True
            if np.all(visited):
                cover_times[walk_idx] = t + 1
                break
    return cover_times


def compute_empirical_stationary_distribution(G, walks):
    """Compute the empirical long-term distribution of the random walk."""
    unique, counts = np.unique(walks, return_counts=True)
    assert np.all(unique == np.arange(len(G.nodes)))
    return counts / np.sum(counts)


def run_tests(G, config, max_radius=10):
    # compute diameter
    diameter = nx.diameter(G)
    radius = list(range(1, min(diameter, max_radius) + 1))

    # generate random walk
    walks, restarts = compute_random_walks(G, config)
    start_nodes = walks[:, 0]

    # estimate stationary distribution
    if config.p == 1 and config.q == 1 and config.alpha == 0 and config.k is None and not config.no_backtrack:
        sample_probs = compute_empirical_stationary_distribution(G, walks)
        true_probs = compute_stationary_distribution(G, config)
        error = np.linalg.norm(sample_probs - true_probs)
        print(f"Stationary distribution error: {error:.2e}")

    # compute cover times
    cover_times = compute_cover_times(G, walks, restarts)
    local_cover_times = {}
    for r in radius:
        local_cover_times[r] = compute_local_cover_times(G, walks, restarts, r)
    if diameter <= max_radius:
        assert np.all(local_cover_times[diameter] == cover_times)

    # compute edge cover times
    edge_cover_times = compute_edge_cover_times(G, walks, restarts)
    local_edge_cover_times = {}
    for r in radius:
        local_edge_cover_times[r] = compute_local_edge_cover_times(G, walks, restarts, r)
    if diameter <= max_radius:
        assert np.all(local_edge_cover_times[diameter] == edge_cover_times)

    # compute undirected edge cover times
    undirected_edge_cover_times = compute_undirected_edge_cover_times(G, walks, restarts)
    local_undirected_edge_cover_times = {}
    for r in radius:
        local_undirected_edge_cover_times[r] = compute_local_undirected_edge_cover_times(G, walks, restarts, r)
    if diameter <= max_radius:
        assert np.all(local_undirected_edge_cover_times[diameter] == undirected_edge_cover_times)

    # return results
    return (
        start_nodes,
        cover_times,
        edge_cover_times,
        undirected_edge_cover_times,
        radius,
        local_cover_times,
        local_edge_cover_times,
        local_undirected_edge_cover_times
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


def plot_results(G, results, name="test"):
    # unpack results
    (
        start_nodes,
        cover_times,
        edge_cover_times,
        undirected_edge_cover_times,
        radius,
        local_cover_times,
        local_edge_cover_times,
        local_undirected_edge_cover_times
    ) = results

    # make result directory
    pathlib.Path(f'experiments/{name}').mkdir(parents=True, exist_ok=True)

    # get layout
    pos = nx.kamada_kawai_layout(G)

    # plot graph
    nx.draw(G, pos, with_labels=True, node_color='blue', font_color='white')
    plt.savefig(f'experiments/{name}/G.png')
    plt.close()

    # plot cover times
    labels = get_labels(G, start_nodes, cover_times)
    nx.draw(G, pos, labels=labels, font_size=8, node_color='blue', font_color='white')
    plt.savefig(f'experiments/{name}/cover_time.png')
    plt.close()
    for r in radius:
        labels = get_labels(G, start_nodes, local_cover_times[r])
        nx.draw(G, pos, labels=labels, font_size=8, node_color='blue', font_color='white')
        plt.savefig(f'experiments/{name}/local_cover_time_{r}.png')
        plt.close()

    # plot edge cover times
    labels = get_labels(G, start_nodes, edge_cover_times)
    nx.draw(G, pos, labels=labels, font_size=8, node_color='blue', font_color='white')
    plt.savefig(f'experiments/{name}/edge_cover_time.png')
    plt.close()
    for r in radius:
        labels = get_labels(G, start_nodes, local_edge_cover_times[r])
        nx.draw(G, pos, labels=labels, font_size=8, node_color='blue', font_color='white')
        plt.savefig(f'experiments/{name}/local_edge_cover_time_{r}.png')
        plt.close()

    # plot undirected edge cover times
    labels = get_labels(G, start_nodes, undirected_edge_cover_times)
    nx.draw(G, pos, labels=labels, font_size=8, node_color='blue', font_color='white')
    plt.savefig(f'experiments/{name}/undirected_edge_cover_time.png')
    plt.close()
    for r in radius:
        labels = get_labels(G, start_nodes, local_undirected_edge_cover_times[r])
        nx.draw(G, pos, labels=labels, font_size=8, node_color='blue', font_color='white')
        plt.savefig(f'experiments/{name}/local_undirected_edge_cover_time_{r}.png')
        plt.close()


def run_all_tests(G, name, alpha, k):
    # natural random walks
    config = RandomWalkConfig(seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_natural")

    # natural random walks with restarts
    config = RandomWalkConfig(alpha=alpha, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_natural_alpha{alpha}")

    # natural random walks with periodic restarts
    config = RandomWalkConfig(k=k, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_natural_k{k}")

    # natural random walks with no backtracking
    config = RandomWalkConfig(no_backtrack=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_natural_no_backtrack")

    # natural random walks with restarts and no backtracking
    config = RandomWalkConfig(alpha=alpha, no_backtrack=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_natural_alpha{alpha}_no_backtrack")

    # natural random walks with periodic restarts and no backtracking
    config = RandomWalkConfig(k=k, no_backtrack=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_natural_k{k}_no_backtrack")

    # minimum degree random walks
    config = RandomWalkConfig(min_degree=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_min_degree")

    # minimum degree random walks with restarts
    config = RandomWalkConfig(min_degree=True, alpha=alpha, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_min_degree_alpha{alpha}")

    # minimum degree random walks with periodic restarts
    config = RandomWalkConfig(min_degree=True, k=k, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_min_degree_k{k}")

    # minimum degree random walks with no backtracking
    config = RandomWalkConfig(min_degree=True, no_backtrack=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_min_degree_no_backtrack")

    # minimum degree random walks with restarts and no backtracking
    config = RandomWalkConfig(min_degree=True, alpha=alpha, no_backtrack=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_min_degree_alpha{alpha}_no_backtrack")

    # minimum degree random walks with periodic restarts and no backtracking
    config = RandomWalkConfig(min_degree=True, k=k, no_backtrack=True, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_min_degree_k{k}_no_backtrack")

    # node2vec random walks
    config = RandomWalkConfig(p=0.25, q=0.25, seed=None)
    results = run_tests(G, config)
    plot_results(G, results, name=f"{name}_node2vec")

    print("Done")


def main():
    # clique
    G = nx.complete_graph(10)
    G = nx.to_undirected(G)
    run_all_tests(G, f"clique_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # grid
    G = nx.grid_2d_graph(3, 3)
    G = nx.convert_node_labels_to_integers(G)
    G = nx.to_undirected(G)
    run_all_tests(G, f"grid_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # tree
    G = nx.full_rary_tree(3, 13)
    G = nx.to_undirected(G)
    run_all_tests(G, f"tree_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # lollipop
    clique = nx.complete_graph(8)
    chain = nx.path_graph(4)
    G = nx.disjoint_union(clique, chain)
    G.add_edge(7, 8)
    G = nx.to_undirected(G)
    run_all_tests(G, f"lollipop_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # clique-star
    G = nx.complete_graph(5)
    for i in range(5):
        G.add_node(5 + i)
        G.add_edge(i, 5 + i)
    G = nx.to_undirected(G)
    run_all_tests(G, f"clique_star_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # glitter star
    G = nx.star_graph(6)
    for i in range(6):
        G.add_node(7 + i)
        G.add_edge(1 + i, 7 + i)
    G = nx.to_undirected(G)
    run_all_tests(G, f"glitter_star_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # double star
    star_1 = nx.star_graph(5)
    star_2 = nx.star_graph(5)
    G = nx.disjoint_union(star_1, star_2)
    G.add_edge(0, 6)
    G = nx.to_undirected(G)
    run_all_tests(G, f"double_star_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # erdos-renyi graph
    G = nx.erdos_renyi_graph(100, 0.05, seed=42)
    G = nx.to_undirected(G)
    run_all_tests(G, f"erdos_renyi_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)

    # barabasi-albert preferential attachment graph
    G = nx.barabasi_albert_graph(100, 5, seed=42)
    G = nx.to_undirected(G)
    run_all_tests(G, f"barabasi_albert_(n={G.number_of_nodes()},m={G.number_of_edges()})", alpha=0.1, k=10)


if __name__ == '__main__':
    main()
