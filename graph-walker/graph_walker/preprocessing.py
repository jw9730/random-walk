import networkx as nx
import numpy as np
from scipy.sparse import diags
from sklearn.preprocessing import normalize


def _weight_node(node, G, m, sub_sampling):
    z = G.degree(node, weight="weight") + 1
    weight = 1 / (z**sub_sampling)
    return weight


def get_normalized_adjacency(G, sub_sampling=0.1):
    A = nx.adjacency_matrix(G)
    A = A.astype(np.float32)
    probs = A.sum(1)
    if sub_sampling != 0:
        m = len(G.edges)
        D_inv = diags([
            _weight_node(node, G, m, sub_sampling)
            for node in G.nodes
        ])
        A = A.dot(D_inv)

    normalize(A, norm="l1", axis=1, copy=False)
    probs = probs / np.clip(probs.sum(), 1, None)
    return A, probs


def get_normalized_minimum_degree(G):
    """Minimum degree local rule
    https://arxiv.org/abs/1604.08326"""
    assert not nx.is_directed(G), "Graph must be undirected for minimum degree local rule"
    A = nx.adjacency_matrix(G)
    A = A.astype(np.float32)
    D = A.sum(1)
    A = A.tocoo()
    i, j = A.nonzero()
    A.data = 1 / np.clip(np.minimum(D[i], D[j]), 1, None)
    A = A.tocsr()
    probs = A.sum(1)

    normalize(A, norm="l1", axis=1, copy=False)
    probs = probs / np.clip(probs.sum(), 1, None)
    return A, probs
