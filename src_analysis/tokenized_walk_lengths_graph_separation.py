import os
import sys
from typing import List
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, to_undirected
from transformers import AutoTokenizer

import graph_walker  # pylint: disable=import-error

# Get the absolute path to the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from src.data.graph_separation_csl import GraphSeparationCSLDataset


DATASET = "SR25"


if DATASET == "CSL":
    dataset = GraphSeparationCSLDataset(root="../experiments/data", split=None, config=None)
    data_list = list(dataset)
elif DATASET == "SR16":
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
elif DATASET == "SR25":
    dataset = nx.read_graph6("../src/data/SR25/raw/sr251256.g6")
    data_list = []
    for i, data in enumerate(dataset):
        x = torch.ones(data.number_of_nodes(), 1, dtype=torch.long)
        edge_index = to_undirected(torch.tensor(list(data.edges())).transpose(1, 0))
        data_list.append(Data(edge_index=edge_index, x=x))
num_graphs = len(data_list)
dataset = Batch.from_data_list(data_list)
G = to_networkx(dataset)
G = nx.to_undirected(G)


def random_walk_text(n_walks: int = 1, include_neighbors: bool = True, seed: int = None) -> List[str]:
    """Sample random walks and convert them to strings."""
    walks, restarts = graph_walker.random_walks(
        G,
        n_walks=n_walks,
        walk_len=1000,
        min_degree=True,
        sub_sampling=0.,
        p=1, q=1, alpha=0, k=None,
        no_backtrack=True,
        seed=seed
    )
    walks_text = graph_walker.as_text(
        walks=walks,
        restarts=restarts,
        G=G,
        include_neighbors=include_neighbors
    )
    return walks_text

new_walks_string = random_walk_text(n_walks=10, include_neighbors=True)

if len(new_walks_string) > 1000:
    indices = np.random.choice(len(new_walks_string), size=1000, replace=False)
    new_walks_string = [new_walks_string[i] for i in indices]

pretrained_model = "microsoft/deberta-base"
pretrained_hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

max_length = 512
token_ids = pretrained_hf_tokenizer(
    new_walks_string,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=max_length
)['input_ids']

decoded_string = pretrained_hf_tokenizer.batch_decode(token_ids)
truncation_rate = 1 - np.array([len(s_new) / len(s_orig) for s_orig, s_new in zip(new_walks_string, decoded_string)])
estimated_lengths = [string.count("-") for string in decoded_string]

print(f"\n{pretrained_model} tokenizer truncation rate: {np.mean(truncation_rate):.2f} ± {np.std(truncation_rate):.2f}")
print(f"Effective walk length estimated using truncation rate of detokenized text: {1000 * (np.mean(1 - truncation_rate)):.2f} ± {1000 * np.std(1 - truncation_rate):.2f} steps")
print(f"Effective walk length estimated by counting transition symbols (more accurate): {np.mean(estimated_lengths):.2f} ± {np.std(estimated_lengths):.2f} steps")
