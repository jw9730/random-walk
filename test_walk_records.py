from pathlib import Path
from typing import List
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing

import graph_walker  # pylint: disable=import-error

from test_walk_statistics import compute_cover_times, compute_edge_cover_times, compute_undirected_edge_cover_times

from src.data import GraphSeparationCSLDataset, GraphSeparationSR16Dataset, GraphSeparationSR25Dataset


DATASET = "CSL"

LABEL_DICT = {
    -1: 'Unknown',
    0: 'Numerical Analysis (cs.NA)',
    1: 'Multimedia (cs.MM)',
    2: 'Logic in Computer Science (cs.LO)',
    3: 'Computers and Society (cs.CY)',
    4: 'Cryptography and Security (cs.CR)',
    5: 'Distributed, Parallel, and Cluster Computing (cs.DC)',
    6: 'Human-Computer Interaction (cs.HC)',
    7: 'Computational Engineering, Finance, and Science (cs.CE)',
    8: 'Networking and Internet Architecture (cs.NI)',
    9: 'Computational Complexity (cs.CC)',
    10: 'Artificial Intelligence (cs.AI)',
    11: 'Multiagent Systems (cs.MA)',
    12: 'General Literature (cs.GL)',
    13: 'Neural and Evolutionary Computing (cs.NE)',
    14: 'Symbolic Computation (cs.SC)',
    15: 'Hardware Architecture (cs.AR)',
    16: 'Computer Vision and Pattern Recognition (cs.CV)',
    17: 'Graphics (cs.GR)',
    18: 'Emerging Technologies (cs.ET)',
    19: 'Systems and Control (cs.SY)',
    20: 'Computational Geometry (cs.CG)',
    21: 'Other Computer Science (cs.OH)',
    22: 'Programming Languages (cs.PL)',
    23: 'Software Engineering (cs.SE)',
    24: 'Machine Learning (cs.LG)',
    25: 'Sound (cs.SD)',
    26: 'Social and Information Networks (cs.SI)',
    27: 'Robotics (cs.RO)',
    28: 'Information Theory (cs.IT)',
    29: 'Performance (cs.PF)',
    30: 'Computation and Language (cs.CL)',
    31: 'Information Retrieval (cs.IR)',
    32: 'Mathematical Software (cs.MS)',
    33: 'Formal Languages and Automata Theory (cs.FL)',
    34: 'Data Structures and Algorithms (cs.DS)',
    35: 'Operating Systems (cs.OS)',
    36: 'Computer Science and Game Theory (cs.GT)',
    37: 'Databases (cs.DB)',
    38: 'Digital Libraries (cs.DL)',
    39: 'Discrete Mathematics (cs.DM)',
}


root = "experiments/data"
if DATASET == "CSL":
    dataset = GraphSeparationCSLDataset(root, split=None, config=None, repeat=1)
    data_list = list(dataset)
elif DATASET == "SR16":
    dataset = GraphSeparationSR16Dataset(root, split=None, config=None, repeat=1)
    data_list = list(dataset)
elif DATASET == "SR25":
    dataset = GraphSeparationSR25Dataset(root, split=None, config=None, repeat=1)
    data_list = list(dataset)
elif DATASET == "counting":
    adj_list, index_dict = np.load("src/data/Counting/raw/graph.npy", allow_pickle=True)
    index_dict = index_dict[()]
    train_idx = index_dict["train"]
    val_idx = index_dict["val"]
    test_idx = index_dict["test"]
    data_list = []
    for adj in adj_list:
        x = torch.ones(len(adj), 1, dtype=torch.int64)
        edge_index = torch.Tensor(np.vstack(np.where(adj != 0))).type(torch.int64)
        data_list.append(Data(x=x, edge_index=edge_index))
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]
    # filter disconnected graphs
    train_data = [data for data in train_data if nx.is_connected(nx.to_undirected(to_networkx(data)))]
    val_data = [data for data in val_data if nx.is_connected(nx.to_undirected(to_networkx(data)))]
    test_data = [data for data in test_data if nx.is_connected(nx.to_undirected(to_networkx(data)))]
    data_list = train_data  # + val_data + test_data
elif DATASET == "ogbn-arxiv":
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    paper_df = pd.read_csv(
        f"{root}/ogbn_arxiv/raw/titleabs.tsv.gz", sep='\t', compression="gzip",
        names=['paper id', 'title', 'abstract'])
    paper_df = paper_df.drop(0,axis=0).dropna()
    paper_df['paper id'] = paper_df['paper id'].astype(int)
    paper_df.set_index('paper id', drop=True, inplace=True)

    index_df = pd.read_csv(
        f"{root}/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz", sep=',', compression="gzip")
    index_df['node idx'] = index_df['node idx'].astype(int)
    index_df['paper id'] = index_df['paper id'].astype(int)
    index_df['title'] = index_df['paper id'].map(paper_df['title'])
    index_df['abstract'] = index_df['paper id'].map(paper_df['abstract'])
    index_df.sort_index(inplace=True)

    title = index_df['title'].tolist()
    abstract = index_df['abstract'].tolist()
    title = [item.capitalize() for item in title]
    abstract = [item.capitalize() for item in abstract]
    abstract = [item + "." if not item.endswith(".") else item for item in abstract]

    target_label = [LABEL_DICT[y] for y in dataset.y.squeeze(1).tolist()]
    input_label = torch.zeros_like(dataset.y).fill_(-1)
    input_label[train_idx] = dataset.y[train_idx]
    input_label = [LABEL_DICT[y] for y in input_label.squeeze(1).tolist()]

    def trunc(string, length, suffix='...'):
        if len(string) <= length:
            return string
        if " " in string[length-1: length]:
            # The given length puts us on a word boundary
            return string[:length].rstrip(' ') + suffix
        # Otherwise add the "tail" of the input, up to just before the first space it contains
        return string[:length] + string[length:].partition(" ")[0] + suffix

    batch = dataset._data  # pylint: disable=protected-access
    batch.title = [trunc(item, 200) for item in title]
    batch.abstract = [trunc(item, 500) for item in abstract]
    batch.input_title = [trunc(item, 100) for item in title]
    batch.input_abstract = [trunc(item, 200) for item in abstract]
    batch.input_label = input_label
    batch.train_idx = train_idx
    batch.val_idx = val_idx
    batch.test_idx = test_idx

    # compute transition probabilities
    min_degree = True
    sub_sampling = 0.
    file_path = Path(f"{root}/ogbn_arxiv/processed/processed_min_deg_True_sub_sample_0.0.pt")
    if not file_path.exists():
        print("Precomputing transition probabilities and adjacency matrices...")
        G_directed = to_networkx(batch)
        G_undirected = nx.to_undirected(G_directed)
        (
            indptr_undirected,
            indices_undirected,
            data_undirected
        ) = graph_walker.transition_probs(
            G=G_undirected,
            min_degree=min_degree,
            sub_sampling=sub_sampling
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
    data_list = [batch]
num_graphs = len(data_list)
dataset = Batch.from_data_list(data_list)
G = to_networkx(dataset)
G = nx.to_undirected(G)


def measure_statistics(min_degree=False, n_walks=2, cover_time=True, edge_cover_time=False, undirected_edge_cover_time=False):
    """Measure statistics of the dataset."""
    disconnected = 0
    for idx, di in enumerate(data_list):
        print(f"\nGraph {idx + 1}/{num_graphs}")

        Gi = to_networkx(di)
        Gi = nx.to_undirected(Gi)

        print(f"Number of nodes: {Gi.number_of_nodes()}")
        print(f"Number of edges: {Gi.number_of_edges()}")
        print(f"Average degree: {np.mean(list(dict(Gi.degree()).values())):.2f}")
        print(f"Average clustering coefficient: {nx.average_clustering(Gi):.2f}")
        try:
            print(f"Average shortest path length: {nx.average_shortest_path_length(Gi):.2f}")
        except nx.NetworkXError:
            print("Graph is not connected")
            disconnected += 1
            continue

        walks, restarts = graph_walker.random_walks(
            Gi,
            n_walks=n_walks,
            walk_len=10000,
            min_degree=min_degree,
            sub_sampling=0.,
            p=1, q=1, alpha=0, k=None,
            no_backtrack=True
        )

        if cover_time:
            cover_times = compute_cover_times(Gi, walks, restarts).astype(float)
            cover_times[cover_times == -1] = np.nan
            cover_times = np.nanmean(cover_times.reshape(n_walks, Gi.number_of_nodes()), axis=0)
            print(f"Cover time: {np.nanmean(cover_times):.2f} ± {np.nanstd(cover_times):.2f}")

        if edge_cover_time:
            edge_cover_times = compute_edge_cover_times(Gi, walks, restarts).astype(float)
            edge_cover_times[edge_cover_times == -1] = np.nan
            edge_cover_times = np.nanmean(edge_cover_times.reshape(n_walks, Gi.number_of_nodes()), axis=0)
            print(f"Edge cover time: {np.nanmean(edge_cover_times):.2f} ± {np.nanstd(edge_cover_times):.2f}")

        if undirected_edge_cover_time:
            undirected_edge_cover_times = compute_undirected_edge_cover_times(Gi, walks, restarts).astype(float)
            undirected_edge_cover_times[undirected_edge_cover_times == -1] = np.nan
            undirected_edge_cover_times = np.nanmean(undirected_edge_cover_times.reshape(n_walks, Gi.number_of_nodes()), axis=0)
            print(f"Undirected edge cover time: {np.nanmean(undirected_edge_cover_times):.2f} ± {np.nanstd(undirected_edge_cover_times):.2f}")

    print(f"\nDisconnected graphs: {disconnected}/{num_graphs}")


def random_walk_text(n_walks: int = 1, include_neighbors: bool = True, seed: int = None) -> List[str]:
    """Sample random walks and convert them to strings."""
    if DATASET != "ogbn-arxiv":
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
    else:
        start_nodes = list(range(100))
        walks, restarts = graph_walker.random_walks_with_precomputed_probs(
            batch.indptr_undirected,
            batch.indices_undirected,
            batch.data_undirected,
            n_walks=n_walks,
            walk_len=100,
            p=1, q=1, alpha=0.1, k=None,
            no_backtrack=True,
            start_nodes=start_nodes,
            seed=seed
        )
        walks_text = graph_walker.as_text_arxiv(
            walks=walks,
            restarts=restarts,
            indptr_undirected=batch.indptr_undirected,
            indices_undirected=batch.indices_undirected,
            indptr_directed=batch.indptr_directed,
            indices_directed=batch.indices_directed,
            title=batch.title,
            abstract=batch.abstract,
            input_title=batch.input_title,
            input_abstract=batch.input_abstract,
            input_label=batch.input_label,
            include_neighbors=True
        )
    return walks_text


# measure_statistics(min_degree=True)

new_walks_string = random_walk_text(n_walks=1, include_neighbors=True)

pretrained_model = "microsoft/deberta-base"
pretrained_hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

# test pretrained tokenizer
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
print(f"\n{pretrained_model} tokenizer truncation rate: {np.mean(truncation_rate):.2f} ± {np.std(truncation_rate):.2f}")
print(f"Estimated walk length: {1000 * (np.mean(1 - truncation_rate)):.2f} ± {1000 * np.std(1 - truncation_rate):.2f} steps")


import pdb; pdb.set_trace()

# test trained tokenizer
vocab_size = 10000  # or pretrained_hf_tokenizer.vocab_size
tokenizer = SentencePieceBPETokenizer()
tokenizer.train_from_iterator(
    random_walk_text(n_walks=1, include_neighbors=True),
    vocab_size=vocab_size,
    special_tokens=["<mask>", "<pad>", "<s>", "</s>", "<unk>", "<sep>", "<cls>"],
    min_frequency=2,
    show_progress=True,
)
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer._tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    sep_token="<sep>",
    pad_token="<pad>",
    cls_token="<cls>",
    mask_token="<mask>"
)
hf_tokenizer._tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[
        ("<s>", hf_tokenizer.bos_token_id),
        ("</s>", hf_tokenizer.eos_token_id),
    ]
)

token_ids = hf_tokenizer(
    new_walks_string,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=max_length
)['input_ids']
decoded_string = hf_tokenizer.batch_decode(token_ids)
truncation_rate = 1 - np.array([len(s_new) / len(s_orig) for s_orig, s_new in zip(new_walks_string, decoded_string)])
print(f"\nTrained tokenizer truncation rate: {np.mean(truncation_rate):.2f} ± {np.std(truncation_rate):.2f}")
print(f"Estimated walk length: {1000 * (np.mean(1 - truncation_rate)):.2f} ± {1000 * np.std(1 - truncation_rate):.2f} steps")

import pdb; pdb.set_trace()
