from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch.distributed
from torch_geometric.utils import to_networkx
import fire
import datasets

import graph_walker

from llama import Tokenizer  # pylint: disable=import-error


LABEL_DICT = {
    -1: 'Unknown',
    0: 'Case-Based',
    1: 'Genetic Algorithms',
    2: 'Neural Networks',
    3: 'Probabilistic Methods',
    4: 'Reinforcement Learning',
    5: 'Rule Learning',
    6: 'Theory',
}
LABEL_PARSE_DICT = {
    'Case_Based': 'Case-Based',
    'Genetic_Algorithms': 'Genetic Algorithms',
    'Neural_Networks': 'Neural Networks',
    'Probabilistic_Methods': 'Probabilistic Methods',
    'Reinforcement_Learning': 'Reinforcement Learning',
    'Rule_Learning': 'Rule Learning',
    'Theory': 'Theory',
}


def trunc(string, length, suffix='...'):
    if len(string) <= length:
        return string
    if " " in string[length-1: length]:
        # The given length puts us on a word boundary
        return string[:length].rstrip(' ') + suffix
    # Otherwise add the "tail" of the input, up to just before the first space it contains
    return string[:length] + string[length:].partition(" ")[0] + suffix


def prepare_data():
    dataset = torch.load('../data/cora/processed/geometric_data_processed.pt')[0]
    train_idx = torch.where(dataset.train_masks[0])[0]
    val_idx = torch.where(dataset.val_masks[0])[0]
    test_idx = torch.where(dataset.test_masks[0])[0]

    texts = dataset.raw_texts
    title, abstract = [], []
    for text in texts:
        segments = text.split(" : ")
        if len(segments) == 2:
            title.append(segments[0])
            abstract.append(segments[1])
        elif len(segments) == 1:
            segments = segments[0].split(": ")
            if len(segments) == 2:
                title.append(segments[0])
                abstract.append(segments[1])
            elif len(segments) == 1:
                title.append(segments[0])
                abstract.append("Unknown")
            else:
                title.append(segments[0])
                abstract.append(": ".join(segments[1:]))
        else:
            title.append(segments[0])
            abstract.append(" : ".join(segments[1:]))
    title = [item.replace('"', '').strip().strip('.').strip(',').capitalize() for item in title]
    abstract = [item.replace('"', '').strip().strip('.').strip(',').capitalize() for item in abstract]
    abstract = [item + "." if not item.endswith(".") else item for item in abstract]

    target_label = [LABEL_PARSE_DICT[dataset.label_names[y]] for y in dataset.y.tolist()]
    input_label = torch.zeros_like(dataset.y).fill_(-1)
    input_label[train_idx] = dataset.y[train_idx]
    input_label = [LABEL_PARSE_DICT[dataset.label_names[y]] if y != -1 else "Unknown" for y in input_label.tolist()]

    batch = dataset  # pylint: disable=protected-access
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
    root = Path('../data/cora/processed_transition_probs/')
    file_path = root / "processed_min_deg_True_sub_sample_0.0.pt"
    if not file_path.exists():
        root.mkdir(parents=True, exist_ok=True)
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
    return batch, target_label


def random_walk_text(batch, all_targets, start_nodes, n_walks, walk_len, alpha, k, no_backtrack, include_neighbors):
    """
    We reuse the code for ogbn-arxiv random walks, which assumes a directed input graph.
    Cora is an undirected graph, but we simply interpret the edges as if they are directed.
    This is a workaround and fixing this may lead to better results.
    """
    walks, restarts = graph_walker.random_walks_with_precomputed_probs(
        batch.indptr_undirected,
        batch.indices_undirected,
        batch.data_undirected,
        n_walks=n_walks,
        walk_len=walk_len,
        p=1, q=1, alpha=alpha, k=k,
        no_backtrack=no_backtrack,
        start_nodes=start_nodes,
        verbose=False
    )
    walks_targets = [all_targets[idx] for idx in walks[:, 0]]
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
        include_neighbors=include_neighbors,
        verbose=False
    )
    return walks_text, walks_targets


def text_format(text, target, tokenizer: Tokenizer, max_seq_len):
    tokens = tokenizer.encode(text, bos=False, eos=False)
    if len(tokens) > max_seq_len - 450:
        text = tokenizer.decode(tokens[:max_seq_len - 450])
    return [
        {
            "role": "system",
            "content": f"A walk on a citation network will be given. Predict the Category that Paper 1 belongs to. It is one of the following: {', '.join(list(LABEL_DICT.values())[1:])}. Only respond with the answer, do not say any word or explain."
        },
        {
            "role": "user",
            "content": f"A walk on a citation network is as follows:\n{text}...\nWhich Category does Paper 1 belong to? It is one of the following: {', '.join(list(LABEL_DICT.values())[1:])}. Only respond with the answer, do not say any word or explain."
        },
        {
            "role": "assistant",
            "content": target
        }
    ]


def main(
    tokenizer_path: str,
    max_seq_len: int = 3584,
    alpha: float = 0.0,
    k: int = None,
    no_backtrack: bool = False,
    include_neighbors: bool = False,
):
    batch, all_targets = prepare_data()

    root = Path('../../experiments/llama3/finetuning_data/cora_20shot')
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    train_texts, train_targets = random_walk_text(
        batch,
        all_targets,
        start_nodes=batch.train_idx,
        n_walks=100,
        walk_len=200,
        alpha=alpha,
        k=k,
        no_backtrack=no_backtrack,
        include_neighbors=include_neighbors
    )
    val_texts, val_targets = random_walk_text(
        batch,
        all_targets,
        start_nodes=batch.val_idx,
        n_walks=1,
        walk_len=200,
        alpha=alpha,
        k=k,
        no_backtrack=no_backtrack,
        include_neighbors=include_neighbors
    )

    tokenizer = Tokenizer(tokenizer_path)
    train_data = [{'messages': text_format(text, target, tokenizer, max_seq_len)} for text, target in zip(train_texts, train_targets)]
    val_data = [{'messages': text_format(text, target, tokenizer, max_seq_len)} for text, target in zip(val_texts, val_targets)]

    train_data = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data), split='train')
    val_data = datasets.Dataset.from_pandas(pd.DataFrame(data=val_data), split='val')
    dataset = datasets.DatasetDict({'train': train_data, 'val': val_data})
    dataset["train"].to_json(root / "train_dataset.json", orient="records", force_ascii=False)
    dataset["val"].to_json(root / "val_dataset.json", orient="records", force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
