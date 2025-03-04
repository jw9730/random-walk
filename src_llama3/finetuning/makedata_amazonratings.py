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
    0: 'Excellent 5/5',
    1: 'Great 4.5/5',
    2: 'Good 4/5',
    3: 'Average 3.5/5',
    4: 'Bad 0-3/5',
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
    dataset = torch.load('../data/amazonratings/processed/geometric_data_processed.pt')[0]
    train_idx = torch.where(dataset.train_mask)[0]
    val_idx = torch.where(dataset.val_mask)[0]
    test_idx = torch.where(dataset.test_mask)[0]
    product = dataset.raw_texts
    target_label = [LABEL_DICT[y] for y in dataset.y.tolist()]
    input_label = torch.zeros_like(dataset.y).fill_(-1)
    input_label[train_idx] = dataset.y[train_idx]
    input_label = [LABEL_DICT[y] for y in input_label.tolist()]

    batch = dataset
    batch.product = [trunc(item, 200) for item in product]
    batch.input_product = [trunc(item, 100) for item in product]
    batch.input_label = input_label
    batch.input_label_array = np.array(input_label)
    batch.train_idx = train_idx
    batch.val_idx = val_idx
    batch.test_idx = test_idx
    # compute transition probabilities
    min_degree = True
    sub_sampling = 0.
    root = Path('../data/amazonratings/processed_transition_probs/')
    file_path = root / "processed_min_deg_True_sub_sample_0.0.pt"
    if not file_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        print("Precomputing transition probabilities and adjacency matrices...")
        G = nx.to_undirected(to_networkx(batch))
        (
            indptr,
            indices,
            data
        ) = graph_walker.transition_probs(
            G=G,
            min_degree=min_degree,
            sub_sampling=sub_sampling
        )
        processed_data = (
            indptr,
            indices,
            data
        )
        torch.save(processed_data, file_path)
    (
        batch.indptr,
        batch.indices,
        batch.data,
    ) = torch.load(file_path)
    return batch, target_label


def random_walk_text(batch, all_targets, start_nodes, n_walks, walk_len, alpha, k, no_backtrack, include_neighbors):
    walks, restarts = graph_walker.random_walks_with_precomputed_probs(
        batch.indptr,
        batch.indices,
        batch.data,
        n_walks=n_walks,
        walk_len=walk_len,
        p=1, q=1, alpha=alpha, k=k,
        no_backtrack=no_backtrack,
        start_nodes=start_nodes,
        verbose=False
    )
    walks_targets = [all_targets[idx] for idx in walks[:, 0]]
    walks_text = graph_walker.as_text_amazon(
        walks=walks,
        restarts=restarts,
        indptr=batch.indptr,
        indices=batch.indices,
        product=batch.product,
        input_product=batch.input_product,
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
            "content": f"A walk on the Amazon product co-purchasing network will be given, where products (books, music CDs, DVDs, VHS video tapes) are linked if they are frequently bought together. Predict the average rating given to Product 1 by reviewers. It is one of the following: {', '.join(list(LABEL_DICT.values())[1:])}. Only respond with the answer, do not say any word or explain."
        },
        {
            "role": "user",
            "content": f"A walk on Amazon product co-purchasing network is as follows:\n{text}...\nWhat would be the average rating given to Product 1 by reviewers? It should be one of the following: {', '.join(list(LABEL_DICT.values())[1:])}. Only respond with the answer, do not say any word or explain."
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
    include_neighbors: bool = False
):
    batch, all_targets = prepare_data()

    root = Path('../../experiments/llama3/finetuning_data/amazonratings')
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    train_texts, train_targets = random_walk_text(
        batch,
        all_targets,
        start_nodes=batch.train_idx,
        n_walks=10,
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
