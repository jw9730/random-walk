# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# pylint: disable=import-error
import os
import argparse
from pathlib import Path
import json
import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.distributed
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

import graph_walker


LLAMA_3_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


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


def trunc(string, length, suffix='...'):
    if len(string) <= length:
        return string
    if " " in string[length-1: length]:
        # The given length puts us on a word boundary
        return string[:length].rstrip(' ') + suffix
    # Otherwise add the "tail" of the input, up to just before the first space it contains
    return string[:length] + string[length:].partition(" ")[0] + suffix


def prepare_data():
    root = "../../experiments/data"
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    paper_df_file_path = Path("../data/ogbn_arxiv/raw/titleabs.tsv.gz")
    assert paper_df_file_path.exists(), f"File not found: {paper_df_file_path}"
    paper_df = pd.read_csv(paper_df_file_path, sep='\t', compression="gzip",
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
    root = Path('../data/ogbn_arxiv/processed_transition_probs/')
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


def random_walk_text(batch, start_nodes, walk_len, alpha, k, no_backtrack, include_neighbors):
    walks, restarts = graph_walker.random_walks_with_precomputed_probs(
        batch.indptr_undirected,
        batch.indices_undirected,
        batch.data_undirected,
        n_walks=1,
        walk_len=walk_len,
        p=1, q=1, alpha=alpha, k=k,
        no_backtrack=no_backtrack,
        start_nodes=start_nodes,
        verbose=False
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
        include_neighbors=include_neighbors,
        verbose=False
    )
    return walks_text


def text_format(text, tokenizer, max_seq_len):
    tokens = tokenizer.encode(text)[1:]
    if len(tokens) > max_seq_len - 450:
        text = tokenizer.decode(tokens[:max_seq_len - 450])
    return [
        {
            "role": "system",
            "content": f"A walk on the arXiv citation network will be given. Predict the arXiv CS sub-category Paper 1 belongs to. It is one of the following: {', '.join(list(LABEL_DICT.values())[1:])}. Only respond with the answer, do not say any word or explain."
        },
        {
            "role": "user",
            "content": f"A walk on arXiv citation network is as follows:\n{text}...\nWhich arXiv CS sub-category does Paper 1 belong to? Only respond with the answer, do not say any word or explain."
        }
    ]


def main(
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 3584,
    max_gen_len: int = 5,
    alpha: float = 0.0,
    k: int = None,
    no_backtrack: bool = True,
    include_neighbors: bool = True,
    n_preds: int = 10,
    ckpt_dir: str = None,
    save_dir: str = None,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = AutoPeftModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.bfloat16,
        quantization_config= {"load_in_4bit": True},
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    batch, all_targets = prepare_data()

    test_data_idx = batch.test_idx.tolist()

    root = Path(save_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    test_data_pred_idx = []
    for i in test_data_idx:
        for j in range(n_preds):
            test_data_pred_idx.append((i, j))

    start_nodes = np.asarray([i for i, j in test_data_pred_idx]).astype(np.uint32)
    all_texts = random_walk_text(
        batch,
        start_nodes=start_nodes,
        walk_len=200,
        alpha=alpha,
        k=k,
        no_backtrack=no_backtrack,
        include_neighbors=include_neighbors
    )

    for idx, (i, j) in tqdm.tqdm(list(enumerate(test_data_pred_idx))):
        file_path = root / f"{i}_{j}.json"
        if file_path.exists():
            continue

        batch_dialogs = [text_format(all_texts[idx], tokenizer, max_seq_len)]
        input_ids = tokenizer.apply_chat_template(batch_dialogs, add_generation_prompt=True, return_tensors="pt").to(generator.device)
        raw_results = generator.generate(input_ids, max_new_tokens=max_gen_len, eos_token_id=tokenizer.encode("<|eot_id|>")[0], pad_token_id=tokenizer.pad_token_id, do_sample=False, temperature=None, top_p=None)
        batch_results = [tokenizer.decode(raw_results[0][input_ids.shape[-1]:], skip_special_tokens=True)]

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump({
                        "dialog": batch_dialogs[0],
                        "result": batch_results[0],
                        "target": all_targets[i]
                    }, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--no_backtrack', action='store_true')
    parser.add_argument('--include_neighbors', action='store_true')
    parser.add_argument('--n_preds', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        k=args.k,
        no_backtrack=args.no_backtrack,
        include_neighbors=args.include_neighbors,
        n_preds=args.n_preds,
        ckpt_dir=args.ckpt_dir,
        save_dir=args.save_dir
    )
