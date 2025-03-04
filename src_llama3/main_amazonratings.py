# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# pylint: disable=import-error
import os
import re
import argparse
from pathlib import Path
import json
import tqdm
import numpy as np
import networkx as nx
import torch
import torch.distributed
from torch_geometric.utils import to_networkx
from vllm import LLM, SamplingParams

import graph_walker


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


def parse_result(result):
    if 'average' in result.lower() or '3.5/5' in result:
        return LABEL_DICT[3]
    if 'great' in result.lower() or '4.5/5' in result:
        return LABEL_DICT[1]
    if 'bad' in result.lower() or '0-3/5' in result:
        return LABEL_DICT[4]
    if 'good' in result.lower():
        return LABEL_DICT[2]
    if 'excellent' in result.lower():
        return LABEL_DICT[0]
    # detect *.*/5 or */5 pattern
    pattern = r'([0-5](?:\.[0-5])?)/5'
    match = re.search(pattern, result)
    if match:
        rating = float(match.group(1))
        if rating >= 4.75:
            return LABEL_DICT[0]
        if rating >= 4.25:
            return LABEL_DICT[1]
        if rating >= 3.75:
            return LABEL_DICT[2]
        if rating >= 3.25:
            return LABEL_DICT[3]
        return LABEL_DICT[4]
    return "Unknown"


def prepare_data(use_val_labels=False):
    dataset = torch.load('data/amazonratings/processed/geometric_data_processed.pt')[0]
    train_idx = torch.where(dataset.train_mask)[0]
    val_idx = torch.where(dataset.val_mask)[0]
    test_idx = torch.where(dataset.test_mask)[0]
    product = dataset.raw_texts
    target_label = [LABEL_DICT[y] for y in dataset.y.tolist()]
    input_label = torch.zeros_like(dataset.y).fill_(-1)
    input_label[train_idx] = dataset.y[train_idx]
    if use_val_labels:
        print("Using validation labels.")
        input_label[val_idx] = dataset.y[val_idx]
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
    root = Path('data/amazonratings/processed_transition_probs/')
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


def random_walk_text(batch, start_nodes, walk_len, alpha, k, no_backtrack, include_neighbors):
    walks, restarts = graph_walker.random_walks_with_precomputed_probs(
        batch.indptr,
        batch.indices,
        batch.data,
        n_walks=1,
        walk_len=walk_len,
        p=1, q=1, alpha=alpha, k=k,
        no_backtrack=no_backtrack,
        start_nodes=start_nodes,
        verbose=False
    )
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
    return walks_text


def text_format(text, tokenizer, max_seq_len):
    tokens = tokenizer.encode(text)[1:]
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
        }
    ]


def main(
    ckpt_dir: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 128,
    alpha: float = 0.0,
    k: int = None,
    no_backtrack: bool = True,
    include_neighbors: bool = True,
    n_preds: int = 10,
    use_val_labels: bool = True,
    save_name: str = '70b_walk',
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = LLM(
        ckpt_dir,
        load_format='safetensors',
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=torch.bfloat16,
        enable_prefix_caching=True
    )
    tokenizer = generator.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_gen_len,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    )

    batch, all_targets = prepare_data(use_val_labels)
    test_data_idx = batch.test_idx.tolist()

    root = Path('../experiments/llama3/amazonratings') / save_name
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

    template_checked_flag = False
    template_should_clean_flag = False

    all_inputs = list(enumerate(test_data_pred_idx))
    print(all_inputs[-1])
    for idx, (i, j) in tqdm.tqdm(all_inputs):
        file_path = root / f"{i}_{j}.json"
        if file_path.exists():
            continue

        batch_dialogs = [text_format(all_texts[idx], tokenizer, max_seq_len)]
        batch_dialogs = tokenizer.apply_chat_template(batch_dialogs, tokenize=False)

        # in some cases, vllm tokenizer adds knowledge cutoff prefix
        if not template_checked_flag:
            template_checked_flag = True
            if "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n" in batch_dialogs[0]:
                print("Template should be cleaned.")
                template_should_clean_flag = True
        if template_should_clean_flag:
            batch_dialogs = [x.replace("Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n", "") for x in batch_dialogs]

        raw_results = generator.generate(batch_dialogs, sampling_params, use_tqdm=False)
        batch_results = [x.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", '') for x in raw_results]

        assert len(batch_dialogs) == len(batch_results) == 1

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
    parser.add_argument('--ckpt_dir', type=str, default='../experiments/checkpoints/Meta-Llama-3-70B-Instruct-HF')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--no_backtrack', action='store_true')
    parser.add_argument('--include_neighbors', action='store_true')
    parser.add_argument('--n_preds', type=int, default=5)
    parser.add_argument('--use_val_labels', action='store_true')
    parser.add_argument('--save_name', type=str, default='8b_walk')
    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        max_seq_len=args.max_seq_len,
        alpha=args.alpha,
        k=args.k,
        no_backtrack=args.no_backtrack,
        include_neighbors=args.include_neighbors,
        n_preds=args.n_preds,
        use_val_labels=args.use_val_labels,
        save_name=args.save_name
    )
