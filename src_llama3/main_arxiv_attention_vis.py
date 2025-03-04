# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# pylint: disable=import-error
import os
from pathlib import Path
import json
import numpy as np
import torch
import torch.distributed

import fire

from llama_vis import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
    load_dir: str = None,
    save_dir: str = None
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    root = Path(save_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    test_data_idx, pred_idx = (2035, 2)

    with open(Path(load_dir) / f"{test_data_idx}_{pred_idx}.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        dialog = json_data['dialog']
        target = json_data['target']

    result, attention_dict = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    torch.distributed.barrier()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:

        result = result[0]

        attention = {}
        for cur_pos, _ in list(attention_dict.items())[1:]:
            attention[cur_pos] = {}
            for layer_idx in list(attention_dict.items())[0][1].keys():
                assert len(attention_dict[cur_pos][layer_idx]) == 1
                attention[cur_pos][layer_idx] = attention_dict[cur_pos][layer_idx][0]

        file_path = root / f"{test_data_idx}_{pred_idx}.json"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump({
                    "dialog": dialog,
                    "result": result,
                    "target": target
                }, file, indent=4)
        file_path = root / f"{test_data_idx}_{pred_idx}_attention.npy"
        if not file_path.exists():
            np.save(file_path, attention)

    torch.distributed.barrier()


if __name__ == "__main__":
    fire.Fire(main)
