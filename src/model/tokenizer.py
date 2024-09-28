import os
from pathlib import Path
from lightning.pytorch.utilities import rank_zero_only
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing

from src.data import Walker

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@rank_zero_only
def train_and_save_tokenizer(walker: Walker, vocab_size) -> None:
    if Path(walker.tokenizer_path).exists():
        print(f'found tokenizer at {walker.tokenizer_path}')
        return
    print(f'training and saving tokenizer at {walker.tokenizer_path}')
    os.makedirs(Path(walker.tokenizer_path).parent, exist_ok=True)
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        walker.get_training_corpus(),
        vocab_size=vocab_size,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
        min_frequency=2,
        show_progress=True
    )
    tokenizer.save(walker.tokenizer_path)


def setup_tokenizer(
    vocab_size,
    max_length,
    backbone: str,
    walker: Walker,
    pretrained=False
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    if pretrained:
        assert not walker.reverse, "Pretrained tokenizer does not support reverse tokenization!"
        tokenizer.add_special_tokens({'additional_special_tokens': walker.special_tokens})
        return tokenizer
    train_and_save_tokenizer(walker, tokenizer.vocab_size if vocab_size == -1 else vocab_size)
    tokenizer = PreTrainedTokenizerFast(
        model_max_length=tokenizer.model_max_length if max_length == -1 else max_length,
        tokenizer_file=walker.tokenizer_path,
        padding_side="left" if walker.reverse else "right"
    )
    tokenizer.add_special_tokens({
        'bos_token': "<s>",
        'eos_token': "</s>",
        'pad_token': "<pad>",
        'unk_token': "<unk>",
        'additional_special_tokens': walker.special_tokens,
    })
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.bos_token_id),
            ("</s>", tokenizer.eos_token_id)
        ]
    )
    return tokenizer
