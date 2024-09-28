from typing import Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from torch_geometric.data import Batch
from transformers import AutoConfig, AutoModel
from transformers import BertModel, RobertaModel, DebertaModel, MambaModel
from transformers.models.deberta.modeling_deberta import ContextPooler as DebertaContextPooler
from transformers.modeling_outputs import ModelOutput

from src.data import Walker
from .checkpoints import HF_CHECKPOINTS
from .tokenizer import setup_tokenizer

TextTransformer = (BertModel, RobertaModel, DebertaModel)
TextSSM = (MambaModel,)


def setup_backbone(backbone, pretrained=True) -> nn.Module:
    """Initialize backbone architecture and load pretrained weights"""
    if backbone in HF_CHECKPOINTS:
        if pretrained:
            return AutoModel.from_pretrained(backbone)
        return AutoModel.from_config(AutoConfig.from_pretrained(backbone))
    raise NotImplementedError(f"Backbone ({backbone}) not supported!")


def nearest_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class Model(nn.Module):
    """Text model (optionally pretrained)"""
    def __init__(
        self,
        walker: Walker,
        backbone,
        dropout,
        att_dropout,
        head_dropout,
        vocab_size,
        max_length,
        pretrained,
        pretrained_tokenizer,
        is_compiled,
        deberta_use_pooler,
        debug_mode
    ):
        super().__init__()
        self.walker = walker
        self.debug_mode = debug_mode
        # setup backbone
        self.backbone = setup_backbone(backbone, pretrained)
        if isinstance(self.backbone, DebertaModel):
            if deberta_use_pooler:
                self.backbone.pooler = DebertaContextPooler(self.backbone.config)
            else:
                self.backbone.pooler = nn.Identity()
        assert isinstance(self.backbone, (TextTransformer, TextSSM))
        if dropout is not None:
            print(f"Setting hidden_dropout_prob to {dropout}")
            self.backbone.config.hidden_dropout_prob = dropout
        if att_dropout is not None:
            print(f"Setting attention_probs_dropout_prob to {att_dropout}")
            self.backbone.config.attention_probs_dropout_prob = att_dropout
        # setup tokenizer
        self.tokenizer = setup_tokenizer(
            vocab_size,
            max_length,
            backbone,
            walker,
            pretrained_tokenizer
        )
        self.max_length = self.tokenizer.model_max_length if max_length == -1 else max_length
        # adjust embeddings to match tokenizer
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        self.pretrained_tokenizer = pretrained_tokenizer
        if not pretrained_tokenizer:
            self.backbone.embeddings.word_embeddings.reset_parameters()
        # adjust special tokens
        assert self.backbone.embeddings.word_embeddings.padding_idx in \
            (None, self.tokenizer.pad_token_id)
        self.backbone.config.pad_token_id = self.tokenizer.pad_token_id
        self.backbone.config.bos_token_id = self.tokenizer.bos_token_id
        self.backbone.config.eos_token_id = self.tokenizer.eos_token_id
        self.backbone.config.sep_token_id = self.tokenizer.sep_token_id
        # setup prediction head
        self.head_dropout = nn.Dropout(head_dropout, inplace=True)
        self.head = nn.Linear(self.backbone.config.hidden_size, walker.out_dim)
        # setup semi-supervised learning
        if isinstance(walker, NodeClassificationArxivWalker):
            dataset = walker.ds_builder.train_dataset()
            self.label_dict = dataset.label_dict
            self.data = dataset._data
            self.data.target_label = torch.zeros_like(self.data.y).fill_(-1)
            self.data.target_label[dataset.train_idx] = self.data.y[dataset.train_idx]
            self.data.input_label = [self.label_dict[y][:-8].lower() if y != -1 else "unknown"
                                     for y in self.data.target_label.squeeze(1).tolist()]
            self.use_pseudo_label = dataset.use_pseudo_label
            assert not self.use_pseudo_label
        # setup compile variables
        self.is_compiled = is_compiled
        self.compiled_size = 0
        self.eval_compiled_size = 0

    def pretrained_parameters(self):
        if isinstance(self.backbone, TextTransformer):
            modules = [self.backbone.encoder,]
        else:
            assert isinstance(self.backbone, TextSSM)
            modules = [self.backbone.layers, self.backbone.norm_f]
        if self.pretrained_tokenizer:
            modules.append(self.backbone.embeddings)
        for module in modules:
            yield from module.parameters()

    def scratch_parameters(self):
        if isinstance(self.backbone, TextTransformer):
            modules = [self.backbone.pooler, self.head]
        else:
            assert isinstance(self.backbone, TextSSM)
            modules = [self.head,]
        if not self.pretrained_tokenizer:
            modules.append(self.backbone.embeddings)
        for module in modules:
            yield from module.parameters()

    @torch.no_grad()
    @torch.compiler.disable
    def tokenize(self, batch, n_targets, target_ids: Tensor):
        # get random walk text
        n_walks = self.walker.n_walks if self.training else self.walker.eval_n_walks
        if isinstance(self.walker, NodeClassificationArxivWalker):
            start_nodes, target_ids = self.walker.get_start_nodes(batch, target_ids, n_walks)
            text = self.walker.random_walk_text(self.data, 1, start_nodes)
        else:
            start_nodes, target_ids = self.walker.get_start_nodes(n_targets, target_ids, n_walks)
            text = self.walker.random_walk_text(batch, 1, start_nodes)
        # tokenize text
        encoded_input = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        encoded_input = encoded_input.to(target_ids.device)
        input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
        if self.debug_mode:
            decoded_text = self.tokenizer.batch_decode(input_ids)
            import pdb; pdb.set_trace()
        # perform necessary postprocessing
        if self.walker.reverse:
            input_ids, attention_mask = input_ids.flip(1), attention_mask.flip(1)
        if self.is_compiled:
            return self._pad_for_compile(input_ids, attention_mask, n_targets, target_ids)
        return input_ids, attention_mask, target_ids

    def forward(self, batch) -> Tuple[Tensor, Union[Tensor, Batch]]:
        if isinstance(self.walker, NodeClassificationArxivWalker):
            target, n_targets, target_ids = self.walker.parse_target(
                batch, self.data.y, self.data.target_label, self.label_dict, self.training)
        else:
            target, n_targets, target_ids = self.walker.parse_target(batch)
        input_ids, attention_mask, target_ids = self.tokenize(batch, n_targets, target_ids)
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        walk_state = self.parse_output(output, input_ids)
        walk_pred = self.head(self.head_dropout(walk_state))
        pred = self.walker.pool_to_target(walk_pred, n_targets, target_ids, reduce='mean')
        if isinstance(self.walker, NodeClassificationArxivWalker):
            if self.use_pseudo_label and self.training:
                self._update_target_label(pred, batch)
        return pred, target

    def parse_output(self, output: ModelOutput, input_ids: Tensor) -> Tensor:
        if isinstance(self.backbone, DebertaModel):
            if isinstance(self.backbone.pooler, DebertaContextPooler):
                return self.backbone.pooler(output.last_hidden_state)
            return output.last_hidden_state[:, 0, :]
        if isinstance(self.backbone, TextTransformer):
            return output.pooler_output
        assert isinstance(self.backbone, TextSSM)
        batch_size = input_ids.shape[0]
        seq_lengths = torch.eq(input_ids, self.backbone.config.pad_token_id).int().argmax(-1) - 1
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[torch.arange(batch_size, device=input_ids.device), seq_lengths]

    def _pad_for_compile(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            n_targets,
            target_ids: Tensor
        ):
        """Pad batch size to nearest power of 2 to avoid recompilations."""
        assert target_ids.shape[0] == input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.ndim == 2 and attention_mask.ndim == 2
        cur_size = input_ids.shape[0]
        new_size = nearest_power_of_2(cur_size)
        if self.training:
            if new_size > self.compiled_size:
                print(f"Train: Compiled size updated {self.compiled_size} -> {new_size}")
                self.compiled_size = new_size
            pad_size = self.compiled_size - cur_size
        else:
            if new_size > self.eval_compiled_size:
                print(f"Eval: Compiled size updated {self.eval_compiled_size} -> {new_size}")
                self.eval_compiled_size = new_size
            pad_size = self.eval_compiled_size - cur_size
        input_ids = pad(input_ids, (0, 0, 0, pad_size), "constant", self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, (0, 0, 0, pad_size), "constant", 0)
        target_ids = pad(target_ids, (0, pad_size), "constant", n_targets)
        return input_ids, attention_mask, target_ids
