from typing import Tuple, Union, List, Any, Iterator, Dict
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx

import graph_walker  # pylint: disable=import-error

from .ds_builder import DatasetBuilder


class Walker:
    """Base class for random walk operations."""
    ds_builder: DatasetBuilder = NotImplemented
    out_dim: int = NotImplemented
    metric_name: Union[str, List[str]] = NotImplemented
    tokenizer_path: str
    special_tokens: list = []

    def __init__(self, config):
        self.walk_length = getattr(config, 'walk_length', 1000)
        self.min_degree = config.walk_type == 'min_degree'
        self.sub_sampling = getattr(config, 'sub_sampling', 0.)
        self.p = getattr(config, 'node2vec_p', 1)
        self.q = getattr(config, 'node2vec_q', 1)
        self.alpha = getattr(config, 'restart_prob', 0)
        self.k = getattr(config, 'restart_period', None)
        self.no_backtrack = getattr(config, 'no_backtrack', False)
        self.neighbors = getattr(config, 'neighbors', False)
        self.tokenizer_n_walks_per_node = getattr(config, 'tokenizer_n_walks_per_node', 1)
        self.tokenizer_path = f"experiments/{config.load_dir}/" \
            + f"{config.dataset.replace('/', '_')}/{self.name}/tokenizer.json"
        self.reverse = getattr(config, 'reverse', False)
        self.n_walks = getattr(config, 'n_walks', 1)
        self.eval_n_walks = getattr(config, 'eval_n_walks', self.n_walks)
        if config.test_mode:
            self.eval_n_walks = getattr(config, 'test_n_walks', self.eval_n_walks)

    @property
    def name(self):
        return f"walk_length={self.walk_length}" \
            + (",min_deg" if self.min_degree else "") \
            + (",sub_sample" if self.sub_sampling else "") \
            + (f",(p,q)=({self.p},{self.q})" if (self.p != 1 or self.q != 1) else "") \
            + (f",alpha={self.alpha}" if self.alpha > 0 else "") \
            + (f",k={self.k}" if self.k else "") \
            + (",no_backtrack" if self.no_backtrack else "") \
            + (",neighbors" if self.neighbors else "") \
            + (f",tok_n_walks={self.tokenizer_n_walks_per_node}" if
               self.tokenizer_n_walks_per_node > 1 else "")

    def __repr__(self):
        return f"Walker({self.name})"

    def register_ds_builder(self, ds_builder: DatasetBuilder):
        """Register dataset builder."""
        self.ds_builder = ds_builder

    def parse_target(self, batch: Batch) -> Tuple[Union[Tensor, Batch], int, Tensor]:
        """Parse target from batch."""
        target = getattr(batch, 'y')
        n_targets = getattr(batch, 'num_graphs')
        target_ids = getattr(batch, 'batch')
        return target, n_targets, target_ids

    def get_start_nodes(
        self,
        n_targets,
        target_ids: Tensor,
        n_walks_per_target
    ) -> Tuple[np.ndarray, Tensor]:
        """Sample start nodes for random walks."""
        # sample start nodes
        probs = one_hot(target_ids).transpose(1, 0)  # pylint: disable=not-callable
        probs = probs / probs.sum(dim=1, keepdim=True)
        determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        start_nodes = torch.multinomial(probs, n_walks_per_target, replacement=True)
        torch.use_deterministic_algorithms(determinism)
        start_nodes = start_nodes.flatten()
        assert start_nodes.shape == (n_targets * n_walks_per_target,)
        # expand target_ids
        target_ids = torch.arange(n_targets, device=target_ids.device)
        target_ids = target_ids.repeat_interleave(n_walks_per_target)
        return start_nodes.cpu().numpy(), target_ids

    def random_walk_text(
        self,
        batch: Batch,
        n_walks=1,
        start_nodes=None,
        seed=None,
        verbose=False
    ) -> List[str]:
        """Sample random walks and convert them to a list of strings.
        Caution: Absence of node and edge attributes is assumed.
        """
        assert batch.is_undirected()
        G = nx.to_undirected(to_networkx(batch))
        walks, restarts = graph_walker.random_walks(
            G=G,
            n_walks=n_walks,
            walk_len=self.walk_length,
            min_degree=self.min_degree,
            sub_sampling=self.sub_sampling,
            p=self.p,
            q=self.q,
            alpha=self.alpha,
            k=self.k,
            no_backtrack=self.no_backtrack,
            start_nodes=start_nodes,
            seed=seed,
            verbose=verbose,
        )
        walks_text = graph_walker.as_text(
            walks=walks,
            restarts=restarts,
            G=G,
            include_neighbors=self.neighbors,
            verbose=verbose,
        )
        return walks_text

    def get_training_corpus(self) -> Iterator[str]:
        """Get the training corpus for the tokenizer."""
        batch = Batch.from_data_list(list(self.ds_builder.train_dataset()))
        yield self.random_walk_text(batch, self.tokenizer_n_walks_per_node, verbose=True)

    @staticmethod
    def pool_to_target(
        source: Tensor,
        n_targets: int,
        target_ids: Tensor,
        reduce='mean'
    ) -> Tensor:
        """Pool source to target."""
        assert target_ids.max().item() <= n_targets <= target_ids.max().item() + 1
        assert source.ndim == 2
        dim = source.shape[1]
        output = torch.zeros(n_targets + 1, dim, dtype=source.dtype, device=source.device)
        target_ids = target_ids[:, None].expand_as(source)
        determinism = torch.are_deterministic_algorithms_enabled()
        if determinism:
            torch.use_deterministic_algorithms(False)
        output.scatter_reduce_(
            dim=0, index=target_ids, src=source, reduce=reduce, include_self=False)
        if determinism:
            torch.use_deterministic_algorithms(True)
        output = output[:-1]
        return output

    def criterion(self, y_hat: Any, y: Any) -> Tensor:
        """Compute loss between prediction y_hat and target y"""
        raise NotImplementedError

    def evaluator(self, y_hat: Any, y: Any) -> Union[Tensor, Dict]:
        """Compute metric between prediction y_hat and target y"""
        raise NotImplementedError
