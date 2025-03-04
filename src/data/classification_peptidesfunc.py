# pylint: disable=protected-access,too-many-locals,unused-argument,line-too-long,too-many-instance-attributes,too-many-arguments,not-callable
import warnings
from typing import List, Dict
import networkx as nx
import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx, to_torch_csr_tensor
from torchmetrics.functional.classification import multilabel_average_precision, binary_average_precision
import periodictable

import graph_walker  # pylint: disable=import-error

from .walker import Walker
from .ogb_features import atom_feature_vector_to_dict, bond_feature_vector_to_dict


atomic_number_to_symbol = {el.number: el.symbol for el in periodictable.elements if el.number}


class ClassificationPeptidesfuncWalker(Walker):
    def __init__(self, config):
        super().__init__(config)
        self.out_dim = 10
        self.global_test_metric = not config.compile
        self.metric_name = 'average-precision' if self.global_test_metric else 'accuracy'

    def criterion(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loss/multilabel_classification_loss.py
        """
        is_labeled = torch.eq(y, y)
        return nn.functional.binary_cross_entropy_with_logits(y_hat[is_labeled], y[is_labeled].to(y_hat.dtype))

    def evaluator(self, y_hat: Tensor, y: Tensor) -> Dict:
        preds, target = y_hat, y
        # compute metrics
        preds = (preds > 0).to(target.dtype)
        metric_val = (preds == target).float().mean()
        batch_size = target.size(0)
        return {
            'metric_sum': metric_val * batch_size,
            'metric_count': batch_size
        }

    def global_evaluator(self, y_hat: List, y: List) -> Dict:
        """Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metric_wrapper.py
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/logger.py
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metrics_ogb.py
        """
        preds, target = torch.cat(y_hat, dim=0), torch.cat(y, dim=0)
        # torchmetrics expects probability preds and long targets
        preds = torch.sigmoid(preds)
        target = target.long()
        # compute metrics
        determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        if torch.isnan(target).any():
            warnings.warn("NaNs in targets, falling back to slow evaluation.")
            # Compute the metric for each column, and output nan if there's an error on a given column
            target_nans = torch.isnan(target)
            target_list = [target[..., ii][~target_nans[..., ii]] for ii in range(target.shape[-1])]
            preds_list = [preds[..., ii][~target_nans[..., ii]] for ii in range(preds.shape[-1])]
            target = target_list
            preds = preds_list
            metric_val = []
            for p, t in zip(preds, target):
                # below is tested to be equivalent with OGB metric
                # https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metrics_ogb.py
                res = binary_average_precision(p, t)
                metric_val.append(res)
            # Average the metric
            # PyTorch 1.10
            metric_val = torch.nanmean(torch.stack(metric_val))
            # PyTorch <= 1.9
            # x = torch.stack(metric_val)
            # metric_val = torch.div(torch.nansum(x), (~torch.isnan(x)).count_nonzero())
        else:
            # below is tested to be equivalent with OGB metric
            # https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metrics_ogb.py
            metric_val = multilabel_average_precision(preds, target, num_labels=self.out_dim)
        torch.use_deterministic_algorithms(determinism)
        return {
            'metric_sum': metric_val,
            'metric_count': 1,
        }

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
        node_attr = batch.x.cpu().numpy()
        edge_attr = to_torch_csr_tensor(batch.edge_index, batch.edge_attr)
        indptr_edge_attr = edge_attr.crow_indices().cpu().numpy()
        indices_edge_attr = edge_attr.col_indices().cpu().numpy()
        data_edge_attr = edge_attr.values().cpu().numpy()

        def node_attr_formatter(x):
            x_dict = atom_feature_vector_to_dict(list(x))
            atomic_num = x_dict['atomic_num']
            element_name = atomic_number_to_symbol[atomic_num]
            chirality = x_dict["chirality"]
            degree = x_dict["degree"]
            formal_charge = x_dict["formal_charge"]
            num_h = x_dict["num_h"]
            num_rad_e = x_dict["num_rad_e"]
            return (
                f"[{element_name}"
                f"{f',{chirality}' if chirality != '' else ''}"
                f"{f',d:{degree}' if degree != 1 else ''}"
                f"{f',+:{formal_charge}' if formal_charge != 0 else ''}"
                f"{f',H:{num_h}' if num_h != 0 else ''}"
                f"{f',e:{num_rad_e}' if num_rad_e != 0 else ''}"
                f",{x_dict['hybridization']}"
                f"{',*' if x_dict['is_aromatic'] else ''}"
                f"{',r' if x_dict['is_in_ring'] else ''}]"
            )

        def edge_attr_formatter(x):
            x_dict = bond_feature_vector_to_dict(list(x))
            # stereochemistry = x_dict['bond_stereo']
            return (
                f"{x_dict['bond_type']}"
                f"{'π' if x_dict['is_conjugated'] else ''}"
            )

        node_attr_str = list(map(node_attr_formatter, node_attr))
        indptr_edge_attr = indptr_edge_attr.astype(np.uint32)
        indices_edge_attr = indices_edge_attr.astype(np.uint32)
        edge_attr_str = list(map(edge_attr_formatter, data_edge_attr))

        walks_text = graph_walker.as_text_peptides(
            walks=walks,
            restarts=restarts,
            G=G,
            node_attr_str=node_attr_str,
            indptr_edge_attr=indptr_edge_attr,
            indices_edge_attr=indices_edge_attr,
            edge_attr_str=edge_attr_str,
            include_neighbors=self.neighbors,
            verbose=verbose,
        )
        return walks_text


class ClassificationPeptidesfuncDataset(LRGBDataset):
    def __init__(self, root, split, config):
        super().__init__(root, name='Peptides-func', split=split)
