import os
import os.path as osp
from typing import Tuple, Dict, Union
from pathlib import Path
from distutils.dir_util import copy_tree
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.utils import to_networkx

from .utils_regression_counting import process
from .walker import Walker


HOM_NAMES = [
    "boat", "chordal6", "chordal4_1", "chordal4_4", "chordal5_13", "chordal5_31", "chordal5_24"]
ISO_NAMES = [
    "cycle3", "cycle4", "cycle5", "cycle6", "cycle7", "cycle8", "chordal4", "chordal5"]
NORMALIZE = {
    "cycle3:v": 3, "cycle3:e": 6,
    "cycle4:v": 4, "cycle4:e": 8,
    "cycle5:v": 5, "cycle5:e": 10,
    "cycle6:v": 6, "cycle6:e": 12,
    "cycle7:v": 7, "cycle7:e": 14,
    "cycle8:v": 8, "cycle8:e": 16,
    "chordal4:v": 4, "chordal4:e": 10,
    "chordal5:v": 5, "chordal5:e": 14,
}


class RegressionCountingWalker(Walker):
    def __init__(self, config):
        super().__init__(config)
        assert config.task_name in HOM_NAMES + ISO_NAMES
        assert config.task_level in ["g", "v", "e"]
        if config.task_level == "e":
            raise NotImplementedError("Edge-level tasks are not supported yet.")
        self.out_dim = 1
        self.task_id = f"{config.task_name}:{config.task_level}"
        task_type = "hom" if config.task_name in HOM_NAMES else "iso"
        self.metric_name = f"{config.task_name}_{task_type}_{config.task_level}_mae"

    def parse_target(self, batch: Batch) -> Tuple[Tensor, int, Tensor]:
        """Parse target from batch."""
        if 'g' in self.task_id:
            target = getattr(batch, self.task_id)
            target_ids = getattr(batch, 'batch')
            n_targets = getattr(batch, 'num_graphs')
            return target, n_targets, target_ids
        if 'v' in self.task_id:
            target = getattr(batch, self.task_id)
            n_targets = target.shape[0]
            assert n_targets == batch.x.shape[0]
            target_ids = torch.arange(n_targets, device=target.device)
            return batch, n_targets, target_ids
        raise NotImplementedError(f"Task {self.task_id} not supported")

    def get_start_nodes(
        self,
        n_targets,
        target_ids: Tensor,
        n_walks_per_target=1
    ) -> Tuple[np.ndarray, Tensor]:
        if 'g' in self.task_id:
            return super().get_start_nodes(n_targets, target_ids, n_walks_per_target)
        if 'v' in self.task_id:
            start_nodes = torch.arange(n_targets, device=target_ids.device)
            start_nodes = start_nodes.repeat_interleave(n_walks_per_target)
            target_ids = target_ids.repeat_interleave(n_walks_per_target)
            return start_nodes.cpu().numpy(), target_ids
        raise NotImplementedError(f"Task {self.task_id} not supported")

    def criterion(self, y_hat: Tensor, y: Union[Tensor, Batch]) -> Tensor:
        """Implementation from homomorphism-expressivity:
        https://github.com/subgraph23/homomorphism-expressivity/blob/main/main.count.py
        """
        preds = y_hat
        assert preds.ndim == 2 and preds.shape[1] == 1
        preds = preds.squeeze(1)
        if 'g' in self.task_id:
            assert isinstance(y, Tensor)
            target = y
            assert target.ndim == 1
            loss = torch.nn.functional.l1_loss(preds, target)
            return loss
        if 'v' in self.task_id:
            assert isinstance(y, Batch)
            target = getattr(y, self.task_id)
            assert target.ndim == 1
            # compute loss for each node
            loss = torch.nn.functional.l1_loss(preds, target, reduction='none')
            if self.task_id in NORMALIZE:
                loss /= NORMALIZE[self.task_id]
            # sum-pool nodes to graphs
            loss = self.pool_to_target(
                loss.unsqueeze(1),
                n_targets=getattr(y, 'num_graphs'),
                target_ids=getattr(y, 'batch'),
                reduce='sum'
            )
            # mean-pool graphs to batch
            loss = loss.mean()
            return loss
        raise NotImplementedError(f"Task {self.task_id} not supported")

    def evaluator(self, y_hat: Tensor, y: Union[Tensor, Batch]) -> Dict:
        """Implementation from homomorphism-expressivity:
        https://github.com/subgraph23/homomorphism-expressivity/blob/main/main.count.py
        """
        metric_val = self.criterion(y_hat, y)
        if 'g' in self.task_id:
            batch_size = y.size(0)
        elif 'v' in self.task_id:
            batch_size = getattr(y, 'num_graphs')
        else:
            raise NotImplementedError(f"Task {self.task_id} not supported")
        return {
            'metric_sum': metric_val * batch_size,
            'metric_count': batch_size
        }


class RegressionCountingDataset(InMemoryDataset):
    """Implementation from homomorphism-expressivity:
    https://github.com/subgraph23/homomorphism-expressivity/blob/main/data/count.py
    https://github.com/subgraph23/homomorphism-expressivity/blob/main/src/dataset.py"""

    def __init__(self, root, split, config):
        assert split in ['train', 'val', 'test']
        self.name = 'Counting'
        super().__init__(root)
        if split == 'train':
            processed_path = self.processed_paths[0]
        elif split == 'val':
            processed_path = self.processed_paths[1]
        elif split == 'test':
            processed_path = self.processed_paths[2]
        else:
            raise ValueError(f"Unrecognized split: {split}")
        self.data, self.slices = torch.load(processed_path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ["graph.npy", "hom.npy", "iso.npy"]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        copy_tree((Path(__file__).parent / "Counting").as_posix(),
                  (Path(self.root) / self.name).as_posix())

        os.chmod((Path(__file__).parent / "utils_regression_counting/count.out").as_posix(), 0o755)

        with Pool(processes=50) as mp:
            data, _ = np.load(f"{self.raw_dir}/graph.npy", allow_pickle=True)
            result = mp.map(func=process, iterable=tqdm(data))

        hom, iso = zip(*result)

        np.save(f"{self.raw_dir}/hom.npy", hom, allow_pickle=True)
        np.save(f"{self.raw_dir}/iso.npy", iso, allow_pickle=True)

    @staticmethod
    def parse_count_dict(count_dict):
        result = {}
        for key, value in count_dict.items():
            for task, count in zip(["g", "v", "e"], value):
                result[f"{key}:{task}"] = torch.from_numpy(count)
        return result

    def process(self):
        adj_list, index_dict = np.load(f"{self.raw_dir}/graph.npy", allow_pickle=True)
        hom_list = np.load(f"{self.raw_dir}/hom.npy", allow_pickle=True)
        iso_list = np.load(f"{self.raw_dir}/iso.npy", allow_pickle=True)
        index_dict = index_dict[()]
        train_idx = index_dict["train"]
        val_idx = index_dict["val"]
        test_idx = index_dict["test"]

        data_list = []
        for adj, hom_dict, iso_dict in zip(adj_list, hom_list, iso_list):
            x = torch.ones(len(adj), 1, dtype=torch.int64)
            edge_index = torch.Tensor(np.vstack(np.where(adj != 0))).type(torch.int64)
            assert list(hom_dict.keys()) == HOM_NAMES
            assert list(iso_dict.keys()) == ISO_NAMES
            data_list.append(Data(
                x=x,
                edge_index=edge_index,
                **self.parse_count_dict(hom_dict),
                **self.parse_count_dict(iso_dict),
            ))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        test_data = [data_list[i] for i in test_idx]

        # filter disconnected graphs
        train_data = [data for data in train_data if
                      nx.is_connected(nx.to_undirected(to_networkx(data)))]
        val_data = [data for data in val_data if
                    nx.is_connected(nx.to_undirected(to_networkx(data)))]
        test_data = [data for data in test_data if
                     nx.is_connected(nx.to_undirected(to_networkx(data)))]

        data_list = train_data + val_data + test_data
        statistics = {
            key: torch.std_mean(torch.cat([getattr(data, f"{key}:g") for data in data_list]))
            for key in HOM_NAMES + ISO_NAMES
        }

        train_data, train_slices = self.collate(train_data)
        val_data, val_slices = self.collate(val_data)
        test_data, test_slices = self.collate(test_data)
        for key, (std, mean) in statistics.items():
            train_data[f"{key}:g"] = (train_data[f"{key}:g"] - mean) / std
            train_data[f"{key}:v"] = (train_data[f"{key}:v"] - mean) / std
            train_data[f"{key}:e"] = (train_data[f"{key}:e"] - mean) / std
            val_data[f"{key}:g"] = (val_data[f"{key}:g"] - mean) / std
            val_data[f"{key}:v"] = (val_data[f"{key}:v"] - mean) / std
            val_data[f"{key}:e"] = (val_data[f"{key}:e"] - mean) / std
            test_data[f"{key}:g"] = (test_data[f"{key}:g"] - mean) / std
            test_data[f"{key}:v"] = (test_data[f"{key}:v"] - mean) / std
            test_data[f"{key}:e"] = (test_data[f"{key}:e"] - mean) / std
        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((val_data, val_slices), self.processed_paths[1])
        torch.save((test_data, test_slices), self.processed_paths[2])
