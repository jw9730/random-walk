# pylint: disable=protected-access,too-many-locals,unused-argument,line-too-long,too-many-instance-attributes,too-few-public-methods,too-many-arguments
from typing import Dict
from pathlib import Path
from distutils.dir_util import copy_tree
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

from .walker import Walker


class GraphSeparationSR25Walker(Walker):
    def __init__(self, config):
        super().__init__(config)
        self.out_dim = 15
        self.metric_name = 'accuracy'

    def criterion(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y)

    def evaluator(self, y_hat: Tensor, y: Tensor) -> Dict:
        preds, target = y_hat, y
        # compute metrics
        preds = preds.argmax(dim=-1)
        metric_val = (preds == target).float().mean()
        batch_size = target.size(0)
        return {
            'metric_sum': metric_val * batch_size,
            'metric_count': batch_size
        }


class GraphSeparationSR25Dataset(InMemoryDataset):
    """Implementation from ELENE (SRDataset):
    https://github.com/nur-ag/ELENE/blob/main/core/data.py"""

    def __init__(self, root, split, config, repeat=1000):
        # prior works use training set for validation and testing
        # https://github.com/LingxiaoShawn/GNNAsKernel/blob/main/train/sr25.py
        # https://github.com/nur-ag/ELENE/blob/main/train/sr25.py
        self.name = 'SR25'
        self.repeat = repeat
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return (Path(self.root) / self.name / 'raw').as_posix()

    @property
    def processed_dir(self) -> str:
        return (Path(self.root) / self.name / 'processed').as_posix()

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]

    @property
    def processed_file_names(self):
        return f'data_{self.repeat}x.pt'

    def download(self):
        copy_tree((Path(__file__).parent / "SR25").as_posix(),
                  (Path(self.root) / self.name).as_posix())

    def process(self):
        # Read data into huge `Data` list.
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, data in enumerate(dataset):
            x = torch.ones(data.number_of_nodes(), 1, dtype=torch.long)
            y = torch.tensor([i], dtype=torch.long)
            edge_index = to_undirected(torch.tensor(list(data.edges())).transpose(1, 0))
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
        data_list = data_list * self.repeat

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
