from typing import Dict
from pathlib import Path
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

from .walker import Walker


class GraphSeparationSR16Walker(Walker):
    def __init__(self, config):
        super().__init__(config)
        self.out_dim = 2
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


class GraphSeparationSR16Dataset(InMemoryDataset):
    """Example from ELENE (Figure 1):
    https://arxiv.org/pdf/2312.05905v1.pdf"""

    def __init__(self, root, split, config, repeat=1000):
        # prior works use training set for validation and testing
        # https://github.com/LingxiaoShawn/GNNAsKernel/blob/main/train/sr25.py
        # https://github.com/nur-ag/ELENE/blob/main/train/sr25.py
        self.name = 'SR16'
        self.repeat = repeat
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return (Path(self.root) / self.name / 'processed').as_posix()

    @property
    def processed_file_names(self):
        return f'data_{self.repeat}x.pt'

    def process(self):
        shrikhande_graph = nx.Graph()
        vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 7), (2, 8), (2, 9), (2, 10),
                (3, 4), (3, 8), (3, 11), (3, 12), (4, 6), (4, 11), (4, 13), (4, 14), (5, 6), (5, 7), (5, 12), (5, 15), (5, 16),
                (6, 9), (6, 13), (6, 15), (7, 10), (7, 14), (7, 16), (8, 9), (8, 12), (8, 13), (8, 16),
                (9, 10), (9, 13), (9, 15), (10, 11), (10, 14), (10, 15), (11, 12), (11, 14), (11, 15),
                (12, 15), (12, 16), (13, 14), (13, 16), (14, 16)]
        shrikhande_graph.add_nodes_from(vertices)
        shrikhande_graph.add_edges_from(edges)
        shrikhande_graph = nx.to_undirected(shrikhande_graph)

        rook_4x4_graph = nx.Graph()
        vertices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 8), (2, 9), (2, 10),
                (3, 4), (3, 11), (3, 12), (3, 13), (4, 14), (4, 15), (4, 16), (5, 6), (5, 7), (5, 8), (5, 11), (5, 14),
                (6, 7), (6, 9), (6, 12), (6, 15), (7, 10), (7, 13), (7, 16), (8, 9), (8, 10), (8, 11), (8, 14),
                (9, 10), (9, 12), (9, 15), (10, 13), (10, 16), (11, 12), (11, 13), (11, 14), (12, 13), (12, 15),
                (13, 16), (14, 15), (14, 16), (15, 16)]
        rook_4x4_graph.add_nodes_from(vertices)
        rook_4x4_graph.add_edges_from(edges)
        rook_4x4_graph = nx.to_undirected(rook_4x4_graph)

        dataset = [shrikhande_graph, rook_4x4_graph]

        # Read data into huge `Data` list.
        data_list = []
        for i, data in enumerate(dataset):
            x = torch.ones(data.number_of_nodes(), 1, dtype=torch.long)
            y = torch.tensor([i], dtype=torch.long)
            edge_index = torch.tensor(list(data.edges())).transpose(1, 0) - 1
            edge_index = to_undirected(edge_index)
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
        data_list = data_list * self.repeat

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
