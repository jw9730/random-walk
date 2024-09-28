from typing import List, Dict
import os
import os.path as osp
import pickle
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import remove_self_loops, to_undirected

from .walker import Walker


class GraphSeparationCSLWalker(Walker):
    def __init__(self, config):
        super().__init__(config)
        self.out_dim = 10
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


class GraphSeparationCSLDataset(InMemoryDataset):
    root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    urls = {'CSL': 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1'}

    def __init__(self, root, split, config, repeat=100):
        self.name = 'CSL'
        self.repeat = repeat
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['graphs_Kary_Deterministic_Graphs.pkl',
                'y_Kary_Deterministic_Graphs.pt']

    @property
    def processed_file_names(self):
        return f'data_{self.repeat}x.pt'

    def download(self) -> None:
        path = download_url(self.urls[self.name], self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        data_list = self.process_csl()
        data_list = data_list * self.repeat
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process_csl(self) -> List[Data]:
        with open(self.raw_paths[0], 'rb') as f:
            adjs = pickle.load(f)

        ys = torch.load(self.raw_paths[1]).tolist()

        data_list = []
        for adj, y in zip(adjs, ys):
            row, col = torch.from_numpy(adj.row), torch.from_numpy(adj.col)
            edge_index = torch.stack([row, col], dim=0).to(torch.long)
            x = torch.ones(adj.shape[0], 1, dtype=torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            data = Data(edge_index=edge_index, x=x, y=y, num_nodes=adj.shape[0])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        return data_list
