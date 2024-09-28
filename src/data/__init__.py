# pylint: disable=line-too-long
from typing import Tuple
from .ds_builder import DatasetBuilder
from .walker import Walker
from .graph_separation_csl import GraphSeparationCSLDataset, GraphSeparationCSLWalker
from .graph_separation_sr16 import GraphSeparationSR16Dataset, GraphSeparationSR16Walker
from .graph_separation_sr25 import GraphSeparationSR25Dataset, GraphSeparationSR25Walker
from .regression_counting import RegressionCountingDataset, RegressionCountingWalker


def setup_data_and_walker(dataset: str, root_dir: str, config) -> Tuple[DatasetBuilder, Walker]:
    # pyg datasets
    is_pyg = True
    if dataset == 'graph_separation_csl':
        walker = GraphSeparationCSLWalker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphSeparationCSLDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_separation_sr16':
        walker = GraphSeparationSR16Walker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphSeparationSR16Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_separation_sr25':
        walker = GraphSeparationSR25Walker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphSeparationSR25Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'regression_counting':
        walker = RegressionCountingWalker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, RegressionCountingDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    # non-pyg datasets
    is_pyg = False
    raise NotImplementedError(f"Dataset ({dataset}) not supported!")
