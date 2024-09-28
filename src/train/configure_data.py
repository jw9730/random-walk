from pathlib import Path
import torch

from src.data import DatasetBuilder, setup_data_and_walker
from .lit_datamodule import LitDataModule, setup_pyg_datamodule


def configure_data(config):
    data_dir = setup_data_directory(
        root_dir=config.root_dir,
        data_dir=config.data_dir
    )
    ds_builder, walker = setup_data_and_walker(
        root_dir=data_dir,
        dataset=config.dataset,
        config=config
    )
    print(walker)
    global_batch_size = config.global_batch_size
    devices = torch.cuda.device_count() if config.accelerator == 'gpu' else 1
    if config.test_mode:
        global_batch_size = getattr(config, 'test_batch_size', global_batch_size // devices)
        devices = 1
    datamodule = setup_datamodule(
        ds_builder=ds_builder,
        global_batch_size=global_batch_size,
        devices=devices,
        num_workers=config.num_workers
    )
    return datamodule, walker


def setup_data_directory(root_dir='experiments', data_dir='data'):
    data_dir = Path(root_dir) / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir.as_posix()


def setup_datamodule(
    ds_builder: DatasetBuilder,
    global_batch_size,
    devices,
    num_workers,
):
    dm_builder = setup_pyg_datamodule if ds_builder.is_pyg else LitDataModule
    datamodule = dm_builder(
        ds_builder=ds_builder,
        global_batch_size=global_batch_size,
        devices=devices,
        num_workers=num_workers
    )
    return datamodule
