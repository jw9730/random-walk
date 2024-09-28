from torch.utils.data import Dataset

from lightning.pytorch.utilities import rank_zero_only


class DatasetBuilder:
    """Dataset configuration class"""
    def __init__(
            self,
            ds_name: str,
            is_pyg: bool,
            ds_builder,
            root_dir: str,
            config
        ):
        self.ds_name = ds_name
        self.is_pyg = is_pyg
        self.ds_builder = ds_builder
        self.root_dir = root_dir
        self.config = config

    @rank_zero_only
    def prepare_data(self):
        self.ds_builder(self.root_dir, split='train', config=self.config)

    def train_dataset(self) -> Dataset:
        return self.ds_builder(self.root_dir, split='train', config=self.config)

    def val_dataset(self) -> Dataset:
        return self.ds_builder(self.root_dir, split='val', config=self.config)

    def test_dataset(self)  -> Dataset:
        return self.ds_builder(self.root_dir, split='test', config=self.config)

    def predict_dataset(self) -> Dataset:
        return NotImplemented
