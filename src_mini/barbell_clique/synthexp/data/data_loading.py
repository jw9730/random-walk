import os
from definitions import ROOT_DIR
from .utils.barbell import gen_many_barbells
from .utils.clique import gen_many_cliques


def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'),
                 pre_transform=None, transform=None, config=None):
    """Returns an AugmentedDataset with the specified name and initialised with the given params."""
    train_data, val_data, test_data = _lookup_dataset(name, root, pre_transform, transform, config)
    return train_data, val_data, test_data


def _lookup_dataset(name, root, pre_transform, transform, config=None):
    """ helper function to lookup and load the datasets"""
    if name == "barbell":
        train_data, val_data, test_data = (gen_many_barbells(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_barbells(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_barbells(num_samples=config.data.samples, n=config.data.num_nodes))
        return train_data, val_data, test_data
    if name == "clique":
        train_data, val_data, test_data = (gen_many_cliques(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_cliques(num_samples=config.data.samples, n=config.data.num_nodes),
                                           gen_many_cliques(num_samples=config.data.samples, n=config.data.num_nodes))
        return train_data, val_data, test_data
    raise ValueError("Dataset not found.")
