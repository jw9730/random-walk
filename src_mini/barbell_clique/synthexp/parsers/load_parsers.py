import yaml
import os.path as osp
import argparse
from definitions import ROOT_DIR
import time
import git


ALL_KEY_VAL = {
    "synthexp": {
        "exp_name": time.time(),
        "start_seed": 0,
        "num_seeds": 1,
        "seed": 0,
        "device": 0,
        "entity": "XXXX-1-XXXX-2",
        "result_folder": osp.join(ROOT_DIR, '', 'results'),  # TODO: remove
        "dump_curves": False, # TODO: remove
        "train_eval_period": 10,
        "sha": git.Repo(search_parent_directories=True).head.object.hexsha
    },

    "data": {
        "dataset": "ZINC",
        "task_type": "regression",
        "graph_level": False,
        "eval_metric": "mae",
        "minimize": False,
        "unbalanced": False,
        "pow": 1,
        "nb_graphs": 1,
        "samples": 10,
        "theta": 0,
        "unique_id": False,
        "num_nodes": 250,
    },

    "model": {
        "model_name": "gingcn",
        "input_dim": 1,
        "hidden_dim": 8,
        "out_dim": 1,
        "bundle_dim": 2,
        "num_bundle": 1,
        "num_layers": 2,
        "num_layers_gin": 2,
        "num_gnn_layers": 1,
        "layer_type": "GCN",
        "time": 100,
        "orth_map": "householder",
        "act": "relu",
        "add_self_loops": True,
        "bias": True,
        "norm": None,
        "dropout_in_feat": 0.0,
        "dropout_rate": 0.0,
        "residual": False,

        "max_deg": 4,
        "gnn_type": "GCN",
        "learn_tau": False,
        "k": 1,  # for chebnet

        "heads": 1,  # for GAT

        "num_layers_decoder": 0,
        "num_layers_encoder": 0,

        "atom_types": 20,
        "lrgb_embedding": False,
        "ogb_embedding": False,

        "readout": "mean",
        "num_layers_post_readout": 0,

        "num_layers_mlp": 2,  # TODO: Combine with num_layers_gin
        "enc_in_dim": 2,
        "time_steps": [1, 2, 4, 8],

        # for RWNN
        "min_degree": True,
        "no_backtrack": True,
        "walk_len": 1000,
        "train_n_walks": 8,
        "eval_n_walks": 8,
        "test_n_walks": 8,
    },

    "optim":{
        "optimizer": "ADAM",
        "lr": 0.001,
        "weight_decay": 0,
        "lr_bunn": 0.001,
        "lr_tau": 0.1,
        "lr_scheduler": None,
        "lr_scheduler_patience": 10,
        "lr_scheduler_decay_rate": 0.5,
        "lr_scheduler_decay_steps": 10,
        "lr_scheduler_min": 0.0000001,
        "batch_size": 50,
        "epochs": 100,
        "early_stop": True
    }
}


def load_yaml(yaml_name):
    path = osp.join(ROOT_DIR, "synthexp", "configs", yaml_name+".yml")
    with open(path, "r") as file:
        dic = yaml.safe_load(file)
    print(dic)
    return dic


def load_config(cfg):
    return Config(cfg)


class Config:
    def __init__(self, cfg):
        self.exp = ExpConfig(cfg["exp"])
        self.data = DatasetConfig(cfg["data"])
        self.model = ModelConfig(cfg["model"])
        self.optim = OptimConfig(cfg["optim"])

        self.dic = cfg

    def get_dic(self):
        return self.dic


class ExpConfig:
    """ General config about the experiment"""
    def __init__(self, cfg):
        self.exp_name: str = str(time.time())  # help='name for specific experiment; if not provided, a name

        self.start_seed: int = 289469  # random seed to use as first seed, useful when running on mutliple seeds
        self.num_seeds: int = 5  # 'number of seeds over which to run experiments'
        self.seed: int = 289469  # seed of current experiment
        self.device: int = 0  # which gpu to use, if there is any

        self.result_folder: str = osp.join(ROOT_DIR, '', 'results')  # filename to output result (default: None, will use `bunbdles/synthexp/results`)
        self.dump_curves: bool = False  # 'whether to dump the training curves to disk'
        self.wandb: bool = False  # whether to record results in wandb
        self.train_eval_period: int = 10

        # bookkeeping
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        self.sha: str = sha  # Stores the hash of the git commit to track code version
        self.entity: str = "XXXX-1-XXXX-2"

        self.set_config(cfg)

    def set_config(self, cfg):
        old_dic = ALL_KEY_VAL["synthexp"].copy()
        old_dic.update(cfg)
        cfg = old_dic
        self.exp_name: str = cfg["exp_name"]

        self.start_seed = cfg["start_seed"]
        self.num_seeds = cfg["num_seeds"]
        self.seed = cfg["seed"]
        self.device = cfg["device"]

        self.result_folder = cfg["result_folder"]
        self.dump_curves = cfg["dump_curves"]
        self.wandb = cfg["wandb"]
        self.train_eval_period = cfg["train_eval_period"]

        # bookkeeping
        self.sha = cfg["sha"]
        self.entity = cfg["entity"]


class DatasetConfig:
    def __init__(self, cfg):
        self.dataset: str = "ZINC"  # dataset name (default: ZINC)
        self.task_type: str = "regression"  # what kind of task, classification, regression etc...
        self.graph_level: bool = False  # true if task is graph level, otherwise assume it is node-level
        self.eval_metric: str = "mae"  # evaluation metric (default: mae)
        self.minimize: bool = False  # whether to minimize or maximize the loss
        self.unbalanced: bool = False
        self.pow: int = 1
        self.nb_graphs: int = 10
        self.samples: int = 10
        self.theta = 0
        self.unique_id = False
        self.num_nodes = 250
        self.set_config(cfg)

    def set_config(self, cfg):
        old_dic = ALL_KEY_VAL["data"].copy()
        old_dic.update(cfg)
        cfg = old_dic
        self.dataset = cfg["dataset"]
        self.task_type = cfg["task_type"]
        self.graph_level = cfg["graph_level"]
        self.eval_metric = cfg["eval_metric"]
        self.minimize = cfg["minimize"]
        self.unbalanced = cfg["unbalanced"]
        self.pow = cfg["pow"]
        self.nb_graphs = cfg["nb_graphs"]
        self.samples = cfg["samples"]
        self.theta = cfg["theta"]
        self.unique_id = cfg["unique_id"]
        self.num_nodes = cfg["num_nodes"]


class ModelConfig:
    def __init__(self, cfg):
        old_dic = ALL_KEY_VAL["model"].copy()
        old_dic.update(cfg)
        cfg = old_dic
        self.model_name: str = cfg["model_name"]
        self.input_dim: int = cfg["input_dim"]
        self.hidden_dim: int = cfg["hidden_dim"]
        self.out_dim: int = cfg["out_dim"]
        self.bundle_dim: int = cfg["bundle_dim"]
        self.num_bundle: int = cfg["num_bundle"]
        self.num_layers: int = cfg["num_layers"]
        self.num_layers_gin: int = cfg["num_layers_gin"]
        self.num_gnn_layers: int = cfg["num_gnn_layers"]
        self.layer_type: str = cfg["layer_type"]
        self.time: int = cfg["time"]
        self.orth_map: str = cfg["orth_map"]
        self.act: int = cfg["act"]
        self.add_self_loops: bool = cfg["add_self_loops"]
        self.bias: bool = cfg["bias"]
        self.norm: str = cfg["norm"]
        self.dropout_in_feat: float = cfg["dropout_in_feat"]
        self.dropout_rate: float = cfg["dropout_rate"]
        self.num_layers_decoder: int = cfg["num_layers_decoder"]
        self.num_layers_encoder: int = cfg["num_layers_encoder"]
        self.max_deg: int = cfg["max_deg"]
        self.gnn_type: str = cfg["gnn_type"]
        self.learn_tau: bool = cfg["learn_tau"]
        self.k: int = cfg["k"]
        self.heads: int = cfg["heads"]

        self.atom_types: int = cfg["atom_types"]
        self.lrgb_embedding: bool = cfg["lrgb_embedding"]
        self.ogb_embedding: bool = cfg["ogb_embedding"]

        self.readout: str = cfg["readout"]
        self.num_layers_post_readout: int = cfg["num_layers_post_readout"]
        self.num_layers_mlp: int = cfg["num_layers_mlp"]
        self.enc_in_dim: int = cfg["enc_in_dim"]
        self.time_steps: list = cfg["time_steps"]

        # for RWNN
        self.min_degree: bool = cfg["min_degree"]
        self.no_backtrack: bool = cfg["no_backtrack"]
        self.walk_len: int = cfg["walk_len"]
        self.train_n_walks: int = cfg["train_n_walks"]
        self.eval_n_walks: int = cfg["eval_n_walks"]
        self.test_n_walks: int = cfg["test_n_walks"]


class OptimConfig:
    def __init__(self, cfg):
        old_dic = ALL_KEY_VAL["optim"].copy()
        old_dic.update(cfg)
        cfg = old_dic
        self.optimizer: str = "ADAM"

        self.lr: float = cfg["lr"]
        self.weight_decay: float = cfg["weight_decay"]
        self.lr_bunn: float = cfg["lr_bunn"]
        self.lr_tau: float = cfg["lr_tau"]
        self.lr_scheduler: str = cfg["lr_scheduler"]
        self.lr_scheduler_patience: int = cfg["lr_scheduler_patience"]
        self.lr_scheduler_decay_steps: int = cfg["lr_scheduler_decay_steps"]
        self.lr_scheduler_decay_rate: float = cfg["lr_scheduler_decay_rate"]
        self.lr_scheduler_min: float = cfg["lr_scheduler_min"]

        self.batch_size: int = cfg["batch_size"]
        self.epochs: int = cfg["epochs"]
        self.early_stop: bool = cfg["early_stop"]

def get_parser():
    parser = argparse.ArgumentParser(description='Common parser for all synthexp')
    parser.add_argument('--yaml_name', type=str)
    return parser
