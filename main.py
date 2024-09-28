# pylint: disable=no-member
import sys
import argparse
import warnings
import yaml
from easydict import EasyDict as edict
import torch
import lightning as L

from src.train import configure_data, configure_model, configure_experiment


def str2bool(v: str) -> bool:
    if v in ('True', 'true'):
        return True
    if v in ('False', 'false'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # necessary arguments
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_mode', default=False, action='store_true')
    parser.add_argument('--resume_mode', default=False, action='store_true')
    parser.add_argument('--test_mode', default=False, action='store_true')
    parser.add_argument('--test_ckpt_path', type=str, default=None)

    # experiment arguments
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--test_seed', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--name_postfix', type=str, default=None)

    # data arguments
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--compile', type=str2bool, default=None)
    parser.add_argument('--strategy', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--global_batch_size', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)

    # model arguments
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--pretrained', type=str2bool, default=None)
    parser.add_argument('--head_dropout', type=float, default=None)

    # random walk arguments
    parser.add_argument('--walk_type', type=str, default=None,
                        choices=['natural', 'min_degree', 'node2vec'])
    parser.add_argument('--walk_length', type=int, default=None)
    parser.add_argument('--restart_prob', type=float, default=None)
    parser.add_argument('--restart_period', type=int, default=None)
    parser.add_argument('--backtrack', type=str2bool, default=None)
    parser.add_argument('--node2vec_p', type=float, default=None)
    parser.add_argument('--node2vec_q', type=float, default=None)
    parser.add_argument('--neighbors', type=str2bool, default=None)
    parser.add_argument('--n_walks', type=int, default=None)
    parser.add_argument('--eval_n_walks', type=int, default=None)
    parser.add_argument('--test_n_walks', type=int, default=None)

    # tokenizer arguments
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--pretrained_tokenizer', type=str2bool, default=None)
    parser.add_argument('--reverse', type=str2bool, default=None)

    # training arguments
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--gradient_clip_val', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_pretrained', type=float, default=None)
    parser.add_argument('--lr_schedule', type=str, default=None,
                        choices=['const', 'sqrt', 'cos', 'poly'])
    parser.add_argument('--early_stopping_monitor', type=str, default=None)
    parser.add_argument('--early_stopping_mode', type=str, default=None,
                        choices=['min', 'max'])
    parser.add_argument('--early_stopping_patience', type=int, default=None)

    # logging arguments
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--val_iter', type=int, default=None)
    parser.add_argument('--save_iter', type=int, default=None)

    return parser


def get_config() -> edict:
    # parse arguments
    parser = argparse.ArgumentParser(description='Random Walks')
    parser = add_args(parser)
    args = parser.parse_args()

    # load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        config = edict(config)

    # update config with parsed arguments
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)

    # create experiment name
    if config.exp_name == '':
        config.exp_name = f"{config.backbone.replace('/', '_')},pt_{config.pretrained}," \
            + (f"drop_{config.head_dropout}," if config.head_dropout > 0 else '') \
            + (f"l_{config.max_length}," if config.max_length > 0 else '')
        config.exp_name += f"w_{config.walk_type}," \
            + f"wl_{config.walk_length}," \
            + (f"rp_{config.restart_prob}," if hasattr(config, 'restart_prob') else '') \
            + (f"rpd_{config.restart_period}," if hasattr(config, 'restart_period') else '') \
            + (f"bt_{config.backtrack}," if hasattr(config, 'backtrack') else '') \
            + (f"(p,q)_({config.node2vec_p},{config.node2vec_q})," if
               (hasattr(config, 'node2vec_p') and hasattr(config, 'node2vec_q')) else '') \
            + (f"n_{config.neighbors}," if hasattr(config, 'neighbors') else '') \
            + (f"nw_{config.n_walks}," if hasattr(config, 'n_walks') else '') \
            + (f"enw_{config.eval_n_walks}," if hasattr(config, 'eval_n_walks') else '') \
            + ("rev," if hasattr(config, 'reverse') and config.reverse else '')
        config.exp_name += f"b_{config.global_batch_size}" \
            + (f"x{config.accumulate_grad_batches}," if
               hasattr(config, 'accumulate_grad_batches') else ',') \
            + f"es_{config.early_stopping_monitor.replace('/', '_')}" \
            + f"_{config.early_stopping_mode}_{config.early_stopping_patience}," \
            + f"lr_{config.lr}_{config.lr_pretrained}," \
            + f"steps_{config.n_steps}," \
            + f"wu_{config.lr_warmup}," \
            + f"wd_{config.weight_decay}," \
            + (f"clip_{config.gradient_clip_val}," if
               hasattr(config, 'gradient_clip_val') else '') \
            + f"seed_{config.seed},"
        config.exp_name += (config.name_postfix if hasattr(config, 'name_postfix') else '')

    # create seed for testing
    if not hasattr(config, 'test_seed'):
        config.test_seed = config.seed

    # create checkpoint for testing
    if not hasattr(config, 'test_ckpt_path'):
        config.test_ckpt_path = None

    # create team name for wandb logging
    config.team_name = 'vl-kaist'

    # setup debugging
    if config.debug_mode:
        config.accelerator = 'cpu'
        config.compile = False
        config.num_workers = 0
        config.global_batch_size = 2
        config.n_steps = 10
        config.log_iter = 1
        config.val_iter = 5
        config.save_iter = 5
        config.exp_name = '_debug_' + config.exp_name

    return config


def main(config):
    # reproducibility (this and deterministic=True in trainer)
    L.seed_everything(config.seed, workers=True)

    # utilize Tensor Cores (RTX 3090)
    torch.set_float32_matmul_precision('medium')

    # configure data and task
    datamodule, walker = configure_data(config)

    # configure model
    model, ckpt_path = configure_model(config, walker)

    # configure experiment
    logger, log_dir, callbacks, precision, strategy, plugins = configure_experiment(config, model)

    if config.compile:
        if config.test_mode:
            warnings.warn("Test mode, compile flag ignored.")
        else:
            # compile the model and *step (training/validation/test/prediction)
            # can lead to nondeterministic behavior
            warnings.warn("Compile mode enabled. This can lead to nondeterministic behavior.")
            model = torch.compile(model)

    if config.test_mode:
        # test routine reproducibility (this and deterministic=True in trainer)
        L.seed_everything(config.test_seed, workers=True)

        # setup trainer
        # during evaluation, it is recommended to use `Trainer(devices=1, num_nodes=1)`
        # to ensure each sample/batch gets evaluated exactly once. Otherwise,
        # multi-device settings use `DistributedSampler` that replicates some
        # samples to make sure all devices have same batch size in case of uneven inputs.
        # https://github.com/Lightning-AI/lightning/issues/12862
        trainer = L.Trainer(
            logger=logger,
            default_root_dir=log_dir,
            accelerator=config.accelerator,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            deterministic=True,
            devices=1,
            num_nodes=1,
            strategy=strategy,
            precision=precision,
            plugins=plugins
        )

        # start evaluation
        trainer.test(model, datamodule=datamodule)

        # terminate
        sys.exit()

    # setup trainer
    trainer = L.Trainer(
        logger=logger,
        default_root_dir=log_dir,
        accelerator=config.accelerator,
        max_steps=config.n_steps,
        val_check_interval=(
            config.val_iter * getattr(config, 'accumulate_grad_batches', 1) if
            hasattr(config, 'val_iter') else None
        ),
        check_val_every_n_epoch=(None if hasattr(config, 'val_iter') else 1),
        log_every_n_steps=-1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        deterministic=(not config.compile),
        devices=(torch.cuda.device_count() if config.accelerator == 'gpu' else 1),
        strategy=strategy,
        precision=precision,
        plugins=plugins,
        gradient_clip_val=getattr(config, 'gradient_clip_val', 0),
        accumulate_grad_batches=getattr(config, 'accumulate_grad_batches', 1)
    )

    if not config.resume_mode:
        # validation at start
        trainer.validate(model, datamodule=datamodule)

    # start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # start evaluation
    # this uses the last checkpoint for testing, and replicates some test samples.
    # for exact evaluation using the best checkpoint, it is recommended to run a
    # separate process with command `python3 main,py ... --test_mode` after training.
    trainer.test(model, datamodule=datamodule)

    # terminate
    sys.exit()


if __name__ == '__main__':
    # get config and start main
    config_ = get_config()
    main(config_)
