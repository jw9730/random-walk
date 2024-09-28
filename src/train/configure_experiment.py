import os
import sys
import shutil
from pathlib import Path
import tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint


def configure_experiment(config, model):
    log_dir = setup_log_directory(
        root_dir=config.root_dir,
        log_dir=config.log_dir,
        dataset=config.dataset,
        exp_name=config.exp_name,
        debug_mode=config.debug_mode,
        resume_mode=config.resume_mode,
        test_mode=config.test_mode,
    )
    save_dir = setup_save_directory(
        root_dir=config.root_dir,
        save_dir=config.save_dir,
        dataset=config.dataset,
        exp_name=config.exp_name,
        debug_mode=config.debug_mode,
        resume_mode=config.resume_mode,
        test_mode=config.test_mode,
    )
    callbacks = setup_callbacks(
        save_dir=save_dir,
        early_stopping_monitor=config.early_stopping_monitor,
        early_stopping_mode=config.early_stopping_mode,
        early_stopping_patience=config.early_stopping_patience
    )
    logger = setup_logger(
        log_dir=log_dir,
        team_name=config.team_name,
        dataset_name=config.dataset,
        exp_name=config.exp_name
    )
    precision = setup_precision(precision=config.precision)
    strategy = setup_strategy(strategy=config.strategy)
    plugins = setup_plugins(model=model)
    return logger, log_dir, callbacks, precision, strategy, plugins


def setup_log_directory(root_dir, log_dir, dataset, exp_name, debug_mode, resume_mode, test_mode):
    dataset = dataset.replace('/', '_')
    log_dir = Path(root_dir) / log_dir / dataset / exp_name
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    if log_dir.exists() and IS_RANK_ZERO:
        if debug_mode:
            print(f'remove existing logs ({exp_name})')
            shutil.rmtree(log_dir)
        elif resume_mode or test_mode:
            pass
        else:
            while True:
                print(f'redundant log directory! ({log_dir}) remove existing logs? (y/n)')
                inp = input()
                if inp == 'y':
                    shutil.rmtree(log_dir)
                    break
                if inp == 'n':
                    print('quit')
                    sys.exit()
                print('invalid input')
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir.as_posix()


def setup_save_directory(root_dir, save_dir, dataset, exp_name, debug_mode, resume_mode, test_mode):
    # create save directory if checkpoint doesn't exist,
    # otherwise ask user to reset it
    dataset = dataset.replace('/', '_')
    save_dir = Path(root_dir) / save_dir / dataset / exp_name
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    if save_dir.exists() and IS_RANK_ZERO:
        if resume_mode:
            print(f'resume from checkpoint ({exp_name})')
        elif test_mode:
            print(f'test existing checkpoint ({exp_name})')
        elif debug_mode:
            print(f'remove existing checkpoint ({exp_name})')
            shutil.rmtree(save_dir)
        else:
            while True:
                print(f'redundant experiment name! ({exp_name}) remove existing checkpoints? (y/n)')
                inp = input()
                if inp == 'y':
                    shutil.rmtree(save_dir)
                    break
                if inp == 'n':
                    print('quit')
                    sys.exit()
                print('invalid input')
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir.as_posix()


class CustomEarlyStopping(EarlyStopping):
    def __init__(
            self,
            monitor,
            patience,
            verbose,
            mode,
            check_on_train_epoch_end
        ):
        super().__init__(
            monitor=monitor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            check_on_train_epoch_end=check_on_train_epoch_end
        )
        self.patience_for_overriding = patience

    def on_validation_end(self, trainer, pl_module):
        if self.patience_for_overriding != self.patience:
            if self.verbose:
                print("Overriding early stopping patience loaded from checkpoint: "
                      f"{self.patience} -> {self.patience_for_overriding}")
            self.patience = self.patience_for_overriding
        self._run_early_stopping_check(trainer)


def setup_callbacks(save_dir, early_stopping_monitor, early_stopping_mode, early_stopping_patience):
    callbacks = [CustomProgressBar(),]
    if early_stopping_monitor is not None and early_stopping_patience > 0:
        callbacks.append(
            CustomEarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                verbose=True,
                mode=early_stopping_mode,
                check_on_train_epoch_end=False
            )
        )
    if save_dir is not None:
        # last checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename='last',
            monitor='epoch',
            verbose=False,
            save_last=False,
            save_top_k=1,
            mode='max',
            auto_insert_metric_name=False,
            every_n_epochs=1
        )
        checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
        callbacks.append(checkpoint_callback)
        # best checkpointing
        if early_stopping_monitor is not None:
            checkpoint_callback = ModelCheckpoint(
                dirpath=save_dir,
                filename='best',
                monitor=early_stopping_monitor,
                save_last=False,
                save_top_k=1,
                mode=early_stopping_mode,
                auto_insert_metric_name=False,
                every_n_epochs=1
            )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)
    return callbacks


def setup_logger(log_dir: str, team_name: str, dataset_name: str, exp_name: str):
    logger = WandbLogger(
        name=exp_name,
        save_dir=log_dir,
        project=dataset_name.replace('/', '_'),
        entity=team_name,
        log_model=False
    )
    return logger


def setup_plugins(model):
    plugins = []
    return plugins


def setup_precision(precision: str):
    return int(precision.strip('fp')) if precision in ['fp16', 'fp32'] else precision


def setup_strategy(strategy: str):
    if strategy == 'ddp':
        strategy = 'ddp'
    elif strategy == 'ddp_spawn':
        strategy = 'ddp_spawn'
    else:
        strategy = 'auto'
    return strategy


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, rescale_validation_batches=1):
        super().__init__()
        self.rescale_validation_batches = rescale_validation_batches

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        pbar = tqdm.tqdm(
            desc="Training",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return pbar

    def init_validation_tqdm(self):
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        pbar = tqdm.tqdm(
            desc="Validation",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return pbar

    def init_test_tqdm(self):
        """Override this to customize the tqdm bar for testing."""
        pbar = tqdm.tqdm(
            desc="Testing",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return pbar
