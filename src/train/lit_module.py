# pylint: disable=arguments-differ,unused-argument
import torch
import lightning as L

from src.model import Model
from src.optim import OptimizerConfig, LRSchedulerConfig


class LitModule(L.LightningModule):
    def __init__(
            self,
            model: Model,
            optimizer_config: OptimizerConfig,
            lr_scheduler_config: LRSchedulerConfig
        ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler = None

        self.criterion = model.walker.criterion
        self.evaluator = model.walker.evaluator
        self.metric_name = model.walker.metric_name

        self.use_global_test_metric = model.walker.global_test_metric
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

    def configure_optimizers(self):
        optimizer = self.optimizer_config.setup(self.model)
        self.lr_scheduler = self.lr_scheduler_config.setup(optimizer)
        return optimizer

    def forward(self, batch):
        return self.model(batch)

    @torch.no_grad()
    @torch.compiler.disable
    def training_log(self, loss, batch_size):
        self.log('training/loss', loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, logger=True, rank_zero_only=True)
        self.log('training/lr', self.lr_scheduler.lr,
                 on_step=True, on_epoch=False, logger=True, rank_zero_only=True)
        self.log('training/step', float(self.global_step),
                 on_step=True, on_epoch=False, logger=True, rank_zero_only=True)

    def training_step(self, batch, batch_idx):
        assert self.model.training
        y_hat, y = self.forward(batch)
        loss = self.criterion(y_hat, y)
        self.lr_scheduler.step(self.global_step)
        self.training_log(loss.detach(), y_hat.shape[0])
        return loss

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, batch):
        assert not self.model.training
        return self.forward(batch)

    @staticmethod
    def parse_perf(perf: dict):
        metric_sum = float(perf['metric_sum'])
        metric_count = max(1, int(perf['metric_count']))
        return metric_sum / metric_count, metric_count

    @torch.compiler.disable
    def inference_log(self, mode, loss, perf, batch_size):
        self.log(f'{mode}/loss', loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if self.use_global_test_metric:
            return
        if isinstance(self.metric_name, str):
            metric_mean, metric_count = self.parse_perf(perf)
            self.log(f'{mode}/{self.metric_name}', metric_mean, batch_size=metric_count,
                     on_step=False, on_epoch=True, logger=True, sync_dist=True)
        else:
            assert isinstance(self.metric_name, list)
            for metric_name in self.metric_name:
                metric_mean, metric_count = self.parse_perf(perf[metric_name])
                self.log(f'{mode}/{metric_name}', metric_mean, batch_size=metric_count,
                         on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.inference(batch)
        loss = self.criterion(y_hat, y)
        perf = self.evaluator(y_hat, y)
        self.inference_log('validation', loss, perf, y_hat.shape[0])
        if self.use_global_test_metric:
            self.val_step_outputs.append((y_hat, y))

    def on_validation_epoch_end(self):
        if self.use_global_test_metric:
            print('global validation')
            y_hat, y = zip(*self.val_step_outputs)
            perfs = self.model.walker.global_evaluator(y_hat, y)
            metric_mean = perfs['metric_sum'] / perfs['metric_count']
            self.log(f'validation/{self.metric_name}', metric_mean,
                     on_epoch=True, logger=True, sync_dist=True)
        self.val_step_outputs.clear()

    def on_validation_model_zero_grad(self) -> None:
        '''https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
        Small hack to avoid first validation on resume. 
        This will NOT work if the gradient accumulation step should be performed at this point.
        '''
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False

    def test_step(self, batch, batch_idx):
        y_hat, y = self.inference(batch)
        if self.use_global_test_metric:
            self.test_step_outputs.append((y_hat, y))
            return
        loss = self.criterion(y_hat, y)
        perf = self.evaluator(y_hat, y)
        self.inference_log('test', loss, perf, y_hat.shape[0])

    def on_test_epoch_end(self):
        if self.use_global_test_metric:
            y_hat, y = zip(*self.test_step_outputs)
            perfs = self.model.walker.global_evaluator(y_hat, y)
            metric_mean = perfs['metric_sum'] / perfs['metric_count']
            self.log(f'test/{self.metric_name}', metric_mean,
                     on_epoch=True, logger=True, sync_dist=True)
            # import pdb; pdb.set_trace()
            # torch.save((y_hat, y), 'test_outputs.pt')
        self.test_step_outputs.clear()

    @torch.compiler.disable
    def predict_step(self, batch, batch_idx):
        y_hat, y = self.inference(batch)
        self.predict_step_outputs.append((batch, y_hat, y))

    def on_predict_epoch_end(self):
        self.predict_step_outputs.clear()
