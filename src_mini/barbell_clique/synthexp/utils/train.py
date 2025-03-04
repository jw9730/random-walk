import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics as met
from sklearn.metrics import average_precision_score
from ogb.graphproppred import Evaluator as OGBEvaluator


def train(model, device, loader, optimizer, task_type='classification', ignore_unlabeled=False, weight=None, mask=None, cfg=None):
    """
    Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """
    assert task_type == 'mse_regression'
    loss_fn = torch.nn.MSELoss()

    curve = []
    model.train()
    for _, batch in enumerate(tqdm(loader, desc="Training iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(x=batch.x, edge_index=batch.edge_index, batch=batch)

        if isinstance(pred, tuple):
            # RWNN
            assert mask is None
            pred, targets = pred
        elif isinstance(loss_fn, torch.nn.CrossEntropyLoss) and len(batch.y.shape) > 1 and batch.y.shape[1] == 1:  # TODO: what is the goal of this?
            targets = batch.y.view(-1, )
        else:
            targets = batch.y

        if mask is not None:
            loss = loss_fn(pred[mask], targets[mask])
        else:
            loss = loss_fn(pred, targets)

        loss.backward()
        optimizer.step()
        curve.append(loss.detach().cpu().item())

    return curve


def infer(model, device, loader, mask=None, cfg=None):
    """
        Runs inference over all the batches of a data loader.
    """
    model.eval()
    y_pred = []
    for _, batch in enumerate(tqdm(loader, desc="Inference iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(x=batch.x, edge_index=batch.edge_index, batch=batch)
            if isinstance(pred, tuple):
                # RWNN
                assert mask is None
                pred, _ = pred
        if mask is not None:
            pred = pred[mask]
        y_pred.append(pred.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if mask is not None:
        y_pred = y_pred[mask]
    return y_pred


@torch.no_grad()
def evaluate(model, device, loader, evaluator, task_type, weight=None, mask=None, cfg=None, is_test=False):
    """
    Evaluates a model over all the batches of a data loader.
    """
    assert task_type == 'mse_regression'
    loss_fn = torch.nn.MSELoss()

    model.eval()

    y_true = []
    y_pred = []
    losses = []
    for _, batch in enumerate(tqdm(loader, desc="Eval iteration")):
        batch = batch.to(device)
        pred = model(x=batch.x, edge_index=batch.edge_index, batch=batch, is_test=is_test)

        if isinstance(pred, tuple):
            # RWNN
            assert mask is None
            pred, targets = pred
            y_true.append(targets.detach().cpu())
        elif isinstance(loss_fn, torch.nn.CrossEntropyLoss) and len(batch.y.shape) > 1 and batch.y.shape[1] == 1:  # TODO: what is the goal of this?
            targets = batch.y.view(-1, )
            if mask is not None:
                y_true.append(batch.y[mask].detach().cpu())
            else:
                y_true.append(batch.y.detach().cpu())
        else:
            if mask is not None:
                targets = batch.y[mask]
                y_true.append(batch.y[mask].detach().cpu())
            else:
                targets = batch.y
                y_true.append(batch.y.detach().cpu())

        if mask is not None:
            pred = pred[mask]
            loss = loss_fn(pred, targets)
        else:
            loss = loss_fn(pred, targets)
        losses.append(loss.detach().cpu().item())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) > 0 else None
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    return evaluator.eval(input_dict), mean_loss


class Evaluator(object):
    def __init__(self, metric, **kwargs):
        if metric == 'isomorphism':
            self.eval_fn = self._isomorphism
            self.eps = kwargs.get('eps', 0.01)
            self.p_norm = kwargs.get('p', 2)
        elif metric == 'accuracy':
            self.eval_fn = self._accuracy
        elif metric == 'mae':  # aligned with regression loss, which is L1. Otherwise the loss is L2 and the perf is L1
            self.eval_fn = self._mae
        elif metric == 'mse':  # aligned with regression loss, which is L1. Otherwise the loss is L2 and the perf is L1
            self.eval_fn = self._mse
        elif metric.startswith('ogbg-mol'):
            self._ogb_evaluator = OGBEvaluator(metric)
            self._key = self._ogb_evaluator.eval_metric
            self.eval_fn = self._ogb
        elif metric == 'avg_prec':
            self.eval_fn = self._avg_prec
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))

    def eval(self, input_dict):
        return self.eval_fn(input_dict)

    def _isomorphism(self, input_dict):
        # NB: here we return the failure percentage... the smaller the better!
        preds = input_dict['y_pred']
        assert preds is not None
        assert preds.dtype == np.float64
        preds = torch.tensor(preds, dtype=torch.float64)
        mm = torch.pdist(preds, p=self.p_norm)
        wrong = (mm < self.eps).sum().item()
        metric = wrong / mm.shape[0]
        return metric

    def _accuracy(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = np.argmax(input_dict['y_pred'], axis=1)
        assert y_true is not None
        assert y_pred is not None
        metric = met.accuracy_score(y_true, y_pred)
        return metric

    def _mae(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = met.mean_absolute_error(y_true, y_pred)
        return metric

    def _mse(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = met.mean_squared_error(y_true, y_pred)
        return metric

    def _ogb(self, input_dict, **kwargs):
        assert 'y_true' in input_dict
        assert input_dict['y_true'] is not None
        assert 'y_pred' in input_dict
        assert input_dict['y_pred'] is not None
        return self._ogb_evaluator.eval(input_dict)[self._key]

    def _avg_prec(self, input_dict, *kwargs):
        '''
            compute Average Precision (AP) averaged across tasks
        '''
        assert 'y_true' in input_dict
        assert input_dict['y_true'] is not None
        assert 'y_pred' in input_dict
        assert input_dict['y_pred'] is not None
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']

        ap_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                ap = average_precision_score(y_true[is_labeled, i],
                                             y_pred[is_labeled, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute Average Precision.')

        return sum(ap_list) / len(ap_list)
