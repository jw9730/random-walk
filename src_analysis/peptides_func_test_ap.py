# pylint: disable=protected-access,too-many-locals,unused-argument,line-too-long,too-many-instance-attributes,too-many-arguments,not-callable
from pathlib import Path
import warnings
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional.classification import multilabel_average_precision, binary_average_precision
from sklearn.metrics import average_precision_score


def eval_ap(y_true, y_pred):
    """
    Compute Average Precision (AP) averaged across tasks.
    """
    ap_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ap_list.append(ap)
    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')
    return {'ap': sum(ap_list) / len(ap_list)}


def classification_multilabel(_true, _pred, test_scores=True):
    """
    GraphGym version
    """
    true, pred_score = torch.cat(_true), torch.cat(_pred)  # [N, 10]
    reformat = lambda x: round(float(x), 5)
    # Send to GPU to speed up TorchMetrics if possible.
    true = true.to('cpu')
    pred_score = pred_score.to('cpu')
    results = {}
    if true.shape[0] < 1e6:
        ogb_ap = reformat(eval_ap(true.cpu().numpy(), pred_score.cpu().numpy())['ap'])
        results['ap'] = ogb_ap
    if test_scores:
        # Compute metric by OGB Evaluator methods.
        true = true.cpu().numpy()
        pred_score = pred_score.cpu().numpy()
        ogb = {'ap': reformat(eval_ap(true, pred_score)['ap']),}
        assert np.isclose(ogb['ap'], results['ap'], atol=1e-05)
    return results


def global_evaluator(y_hat: List, y: List) -> Dict:
    """Implementation from LRGB:
    https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metric_wrapper.py
    https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/logger.py
    https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metrics_ogb.py
    """
    preds, target = torch.cat(y_hat, dim=0), torch.cat(y, dim=0)
    # torchmetrics expects probability preds and long targets
    preds = torch.sigmoid(preds)
    target = target.long()
    # compute metrics
    determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    if torch.isnan(target).any():
        warnings.warn("NaNs in targets, falling back to slow evaluation.")
        # Compute the metric for each column, and output nan if there's an error on a given column
        target_nans = torch.isnan(target)
        target_list = [target[..., ii][~target_nans[..., ii]] for ii in range(target.shape[-1])]
        preds_list = [preds[..., ii][~target_nans[..., ii]] for ii in range(preds.shape[-1])]
        target = target_list
        preds = preds_list
        metric_val = []
        for p, t in zip(preds, target):
            # below is tested to be equivalent with OGB metric
            # https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metrics_ogb.py
            res = binary_average_precision(p, t)
            metric_val.append(res)
        # Average the metric
        # PyTorch 1.10
        metric_val = torch.nanmean(torch.stack(metric_val))
        # PyTorch <= 1.9
        # x = torch.stack(metric_val)
        # metric_val = torch.div(torch.nansum(x), (~torch.isnan(x)).count_nonzero())
    else:
        # below is tested to be equivalent with OGB metric
        # https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/metrics_ogb.py
        metric_val = multilabel_average_precision(preds, target, num_labels=10)
    torch.use_deterministic_algorithms(determinism)
    return metric_val.item()


# each file contains 20-sample logit averages
path = Path('../experiments/checkpoints/classification_peptidesfunc/microsoft_deberta-base,pt_True,l_1024,w_min_degree,wl_200,n_True,nw_1,enw_1,b_128,es_validation_accuracy_max_100,lr_2e-05_2e-05,steps_100000,wu_5000,wd_0.01,seed_42,/test_outputs_n_walks_20/')
indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
test_outputs = {}
for i in indices:
    y_hat_list, y_list = torch.load(path / f'seed_{i}.pt')
    test_outputs[i] = (list(y_hat_list), list(y_list))

# consistency check for y_list
y_list = test_outputs[indices[0]][1]
for i in indices[1:]:
    y_list_i = test_outputs[i][1]
    assert (torch.cat(y_list) == torch.cat(y_list_i)).all()
num_test_batches = len(y_list)

# Monte Carlo averaging samples sizes to evaluate the performances
sample_sizes = [20, 40, 60, 80, 100, 160, 320]

# estimate the mean and std of the metric
results_mean = []
results_std = []
for sample_size in sample_sizes:
    # each test output is already an 20-sample Monte Carlo average
    # so average 1/20 of the target sample size
    num_to_average = sample_size // 20
    # perform averaging
    results = []
    truncated_index_list = indices[:(len(indices) // num_to_average) * num_to_average]
    chunked_indices = np.array(truncated_index_list).reshape(-1, num_to_average)
    for indices_to_average in chunked_indices:
        y_hat_list_list = [test_outputs[index][0] for index in indices_to_average]
        # reduce y_hat_list_list by averaging
        avg_y_hat_list = [0.] * num_test_batches
        l = len(avg_y_hat_list)
        for y_hat_list_i in y_hat_list_list:
            for j in range(num_test_batches):
                avg_y_hat_list[j] += y_hat_list_i[j].clone().detach()
        for j in range(num_test_batches):
            avg_y_hat_list[j] /= num_to_average
        # evaluate the metric
        result = global_evaluator(avg_y_hat_list, y_list)
        alt_result = classification_multilabel(y_list, avg_y_hat_list, test_scores=True)
        if not np.isclose(result, alt_result['ap'], atol=1e-05):
            print(f"{result} != {alt_result['ap']}")
        results.append(result)
    results_mean.append(np.mean(results))
    results_std.append(np.std(results, ddof=1))

for sample_size, mean, std in zip(sample_sizes, results_mean, results_std):
    print(f"Sample size: {sample_size},\tTest AP: {mean:.4f} ± {std:.4f}")

# the below were measured separately
sample_sizes_prefix = [1, 10]
results_mean_prefix = [0.5300, 0.6937]
results_std_prefix = [0.0092, 0.0039]
sample_sizes = sample_sizes_prefix + sample_sizes
results_mean = results_mean_prefix + results_mean
results_std = results_std_prefix + results_std

# draw errorbar plot
root = Path('../experiments/figures/peptides_func')
root.mkdir(parents=True, exist_ok=True)
plt.errorbar(sample_sizes, results_mean, yerr=results_std, capsize=5)
plt.xscale('log')
plt.xlabel('Sample size')
plt.ylabel('Test AP')
plt.xlim(1, 320)
plt.ylim(0.51, 0.73)
plt.xticks([1, 10, 20, 40, 60, 80, 100, 160, 320])
plt.yticks([0.51, 0.55, 0.6, 0.65, 0.7, 0.73])
plt.tight_layout()
plt.title('Peptides-func test AP vs. sample size')
plt.savefig(root / 'plot.pdf')
