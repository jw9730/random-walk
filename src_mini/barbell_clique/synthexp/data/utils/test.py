import numpy as np
import torch

def test_baseline_error():
    n = 100
    mu = np.sqrt(3*n)
    std = np.sqrt(n)

    x = torch.empty([n, 10000]).uniform_(mu - np.sqrt(3) * std, mu + np.sqrt(3) * std)
    est = mu - x.mean(axis=0)
    err = (est ** 2).mean()
    assert np.abs(err.item() - 0.5) < 0.1  # want the error to be 0.5