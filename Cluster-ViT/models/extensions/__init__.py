import torch

from .broadcast import _broadcast
from .weighted_sum import _weighted_sum

def weighted_sum(x, groups, weights):
    assert x.size(1) == groups.size(1)
    B, N, D = x.shape
    C = weights.size(1)
    y = torch.zeros((B, C, D), dtype=torch.float, device=x.device)
    _weighted_sum(x, groups, weights, y)
    return y


def broadcast(y, groups, weights):
    assert y.size(1) == weights.size(1)
    B, C, D = y.shape
    N = groups.size(1)
    x = torch.zeros((B, N, D), dtype=torch.float, device=y.device)
    _broadcast(y, groups, weights, x)
    return x
