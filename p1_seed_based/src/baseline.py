"""Seed-free structural baseline: degree, mean neighbor degree and clustering per node,
standardized with pooled std, Hungarian on Euclidean feature distance. Ignores seeds."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from features import node_features


def baseline_match(inst):
    fa = node_features(inst["A"])[:, :3]
    fb = node_features(inst["B"])[:, :3]
    sd = np.vstack([fa, fb]).std(0) + 1e-9
    fa, fb = fa / sd, fb / sd
    d2 = (fa ** 2).sum(1)[:, None] + (fb ** 2).sum(1)[None, :] - 2 * fa @ fb.T
    r, c = linear_sum_assignment(np.maximum(d2, 0))
    return {int(a): int(b) for a, b in zip(r, c)}
