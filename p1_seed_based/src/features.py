"""Structural node features and a structural similarity matrix."""
import numpy as np


def node_features(A):
    deg = np.asarray(A.sum(1)).ravel()
    nbr_deg_sum = A @ deg
    mean_nbr = np.divide(nbr_deg_sum, deg, out=np.zeros_like(deg), where=deg > 0)
    tri = np.asarray(A.multiply(A @ A).sum(1)).ravel() / 2.0
    possible = deg * (deg - 1) / 2.0
    clust = np.divide(tri, possible, out=np.zeros_like(deg), where=possible > 0)
    two_hop = A @ deg  # number of length-2 walks, cheap 2-hop size proxy
    return np.column_stack([np.log1p(deg), np.log1p(mean_nbr), clust, np.log1p(two_hop)])


def structural_similarity(A, B, weights=(1.0, 1.0, 0.5, 0.5), bandwidth=1.0):
    """Return K in (0,1], n_a x n_b. Features standardized with pooled std."""
    fa, fb = node_features(A), node_features(B)
    pooled = np.vstack([fa, fb])
    sd = pooled.std(0) + 1e-9
    w = np.asarray(weights)
    fa, fb = fa / sd * np.sqrt(w), fb / sd * np.sqrt(w)
    d2 = (fa ** 2).sum(1)[:, None] + (fb ** 2).sum(1)[None, :] - 2 * fa @ fb.T
    d2 = np.maximum(d2, 0)
    return np.exp(-d2 / (2 * bandwidth ** 2 * w.sum()))
