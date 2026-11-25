"""Validation metrics: label-constrained Hungarian alignment within a trial, then per-image MSE/PSNR."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def align(recon, orig, labels):
    """Permute reconstructed slots among same-label slots to minimize MSE. Returns aligned recon."""
    cost = ((recon[:, None, :] - orig[None, :, :]) ** 2).mean(-1)
    cost[labels[:, None] != labels[None, :]] = 1e9
    r, c = linear_sum_assignment(cost)
    out = np.empty_like(recon)
    out[c] = recon[r]
    return out


def per_image_mse(recon, orig, labels):
    return ((align(recon, orig, labels) - orig) ** 2).mean(-1)


def psnr(mse):
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
