"""Sanity checks (public validation only): finite-difference check of the analytic gradient, and the
residual between observed g_t and the true clipped gradient of the PUBLIC validation originals,
which should be 0 without noise and have per-coordinate std sigma_per_step with DP noise.
Run from the privacy folder: conda run -n privacy_projects python p4_dp_reconstruction/sanity.py"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import numpy as np  # noqa: E402

from grad import batch_grad, objective, observed_grads, unpack  # noqa: E402

DATA = os.path.join(HERE, "..", "student_projects", "04_dp_reconstruction")


def fd_check():
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(3, 10, 64)) * 0.3, rng.normal(size=(3, 10)) * 0.3
    X, y = rng.uniform(size=(2, 64)), np.array([3, 3])
    G = rng.normal(size=(3, 10, 65)) * 0.05
    for C in (np.inf, 1.0, 0.05):
        f, g = objective(X, y, W, b, G, C)
        num = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(64):
                E = np.zeros_like(X)
                E[i, j] = 1e-6
                num[i, j] = (objective(X + E, y, W, b, G, C)[0] - objective(X - E, y, W, b, G, C)[0]) / 2e-6
        print("C=%s  max |analytic - numeric| = %.2e (scale %.2e)" % (C, np.abs(g - num).max(), np.abs(num).max()))


def residuals():
    for f in sorted(glob.glob(os.path.join(DATA, "validation", "*.npz"))):
        v = np.load(f)
        C = float(v["clip"]) if np.isfinite(v["clip"]) else np.inf
        res = []
        for k in range(v["states"].shape[0]):
            st = v["states"][k]
            W, b = unpack(st[:-1])
            g = observed_grads(st, float(v["lr"]))
            Gt = batch_grad(W, b, v["originals"][k], v["labels"][k], C)
            Gm = np.concatenate([Gt[:, :, :64].reshape(10, 640), Gt[:, :, 64]], 1)
            res.append(g - Gm)
        res = np.concatenate(res)
        print("%-24s trials %2d  residual std %.4f  sigma_per_step %.4f" % (
            os.path.basename(f), v["states"].shape[0], res.std(), float(v["sigma_per_step"])))


if __name__ == "__main__":
    fd_check()
    residuals()
