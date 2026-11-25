"""Tuning on PUBLIC validation only: prior factor, Adam lr, iterations. Prints mean MSE per setting.
Run from the privacy folder: conda run -n privacy_projects python p4_dp_reconstruction/tune.py"""
import glob
import itertools
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from attack import Prior  # noqa: E402
from evaluate import per_image_mse  # noqa: E402
from run import DATA, attack_file, public  # noqa: E402

BASE = {"lr": 0.05, "iters": 1000, "restarts": 3, "prior_factor": 1.0, "sigma0_prior": 0.0, "var_floor": 0.01}


def mse_file(path, attack, cfg, pub):
    v = np.load(path)
    X, _, _ = attack_file(path, attack, cfg, Prior(pub, cfg["var_floor"]), seed=7)
    return float(np.mean([per_image_mse(X[k], v["originals"][k], v["labels"][k]).mean() for k in range(len(X))]))


def grid(attack, files, settings, pub):
    jobs = [(f, s) for s in settings for f in files]
    res = Parallel(n_jobs=10)(delayed(mse_file)(f, attack, {**BASE, **s}, pub) for f, s in jobs)
    rows = [{"file": os.path.basename(f)[:-4], **s, "mse": m} for (f, s), m in zip(jobs, res)]
    return pd.DataFrame(rows)


def main():
    pub = public()
    allf = sorted(glob.glob(os.path.join(DATA, "validation", "*.npz")))
    dp = [f for f in allf if "eps" in f]
    nodp = [f for f in allf if "eps" not in f]
    pd.set_option("display.width", 220)
    out = []
    print("== plain, optimizer settings, all validation files")
    g = grid("plain", allf, [dict(lr=lr, iters=it) for lr, it in itertools.product([0.02, 0.05, 0.1], [500, 1500])], pub)
    out.append(g.assign(attack="plain"))
    print(g.pivot_table(index="file", columns=["lr", "iters"], values="mse").round(4).to_string())
    print("== prior, prior_factor, DP validation files")
    g = grid("prior", dp, [dict(prior_factor=pf) for pf in [0.1, 0.3, 1.0, 3.0, 10.0]], pub)
    out.append(g.assign(attack="prior"))
    print(g.pivot_table(index="file", columns="prior_factor", values="mse").round(4).to_string())
    print(g.groupby("prior_factor").mse.mean().round(5).to_string())
    print("== prior with a small prior on sigma = 0 files (sigma0_prior)")
    g = grid("prior", nodp, [dict(sigma0_prior=s0) for s0 in [0.0, 1e-4, 1e-3]], pub)
    out.append(g.assign(attack="prior"))
    print(g.pivot_table(index="file", columns="sigma0_prior", values="mse").round(5).to_string())
    pd.concat(out).to_csv(os.path.join(HERE, "results", "tuning_validation.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    main()
