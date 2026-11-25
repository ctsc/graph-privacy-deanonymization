"""Robustness probe on Val (analysis only, frozen config unchanged): replace an extra fraction f of
B's edges with random non-edges (same protocol as the data: deletion + replacement), then match.
Run from the privacy folder: conda run -n privacy_projects python p2_seed_free/robustness.py"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p1_seed_based", "src"))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from data import list_instances, load_instance  # noqa: E402
from metrics import instance_metrics  # noqa: E402
from seedfree import initial_matching, match_seedfree  # noqa: E402

FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def perturb(B, f, rng):
    n = B.shape[0]
    iu = sp.triu(B, 1).tocoo()
    e = np.c_[iu.row, iu.col]
    k = int(round(f * len(e)))
    keep = e[rng.permutation(len(e))[k:]]
    have = set(map(tuple, e))
    add = []
    while len(add) < k:
        a, b = rng.integers(0, n, 2)
        if a != b and (min(a, b), max(a, b)) not in have:
            have.add((min(a, b), max(a, b)))
            add.append((min(a, b), max(a, b)))
    allE = np.vstack([keep, np.array(add, dtype=int).reshape(-1, 2)])
    M = sp.coo_matrix((np.ones(len(allE)), (allE[:, 0], allE[:, 1])), shape=(n, n)).tocsr()
    M = M + M.T
    M.data[:] = 1
    return M.tocsr()


def one(name, f, cfg):
    inst = load_instance("val", name)
    inst["B"] = perturb(inst["B"], f, np.random.default_rng(1000 + int(f * 100)))
    p, _, _ = initial_matching(inst, cfg)
    r0 = instance_metrics({int(a): int(b) for a, b in enumerate(p)}, inst["truth"], [], inst["n_a"])["f1"]
    m = instance_metrics(match_seedfree(inst, cfg), inst["truth"], [], inst["n_a"])
    return {"instance": name, "family": inst["meta"]["family"], "difficulty": inst["meta"]["difficulty"],
            "extra_replaced": f, "f1_initial": r0, "f1_full": m["f1"], "precision_full": m["precision"],
            "coverage_full": m["coverage"]}


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json")))["method"]
    jobs = [(n, f) for f in FRACS for n in list_instances("val")]
    df = pd.DataFrame(Parallel(n_jobs=9)(delayed(one)(n, f, cfg) for n, f in jobs))
    out = os.path.join(HERE, "results")
    df.to_csv(os.path.join(out, "val_robustness.csv"), index=False)
    piv = df.pivot_table(index="extra_replaced", columns="difficulty", values="f1_full")[["easy", "medium", "hard"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, col in zip(piv.columns, ["#009E73", "#0072B2", "#D55E00"]):
        ax.plot(piv.index, piv[c], marker="o", color=col, label=c)
    ax.set_xlabel("extra fraction of B edges replaced (on top of supplied noise)")
    ax.set_ylabel("macro F1 (3 Val instances each)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="difficulty")
    ax.set_title("Seed-free robustness on Val")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "val_robustness.png"), dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index="extra_replaced", columns=["difficulty"], values=["f1_initial", "f1_full"]).round(3).to_string())
    print(df.pivot_table(index="extra_replaced", columns=["family"], values="f1_full").round(3).to_string())


if __name__ == "__main__":
    main()
