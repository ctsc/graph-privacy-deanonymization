"""Load graph instances from student_projects (read only, in place)."""
import json
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "student_projects",
                                    "01_seed_based", "student", "seed_based"))
SEED_CONFIGS = [f"{s}_{p}" for s in ("random", "degree") for p in ("01", "05", "10")]


def list_instances(split):
    return sorted(d for d in os.listdir(os.path.join(ROOT, split))
                  if os.path.isdir(os.path.join(ROOT, split, d)))


def _adj(path, n):
    e = pd.read_csv(path)
    r = e["source"].to_numpy()
    c = e["target"].to_numpy()
    m = r != c
    r, c = r[m], c[m]
    a = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr()


def load_instance(split, name, seed_config=None):
    d = os.path.join(ROOT, split, name)
    na = len(pd.read_csv(os.path.join(d, "nodes_a.csv")))
    nb = len(pd.read_csv(os.path.join(d, "nodes_b.csv")))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    inst = {"name": name, "split": split, "n_a": na, "n_b": nb, "meta": meta,
            "A": _adj(os.path.join(d, "edges_a.csv"), na),
            "B": _adj(os.path.join(d, "edges_b.csv"), nb)}
    if seed_config is not None:
        s = pd.read_csv(os.path.join(d, f"seeds_{seed_config}.csv"))
        inst["seeds"] = list(zip(s["node_a"].tolist(), s["node_b"].tolist()))
    gt = os.path.join(d, "ground_truth.csv")
    if os.path.exists(gt) and split != "test":
        g = pd.read_csv(gt)
        inst["truth"] = dict(zip(g["node_a"].tolist(), g["node_b"].tolist()))
    return inst
