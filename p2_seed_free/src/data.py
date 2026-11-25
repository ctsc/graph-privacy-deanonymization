"""Load seed-free graph instances (read only, in place). Never reads any seed file."""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "student_projects",
                                    "02_seed_free", "student", "seed_free"))


def list_instances(split):
    return sorted(d for d in os.listdir(os.path.join(ROOT, split))
                  if os.path.isdir(os.path.join(ROOT, split, d)))


def _adj(path, n):
    e = pd.read_csv(path)
    r, c = e["source"].to_numpy(), e["target"].to_numpy()
    m = r != c
    a = sp.coo_matrix((np.ones(m.sum()), (r[m], c[m])), shape=(n, n)).tocsr()
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr()


def load_instance(split, name):
    d = os.path.join(ROOT, split, name)
    na = len(pd.read_csv(os.path.join(d, "nodes_a.csv")))
    nb = len(pd.read_csv(os.path.join(d, "nodes_b.csv")))
    inst = {"name": name, "split": split, "n_a": na, "n_b": nb,
            "meta": json.load(open(os.path.join(d, "metadata.json"))),
            "A": _adj(os.path.join(d, "edges_a.csv"), na), "B": _adj(os.path.join(d, "edges_b.csv"), nb),
            "seeds": []}
    gt = os.path.join(d, "ground_truth.csv")
    if split != "test" and os.path.exists(gt):
        g = pd.read_csv(gt)
        inst["truth"] = dict(zip(g["node_a"].tolist(), g["node_b"].tolist()))
    return inst
