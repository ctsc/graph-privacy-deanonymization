"""Validity checker for Test predictions: every test instance x seed config has a file with
columns exactly node_a,node_b, integer ids that exist in nodes_a.csv / nodes_b.csv, no
duplicate node_a or node_b, and every seed pair of that config included unchanged."""
import os

import pandas as pd

from data import ROOT, SEED_CONFIGS, list_instances


def check_file(path, inst_dir, sc):
    errs = []
    if not os.path.exists(path):
        return ["missing file"]
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip()
    if header != "node_a,node_b":
        errs.append("bad header %r" % header)
    df = pd.read_csv(path)
    if len(df) == 0:
        errs.append("empty")
    if not all(pd.api.types.is_integer_dtype(df[c]) for c in df.columns):
        errs.append("non-integer ids")
        return errs
    na = set(pd.read_csv(os.path.join(inst_dir, "nodes_a.csv"))["node_id"])
    nb = set(pd.read_csv(os.path.join(inst_dir, "nodes_b.csv"))["node_id"])
    if df.node_a.duplicated().any():
        errs.append("duplicate node_a")
    if df.node_b.duplicated().any():
        errs.append("duplicate node_b")
    if not set(df.node_a) <= na:
        errs.append("invalid node_a")
    if not set(df.node_b) <= nb:
        errs.append("invalid node_b")
    seeds = pd.read_csv(os.path.join(inst_dir, f"seeds_{sc}.csv"))
    got = dict(zip(df.node_a, df.node_b))
    if any(got.get(a) != b for a, b in zip(seeds.node_a, seeds.node_b)):
        errs.append("seed pair missing or changed")
    return errs


def main(pred_dir):
    names = list_instances("test")
    n_ok, bad = 0, []
    for sc in SEED_CONFIGS:
        for n in names:
            e = check_file(os.path.join(pred_dir, sc, n, "predictions.csv"), os.path.join(ROOT, "test", n), sc)
            if e:
                bad.append((sc, n, e))
            else:
                n_ok += 1
    total = len(SEED_CONFIGS) * len(names)
    for b in bad:
        print("FAIL", *b)
    print("%d / %d prediction files valid" % (n_ok, total))
    return 0 if n_ok == total else 1
