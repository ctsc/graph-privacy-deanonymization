"""Parallel evaluator for tuning on Train/Val only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p1_seed_based", "src"))
sys.path.insert(0, os.path.join(HERE, "src"))
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data import load_instance, list_instances
from seedfree import match_seedfree, initial_matching
from metrics import instance_metrics


def _one(split, name, cfg, stage):
    inst = load_instance(split, name)
    t = time.perf_counter()
    if stage == "initial":
        p, _, _ = initial_matching(inst, {**__import__("seedfree").DEFAULT, **cfg})
        pred = {int(a): int(b) for a, b in enumerate(p)}
    else:
        pred = match_seedfree(inst, cfg)
    r = instance_metrics(pred, inst["truth"], [], inst["n_a"])
    r.update(instance=name, family=inst["meta"]["family"], difficulty=inst["meta"]["difficulty"],
             runtime_s=time.perf_counter() - t)
    return r


def evaluate(split, cfg, stage="full", n_jobs=10, names=None):
    names = names or list_instances(split)
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_one)(split, n, cfg, stage) for n in names))
