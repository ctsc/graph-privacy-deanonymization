"""Parallel evaluation helper used for tuning on Train/Val only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from joblib import Parallel, delayed
from data import load_instance, list_instances
from match import match
from metrics import instance_metrics


def _one(split, name, sc, cfg):
    inst = load_instance(split, name, sc)
    pred = match(inst, cfg)
    return instance_metrics(pred, inst["truth"], inst["seeds"], inst["n_a"])


def evaluate(split, cfg, configs, key="f1", n_jobs=10):
    names = list_instances(split)
    jobs = [(n, sc) for sc in configs for n in names]
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(split, n, sc, cfg) for n, sc in jobs)
    out = {}
    for sc in configs:
        out[sc] = float(np.mean([r[key] for (n, s), r in zip(jobs, res) if s == sc]))
    return out
