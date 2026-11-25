"""Train-only sweep: reject_tau (offline from confidences), rounds and bandwidth.
Run from p1_seed_based: conda run -n privacy_projects python tuning/sweep_tau_rounds_bw.py"""
import sys; sys.path.insert(0, '.')
import os
import time
import numpy as np
import tune_util  # noqa: F401  (sets thread env vars and sys.path)
from joblib import Parallel, delayed
from data import load_instance, list_instances, SEED_CONFIGS
from match import match
from metrics import instance_metrics, evaluate_rejection

BASE = {'min_common': 3, 'min_common_floor': 1, 'lam_struct': 0.02, 'idf': 1, 'rounds': 40,
        'bandwidth': 1.0, 'accept_margin': 0.0, 'reject_tau': 0.0}
TAUS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def one(name, sc, cfg):
    inst = load_instance('train', name, sc)
    t = time.time()
    pred, conf = match(inst, cfg, return_conf=True)
    return name, sc, time.time() - t, evaluate_rejection(pred, conf, inst, TAUS)


names = list_instances('train')
jobs = [(n, sc) for sc in SEED_CONFIGS for n in names]
res = Parallel(n_jobs=10)(delayed(one)(n, sc, BASE) for n, sc in jobs)
print('reject_tau sweep (Train macro over 27 instances, mean over 6 configs)')
for k, tau in enumerate(TAUS):
    m = {key: np.mean([r[3][k][key] for r in res]) for key in ('precision', 'recall', 'f1', 'coverage')}
    print(tau, {a: round(b, 4) for a, b in m.items()})
print('mean runtime s', round(np.mean([r[2] for r in res]), 3))

for rounds, bw in [(20, 1.0), (40, 1.0), (80, 1.0), (40, 0.5), (40, 2.0)]:
    cfg = {**BASE, 'rounds': rounds, 'bandwidth': bw}
    out = tune_util.evaluate('train', cfg, SEED_CONFIGS)
    print('rounds', rounds, 'bw', bw, {k: round(v, 4) for k, v in out.items()}, round(np.mean(list(out.values())), 4), flush=True)
