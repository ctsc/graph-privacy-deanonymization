"""Seed-free matching.

1. Spectral initial similarity (GRAMPA, Fan et al. 2019): X = U diag(U^T 1) W diag(V^T 1) V^T with
   W_ij = 1 / ((l_i - m_j)^2 + eta^2) on degree-normalized adjacency spectra. Hungarian gives a full
   initial matching.
2. Refinement: repeat S = A P B^T (count of neighbors kept consistent by the current matching P),
   Hungarian on S, for refine_rounds rounds.
3. Pseudo seeds: pairs whose consistent-neighbor count is at least anchor_min and at least
   anchor_frac of the smaller degree. These feed the P1 percolation matcher, which re-matches
   everything else and applies the rejection threshold.
No seed file is ever read."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from match import match as percolate

DEFAULT = {"eta": 0.2, "refine_rounds": 5, "anchor_min": 3, "anchor_frac": 0.3,
           "perc": {"min_common": 3, "min_common_floor": 1, "lam_struct": 0.02, "idf": 1, "rounds": 40,
                    "bandwidth": 1.0, "accept_margin": 0.0, "reject_tau": 0.02}}


def grampa(Ad, Bd, eta):
    def spec(M):
        d = M.sum(1).mean()
        lam, U = np.linalg.eigh(M / np.sqrt(max(d, 1.0)))
        return lam, U
    la, U = spec(Ad)
    lb, V = spec(Bd)
    W = 1.0 / ((la[:, None] - lb[None, :]) ** 2 + eta ** 2)
    a = U.sum(0)
    b = V.sum(0)
    return (U * a) @ W @ (V * b).T


def _assign(S):
    r, c = linear_sum_assignment(-S)
    p = np.empty(S.shape[0], dtype=int)
    p[r] = c
    return p


def consistency(Ad, Bd, p):
    """For each a: number of neighbors a' of a with p[a'] adjacent to p[a]."""
    return (Ad * Bd[p][:, p]).sum(1)


def initial_matching(inst, cfg):
    Ad = inst["A"].toarray().astype(np.float64)
    Bd = inst["B"].toarray().astype(np.float64)
    p = _assign(grampa(Ad, Bd, cfg["eta"]))
    for _ in range(cfg["refine_rounds"]):
        P = np.zeros_like(Bd)
        P[np.arange(len(p)), p] = 1.0
        p = _assign(Ad @ P @ Bd.T)
    return p, Ad, Bd


def match_seedfree(inst, cfg=None, return_info=False):
    cfg = {**DEFAULT, **(cfg or {})}
    p, Ad, Bd = initial_matching(inst, cfg)
    cons = consistency(Ad, Bd, p)
    dmin = np.minimum(Ad.sum(1), Bd.sum(1)[p])
    keep = (cons >= cfg["anchor_min"]) & (cons >= cfg["anchor_frac"] * np.maximum(dmin, 1))
    seeds = [(int(a), int(p[a])) for a in np.where(keep)[0]]
    if not seeds:
        pred = {int(a): int(b) for a, b in enumerate(p)}
        info = {a: ("initial", float(cons[a])) for a in pred}
        return (pred, info) if return_info else pred
    sub = {**inst, "seeds": seeds}
    pred, info = percolate(sub, cfg["perc"], return_info=True)
    info = {a: ("anchor" if k == "seed" else k, s) for a, (k, s) in info.items()}
    return (pred, info) if return_info else pred


def degree_baseline(inst):
    """Degree-only baseline: Hungarian on |deg_a - deg_b|."""
    da = np.asarray(inst["A"].sum(1)).ravel()
    db = np.asarray(inst["B"].sum(1)).ravel()
    r, c = linear_sum_assignment(np.abs(da[:, None] - db[None, :]))
    return {int(a): int(b) for a, b in zip(r, c)}
