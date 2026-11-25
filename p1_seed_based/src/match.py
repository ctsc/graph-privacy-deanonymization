"""Seed-based matching: seed-neighborhood overlap (percolation style) plus a small
structural term, Hungarian assignment, iterative confident-pair expansion inside ONE
seed config. Only the supplied seeds of the given config are ever used."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from features import structural_similarity

DEFAULT = {
    "lam_struct": 0.0,      # weight of structural similarity in the score
    "bandwidth": 1.0,       # structural kernel bandwidth
    "rounds": 12,           # max expansion rounds
    "accept_margin": 0.0,   # min (best - runner up) margin to accept a pseudo seed
    "min_common": 3,        # min matched common neighbors to accept a pseudo seed
    "min_common_floor": 2,  # when stuck, relax min_common down to this
    "reject_tau": 0.0,      # final rejection threshold on the final score
    "idf": 1,               # 1 = weight matched neighbors by 1/log(2+deg)
}


def _top2_margin(sub):
    """For each entry, best competing score in its row or column (excluding itself)."""
    n, m = sub.shape
    if n < 2 or m < 2:
        return np.zeros_like(sub)
    r2 = np.partition(sub, -2, axis=1)[:, -2]
    c2 = np.partition(sub, -2, axis=0)[-2, :]
    r1 = sub.max(1)
    c1 = sub.max(0)
    return r1, r2, c1, c2


def _scores(Ad, Bd, M, degA, degB, K, cfg):
    ma = np.fromiter(M.keys(), dtype=int)
    mb = np.fromiter(M.values(), dtype=int)
    if cfg["idf"]:
        wa = 1.0 / np.log(2.0 + degA[ma])
        wb = 1.0 / np.log(2.0 + degB[mb])
    else:
        wa = np.ones(len(ma), dtype=np.float32)
        wb = np.ones(len(mb), dtype=np.float32)
    Am, Bm = Ad[:, ma], Bd[:, mb]
    ncommon = Am @ Bm.T
    common = (Am * wa) @ (Bm * wb).T
    cnt_a = Am.sum(1)
    cnt_b = Bm.sum(1)
    seedsim = common / np.sqrt((cnt_a[:, None] + 1.0) * (cnt_b[None, :] + 1.0))
    T = seedsim + (cfg["lam_struct"] * K if K is not None else 0.0)
    return T, ncommon, ma, mb


def match(inst, cfg=None, return_conf=False, return_info=False):
    """return_conf: also return {a: score} for final-assignment pairs (used by rejection).
    return_info: instead return (pred, info) where info[a] = (kind, score) for every
    predicted pair, kind in {seed, pseudo, final}, score = final T[a, b]."""
    cfg = {**DEFAULT, **(cfg or {})}
    A, B = inst["A"], inst["B"]
    na, nb = inst["n_a"], inst["n_b"]
    Ad = A.toarray().astype(np.float32)
    Bd = B.toarray().astype(np.float32)
    degA, degB = Ad.sum(1), Bd.sum(1)
    K = structural_similarity(A, B, bandwidth=cfg["bandwidth"]).astype(np.float32) \
        if cfg["lam_struct"] > 0 else None
    M = dict(inst["seeds"])
    mc = cfg["min_common"]
    for _ in range(cfg["rounds"]):
        T, ncommon, ma, mb = _scores(Ad, Bd, M, degA, degB, K, cfg)
        ua = np.setdiff1d(np.arange(na), ma)
        ub = np.setdiff1d(np.arange(nb), mb)
        if len(ua) == 0:
            break
        sub = T[np.ix_(ua, ub)]
        nc = ncommon[np.ix_(ua, ub)]
        r, c = linear_sum_assignment(-sub)
        s = sub[r, c]
        if sub.shape[0] > 1 and sub.shape[1] > 1:
            row2 = np.partition(sub, -2, axis=1)[:, -2]
            col2 = np.partition(sub, -2, axis=0)[-2, :]
            margin = s - np.maximum(row2[r], col2[c])
        else:
            margin = s
        ok = (margin >= cfg["accept_margin"]) & (nc[r, c] >= mc)
        if not ok.any():
            if mc > cfg["min_common_floor"]:
                mc -= 1
                continue
            break
        for i, j in zip(r[ok], c[ok]):
            M[int(ua[i])] = int(ub[j])
    T, _, ma, mb = _scores(Ad, Bd, M, degA, degB, K, cfg)
    ua = np.setdiff1d(np.arange(na), ma)
    ub = np.setdiff1d(np.arange(nb), mb)
    pred, conf = dict(M), {}
    if len(ua):
        sub = T[np.ix_(ua, ub)]
        r, c = linear_sum_assignment(-sub)
        for i, j in zip(r, c):
            if sub[i, j] >= cfg["reject_tau"]:
                pred[int(ua[i])] = int(ub[j])
                conf[int(ua[i])] = float(sub[i, j])
    if return_info:
        seed_a = {a for a, _ in inst["seeds"]}
        info = {a: ("seed" if a in seed_a else "final" if a in conf else "pseudo", float(T[a, b]))
                for a, b in pred.items()}
        return pred, info
    return (pred, conf) if return_conf else pred
