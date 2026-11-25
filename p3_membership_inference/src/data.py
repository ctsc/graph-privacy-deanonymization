"""Load Digits, dedupe pixels, build disjoint stratified pools (read in place, never modified)."""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "student_projects",
                                    "03_membership_inference", "data", "digits", "digits.csv"))


def load():
    d = pd.read_csv(PATH)
    # the label column is named "income" in the supplied CSV; it holds the digit class 0-9
    label_col = "label" if "label" in d.columns else "income"
    px = [c for c in d.columns if c.startswith("pixel_")]
    n0 = len(d)
    d = d.drop_duplicates(subset=px, keep="first").reset_index(drop=True)
    X = d[px].to_numpy(float)
    y = d[label_col].to_numpy(int)
    return X, y, d["record_id"].to_numpy(int), n0 - len(d)


def _take(idx, y, n, seed):
    """Stratified draw of n from idx; returns (chosen, rest)."""
    a, b = train_test_split(idx, train_size=n, stratify=y[idx], random_state=seed)
    return np.sort(a), np.sort(b)


def make_pools(y, cfg, seed):
    p, s = cfg["pools"], cfg["shadow_sizes"]
    rest = np.arange(len(y))
    pools = {}
    for k in ("target_train", "target_val", "target_test", "shadow_pool"):
        pools[k], rest = _take(rest, y, p[k], seed)
    pools["unused"] = rest
    cal, dev = _take(pools["shadow_pool"], y, p["shadow_calibration"], seed)
    assert len(dev) == p["shadow_attack_train"]
    shadows = []
    for k in range(cfg["n_dev_shadows"]):
        sh = {}
        for part in ("train", "val", "nonmember"):
            if part == "nonmember" and k == cfg["n_dev_shadows"] - 1:
                sh[part], dev = dev, dev[:0]
            else:
                sh[part], dev = _take(dev, y, s[part], seed + k + 1)
        shadows.append(sh)
    calib = {}
    for part in ("train", "val"):
        calib[part], cal = _take(cal, y, s[part], seed + 100)
    calib["nonmember"] = cal
    pools["dev_shadows"], pools["calib_shadow"] = shadows, calib
    _check_disjoint(pools)
    return pools


def _check_disjoint(pools):
    parts = [pools["target_train"], pools["target_val"], pools["target_test"], pools["unused"]]
    for sh in pools["dev_shadows"] + [pools["calib_shadow"]]:
        parts += [sh["train"], sh["val"], sh["nonmember"]]
    allidx = np.concatenate(parts)
    assert len(allidx) == len(set(allidx.tolist())), "pools overlap"
