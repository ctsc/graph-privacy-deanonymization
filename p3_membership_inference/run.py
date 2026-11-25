"""P3 membership inference vs early stopping, single entry point.

Usage (from the privacy folder):
  conda run -n privacy_projects python p3_membership_inference/run.py

For each seed in config.json and each training mode (fixed 600 epochs, early stopping):
  1. build disjoint stratified pools, 2. train target, 3 dev shadows and 1 calibration shadow,
  3. fit attacks on dev shadow records, choose C and thresholds on the calibration shadow only (frozen),
  4. evaluate once on target attack-Test: 100 target-train members vs 100 target-test nonmembers.
Target validation is never used in attack evaluation."""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

import attack as atk  # noqa: E402
from data import load, make_pools  # noqa: E402
from mlp import train  # noqa: E402

RESULTS = os.path.join(HERE, "results")
MODES = ("fixed", "early")


def records(model, X, y, members, nonmembers):
    idx = np.concatenate([members, nonmembers])
    F, loss = atk.features(model.proba(X[idx]), y[idx])
    lab = np.r_[np.ones(len(members)), np.zeros(len(nonmembers))].astype(int)
    return F, loss, lab


def acc(model, X, y, idx):
    return float((model.proba(X[idx]).argmax(1) == y[idx]).mean())


def run_one(seed, mode, X, y, cfg):
    pools = make_pools(y, cfg, seed)
    tt, tv, te = pools["target_train"], pools["target_val"], pools["target_test"]
    target, hist, ep = train(X, y, tt, tv, mode, cfg, seed)
    shadow_info, devF, devL, devY = [], [], [], []
    for k, sh in enumerate(pools["dev_shadows"]):
        m, _, sep = train(X, y, sh["train"], sh["val"], mode, cfg, seed * 10 + k + 1)
        F, L, lab = records(m, X, y, sh["train"], sh["nonmember"])
        devF.append(F), devL.append(L), devY.append(lab)
        shadow_info.append({"shadow": f"dev{k}", "epochs": sep, "train_acc": acc(m, X, y, sh["train"]),
                            "nonmember_acc": acc(m, X, y, sh["nonmember"])})
    devF, devL, devY = np.vstack(devF), np.concatenate(devL), np.concatenate(devY)
    cs = pools["calib_shadow"]
    cm, _, cep = train(X, y, cs["train"], cs["val"], mode, cfg, seed * 10 + 9)
    calF, calL, calY = records(cm, X, y, cs["train"], cs["nonmember"])
    shadow_info.append({"shadow": "calib", "epochs": cep, "train_acc": acc(cm, X, y, cs["train"]),
                        "nonmember_acc": acc(cm, X, y, cs["nonmember"])})

    # attack selection on the calibration shadow only, then frozen
    acfg = cfg["attack"]
    attacks, choice = {}, {}
    la = atk.LossAttack().fit(devF, devL, devY)
    attacks[la.name] = (la, atk.choose_threshold(calY, la.score(calF, calL), acfg["target_fpr"]))
    best = None
    for C in acfg["logistic_C_grid"]:
        lr = atk.LogisticAttack(C, seed).fit(devF, devL, devY)
        a = roc_auc_score(calY, lr.score(calF, calL))
        if best is None or a > best[0]:
            best = (a, C, lr)
    choice["logistic_C"] = best[1]
    attacks[best[2].name] = (best[2], atk.choose_threshold(calY, best[2].score(calF, calL), acfg["target_fpr"]))

    # single evaluation on target attack-Test
    tF, tL, tY = records(target, X, y, tt, te)
    rows, rocs = [], []
    for name, (a, t) in attacks.items():
        cal = atk.evaluate(calY, a.score(calF, calL), t)
        s = a.score(tF, tL)
        res = atk.evaluate(tY, s, t)
        fpr, tpr, _ = roc_curve(tY, s)
        rocs.append({"seed": seed, "mode": mode, "attack": name, "fpr": fpr.tolist(), "tpr": tpr.tolist()})
        rows.append({"seed": seed, "mode": mode, "attack": name, "target_epochs": ep,
                     "target_train_acc": acc(target, X, y, tt), "target_val_acc": acc(target, X, y, tv),
                     "target_test_acc": acc(target, X, y, te),
                     **{f"target_{k}": v for k, v in res.items()},
                     **{f"calib_{k}": v for k, v in cal.items() if k != "threshold"},
                     "logistic_C": choice["logistic_C"] if name == "shadow_logistic" else np.nan})
    split_ids = {k: v.tolist() for k, v in pools.items() if isinstance(v, np.ndarray)}
    split_ids["dev_shadows"] = [{p: v.tolist() for p, v in sh.items()} for sh in pools["dev_shadows"]]
    split_ids["calib_shadow"] = {p: v.tolist() for p, v in cs.items()}
    return rows, rocs, {"seed": seed, "mode": mode, **hist}, [{"seed": seed, "mode": mode, **s} for s in shadow_info], split_ids


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    X, y, rid, ndup = load()
    print("records %d, pixel duplicates removed %d" % (len(y), ndup))
    jobs = [(s, m) for s in cfg["seeds"] for m in MODES]
    out = Parallel(n_jobs=len(jobs))(delayed(run_one)(s, m, X, y, cfg) for s, m in jobs)
    os.makedirs(RESULTS, exist_ok=True)
    per = pd.DataFrame([r for o in out for r in o[0]])
    per.to_csv(os.path.join(RESULTS, "per_seed.csv"), index=False)
    pd.DataFrame([r for o in out for r in o[3]]).to_csv(os.path.join(RESULTS, "shadow_models.csv"), index=False)
    json.dump([r for o in out for r in o[1]], open(os.path.join(RESULTS, "roc_curves.json"), "w"))
    json.dump([o[2] for o in out], open(os.path.join(RESULTS, "learning_curves.json"), "w"))
    splits = {f"seed{s}": o[4] for (s, m), o in zip(jobs, out) if m == "fixed"}
    for v in splits.values():
        for k in list(v):
            if isinstance(v[k], list) and v[k] and isinstance(v[k][0], int):
                v[k] = [int(rid[i]) for i in v[k]]
        v["dev_shadows"] = [{p: [int(rid[i]) for i in ids] for p, ids in sh.items()} for sh in v["dev_shadows"]]
        v["calib_shadow"] = {p: [int(rid[i]) for i in ids] for p, ids in v["calib_shadow"].items()}
    json.dump(splits, open(os.path.join(RESULTS, "split_record_ids.json"), "w"))

    mets = [c for c in per.columns if c.startswith(("target_", "calib_"))]
    summ = per.groupby(["mode", "attack"])[mets].agg(["mean", lambda s: s.std(ddof=1)])
    summ.columns = [f"{a}_{'mean' if b == 'mean' else 'std'}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(RESULTS, "summary_mean_std.csv"), index=False)
    import figures
    figures.make_all(RESULTS)
    pd.set_option("display.width", 250)
    show = ["seed", "mode", "attack", "target_epochs", "target_train_acc", "target_test_acc", "target_auc",
            "target_tpr_at_5fpr", "target_tpr_at_threshold", "target_fpr_at_threshold", "calib_auc", "logistic_C"]
    print(per[show].round(3).to_string(index=False))
    keep = [c for c in summ.columns if c.startswith("target_") and "threshold_" not in c[:17]]
    print(summ[keep].round(3).T.to_string())


if __name__ == "__main__":
    main()
