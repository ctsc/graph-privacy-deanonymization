"""P2 seed-free graph matching, single entry point. No seed file is read anywhere.

Usage (from the privacy folder):
  conda run -n privacy_projects python p2_seed_free/run.py val     # Val metrics, tables, figures
  conda run -n privacy_projects python p2_seed_free/run.py test    # Test predictions (18 files)
  conda run -n privacy_projects python p2_seed_free/run.py check   # validity check"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p1_seed_based", "src"))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from data import ROOT, list_instances, load_instance  # noqa: E402
from metrics import instance_metrics  # noqa: E402
from seedfree import degree_baseline, initial_matching, match_seedfree  # noqa: E402

RESULTS = os.path.join(HERE, "results")
PRED_DIR = os.path.join(HERE, "predictions")
METS = ["precision", "recall", "f1", "coverage", "runtime_s"]


def _row(inst, method, pred, rt):
    r = {"method": method, "instance": inst["name"], "family": inst["meta"]["family"],
         "difficulty": inst["meta"]["difficulty"], "n": inst["n_a"], "runtime_s": rt}
    m = instance_metrics(pred, inst["truth"], [], inst["n_a"])
    r.update({k: m[k] for k in ("precision", "recall", "f1", "coverage")})
    return r


def _val_one(name, cfg):
    inst = load_instance("val", name)
    rows = []
    t = time.perf_counter()
    rows.append(_row(inst, "degree_baseline", degree_baseline(inst), time.perf_counter() - t))
    t = time.perf_counter()
    p, _, _ = initial_matching(inst, cfg)
    rows.append(_row(inst, "spectral_initial_only", {int(a): int(b) for a, b in enumerate(p)}, time.perf_counter() - t))
    t = time.perf_counter()
    pred, info = match_seedfree(inst, cfg, return_info=True)
    rows.append(_row(inst, "spectral_refine_percolate", pred, time.perf_counter() - t))
    pairs = [{"instance": name, "node_a": a, "kind": k, "score": s, "correct": int(inst["truth"].get(a) == pred[a])}
             for a, (k, s) in info.items()]
    return rows, pairs


def run_val(cfg, jobs):
    os.makedirs(RESULTS, exist_ok=True)
    out = Parallel(n_jobs=jobs)(delayed(_val_one)(n, cfg) for n in list_instances("val"))
    per = pd.DataFrame([r for rs, _ in out for r in rs])
    pairs = pd.DataFrame([p for _, ps in out for p in ps])
    per.to_csv(os.path.join(RESULTS, "val_per_instance.csv"), index=False)
    pairs.to_csv(os.path.join(RESULTS, "val_pair_scores.csv"), index=False)
    macro = per.groupby("method", sort=False)[METS].mean().reset_index()
    macro.to_csv(os.path.join(RESULTS, "val_macro.csv"), index=False)
    for g in ("family", "difficulty"):
        per.groupby(["method", g], sort=False)[METS].mean().reset_index().to_csv(
            os.path.join(RESULTS, f"val_macro_by_{g}.csv"), index=False)
    kinds = pairs.groupby("kind").agg(pairs=("correct", "size"), precision=("correct", "mean")).reset_index()
    kinds.to_csv(os.path.join(RESULTS, "val_precision_by_pair_kind.csv"), index=False)
    figures(per)
    pd.set_option("display.width", 200)
    print(macro.round(4).to_string(index=False))
    print(per.pivot_table(index=["family", "difficulty"], columns="method", values="f1").round(4).to_string())
    print(kinds.to_string(index=False))


def figures(per):
    piv = per.pivot_table(index="difficulty", columns="method", values="f1").loc[["easy", "medium", "hard"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    piv.plot.bar(ax=ax, color=["#7F7F7F", "#0072B2", "#D55E00"][:piv.shape[1]], rot=0)
    ax.set_ylabel("macro F1 over Val instances")
    ax.set_ylim(0, 1.05)
    ax.set_title("Val F1 by difficulty, seed-free")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "val_f1_by_difficulty.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for meth, g in per.groupby("method"):
        s = g.groupby("n").runtime_s.mean()
        ax.plot(s.index, s.values, marker="o", label=meth)
    ax.set_xlabel("nodes per graph")
    ax.set_ylabel("mean runtime per instance (s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "val_runtime.png"), dpi=150)
    plt.close(fig)


def _test_one(name, cfg):
    inst = load_instance("test", name)
    assert "truth" not in inst
    t = time.perf_counter()
    pred = match_seedfree(inst, cfg)
    rt = time.perf_counter() - t
    d = os.path.join(PRED_DIR, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(sorted(pred.items()), columns=["node_a", "node_b"]).to_csv(os.path.join(d, "predictions.csv"), index=False)
    return {"instance": name, "pairs": len(pred), "runtime_s": rt}


def run_test(cfg, jobs):
    log = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(_test_one)(n, cfg) for n in list_instances("test")))
    os.makedirs(RESULTS, exist_ok=True)
    log.to_csv(os.path.join(RESULTS, "test_prediction_log.csv"), index=False)
    print("wrote %d prediction files" % len(log))


def check():
    names = list_instances("test")
    ok = 0
    for n in names:
        path = os.path.join(PRED_DIR, n, "predictions.csv")
        errs = []
        if not os.path.exists(path):
            errs.append("missing")
        else:
            header = open(path, encoding="utf-8").readline().strip()
            df = pd.read_csv(path)
            na = set(pd.read_csv(os.path.join(ROOT, "test", n, "nodes_a.csv"))["node_id"])
            nb = set(pd.read_csv(os.path.join(ROOT, "test", n, "nodes_b.csv"))["node_id"])
            if header != "node_a,node_b":
                errs.append("bad header")
            if len(df) == 0 or not all(pd.api.types.is_integer_dtype(df[c]) for c in df.columns):
                errs.append("empty or non-integer")
            elif df.node_a.duplicated().any() or df.node_b.duplicated().any():
                errs.append("duplicates")
            elif not (set(df.node_a) <= na and set(df.node_b) <= nb):
                errs.append("invalid node")
        if errs:
            print("FAIL", n, errs)
        else:
            ok += 1
    print("%d / %d prediction files valid" % (ok, len(names)))
    return 0 if ok == len(names) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "test", "check"])
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    if a.mode == "check":
        sys.exit(check())
    cfg = json.load(open(os.path.join(HERE, "config.json")))["method"]
    run_val(cfg, a.jobs) if a.mode == "val" else run_test(cfg, a.jobs)


if __name__ == "__main__":
    main()
