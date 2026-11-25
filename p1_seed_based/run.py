"""P1 seed-based graph matching, single entry point.

Usage (from the privacy folder):
  conda run -n privacy_projects python p1_seed_based/run.py val     # Val metrics, tables, figures
  conda run -n privacy_projects python p1_seed_based/run.py test    # Test predictions (108 files)
  conda run -n privacy_projects python p1_seed_based/run.py check   # validity check of predictions

The method settings are read from config.json (frozen). Each seed config uses only its own
seeds file; nothing is shared between configs."""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from baseline import baseline_match  # noqa: E402
from data import SEED_CONFIGS, list_instances, load_instance  # noqa: E402
from match import match  # noqa: E402
from metrics import instance_metrics  # noqa: E402

RESULTS = os.path.join(HERE, "results")
PRED_DIR = os.path.join(HERE, "predictions")


def load_config():
    with open(os.path.join(HERE, "config.json")) as fh:
        return json.load(fh)["method"]


def _row(inst, sc, method, pred, runtime):
    m = inst["meta"]
    strat, frac = sc.split("_")
    r = {"method": method, "seed_config": sc, "strategy": strat, "fraction_pct": int(frac),
         "instance": inst["name"], "family": m["family"], "difficulty": m["difficulty"],
         "n": inst["n_a"], "n_seeds": len(inst["seeds"]), "runtime_s": runtime}
    r.update(instance_metrics(pred, inst["truth"], inst["seeds"], inst["n_a"]))
    return r


def _val_one(name, sc, cfg):
    inst = load_instance("val", name, sc)
    t = time.perf_counter()
    pred, info = match(inst, cfg, return_info=True)
    rt = time.perf_counter() - t
    truth = inst["truth"]
    pairs = [{"seed_config": sc, "instance": name, "node_a": a, "kind": k, "score": s,
              "correct": int(truth.get(a) == pred[a])} for a, (k, s) in info.items()]
    return _row(inst, sc, "seed_percolation", pred, rt), pairs


def _baseline_one(name):
    inst = load_instance("val", name, "random_05")
    t = time.perf_counter()
    pred = baseline_match(inst)
    rt = time.perf_counter() - t
    inst["seeds"] = []  # baseline uses no seeds; unseeded recall equals recall
    r = _row(inst, "random_05", "baseline_structural", pred, rt)
    r.update(seed_config="none", strategy="none", fraction_pct=0, n_seeds=0)
    return r


def run_val(cfg, n_jobs):
    os.makedirs(RESULTS, exist_ok=True)
    names = list_instances("val")
    jobs = [(n, sc) for sc in SEED_CONFIGS for n in names]
    out = Parallel(n_jobs=n_jobs)(delayed(_val_one)(n, sc, cfg) for n, sc in jobs)
    rows = [r for r, _ in out]
    pairs = pd.DataFrame([p for _, ps in out for p in ps])
    rows += Parallel(n_jobs=n_jobs)(delayed(_baseline_one)(n) for n in names)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(RESULTS, "val_per_instance.csv"), index=False)
    pairs.to_csv(os.path.join(RESULTS, "val_pair_scores.csv"), index=False)

    mets = ["precision", "recall", "f1", "coverage", "unseeded_recall", "unseeded_precision", "runtime_s"]
    order = ["seed_config", "strategy", "fraction_pct"]
    macro = per.groupby(["method"] + order, sort=False)[mets].mean().reset_index()
    macro.to_csv(os.path.join(RESULTS, "val_macro_by_config.csv"), index=False)
    for g in ("family", "difficulty", "n"):
        t = per.groupby(["method", "seed_config", g], sort=False)[mets].mean().reset_index()
        t.to_csv(os.path.join(RESULTS, f"val_macro_by_config_{g}.csv"), index=False)
    strat = per[per.method == "seed_percolation"].groupby(["strategy", "fraction_pct"])[
        ["f1", "unseeded_recall"]].mean().unstack("strategy")
    strat.to_csv(os.path.join(RESULTS, "val_random_vs_degree.csv"))

    conf = confidence_table(pairs)
    conf.to_csv(os.path.join(RESULTS, "val_confidence_vs_precision.csv"), index=False)
    kinds = pairs.groupby(["seed_config", "kind"]).agg(pairs=("correct", "size"), precision=("correct", "mean")).reset_index()
    kinds.to_csv(os.path.join(RESULTS, "val_precision_by_pair_kind.csv"), index=False)

    import figures
    figures.make_all(per, conf, RESULTS)
    pd.set_option("display.width", 200)
    print(macro.round(4).to_string(index=False))
    print(kinds.round(4).to_string(index=False))
    print(conf.round(4).to_string(index=False))


def confidence_table(pairs, nbins=5):
    """Precision of non-seed predicted pairs by score quintile, pooled over Val instances per config."""
    rows = []
    ns = pairs[pairs.kind != "seed"].copy()
    for sc, g in ns.groupby("seed_config", sort=False):
        g = g.assign(bin=pd.qcut(g.score.rank(method="first"), nbins, labels=False))
        for b, h in g.groupby("bin"):
            rows.append({"seed_config": sc, "score_quintile": int(b) + 1, "score_min": h.score.min(),
                         "score_max": h.score.max(), "pairs": len(h), "precision": h.correct.mean()})
    return pd.DataFrame(rows)


def _test_one(name, sc, cfg):
    inst = load_instance("test", name, sc)
    assert "truth" not in inst
    t = time.perf_counter()
    pred = match(inst, cfg)
    rt = time.perf_counter() - t
    d = os.path.join(PRED_DIR, sc, name)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame(sorted(pred.items()), columns=["node_a", "node_b"])
    df.to_csv(os.path.join(d, "predictions.csv"), index=False)
    return {"seed_config": sc, "instance": name, "pairs": len(df), "runtime_s": rt}


def run_test(cfg, n_jobs):
    names = list_instances("test")
    jobs = [(n, sc) for sc in SEED_CONFIGS for n in names]
    log = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_test_one)(n, sc, cfg) for n, sc in jobs))
    os.makedirs(RESULTS, exist_ok=True)
    log.to_csv(os.path.join(RESULTS, "test_prediction_log.csv"), index=False)
    print("wrote %d prediction files" % len(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "test", "check"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "check":
        import check_predictions
        sys.exit(check_predictions.main(PRED_DIR))
    cfg = load_config()
    run_val(cfg, a.jobs) if a.mode == "val" else run_test(cfg, a.jobs)


if __name__ == "__main__":
    main()
