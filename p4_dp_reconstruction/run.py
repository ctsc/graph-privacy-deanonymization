"""P4 ten-step DP reconstruction, single entry point.

Usage (from the privacy folder):
  conda run -n privacy_projects python p4_dp_reconstruction/run.py val      # public validation, all scenarios
  conda run -n privacy_projects python p4_dp_reconstruction/run.py test     # predict observations/ (frozen config)
  conda run -n privacy_projects python p4_dp_reconstruction/run.py check    # validate output NPZ files
Every trial is attacked independently; nothing is shared across trials, scenarios or seeds."""
import argparse
import glob
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

from attack import Prior, reconstruct  # noqa: E402
from evaluate import per_image_mse, psnr  # noqa: E402
from grad import unpack  # noqa: E402

DATA = os.path.join(HERE, "..", "student_projects", "04_dp_reconstruction")
RESULTS = os.path.join(HERE, "results")
OUTPUTS = os.path.join(HERE, "outputs")
ATTACKS = ("plain", "prior", "mean")


def load_cfg():
    return json.load(open(os.path.join(HERE, "config.json")))["attack"]


def public():
    return dict(np.load(os.path.join(DATA, "public.npz")))


def scenario_info(v):
    return dict(lr=float(v["lr"]), clip=float(v["clip"]), sigma=float(v["sigma_per_step"]))


def attack_file(path, attack, cfg, prior, seed, track=False):
    """Reconstruct every trial in one NPZ. Returns images (trials, B, 64), objectives, histories."""
    v = np.load(path)
    info = scenario_info(v)
    out, objs, hists = [], [], []
    for k in range(v["states"].shape[0]):
        rng = np.random.default_rng([seed, k])
        x, f, h = reconstruct(v["states"][k], v["labels"][k], info["lr"], info["clip"], info["sigma"],
                              attack, prior, cfg, rng, track)
        out.append(x)
        objs.append(f)
        hists.append(h)
    return np.stack(out), np.array(objs), hists


def utility(states, pub):
    """Accuracy of theta_t on the 200 public calibration images, t = 0..10."""
    W, b = unpack(states)
    z = np.einsum("tkd,nd->tnk", W, pub["calibration_images"]) + b[:, None, :]
    return (z.argmax(-1) == pub["calibration_labels"][None, :]).mean(1)


def _val_one(path, attack, cfg, pub, seed):
    prior = Prior(pub, cfg["var_floor"])
    v = np.load(path)
    X, objs, hists = attack_file(path, attack, cfg, prior, seed, track=(attack != "mean"))
    name = os.path.basename(path)[:-4]
    B, scen = name.split("_", 1)
    rows = []
    for k in range(X.shape[0]):
        mse = per_image_mse(X[k], v["originals"][k], v["labels"][k])
        for j, m in enumerate(mse):
            rows.append({"file": name, "batch": int(B[5:]), "scenario": scen, "attack": attack, "trial": k,
                         "slot": j, "mse": m, "psnr": psnr(m), "objective": objs[k]})
    return rows, {"file": name, "attack": attack, "images": X, "hist": hists[0] if hists else []}


def run_val(cfg, jobs, seed=42):
    os.makedirs(RESULTS, exist_ok=True)
    pub = public()
    files = sorted(glob.glob(os.path.join(DATA, "validation", "*.npz")))
    out = Parallel(n_jobs=jobs)(delayed(_val_one)(f, a, cfg, pub, seed) for f in files for a in ATTACKS)
    df = pd.DataFrame([r for rs, _ in out for r in rs])
    df.to_csv(os.path.join(RESULTS, "val_per_image.csv"), index=False)
    s = df.groupby(["batch", "scenario", "attack"]).agg(mse=("mse", "mean"), mse_std=("mse", "std"),
                                                        psnr=("psnr", "mean"), images=("mse", "size")).reset_index()
    base = s[s.attack == "mean"].set_index(["batch", "scenario"]).mse
    s["mse_over_mean_baseline"] = [m / base[(b, sc)] for b, sc, m in zip(s.batch, s.scenario, s.mse)]
    tr = df.groupby(["batch", "scenario", "attack", "trial"]).mse.mean().unstack("attack")
    beat = (tr["prior"] < tr["mean"]).groupby(["batch", "scenario"]).mean().rename("prior_beats_mean_trial_frac")
    s = s.merge(beat.reset_index(), on=["batch", "scenario"])
    s.to_csv(os.path.join(RESULTS, "val_summary.csv"), index=False)
    util = []
    for f in files:
        v = np.load(f)
        acc = np.mean([utility(v["states"][k], pub) for k in range(v["states"].shape[0])], 0)
        util.append({"file": os.path.basename(f)[:-4], **{f"acc_step{t}": a for t, a in enumerate(acc)}})
    pd.DataFrame(util).to_csv(os.path.join(RESULTS, "val_utility_by_step.csv"), index=False)
    import figures
    figures.make_all(out, files, RESULTS)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))


def run_test(cfg, jobs, seed=42):
    pub = public()
    files = sorted(glob.glob(os.path.join(DATA, "observations", "seed*", "*.npz")))

    def one(path):
        prior = Prior(pub, cfg["var_floor"])
        X, objs, _ = attack_file(path, cfg["submitted_attack"], cfg, prior, seed)
        rel = os.path.relpath(path, os.path.join(DATA, "observations"))
        dst = os.path.join(OUTPUTS, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez(dst, images=np.clip(X, 0.0, 1.0))
        return rel, X.shape, float(np.nanmean(objs))
    for rel, shp, f in Parallel(n_jobs=jobs)(delayed(one)(p) for p in files):
        print(rel, shp, "mean objective %.3f" % f)


def check():
    files = sorted(glob.glob(os.path.join(DATA, "observations", "seed*", "*.npz")))
    ok = 0
    for path in files:
        rel = os.path.relpath(path, os.path.join(DATA, "observations"))
        v = np.load(path)
        errs = []
        dst = os.path.join(OUTPUTS, rel)
        if not os.path.exists(dst):
            errs.append("missing")
        else:
            o = np.load(dst)
            T, Bn = v["labels"].shape
            if o.files != ["images"]:
                errs.append("keys %s" % o.files)
            x = o["images"]
            if x.shape != (T, Bn, 64):
                errs.append("shape %s" % (x.shape,))
            if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                errs.append("range")
        print(("OK  " if not errs else "FAIL"), rel, errs or "")
        ok += not errs
    print("%d / %d output files valid" % (ok, len(files)))
    return 0 if ok == len(files) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "test", "check"])
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    if a.mode == "check":
        sys.exit(check())
    cfg = load_cfg()
    run_val(cfg, a.jobs) if a.mode == "val" else run_test(cfg, a.jobs)


if __name__ == "__main__":
    main()
