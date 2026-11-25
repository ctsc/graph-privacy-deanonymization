"""Val figures for P1, regenerated from the result tables."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"random": "#0072B2", "degree": "#D55E00", "none": "#7F7F7F"}


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def f1_by_config(per, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.35
    for k, s in enumerate(("random", "degree")):
        g = per[(per.method == "seed_percolation") & (per.strategy == s)].groupby("fraction_pct")
        mean, std = g.f1.mean(), g.f1.std(ddof=1)
        x = np.arange(len(mean)) + (k - 0.5) * w
        ax.bar(x, mean, w, yerr=std, capsize=3, color=COLORS[s], label=f"{s} seeds")
    base = per[per.method == "baseline_structural"].f1.mean()
    ax.axhline(base, color=COLORS["none"], ls="--", label="seed-free baseline (%.3f)" % base)
    ax.set_xticks(range(3), ["1%", "5%", "10%"])
    ax.set_xlabel("seed fraction")
    ax.set_ylabel("macro F1 over 9 Val instances")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Val F1 by seed config (error bar = sample std)")
    _save(fig, out, "val_f1_by_config.png")


def f1_heat(per, col, out):
    d = per[per.method == "seed_percolation"].pivot_table(index=col, columns="seed_config", values="f1")
    d = d[[c for c in ("random_01", "random_05", "random_10", "degree_01", "degree_05", "degree_10") if c in d]]
    if col == "difficulty":
        d = d.loc[[x for x in ("easy", "medium", "hard") if x in d.index]]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(d.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, "%.3f" % v, ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_xticks(range(d.shape[1]), d.columns, rotation=20)
    ax.set_yticks(range(d.shape[0]), d.index)
    ax.set_title(f"Val macro F1 by {col}")
    fig.colorbar(im, ax=ax)
    _save(fig, out, f"val_f1_by_{col}.png")


def runtime(per, out):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for meth, mk in (("seed_percolation", "o"), ("baseline_structural", "x")):
        g = per[per.method == meth].groupby("n").runtime_s.mean()
        ax.plot(g.index, g.values, marker=mk, label=meth)
    ax.set_xlabel("nodes per graph")
    ax.set_ylabel("mean runtime per instance (s)")
    ax.legend()
    ax.set_title("Val runtime")
    _save(fig, out, "val_runtime.png")


def confidence(conf, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for sc, g in conf.groupby("seed_config", sort=False):
        s = sc.split("_")[0]
        ax.plot(g.score_quintile, g.precision, marker="o", color=COLORS[s],
                ls="-" if sc.endswith("05") else ":" if sc.endswith("01") else "--", label=sc)
    ax.set_xlabel("score quintile of predicted non-seed pairs (1 = lowest)")
    ax.set_ylabel("precision")
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=7)
    ax.set_title("Val confidence vs precision (pooled per config)")
    _save(fig, out, "val_confidence_vs_precision.png")


def make_all(per, conf, out):
    f1_by_config(per, out)
    f1_heat(per, "family", out)
    f1_heat(per, "difficulty", out)
    runtime(per, out)
    confidence(conf, out)
