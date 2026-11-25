"""P4 validation figures: MSE by scenario, example reconstructions, convergence curves."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluate import align  # noqa: E402

SCEN = ["non_private", "clip_only", "eps_16", "eps_4", "eps_1"]
COL = {"plain": "#0072B2", "prior": "#D55E00", "mean": "#7F7F7F"}


def mse_bars(out):
    s = pd.read_csv(os.path.join(out, "val_summary.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, B in zip(axes, (1, 2, 4)):
        d = s[s.batch == B].pivot_table(index="scenario", columns="attack", values="mse").loc[SCEN]
        x = np.arange(len(SCEN))
        for k, a in enumerate(("plain", "prior", "mean")):
            ax.bar(x + (k - 1) * 0.27, d[a], 0.27, color=COL[a], label=a)
        ax.set_xticks(x, SCEN, rotation=20)
        ax.set_title(f"batch {B}")
        ax.set_yscale("log")
    axes[0].set_ylabel("mean per-image MSE (log scale)")
    axes[0].legend()
    fig.suptitle("Public validation: reconstruction MSE (lower is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "val_mse_by_scenario.png"), dpi=150)
    plt.close(fig)


def examples(results, files, out):
    byk = {(r[1]["file"], r[1]["attack"]): r[1]["images"] for r in results}
    show = ["batch1_non_private", "batch1_eps_16", "batch1_eps_4", "batch2_clip_only", "batch2_eps_4", "batch4_eps_4"]
    rows = []
    for name in show:
        path = [f for f in files if os.path.basename(f)[:-4] == name][0]
        v = np.load(path)
        o, lab = v["originals"][0], v["labels"][0]
        for j in range(len(lab)):
            rows.append((name, j, o[j], [align(byk[(name, a)][0], o, lab)[j] for a in ("plain", "prior", "mean")]))
    fig, axes = plt.subplots(len(rows), 4, figsize=(5.2, 1.3 * len(rows)))
    for i, (name, j, o, recs) in enumerate(rows):
        for c, img in enumerate([o] + recs):
            ax = axes[i, c]
            ax.imshow(img.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(["original", "plain", "prior", "public mean"][c], fontsize=8)
        axes[i, 0].set_ylabel(f"{name}\nslot {j}", fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("Validation trial 0 (public originals)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "val_examples.png"), dpi=150)
    plt.close(fig)


def convergence(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    for i, a in enumerate(("plain", "prior")):
        for j, dp in enumerate((False, True)):
            ax = axes[i, j]
            k = 0
            for r in results:
                h = r[1]
                if h["attack"] == a and len(h["hist"]) and (("eps" in h["file"]) == dp):
                    ax.plot(np.maximum(h["hist"], 1e-12), lw=1, color=cmap(k % 10), label=h["file"])
                    k += 1
            ax.set_yscale("log")
            ax.set_xlabel("Adam iteration")
            ax.set_ylabel("objective (absolute)")
            ax.set_title(f"{a}, {'DP files' if dp else 'sigma = 0 files'}: trial 0, chosen restart")
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "val_convergence.png"), dpi=150)
    plt.close(fig)


def make_all(results, files, out):
    mse_bars(out)
    examples(results, files, out)
    convergence(results, out)
