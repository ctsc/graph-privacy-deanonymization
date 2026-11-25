"""P3 figures from results/*.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED_COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#009E73"}


def roc(out):
    rocs = json.load(open(os.path.join(out, "roc_curves.json")))
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for ax, (mode, name) in zip(axes.ravel(), [(m, a) for m in ("fixed", "early") for a in ("loss_threshold", "shadow_logistic")]):
        for r in rocs:
            if r["mode"] == mode and r["attack"] == name:
                ax.plot(r["fpr"], r["tpr"], color=SEED_COLORS[r["seed"]], label=f"seed {r['seed']}")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.axvline(0.05, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{mode}: {name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Target attack-Test ROC (100 members vs 100 nonmembers)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc_curves.png"), dpi=150)
    plt.close(fig)


def learning(out):
    lc = json.load(open(os.path.join(out, "learning_curves.json")))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, mode in enumerate(("fixed", "early")):
        for h in lc:
            if h["mode"] != mode:
                continue
            c = SEED_COLORS[h["seed"]]
            ep = range(1, len(h["train_loss"]) + 1)
            axes[0, col].plot(ep, h["train_loss"], color=c, label=f"seed {h['seed']} train")
            axes[0, col].plot(ep, h["val_loss"], color=c, ls="--", label=f"seed {h['seed']} val")
            axes[1, col].plot(ep, h["train_acc"], color=c)
            axes[1, col].plot(ep, h["val_acc"], color=c, ls="--")
        axes[0, col].set_title(f"target model, {mode} (solid train, dashed val)")
        axes[0, col].set_ylabel("cross-entropy loss")
        axes[1, col].set_ylabel("accuracy")
        axes[1, col].set_xlabel("epoch")
        axes[0, col].set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learning_curves.png"), dpi=150)
    plt.close(fig)


def make_all(out):
    roc(out)
    learning(out)
