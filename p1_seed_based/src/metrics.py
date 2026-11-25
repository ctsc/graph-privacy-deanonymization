"""Per-instance metrics, macro averaged across instances."""
import numpy as np


def instance_metrics(pred, truth, seeds, n_a):
    """pred: dict a->b. truth: dict a->b. seeds: list of (a,b)."""
    seed_a = {a for a, _ in seeds}
    correct = sum(1 for a, b in pred.items() if truth.get(a) == b)
    npred = len(pred)
    prec = correct / npred if npred else 0.0
    rec = correct / len(truth)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    cov = npred / n_a
    # unseeded: restrict to nodes that are not seeds
    un_truth = {a: b for a, b in truth.items() if a not in seed_a}
    un_pred = {a: b for a, b in pred.items() if a not in seed_a}
    un_correct = sum(1 for a, b in un_pred.items() if un_truth.get(a) == b)
    un_rec = un_correct / len(un_truth) if un_truth else 0.0
    un_prec = un_correct / len(un_pred) if un_pred else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "coverage": cov,
            "unseeded_recall": un_rec, "unseeded_precision": un_prec}


def evaluate_rejection(pred, conf, inst, taus):
    """Metrics if final-assignment pairs with confidence < tau were dropped. Seeds and
    pseudo seeds (not in conf) are always kept."""
    out = []
    for tau in taus:
        p = {a: b for a, b in pred.items() if a not in conf or conf[a] >= tau}
        out.append(instance_metrics(p, inst["truth"], inst["seeds"], inst["n_a"]))
    return out


def check_valid(pred, n_a, n_b):
    bs = list(pred.values())
    assert len(set(bs)) == len(bs), "duplicate node_b"
    assert all(0 <= a < n_a for a in pred) and all(0 <= b < n_b for b in bs), "invalid node"
