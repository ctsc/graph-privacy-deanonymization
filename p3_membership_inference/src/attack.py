"""Attack features, attacks and metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def features(P, y):
    """P: (n,10) predicted probabilities, y: true task labels. Returns (feature matrix, loss)."""
    pt = np.clip(P[np.arange(len(y)), y], EPS, 1.0)
    loss = -np.log(pt)
    ent = -(P * np.log(np.clip(P, EPS, 1.0))).sum(1)
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    onehot = np.eye(10)[y]
    return np.column_stack([pt, loss, ent, conf, correct, onehot]), loss


def tpr_at_fpr(labels, scores, max_fpr=0.05):
    """Largest empirical TPR over thresholds with FPR <= max_fpr (predict member if score >= t)."""
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max())


def choose_threshold(labels, scores, max_fpr=0.05):
    """Threshold with largest TPR subject to FPR <= max_fpr; ties broken by the higher threshold."""
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    ok = np.where(fpr <= max_fpr + 1e-12)[0]
    k = ok[np.argmax(tpr[ok])]
    return float(thr[k]) if np.isfinite(thr[k]) else float(np.max(scores) + 1.0)


def rates_at(labels, scores, t):
    pred = scores >= t
    labels = np.asarray(labels).astype(bool)
    return float(pred[labels].mean()), float(pred[~labels].mean())


def evaluate(labels, scores, t):
    tpr, fpr = rates_at(labels, scores, t)
    return {"auc": float(roc_auc_score(labels, scores)), "tpr_at_5fpr": tpr_at_fpr(labels, scores),
            "tpr_at_threshold": tpr, "fpr_at_threshold": fpr, "threshold": t}


class LossAttack:
    name = "loss_threshold"

    def fit(self, F, loss, labels):  # nothing to learn from dev shadows
        return self

    def score(self, F, loss):
        return -loss


class LogisticAttack:
    name = "shadow_logistic"

    def __init__(self, C, seed):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed))

    def fit(self, F, loss, labels):
        self.m.fit(F, labels)
        return self

    def score(self, F, loss):
        return self.m.decision_function(F)
