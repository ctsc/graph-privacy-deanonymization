"""MLP training with a manual partial_fit epoch loop: fixed epochs or val-loss early stopping."""
import copy
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CLASSES = np.arange(10)


class Model:
    """Scaler fit on the model's own train members only, plus the MLP."""

    def __init__(self, scaler, clf):
        self.scaler, self.clf = scaler, clf

    def proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))


def train(X, y, tr, va, mode, cfg, seed):
    """mode 'fixed': cfg fixed_epochs epochs. mode 'early': stop on val loss with patience,
    restore best. Returns (Model, history dict of per-epoch losses and accuracies, best epoch)."""
    m = cfg["mlp"]
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    clf = MLPClassifier(hidden_layer_sizes=tuple(m["hidden_layer_sizes"]), solver=m["solver"],
                        learning_rate_init=m["learning_rate_init"], alpha=m["alpha"],
                        batch_size=m["batch_size"], activation=m["activation"], random_state=seed)
    es = cfg["early_stopping"]
    max_ep = cfg["fixed_epochs"] if mode == "fixed" else es["max_epochs"]
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best, best_ep, best_clf, wait = np.inf, 0, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for ep in range(1, max_ep + 1):
            clf.partial_fit(Xtr, y[tr], classes=CLASSES)
            ptr, pva = clf.predict_proba(Xtr), clf.predict_proba(Xva)
            hist["train_loss"].append(log_loss(y[tr], ptr, labels=CLASSES))
            hist["val_loss"].append(log_loss(y[va], pva, labels=CLASSES))
            hist["train_acc"].append(float((ptr.argmax(1) == y[tr]).mean()))
            hist["val_acc"].append(float((pva.argmax(1) == y[va]).mean()))
            if mode == "early":
                if hist["val_loss"][-1] < best:
                    best, best_ep, best_clf, wait = hist["val_loss"][-1], ep, copy.deepcopy(clf), 0
                else:
                    wait += 1
                    if wait >= es["patience"]:
                        break
    if mode == "early" and es["restore_best"]:
        clf = best_clf
    else:
        best_ep = len(hist["train_loss"])
    return Model(scaler, clf), hist, best_ep
