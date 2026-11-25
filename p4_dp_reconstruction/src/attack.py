"""Reconstruction attacks for one trial (one 11-state transcript). Trials are never combined.

plain : minimize sum_t 0.5 ||G_t(x) - g_t||^2 over x in [0,1] (sigmoid parameterization), Adam,
        random restarts; for B = 1 with sigma = 0 the weight/bias gradient ratio gives an exact init.
prior : same plus factor * sigma^2 * sum_i sum_d (x_id - mu_{y_i,d})^2 / (2 var_{y_i,d}), with class
        means / variances from the 800 public images; init at the class mean plus random restarts.
mean  : public class-mean image per known label, ignores all updates.
The true images are never used here."""
import numpy as np

from grad import objective, observed_grads, unpack

EPS = 1e-4


class Prior:
    def __init__(self, public, var_floor):
        X, y = public["images"], public["labels"]
        self.mu = np.stack([X[y == k].mean(0) for k in range(10)])
        self.var = np.stack([X[y == k].var(0) for k in range(10)]) + var_floor


def _logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def _sig(u):
    return 1.0 / (1.0 + np.exp(-u))


def exact_init_b1(G):
    """B = 1, no noise: dW[k] = c r_k x, db[k] = c r_k, so x = dW[k] / db[k] for the largest |db[k]|."""
    g = G[0]
    k = np.argmax(np.abs(g[:, 64]))
    return np.clip(g[k, :64] / g[k, 64], 0.0, 1.0)


def _optimize(u0, y, W, b, G, C, prior, lam, cfg, track=False):
    u = u0.copy()
    m, v = np.zeros_like(u), np.zeros_like(u)
    b1, b2, lr = 0.9, 0.999, cfg["lr"]
    hist = []
    best_f, best_u = np.inf, u.copy()
    for it in range(1, cfg["iters"] + 1):
        x = _sig(u)
        f, gx = objective(x, y, W, b, G, C)
        if lam > 0:
            d = (x - prior.mu[y]) / prior.var[y]
            f += lam * 0.5 * ((x - prior.mu[y]) * d).sum()
            gx = gx + lam * d
        if f < best_f:
            best_f, best_u = f, u.copy()
        if track:
            hist.append(f)
        gu = gx * x * (1 - x)
        m = b1 * m + (1 - b1) * gu
        v = b2 * v + (1 - b2) * gu ** 2
        u -= lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + 1e-12)
    return best_f, _sig(best_u), hist


def reconstruct(states, labels, lr, clip, sigma, attack, prior, cfg, rng, track=False):
    """Returns (images (B,64), final objective, convergence history of the chosen restart)."""
    y = np.asarray(labels, int)
    if attack == "mean":
        return prior.mu[y].copy(), np.nan, []
    C = float(clip) if np.isfinite(clip) else np.inf
    W, b = unpack(states[:-1])
    g = observed_grads(states, lr)
    G = np.concatenate([g[:, :640].reshape(-1, 10, 64), g[:, 640:, None]], axis=2)
    Bn = len(y)
    inits = []
    if Bn == 1 and sigma == 0:
        inits.append(_logit(exact_init_b1(G))[None, :])
    if attack == "plain":
        lam = 0.0
    else:
        lam = cfg["prior_factor"] * sigma ** 2 if sigma > 0 else cfg["prior_factor"] * cfg["sigma0_prior"]
        inits.append(_logit(prior.mu[y]))
    while len(inits) < cfg["restarts"]:
        inits.append(rng.normal(0.0, 1.0, size=(Bn, 64)))
    best = None
    for u0 in inits:
        f, x, h = _optimize(u0, y, W, b, G, C, prior, lam, cfg, track)
        if best is None or f < best[0]:
            best = (f, x, h)
    return best[1], best[0], best[2]
