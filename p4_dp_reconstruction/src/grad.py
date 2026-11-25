"""Softmax-regression gradients and the gradient-matching objective with analytic derivatives.

theta: 650 = row-major W (10, 64) then b (10). Per-example loss: cross-entropy of softmax(W x + b).
Per-example gradient: r xt^T with r = p - e_y, xt = [x, 1]. Clipped to norm C (if C is finite),
then averaged over the batch. Observed update: g_t = (theta_t - theta_{t+1}) / lr, per step."""
import numpy as np


def unpack(theta):
    return theta[..., :640].reshape(theta.shape[:-1] + (10, 64)), theta[..., 640:]


def observed_grads(states, lr):
    """states (11, 650) -> per-step g_t (10, 650). Never the endpoint difference."""
    return (states[:-1] - states[1:]) / lr


def _forward(W, b, X):
    """W (T,10,64), b (T,10), X (B,64) -> p (T,B,10)."""
    z = np.einsum("tkd,bd->tbk", W, X) + b[:, None, :]
    z -= z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def batch_grad(W, b, X, y, C):
    """Clipped averaged gradient for every step: (T, 10, 65) as [dW | db]."""
    p = _forward(W, b, X)
    r = p.copy()
    r[:, np.arange(len(y)), y] -= 1.0
    s = np.sqrt((X ** 2).sum(1) + 1.0)
    n = np.linalg.norm(r, axis=-1) * s[None, :]
    c = np.minimum(1.0, C / np.maximum(n, 1e-12)) if np.isfinite(C) else np.ones_like(n)
    Xt = np.c_[X, np.ones(len(X))]
    return np.einsum("tb,tbk,bd->tkd", c, r, Xt) / len(y)


def objective(X, y, W, b, G, C):
    """0.5 * sum_t ||batch_grad_t(X) - G_t||^2 and its gradient wrt X (B, 64).
    G: (T, 10, 65) observed gradients."""
    T, Bn = W.shape[0], len(y)
    p = _forward(W, b, X)
    r = p.copy()
    r[:, np.arange(Bn), y] -= 1.0
    sx2 = (X ** 2).sum(1)
    s = np.sqrt(sx2 + 1.0)
    rn = np.linalg.norm(r, axis=-1)
    n = rn * s[None, :]
    clipped = np.isfinite(C) & (n > C)
    c = np.where(clipped, C / np.maximum(n, 1e-12), 1.0)
    Xt = np.c_[X, np.ones(Bn)]
    D = np.einsum("tb,tbk,bd->tkd", c, r, Xt) / Bn - G
    f = 0.5 * (D ** 2).sum()
    Dp = D / Bn
    Dx = np.einsum("tkd,bd->tbk", Dp, Xt)
    q = (r * Dx).sum(-1)
    gx = np.einsum("tb,tkd,tbk->bd", c, Dp, r)[:, :64]

    def jvec(v):  # J^T v for softmax Jacobian, J symmetric: p*v - p (p.v)
        return p * v - p * (p * v).sum(-1, keepdims=True)

    gx += np.einsum("tkd,tbk->bd", W, jvec(c[..., None] * Dx))
    dcdn = np.where(clipped, -C / np.maximum(n, 1e-12) ** 2, 0.0)
    w = q * dcdn
    gx += np.einsum("tkd,tbk->bd", W, jvec((w * s[None, :] / np.maximum(rn, 1e-12))[..., None] * r))
    gx += (w * rn / s[None, :]).sum(0)[:, None] * X
    return f, gx
