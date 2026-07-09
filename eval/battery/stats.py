"""Statistics for the v2 eval battery.

Ported verbatim (up to type hints) from the campaign session artifacts in
eval/session_artifacts/ — rank-based AUROC, episode-bootstrap CIs, paired
bootstrap deltas, TPR@FPR, and the ridge IRLS logistic. Numerics must stay
identical: the §6 expected numbers in expected.json depend on them.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC with tie handling."""
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = (csum - cnt + csum + 1) / 2.0
    ranks = avg_rank[inv]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def boot_ci(
    scores: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    """95% bootstrap CI on AUROC (run_deception_eval.py protocol)."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(auroc(s[b], y[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired(
    sa: np.ndarray,
    sb: np.ndarray,
    y: np.ndarray,
    n: int = 3000,
    seed: int = 1,
) -> Tuple[float, Tuple[float, float]]:
    """Paired bootstrap of AUROC(sa) - AUROC(sb) (loto_analysis.py protocol;
    curvature_kill_test.py uses n=4000, seed=3 — pass those explicitly)."""
    rng = np.random.default_rng(seed)
    sa = np.asarray(sa, float)
    sb = np.asarray(sb, float)
    y = np.asarray(y, int)
    idx = np.arange(len(y))
    dd = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        dd.append(auroc(sa[b], y[b]) - auroc(sb[b], y[b]))
    arr = np.array(dd)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(arr.mean()), (float(lo), float(hi))


def tpr_at_fpr(scores: np.ndarray, y: np.ndarray, f: float = 0.2) -> float:
    """TPR at a fixed FPR budget (curvature_kill_test.py protocol)."""
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    neg = np.sort(s[y == 0])[::-1]
    k = max(int(np.floor(f * len(neg))) - 1, 0)
    thr = neg[k] if len(neg) else np.inf
    return float((s[y == 1] > thr).mean())


def fit_logistic(
    X: np.ndarray, y: np.ndarray, ridge: float = 1e-2, iters: int = 60
) -> Tuple[np.ndarray, float]:
    """Ridge IRLS logistic; returns (weights, bias)."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xa @ w)))
        W = p * (1 - p) + 1e-9
        H = (Xa.T * W) @ Xa + ridge * np.eye(Xa.shape[1])
        w = w + np.linalg.solve(H, Xa.T @ (y - p) - ridge * w)
    return w[:-1], float(w[-1])
