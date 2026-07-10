"""Q2 (post-v2): how many in-domain labels does calibration actually need?

Q1 made raw-384-D step-mean the detection stack and Q3 made in-domain
calibration fundamental — so the binding deployment input is labeled
in-domain episodes. This measures AUROC vs label budget for the candidate
calibrators, including the one regime where the demoted 4-D chart could
still earn its keep: tiny budgets, where a 4x4 estimated metric may be
stabler than a 384-D prototype direction.

Methods (all supervised, label-oriented, no train-sign step):
  raw_stepmean     prototype direction on step-mean raw embeddings
  raw_single       prototype direction on full-text raw embeddings
  chart_stepmean   centroid margin on chart step-means under estimated
                   Sigma_w^-1 (ridge 1e-3, campaign protocol)
  chart_stepmean_id  same centroids under the identity metric (isolates
                   metric-estimation cost from chart compression)

Protocol: per task (matrix_{task}.npz caches from step_mean_raw_matrix),
5-fold CV; within each fold, R=10 stratified subsamples of the train pool
per budget (>=2 per class), scored on the held-out fold. Budgets
8..256 + full pool. Reported: mean AUROC over 5 folds x 10 draws, and the
smallest budget within 0.02 AUROC of the full-pool value.

Run: python -m eval.label_budget_curve  (requires the Q1 caches)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.battery.data import TASKS
from eval.battery.stats import auroc
from eval.step_mean_raw_matrix import encode_task

RESULTS = Path(__file__).resolve().parent / "results"
BUDGETS = [8, 16, 32, 64, 128, 256]
R = 10
FOLD_SEED = 11
METHODS = ["raw_stepmean", "raw_single", "chart_stepmean", "chart_stepmean_id"]


def proto(E: np.ndarray, Y: np.ndarray) -> np.ndarray:
    d = E[Y == 1].mean(0) - E[Y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def est_metric(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    d = X.shape[1]
    Sw = np.zeros((d, d))
    for c in (0, 1):
        Z = X[y == c] - X[y == c].mean(0)
        Sw += Z.T @ Z
    M = np.linalg.inv(Sw / max(len(X) - 2, 1) + ridge * np.eye(d))
    return M * d / np.trace(M)


def margin(X: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, M: np.ndarray) -> np.ndarray:
    d0 = np.einsum("ni,ij,nj->n", X - mu0, M, X - mu0)
    d1 = np.einsum("ni,ij,nj->n", X - mu1, M, X - mu1)
    return d0 - d1


def score_method(method: str, data: Dict, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    y = data["Y"][tr]
    if method == "raw_stepmean":
        return data["raw_mean"][te] @ proto(data["raw_mean"][tr], y)
    if method == "raw_single":
        return data["raw_full"][te] @ proto(data["raw_full"][tr], y)
    mu0 = data["chart_mean"][tr][y == 0].mean(0)
    mu1 = data["chart_mean"][tr][y == 1].mean(0)
    if method == "chart_stepmean":
        M = est_metric(data["chart_full"][tr], y)
    else:  # chart_stepmean_id
        M = np.eye(data["chart_mean"].shape[1])
    return margin(data["chart_mean"][te], mu0, mu1, M)


def stratified_draw(
    pool: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """n indices from pool, proportional by class, >=2 per class."""
    by_class = {c: pool[y[pool] == c] for c in (0, 1)}
    frac1 = len(by_class[1]) / len(pool)
    n1 = int(np.clip(round(n * frac1), 2, n - 2))
    take = {1: n1, 0: n - n1}
    idx = np.concatenate(
        [rng.choice(by_class[c], take[c], replace=False) for c in (0, 1)]
    )
    return idx


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    report: Dict[str, Dict] = {}
    for task in TASKS:
        data = encode_task(task)
        Y = data["Y"]
        n = len(Y)
        if n < 25 or Y.min() == Y.max():
            continue
        rng_folds = np.random.default_rng(FOLD_SEED)
        folds = np.array_split(rng_folds.permutation(n), 5)
        pool_size = min(len(np.concatenate(folds[1:])), n - len(folds[0]))
        budgets = [b for b in BUDGETS if b <= pool_size] + ["full"]

        curves: Dict[str, Dict] = {m: {} for m in METHODS}
        for bi, budget in enumerate(budgets):
            samples: Dict[str, List[float]] = {m: [] for m in METHODS}
            for f in range(5):
                te = folds[f]
                pool = np.concatenate([folds[j] for j in range(5) if j != f])
                reps = 1 if budget == "full" else R
                for r in range(reps):
                    if budget == "full":
                        tr = pool
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([FOLD_SEED, bi, f, r])
                        )
                        min_class = min((Y[pool] == 0).sum(), (Y[pool] == 1).sum())
                        if min_class < 2:
                            continue
                        tr = stratified_draw(pool, Y, budget, rng)
                    if Y[tr].min() == Y[tr].max():
                        continue
                    for m in METHODS:
                        samples[m].append(auroc(score_method(m, data, tr, te), Y[te]))
            for m in METHODS:
                v = np.array(samples[m])
                curves[m][str(budget)] = dict(
                    mean=float(v.mean()), std=float(v.std()), n_est=len(v)
                )

        # smallest budget within 0.02 of the full-pool AUROC
        summary = {}
        for m in METHODS:
            full = curves[m]["full"]["mean"]
            enough = [
                b for b in budgets[:-1]
                if curves[m][str(b)]["mean"] >= full - 0.02
            ]
            summary[m] = dict(
                full_auroc=full,
                budget_within_002=(enough[0] if enough else None),
            )

        report[task] = dict(
            n_episodes=int(n), budgets=budgets, curves=curves, summary=summary
        )
        print(f"== {task} (n={n}) ==")
        hdr = "  budget " + "".join(f"{m:>20s}" for m in METHODS)
        print(hdr)
        for budget in budgets:
            row = f"  {str(budget):>6s} "
            for m in METHODS:
                c = curves[m][str(budget)]
                row += f"      {c['mean']:.3f}±{c['std']:.3f}"
            print(row)
        for m in METHODS:
            print(
                f"  within-0.02-of-full: {m:18s} -> {summary[m]['budget_within_002']}"
            )

    out = RESULTS / "label-budget-2026-07-10.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
