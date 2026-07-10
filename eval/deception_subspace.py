"""Q3 (post-v2): do the four task deception directions share a subspace?

The external-transfer check showed task prototype directions in raw 384-D
are near-orthogonal (mean off-diag |cos| 0.071) and transfer poorly
(0.33-0.57 vs 0.64-0.97 in-task). This asks the follow-up that decides
whether "in-domain only" is fundamental or fixable: is each held-out task's
direction largely contained in the span of the other three (a shared
low-rank "deception subspace"), or is it genuinely novel?

Measurements:
  1. Singular values / participation ratio of the stacked directions.
  2. LOTO containment: R^2 of each held-out direction projected onto the
     span of the other three, vs the random-direction null E[R^2] = 3/384,
     with episode-bootstrap CIs (B=300).
  3. Practical ceiling: held-out AUROC of the best subspace direction
     (least-squares projection of the held-out prototype onto the span —
     an oracle upper bound for zero-shot subspace transfer).

Uses the battery's cached raw encodings (run `python -m eval.battery` first).

Run: python -m eval.deception_subspace
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from eval.battery.data import TASKS, cache_dir
from eval.battery.stats import auroc

RESULTS = Path(__file__).resolve().parent / "results"
B = 300


def proto(E: np.ndarray, Y: np.ndarray) -> np.ndarray:
    d = E[Y == 1].mean(0) - E[Y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = {}
    for t in TASKS:
        z = np.load(cache_dir() / f"raw_{t}.npz")
        data[t] = (z["E"], z["Y"])
    W = {t: proto(*data[t]) for t in TASKS}
    dim = len(next(iter(W.values())))

    # 1. Spectrum of the stacked directions
    M = np.stack([W[t] for t in TASKS])
    sv = np.linalg.svd(M, compute_uv=False)
    sv_norm = sv**2 / (sv**2).sum()
    participation = float(1.0 / (sv_norm**2).sum())
    print("singular value energy fractions:", np.round(sv_norm, 3))
    print(f"participation ratio (effective rank): {participation:.2f} / 4")

    # 2 + 3. LOTO containment and oracle subspace transfer
    rng = np.random.default_rng(17)
    null_r2 = (len(TASKS) - 1) / dim
    loto: Dict[str, Dict] = {}
    print(f"\nLOTO containment (null E[R^2] = {null_r2:.4f}):")
    for hold in TASKS:
        others = [t for t in TASKS if t != hold]
        Q, _ = np.linalg.qr(np.stack([W[t] for t in others]).T)  # (384, 3)
        w_h = W[hold]
        r2 = float(np.linalg.norm(Q.T @ w_h) ** 2)

        # episode-bootstrap CI on R^2 (resample every task's episodes)
        r2_boot = []
        for _ in range(B):
            Wb = {}
            ok = True
            for t in TASKS:
                E, Y = data[t]
                b = rng.choice(len(Y), len(Y), replace=True)
                if Y[b].min() == Y[b].max():
                    ok = False
                    break
                Wb[t] = proto(E[b], Y[b])
            if not ok:
                continue
            Qb, _ = np.linalg.qr(np.stack([Wb[t] for t in others]).T)
            r2_boot.append(float(np.linalg.norm(Qb.T @ Wb[hold]) ** 2))
        lo, hi = np.percentile(r2_boot, [2.5, 97.5])

        # oracle: best direction inside the train span, evaluated held-out
        E_h, Y_h = data[hold]
        w_oracle = Q @ (Q.T @ w_h)
        norm = np.linalg.norm(w_oracle)
        a_oracle = auroc(E_h @ (w_oracle / norm), Y_h) if norm > 1e-9 else float("nan")
        a_intask_full = auroc(E_h @ w_h, Y_h)

        loto[hold] = dict(
            r2=r2,
            r2_ci=[float(lo), float(hi)],
            oracle_subspace_auroc=a_oracle,
            intask_fullfit_auroc=a_intask_full,
        )
        print(
            f"  {hold:16s} R^2={r2:.3f} CI [{lo:.3f},{hi:.3f}]  "
            f"oracle-subspace AUROC={a_oracle:.3f}  "
            f"(in-task full-fit {a_intask_full:.3f})"
        )

    out = RESULTS / "deception-subspace-2026-07-10.json"
    with open(out, "w") as f:
        json.dump(
            dict(
                singular_value_energy=[float(x) for x in sv_norm],
                participation_ratio=participation,
                null_r2=null_r2,
                loto=loto,
            ),
            f,
            indent=2,
        )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
