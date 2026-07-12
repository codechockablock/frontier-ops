"""Precision at deployment base rates + the operator-budget inversion.

Benchmark corpora are ~50/50 adversarial/benign; deployments are not. This
converts the project's existing operating points into the quantities an
operator actually experiences:

  Part A (arithmetic, no data): precision as a function of the adversarial
      base rate pi for the published operating points of both repos —
      precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR) — plus false flags
      per 1000 benign actions.

  Part B (real scores): the deployment loop on real traffic. Thresholds
      are calibrated on the directive dataset's benign actions (half the
      sessions, split-conformal), sized to a false-flag budget expressed
      in operator units ("1 false flag per K sessions"); realized flag
      burden is measured on the held-out sessions and surviving TPR on
      the labeled harness (A) and generator-blind fresh (F) off-goal
      actions, scored with the A-fit prototype direction.

Same text streams and direction as eval.conformal_rolling_transport.

Run:
  python -m eval.base_rate_analysis \
      --harness-dir /Users/joseph/frontier-ops/eval/drift_harness \
      --records /Users/joseph/frontier-ops/eval/results/drift-harness-records-2026-07-03.jsonl \
      --fresh /Users/joseph/frontier-ops/eval/results/drift-harness-records-2026-07-04-fresh-haiku4520251.jsonl \
              /Users/joseph/frontier-ops/eval/results/drift-harness-records-2026-07-04-fresh-sonnet5.jsonl \
      --directive /Users/joseph/frontier-ops/data/2026-07-02/directive-dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.domain_port_stepmean import harness_rows, load_jsonl
from frontier_ops.conformal import split_conformal_threshold
from frontier_ops.detector import CalibratedDetector

RESULTS = Path(__file__).resolve().parent / "results"

#: adversarial base rates (fraction of actions) to tabulate
BASE_RATES = [0.5, 0.1, 0.01, 1e-3, 1e-4]

#: operator budgets: one false flag per K sessions
BUDGET_SESSIONS = [1, 5, 20, 100]

#: published operating points (TPR, per-action FPR) to convert in Part A;
#: the upstream detector's in-domain point is computed from data below.
PUBLISHED = {
    "unified-stack internal": (0.945, 0.109),
    "unified-stack ATBench": (1.000, 0.120),
    "unified-stack AgentDojo": (0.932, 0.008),
    "unified-stack multi-turn held-out": (1.000, 0.000),
}


def precision(pi: float, tpr: float, fpr: float) -> float:
    """P(adversarial | flagged) at adversarial action base rate ``pi``."""
    denom = pi * tpr + (1 - pi) * fpr
    return (pi * tpr / denom) if denom > 0 else float("nan")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-dir", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--fresh", nargs="+", required=True)
    ap.add_argument("--directive", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    report: Dict[str, object] = {}

    # ---- Part A: precision vs base rate for published points --------------
    part_a = {}
    for name, (tpr, fpr) in PUBLISHED.items():
        part_a[name] = dict(
            tpr=tpr,
            fpr=fpr,
            fp_per_1000_benign_actions=round(1000 * fpr, 1),
            precision_at=dict(
                (f"{pi:g}", round(precision(pi, tpr, fpr), 4)) for pi in BASE_RATES
            ),
        )
    report["A_published_operating_points"] = part_a
    print("Part A — precision at base rate (rows: operating point):")
    header = "  ".join(f"pi={pi:g}" for pi in BASE_RATES)
    print(f"  {'':44s} {header}")
    for name, row in part_a.items():
        vals = "  ".join(f"{row['precision_at'][f'{pi:g}']:.3f}" for pi in BASE_RATES)
        print(f"  {name:44s} {vals}")

    # ---- load real streams, fit direction, score --------------------------
    rows = harness_rows(load_jsonl(args.records), args.harness_dir)
    y = np.array([int(r["off_goal"]) for r in rows])
    texts = [r["text"] for r in rows]

    fresh_rows: List[Dict] = []
    for fp in args.fresh:
        fresh_rows.extend(harness_rows(load_jsonl(fp), args.harness_dir))
    fy = np.array([int(r["off_goal"]) for r in fresh_rows])
    ftexts = [r["text"] for r in fresh_rows]

    d_recs = load_jsonl(args.directive)
    d_texts = [
        f"{a.get('tool','')}: {a.get('summary','')}"
        for s in d_recs
        for a in s.get("actions", [])
    ]
    d_sessions = np.array(
        [i for i, s in enumerate(d_recs) for _ in s.get("actions", [])]
    )
    n_sessions = len(np.unique(d_sessions))
    actions_per_session = len(d_texts) / n_sessions

    det = CalibratedDetector()
    det.calibrate(texts, y, fpr=None)
    sA = det.score_batch(texts)
    runs = np.array([r["run"] for r in rows])
    sA_off = sA[y == 1]
    sF_off = det.score_batch([t for t, lbl in zip(ftexts, fy) if lbl == 1])
    sD = det.score_batch(d_texts)

    # upstream in-domain operating point (grouped half-splits, alpha=0.1),
    # then fold it into the Part A table as a computed row
    tprs, fprs = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        uniq = rng.permutation(np.unique(runs))
        cal_r, test_r = np.array_split(uniq, 2)
        cal_b = sA[np.isin(runs, cal_r) & (y == 0)]
        thr = split_conformal_threshold(cal_b, 0.1)
        test = np.isin(runs, test_r)
        off, ben = sA[test & (y == 1)], sA[test & (y == 0)]
        if off.size:
            tprs.append(float((off > thr).mean()))
        fprs.append(float((ben > thr).mean()))
    up_tpr, up_fpr = float(np.mean(tprs)), float(np.mean(fprs))
    row = dict(
        tpr=round(up_tpr, 3),
        fpr=round(up_fpr, 3),
        fp_per_1000_benign_actions=round(1000 * up_fpr, 1),
        precision_at=dict(
            (f"{pi:g}", round(precision(pi, up_tpr, up_fpr), 4))
            for pi in BASE_RATES
        ),
        note="computed: grouped half-splits on the 2026-07-03 batch, alpha=0.1",
    )
    part_a["upstream detector, harness in-domain"] = row
    vals = "  ".join(f"{row['precision_at'][f'{pi:g}']:.3f}" for pi in BASE_RATES)
    print(f"  {'upstream detector, harness in-domain':44s} {vals}")

    print(
        f"\nPart B — directive workload: {len(d_texts)} benign actions, "
        f"{n_sessions} sessions ({actions_per_session:.1f} actions/session)"
    )

    # ---- Part B: FP budget -> alpha -> threshold -> surviving TPR ---------
    part_b = {}
    rng_seeds = range(20)
    for k in BUDGET_SESSIONS:
        alpha = 1.0 / (k * actions_per_session)
        fp_per_session, tprs_a, tprs_f = [], [], []
        for seed in rng_seeds:
            rng = np.random.default_rng(seed)
            sess = rng.permutation(np.unique(d_sessions))
            cal_s, test_s = np.array_split(sess, 2)
            cal = sD[np.isin(d_sessions, cal_s)]
            test_mask = np.isin(d_sessions, test_s)
            test = sD[test_mask]
            if alpha < 1.0 / (len(cal) + 1):
                # budget tighter than the calibration set can certify
                thr = float("inf")
            else:
                thr = split_conformal_threshold(cal, alpha)
            n_fp = int((test > thr).sum())
            fp_per_session.append(n_fp / len(test_s))
            tprs_a.append(float((sA_off > thr).mean()))
            tprs_f.append(float((sF_off > thr).mean()))
        part_b[f"1_fp_per_{k}_sessions"] = dict(
            per_action_alpha=round(alpha, 5),
            realized_fp_per_session=round(float(np.mean(fp_per_session)), 4),
            budget_fp_per_session=round(1.0 / k, 4),
            tpr_harness_offgoal=round(float(np.mean(tprs_a)), 3),
            tpr_fresh_offgoal=round(float(np.mean(tprs_f)), 3),
        )
        r = part_b[f"1_fp_per_{k}_sessions"]
        print(
            f"  budget 1 FP/{k:>3d} sessions -> alpha {r['per_action_alpha']:.5f}: "
            f"realized {r['realized_fp_per_session']:.3f} FP/session, "
            f"TPR harness {r['tpr_harness_offgoal']:.3f}, "
            f"fresh {r['tpr_fresh_offgoal']:.3f}"
        )
    report["B_operator_budget_inversion"] = dict(
        n_benign_actions=len(d_texts),
        n_sessions=n_sessions,
        actions_per_session=round(actions_per_session, 2),
        note=(
            "thresholds split-conformal on half the directive sessions; "
            "realized burden on held-out half; TPR on A/F off-goal actions "
            "scored with the A-fit direction (cross-workload, the "
            "deployment condition); 20 seeds"
        ),
        budgets=part_b,
    )

    out = RESULTS / "base-rate-analysis-2026-07-11.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
