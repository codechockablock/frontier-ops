"""Do calibrate_conformal and RollingThreshold survive the measured
workload-transport failure — on the real recorded streams?

The 0.5.0 features were validated on synthetic data only. This replays the
project's own records through the shipped APIs:

  A = drift-harness 2026-07-03 emergent batch (labeled; the E5 source)
  F = 2026-07-04 generator-blind fresh batch (labeled)
  D = directive dataset (real session traffic, benign-only)

using the identical text stream + labels as eval.domain_port_stepmean
(whose references are: fixed-threshold FPR harness->directive 0.274,
harness->fresh 0.239, nominal alpha 0.1).

Experiments:
  R1  in-domain conformal validity: grouped half-splits of A's runs,
      conformal threshold from cal-half benign -> FPR on test-half benign.
      Also the plug-in (np.quantile) threshold for contrast.
  R2  conformal transport: the same per-seed conformal thresholds ->
      FPR on D and on F benign; plus directive-internal validity
      (cal on D-half -> FPR on D-half) and the reverse direction D -> A.
  R3  rolling recovery across the real shift: stream A-benign then
      D-benign; frozen conformal threshold vs RollingThreshold at
      window in {32, 64, 128, 256}; flag-then-update (operator confirms
      benign after the decision). Reports post-refill FPR per window and
      the Q8 verdict band [alpha-0.05, alpha+0.05].
  R4  detection cost of adaptation on F (labeled): frozen threshold
      (TPR ref 1.00 / FPR ref 0.239) vs the rolling threshold adapted on
      F benign — does pulling FPR back to alpha cost TPR?

Run (paths point at the checkout holding the untracked records):
  python -m eval.conformal_rolling_transport \
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
from frontier_ops.adaptive_threshold import RollingThreshold
from frontier_ops.conformal import split_conformal_threshold
from frontier_ops.detector import CalibratedDetector

RESULTS = Path(__file__).resolve().parent / "results"
ALPHA = 0.1
BAND = (ALPHA - 0.05, ALPHA + 0.05)  # the Q8 verdict band
SEEDS = list(range(20))
WINDOWS = [32, 64, 128, 256]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-dir", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--fresh", nargs="+", required=True)
    ap.add_argument("--directive", required=True)
    return ap.parse_args()


def grouped_half_split(runs: np.ndarray, seed: int):
    """Index halves that never split a run."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(runs))
    half = np.array_split(uniq, 2)
    return (
        np.flatnonzero(np.isin(runs, half[0])),
        np.flatnonzero(np.isin(runs, half[1])),
    )


def fpr(scores: np.ndarray, thr: float) -> float:
    return float((scores > thr).mean())


def rolling_replay(
    phase1: np.ndarray, phase2: np.ndarray, window: int, alpha: float = ALPHA
) -> Dict:
    """Warm a RollingThreshold on phase1, then flag-then-update through
    phase2 (every score is operator-confirmed benign). Returns realized
    phase-2 FPR overall and post-refill (after `window` updates)."""
    rt = RollingThreshold(alpha=alpha, window=window, min_n=20)
    for s in phase1:
        rt.update(float(s))
    flags = []
    for s in phase2:
        flags.append(bool(s > rt.threshold))
        rt.update(float(s))
    flags_arr = np.array(flags)
    post = flags_arr[window:]
    return dict(
        fpr_overall=float(flags_arr.mean()),
        # None when the stream is shorter than the window (never refills)
        fpr_post_refill=float(post.mean()) if post.size else None,
        final_threshold=float(rt.threshold),
    )


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)
    report: Dict[str, Dict] = {}

    # ---- load the exact domain-port streams --------------------------------
    rows = harness_rows(load_jsonl(args.records), args.harness_dir)
    runs = np.array([r["run"] for r in rows])
    y = np.array([int(r["off_goal"]) for r in rows])
    texts = [r["text"] for r in rows]

    fresh_rows: List[Dict] = []
    for fp in args.fresh:
        fresh_rows.extend(harness_rows(load_jsonl(fp), args.harness_dir))
    fy = np.array([int(r["off_goal"]) for r in fresh_rows])
    ftexts = [r["text"] for r in fresh_rows]

    d_texts = [
        f"{a.get('tool','')}: {a.get('summary','')}"
        for s_rec in load_jsonl(args.directive)
        for a in s_rec.get("actions", [])
    ]
    d_sessions = np.array(
        [
            i
            for i, s_rec in enumerate(load_jsonl(args.directive))
            for _ in s_rec.get("actions", [])
        ]
    )

    # ---- shipped API end-to-end: fit direction, embed once -----------------
    det = CalibratedDetector()
    det.calibrate(texts, y, fpr=None)  # direction on all of A (mirrors E4/E5)
    sA = det.score_batch(texts)
    sF = det.score_batch(ftexts)
    sD = det.score_batch(d_texts)
    bA = sA[y == 0]
    print(
        f"A: {len(y)} actions ({y.sum()} off-goal) | F: {len(fy)} "
        f"({fy.sum()} off-goal) | D: {len(sD)} benign"
    )

    # sanity: the shipped calibrate_conformal on all A-benign text
    api_thr = det.calibrate_conformal(
        [t for t, lbl in zip(texts, y) if lbl == 0], alpha=ALPHA
    )
    assert np.isclose(api_thr, split_conformal_threshold(bA, ALPHA))

    # ---- R1 + R2: conformal validity in-domain, decay across workloads -----
    r1_conf, r1_plug, r2_D, r2_F = [], [], [], []
    for seed in SEEDS:
        cal_idx, test_idx = grouped_half_split(runs, seed)
        cal_b = sA[cal_idx][y[cal_idx] == 0]
        test_b = sA[test_idx][y[test_idx] == 0]
        thr_c = split_conformal_threshold(cal_b, ALPHA)
        thr_p = float(np.quantile(cal_b, 1 - ALPHA))
        r1_conf.append(fpr(test_b, thr_c))
        r1_plug.append(fpr(test_b, thr_p))
        r2_D.append(fpr(sD, thr_c))
        r2_F.append(fpr(sF[fy == 0], thr_c))
    report["R1_in_domain_validity"] = dict(
        alpha=ALPHA,
        conformal_fpr_mean=float(np.mean(r1_conf)),
        conformal_fpr_max=float(np.max(r1_conf)),
        plugin_fpr_mean=float(np.mean(r1_plug)),
        n_cal_benign_typical=int(len(cal_b)),
    )
    report["R2_transport_decay"] = dict(
        fpr_on_directive_mean=float(np.mean(r2_D)),
        fpr_on_fresh_benign_mean=float(np.mean(r2_F)),
        reference_fixed_threshold=dict(directive=0.274, fresh=0.239),
    )
    print(
        f"R1 in-domain: conformal FPR {np.mean(r1_conf):.3f} "
        f"(plug-in {np.mean(r1_plug):.3f}) at alpha {ALPHA}\n"
        f"R2 transport: FPR on directive {np.mean(r2_D):.3f}, "
        f"on fresh {np.mean(r2_F):.3f}"
    )

    # directive-internal validity + reverse direction
    dv, rev = [], []
    for seed in SEEDS:
        cal_idx, test_idx = grouped_half_split(d_sessions, seed)
        thr_d = split_conformal_threshold(sD[cal_idx], ALPHA)
        dv.append(fpr(sD[test_idx], thr_d))
        rev.append(fpr(bA, thr_d))
    report["R2b_directive_internal_and_reverse"] = dict(
        directive_internal_fpr_mean=float(np.mean(dv)),
        reverse_D_to_A_fpr_mean=float(np.mean(rev)),
    )
    print(
        f"R2b directive-internal conformal FPR {np.mean(dv):.3f}; "
        f"reverse D->A FPR {np.mean(rev):.3f}"
    )

    # ---- R3: rolling recovery across the real A -> D shift -----------------
    frozen_thr = split_conformal_threshold(bA, ALPHA)
    r3: Dict[str, Dict] = {}
    rng = np.random.default_rng(0)
    for window in WINDOWS:
        per_seed = []
        for seed in range(5):
            # session-grouped shuffle of D preserves within-session order
            order = rng.permutation(np.unique(d_sessions))
            idx = np.concatenate([np.flatnonzero(d_sessions == s) for s in order])
            per_seed.append(rolling_replay(bA, sD[idx], window))
        r3[f"window_{window}"] = dict(
            fpr_overall_mean=float(np.mean([p["fpr_overall"] for p in per_seed])),
            fpr_post_refill_mean=float(
                np.mean([p["fpr_post_refill"] for p in per_seed])
            ),
            in_band=bool(
                BAND[0]
                <= float(np.mean([p["fpr_post_refill"] for p in per_seed]))
                <= BAND[1]
            ),
        )
    report["R3_rolling_recovery"] = dict(
        frozen_conformal_fpr_on_D=fpr(sD, frozen_thr),
        band=list(BAND),
        windows=r3,
    )
    print(f"R3 frozen threshold FPR on D: {fpr(sD, frozen_thr):.3f}")
    for w, res in r3.items():
        print(
            f"R3 {w}: overall {res['fpr_overall_mean']:.3f}, post-refill "
            f"{res['fpr_post_refill_mean']:.3f}, in-band {res['in_band']}"
        )

    # ---- R4: detection cost of adaptation on the labeled fresh batch -------
    fb, fo = sF[fy == 0], sF[fy == 1]
    r4: Dict[str, Dict] = {
        "frozen": dict(
            fpr=fpr(fb, frozen_thr),
            tpr=float((fo > frozen_thr).mean()),
            reference=dict(fpr=0.239, tpr=1.00),
        )
    }
    for window in WINDOWS:
        if window >= len(fb):
            continue  # stream shorter than the window; post-refill undefined
        res = rolling_replay(bA, fb, window)
        adapted = res["final_threshold"]
        r4[f"window_{window}"] = dict(
            fpr_post_refill=res["fpr_post_refill"],
            tpr_at_final_threshold=float((fo > adapted).mean()),
            final_threshold=adapted,
        )
    report["R4_detection_cost_fresh"] = r4
    print(
        f"R4 frozen: FPR {r4['frozen']['fpr']:.3f} TPR {r4['frozen']['tpr']:.2f}"
    )
    for window in WINDOWS:
        res = r4.get(f"window_{window}")
        if res is None:
            continue
        print(
            f"R4 window {window}: FPR {res['fpr_post_refill']:.3f}, "
            f"TPR {res['tpr_at_final_threshold']:.2f}"
        )

    out = RESULTS / "conformal-rolling-transport-2026-07-11.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
