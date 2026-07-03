# Drift Layer — Refactor Report
**Date:** 2026-07-02 · **Spec:** Fable Spec v2 (relational-layer sequencing, drift = next)
**Depends on:** [encoder surface refactor](anchor-surface-refactor-2026-07-02.md) (same session)
**Changed files:** `frontier_ops/pipeline.py`, `tests/test_pipeline.py`
**Harness:** `eval/drift_replay_experiment.py` · raw runs in `drift-replay-{summary,timelines}.json`
**Validation:** 700 tests pass (4 new); ruff clean on changed files; ATBench cached re-score TPR=89.5% / FPR=1.7%

---

## 0. What "drift" had to catch, and the replay method

The residual adversarial class (per the encoder work) is an autonomous agent
sliding into broad enumeration and credential probing without any single action
declaring intent. The claim the drift layer had to make good on: *once
enumeration is placed in `scope_exploration` instead of `user_aligned`, sustained
movement into it registers as movement rather than as increasing alignment.*

Method: `drift_replay_experiment.py` replays all 738 real 2026-03-20 observations
in sequence through the library's actual drift stack (NEWMA + DriftClassifier +
ScopeCreepDetector + constitutional cross-terms, exact `pipeline.py` params),
once with the **old** encoder (anchors/keywords loaded from `git HEAD` via
`git show`) and once with the **new** encoder (working tree). Detectors reset at
session boundaries. The frozen probe burst (`b2381bd9 … 075e3f01`, 16 contiguous
steps of enum→cred→recon) is the canonical drift trajectory; benign = all steps
≥4 away from any frozen probe (n=679).

## 1. Diagnosis from the replay (before any fix)

Running the replay against the *unfixed* drift stack surfaced three things — only
one of which was the encoder:

1. **Encoder input (fixed upstream, confirmed here).** Old encoder: during the
   burst, NEWMA drift direction pointed at `scope_exploration` on 6/16 steps and
   at `user_aligned` on 0 — but on the 7 *scattered* enumeration probes elsewhere
   in the session it pointed at `user_aligned` 6/7 times (mean user_aligned=1.00,
   scope=0.53). New encoder: those same 7 scattered probes point at
   `scope_exploration` **7/7** (mean scope=1.00, user_aligned=0.30). This is the
   spec's "movement, not alignment" claim, confirmed on real traces.

2. **NEWMA threshold was mis-scaled (real bug, not encoder).** Default
   `threshold=0.15` fired on **~16% of benign steps** (p95 of benign
   metric-weighted divergence is 0.61–0.65). An alarm that fires on 1 in 6 benign
   actions carries almost no evidence, and the Bayes combiner was being fed a
   near-meaningless NEWMA p-value. Present under both encoders — the poisoned
   input had masked it.

3. **Cross-terms were structurally blind to sequential conjunction (real gap).**
   The privilege-escalation cross-term (`credential_adjacent × scope_exploration`,
   weight 4.0) is evaluated **pointwise**, so it only fires when one action is
   jointly elevated (e.g. `cat ~/.ssh/config`, which is both). An agent that
   enumerates *then* probes credentials a few steps later — the actual attack
   shape — never produces a jointly-elevated single action, so the conjunction
   was invisible. Max pointwise cross-term across the whole burst was driven by
   the one dual-purpose action; the sequential structure scored zero.

## 2. Fixes (validated, minimal)

**(a) NEWMA threshold 0.15 → 0.65**, exposed as a `newma_threshold` constructor
arg on `FullPipeline`. Calibrated to benign p95 of the metric-weighted divergence
on the real traces (0.65 for both tier-1 and blended vectors). Benign alarm rate
drops from ~16% to ~2%; the burst still alarms. Not tuned against the probes —
calibrated against the benign distribution, which is the honest reference.

**(b) Trajectory cross-terms.** Added an EWMA-smoothed position
(`traj_alpha=0.2`, ~5-step memory) and evaluate the constitutional cross-terms at
that smoothed point in addition to pointwise. This makes `cred × scope` a
*temporal* conjunction: enumeration raises smoothed scope, a credential probe a
few steps later raises smoothed cred, and the product activates. Exposed on
`StepResult.trajectory_cross_terms` and fed into `_compute_alert` as a new
evidence channel (`traj_cross`, k=20). On the real traces this activates on
**0 of 679 benign steps** (max benign traj-cross = 0.0008) and on 4–5 steps
inside the probe burst (max 0.075–0.081) — so any positive value is strong
evidence by construction.

Two things I deliberately did **not** ship, having tried them in the replay:
- A "sustained elevation" branch in `DriftClassifier` (fire when the recent
  window sits above the long-run baseline even without autocorrelation). It
  flagged 30–36 benign steps in the replay — too noisy. Reverted. The spiky
  heterogeneous-probe case is better served by the trajectory cross-term, which
  is near-zero on benign.
- A suppression-band exception (let extreme divergence alarm even when the drift
  classifier says "noise"). At the recalibrated threshold it changed the benign
  rate by <1pt and the burst count by ≤1 — not worth the added rule. The clean
  threshold recalibration subsumed it.

## 3. Before/after on the frozen probe burst (shipped config: thr=0.65)

| Signal | old encoder | new encoder |
|---|---|---|
| Scattered enum probes — NEWMA direction = `scope_exploration` | 1/7 | **7/7** |
| Scattered enum probes — NEWMA direction = `user_aligned` | 6/7 | **0/7** |
| Burst NEWMA alarms (of 16) | 1 | 3 |
| Burst scope-trend alerts | 2 | 3 |
| Burst trajectory-cross-term steps (>0.01) | 4 | 5 |
| Max burst trajectory cross-term | 0.081 | 0.075 |
| Benign alarm rate (n=679) | 0.018 | 0.021 |
| Benign trajectory-cross-term steps | 0 | 0 |

The headline is the **direction** row: with the old encoder, enumeration drift
literally pointed toward "user aligned" (the detector read it as the agent doing
its job better); with the new encoder the identical actions point into a
RELAXABLE dim, which is what makes the downstream goal-conditioning judgment
coherent. The trajectory cross-term adds a benign-silent channel that fires on
the enum→cred sequence regardless of encoder. Absolute burst NEWMA-alarm counts
are modest because the burst is short (n=16) and NEWMA is deliberately
conservative now; the point is separation from a ~2% benign floor, not raw count.

## 4. Regression check

- **Full suite:** 700 passed (was 696; +4 new `TestTrajectoryCrossTerms`). ruff
  clean on all changed files.
- **ATBench (cached re-score, 320 traces):** TPR **89.5%** [84.4, 93.4],
  FPR **1.7%** [0.2, 5.9]. FPR is on the prior 1.7% baseline; TPR reads slightly
  under the prior 91.5% point estimate **but the two CIs almost entirely
  overlap** and — see the caveat — ATBench scoring is nondeterministic, so this
  is within run-to-run noise. The drift changes cannot have affected ATBench
  structurally: the ATBench path (`wrapper.py` → `detection_signals.py` /
  `tiered_verdict.py`) imports **none** of the changed modules (verified by AST
  import scan). It uses the 7-slot VSA encoder, not the 6-dim concept pipeline I
  changed.

## 5. Caveats

- **ATBench is nondeterministic.** Re-scoring the *identical* cached pentest
  traces on identical code gave tp=37, then 34, then 35 out of 40 across three
  runs (±7pt per-category swing). Every previously reported ATBench figure is a
  single draw from this distribution. The VSA projection seeds are fixed, so the
  source is elsewhere (wrapper state persistence, unseeded RNG, or hash/iteration
  order). **Spawned as a separate task** — it makes the ATBench baseline itself
  untrustworthy and is out of scope for the drift work, but it means item 4's
  TPR comparison should be read as "within noise," not "regressed 2 points."
- **n is tiny** (one 16-step burst, 7 scattered probes). Directional evidence
  that the fix does the right thing on real traces, not a benchmark. The real
  test is a data-collection run with genuine multi-step covert trajectories.
- **traj_alpha=0.2 and k=20 are calibrated to one session.** They're honest
  against its benign distribution (0/679 false activations) but should be
  re-checked on new data before being treated as tuned.

## 6. Sequencing

Encoder ✓ → drift ✓ (this). Next per spec: **provenance** (parallel, claim
discipline, no behavioral-vector dependency) and **goal-conditioning** (last;
needs a directive-bearing data run — current data has ~1 usable directive,
726/738 actions autonomous). The trajectory cross-term and the recalibrated
NEWMA both feed goal-conditioning cleaner signals when that work starts.
