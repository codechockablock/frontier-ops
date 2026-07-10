# Q1 — The fourth cell: step-mean × raw 384-D (2026-07-10)

`python -m eval.step_mean_raw_matrix`. All four cells of
{single-point, step-mean} × {4-D chart, raw 384-D} on every Apollo task,
episodes with ≥3 sentence steps, 5-fold CV. Chart cells use the campaign
protocol (nearest-centroid under estimated Σ_w⁻¹, ridge 1e-3, via
StepMeanScorer); raw cells use the prototype direction (= identity-metric
centroid margin; a 384² covariance can't be estimated from ~140 episodes).
Canonical fold seed 11 (battery-comparable) plus seeds 0–9. Batched chart
encoding validated against the battery's per-text cache (max |Δ| = 0.00000).

## AUROC — mean over 11 fold seeds [min, max] (canonical seed 11 in parens)

| task (n eps) | chart single | chart step-mean | raw single | raw step-mean |
|---|---|---|---|---|
| roleplaying (87)* | 0.645 (0.677) | 0.658 (0.692) | 0.694 (0.704) | **0.722 (0.739)** |
| ai_liar (53) | 0.605 (0.541) | 0.566 (0.506) | 0.611 (0.511) | **0.632 (0.557)** |
| insider_trading (171) | 0.615 (0.611) | 0.772 (0.790) | 0.677 (0.695) | **0.824 (0.826)** |
| sandbagging (932) | 0.621 (0.615) | 0.651 (0.649) | 0.968 (0.968) | **0.985 (0.985)** |

\* the ≥3-step filter drops roleplaying from 287 to 87 episodes (short
responses), so this row is not comparable to the 287-row battery numbers;
within-row cell comparisons are fair. ai_liar (n=53) has huge fold noise —
treat its point values as indicative only.

## Findings

1. **raw step-mean is the best cell on all four tasks.** The campaign
   champion (chart step-mean, insider 0.790) is beaten by raw step-mean
   0.826 (seed-mean 0.824, worst seed 0.793 — its *minimum* matches the
   chart's best). Sandbagging: 0.985, paired Δ vs chart step-mean
   +0.336 CI [+0.301, +0.371].

2. **Change-of-mind threshold ("raw ≥ chart + 0.05 on ≥3/4 tasks"):** met
   4/4 on seed-means (+0.064, +0.066, +0.052, +0.334); at the single
   canonical seed only 2/4 (+0.047, +0.051, +0.035, +0.336), and per-task
   paired CIs cross zero everywhere except sandbagging (small n). Verdict:
   direction is consistent 4/4 everywhere and decisive where n is large —
   **the 4-D chart is not load-bearing for detection.** It compresses
   384-D → 4-D through anchors and loses signal on every task; its value is
   interpretability/governance semantics (named dimensions, boundaries,
   costs), which can ride on top of a raw-space detector.

3. **Step-mean is representation-independent.** It beats single-point in
   raw space too (+0.028/+0.021/+0.147/+0.017 seed-mean) — the aggregator,
   not the chart, was the v2 discovery. The gain is largest exactly where
   responses are long (insider, +0.15), consistent with a substantial
   truncation-rescue component (MiniLM's 256-wordpiece limit), plus a
   smaller genuine aggregation gain elsewhere.

## Implications

- The detection stack for v3: **raw-embedding step-mean + in-domain
  supervised direction/metric**, with the chart layered on top for
  human-readable reporting only. This also dissolves the Phase 5 problem —
  the encoder-specific anchors leave the detection path, and a better
  encoder now *helps* (raw path improved under bge) instead of breaking
  everything.
- Combined with Q3 (no shared subspace: effective rank 3.93/4, LOTO
  containment ≈ null): what transfers across domains is the *procedure*
  (split → embed → mean → calibrate in-domain), not any fixed direction,
  metric, or anchor set.
