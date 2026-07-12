# Calibration operating manual

The library ships several ways to set and maintain the detector's flag
threshold. They are not interchangeable — each exists for a specific
regime, and the numbers behind every recommendation here are measured on
the project's own recorded traffic
(`eval/results/conformal-rolling-transport-2026-07-11.md`,
`domain-port-2026-07-10.md`, `calibration-transport-2026-07-04.md`).

## The one rule that dominates everything else

**Thresholds do not transport across workloads.** A threshold calibrated
on one benign task mix runs at 2–3× its advertised false-positive rate on
another (measured: nominal α=0.1 → realized 0.21–0.27 across the
harness→directive boundary), and TPR quoted at an uncalibrated threshold
is inflated by the same mechanism (measured: "TPR 0.95" at 2×α realized
FPR became TPR 0.55–0.64 at the honest operating point). Ship the
procedure, never the constant.

## Decision tree

**1. Fit the direction** — `calibrate(texts, labels, fpr=None)` with
~32–64 labeled in-domain examples (more adds little: label-budget curve in
`label-budget-2026-07-10.md`). The *direction* does transport within a
workload family; the threshold does not.

**2. Set the initial threshold on held-out benign traffic:**

| situation | use | why |
|---|---|---|
| ≥ ~30 held-out benign examples (normal case) | `calibrate_conformal(benign_texts, alpha)` | finite-sample FPR ≤ α under exchangeability; measured in-domain FPR 0.094 at α=0.1 |
| very large benign set (n ≥ ~500) and you prefer the historical behavior | `set_threshold(benign_texts, alpha)` | plug-in quantile; the finite-sample correction is negligible at this n |
| labels/benign set unavailable at deploy time (pipeline setting) | `FullPipeline(enable_warmup_calibration=True)` | label-free warmup from early traffic; only as good as early-traffic representativeness (`warmup-calibration-2026-07-04.md`) |

Expect ~±0.02 realized-FPR slack even in-domain: real session traffic is
heterogeneous at the session level (measured 0.116 at nominal 0.1 across
directive-session splits), so exchangeability is an approximation from
the start.

**3. Maintain the operating point with a rolling window:**

```python
det = CalibratedDetector.load("det.npz")
rt = RollingThreshold(alpha=0.1, window=64, conformal=True)
det.attach_rolling_threshold(rt)

for text in stream:
    flagged = det.flag(text)          # rolling threshold once warm
    ...
    if operator_confirms_benign(text):
        rt.update(det.score(text))
```

- `conformal=True` — the plug-in window quantile overshoots α at small
  windows (+0.04 at window 32 on real streams); the conformal order
  statistic removes the bias (measured post-shift FPR 0.100–0.109 at every
  window from 32 to 256 across the real workload shift, vs frozen 0.211).
- Window ≥ 32 confirmed-benign scores suffices with `conformal=True`.
  These are *unlabeled benign confirmations*, cheaper than the labeled
  examples in step 1.
- Track faster shifts with a smaller window. (A decayed-window mode was
  cut before release: no measured regime needed it, and it is
  incompatible with the conformal correction.)

**4. On a workload change** (new task mix, new deployment, new model
generating the traffic): re-run step 2 on that workload's benign traffic,
or let the rolling window re-fill (it recovers within ~one window of
confirmed-benign scores). Do not carry the old constant across.

**5. Quote detection numbers only at a calibrated operating point.** TPR
at a stale threshold is an artifact of FPR inflation, not detection skill.
On labeled fresh-batch data, matched-FPR TPR was 0.55–0.64 (own-label
ceiling ≈ 0.73) where the stale threshold claimed 0.95.

**6. Choose α from the operator's false-flag budget, not by convention.**
α = 1 / (K × actions-per-session) for a budget of one false flag per K
sessions; the conformal machinery holds the realized burden to that budget
across two orders of magnitude on real traffic. Expect recall ~0.5–0.6 at
tolerable budgets (1 FP per 5–20 sessions), not the α=0.1 numbers — see
`eval/results/base-rate-analysis-2026-07-11.md` for the measured
budget→recall table and the precision-vs-base-rate arithmetic that makes
per-action α=0.1 an alert-fatigue machine in deployment.

## What each guarantee actually covers

- `calibrate_conformal` / `split_conformal_threshold`: marginal FPR ≤ α
  **within the calibration workload** (exchangeability). It is not a
  robustness claim — across a workload boundary it breaks exactly like a
  plug-in threshold (measured 0.215–0.239 at nominal 0.1).
- `RollingThreshold(conformal=True)`: exact under local exchangeability of
  the window; under active drift it is a bias correction plus tracking,
  not a certificate.
- Nothing in this file changes ranking quality: AUROC is threshold-free
  and transports far better than operating points (direction AUC 0.921 on
  a generator-blind batch). Calibration governs *decisions*, not scores.
