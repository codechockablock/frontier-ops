# Conformal + rolling thresholds vs the real workload shift (2026-07-11)

`python -m eval.conformal_rolling_transport` — the 0.5.0 calibration
features (`CalibratedDetector.calibrate_conformal`, `RollingThreshold`),
which shipped validated only on synthetic streams, replayed against the
project's own recorded traffic using the identical text stream + labels as
`eval.domain_port_stepmean`:

- **A** — drift-harness 2026-07-03 emergent batch: 646 actions, 50 off-goal
- **F** — 2026-07-04 generator-blind fresh batch: 240 actions, 22 off-goal
- **D** — directive dataset (real session traffic): 2212 benign actions

Direction fit on all of A (mirrors E4/E5); α = 0.1 throughout; 20-seed
grouped splits. Companion JSON: `conformal-rolling-transport-2026-07-11.json`.

## Results

| experiment | result | reference |
|---|---|---|
| R1 conformal validity in-domain (A half → A half) | FPR **0.094** ≤ α (plug-in 0.100) | guarantee: ≤ 0.1 |
| R2 conformal threshold A → D | FPR **0.239** — BREAKS | fixed-threshold ref 0.274 |
| R2 conformal threshold A → F benign | FPR **0.215** — BREAKS | fixed-threshold ref 0.239 |
| R2b conformal *within* D (session half-splits) | FPR **0.116** > α | nominal 0.1 |
| R2b reverse D → A | FPR 0.065 (conservative) | — |
| R3 frozen conformal threshold on D | FPR **0.211** | — |
| R3 rolling, window 32 / 64 / 128 / 256 (post-refill) | **0.143 / 0.130 / 0.116 / 0.106** — all in the Q8 band [0.05, 0.15] | — |
| R4 frozen on F | FPR 0.193, **TPR 0.95** | E4: 0.239 / 1.00 |
| R4 rolling-adapted on F (w=32/64/128) | FPR 0.161/0.117/0.089, **TPR 0.64/0.55/0.64** | own-label recal TPR ≈ 0.73 (E4) |

## Reading

1. **Conformal does what it promises — and only that.** In-domain on real
   traffic the finite-sample guarantee holds (0.094 ≤ 0.1). Across the
   workload boundary it decays to ~2.2–2.4×α, indistinguishable from the
   plug-in threshold's break. Conformal is an exchangeability guarantee,
   not a transport fix; nothing about the correction term survives a
   distribution shift. Document it as "per-workload validity," never as
   robustness.
2. **Exchangeability is already imperfect *inside* real traffic.** Session
   half-splits of D give FPR 0.116 at nominal 0.1 — session-level
   heterogeneity breaks exchangeability mildly even with no workload
   change. The unit of "in-domain" is the session mix — the same
   conclusion the calibration-transport study reached for the radius, now
   reproduced for the conformal detector threshold.
3. **The rolling threshold survives its falsification test (Q8: does not
   fire).** Across the real A→D shift the frozen threshold runs at 0.211;
   the rolling window recovers to 0.106–0.143 post-refill, inside the
   pre-registered band at every window size. Improvement is monotone in
   window size; w=32's 0.143 is mostly plug-in quantile bias at small n —
   a conformal-corrected rolling quantile (use `split_conformal_threshold`
   over the window instead of `np.quantile`) is the obvious follow-up.
4. **R4 is the finding that changes the story: the frozen threshold's high
   TPR was purchased with silent FPR inflation.** On the fresh batch the
   frozen threshold reads TPR 0.95 — at a realized FPR of 0.193, double
   the advertised rate. Pull the operating point back to nominal (rolling,
   any window) and TPR lands at 0.55–0.64, near the own-label
   recalibration ceiling of ~0.73. Adaptation doesn't "cost" detection;
   it reveals what detection at the advertised FPR actually is on the
   shifted workload. Any TPR quoted at an uncalibrated threshold on
   shifted traffic is inflated by exactly this mechanism.

## Operational readout

Calibrate per deployment (conformal, for the finite-sample guarantee);
maintain with a rolling window of ≥64 confirmed-benign scores (matches the
32–64 label-budget scale; unlabeled confirmations, cheaper than labels);
treat TPR as workload-specific and only meaningful at a calibrated
operating point. Expect ~±0.02 FPR slack even in-domain from session
heterogeneity (R2b).

## Provenance

Raw records are untracked working-tree data in the main checkout
(`eval/results/drift-harness-records-*.jsonl`,
`data/2026-07-02/directive-dataset.jsonl`); the run is deterministic given
those files (fixed seeds, CPU MiniLM).
