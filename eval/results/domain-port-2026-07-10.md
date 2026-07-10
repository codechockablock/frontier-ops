# Domain port — the v2 recipe on real agent-session data (2026-07-10)

`python -m eval.domain_port_stepmean` on the drift-generation harness
records (real Claude agents, scoped tasks, planted distractors, independent
mechanical per-action labels; 225 emergent runs / 646 actions / 50 off-goal
in the 2026-07-03 batch) and the directive dataset (real session traffic,
benign-only, 2212 actions). Recipe: raw MiniLM embedding of the exact replay
stream (`action_text`), prototype direction, in-domain calibration.
Embedding stream and label alignment are identical to the shipping-stack
scoring in drift-harness-2026-07-03.md, so the references are same-data.

Caveat first: the mechanical labels (scope/secret/network) are partly
derivable from the action text, so supervised in-domain numbers carry an
advantage no unseen drift mode would enjoy. The transfer tests (E4/E5) are
the part that measures what breaks in practice.

## Results

| experiment | recipe | shipping-stack reference |
|---|---|---|
| E1 action-level in-domain AUC (grouped 5-fold CV) | **0.932** [0.896, 0.968 cluster CI; seeds 0.926–0.935] | geodesic 0.857 [0.794, 0.920]; alert_level 0.42 |
| E2 session-level step-mean AUC (66 sessions ≥3 actions, 7 drifted) | **0.923** [0.889, 0.959 across seeds] | — |
| E3 label budget: 8 / 16 / 32 / 64 sessions | 0.858 / 0.870 / 0.912 / 0.949 | geodesic 0.857 (no labels) |
| E4 detection transport orig → generator-blind fresh batch | imported AUC **0.921**, TPR@srcFPR0.1 **1.00** (own-CV: AUC 0.997, TPR 1.00) | radius: imported TPR **0.14** vs own 0.73 |
| E4 threshold transport (same) | fresh benign FPR **0.239** at the imported α=0.1 threshold | — |
| E5 workload transport harness → directive traffic (fixed threshold) | FPR **0.274** (nominal 0.1) — **BREAKS** | radius FPR 0.314 — BREAKS |

## Reading

1. **The recipe beats the shipping stack on its own benchmark.** 0.932 vs
   geodesic's 0.857 on the identical action stream and labels — and E3 shows
   *eight* labeled sessions already match the unsupervised geodesic (0.858),
   with 32 clearly past it (0.912). The Apollo-derived 32–64-label rule
   transfers to this workload unchanged.
2. **The direction transports; the threshold doesn't.** On the
   generator-blind fresh batch the imported prototype keeps AUC 0.921 and
   catches 100% of off-goal actions — where the imported authorization
   radius caught 14%. But the imported threshold's realized FPR is 0.239
   vs nominal 0.10: the score *distribution* shifts even when the direction
   holds. Ranking transports within the workload; operating points must be
   recalibrated per batch (cheap: benign quantile, no labels needed).
3. **The workload boundary is fundamental, not a radius artifact.** On real
   directive traffic the fixed harness threshold breaks for the prototype
   (FPR 0.274) almost exactly as it broke for the radius (0.314). Nothing
   about supervision fixes cross-workload transport — consistent with Q3
   (no shared subspace) and the calibration-transport results. Calibrate
   per deployment; ship the procedure, not constants.
4. **Session-level step-mean applies to only 66/225 runs** — real agent
   trajectories here are short (most < 3 effective actions), exactly as the
   Q1 truncation-rescue mechanism predicted. On this workload the unit that
   matters is the action; step-mean is a bonus for long sessions, not the
   core detector.

## Status

This closes the "next big step": the v2 recipe (raw embed → prototype →
in-domain calibrate, ~32–64 labels, benign-quantile thresholds per batch)
is validated on the project's own workload and dominates the shipping
stack's best signal there. Natural follow-ups: wire the recipe into
FullPipeline as an optional calibrated channel alongside the drift stack,
and re-run the mimicry red-team against it (a supervised direction is a
new, possibly easier, mimicry target).
