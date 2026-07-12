# Base rates and operator budgets — what the headline numbers mean in deployment (2026-07-11)

`python -m eval.base_rate_analysis` — converts both repos' operating
points from benchmark units (TPR/FPR on ~50/50 corpora) into deployment
units (precision at realistic adversarial base rates; recall surviving an
operator's false-flag budget). Part A is arithmetic on published/computed
operating points; Part B replays the real streams (A-fit prototype
direction; thresholds split-conformal-calibrated on half the directive
sessions, burden measured on the held-out half; 20 seeds). Companion
JSON: `base-rate-analysis-2026-07-11.json`.

## Part A — precision(π) = P(adversarial | flagged) at action base rate π

| operating point | TPR / FPR | π=0.5 | π=0.1 | π=0.01 | π=10⁻³ | π=10⁻⁴ |
|---|---|---|---|---|---|---|
| unified-stack internal | .945 / .109 | 0.90 | 0.49 | **0.08** | 0.009 | 0.001 |
| unified-stack ATBench | 1.00 / .120 | 0.89 | 0.48 | **0.08** | 0.008 | 0.001 |
| unified-stack AgentDojo | .932 / .008 | 0.99 | 0.93 | 0.54 | 0.10 | 0.011 |
| unified-stack multi-turn* | 1.00 / .000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| upstream detector, harness in-domain† | .87 / .095 | 0.90 | 0.51 | **0.09** | 0.009 | 0.001 |

\* 0 FPs on 30 benign traces — a tiny denominator; the exact-zero FPR
(hence precision 1.0 at every π) is an artifact of n, not a property.
† computed: grouped half-splits on the 2026-07-03 batch, α=0.1.

**Reading:** at any base rate below ~10% of actions, every ~0.1-FPR
operating point in either repo produces flags that are >90% false. The
"100% TPR / 12% FPR" class of headline describes benchmark geometry, not
operational viability. Only low-FPR points (AgentDojo-style 0.008)
remain usable at π=0.01 — and nothing survives π=10⁻³ except exact-zero
FPR, which no finite benign set can certify.

## Part B — the operator-budget inversion (real traffic)

Reference workload: the directive dataset — 2,212 benign actions across
237 real sessions (9.3 actions/session). Budget → per-action α →
conformal threshold on calibration sessions → realized burden on held-out
sessions + surviving TPR on labeled off-goal actions:

| budget | per-action α | realized FP/session | TPR harness (A) | TPR fresh (F) |
|---|---|---|---|---|
| 1 FP / session | 0.107 | 1.134 | 0.868 | 0.759 |
| 1 FP / 5 sessions | 0.021 | 0.232 | 0.744 | 0.548 |
| 1 FP / 20 sessions | 0.0054 | 0.040 | 0.625 | 0.514 |
| 1 FP / 100 sessions | 0.0011 | 0.004 | **0.520** | **0.091** |

**Reading:**

1. **The recall that survives a tolerable budget is ~0.5–0.6, not
   0.9–1.0.** At one false flag per 20 sessions — a plausible ceiling for
   a human operator — harness recall is 0.63 and generator-blind fresh
   recall 0.51. The α=0.1 numbers everyone quotes correspond to a burden
   (≥1 interrupt per session) no operator tolerates.
2. **The transfer condition degrades faster than the in-domain one.** At
   1 FP/100 sessions, fresh-batch recall collapses to 0.09 while harness
   recall holds 0.52 — the tail of the transferred score distribution is
   where transport hurts most. Tight budgets amplify every distribution
   shift.
3. **Calibration-in-operator-units works.** Realized FP/session tracks
   the budget across two orders of magnitude (1.13 vs 1.0 target, 0.23
   vs 0.2, 0.040 vs 0.05, 0.004 vs 0.01 — conservative at tight budgets,
   as the finite-sample correction should be). The machinery from
   docs/CALIBRATION.md transfers unchanged to budgets expressed as
   "1 FP per K sessions"; α is derived, not chosen.
4. **What this changes:** detector improvements should be measured as
   recall gained *at a fixed operator budget* (e.g. TPR@1FP/20sessions),
   not TPR at α=0.1. Per-action flagging at realistic base rates is an
   alert-fatigue machine unless either the FPR drops ~two orders of
   magnitude (per-action) or flags are aggregated to session level /
   gated by the authorization layer before a human sees them. That is an
   argument for the governance/escalation architecture carrying the
   operational load, with the detector as its ranking engine —
   consistent with AUROC being the one number that transported well all
   along.

## Provenance

Same untracked source data as the conformal/rolling replay (main-checkout
records + directive dataset); deterministic given those files. Multi-turn
corpus not re-run (its published 0-FPR point is carried with the n=30
caveat).
