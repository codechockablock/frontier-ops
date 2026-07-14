# Operating-Point Recalibration — Results
_Generated 2026-07-03._
- Source records: `eval/results/drift-harness-records-2026-07-03.jsonl` (225 emergent runs, 646 actions)
- Off-goal / on-goal emergent actions: 50 / 596

## The defect (from the drift harness)
The stack separates emergent drift well (`geodesic_distance` AUC 0.857), but the DEFAULT authorization radius (0.5) contains only 1.7% of benign on-goal actions — so it escalates almost everything, and the `needs_escalation` floor pins `alert_level` at 0.6, collapsing the combined alert to chance. The radius default is not calibrated in-domain.

## (A) Held-out conformal recalibration
Splits are **clustered by run** (actions within a run are correlated); the radius is calibrated on one split's benign distances and validated on the other. A conformal calibration that transports in-domain should give held-out FPR ≈ α.

Over 300 splits at α=0.1 (median 25 off-goal actions per test split):

| Quantity | Value [95% CI] |
|---|---|
| Calibrated radius | 1.233 [1.133, 1.293] |
| Held-out on-goal false-escalation (target ≈ 0.1) | 0.094 [0.045, 0.139] |
| Held-out off-goal escalation (TPR) | 0.546 [0.350, 0.830] |

Baseline default radius 0.5: on-goal false-escalation **98.3%**, off-goal escalation 100.0%. Calibration converts an unusable 98%-FPR operating point into a conformally-controlled one whose FPR tracks α out-of-sample.

### Operating-point curve (full-data, escalate iff distance > radius)
| α | radius | on-goal FPR | off-goal TPR |
|---|---|---|---|
| 0.05 | 1.313 | 0.044 | 0.52 |
| 0.1 | 1.245 | 0.097 | 0.52 |
| 0.2 | 1.133 | 0.111 | 0.64 |
| 0.3 | 0.872 | 0.284 | 0.86 |

## (B) Does the combined `alert_level` recover?
Single held-out split (75 test runs, 303 actions), radius calibrated to 1.293 on the other split:

| | default radius 0.5 | recalibrated |
|---|---|---|
| `alert_level` AUC | 0.440 | 0.761 |
| on-goal escalation FPR | 98.2% | 5.0% |
| on-goal `alert>0.5` | 99.3% | 20.3% |
| off-goal escalation TPR | 100.0% | 45.5% |

## Conclusion
The 98%-false-positive operating point is a **calibration** defect, not a representation one: the geodesic signal already separates emergent drift; the fixed default radius simply sits far below the benign distance distribution. Calibrating the radius in-domain (which `AuthorizationRadius.calibrate` already supports) restores a usable, conformally-controlled operating point and lets `alert_level` recover. No constant was tuned on this eval set; the recommendation is to calibrate per deployment before relying on `authorized` / `needs_escalation`.

## Limitations
- Same sandbox, same 265-run dataset as the drift harness; no ecological validity claimed beyond it.
- Off-goal actions are few (mostly credential-echo) and concentrated in credential_adjacent runs; TPR CIs are wide.
- Conformal FPR control assumes in-domain exchangeability; the task-affinity work showed cross-session transport breaks it, so calibrate per deployment, not once and forever.
