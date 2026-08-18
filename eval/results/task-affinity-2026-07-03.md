# Task-Affinity Channel — Held-Out Evaluation
**Date:** 2026-07-03 · **Spec:** `fable-spec-task-affinity.md` · **Channel:** `frontier_ops/authorization/task_affinity.py` (all-MiniLM-L6-v2 cosine, escalate-only)
**Split:** deterministic session hash — 5 calibration sessions (20 records) / 10 held-out sessions (123 records). No channel constant was fitted to held-out sessions; the frozen seed-7 goal-conditioning pairs were not reused.
**Determinism:** seed=7, K=5, 1000 bootstrap resamples (cluster bootstrap over records).

---

## Held-out directional result (threshold-free)

- **Affinity AUC = 0.631** (95% CI [0.560, 0.701]) on 123 held-out records / 8345 pairs — P(affinity-to-own-directive > affinity-to-mismatched), ties 0.5.
- Masked-action robustness variant (paths/files/ids/numbers masked): **0.584** (95% CI [0.527, 0.641]).

| Band | AUC | 95% CI | AUC (masked) | records | pairs |
|---|---|---|---|---|---|
| short (<=6 words) | 0.403 | [0.325, 0.481] | 0.506 | 30 | 1880 |
| medium (7-15) | 0.588 | [0.489, 0.684] | 0.552 | 51 | 3780 |
| long (>15) | 0.850 | [0.771, 0.909] | 0.684 | 42 | 2685 |

Similarity strata (negative-quality view — same methodology as the goal-conditioning controls):

| Stratum (directive↔directive affinity) | AUC | 95% CI | pairs |
|---|---|---|---|
| least similar | 0.737 | [0.620, 0.834] | 2065 |
| second | 0.638 | [0.533, 0.740] | 2062 |
| third | 0.677 | [0.601, 0.750] | 1773 |
| most similar | 0.501 | [0.388, 0.628] | 2445 |

## Conformal calibration demo (escalate-only threshold)

`AffinityThreshold` calibrated on the calibration split only (334 benign own-pairs, alpha=0.1) → threshold = 0.052. On 1669 held-out benign own-pairs the false-escalation rate is **19.8%** (target ≲ 10%, distribution-free).

**The miss is informative, not a tuning target.** The conformal guarantee assumes exchangeability between calibration and deployment pairs; splitting by SESSION breaks it — different sessions have different directive styles and affinity baselines, and the calibration half here is small (5 sessions / 20 records). A 20%-vs-10% inflation under cross-session transfer means the threshold must be calibrated in-domain (per deployment/workload, or rolling), exactly as radius calibration is already framed. Alpha was not adjusted after seeing this number.

## Scope of the claim

- **Directionality only.** All actions are on-goal; negatives are mismatched real pairs. This does not test detection of genuinely off-goal or covert actions.
- **Escalate-only by construction.** Low affinity adds an escalation signal; high affinity authorizes nothing (`authorized` is provably unaffected — see `tests/authorization/test_task_affinity.py::TestAsymmetry`). Under this rule, vocabulary mimicry can only restore an attacker's no-channel baseline, never unlock anything.
- **Mimicry robustness is UNTESTED.** No adversarial data exists; the masking variant bounds the benign lexical share only.
- One user, one workload, 22 sessions; session split is not a time split. MiniLM was fixed in advance, not searched.
- No constants were tuned on held-out data; the only fitted quantity (the threshold) comes from the calibration split and is evaluated once on held-out.
