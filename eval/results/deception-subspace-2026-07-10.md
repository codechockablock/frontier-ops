# Q3 — Is there a shared deception subspace? (2026-07-10)

`python -m eval.deception_subspace` on the battery's cached raw 384-D
MiniLM encodings (Apollo @ f8ec4010). Follow-up to the external-transfer
check (task prototype directions near-orthogonal, mean off-diag |cos| 0.071):
is each task's deception direction contained in the span of the others — a
shared low-rank subspace that multi-task calibration could exploit — or
genuinely novel?

## Answer: no shared subspace. "In-domain only" is fundamental at this representation level.

**Spectrum.** Singular-value energy of the four stacked directions:
[0.293, 0.262, 0.242, 0.203] — participation ratio (effective rank)
**3.93 / 4**. Four mutually orthogonal directions would give 4.0.

**LOTO containment** (R² of the held-out direction onto the span of the
other three; random-direction null E[R²] = 3/384 = 0.0078; episode-bootstrap
95% CIs, B=300):

| held-out | R² | 95% CI | oracle-subspace AUROC | in-task full-fit AUROC |
|---|---|---|---|---|
| roleplaying | 0.016 | [0.002, 0.051] | 0.562 | 0.855 |
| ai_liar | 0.015 | [0.002, 0.056] | 0.654 | 0.927 |
| insider_trading | 0.018 | [0.002, 0.044] | 0.597 | 0.762 |
| sandbagging | 0.023 | [0.003, 0.043] | 0.735 | 0.972 |

Containment is 2–3× the random null with CIs nearly touching it. The
"oracle-subspace" column is the *upper bound* for any zero-shot transfer
built from the other tasks' directions (least-squares projection of the true
held-out direction onto their span, then scored held-out) — it caps at
0.56–0.74 against 0.76–0.97 for in-task fitting. (Both columns are
full-data fits, so the comparison is apples-to-apples upper bounds.)

## Implications

- Pooling or cleverly re-weighting existing task directions cannot fix
  cross-task transfer: there is nothing shared to pool. Fixing transfer
  requires a different representation or per-domain calibration.
- This is the third independent line pointing at the same thesis
  (calibration-transport across workloads breaks; estimated in-task metric
  wins; no shared subspace): **the unit of validity is the in-domain
  distribution, and in-domain calibration is the product, not a caveat.**
- Q2 (label-budget curve for in-domain calibration) is now the binding
  question for deployment.
