# Goal-Conditioning Controls — Negative Quality (Q1) + Lexical Overlap (Q2)
**Date:** 2026-07-03 · **Harness pairs:** identical to `goal_conditioning_eval.py` seed=7, K=5 (via `sample_mismatches`)
**Dataset:** `data/2026-07-02/directive-dataset.jsonl` (gitignored, local-only) · Tier 2 encoders, all-MiniLM-L6-v2 embeddings

Both controls stress the 2026-07-03 conclusion ([main report](goal-conditioning-2026-07-03.md)): concept-space AUC 0.467 (null) vs raw-embedding 0.654 — 'representation ceiling'. Q1 asks whether the negatives were negative at all; Q2 asks whether the embedding signal is just shared vocabulary.

---

## Q1 — AUC stratified by directive↔directive similarity

Mismatched pairs binned by cosine similarity between the OWN and MISMATCHED directive embeddings (quartiles over all sampled units). If 'mismatched' directives were often compatible with the action anyway, AUC should rise sharply in the least-similar bin.

| Stratum | concept AUC | 95% CI | embed AUC | 95% CI | pairs |
|---|---|---|---|---|---|
| Q1 least similar (sim <= 0.10) | 0.454 | [0.349, 0.583] | 0.730 | [0.613, 0.824] | 2508 |
| Q2 (0.10 < sim <= 0.20) | 0.519 | [0.424, 0.603] | 0.725 | [0.640, 0.801] | 2377 |
| Q3 (0.20 < sim <= 0.35) | 0.434 | [0.322, 0.557] | 0.621 | [0.527, 0.711] | 2346 |
| Q4 most similar (sim > 0.35) | 0.462 | [0.358, 0.558] | 0.553 | [0.463, 0.651] | 2784 |

(Aggregate for reference: concept 0.467, embed 0.654.)

## Q2 — embedding AUC with lexical tokens masked from actions

Masking rule (aggressive, applied to ACTION texts only): tokens containing `/` → `<PATH>`; long hex → `<HEX>`; dotted names → `<FILE>`; digit runs → `<NUM>`. 13.4% of action tokens masked. Masked AUC is a LOWER bound on non-lexical signal (masking also destroys legitimate content).

| Band | concept AUC | embed AUC | embed AUC (masked) | Δ masked |
|---|---|---|---|---|
| short (<=6 words) | 0.488 | 0.479 | 0.540 | +0.061 |
| medium (7-15) | 0.526 | 0.580 | 0.570 | -0.011 |
| long (>15) | 0.392 | 0.837 | 0.712 | -0.124 |
| **overall** | 0.467 | 0.654 | 0.617 | -0.037 |

## Reading the results

*Interpretation thresholds were fixed BEFORE running (session notes, 2026-07-03): Q1 kills the representation-ceiling story if least-similar-stratum concept AUC ≥ ~0.6; Q2 undermines the embedding channel if the masked long-band AUC collapses toward 0.5.*

**Q1 verdict — the negatives WERE noisy, and the ceiling story survives anyway.** Embedding AUC rises monotonically as the mismatched directive gets less similar (0.553 in the most-similar stratum → 0.730 in the least-similar), confirming that many aggregate 'negatives' were semantically compatible pairs and the aggregate embedding number understates the true ceiling — on the cleanest negatives it is ~0.73, not 0.65. Concept-space AUC, by contrast, stays at chance in EVERY stratum (0.454 on the least-similar, CI [0.349, 0.583]): the 6-dim projection cannot discriminate even clearly-different directives. The gradient in one space and flatness in the other is the sharpest form of the representation-ceiling finding.

**Q2 verdict — the embedding signal survives aggressive masking.** Long-band AUC drops 0.837 → 0.712 with 13% of action tokens masked; the drop (an UPPER bound on the lexical contribution, since masking also destroys legitimate content) leaves the masked signal well above both chance and concept space. The prospective task-affinity channel is grounded in semantic affinity, with a quantified lexical component — but note the adversarial caveat: cosine affinity to the directive is mimicable by a covert action that borrows the directive's vocabulary; masking bounds the benign lexical share, not adversarial robustness.

