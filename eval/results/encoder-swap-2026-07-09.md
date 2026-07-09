# Phase 5 encoder-swap experiment — 2026-07-09

`python -m eval.battery --encoder BAAI/bge-small-en-v1.5` (informational;
default encoder unchanged per handoff §5/§8). Baselines are the §6
all-MiniLM-L6-v2 numbers; sentence-transformers 5.6.0 for both runs.

| check | metric | MiniLM (baseline) | bge-small-en-v1.5 | Δ |
|---|---|---|---|---|
| zero-shot roleplaying | B response-only AUROC | 0.647 | 0.531 | −0.116 |
| stock pipeline control | A AUROC | 0.529 | 0.500 | −0.029 |
| metric ordering | estimated AUROC | 0.644 (recon 0.617) | 0.576 | −0.068 |
| metric ordering | identity AUROC | 0.588 (recon 0.574) | 0.590 | ≈ |
| step-mean insider | AUROC | 0.790 | 0.616 | −0.174 |
| step-mean insider | TPR@FPR0.2 | 0.648 | 0.297 | −0.351 |
| step-mean − single-point | AUROC gap | +0.180 | −0.004 | collapse |
| external transfer | mean off-diag \|cos\| | 0.071 | 0.052 | more orthogonal |
| external transfer | sandbagging in-task | 0.968 | 0.977 | +0.009 |
| external transfer | min in-task AUROC | 0.643 | 0.706 | +0.063 |

## Reading

- **The zero-shot anchor signal is encoder-specific.** Under bge the earned
  4-D chart collapses to near-chance (0.531) and the step-mean advantage
  disappears entirely. The anchors were written blind *for MiniLM's
  similarity geometry*; both the phrases and the fixed rescale in
  `SemanticConceptExtractor._rescale` (clip((cos−0.1)/0.5), tuned to
  MiniLM's typical cosine range) fail to transfer. Swapping encoders would
  require re-earning anchors and recalibrating the rescale — not a drop-in.
- **The raw representation is better under bge** where supervision exists:
  every in-task 384-D prototype AUROC improves (min 0.643 → 0.706,
  sandbagging 0.968 → 0.977) and task directions are more mutually
  orthogonal (mean off-diag |cos| 0.052). A stronger encoder helps the
  supervised path while breaking the zero-shot anchor path.
- Per handoff §5/§8: the default encoder stays all-MiniLM-L6-v2; this run
  changes nothing.
