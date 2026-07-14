# v3 lean core — what's on the detection path, what's legacy

As of 0.4.0 the detector is one thing: `frontier_ops.CalibratedDetector`, an
in-domain-calibrated prototype direction (AUROC 0.93 on real agent-session
traffic). A repo-wide evidence audit
([eval/results/repo-open-questions-2026-07-10.md](../eval/results/repo-open-questions-2026-07-10.md))
established that no subset of the old multi-signal pipeline beat it, and that
several signals were actively harmful. This document is the boundary: what
carries detection signal, what is kept for a non-detection reason, and what is
dead weight quarantined out of the default path.

**Nothing is deleted.** frontier-ops is an upstream library with a downstream
production consumer (unified-stack); every exported symbol still imports. Dead
weight is removed from the *default execution path* and marked, so removal
stays a one-line change later and downstreams don't break today.

## On the detection path (validated)

| Component | Role | Evidence |
|---|---|---|
| `CalibratedDetector` | the detector — prototype direction, in-domain calibrated | AUROC 0.93 (domain port); beats pipeline alert_level 0.59 and best channel subset 0.90 |
| `boundary` — encoder, `StepMeanScorer`, `CalibratedMetric` | encoding + estimated-metric scoring | Apollo battery 0.79–0.93 |
| `authorization` — goal / radius / provenance | scope envelope + audit semantics | drift-harness geodesic AUROC 0.86; radius transport mapped |
| `sensing.newma` (`DualEWMA`) | streaming drift channel | benign-stable over long real sessions (FPR ~3.6%, flat with length); complements the prototype |
| `pipeline.alert_level` | combiner output, post-fix | 0.59 → **0.86** after cutting the surprise/direction channels + escalation overwrite |

## Kept, but NOT for detection

| Component | Why it stays | Not cut because |
|---|---|---|
| `governance` (Ed25519 chain, ledger, market audit) | tamper-evident audit trail | infrastructure — a cryptographic correctness claim, not an AUROC |
| `sensing` cross-term / trend / traj-cross channels | designed for the drift/NEWMA long-session regime (probe bursts) | untested-in-regime ≠ refuted; they fired 0.00 on the short drift workload, which is out of their design regime |
| `sensing.cusum`, `market_signals`, `market_gate`, `market_entropy`, `cold_start` | the live "Market Architecture" | used by the `integration` layer; a different (glue) subsystem, out of scope for the detection cut |
| `sensing.efference` (`EfferenceCopyPredictor`) | still produces `proprioceptive_context` / `pred_error` on `StepResult` | **removed from the alert** (both its channels were at/below chance) but kept as a diagnostic field |
| `sensing.extractors` (`BehavioralExtractor`) | legacy pre-concept extractor | still exercised by the ATBench harnesses |

## Quarantined dead weight (off the default path)

| Component | Action (0.4.0) | Basis |
|---|---|---|
| efference **surprise** + **direction** alert channels | removed from `_compute_alert` | only-surprise AUROC 0.429, only-direction 0.469 (at/below chance); surprise dumped the most benign-side evidence of any channel (Q1) |
| authorization escalation `max(alert_level, 0.6)` | removed (escalation now a flag + reason, never overwrites the ranking) | fired on ~98% of actions, flattened the ranking (Q1) |
| `memory` (VSA phasor, activation priming) | `FullPipeline(enable_memory=...)` default `True` → `False`; opt-in | no measured detection value (Q7) |
| `sensing.modes` (`ViolationModeDetector`, ICA) | `DeprecationWarning` on import | orphaned: imported nowhere, unexported, zero benchmark contact; premise predates v3 |

## Already deprecated in v2 (0.3.0)

Position-dependent curvature-as-metric (`metric_weighted_distance`),
path-energy features (`metric_weighted_path_length`), and expert-asserted-G
construction all emit `DeprecationWarning` (they lost their pre-registered
kill tests on the Apollo benchmarks). The static metric was renamed
`CalibratedMetric` (alias kept).

## Migration

```python
# before (multi-signal pipeline as the detector)
from frontier_ops import FullPipeline
pipe = FullPipeline(constitution=...)
score = pipe.process_step(text).alert_level      # 0.59 AUROC on drift

# after (lean detector)
from frontier_ops import CalibratedDetector
det = CalibratedDetector()
det.calibrate(train_texts, train_labels)         # ~32–64 labeled in-domain examples
det.set_threshold(benign_texts, alpha=0.1)       # recalibrate per deployment
score = det.score(text)                          # 0.93 AUROC on drift
```

`FullPipeline` remains for the authorization + governance + provenance layer;
read `alert_level` (post-fix) or the NEWMA channel for its detection signal,
never the full-ensemble gating.

## 0.6.0 — the attic

The quarantine table above described 0.4.0's default-path cuts. In 0.6.0
the quarantined and no-contact components left `main` entirely — the Q2
audit established there are no downstream consumers, so "importable but
deprecated" no longer bought anything. The complete pre-cut tree is
preserved on the **`attic/pre-v0.6`** branch; the CHANGELOG lists every
removed module. Archival eval scripts that reference atticked modules
document their experiments as run and execute against that branch.
