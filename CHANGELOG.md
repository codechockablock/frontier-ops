# Changelog

## 0.5.0 — 2026-07-11

Productionization of the v3 detector plus repo hygiene and product seams.
No breaking changes; every pre-0.5 import keeps working.

**Detector productionization**

- **Persistence** — `CalibratedDetector.save(path)` / `load(path)`:
  direction + threshold + calibration provenance in a single `.npz` with a
  format version. Round-trips bit-identically; a `model_name` mismatch or a
  newer format raises a clear error; the encoder stays lazy through `load`.
- **Conformal calibration** — the finite-sample split-conformal quantile
  now lives in `frontier_ops.conformal.split_conformal_threshold`, shared
  by the authorization-radius calibrator (historical radii unchanged) and
  the new `CalibratedDetector.calibrate_conformal(benign_texts, alpha)`,
  which guarantees marginal FPR ≤ α under exchangeability (Monte-Carlo
  verified over 20 seeds). Calibration scores ride along in `save()`.
- **Adaptive thresholding** — `frontier_ops.RollingThreshold`: a sliding /
  exponentially-decayed benign-quantile threshold fed by operator-confirmed
  scores, attachable via `detector.attach_rolling_threshold()`. On a +1σ
  mean-shifted stream it holds FPR within ±0.05 of α while a frozen
  threshold drifts past α + 0.1 — the transport failure measured in
  `eval/results/calibration-transport-2026-07-04.md`, now mitigated.

**Architecture**

- **Encoder protocol** — `frontier_ops.encoder.Encoder`
  (`encode(texts) -> (n, d)`): `CalibratedDetector`,
  `SemanticConceptExtractor`, and `StepMeanScorer` accept any conforming
  encoder by constructor injection; defaults unchanged.
- **Product seams** — `pip install frontier-ops[detect]` (encoder, no
  cryptography) and `[govern]` (cryptography, no encoder), with a
  `frontier_ops.detection` façade. Verified: a detect-only environment
  runs the detector end-to-end with cryptography imports blocked.
- **Governance e2e test** — full producer → export → independent-auditor
  lifecycle, with tamper cases (mutated payload fails at its exact
  sequence number; dropped entries and foreign keys fail verification).

**Hygiene**

- Lint: `ruff check . --select E,F,W --ignore E501` exits 0 repo-wide;
  `eval/session_artifacts/` excluded as verbatim provenance; terse eval
  scripts get per-file style ignores.
- Types: `mypy frontier_ops/` reports 0 errors (lenient config; a
  4-module documented baseline covers quarantined/legacy glue). Fixed a
  real shadowing bug in `constitution.py`'s cross-term loop (rename only).
- CI: runs on all branches; matrix extended to 3.13/3.14; a semantic lane
  installs sentence-transformers so detector tests can no longer silently
  skip; a numpy-only lane guards the bare-import contract; the slow
  battery sits behind `workflow_dispatch`.
- Packaging: `python -m build` + `twine check` pass; the wheel ships
  `py.typed`. Nothing published — that stays a human decision.
- `CLAUDE.md` rewritten for v3 (public repo, detector-first, LEGACY.md
  boundary).

## 0.4.0 — 2026-07-10

v3 lean core — cut the dead weight. A repo-wide evidence audit (`eval/`,
`eval/results/repo-open-questions-2026-07-10.md`) established that detection
reduces to one signal: an in-domain-calibrated **prototype direction**. This
release ships that as the canonical detector and removes the proven defects
from the live path, without deleting exported API (deprecate/quarantine, not
rewrite):

- **`CalibratedDetector`** (`frontier_ops.CalibratedDetector`) — the lean
  detector: raw-encoder embedding + prototype direction + benign-quantile
  threshold. AUROC 0.93 on real agent-session traffic vs the pipeline's
  `alert_level` 0.59 and its best channel subset 0.90. Standalone, in-domain
  calibrated (~32–64 labels), thresholds recalibrated per deployment.
- **Combiner fix** — the efference *surprise* channel was anti-informative
  (only-surprise AUROC 0.429) yet contributed the most benign-side evidence;
  removed from the alert. The authorization escalation `max(alert_level, 0.6)`
  fired on ~98% of actions and flattened the ranking; it no longer overwrites
  `alert_level`. Net: `alert_level` drift AUROC 0.59 → 0.85.
- **Memory quarantined** — `FullPipeline(enable_memory=...)` now defaults to
  `False`; the VSA path has no measured detection value and is opt-in.
- **Component-status table** in the README states which packages the
  benchmarks back (validated / partial / legacy / infrastructure /
  speculative). NEWMA is documented as benign-stable over long real sessions.
- Fixed `pyproject` project URLs (were pointing at the wrong org) and the
  `__version__` string (was `0.1.0`).

## 0.3.0 — 2026-07-09

Post-mortem-driven v2 refactor. A pre-registered evaluation campaign against
the Apollo Research deception benchmarks (see [`eval/`](eval/) — battery,
expected numbers, and the original session harnesses) found that the
expert-asserted metric tensors, position-dependent curvature walls used as a
metric, path-energy trajectory features, and alert-level gating on ranking
paths all carried no signal or destroyed it, while an estimated shrinkage
metric (pooled within-class covariance, ridge 1e-3) and chart-space step-mean
scoring (`StepMeanScorer`, AUROC 0.790 on insider-trading episodes vs 0.611
single-point) won everywhere they ran. This release promotes the winners
(estimated metric as the default path, `StepMeanScorer`, softplus costs
rehoused as a decision layer over calibrated scores), deprecates the losers
with `DeprecationWarning` (nothing deleted), makes custom concept dimensions
first-class constructor parameters end-to-end, renames the static metric to
`CalibratedMetric` (alias kept) to resolve a name collision, annotates the
drift/NEWMA/trend detectors as untested in their design regime (long-running
sessions) rather than refuted, and fixes packaging so fresh-clone editable
installs work on a numpy-only environment.
