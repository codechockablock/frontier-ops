# Changelog

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
