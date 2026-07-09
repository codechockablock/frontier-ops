# Changelog

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
