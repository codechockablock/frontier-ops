# Architecture Evolution — frontier-ops Market Architecture

## Timeline

### 2026-03-07/08: frontier-ops v0.1.0
- 287 tests, 4 core packages: boundary, sensing, governance, memory
- ConstitutionalMetric, EfferenceCopyPredictor, NEWMA, GovernanceChain
- Deployed as sidecar monitoring Chocka sessions

### 2026-03-20: Market Architecture v0.2.0
- Proprioceptive market with DAS-CUSUM sequential detection
- Entropy-based antitrust prevents single-signal monopoly
- Qualitative-only signal surfacing (Skalse impossibility theorem)
- Ed25519-signed audit chain for every market evaluation
- Label rotation defense against adversarial meta-learning

### Data-Driven Pivot (same day)
- Calibrated against 330K sidecar entries + 538 frontier-ops observations + 24 labeled traces
- **Geodesic ratio (D_raw) invalidated** — behavioral space is discrete, not continuous
- **Per-step CUSUM on raw signals** — 25-49% FPR, unusable
- **Pivoted to verdict-severity CUSUM** — consumes sidecar output instead of recomputing
- **Interval percentile normalization** — handles bimodal burst/idle distribution

### Key Insight
The sidecar already computes good per-step signals. The market architecture
adds value by detecting **sustained patterns** (trajectory-level) that single-step
evaluation misses. A single FLAG is normal; 10 consecutive FLAGs is the signal.

## Architecture Layers

```
Layer 1: Per-step detection (existing sidecar)
  → spike, drift, EWMA, conjunction flags
  → verdict: PASS / MONITOR / FLAG / BLOCK

Layer 2: Trajectory-level detection (market architecture)
  → DAS-CUSUM over windowed verdict severity
  → DAS-CUSUM over interval percentile
  → entropy antitrust (Shannon + Renyi-2)
  → SPRT formal accept/reject (planned)

Layer 3: Governance (audit chain)
  → Ed25519-signed hash chain
  → every market evaluation recorded
  → tamper-evident, externally verifiable

Layer 4: Agent interface (qualitative gate)
  → labels from rotating pool
  → no numeric values exposed
  → conscience, not scoreboard
```

## Test Coverage
- 436 tests across 7 categories
- Invariant, statistical, boundary, adversarial, regression, integration, performance
- Correctness specification: CORRECTNESS_SPEC.md
- Calibration report: CALIBRATION_REPORT.md
