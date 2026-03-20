# Proprioceptive Market Architecture — Implementation Plan
## frontier-ops v0.2.0 Target

**Created:** 2026-03-20
**Status:** Phase 1 complete. Phase 2 ready to start.
**Repo:** `/Users/joseph/frontier-ops/`

---

## Overview

Ten priorities across five phases. Each phase has a gate condition — don't advance until the gate passes. Earlier phases are foundations; later phases are refinements that can be deferred without losing the core value.

---

## Phase 1: Sequential Detection Foundation ✅ COMPLETE

The base layer. CUSUM accumulators over the two most actionable signals, wired through a binary gate that emits qualitative-only labels.

### Step 1.1: DAS-CUSUM Sequential Detector ✅
**File:** `frontier_ops/sensing/cusum.py`
**What:** Data-Adaptive Symmetric CUSUM (Ahmed et al. 2024) with decay (0.98/step) and ceiling (30.0) to prevent the 125x runaway accumulation bug. Includes SPRT wrapper for formal accept/reject decisions.
**Tests:** 15 tests covering basics, adaptive window, runaway prevention, SPRT boundaries.
**Status:** Implemented. 323 tests passing.

### Step 1.2: Signal D — Deception Tax ✅
**File:** `frontier_ops/sensing/market_signals.py`
**What:** CUSUM over geodesic ratio. Uses existing `ConstitutionalMetric.metric_weighted_distance` for the ratio computation. D_raw = PathLength_G / GeodesicLength_G. Straight trajectory → 1.0, zigzag evasion → >> 1.0.
**Tests:** 5 tests (straight line ≈ 1.0, zigzag > 1.5, stationary = 1.0, reset, detect interface).
**Status:** Implemented.

### Step 1.3: Signal S — Stagnation Tax ✅
**File:** `frontier_ops/sensing/market_signals.py`
**What:** CUSUM over action-rate deviation from empirical baseline. Replaces binary paralysis detector. S_raw = baseline_rate - observed_rate. Positive = stagnation.
**Tests:** 5 tests (baseline rate no alarm, alarm on rate drop, set_baseline calibration, reset).
**Status:** Implemented.

### Step 1.4: Binary Gate — Qualitative Signal Injection ✅
**File:** `frontier_ops/sensing/market_gate.py`
**What:** Intercepts concept_vec + timestamp, runs D + S, returns `None` (silent) or qualitative label string. No numeric values exposed to agent. SIGNAL_LABELS dict for label mapping.
**Tests:** 8 tests including regex verification that no floats leak into labels.
**Status:** Implemented.

### Phase 1 Gate: ✅ PASSED
- [x] All 38 new tests pass
- [x] All 323 total tests pass (no regressions)
- [x] Qualitative-only constraint verified by regex tests
- [x] Runaway prevention verified (125x bug test)
- [x] SPRT can affirmatively accept benign behavior

---

## Phase 2: Calibration & Governance Hardening

Wire the market signals into the existing governance chain so every evaluation is signed and auditable. Then calibrate thresholds from real data so the signals actually fire at the right sensitivity.

### Step 2.1: Governance Chain Integration
**File:** `frontier_ops/governance/market_audit.py` (new)
**Depends on:** Phase 1 complete, existing `GovernanceChain` + `GovernanceAuditor`
**What:**
- After each `MarketGate.evaluate()`, sign a `MarketEntry` into the Ed25519 hash chain
- Entry contains: signal vector [D, S, X, U], alarm states, qualitative label emitted (if any), timestamp
- The agent CANNOT modify these entries — tamper-evidence from existing chain infrastructure
- Add `MarketAuditor` that can replay the chain and verify no entries were altered

**Implementation steps:**
1. Define `MarketEntry` dataclass (signal states + label + timestamp)
2. Add `sign_market_evaluation()` to `GovernanceChain` (or create a parallel chain)
3. Add `verify_market_chain()` to `GovernanceAuditor`
4. Wire into `MarketGate.evaluate()` — sign after every evaluation, not just alarms
5. Add governance chain reference to `MarketGate.__init__` (optional, for backward compat)

**Tests:**
- Chain integrity: sign 50 evaluations, verify chain
- Tamper detection: modify one entry, verify auditor catches it
- Backward compat: MarketGate works without governance chain

**Effort:** Low (2-3 hours). Existing chain infra does the heavy lifting.

### Step 2.2: Threshold Calibration from Telemetry
**File:** `frontier_ops/calibration/market_calibration.py` (new)
**Depends on:** Step 2.1, access to 330K telemetry dataset
**What:**
- Load historical behavioral telemetry (330K entries)
- Segment by HMM task state (different states have different natural rates)
- For Signal D: compute geodesic ratio distribution under benign conditions → set CUSUM threshold via ARL analysis (target ARL > 1000 under benign, ARL < 50 under known-bad)
- For Signal S: compute action-rate distribution per task state → set per-state baseline rates and CUSUM thresholds
- Output: `CalibrationResult` with optimal thresholds, ARL estimates, and diagnostic plots

**Implementation steps:**
1. Build telemetry loader (parse the 330K dataset format)
2. Segment traces by HMM state labels
3. Compute D_raw distribution for all benign trajectories
4. Run ARL sweep for D threshold (use `DASCUSUM.calibrate_from_data` as starting point, then refine with proper ARL simulation)
5. Compute action-rate distributions per state for S
6. Run ARL sweep for S threshold per state
7. Save calibration results as JSON (reproducible)
8. Add CLI entry point: `python -m frontier_ops.calibration.market_calibration --data PATH --output PATH`

**Tests:**
- Calibration runs on synthetic data without errors
- Threshold increases with stricter ARL targets
- Per-state S thresholds differ from global threshold

**Effort:** Medium (4-6 hours). The ARL simulation loop is the core work.

### Step 2.3: SPRT Integration with Market Gate
**File:** Edit `frontier_ops/sensing/market_gate.py`
**Depends on:** Step 2.2 (need calibrated thresholds for meaningful LLR)
**What:**
- Wire SPRTWrapper into MarketGate alongside CUSUM alarms
- SPRT runs in parallel: converts CUSUM statistics to log-likelihood ratios
- Three-state output: "benign" (SPRT accepted) | "violation" (SPRT rejected) | "monitoring" (SPRT continuing)
- "benign" state can emit positive reinforcement label ("Behavioral trajectory consistent with expected path.")
- This is the formal clearance capability that sentinel-guard v1 lacked

**Implementation steps:**
1. Add SPRTWrapper instances to MarketGate (one per signal)
2. Convert CUSUM statistics to LLR using calibrated null/alt distributions
3. Add "accept" labels to SIGNAL_LABELS
4. Update evaluate() to return positive reinforcement on SPRT accept
5. Add session-level verdict: benign/monitoring/violation based on combined SPRT states

**Tests:**
- SPRT accepts after sustained benign behavior
- SPRT rejects after sustained violation
- Positive reinforcement label contains no numeric values
- Session verdict transitions correctly

**Effort:** Low-Medium (2-4 hours).

### Phase 2 Gate: ✅ PASSED (partial — calibration deferred to dataset availability)
- [x] Every market evaluation is signed into governance chain (A1)
- [x] Auditor can verify full chain integrity (A3)
- [x] Tamper detection catches modified entries (A4)
- [ ] D and S thresholds calibrated from real telemetry data (deferred — needs 330K dataset)
- [ ] ARL > 1000 under benign conditions verified empirically (deferred — needs calibration)
- [ ] SPRT can formally accept benign AND reject violation (deferred — needs calibrated LLR)
- [x] All tests pass

---

## Phase 3: Antitrust & Market Health

Prevent any single signal from monopolizing the market. Add entropy-based monitoring that alerts the HUMAN (not the agent) when the market itself is malfunctioning.

### Step 3.1: Shannon Entropy Redistribution
**File:** `frontier_ops/sensing/market_entropy.py` (new)
**Depends on:** Phase 1 signals producing meaningful statistics
**What:**
- `market_entropy(signals)` → normalized Shannon entropy of signal magnitude vector [0, 1]
- `redistribute(signals, floor=0.5)` → soft blend toward uniform when entropy drops below floor
- **Emergency override:** when any signal > critical threshold, bypass redistribution entirely
- Continuous (no hard discontinuity), scale-invariant, preserves signal direction

**Implementation steps:**
1. Implement `market_entropy()` — Shannon entropy normalized by log(N)
2. Implement `redistribute()` — alpha-blend toward uniform based on entropy deficit
3. Add `critical_threshold` parameter — signals above this skip redistribution
4. Calibrate entropy floor from 330K dataset (fire < 5% under benign conditions)
5. Wire into MarketGate: after computing raw signals, apply redistribution before threshold check

**Tests:**
- Uniform signals → entropy = 1.0
- Single dominant signal → entropy < 0.5
- Redistribution preserves signal direction (sign)
- Emergency override: dominant signal at critical threshold is NOT redistributed
- Redistribution is idempotent on uniform distributions

**Effort:** Medium (3-4 hours).

### Step 3.2: Renyi-2 Market Health Monitor
**File:** `frontier_ops/sensing/market_entropy.py` (same file)
**Depends on:** Step 3.1
**What:**
- `market_health(signals)` → normalized Renyi-2 entropy [0, 1]. More sensitive to dominance than Shannon.
- Alert conditions (to HUMAN, not agent):
  - `health < 0.3` sustained > T seconds → MONOPOLY alert
  - `total < epsilon` sustained > T seconds → COLLAPSE alert
- Separate from the agent-facing gate — this is operator telemetry

**Implementation steps:**
1. Implement `market_health()` using Renyi-2 entropy
2. Add `MarketHealthMonitor` class with sustained-alert logic (requires N consecutive readings below threshold)
3. Define alert types: MONOPOLY, COLLAPSE, HEALTHY
4. Wire into MarketGate: compute health after every evaluation, store in MarketSignalState
5. Add health field to governance chain entries (Step 2.1)

**Tests:**
- Healthy market (uniform signals) → health ≈ 1.0
- Monopoly (one signal dominant) → health < 0.3
- Collapse (all signals near zero) → health triggers COLLAPSE
- Sustained alert requires N consecutive readings (not a single spike)
- Health value is in governance chain entry

**Effort:** Low-Medium (2-3 hours).

### Step 3.3: Label Rotation Defense
**File:** Edit `frontier_ops/sensing/market_gate.py`
**Depends on:** Step 3.1
**What:**
- Maintain a pool of semantically equivalent but textually different label templates per signal state
- Rotate stochastically on each evaluation
- Defense against adversarial meta-learning (Open Question #4): agent can't learn a fixed mapping from specific label text to underlying signal state

**Implementation steps:**
1. Expand SIGNAL_LABELS from single strings to lists of equivalent strings (3-5 per state)
2. Add deterministic-random selection (seeded by timestamp + chain position, so it's reproducible for audit but unpredictable to the agent)
3. Update tests to verify all label variants are valid qualitative strings

**Tests:**
- Multiple evaluations with same signal state produce different label text
- All label variants pass the no-numeric-values regex check
- Label selection is reproducible given same seed

**Effort:** Low (1-2 hours).

### Phase 3 Gate: ✅ PASSED
- [x] Entropy redistribution (Shannon) with emergency override (RD1-RD7)
- [x] Emergency override correctly bypasses redistribution for critical signals (RD3)
- [x] Market health monitor identifies monopoly and collapse (MH1-MH5)
- [x] Health values signed into governance chain via MarketAuditChain
- [x] Label rotation produces varied text (3-5 variants per signal state, LR1-LR5)
- [x] Import-time validation on all label variants
- [ ] Entropy floor calibrated from 330K dataset (deferred — needs data)
- [x] All tests pass (51 entropy + 12 audit + 70 Phase 1 = 419 total)

---

## Phase 4: Complete the Signal Vector

Add the remaining two market actors (X and U) to complete the four-signal market. These are lower priority because D + S alone catch the most important failure modes.

### Step 4.1: Signal X — Exploration Credit (KDTree)
**File:** `frontier_ops/sensing/market_signals.py` (add to existing)
**Depends on:** Phase 2 calibration (need historical behavioral vectors for the index)
**What:**
- `ExplorationCreditSignal`: novelty score via k-nearest-neighbor distance in behavioral embedding space
- X(x) = Novelty(x) × (1 - BoundaryProximity(x))
- Novelty = min distance to k=5 nearest historical observations
- Boundary proximity from existing `ConstitutionalMetric.boundary_proximity()`
- Use `scipy.spatial.KDTree` (exact, ~0.1ms at 330K vectors in 6D — FAISS is overkill at this scale)
- Normalized to [0, 1]

**Implementation steps:**
1. Build KDTree index from historical behavioral vectors at init time
2. At runtime: query for k=5 nearest neighbors, take minimum distance
3. Normalize distance to [0, 1] using calibrated scale (max novelty from benign data)
4. Discount by boundary proximity (1 - max boundary activation)
5. Add to MarketGate signal vector
6. Add exploration-specific labels to SIGNAL_LABELS pool

**Tests:**
- Known vector in dataset → low novelty (near 0)
- Vector far from all training data → high novelty (near 1)
- High novelty near boundary → X suppressed
- High novelty far from boundary → X preserved
- KDTree query < 0.5ms on 330K vectors

**Effort:** Medium (3-4 hours).

### Step 4.2: Signal U — Symmetric Deviation from Baseline
**File:** `frontier_ops/sensing/market_signals.py` (add to existing)
**Depends on:** Phase 2 calibration (need per-task-state baselines)
**What:**
- `UtilityDeviationSignal`: deviation from expected behavioral trajectory per task state
- U(t) = behavior_vector(t) - reference(t), where reference = mean behavior in current HMM state
- **Symmetric weighting only** (no Prospect Theory asymmetry — PT parameters unstable across models per literature review)
- Normalized to [-1, 1]
- Negative = underperformance relative to baseline in this state
- Positive = overperformance (more active/diverse than usual)

**Implementation steps:**
1. Load per-task-state behavioral baselines from calibration output
2. At runtime: look up current HMM state → get reference vector
3. Compute signed deviation as L2 distance with sign from dominant dimension
4. Normalize to [-1, 1] using calibrated scale
5. Add to MarketGate signal vector
6. Cold-start fallback: when HMM state is unknown or novel, set U = 0.0 and flag reduced-signal mode

**Tests:**
- Behavior matching baseline → U ≈ 0.0
- Behavior below baseline → U < 0
- Behavior above baseline → U > 0
- Novel task state → U = 0.0 with reduced-signal flag
- Symmetric: |U| for above-baseline = |U| for equivalent below-baseline

**Effort:** Medium (3-4 hours).

### Step 4.3: Four-Signal Gate Integration
**File:** Edit `frontier_ops/sensing/market_gate.py`
**Depends on:** Steps 4.1, 4.2
**What:**
- Wire X and U into MarketGate alongside D and S
- Update entropy redistribution to operate on full 4-signal vector
- Add combined labels for multi-signal states
- Add context window budget: rolling window of last 10 flags + compressed session summary
- Total token budget: max 10 active flags in context at any time

**Implementation steps:**
1. Add X and U signal instances to MarketGate
2. Update `evaluate()` to compute all four signals
3. Apply entropy redistribution to full vector
4. Expand SIGNAL_LABELS for X, U, and multi-signal combinations
5. Add rolling window buffer: store last N labels, compress older ones to summary
6. Add `context_budget()` method that returns the token-efficient representation

**Tests:**
- All four signals fire independently
- Entropy redistribution operates on full vector
- Rolling window caps at 10 active flags
- Session summary compresses correctly
- All combined labels pass qualitative-only check

**Effort:** Medium (4-5 hours).

### Phase 4 Gate:
- [ ] X correctly scores novelty from KDTree index
- [ ] X suppressed near boundaries
- [ ] U symmetric (no PT asymmetry)
- [ ] U cold-start fallback works for novel task states
- [ ] Four-signal gate produces meaningful combined labels
- [ ] Context window budget caps token overhead
- [ ] All tests pass
- [ ] Full pipeline latency < 0.48ms (benchmark)

---

## Phase 5: Proprioceptive Integration

Wire the market into the agent's actual runtime loop. Pre-action estimation, post-action logging, and the efference copy feedback loop.

### Step 5.1: Pre-Action Signal Estimation
**File:** `frontier_ops/sensing/market_predictor.py` (new)
**Depends on:** Phase 4 complete, existing `EfferenceCopyPredictor`
**What:**
- Before action executes: use efference copy to predict post-action state → compute predicted market signals
- If predicted signals elevated: inject pre-action warning
- This is true proprioception — the agent feels resistance before committing
- Start with simple linear predictor: (action_type, current_state) → predicted_signal_vector
- Train on flight recorder data from governance chain

**Implementation steps:**
1. Build `MarketPredictor` class wrapping EfferenceCopyPredictor
2. Add `predict_market_signals(intended_action, current_state)` method
3. Map predicted state through market signals (D, S, X, U) without updating them
4. Generate pre-action label if any predicted signal elevated
5. Add to MarketGate as optional pre-evaluation step

**Tests:**
- Predict that zigzag action will elevate D
- Predict that long delay will elevate S
- Pre-action label is qualitative only
- Prediction doesn't pollute actual signal state

**Effort:** Medium (4-5 hours).

### Step 5.2: Post-Action Flight Recorder
**File:** `frontier_ops/sensing/market_recorder.py` (new)
**Depends on:** Step 2.1 (governance chain), Step 5.1
**What:**
- After every action: log actual vs. predicted market signals
- Prediction error feeds back to improve the efference copy model
- Full audit trail: governance chain entry + flight recorder entry
- Growing dataset: every action grows the baseline distributions for future calibration

**Implementation steps:**
1. Build `MarketFlightRecorder` class
2. Log: timestamp, action_type, concept_vec, predicted_signals, actual_signals, label_emitted, SPRT_state
3. Compute prediction error: actual - predicted for each signal
4. Feed prediction error into EfferenceCopyPredictor.update()
5. Periodic export: dump recorder to calibration-compatible format for re-calibration

**Tests:**
- Recorder stores complete entries
- Prediction error computed correctly
- Export format matches calibration loader
- Recorder doesn't blow memory (ring buffer with configurable max)

**Effort:** Medium (3-4 hours).

### Step 5.3: Curvature Augmentation to D (After Profiling)
**File:** `frontier_ops/boundary/geometry.py` (new)
**Depends on:** Phase 1 D signal working, latency profiling
**What:**
- Add `sectional_curvature(trajectory, G)` computation
- D_raw(τ) = (PathLength / Geodesic) × (1 + κ_max(τ))
- κ_max catches zigzag evasion patterns with same path-length ratio as smooth arcs but different curvature
- **MUST profile first**: full Riemann tensor is O(n^4). In 6D behavioral space this is 1,296 operations — probably fine, but verify < 0.1ms
- If too slow: fall back to Ricci curvature (O(n^2) = 36 operations) or graph-based discrete curvature

**Implementation steps:**
1. Benchmark: time sectional curvature computation for n=6 metric tensor (1000 iterations)
2. If < 0.1ms: implement full sectional curvature via Christoffel symbols → Riemann tensor → sectional curvature
3. If > 0.1ms: implement Ricci scalar (trace of Ricci tensor, O(n^2)) as fallback
4. Alternatively: use Schild's ladder parallel transport for discrete curvature (Rumpf & Wirth 2019)
5. Wire κ_max into DeceptionTaxSignal as multiplicative augmentation
6. Re-calibrate D threshold with curvature-augmented signal

**Tests:**
- Flat metric → zero curvature
- Known curved metric → matches analytical curvature
- Curvature augmentation increases D_raw for zigzag trajectories
- Computation time < 0.1ms for 6D metric

**Effort:** Medium-High (5-7 hours). New math, needs careful validation.

### Step 5.4: OpenClaw Hook Integration
**File:** `frontier_ops/integration/market_hook.py` (new)
**Depends on:** Phases 1-4, OpenClaw tool-call interception layer
**What:**
- Hook into OpenClaw's tool-call pipeline
- Before tool result returns to agent: run MarketGate.evaluate()
- If label returned: inject into tool result context
- If pre-action prediction available: inject before tool execution
- Latency budget: total market pipeline must complete in < 0.48ms

**Implementation steps:**
1. Define `MarketHook` class implementing OpenClaw's hook interface
2. `pre_tool_call(action)` → pre-action prediction
3. `post_tool_call(action, result)` → market evaluation + label injection
4. Add configuration: enable/disable, signal selection, label verbosity
5. Integration test with mock OpenClaw pipeline

**Tests:**
- Hook injects label into tool result when signals elevated
- Hook is silent when signals nominal
- Total latency < 0.48ms (end-to-end benchmark)
- Configuration correctly enables/disables individual signals

**Effort:** Medium (4-5 hours). Depends on OpenClaw hook interface details.

### Phase 5 Gate:
- [ ] Pre-action prediction correctly anticipates signal elevation
- [ ] Flight recorder captures all evaluations with prediction error
- [ ] Curvature augmentation improves D signal (or deferred with justification)
- [ ] OpenClaw hook integration works end-to-end
- [ ] Full pipeline latency < 0.48ms verified by benchmark
- [ ] All tests pass
- [ ] Live session with Chocka produces meaningful market signals

---

## Cross-Phase Concerns

### Non-Stationarity Handling
- **When:** Phase 2 calibration and ongoing
- **What:** Behavioral distributions drift as the agent learns. Baselines go stale.
- **Solution:** Sliding window baselines with exponential decay. Re-calibrate monthly from accumulated flight recorder data.
- **Owner:** Calibration pipeline (Step 2.2)

### Signal Correlation
- **When:** Phase 3 antitrust
- **What:** D and S are likely correlated (deceptive agent may also stagnate)
- **Solution:** Entropy redistribution partially handles this. Monitor correlation coefficient in flight recorder. If > 0.7, consider decorrelation.
- **Owner:** Market health monitor (Step 3.2)

### Context Window Budget
- **When:** Phase 4 integration
- **What:** Token overhead from injected labels must be bounded
- **Solution:** Rolling window of last 10 flags + compressed session summary. Hard cap at ~200 tokens per evaluation context.
- **Owner:** MarketGate (Step 4.3)

### Cold Start
- **When:** Phase 4 (U and X signals)
- **What:** Novel task types have no historical baseline
- **Solution:** Drop to D + S only, set X and U to neutral, flag reduced-signal mode to human
- **Owner:** MarketGate (Step 4.3)

---

## Validation Milestones

| Milestone | Phase | Test |
|-----------|-------|------|
| D detects zigzag evasion | 1 ✅ | Geodesic ratio > 1.5 on zigzag trace |
| S detects stagnation | 1 ✅ | CUSUM fires when action rate drops |
| No numeric values leak | 1 ✅ | Regex test on all labels |
| SPRT formally clears benign | 2 | Accept decision after sustained benign trace |
| ARL > 1000 on benign data | 2 | Calibration from 330K dataset |
| Governance chain tamper-proof | 2 | Auditor catches modified entries |
| Entropy prevents monopoly | 3 | Redistribution fires when single signal dominates |
| Emergency override works | 3 | Critical signal bypasses redistribution |
| Label rotation varies text | 3 | Same state → different labels |
| X scores novelty correctly | 4 | KDTree distance matches expected |
| U symmetric deviation works | 4 | No PT asymmetry in signal |
| Pre-action prediction useful | 5 | Predicted elevated → actual elevated > 70% |
| End-to-end latency < 0.48ms | 5 | Benchmark on M4 MacBook Air |
| Live Chocka session produces signals | 5 | Real market output on real behavior |

---

## Estimated Total Effort

| Phase | Effort | Status |
|-------|--------|--------|
| Phase 1: Sequential Detection Foundation | 4-6 hours | ✅ COMPLETE |
| Phase 2: Calibration & Governance | 8-13 hours | Ready to start |
| Phase 3: Antitrust & Market Health | 6-9 hours | Blocked on Phase 2 |
| Phase 4: Complete Signal Vector | 10-14 hours | Blocked on Phase 2 calibration |
| Phase 5: Proprioceptive Integration | 16-21 hours | Blocked on Phase 4 |
| **Total** | **~44-63 hours** | |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-20 | DAS-CUSUM over vanilla CUSUM | Adaptive symmetric handles varying mean+variance across behavioral regimes |
| 2026-03-20 | Drop PT asymmetric weighting from U | PT parameters unstable across models (2026 arXiv paper) |
| 2026-03-20 | Three-actor market first (D+S+X) then U | U depends on stable HMM state assignments that haven't converged yet |
| 2026-03-20 | KDTree over FAISS for X | 330K vectors in 6D — brute force is fine, FAISS adds complexity for no gain |
| 2026-03-20 | Emergency override in antitrust | Near genuine safety boundary, D MUST be allowed to dominate |
| 2026-03-20 | Governance chain integration moved to Phase 2 | Unsigned window is adversary's operating space |
| 2026-03-20 | Curvature augmentation deferred to Phase 5 | O(n^4) cost needs profiling before commitment |
| 2026-03-20 | Context window budget: 10-flag rolling window | 1000 actions × 30% flag rate × 25 tokens = 7500 tokens unbounded — unacceptable |
| 2026-03-20 | Qualitative-only surfacing is non-negotiable | Skalse impossibility theorem + Anthropic/OpenAI empirical evidence |
