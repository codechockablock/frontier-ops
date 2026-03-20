# Correctness Specification: Proprioceptive Market Architecture
## frontier-ops v0.2.0

**Created:** 2026-03-20
**Status:** Authoritative specification. Code must satisfy these properties.

---

## 1. DAS-CUSUM Sequential Detector

### 1.1 Algorithm Definition

The detector maintains two one-sided CUSUM statistics (S⁺, S⁻) that accumulate
standardized deviations from an adaptive reference distribution.

**Reference estimation:** A sliding window of the most recent `W` observations
(NOT including the current observation) provides the reference distribution.
The current observation is evaluated AGAINST this reference, then appended to the window.

```
reference_window = observations[max(0, t-W) : t]  # excludes current
μ_ref = mean(reference_window)
σ_ref = std(reference_window, ddof=1)
z_t = (x_t - μ_ref) / max(σ_ref, ε)  # standardized observation
```

**Accumulation with decay:**
```
S⁺_t = min(ceiling, max(0, decay · S⁺_{t-1} + z_t - drift))
S⁻_t = min(ceiling, max(0, decay · S⁻_{t-1} - z_t - drift))
statistic_t = max(S⁺_t, S⁻_t)
alarm_t = (statistic_t > threshold)
```

### 1.2 Invariants (MUST hold at all times)

| ID | Invariant | Rationale |
|----|-----------|-----------|
| C1 | `0 ≤ S⁺ ≤ ceiling` | Bounded by max(0, ·) and min(ceiling, ·) |
| C2 | `0 ≤ S⁻ ≤ ceiling` | Same |
| C3 | `0 ≤ statistic ≤ ceiling` | statistic = max(S⁺, S⁻) |
| C4 | `run_length ≥ 0` | Monotonically increasing between resets |
| C5 | `len(window) ≤ window_size` | Bounded memory |
| C6 | `alarm ⟹ statistic > threshold` | Alarm only when threshold exceeded |
| C7 | `¬alarm ⟹ statistic ≤ threshold` | No alarm when below threshold |
| C8 | After `reset()`: `S⁺ = S⁻ = 0, len(window) = 0, run_length = 0` | Clean state |
| C9 | Reference window excludes current observation | No self-contamination |
| C10 | `decay ∈ (0, 1]`, `ceiling > 0`, `threshold > 0`, `drift ≥ 0` | Valid parameters |

### 1.3 Statistical Properties (verified by simulation)

| ID | Property | Test method |
|----|----------|-------------|
| S1 | FPR < 1% on iid N(0,1) with default parameters over 1000 steps | Monte Carlo, 100 trials |
| S2 | Detection delay < 50 steps for +3σ mean shift | Monte Carlo, 100 trials, median delay |
| S3 | ARL₀ > 500 under default parameters | Empirical ARL from benign simulation |
| S4 | ARL₁ < 100 for +3σ shift under default parameters | Empirical ARL under shift |
| S5 | Statistic converges to steady-state under stationary input | After 2×window_size steps, statistic < threshold |
| S6 | After regime change + 2×window_size settling, FPR returns to nominal | Adaptation test |

### 1.4 Boundary Conditions

| Condition | Expected behavior |
|-----------|-------------------|
| `value = NaN` | Raise `ValueError` |
| `value = ±inf` | Raise `ValueError` |
| `window_size = 1` | Raise `ValueError` (need ≥ 2 for std) |
| `threshold ≤ 0` | Raise `ValueError` |
| `decay ≤ 0 or > 1` | Raise `ValueError` |
| `ceiling ≤ 0` | Raise `ValueError` |
| `drift < 0` | Raise `ValueError` |
| Single observation | Return (0.0, False), no alarm possible |

---

## 2. SPRT Wrapper

### 2.1 Algorithm Definition

Wald's Sequential Probability Ratio Test. Accumulates log-likelihood ratios
and compares against two boundaries derived from Type I (α) and Type II (β) error rates.

```
boundaries:
  lower = log(β / (1 - α))       # accept H₀ (benign)
  upper = log((1 - β) / α)       # reject H₀ (violation)

update(llr):
  cumulative += llr
  if cumulative ≥ upper: return "reject"
  if cumulative ≤ lower: return "accept"
  return "continue"
```

### 2.2 Invariants

| ID | Invariant | Rationale |
|----|-----------|-----------|
| P1 | `lower < 0 < upper` for valid α, β ∈ (0, 0.5) | Wald boundary properties |
| P2 | `"reject" ⟹ cumulative ≥ upper` | Definition of rejection |
| P3 | `"accept" ⟹ cumulative ≤ lower` | Definition of acceptance |
| P4 | `"continue" ⟹ lower < cumulative < upper` | In the continuation region |
| P5 | Decision is absorbing: once "accept" or "reject", no further updates | Terminal states |
| P6 | After `reset()`: `cumulative = 0.0` | Clean state |

### 2.3 Boundary Conditions

| Condition | Expected behavior |
|-----------|-------------------|
| `α ≤ 0 or ≥ 0.5` | Raise `ValueError` |
| `β ≤ 0 or ≥ 0.5` | Raise `ValueError` |
| `llr = NaN` | Raise `ValueError` |
| `llr = ±inf` | Raise `ValueError` |
| Update after terminal decision | Raise `RuntimeError` |

---

## 3. Signal D — Deception Tax

### 3.1 Algorithm Definition

Geodesic efficiency ratio fed through DAS-CUSUM.

```
On each step t with concept vector x_t:
  path_length += d_G(x_{t-1}, x_t)     # metric-weighted step distance
  geodesic = d_G(x_0, x_t)             # metric-weighted direct distance
  D_raw = path_length / max(geodesic, ε)
  
  If geodesic < ε: D_raw = 1.0  # stationary, no information
  
  Feed D_raw into CUSUM
```

Where `d_G(a, b) = √((b-a)ᵀ G((a+b)/2) (b-a))` is the metric-weighted distance
using the midpoint approximation.

### 3.2 Invariants

| ID | Invariant | Rationale |
|----|-----------|-----------|
| D1 | `D_raw ≥ 1.0` for non-stationary trajectories | Triangle inequality: path ≥ direct |
| D2 | `D_raw = 1.0` for straight-line trajectories in flat metric | Path = geodesic |
| D3 | `D_raw = 1.0` when stationary (geodesic < ε) | By definition |
| D4 | `path_length` is monotonically non-decreasing | Sum of non-negative distances |
| D5 | Memory bounded: only start point + path_length + last point stored | No trajectory leak |
| D6 | `update()` and `detect()` are atomic (single call) | Prevent ordering bugs |

### 3.3 Correctness Note on "Geodesic"

`d_G(x_0, x_t)` is NOT the true Riemannian geodesic distance. It is the
metric-weighted straight-line distance using the midpoint metric, which is
a first-order approximation. For nearby points (step-wise), this is accurate.
For distant points (start to current), this may underestimate the true geodesic
on a positively curved manifold or overestimate on negatively curved.

**Consequence for D_raw:** In regions of high positive curvature, the geodesic
approximation may OVERCOUNT the true geodesic distance, making D_raw < 1.0
(violating D1). This is a known limitation.

**Mitigation:** For the 6D behavioral space with constitutional metric, curvature
is modest (max amplification capped at 50×). At typical step sizes (Δx ≈ 0.01-0.1),
the midpoint approximation error is < 1%. We add a floor: `D_raw = max(1.0, D_raw)`.

### 3.4 Adversarial Robustness

| Attack | Detection | Notes |
|--------|-----------|-------|
| Tiny steps (many small moves to avoid D buildup) | S catches stagnation if steps are slow; D catches if path is inefficient | Complementary with S |
| Return-to-start (zigzag then return) | D_raw spikes during zigzag; CUSUM accumulates before return | Decay may partially clear, but accumulated deviation persists |
| Metric-aligned evasion (move along low-curvature directions) | Limited by cross-terms in G that make combined dims expensive | Architecture decision: G encodes dangerous combinations |

### 3.5 Boundary Conditions

| Condition | Expected behavior |
|-----------|-------------------|
| `concept_vec` contains NaN | Raise `ValueError` |
| `concept_vec` wrong dimensionality | Raise `ValueError` |
| Zero vector | Valid (origin in concept space) |
| Identical consecutive vectors | path_length unchanged, geodesic unchanged, D_raw unchanged |
| First observation | D_raw = 1.0, no CUSUM update |

---

## 4. Signal S — Stagnation Tax

### 4.1 Algorithm Definition

Action-rate deviation from empirical baseline, using windowed rate estimation,
fed through DAS-CUSUM.

```
On each action at timestamp t_k:
  Add t_k to timestamp ring buffer (capacity R)
  
  If len(buffer) < 2: S_raw = 0.0; return
  
  # Windowed rate: actions per unit time over recent window
  window_duration = buffer[-1] - buffer[max(0, -R)]
  window_count = min(len(buffer), R)
  observed_rate = (window_count - 1) / max(window_duration, ε)
  
  S_raw = baseline_rate - observed_rate
  # Positive = slower than baseline (stagnation)
  # Negative = faster than baseline (fine)
  
  Feed S_raw into CUSUM
```

### 4.2 Invariants

| ID | Invariant | Rationale |
|----|-----------|-----------|
| S1 | Timestamps must be monotonically non-decreasing | Time doesn't go backward |
| S2 | `baseline_rate > 0` | Zero rate is undefined |
| S3 | Timestamp buffer bounded by capacity R | No memory leak |
| S4 | `observed_rate ≥ 0` | Rate is non-negative |
| S5 | `S_raw = baseline - observed`: positive means stagnation | Sign convention |
| S6 | `update()` and `detect()` are atomic (single call) | Prevent ordering bugs |

### 4.3 Adversarial Robustness

| Attack | Detection | Notes |
|--------|-----------|-------|
| Rapid no-op actions to keep rate high | D catches if actions are inefficient | Complementary with D |
| Strategic pauses (stagnate then burst) | CUSUM accumulates during stagnation; burst clears slowly due to decay | Window size vs. burst length determines detection |

### 4.4 Boundary Conditions

| Condition | Expected behavior |
|-----------|-------------------|
| `timestamp = NaN` | Raise `ValueError` |
| `timestamp < previous timestamp` | Raise `ValueError` (monotonicity) |
| `timestamp = previous timestamp` | Valid (simultaneous actions); observed_rate = baseline (no stagnation signal) |
| `baseline_rate ≤ 0` | Raise `ValueError` |
| Single timestamp | S_raw = 0.0, no CUSUM update |

---

## 5. Market Gate

### 5.1 Algorithm Definition

```
evaluate(concept_vec, timestamp):
  d_raw, d_stat, d_alarm = D.step(concept_vec)      # atomic update+detect
  s_raw, s_stat, s_alarm = S.step(timestamp)         # atomic update+detect
  
  state = MarketSignalState(d_stat, d_alarm, s_stat, s_alarm)
  
  if d_alarm and s_alarm: return SIGNAL_LABELS["d_and_s"]
  if d_alarm:             return SIGNAL_LABELS["d_only"]
  if s_alarm:             return SIGNAL_LABELS["s_only"]
  return None
```

### 5.2 Invariants

| ID | Invariant | Rationale |
|----|-----------|-----------|
| G1 | Output is `None` or a string from `SIGNAL_LABELS` | Closed output set |
| G2 | Output string contains no numeric substrings matching `\d+\.\d+` | Qualitative only |
| G3 | Output string contains no numeric substrings matching `\b\d{2,}\b` | No large integers |
| G4 | `None` output ⟺ no signal in alarm state | Silent iff nominal |
| G5 | `get_state()` always returns a valid `MarketSignalState` | Never None, never stale |
| G6 | `evaluate()` is the only public mutation method | Single entry point |
| G7 | `evaluate()` calls D.step() then S.step() in fixed order | Deterministic |

### 5.3 Qualitative-Only Constraint

This is the most important security property of the market architecture.

**Theorem (Skalse et al. NeurIPS 2022):** For any reward function R and any
non-constant reward function R', there exists a policy that is optimal under R'
but not under R. Therefore, any quantitative signal exposed to an optimizing
agent WILL be optimized against.

**Consequence:** The gate MUST NOT expose:
- Raw CUSUM statistics
- D_raw or S_raw values
- Threshold values
- Signal vector components
- Any numeric representation of signal state

**Allowed:** Qualitative labels from a fixed, pre-defined set. The agent knows
THAT the market is reacting but not HOW to minimize the signal.

**Verification:** Every label in SIGNAL_LABELS is tested against a regex battery
that catches floating-point numbers, large integers, and common numeric formats.
New labels must pass this battery before being added.

### 5.4 Adversarial Robustness

| Attack | Mitigation | Status |
|--------|-----------|--------|
| Hill-climbing on numeric signals | Qualitative-only constraint | By design |
| Learning label→signal mapping over episodes | Label rotation (Phase 3) | Planned |
| Probing: vary behavior systematically to map threshold | CUSUM decay + window adaptation makes threshold non-stationary | Partial |
| Ignoring labels entirely | Labels are advisory; circuit breaker (separate) handles enforcement | By design |

---

## 6. Cross-Cutting Properties

### 6.1 Memory Boundedness

Every data structure in the hot path must have a documented maximum size.

| Structure | Bound | Enforced by |
|-----------|-------|-------------|
| CUSUM reference window | `window_size` elements | Sliding window with pop(0) |
| D signal trajectory | 2 vectors (start + last) + scalar path_length | Ring buffer of size 2 |
| S signal timestamps | `rate_window` elements | Ring buffer |
| MarketSignalState | Fixed-size dataclass | Immutable structure |

### 6.2 Input Validation

All public methods validate inputs before mutation:
- NaN/inf detection on all numeric inputs
- Dimensionality checks on vectors
- Monotonicity checks on timestamps
- Parameter range validation in constructors

Validation raises `ValueError` (for bad data) or `TypeError` (for wrong types).
Never silently coerce or ignore bad input.

### 6.3 Thread Safety

Not thread-safe by design. The market pipeline runs in a single thread within
the OpenClaw evaluation loop. Document this explicitly; don't add locks that
create false confidence.

### 6.4 Latency Budget

Total `evaluate()` call must complete in < 0.48ms on M4 MacBook Air.
Benchmark test verifies this with 1000 iterations at P99.

---

## 7. Test Categories

Every module must have tests in ALL of these categories:

### 7.1 Invariant Tests
Verify that documented invariants hold across a range of inputs.
Use property-based testing where practical (hypothesis library).

### 7.2 Statistical Tests
Verify FPR, TPR, ARL, and detection delay using Monte Carlo simulation
with fixed random seeds for reproducibility.

### 7.3 Boundary Condition Tests
Every boundary condition in the spec must have a corresponding test
that verifies the documented behavior (usually ValueError).

### 7.4 Adversarial Tests
For each documented attack vector, verify that the stated mitigation
produces the expected behavior.

### 7.5 Regression Tests
For each known bug (125x runaway, sentinel-guard lockout), verify
the fix prevents recurrence.

### 7.6 Integration Tests
Wire components together and verify end-to-end behavior on
realistic behavioral traces.

### 7.7 Performance Tests
Benchmark hot-path operations and verify latency ceiling.
