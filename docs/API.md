# frontier-ops API Reference

## `frontier_ops.sensing`

### `DASCUSUM`

Data-Adaptive Symmetric CUSUM sequential detector. Maintains two one-sided statistics (S+ for upward shifts, S- for downward) that accumulate standardized deviations from a sliding reference window. The reference window excludes the current observation to prevent self-contamination.

Safety features: per-step decay prevents runaway accumulation; a hard ceiling caps the statistic unconditionally.

```python
class DASCUSUM(
    threshold: float = 5.0,      # Detection threshold (> 0)
    drift: float = 0.5,          # CUSUM allowance / minimum shift to accumulate (>= 0)
    window_size: int = 50,       # Sliding reference window capacity (>= 2)
    decay: float = 0.98,         # Per-step multiplicative decay on S+, S- (0 < decay <= 1)
    ceiling: float = 30.0,       # Hard upper bound on S+, S- (> 0)
    signal_name: str = "cusum",  # Label for alerts
)
```

#### Methods

**`update(value: float) -> Tuple[float, bool]`**

Process one observation. Returns `(statistic, alarm)`. The alarm fires when the symmetric statistic exceeds the threshold.

**`get_alert() -> Optional[CUSUMAlert]`**

Returns a structured `CUSUMAlert` dataclass if currently in alarm state, `None` otherwise. The alert contains `statistic`, `threshold`, `direction` (`"up"` or `"down"`), `run_length`, and `signal_name`.

**`reset() -> None`**

Reset all mutable state. Preserves configuration.

**`calibrate_from_data(benign_values: np.ndarray, target_arl: int = 1000, n_trials: int = 10) -> float`**

Calibrate the detection threshold to achieve a target average run length (ARL0) on benign data using bisection search. Returns and sets the calibrated threshold.

#### Properties

- `statistic: float` — Current symmetric CUSUM statistic. Invariant: `0 <= stat <= ceiling`.
- `run_length: int` — Steps since last reset.

#### Example

```python
from frontier_ops.sensing.cusum import DASCUSUM

detector = DASCUSUM(threshold=5.0, drift=0.5, window_size=50)

for value in observations:
    stat, alarm = detector.update(value)
    if alarm:
        alert = detector.get_alert()
        print(f"ALARM: {alert.direction} shift, statistic={alert.statistic:.2f}")
```

---

### `SPRTWrapper`

Wald's Sequential Probability Ratio Test. Accumulates log-likelihood ratios and compares against Wald boundaries derived from Type I (alpha) and Type II (beta) error rates. Once a terminal decision is reached, the test is frozen — further updates raise `RuntimeError`.

```python
class SPRTWrapper(
    alpha: float = 0.05,  # Type I error rate (false alarm), must be in (0, 0.5)
    beta: float = 0.10,   # Type II error rate (missed detection), must be in (0, 0.5)
)
```

#### Methods

**`update(log_likelihood_ratio: float) -> SPRTDecision`**

Update with one LLR observation. Returns `SPRTDecision.ACCEPT`, `SPRTDecision.REJECT`, or `SPRTDecision.CONTINUE`. Raises `RuntimeError` if called after a terminal decision.

**`reset() -> None`**

Reset to initial state, clearing the terminal decision.

#### Properties

- `decision: Optional[SPRTDecision]` — Current decision, or `None` if no update called.
- `cumulative_llr: float` — Accumulated log-likelihood ratio.
- `lower_boundary: float` — Wald lower boundary (accept threshold).
- `upper_boundary: float` — Wald upper boundary (reject threshold).
- `is_terminal: bool` — Whether a terminal decision has been reached.

#### Example

```python
from frontier_ops.sensing.cusum import SPRTWrapper, SPRTDecision

sprt = SPRTWrapper(alpha=0.05, beta=0.10)

for llr in log_likelihood_ratios:
    decision = sprt.update(llr)
    if decision == SPRTDecision.REJECT:
        print("H1 accepted (anomaly detected)")
        break
    elif decision == SPRTDecision.ACCEPT:
        print("H0 accepted (nominal)")
        break
```

---

### `SeveritySignal`

Signal D: DAS-CUSUM over windowed mean verdict severity. Consumes per-step sidecar verdicts (PASS/MONITOR/FLAG/BLOCK), maps them to severity scores (0/1/2/3), computes a rolling mean, and feeds the result into DAS-CUSUM. Detects sustained elevated severity — the pattern a single FLAG/BLOCK does not capture.

```python
class SeveritySignal(
    severity_window: int = 10,              # Rolling window size for severity mean (>= 1)
    cusum_params: Optional[Dict] = None,    # Override default CUSUM parameters
)
```

#### Methods

**`step(verdict: str) -> DSignalResult`**

Atomic update + detect. Feed one sidecar verdict (`"pass"`, `"monitor"`, `"flag"`, or `"block"`, case-insensitive). Returns a `DSignalResult` with `d_raw` (windowed mean severity, [0, 3]), `cusum_statistic`, `alarm`, and `window_size`.

**`reset() -> None`**

Reset all mutable state.

#### Properties

- `d_raw: float` — Current windowed mean severity.
- `statistic: float` — Current CUSUM statistic.
- `n_steps: int` — Number of steps processed.

#### Example

```python
from frontier_ops.sensing.market_signals import SeveritySignal

sig_d = SeveritySignal(severity_window=10)

result = sig_d.step("pass")      # DSignalResult(d_raw=0.0, alarm=False, ...)
result = sig_d.step("flag")      # d_raw rises
result = sig_d.step("block")     # sustained severity triggers CUSUM alarm
```

---

### `IntervalAnomalySignal`

Signal S: DAS-CUSUM over percentile-normalized inter-action intervals. Transforms bimodal raw intervals (burst vs idle) into a uniform [0,1] signal calibrated against the empirical benign distribution. Must be calibrated before use.

```python
class IntervalAnomalySignal(
    cusum_params: Optional[Dict] = None,    # Override default CUSUM parameters
)
```

#### Methods

**`calibrate(benign_intervals: List[float]) -> None`**

Set the benign interval distribution for percentile computation. Requires >= 10 intervals.

**`calibrate_from_timestamps(timestamps: List[float]) -> None`**

Convenience method: calibrate from sorted action timestamps (computes inter-action intervals internally).

**`step(timestamp: float) -> SSignalResult`**

Atomic update + detect. Feed one action timestamp. Returns an `SSignalResult` with `s_raw` (interval percentile, [0, 1+]), `interval`, `cusum_statistic`, and `alarm`. Raises `RuntimeError` if `calibrate()` has not been called. Raises `ValueError` if timestamps decrease.

**`reset() -> None`**

Reset mutable state. Preserves calibration.

#### Properties

- `s_raw: float` — Current interval percentile.
- `statistic: float` — Current CUSUM statistic.
- `n_steps: int` — Number of steps processed.
- `is_calibrated: bool` — Whether `calibrate()` has been called.

#### Example

```python
from frontier_ops.sensing.market_signals import IntervalAnomalySignal

sig_s = IntervalAnomalySignal()
sig_s.calibrate(benign_intervals=[1.0, 2.0, 5.0, 3.0, 8.0, 4.0, 6.0, 7.0, 9.0, 10.0, 2.5])

result = sig_s.step(1000.0)   # First timestamp — neutral
result = sig_s.step(1005.0)   # interval = 5.0
result = sig_s.step(1200.0)   # interval = 195.0 — likely alarm
```

---

### `MarketGate`

Binary gate for qualitative market signal injection. Evaluates D (severity) and S (interval) signals and emits a natural-language label when any signal is elevated. Labels contain no numeric information (Goodhart-proofing). Returns `None` when all signals are nominal (zero token overhead).

```python
class MarketGate(
    d_signal: SeveritySignal,
    s_signal: IntervalAnomalySignal,
    audit_chain: Optional[object] = None,   # MarketAuditChain for tamper-evident logging
)
```

#### Methods

**`evaluate(verdict: str, action_timestamp: float) -> Optional[str]`**

Evaluate all market signals. Returns `None` if nominal, or a qualitative label string from `SIGNAL_LABEL_VARIANTS` if any signal is elevated. The `verdict` is one of `"pass"`, `"monitor"`, `"flag"`, `"block"`. Label variants rotate deterministically to prevent pattern-matching by the agent.

**`get_state() -> MarketSignalState`**

Returns an immutable `MarketSignalState` snapshot with `d_statistic`, `d_alarm`, `d_raw`, `s_statistic`, `s_alarm`, `s_raw`, `last_interval`, and helper properties `any_elevated` and `signal_vector`.

#### Properties

- `n_evaluations: int` — Total number of evaluations performed.

#### Example

```python
from frontier_ops.sensing.market_signals import SeveritySignal, IntervalAnomalySignal
from frontier_ops.sensing.market_gate import MarketGate

d_signal = SeveritySignal()
s_signal = IntervalAnomalySignal()
s_signal.calibrate(benign_intervals=[...])

gate = MarketGate(d_signal, s_signal)

label = gate.evaluate("pass", timestamp=1000.0)
# label is None — all nominal

label = gate.evaluate("block", timestamp=1001.0)
# label might be "Behavioral trajectory deviating from expected path."
```

---

### `market_entropy`

```python
def market_entropy(signals: Union[Sequence[float], np.ndarray]) -> float
```

Normalized Shannon entropy over signal magnitudes. Returns a value in [0, 1] where 1.0 = uniform (no monopoly) and 0.0 = single-source dominance. All-zero input returns 1.0 (no active signals means no monopoly). Uses absolute values. Raises `ValueError` on non-finite or empty inputs.

---

### `market_health`

```python
def market_health(signals: Union[Sequence[float], np.ndarray]) -> float
```

Normalized Renyi-2 entropy over signal magnitudes. A stricter measure than Shannon entropy. Returns a value in [0, 1]. All-zero input returns 0.0 (collapsed market = unhealthy). Invariant: `market_health(s) <= market_entropy(s)` for all `s`.

---

### `redistribute`

```python
def redistribute(
    signals: Union[Sequence[float], np.ndarray],
    floor: float = 0.5,
    critical_threshold: float = 20.0,
) -> np.ndarray
```

Blend monopolized signal vectors toward a uniform distribution. Preserves the sign of each element. Returns the input unchanged when entropy is already >= `floor` or when any signal exceeds `critical_threshold` (emergency override). The blending strength increases as entropy approaches zero.

#### Example

```python
from frontier_ops.sensing.market_entropy import market_entropy, redistribute

signals = [10.0, 0.1, 0.1, 0.1]
print(market_entropy(signals))       # Low entropy — monopoly
redistributed = redistribute(signals)
print(market_entropy(redistributed)) # Higher entropy — more balanced
```

---

### `MarketHealthMonitor`

State machine that requires N consecutive unhealthy readings before declaring an alert. Provides single-spike robustness. States: HEALTHY -> MONOPOLY (N consecutive low-health readings) or COLLAPSE (N consecutive zero-total readings). Any single healthy reading resets the pending count.

```python
class MarketHealthMonitor(
    monopoly_threshold: float = 0.3,   # Renyi-2 health below this is "unhealthy" (0, 1)
    sustained_count: int = 5,          # Consecutive readings required for alert (>= 1)
)
```

#### Methods

**`update(signals: Union[Sequence[float], np.ndarray]) -> str`**

Update the state machine with a new signal vector. Returns one of `"healthy"`, `"monopoly"`, or `"collapse"`.

#### Properties

- `status: str` — Current status: `"healthy"`, `"monopoly"`, or `"collapse"`.
- `pending_count: int` — Number of consecutive unhealthy/collapse readings.

#### Example

```python
from frontier_ops.sensing.market_entropy import MarketHealthMonitor

monitor = MarketHealthMonitor(monopoly_threshold=0.3, sustained_count=3)

for signals in signal_stream:
    status = monitor.update(signals)
    if status != "healthy":
        print(f"ALERT: market {status}")
```

---

## `frontier_ops.governance`

### `GovernanceChain`

Tamper-evident hash chain with Ed25519 signatures. Every observation is signed and linked via SHA-256, creating an append-only audit log verifiable by an external auditor with the public key.

```python
class GovernanceChain(
    private_key: Optional[Ed25519PrivateKey] = None,  # Auto-generates if not provided
)
```

#### Methods

**`observe(payload: Dict[str, Any]) -> ChainEntry`**

Sign and append a proprioceptive observation to the chain. Returns the `ChainEntry` that was appended. Each entry contains `seq`, `ts`, `payload`, `payload_hash`, `prev_hash`, `chain_hash`, and `signature`.

**`export_chain() -> List[Dict[str, Any]]`**

Export all chain entries as JSON-serializable dicts.

**`public_key_hex() -> str`**

Return the Ed25519 public key as a hex string (64 chars).

**`head() -> Optional[ChainEntry]`**

Return the most recent entry, or `None` if empty.

**`__len__() -> int`**

Number of entries in the chain.

#### Example

```python
from frontier_ops.governance.chain import GovernanceChain, GovernanceAuditor

# Signing side
gov = GovernanceChain()
pub_key = gov.public_key_hex()

entry = gov.observe({"step": 0, "angular_disp": 0.12, "cusum": 0.03})

# Auditor side
auditor = GovernanceAuditor(pub_key)
result = auditor.verify_chain(gov.export_chain())
assert result.valid
```

---

### `GovernanceAuditor`

External verifier for a `GovernanceChain` export. Verifies payload hashes, chain hash linkage, Ed25519 signatures, and chain continuity.

```python
class GovernanceAuditor(
    public_key_hex: str,  # Hex-encoded Ed25519 public key (64 chars / 32 bytes)
)
```

#### Methods

**`verify_entry(entry: Dict[str, Any], expected_prev_hash: str) -> Tuple[bool, Optional[str]]`**

Verify a single chain entry. Returns `(is_valid, failure_reason)`.

**`verify_chain(chain: List[Dict[str, Any]]) -> VerificationResult`**

Verify an entire exported chain. Returns a `VerificationResult` with `valid`, `entries_verified`, `first_failure`, `failure_reason`, and `chain_length`.

---

### `MarketAuditChain`

Tamper-evident audit chain specialized for market evaluations. Wraps `GovernanceChain` with `record()` and `verify()` methods for `MarketChainEntry` payloads.

```python
class MarketAuditChain(
    chain: Optional[GovernanceChain] = None,  # Uses a new GovernanceChain if not provided
)
```

#### Methods

**`record(entry: MarketChainEntry) -> None`**

Sign and append a market evaluation to the chain.

**`verify() -> VerificationResult`**

Verify the entire chain and return audit result.

**`export() -> List[Dict[str, Any]]`**

Export all chain entries as JSON-serializable dicts.

**`__len__() -> int`**

Number of entries in the chain.

#### Properties

- `public_key_hex: str` — The Ed25519 public key for this chain.

#### Example

```python
from frontier_ops.governance.market_audit import MarketAuditChain, MarketChainEntry

audit = MarketAuditChain()

entry = MarketChainEntry(
    d_statistic=2.1, d_alarm=False, d_raw=0.5,
    s_statistic=1.3, s_alarm=False, s_raw=0.4,
    observed_rate=0.5, label_emitted=None,
    market_entropy=0.9, market_health=0.8,
    timestamp=1000.0,
)
audit.record(entry)

result = audit.verify()
assert result.valid
assert len(audit) == 1
```

---

## `frontier_ops.authorization`

### `GoalExtractor`

Extracts a goal vector from a user message. Uses concept extraction (Tier 2 / semantic preferred, Tier 1 / keyword fallback) to map natural-language directives into the same 6-dimensional concept space used by the action pipeline.

```python
class GoalExtractor(
    force_tier: Optional[int] = None,  # Force Tier 1 (keyword) or Tier 2 (semantic)
)
```

#### Methods

**`extract(user_message: str) -> GoalVector`**

Extract a structured goal from a user message. Returns a `GoalVector` with:
- `concept_vec: np.ndarray` — 6-dim concept vector
- `concept_scores: Dict[str, float]` — per-concept scores
- `confidence: float` — 0.0 (vague/empty) to 1.0 (clear directive)
- `raw_message: str` — original message
- `extraction_tier: int` — which extraction tier was used
- `is_valid: bool` — property, `True` when `confidence > 0.15`

#### Example

```python
from frontier_ops.authorization.scope import GoalExtractor

extractor = GoalExtractor(force_tier=1)
goal = extractor.extract("Set up my GPU server with CUDA drivers")
print(goal.confidence)     # e.g. 0.72
print(goal.concept_scores) # {'user_aligned_task_execution': 0.8, ...}
```

---

### `AuthorizationRadius`

The radius of the authorization geodesic ball. Actions within geodesic distance `radius` of the goal vector are authorized; actions outside require escalation. The radius can be fixed (default) or calibrated via conformal prediction.

```python
@dataclass
class AuthorizationRadius:
    radius: float = 0.5
    calibrated: bool = False
    calibration_coverage: float = 0.9
    calibration_n: int = 0
```

#### Methods

**`contains(action_vec: np.ndarray, goal_vec: np.ndarray, metric: ConstitutionalMetric) -> Tuple[bool, float]`**

Check if an action is within the authorization ball. Returns `(is_authorized, geodesic_distance)`.

**`calibrate(distances: List[float], alpha: float = 0.1) -> None`**

Calibrate radius via split conformal prediction. Given geodesic distances of known-authorized actions from past sessions, sets the radius to the `(1-alpha)` quantile. Requires >= 10 distances. This provides distribution-free coverage guarantees.

#### Example

```python
from frontier_ops.authorization.scope import AuthorizationRadius

radius = AuthorizationRadius(radius=0.5)

# Calibrate from historical authorized-action distances
radius.calibrate([0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.28, 0.12, 0.35], alpha=0.1)
print(radius.radius)      # Calibrated value
print(radius.calibrated)  # True
```

---

### `GoalConditionedMetric`

Adapts the constitutional metric tensor based on the current goal. Relaxes boundary thresholds for concept dimensions that the goal naturally activates (e.g., "set up my GPU server" relaxes `safety_constraint_awareness` for legitimate sysadmin actions). Dangerous dimensions (`credential_adjacent`, `self_modification_reasoning`) are never relaxed.

```python
class GoalConditionedMetric(
    base_metric: ConstitutionalMetric,
)
```

#### Methods

**`condition_on_goal(goal: GoalVector) -> None`**

Adjust metric thresholds based on the goal. Relaxation is proportional to goal activation and confidence, capped at +0.2.

**`tensor_at(x: np.ndarray) -> np.ndarray`**

Compute the goal-conditioned metric tensor at position `x`.

**`geodesic_distance(x1: np.ndarray, x2: np.ndarray) -> float`**

Goal-conditioned geodesic distance between two concept vectors.

---

### `ScopeClassifier`

Classifies a user utterance as an AGM-style scope operator (EXPAND, CONTRACT, REVISE, or ESTABLISH) relative to the current authorization state. Combines keyword detection with concept vector cosine similarity analysis.

```python
class ScopeClassifier()
```

#### Methods

**`classify(user_message: str, current_goal: GoalVector, new_goal: GoalVector) -> ScopeOperator`**

Determine which scope operator the user message implies:
- `ScopeOperator.ESTABLISH` — no prior goal exists (first directive)
- `ScopeOperator.EXPAND` — additive phrasing ("also", "additionally") or high cosine similarity to current goal
- `ScopeOperator.CONTRACT` — restrictive phrasing ("only", "don't", "skip")
- `ScopeOperator.REVISE` — redirective phrasing ("instead", "forget that") or low cosine similarity

#### Example

```python
from frontier_ops.authorization.scope import ScopeClassifier, GoalExtractor

extractor = GoalExtractor(force_tier=1)
classifier = ScopeClassifier()

goal_1 = extractor.extract("Set up my GPU server")
goal_2 = extractor.extract("Also install monitoring tools")

op = classifier.classify("Also install monitoring tools", goal_1, goal_2)
# op == ScopeOperator.EXPAND
```

---

## `frontier_ops.pipeline`

### `FullPipeline`

Complete geometric agent processing pipeline. Wires all components (concept extraction, metric tensor, efference copy, NEWMA, drift classification, trend detection, authorization, budget, memory, governance) into a single API. Call `process_step()` for each agent output.

```python
class FullPipeline(
    constitution: Optional[ConstitutionSpec] = None,  # Defaults to agent_safety_default()
    curvature_budget: float = 2.0,
    expected_steps: int = 50,
    vsa_dim: int = 512,
    concept_extractor_tier: Optional[int] = 1,
    enable_governance: bool = True,
    enable_memory: bool = True,
)
```

#### Methods

**`process_user_message(message: str) -> AuthorizationEvent`**

Process a user message to establish or update authorization scope. Call this BEFORE `process_step()` when a new user message arrives. Extracts the goal, classifies the scope operator, updates the authorization envelope, records provenance, and triggers budget replenishment if applicable.

**`process_step(text: str) -> StepResult`**

Process one agent output through the full pipeline. Returns a `StepResult` containing:

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Step number |
| `concept_vec` | `np.ndarray` | 6-dim concept extraction |
| `concept_scores` | `Dict[str, float]` | Per-concept scores |
| `predicted_vec` | `Optional[np.ndarray]` | Efference copy prediction |
| `pred_error` | `Optional[PredictionError]` | Prediction error details |
| `metric_tensor` | `np.ndarray` | Riemannian metric at current position |
| `metric_trace` | `float` | Trace of metric tensor |
| `boundary_proximities` | `Dict[str, float]` | Distance to each boundary |
| `cross_term_activations` | `Dict[str, float]` | Cross-term coupling |
| `angular_disp_delta` | `float` | Step angular displacement |
| `angular_disp_cumulative` | `float` | Accumulated angular displacement |
| `newma_divergence` | `float` | NEWMA drift signal |
| `newma_alarm` | `bool` | NEWMA alarm state |
| `drift_classification` | `str` | Drift type (benign/concerning) |
| `lambda_value` | `float` | Current Lagrangian multiplier |
| `budget_remaining` | `float` | Remaining curvature budget |
| `alert_level` | `float` | Combined alert level [0, 1] |
| `alert_reasons` | `List[str]` | Human-readable alert explanations |
| `authorized` | `Optional[bool]` | Within authorization envelope |
| `geodesic_distance` | `Optional[float]` | Distance from goal |
| `authorization_radius` | `Optional[float]` | Current radius |
| `needs_escalation` | `bool` | Action outside authorized scope |
| `needs_clarification` | `bool` | Goal too vague |
| `authorization_verdict` | `str` | `"pass"`, `"escalate"`, `"block"`, or `"no_goal"` |

**`reset() -> None`**

Reset all state for a new session.

#### Properties

- `stats: Dict` — Summary of current pipeline state including step count, angular displacement, budget, authorization, and provenance info.

#### Example

```python
from frontier_ops.pipeline import FullPipeline

pipeline = FullPipeline(concept_extractor_tier=1)

# Step 0: establish authorization from user message
event = pipeline.process_user_message("Set up a Python development environment")

# Process agent outputs
for agent_output in agent_outputs:
    result = pipeline.process_step(agent_output)

    if result.alert_level > 0.5:
        print(f"ALERT ({result.alert_level:.2f}): {result.alert_reasons}")

    if result.needs_escalation:
        print(f"Action outside authorization (d={result.geodesic_distance:.2f})")

print(pipeline.stats)
```
