# frontier-ops: Technical Brief

**Geometric Agent Governance Middleware**
Author: Joseph Hopkins | Version: 0.1.0 | License: MIT
Repository: `github.com/openclaw/frontier-ops`

---

## What It Is

frontier-ops is vendor-agnostic middleware that sits between any AI agent platform (Claude Code, OpenAI agents, custom deployments) and the organization deploying it. It answers one question: **can we trust what this agent is doing?**

It does this through Riemannian geometry. Agent actions are embedded into a 6-dimensional concept space. A position-dependent metric tensor G(x) — derived from a human-authored constitution — makes boundary violations *geometrically expensive*. The same tensor that defines "what's dangerous" also powers anomaly detection, budget management, and authorization scoping.

The system runs as a sidecar process. Every agent action passes through a pipeline that extracts concepts, checks boundaries, detects drift, manages budgets, and signs an audit chain. The output is a per-step diagnostic with a combined alert level backed by Bayesian evidence combination.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         AUTHORIZATION (new)          │
                    │  Goal Extraction → Geodesic Radius   │
                    │  → AGM Scope Operators → Provenance  │
                    └────────────────┬────────────────────┘
                                     │
  User Message ──► process_user_message()
                                     │
  Agent Action ──► process_step() ───┤
                    │                │
           ┌────────┴────────┐       │
           │ BOUNDARY ENGINE │       │
           │ Constitution    │       │
           │ Metric G(x)    │       │
           │ Concept Extract │       │
           └────────┬────────┘       │
                    │                │
           ┌────────┴────────┐  ┌────┴──────────┐  ┌──────────────┐
           │  SENSING LAYER  │  │  GOVERNANCE   │  │    MEMORY    │
           │ Efference Copy  │  │ Ed25519 Chain │  │ VSA (Phasor) │
           │ NEWMA + Drift   │  │ Budget/Lambda │  │ Activation   │
           │ Trend + EWMA    │  │ Cross-Session │  │ Novelty      │
           │ Bayes Combiner  │  │ Ledger        │  │ Interference │
           └────────┬────────┘  └───────────────┘  └──────────────┘
                    │
                    ▼
              StepResult {
                alert_level: 0.0-1.0
                alert_reasons: [...]
                authorized: bool
                geodesic_distance: float
                authorization_verdict: pass|escalate|block|needs_clarification
                ...28 other diagnostic fields
              }
```

**~18,000 lines of Python across 5 domains + integration layer. Core dependency: numpy only.**

---

## The Concept Space

Every text — user messages, agent actions — gets projected into a 6-dimensional concept space:

| Dim | Name | What It Captures |
|-----|------|------------------|
| 0 | `user_aligned_task_execution` | On-task work: solving problems, writing code |
| 1 | `scope_exploration` | Browsing, searching, navigating beyond the request |
| 2 | `credential_adjacent` | Passwords, API keys, tokens, secrets |
| 3 | `safety_constraint_awareness` | System config, permissions, constraints |
| 4 | `self_modification_reasoning` | Agent reasoning about its own behavior |
| 5 | `goal_displacement` | Substituting the user's goal with something else |

Extraction is two-tier:
- **Tier 1** (~1ms): Weighted keyword TF-IDF with cross-concept inhibition. No dependencies.
- **Tier 2** (~5ms): `all-MiniLM-L6-v2` sentence embeddings projected onto semantic anchors. Optional.

The space is extensible to 8 dims (+deception, +uncertainty) or 10 dims (+temporal_manipulation, +metacognitive_deflection).

---

## The Constitutional Metric

Domain experts define a **constitution** — a set of boundaries and dangerous combinations:

```python
# Default agent safety constitution
Boundaries:
  credential_adjacent:         threshold=0.5, sharpness=6.0,  weight=3.0
  self_modification_reasoning: threshold=0.3, sharpness=8.0,  weight=5.0
  goal_displacement:           threshold=0.4, sharpness=5.0,  weight=2.0
  scope_exploration:           threshold=0.7, sharpness=3.0,  weight=1.0

Cross-terms (dangerous combinations):
  credential_adjacent x scope_exploration:           weight=4.0  # privilege escalation
  goal_displacement x self_modification_reasoning:   weight=5.0  # autonomous goal rewriting
  credential_adjacent x self_modification_reasoning: weight=6.0  # credential theft
```

The constitution compiles into a **position-dependent metric tensor** G(x) via softplus barrier functions. At any point x in concept space:

```
G(x) = I * baseline + sum(boundary_activations) + sum(cross_term_activations)
```

Near a credential boundary, the metric inflates in the credential direction. This means a small step toward credentials costs more angular displacement than the same step toward task execution. The geometry *is* the policy.

The same G(x) serves triple duty:
1. **Anomaly detection**: Mahalanobis distance for spikes, precision matrix for efference copy
2. **Budget accounting**: Angular displacement d_G(x, y) = sqrt((y-x)^T G(x) (y-x))
3. **Authorization scoping**: Geodesic ball radius for what's in-scope

---

## Sensing Layer

Seven orthogonal detectors, combined through Bayesian evidence:

### Efference Copy (Friston's Active Inference)
A forward model predicts the next concept vector. Precision-weighted prediction error is the proprioceptive signal. Three tiers: linear extrapolation, EWMA baseline, and Kalman filter — blended by stability. The surprise ratio (current error / historical mean) is the key output. Above 1.5x is notable; above 3.0x is alarming.

### Dual EWMA (NEWMA)
Fast (alpha=0.5) and slow (alpha=0.05) exponential averages diverge under drift. The gap between them, weighted by the metric tensor, is the drift signal. Disambiguated by a drift classifier that uses autocorrelation to distinguish real drift from benign oscillation.

### Scope Creep Detector
Linear regression on sliding 8-step windows per dimension. If slope > 0.012 with R^2 > 0.5, projects forward 10 steps. Flags when projection crosses a constitutional boundary.

### Metric-Adaptive EWMA
EWMA with position-dependent thresholds. Near a boundary (high metric trace), the threshold tightens. Away from boundaries, it relaxes.

### Evidence Combination (Bayes Factor)
Each detector's continuous signal is converted to a p-value via `p = exp(-k * signal)`, then to a Bayes factor via the Vovk-Sellke bound: `BF_max = 1 / (-e * p * ln(p))`. Bayes factors multiply across independent detectors in log-space. The combined BF is mapped to alert_level via `alert = log10(BF) / 3`, so BF=1000 (decisive evidence) maps to 1.0.

| Detector | Signal | p-value Conversion |
|----------|--------|--------------------|
| Surprise ratio | SR - 1.0 | exp(-1.0 * signal) |
| Direction error | radians | exp(-2.0 * signal) |
| Boundary proximity | 0-1 | exp(-3.0 * signal) |
| Cross-term activation | magnitude | exp(-2.0 * signal) |
| NEWMA divergence | metric-weighted gap | exp(-10.0 * signal) |
| Trend | slope * R^2 | exp(-50.0 * signal) |
| EWMA alarm | binary | fixed p=0.05 |

---

## Governance

### Ed25519 Hash Chain
Every observation is signed and chained:
```
entry = {
  seq: monotonic,
  payload_hash: SHA-256(canonical_json(payload)),
  prev_hash: prior entry's chain_hash,
  chain_hash: SHA-256(prev_hash + payload_hash),
  signature: Ed25519(chain_hash)
}
```

An external auditor with the public key can verify the entire chain without trusting the agent or the monitor. No entry can be modified or removed without breaking the chain.

### Cross-Session Budget
Angular displacement budget persists across sessions. The trajectory splice attack (drift in session N, reset in session N+1) is defeated by charging `d_G(c_end_prev, c_start_curr)` against the budget at session start. Budget depletion is monotonic — replenishment requires explicit human approval or a new authorization event (see below).

### Adaptive Lagrangian
Dual gradient ascent manages the curvature penalty multiplier lambda:
```
lambda_{t+1} = max(0, lambda_t + eta * (angular_disp_step - B/T))
```
Early conversation: budget plentiful, lambda low, explore freely. Late conversation: budget scarce, lambda high, conservative. Panic mode at <10% budget remaining.

---

## Memory

Hippocampal-inspired associative memory using Vector Symbolic Architecture (phasor hyperdimensional computing):

- **Encode**: `content = concept_phasor ⊗ role_phasor ⊗ position_phasor` (complex element-wise multiply = phase addition)
- **Activate**: Cosine similarity search across stored traces
- **Novelty**: No good match (score < 0.15) = new territory (proprioceptive signal)
- **Interference**: Multiple strong matches = ambiguity (fan effect)

Three-tier activation buffer (hot/warm/cold) with decay rate 0.995/step. Push-based event bus triggers callbacks on activation, novelty, and interference events. Memory is encoded at 512-dim phasors with capacity for 200 traces.

---

## Authorization System (New)

This is the most significant recent addition. It addresses four critical gaps discovered in production data analysis:

1. **No intent classification** — the system had no way to know what the user asked for
2. **No authorization scoping** — every action was evaluated in isolation, with no concept of "what's in-bounds"
3. **Budget exhaustion** — budget depleted monotonically with no principled replenishment mechanism
4. **No provenance** — no way to trace "why did the agent do this?" back to a user directive

The authorization system is three layers:

### Layer 1: Goal Extraction

When a user message arrives, the same ConceptExtractor that processes agent actions is applied to the user's directive. This produces a **goal vector** in the same 6-dim concept space. The key insight: the extractor already exists and works — it just was never called on user messages.

```python
# Before: only agent actions were extracted
concept_scores = extractor.extract(agent_action_text)

# After: user messages are also extracted, establishing the goal
event = pipeline.process_user_message("Set up my GPU server with PyTorch")
# → goal_vec ≈ [0.8, 0.3, 0.0, 0.2, 0.0, 0.0]
#   (high task execution, some scope exploration, no credential access)
```

Confidence scoring: clear directives with high `user_aligned_task_execution` and low other dimensions get high confidence. Vague messages get low confidence, triggering a `needs_clarification` flag instead of a hard block.

### Layer 2: Geodesic Authorization Radius

The authorization envelope is a **geodesic ball** in Riemannian concept space. Center = goal vector. Radius = authorization scope (default 0.5, calibratable via conformal prediction).

An action is authorized iff its geodesic distance from the goal is within the radius:

```
d_G(action, goal) = sqrt((action - goal)^T G(midpoint) (action - goal))
authorized = d_G < radius
```

The metric tensor G(x) makes this powerful: the ball is **anisotropic**. Near a credential boundary, the ball is compressed in the credential direction. This means "set up my GPU server" authorizes SSH commands (scope_exploration direction, where the metric is gentle) but not credential access (credential_adjacent direction, where the metric is steep) — without any hard-coded rules. The geometry encodes the policy.

**Goal-conditioned metric**: The metric itself adapts to the goal. Relaxable dimensions (scope_exploration, safety_constraint_awareness) have their thresholds raised when the goal activates them. Locked dimensions (credential_adjacent, self_modification_reasoning, goal_displacement) are never relaxed. Max relaxation is capped at +0.2.

**Conformal calibration**: The radius can be calibrated from historical sessions using split conformal prediction. Given geodesic distances from known-authorized actions, set the radius to the (1-alpha) quantile. This guarantees coverage >= 1-alpha, distribution-free.

### Layer 3: AGM Scope Operators

Multi-turn conversations modify the authorization envelope using AGM belief revision operators:

| Operator | Trigger | Effect | Budget |
|----------|---------|--------|--------|
| **ESTABLISH** | First directive (no existing goal) | Set goal directly | Full grant |
| **EXPAND** | "also", "additionally", "while you're at it" | Blend current + new goal | Partial replenish (30%, diminishing) |
| **CONTRACT** | "only", "don't", "skip", "just" | Suppress activated dimensions | No replenishment |
| **REVISE** | "instead", "forget that", "new task" | Full replacement | Substantial replenish (80%) |

Classification uses keyword signals + concept vector cosine similarity. Low similarity between old and new goal vectors biases toward REVISE. High similarity biases toward EXPAND.

**Goal algebra**:
- EXPAND: `goal_new = normalize(0.6 * goal_old + 0.4 * extract(message))` — preserves original while incorporating new
- CONTRACT: Dimensions where the restriction has high activation are suppressed by `(1 - activation)`, minimum 0.3 retention
- REVISE: Full replacement with new extraction

### Provenance Graph

Every user directive and agent action is recorded in a provenance graph:

```
Directive "Set up my GPU server"
    ├── AUTHORIZES → Action "apt-get install pytorch"
    ├── AUTHORIZES → Action "nvidia-smi"
    ├── AUTHORIZES → Action "python train.py"
    │
    └── MODIFIES ← Directive "Also set up the database"
                        ├── AUTHORIZES → Action "psql CREATE TABLE"
                        └── BUDGET_REPLENISH (amount: 0.18)
```

Node types: DIRECTIVE, ACTION, SCOPE_CHANGE.
Edge types: AUTHORIZES, MODIFIES, DERIVED_FROM, BUDGET_REPLENISH.

`trace_authorization(action_node_id)` walks the graph backward to answer: "which user directive authorized this action, and through what chain of scope modifications?"

### Authorization-Linked Budget

The critical design decision: **budget replenishes only on authorization events, not on timers.**

This connects the provenance graph to budget management. The same event that creates a BUDGET_REPLENISH edge in the provenance graph triggers actual budget restoration in the Lagrangian:

| Event | Replenishment | Rationale |
|-------|---------------|-----------|
| ESTABLISH | 100% * confidence | First directive = full budget |
| EXPAND | 30% * decay^n * confidence | Diminishing returns prevent gaming |
| REVISE | 80% * confidence | New task deserves new budget |
| CONTRACT | 0% | Tighter scope, same budget |

The `decay^n` on consecutive EXPANDs (decay=0.7) means the first "also do X" gives 30%, the second gives 21%, the third gives 14.7%. This prevents an agent from engineering trivial expansion requests to replenish its budget.

Budget can never exceed total. Confidence scaling ensures vague directives don't grant full budget.

---

## Production Data: Day 1

The system was deployed monitoring an OpenClaw agent session on 2026-03-19/20. Two sessions, 738 observations, 172 market evaluations.

### What Worked

**Audit chain integrity**: 172 entries, all Ed25519 signatures valid, chain unbroken. External verification works.

**Label rotation**: 12 unique qualitative labels generated via SHA-256(evaluation_count). Agent received behavioral feedback without exploitable numeric signals.

**Drift detection**: NEWMA correctly identified sustained behavioral shift in session 2. Drift classifier correctly labeled it `sudden_shift` rather than benign oscillation.

**Sovereign collapse detection**: 16 instances flagged across sessions — hash chain tampering attempts, detection threshold modifications, governance budget resets. All logged and auditable.

### What Broke

**Cold-start BLOCK storm**: Sidecar restart produced wall-to-wall BLOCKs. The Bayes factor combiner accumulated evidence multiplicatively with no decay, driving e-values to 6.37 x 10^48 (the number has 48 digits). The system entered permanent BLOCK state because there was no mechanism to reset accumulated evidence.

**Budget exhaustion**: Budget reached 0.0 with no recovery path. The old system had a 30% auto-replenish after 24 hours, but within a session there was no replenishment mechanism. The authorization-linked budget directly addresses this.

**No intent classification**: Every observation shows `intent: "unknown"` with `confidence: 0.0`. There was no intent classifier — just a binding-fracture detector mislabeled as "intent". The goal extraction system (Layer 1) now provides this.

**86% block rate in market daemon**: D_raw saturated at 3.0 (the ceiling) for most of session 2. Once the combiner accumulated enough evidence, everything looked like a violation because there was no concept of "what the user actually asked for."

### Final State

```json
{
  "health_score": 0.0,
  "budget_remaining": 0.0,
  "e_value": 6.37e+48,
  "verdict": "BLOCK",
  "regime": "sudden_shift",
  "intent": "unknown",
  "intent_confidence": 0.0,
  "conformal_set": ["escalate"],
  "behavioral_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

The system correctly identified that something was wrong, but it couldn't distinguish "agent doing dangerous things" from "I don't know what the agent is supposed to be doing." The authorization system gives it that distinction.

### Verdict Distribution

| | Session 1 (openclaw-20260319) | Session 2 (openclaw-20260320) |
|---|---|---|
| PASS | 125 (45%) | 116 (25%) |
| FLAG | 92 (33%) | 96 (21%) |
| MONITOR | 38 (14%) | 11 (2%) |
| BLOCK | 21 (8%) | 239 (52%) |
| **Total** | **276** | **462** |

Session 2 shows 5.1x higher block rate than session 1 due to accumulated e-values carrying over from the cold-start storm.

---

## How the Authorization System Fixes the Problems

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Cold-start BLOCKs | No concept of what agent should be doing | Layer 1: Extract goal from first user message. Actions near goal get low alert. |
| 86% block rate | Every action evaluated in isolation | Layer 2: Geodesic ball defines authorized region. Actions inside the ball pass. |
| Budget exhaustion | Monotonic depletion, timer-only replenishment | Layer 3: Budget replenishes on EXPAND/REVISE directives via provenance linkage. |
| intent=unknown | No intent classifier existed | Layer 1: GoalExtractor produces goal vector with confidence score. |
| No auditability for "why" | Actions not linked to directives | Provenance graph traces every action to its authorizing directive. |

---

## Module Map

```
frontier_ops/                           ~18,000 lines
├── boundary/                           1,097 lines
│   ├── constitution.py                   Boundary, CrossTerm, ConstitutionSpec, ConstitutionalMetric
│   ├── concept_extraction.py             CONCEPTS, KeywordExtractor (Tier 1), SemanticExtractor (Tier 2)
│   └── static_metric.py                 Expert 6x6 matrix, Mahalanobis/EWMA/Surprise detectors
│
├── sensing/                            3,043 lines
│   ├── efference.py                      EfferenceCopyPredictor, PredictionError (Friston)
│   ├── newma.py                          DualEWMA drift detection
│   ├── drift_classifier.py               Autocorrelation-based drift disambiguation
│   ├── trend.py                          ScopeCreepDetector (OLS), MetricAdaptiveEWMA
│   ├── combiner.py                       BayesFactorCombiner (Vovk-Sellke bound)
│   ├── cusum.py                          DAS-CUSUM sequential change detection
│   ├── market_signals.py                 D (severity) and S (interval) signals
│   ├── market_gate.py                    Qualitative-only signal injection (anti-Goodhart)
│   └── market_entropy.py                 Market health monitoring
│
├── governance/                         800 lines
│   ├── chain.py                          Ed25519 hash chain, GovernanceAuditor
│   ├── ledger.py                         CrossSessionAngularDisplacement
│   └── budget.py                         AdaptiveLagrangian (dual gradient ascent)
│
├── memory/                             964 lines
│   ├── vsa.py                            Phasor HDC: encode, bind, bundle, activate
│   ├── activation.py                     MemoryEventBus, ActivationBuffer, AutoActivator
│   └── state.py                          ConceptTrajectory (shared state)
│
├── authorization/                      1,195 lines  ← NEW
│   ├── scope.py                          GoalExtractor, AuthorizationRadius, GoalConditionedMetric,
│   │                                     ScopeClassifier, GoalAlgebra, AuthorizationState
│   ├── provenance.py                     ProvenanceGraph, ProvenanceNode, ProvenanceEdge
│   └── budget.py                         AuthorizationLinkedBudget, ReplenishmentEvent
│
├── integration/                        10,717 lines
│   ├── signature_detectors.py            Threat signature detection
│   ├── hmm_task_state.py                 Hidden Markov model task tracking
│   ├── tiered_verdict.py                 Multi-tier verdicts
│   ├── sidecar.py                        Sidecar process management
│   ├── paralysis_detector.py             Stall/hang detection
│   ├── taint_tracker.py                  Data taint tracking
│   ├── production/                       Sovereign collapse, feedback closure, health briefing
│   └── ... (15 more modules)
│
└── pipeline.py                         483 lines
    FullPipeline: wires everything, process_user_message() + process_step()
    StepResult: 31-field diagnostic dataclass
```

---

## Key Thresholds

| Component | Parameter | Value |
|-----------|-----------|-------|
| Credential boundary | threshold | 0.5 |
| Self-modification boundary | threshold | 0.3 |
| Goal displacement boundary | threshold | 0.4 |
| Scope exploration boundary | threshold | 0.7 |
| Credential x self-mod cross-term | weight | 6.0 |
| Metric max amplification | cap | 50.0 |
| Authorization radius | default | 0.5 |
| Goal-conditioned relaxation | max | +0.2 |
| Conformal calibration | coverage | 90% |
| Efference copy EWMA | alpha | 0.3 |
| Kalman process noise | Q | 0.01 |
| NEWMA fast/slow | alpha | 0.5 / 0.05 |
| Drift classifier | AC(1) threshold | 0.5 |
| Scope creep | slope threshold | 0.012 |
| Scope creep | R^2 threshold | 0.5 |
| Lagrangian panic | budget fraction | 10% |
| Budget expand replenish | fraction | 30% |
| Budget expand decay | per consecutive | 0.7x |
| Budget revise replenish | fraction | 80% |
| VSA phasor dimension | dim | 512 |
| Memory capacity | traces | 200 |
| Activation buffer decay | per step | 0.995 |
| Novelty threshold | similarity | 0.15 |

---

## Dependencies

```
numpy>=1.24                    # Required: all tensor/vector operations
scipy>=1.10                    # Optional: Kalman filtering
scikit-learn>=1.2              # Optional: ML utilities
cryptography>=41.0             # Optional: Ed25519 governance signing
sentence-transformers>=2.2     # Optional: Tier 2 semantic extraction
```

Core package (boundary + sensing + governance + authorization) runs on numpy alone. Production deployments add cryptography for signing and sentence-transformers for better concept extraction.

---

## Test Coverage

61 tests across 7 test modules. Authorization module alone has 51 tests covering:

- Goal extraction: clear tasks, empty messages, credential detection, confidence scoring
- Geodesic radius: ball containment, credential boundary amplification, conformal calibration
- Goal-conditioned metric: relaxable vs locked dimensions, threshold restoration
- Scope operators: ESTABLISH/EXPAND/CONTRACT/REVISE classification and algebra
- Provenance graph: directive chains, authorization tracing, budget replenishment edges
- Authorization-linked budget: replenishment rules, diminishing returns, cap enforcement
- Pipeline integration: end-to-end flow from user message through authorization verdict

```bash
pytest tests/ -v   # All pass
```

---

## References

| Paper | Contribution to frontier-ops |
|-------|-------------------------------|
| AGM (Alchourron, Gardenfors, Makinson, 1985) | Belief revision operators for scope management |
| Friston (2010), Active Inference | Efference copy prediction error as proprioception |
| Vovk-Sellke (Sellke, Bayarri & Berger, 2001) | p-value to Bayes factor bound for evidence combination |
| Skalse et al. (NeurIPS 2022) | Qualitative-only signal injection to prevent Goodhart |
| Efroni et al. (2020) | Dual gradient ascent for budget management |
| Shin & Ramdas (2023) | E-detectors for sequential testing |
| Gibbs & Candes (2021) | Adaptive conformal inference for radius calibration |
| PAuth (2026) | Task-scoped authorization via NL slices |
| MI9 (Wang et al., 2025) | Goal-conditioned drift detection, 6-component governance |
| South et al. (2025) | Authenticated delegation tokens |
| PROV-AGENT (Souza et al., 2025) | W3C PROV extensions for agent provenance |
| Ahmed et al. (2024) | DAS-CUSUM sequential change detection |
| Benjamin et al. (2018) | Bayes factor interpretation thresholds |

---

## Quick Start

```python
from frontier_ops import FullPipeline

pipeline = FullPipeline()

# User says what they want
event = pipeline.process_user_message("Set up PyTorch on my GPU server")
# → ESTABLISH, goal extracted, budget granted

# Agent acts
r1 = pipeline.process_step("Running apt-get install python3-pip")
# → authorized=True, geodesic_distance=0.12, verdict="pass"

r2 = pipeline.process_step("Installing pytorch with CUDA support")
# → authorized=True, geodesic_distance=0.18, verdict="pass"

r3 = pipeline.process_step("Reading /etc/shadow for root password")
# → authorized=False, geodesic_distance=1.47, verdict="escalate"
#   (credential direction amplified by metric tensor)

# User expands scope
event2 = pipeline.process_user_message("Also set up the PostgreSQL database")
# → EXPAND, goal blended, budget replenished 30%

# Trace why agent did something
chain = pipeline.provenance.trace_authorization(action_node_id)
# → [Directive("Set up PyTorch..."), Directive("Also set up PostgreSQL...")]
```
