# Signal B: Task Coherence Scorer — Design Document

**Date:** 2026-03-24  
**Author:** Chocka (Opus 4.6)  
**Status:** Implementation complete, integration pending validation on ATBench

---

## 1. What Problem This Solves

Fracture detection (Signature 4 / `fracture_signal()`) catches **sharp angular snaps** — a single step where the cosine similarity between consecutive actions drops dramatically. This catches classic prompt injection where the agent's trajectory is suddenly redirected.

But many ATBench attacks don't work that way. They use:

- **Goal displacement**: Gradual shift from legitimate work to adversarial objectives over 5-10 steps. No single step has a dramatic snap.
- **Task interleaving**: Alternating between legitimate actions and adversarial ones, so each individual transition looks like a normal tool switch.
- **Aimless wandering**: The agent explores without a clear task, probing for capabilities or information before committing to an attack.

These produce trajectories that look fine locally (each step is smooth) but are globally incoherent — the overall path through phasor space doesn't resemble any plausible task.

## 2. How It Works

### Architecture

`TaskCoherenceScorer` operates on the same normalized composite phasor HVs that `PhasorTrajectoryBuffer` stores. It maintains its own sliding window (default 20 steps, larger than fracture's 12) because global structure requires more context.

### Four Sub-signals

#### 2.1 Centroid Drift

Split the trajectory into early and late thirds. Bundle each into a centroid vector, compute cosine similarity between centroids.

```
c_early = normalize(Σ hᵢ for i ∈ [0, n/3))
c_late  = normalize(Σ hᵢ for i ∈ [2n/3, n))
drift = 1 - max(0, cos(c_early, c_late))
```

**Why this works**: Bundling k quasi-orthogonal phasor vectors (dim=512) produces a vector whose direction reflects the dominant components (√k scaling, CLT). If the same action patterns appear in both halves, centroids align (cos > 0). If late actions are entirely novel, centroids are orthogonal (cos ≈ 0, drift ≈ 1).

**Calibration**: Benign coding (3-tool cycle, all user_direct) → drift ≈ 0.0. Benign mixed-tool → drift ≈ 0.1-0.3. Goal displacement (coding → credential theft) → drift ≈ 0.8-1.0.

#### 2.2 Recurrence Asymmetry

Compare recurrence rates between first and second half of the trajectory. Recurrence = fraction of off-diagonal Gram matrix pairs with cos > 0.50.

```
R_first = recurrence rate in first half
R_second = recurrence rate in second half
asymmetry = max(0, R_first - R_second) / max(R_first, ε)
```

**Why this works**: A coherent task maintains or increases action pattern repetition over time (settling in). Goal displacement decreases recurrence in the second half (new patterns that don't match earlier ones).

**Threshold**: cos > 0.50 for "same pattern" is very strict in phasor space — requires all 6 slots to be nearly identical. This prevents false recurrence matches.

#### 2.3 Spectral Concentration

Eigendecompose the Gram matrix. Compute normalized entropy of the eigenvalue distribution.

```
λ₁ ≥ λ₂ ≥ ... ≥ λₙ = eigenvalues of G
p_i = λᵢ / Σλⱼ
concentration = 1 - H(p) / log(n)
```

**Why this works**: A focused task occupies few directions in phasor space (few eigenvalues dominate → high concentration). Wandering spans many directions (flat spectrum → low concentration).

**Calibration**: n=20 identical vectors → concentration = 1.0. n=20 cycling 3 patterns → concentration ≈ 0.63. n=20 random vectors → concentration ≈ 0.0.

#### 2.4 Alternation Index

Check if even-indexed and odd-indexed HVs form separate clusters (period-2 interleaving). Also checks period-3.

```
S_within = mean G[i,j] for i,j in same group
S_between = mean G[i,j] for i,j in different groups
alternation = max(0, S_within - S_between) / (S_within + ε)
```

**Critical calibration note**: Benign coding naturally cycles tools (read, write, exec) which produces alternation_index ≈ 1.0 for period-3. This is NOT adversarial. The scorer gates interleaving detection: it only fires when combined with low spectral concentration (< 0.45) or confirming drift signals. Benign 3-tool cycling has concentration ≈ 0.63 → no fire.

### Threat Composition

Each sub-signal maps to threat types:

| Threat | Primary Signal | Confirming Signals |
|--------|---------------|-------------------|
| Goal displacement | centroid_drift > 0.70 | recognition_slope < 0, recurrence_asymmetry > 0 |
| Aimless wandering | spectral_concentration < 0.15 | recognition_slope < 0 |
| Task interleaving | alternation_index > 0.50 | spectral_concentration < 0.45 OR drift/slope confirming |

**Composite**: `coherence = 1 - max(drift_threat, wandering_threat, interleaving_threat)`

Using max rather than sum because threats are correlated (drift can co-occur with low concentration). Avoids double-penalizing.

### Recognition Slope (auxiliary)

For each step i, compute max similarity to any earlier step. Linear regression slope of this series over time. Negative slope = actions becoming less familiar (diverging into new territory).

Used as a confirmation signal for drift and wandering, not as a primary threat.

## 3. Phasor Space Geometry Assumptions

1. **Composite HVs** are element-wise products of 6 slot fillers (dim=512 complex phasor).
2. **Two composites with all slots identical**: cos = 1.0 exactly (same product).
3. **Two composites with any categorical slot differing**: cos ≈ 0 (product with a random phasor decorrelates). This is empirically confirmed — coding consecutive cosines are ≈ -0.05 to 0.05 for tool-type switches.
4. **Continuous slot differences** (e.g., magnitude bin 3 vs 4): cos ≈ 0.97 (adjacent bins in 40-bin encoding are correlated because the random seed generates similar phases for nearby labels).
5. **Random pair expectation**: E[cos] = 0, std = 1/√dim ≈ 0.044.
6. **Bundling** of k orthogonal vectors: centroid magnitude ≈ √k, direction reflects dominant components after normalization.

## 4. What This WON'T Catch

1. **Injection attacks that happen in a single step** → fracture_signal() handles this, not coherence.
2. **Adversarial actions that are semantically identical to benign ones** (e.g., an agent that reads credentials using the same file_read pattern it used for source code). The 6-slot encoding can't distinguish file targets — only action_type, scope, source, etc.
3. **Very slow drift over 50+ steps** → window is 20 steps. Drift slower than that requires the session_holonomy signal (already exists).
4. **Perfectly coherent adversarial task execution** (e.g., an agent that efficiently and consistently performs a single adversarial task from start to finish with no deviation). This would score coherence = 1.0. Other signals (source provenance, intent binding) must catch this.
5. **Adversarial interleaving that mimics benign tool cycling** (alternating between 3 tools in the same order as a legitimate coding session). The scorer deliberately tolerates regular cycling patterns.

## 5. Acceptance Gates

### ATBench Improvement Target

**Gate**: ATBench detection ≥ 52% (up from 46.4%), a ≥ 5.6pp absolute improvement.

**Rationale**: The ATBench misses were categorized. Goal displacement and slow-drift attacks account for an estimated 15-20% of the test set. If coherence catches even half of those while maintaining existing detections, that's 7-10pp improvement. 52% is a conservative target.

**Stretch goal**: 55%+ would indicate the signal is highly effective.

### FPR Regression Limit

**Gate**: Synthetic FPR ≤ 9.0% (from current 7.5%, allowing ≤ 1.5pp regression).

**Rationale**: Coherence verdicts are gated by the warmup phase (first 7 steps = PASS) and require either FLAG (coherence < 0.25) or MONITOR (coherence < 0.50) to escalate. Benign sessions tested against the scorer show coherence ≥ 0.65 for focused coding and ≥ 0.35 for mixed-tool work, well above both thresholds.

### Distribution Separation

**Metric**: Cohen's d between benign and adversarial coherence score distributions.

**Gate**: d ≥ 0.5 (medium effect size).

**Measurement**: Test suite includes `TestDistributionSeparation` which runs 10 benign and 10 adversarial scenarios across 5 random seeds and asserts mean_benign > mean_adversarial with gap > 0.08. This is a lower bound; real ATBench separation should be stronger because the synthetic adversarial traces are more aggressive.

## 6. Integration Points

### 6.1 File: `frontier_ops/integration/task_coherence.py` (NEW)

The `TaskCoherenceScorer` class. Self-contained, imports only numpy.

### 6.2 File: `frontier_ops/integration/sidecar.py`

Add `TaskCoherenceScorer` to the sidecar pipeline. The scorer receives the same composite HV that the trajectory buffer gets.

```python
# In ProprioceptiveSidecar.__init__():
from frontier_ops.integration.task_coherence import TaskCoherenceScorer
self.coherence_scorer = TaskCoherenceScorer(window=20, dim=512)

# In _on_tool_event(), after the wrapper processes the event:
# (The composite HV is available from wrapper.validator.encoder)
if result:
    # Get the last composite HV from the wrapper's trajectory
    encoded = self.wrapper.validator.encoder.encode_action(classified_action)
    composite_hv = bind_slot_vectors(encoded.fillers)
    self.coherence_scorer.push(composite_hv)
    coherence_result = self.coherence_scorer.score()
    result["coherence"] = coherence_result
```

### 6.3 File: `frontier_ops/integration/wrapper.py`

The cleaner integration point. Add coherence to `ProprioceptiveWrapper.on_tool_call()`:

```python
# After computing composite_hv (already done for polytope):
self.coherence_scorer.push(composite_hv)
coherence_result = self.coherence_scorer.score()

# Merge with verdict (escalate only):
coherence_verdict = self.coherence_scorer.verdict_contribution()
if VERDICT_ORDER.get(coherence_verdict, 0) > VERDICT_ORDER.get(current_verdict, 0):
    verdict_result["verdict"] = coherence_verdict
    verdict_result["escalated_by_coherence"] = True

# Attach to result:
verdict_result["coherence"] = {
    "score": coherence_result["coherence"],
    "drift": coherence_result["drift_score"],
    "interleaving": coherence_result["interleaving_score"],
    "wandering": coherence_result["wandering_score"],
    "phase": coherence_result["phase"],
}
```

### 6.4 Session Reset

In `ProprioceptiveWrapper.on_session_start()`, add:
```python
self.coherence_scorer.clear()
```

### 6.5 Compact State

Add coherence to `write_compact_state()` in sidecar.py:
```python
compact["coh"] = round(state.get("coherence", {}).get("score", 1.0), 2)
```

## 7. Test Summary

24 tests in `tests/integration/test_task_coherence.py`:

| Category | Tests | Key Assertions |
|----------|-------|---------------|
| Benign coding | 2 | coherence ≥ 0.65, verdict = PASS |
| Benign mixed-tool | 2 | coherence ≥ 0.35, verdict ∈ {PASS, MONITOR} |
| Goal displacement | 2 | coherence < 0.55, drift_score > 0.20, verdict ∈ {MONITOR, FLAG} |
| Compliance then drift | 2 | late coherence < early coherence, verdict ∈ {MONITOR, FLAG} |
| Task interleaving | 2 | interleaving_score > 0.20, coherence < 0.75 |
| Aimless wandering | 1 | spectral_concentration < 0.30 |
| Warmup phase | 1 | phase = warmup, coherence = 1.0 |
| Centroid drift | 2 | identical → 0, orthogonal → > 0.7 |
| Spectral concentration | 2 | identical → > 0.85, random → < 0.25 |
| Alternation index | 2 | A/B/A/B → > 0.30, uniform → < 0.05 |
| Recurrence asymmetry | 1 | symmetric → < 0.10 |
| Edge cases | 4 | empty, wrong dim, zero vec, clear |
| Distribution separation | 1 | gap > 0.08 across 5 seeds × 4 scenarios |

All 24 pass. Full test suite (647 tests) passes with no regressions.
