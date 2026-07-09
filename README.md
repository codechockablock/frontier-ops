# frontier-ops

Define your boundaries. Monitor your agents. Prove they stayed within bounds.

**frontier-ops** is vendor-agnostic middleware for AI agent governance. It sits between any AI agent platform (Claude Code, OpenAI agents, custom deployments) and the organization deploying it. It answers one question: *can we trust what this agent is doing?*

## Components

**Boundary Engine** (`frontier_ops.boundary`) — Domain experts define constitutions: which behaviors matter, where the limits are, and what crossing them costs. As of v2, experts specify *costs* (softplus λ weights, applied by `DecisionCosts` to calibrated scores) while the scoring *geometry* is estimated from data (`CalibratedMetric.from_labeled` — pooled within-class covariance, ridge 1e-3). On the Apollo deception benchmarks the estimated metric beat both the identity metric and expert-asserted tensors everywhere it ran; asserted-G construction is deprecated but still available.

**Step-mean scoring** (`frontier_ops.boundary.step_mean.StepMeanScorer`) — the strongest detector from the v2 evaluation: split a response into sentence steps, encode each step in the concept chart, average in chart space, classify with nearest-centroid under the calibrated metric. Insider-trading episodes: AUROC 0.790 vs 0.611 for whole-text single-point encoding (reproducible via `python -m eval.battery`).

**Custom concept dimensions** — constitutions, semantic anchors, and the full pipeline accept custom dimensions as constructor parameters (`FullPipeline(dim_names=..., semantic_anchors=...)`, Tier-2 extraction; the Tier-1 keyword table is skipped automatically for dims it does not cover).

**Sensing Layer** (`frontier_ops.sensing`) — real-time detectors for live sessions: spike detection, drift detection (NEWMA), scope-creep trend detection, evidence combination. *Validation note:* the drift/NEWMA/trend detectors are designed for long-running live sessions; the v2 benchmark campaign only exercised them on short response-internal paths, which says nothing about their design regime.

**Governance Ledger** (`frontier_ops.governance`) — Tamper-evident audit chain using Ed25519 signatures. Every observation is signed and chained. Cross-session budget tracking prevents trajectory splice attacks. An external auditor can verify the entire history.

## Installation

```bash
pip install frontier-ops
```

Core package requires only numpy. Optional extras:

```bash
pip install frontier-ops[semantic]    # Tier-2 semantic extraction (MiniLM)
pip install frontier-ops[governance]  # Ed25519 signing
pip install frontier-ops[kalman]      # Kalman filter prediction
pip install frontier-ops[all]         # Everything
```

## Quick Start

```python
from frontier_ops import ConstitutionSpec, FullPipeline

# Define your boundaries
constitution = ConstitutionSpec.agent_safety_default()

# Create the monitoring pipeline
pipeline = FullPipeline(constitution=constitution)

# Monitor each agent action
result = pipeline.process_step("Reading config from ~/.ssh/config")
print(f"Alert: {result.alert_level:.2f}")
print(f"Reasons: {result.alert_reasons}")
print(f"Boundary proximity: {result.boundary_proximities}")
```

**Ranking vs alerting:** `alert_level` is a gated decision signal for live
monitoring. For offline evaluation or ranking (AUROC etc.) always use raw
scores (`concept_vec`, `StepMeanScorer.score`) — saturating alert thresholds
destroy ranking information.

## Custom Constitutions (custom dimensions)

```python
from frontier_ops import ConstitutionSpec, Boundary, CrossTerm, FullPipeline

dims = ["fabricated_justification", "strategic_vagueness",
        "impression_management", "honest_disclosure"]
anchors = {d: ["anchor phrase one", "anchor phrase two"] for d in dims}

constitution = ConstitutionSpec(
    name="honesty_monitor",
    boundaries=[
        Boundary("fabricated_justification", threshold=0.45, sharpness=5.0, base_weight=3.0),
        Boundary("strategic_vagueness", threshold=0.50, sharpness=4.0, base_weight=2.0),
    ],
    cross_terms=[
        CrossTerm(("fabricated_justification", "impression_management"), weight=3.0),
    ],
)

pipeline = FullPipeline(
    constitution=constitution,
    concept_extractor_tier=2,      # semantic anchors need [semantic] extra
    dim_names=dims,
    semantic_anchors=anchors,
)
```

## Evaluation

`python -m eval.battery` reproduces the v2 evaluation against the Apollo
Research deception benchmarks (data fetched at runtime at a pinned commit;
see `eval/battery/expected.json` for the expected numbers and
`eval/session_artifacts/` for the original campaign harnesses).

## Examples

See the [`examples/`](examples/) directory:

- `monitor_claude_code.py` — Monitor a Claude Code session
- `monitor_openclaw.py` — Hook into OpenClaw's event loop
- `custom_constitution.py` — Define domain-specific boundaries

## License

MIT
