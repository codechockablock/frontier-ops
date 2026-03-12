# frontier-ops

Define your boundaries. Monitor your agents. Prove they stayed within bounds.

**frontier-ops** is vendor-agnostic middleware for AI agent governance. It sits between any AI agent platform (Claude Code, OpenAI agents, custom deployments) and the organization deploying it. It answers one question: *can we trust what this agent is doing?*

## Components

**Boundary Engine** (`frontier_ops.boundary`) — Domain experts define constitutions: which behaviors are acceptable, where the limits are, and which combinations are dangerous. The engine turns these specifications into a mathematical structure that makes boundary violations expensive to approach.

**Sensing Layer** (`frontier_ops.sensing`) — A suite of detectors that monitor agent behavior in real time. Spike detection catches sudden dangerous actions. Drift detection catches slow trajectory changes. Trend detection catches scope creep. Evidence combination weighs multiple signals into a single confidence measure.

**Governance Ledger** (`frontier_ops.governance`) — Tamper-evident audit chain using Ed25519 signatures. Every observation is signed and chained. Cross-session budget tracking prevents trajectory splice attacks. An external auditor can verify the entire history.

## Installation

```bash
pip install frontier-ops
```

Core package requires only numpy. Optional extras:

```bash
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

## Custom Constitutions

```python
from frontier_ops import ConstitutionSpec, Boundary, CrossTerm

constitution = ConstitutionSpec(
    name="my_deployment",
    boundaries=[
        Boundary("credential_adjacent", threshold=0.5, sharpness=6.0),
        Boundary("scope_exploration", threshold=0.7, sharpness=3.0),
    ],
    cross_terms=[
        CrossTerm(("credential_adjacent", "scope_exploration"), weight=4.0),
    ],
)
```

## Examples

See the [`examples/`](examples/) directory:

- `monitor_claude_code.py` — Monitor a Claude Code session
- `monitor_openclaw.py` — Hook into OpenClaw's event loop
- `custom_constitution.py` — Define domain-specific boundaries

## License

MIT
