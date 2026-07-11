# frontier-ops — Agent Instructions

## What This Is

Vendor-agnostic middleware for AI agent governance: a calibrated text
classifier for agent-session actions, wrapped in an authorization/audit
layer. pip-installable Python library (numpy core; heavier deps are
optional extras behind lazy imports). Version: see `pyproject.toml`.

**The detector story leads with `CalibratedDetector`** (`frontier_ops/detector.py`):
embed action text → score against a prototype direction calibrated from a
few dozen labeled in-domain examples → threshold at a benign quantile
(AUROC 0.93 on real agent-session traffic). The old 12-signal pipeline is a
legacy shell around two good signals (prototype + NEWMA) — read
`docs/LEGACY.md` before touching anything that looks deprecated; it defines
what is validated, what is infrastructure, and what is quarantined.

**Relationship to unified-stack:** `~/unified-stack` is a *diverged fork*,
not a consumer (verified 2026-07-11): it vendors its own `frontier_ops`
package at version 2.0.0 with a different architecture and imports nothing
from this repo, so changes here cannot break it and its tests need no
re-run. One live coupling remains: both claim the distribution name
`frontier-ops` — resolve before any PyPI publish.

## Repo Structure

```
frontier_ops/
  detector.py       — CalibratedDetector (THE detector; save/load, conformal)
  adaptive_threshold.py — RollingThreshold (drift-tracking benign quantile)
  conformal.py      — shared split-conformal quantile core
  pipeline.py       — FullPipeline (authorization/governance layer, alert_level)
  boundary/         — encoder, StepMeanScorer, CalibratedMetric
  sensing/          — NEWMA (validated), CUSUM/SPRT/entropy, EWMA, quarantined modes
  governance/       — Ed25519 audit chain (infrastructure, not detection)
  memory/           — VSA phasor (speculative; OFF by default)
  authorization/    — GoalExtractor, AuthorizationRadius, AGM scope, conformal radius
  integration/      — sidecar/daemon glue (ATBench; noisy tier)
tests/              — non-slow suite must stay green
paper/              — research paper drafts
docs/               — LEGACY.md (validation boundary), API.md, briefs
eval/               — evaluation scripts; session_artifacts/ is verbatim provenance — NEVER edit
```

## Running Tests

```bash
pytest tests/ -q -m "not slow"     # the gate (slow battery needs Apollo data)
ruff check frontier_ops/ tests/ --select E,F,W --ignore E501   # lint gate
mypy frontier_ops/                 # 0 errors (4-module baseline in pyproject)
```

## Key Facts (v3)

- `CalibratedDetector`: `calibrate(texts, labels)` (~32–64 labels suffice),
  `set_threshold(benign, alpha)` per deployment — **thresholds do not
  transport across workloads**; `calibrate_conformal(benign, alpha)` for a
  finite-sample FPR ≤ α guarantee; `save(path)`/`load(path)` round-trip.
- `RollingThreshold` tracks benign drift from operator-confirmed scores;
  attach via `detector.attach_rolling_threshold(rt)`.
- `FullPipeline(enable_memory=False)` is the v3 default (memory is opt-in;
  no measured detection value). Read `alert_level` post-fix or the NEWMA
  channel; never the removed surprise/direction channels or the
  `max(alert_level, 0.6)` escalation overwrite (both cut on purpose, with
  regression tests).
- Numpy-only import must never break: `python -c "import frontier_ops"`
  with only numpy installed works; scipy/cryptography/sentence-transformers
  stay behind lazy guards.

## Hard Rules

- CI/CD: GitHub Actions (pytest 3.10–3.14, ruff, non-blocking mypy,
  numpy-only import lane, semantic lane). All PRs must pass.
- Repo is **public** (codechockablock/frontier-ops). Nothing is published
  to PyPI; publishing is a human decision.
- No credentials in code.
- Never delete an exported symbol (deprecate/quarantine instead).
- Never edit `eval/session_artifacts/` (verbatim provenance records).

## Token Efficiency Rules

- **Check `FRONTIER_OPS_TECHNICAL_BRIEF.md`** before exploring structure.
- **Never read full test files** — grep for the specific test name.
