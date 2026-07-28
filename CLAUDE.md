# frontier-ops — Agent Instructions

## What This Is

Vendor-agnostic middleware for AI agent governance. v0.2.0. pip-installable Python library.
6 packages: boundary, sensing, governance, memory, authorization, integration.
514 tests, Python 3.10–3.14.

**Relationship to unified-stack:** `~/unified-stack` is a *diverged fork*, not a downstream
consumer (verified 2026-07-11, re-verified 2026-07-28): it vendors its own `frontier_ops`
package at v2.0.0 with a different architecture, and neither repo imports the other. Changes
here cannot break it, and its tests need no re-run. Two couplings remain: both repos claim the
distribution name `frontier-ops` (resolve before any PyPI publish), and the OpenClaw sidecar in
`~/.openclaw/workspace` imports *unified-stack's* `Pipeline` for live evaluation while running
this repo's market architecture in a separate subprocess — the two `frontier_ops` packages
cannot coexist in one interpreter.

## Repo Structure

```
frontier_ops/
  boundary/         — constitution, metric (G matrix)
  sensing/          — CUSUM, SPRT, entropy, efference, EWMA
  governance/       — Ed25519 audit chain
  memory/           — VSA phasor
  authorization/    — GoalExtractor, AuthorizationRadius, AGM scope
  integration/      — sidecar/daemon
tests/              — 514 tests
paper/              — research paper drafts
docs/               — technical briefs
eval/               — evaluation scripts
```

## Running Tests

```bash
pytest tests/ -x -q           # fast run
pytest tests/ --tb=short      # with tracebacks
ruff check .                  # lint (run before pushing)
```

## Key Architecture Decisions

- G matrix: expert-specified positive definite (Gershgorin). G IS the Fisher information of behavioral policy.
- CALM theorem: monotonic G updates = coordination-free; non-monotonic = requires human.
- Market Architecture (live): DAS-CUSUM + entropy antitrust + Ed25519 audit chain.
- Authorization: GoalExtractor → AuthorizationRadius → AGM scope ops → budget linkage.

## Hard Rules

- CI/CD: GitHub Actions (pytest 3.10–3.12 + ruff). All PRs must pass.
- Repo is private (codechockablock/frontier-ops).
- No credentials in code.

## Token Efficiency Rules

- **Check `FRONTIER_OPS_TECHNICAL_BRIEF.md`** before exploring structure.
- **Never read full test files** — grep for the specific test name.
