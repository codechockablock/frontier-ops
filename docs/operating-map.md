# Operating Map — frontier-ops

Read-first rule: any session joining this project reads this file before starting work,
and records itself as owning session when it takes a lane.
Update on state change only (start / block / handoff / done), not as a journal.

## Decisions
- 2026-08-06 — Map initialized. Lanes inferred from branches, dirty tree, and spec files (⚠ = unconfirmed; operator to correct).

## Lanes

### fable-spec-v2-encoder-drift ⚠
- **Objective:** v2 encoder-drift harness per fable-spec-drift-harness.md
- **Owning session:** unassigned — record on next session
- **State:** active — current branch with ~46 uncommitted files (pipeline.py, test_pipeline.py in flight)
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred from working tree

### calibration-transport ⚠
- **Objective:** Calibration transport spec/implementation (fable-spec-calibration-transport.md, CALIBRATION_REPORT.md)
- **Owning session:** unassigned
- **State:** active — spec recently modified
- **Blockers:** possibly sequenced behind encoder-drift lane (same spec family)
- **Last update:** 2026-08-06 — inferred

### v2-postmortem-refactor ⚠
- **Objective:** Post-mortem-driven refactor branch (+v2-postmortem-refactor)
- **Owning session:** unassigned
- **State:** paused — branch exists, no recent commits observed
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred

### market-daemon ⚠
- **Objective:** Market daemon per MARKET_IMPLEMENTATION_PLAN.md (SIGTERM orphan fix landed 2026-07-28)
- **Owning session:** unassigned
- **State:** paused / maintenance
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred

## Done
_(finished lanes move here with a one-line outcome)_
