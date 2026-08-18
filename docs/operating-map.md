# Operating Map — frontier-ops

**Project status: ACTIVE (operator, 2026-08-17). The 2026-08-06 LEGACY/parked
declaration is reversed. Lane states below still record where things stood at
the 2026-08-06 parking and have not been re-confirmed — treat them as the
resume point, not live truth, until the operator refreshes them.**

Read-first rule: any session joining this project reads this file before starting work,
and records itself as owning session when it takes a lane.
Update on state change only (start / block / handoff / done), not as a journal.

## Decisions
- 2026-08-17 — Operator: unparked. The project is active again; the 2026-08-06
  legacy declaration no longer reflects intent. Same ruling set main's license
  to MIT (the AGPLv3 LICENSE from 2026-08-08 contradicted pyproject's
  long-standing MIT declaration and is replaced on main). Lane states below
  were not re-adjudicated in this edit.
- 2026-08-06 — Operator: this project is legacy; nothing is in flight. Map retained as historical state record. *(Reversed 2026-08-17, above.)*
- 2026-08-06 — Map initialized. Lanes inferred from branches, dirty tree, and spec files (⚠ = unconfirmed; operator to correct).

## Lanes

### fable-spec-v2-encoder-drift ⚠
- **Objective:** v2 encoder-drift harness per fable-spec-drift-harness.md
- **Owning session:** unassigned — record on next session
- **State:** parked (project legacied — operator, 2026-08-06)
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred from working tree

### calibration-transport ⚠
- **Objective:** Calibration transport spec/implementation (fable-spec-calibration-transport.md, CALIBRATION_REPORT.md)
- **Owning session:** unassigned
- **State:** parked (project legacied — operator, 2026-08-06)
- **Blockers:** possibly sequenced behind encoder-drift lane (same spec family)
- **Last update:** 2026-08-06 — inferred

### v2-postmortem-refactor ⚠
- **Objective:** Post-mortem-driven refactor branch (+v2-postmortem-refactor)
- **Owning session:** unassigned
- **State:** parked (project legacied — operator, 2026-08-06)
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred

### market-daemon ⚠
- **Objective:** Market daemon per MARKET_IMPLEMENTATION_PLAN.md (SIGTERM orphan fix landed 2026-07-28)
- **Owning session:** unassigned
- **State:** parked (project legacied — operator, 2026-08-06)
- **Blockers:** —
- **Last update:** 2026-08-06 — inferred

## Done
_(finished lanes move here with a one-line outcome)_
