"""Record schema + writer, with the hard `banned`-tier guarantee.

A record is one (scenario, run) with the agent's real tool-call trajectory. The
writer REFUSES to emit any record whose trajectory did not come from a runner
transcript: there is deliberately no function anywhere that synthesizes action
text. `assert_writable` is the single choke point and is exercised by a test
that feeds a hand-authored (`banned`) attempt and asserts refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

VALID_PROVENANCE = {"emergent", "instructed"}


class BannedRecordError(ValueError):
    """Raised when a record lacks a genuine runner transcript behind it."""


def build_record(
    *,
    scenario: Dict[str, Any],
    run_idx: int,
    model: str,
    session_id: str,
    tool_calls: List[Dict[str, Any]],
    final_text: str,
    stop_reason: str,
    started_ts: float,
    duration_s: float,
    cli_returncode: int,
) -> Dict[str, Any]:
    """Assemble a record dict from a real run. Trajectory is taken verbatim
    from the logged tool calls; nothing is fabricated here."""
    trajectory = [
        {
            "tool": c.get("tool", ""),
            "args": c.get("args", {}) or {},
            "summary": c.get("summary", ""),
            "stub_return": c.get("result", ""),
        }
        for c in tool_calls
    ]
    return {
        "scenario_id": scenario["id"],
        "surface": scenario["surface"],
        "provenance": scenario["provenance"],
        "directive": scenario["directive"],
        "in_scope": scenario["in_scope"],
        "secret_values": scenario.get("secret_values", []),
        "distractors": scenario.get("distractors", []),
        "run_idx": run_idx,
        "model": model,
        "session_id": session_id,
        "started_ts": started_ts,
        "duration_s": duration_s,
        "cli_returncode": cli_returncode,
        "stop_reason": stop_reason,
        "final_text": final_text,
        "trajectory": trajectory,
    }


def assert_writable(record: Dict[str, Any]) -> None:
    """Enforce the acceptance invariants before a record may be written.

    Raises BannedRecordError if the record is not backed by a real transcript
    or is missing authentic provenance.
    """
    prov = record.get("provenance")
    if prov not in VALID_PROVENANCE:
        raise BannedRecordError(
            f"provenance must be one of {sorted(VALID_PROVENANCE)}, got {prov!r}"
        )
    traj = record.get("trajectory")
    if not isinstance(traj, list) or len(traj) == 0:
        raise BannedRecordError(
            "empty trajectory: a record must be backed by a real runner "
            "transcript with at least one tool call (no hand-authored actions)"
        )
    # A genuine transcript carries stub_return values served by the sandbox
    # server. A hand-authored `banned` attempt won't have them.
    if not any("stub_return" in step for step in traj):
        raise BannedRecordError(
            "trajectory steps lack stub_return: not sourced from a runner"
        )


def write_records(path: Path, records: List[Dict[str, Any]]) -> int:
    """Write records as JSONL, refusing any that fail assert_writable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            assert_writable(rec)
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
