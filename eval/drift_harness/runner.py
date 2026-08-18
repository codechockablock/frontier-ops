"""Runner: invoke the Claude CLI for one (scenario, run) and collect the trace.

Follows the claude-p-runner pattern. For each run we:
  1. Generate a fresh UUID -> session id.
  2. Materialize the scenario's synthetic workspace as a JSON fixture and a
     per-run MCP config that points the sandbox server at it.
  3. Spawn `claude -p ...` in a clean tempdir (no project CLAUDE.md leaks in).
  4. Read {LOG_ROOT}/{uuid}/tool_calls.jsonl written by the sandbox server.
  5. Return a structured dict ready for records.build_record.

Billing routes through the user's Claude Code OAuth subscription allowance —
`--max-budget-usd` is a guardrail in equivalent dollars, not a literal charge.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scenarios import SCENARIOS  # noqa: E402
from scenarios_fresh import FRESH_SCENARIOS  # noqa: E402

# Fresh batches are kept in their own module so the original set stays frozen;
# the runner accepts ids from either.
ALL_SCENARIOS = {**SCENARIOS, **FRESH_SCENARIOS}

CLAUDE_BIN = os.environ.get("DRIFT_CLAUDE_BIN", "/Users/joseph/.local/bin/claude")
SANDBOX_SERVER = str(_HERE / "sandbox_server.py")
LOG_ROOT = Path(os.environ.get("DRIFT_LOG_ROOT", "/tmp/drift_harness_runs"))


def _build_mcp_config(session_id: str, workspace_path: str) -> Dict[str, Any]:
    return {
        "mcpServers": {
            "workspace-tools": {
                "command": "python3",
                "args": [SANDBOX_SERVER],
                "env": {
                    "DRIFT_SESSION_ID": session_id,
                    "DRIFT_WORKSPACE_PATH": workspace_path,
                    "DRIFT_LOG_ROOT": str(LOG_ROOT),
                    "MCP_CONNECTION_NONBLOCKING": "false",
                    "ENABLE_TOOL_SEARCH": "off",
                },
            }
        }
    }


def _read_tool_calls(session_id: str) -> List[Dict[str, Any]]:
    log_path = LOG_ROOT / session_id / "tool_calls.jsonl"
    if not log_path.exists():
        return []
    calls: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return calls


def run_one(
    scenario_id: str,
    run_idx: int,
    *,
    model: str = "claude-haiku-4-5-20251001",
    max_budget_usd: float = 0.40,
    timeout_s: int = 180,
) -> Dict[str, Any]:
    """Run one (scenario, run) end-to-end; return a structured result dict."""
    spec = ALL_SCENARIOS[scenario_id]
    session_id = str(uuid.uuid4())
    log_dir = LOG_ROOT / session_id
    log_dir.mkdir(parents=True, exist_ok=True)

    ws_path = log_dir / "workspace.json"
    ws_path.write_text(json.dumps(spec["workspace"]), encoding="utf-8")

    cfg_path = log_dir / "mcp_config.json"
    cfg_path.write_text(
        json.dumps(_build_mcp_config(session_id, str(ws_path)), indent=2),
        encoding="utf-8",
    )

    work_dir = Path(tempfile.mkdtemp(prefix=f"drift_{scenario_id}_"))

    cmd = [
        CLAUDE_BIN, "-p", spec["directive"],
        "--system-prompt", spec["system_prompt"],
        "--mcp-config", str(cfg_path),
        "--tools", "",
        "--strict-mcp-config",
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
        "--max-budget-usd", str(max_budget_usd),
        "--model", model,
        "--output-format", "json",
        "--session-id", session_id,
    ]

    env = os.environ.copy()
    env["MCP_CONNECTION_NONBLOCKING"] = "false"
    env["ENABLE_TOOL_SEARCH"] = "off"

    started = time.time()
    returncode = 0
    stderr_tail = ""
    final_text = ""
    cost_usd = None
    stop_reason = "unknown"
    try:
        proc = subprocess.run(
            cmd, cwd=str(work_dir), env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        returncode = proc.returncode
        stderr_tail = (proc.stderr or "")[-2000:]
        if proc.stdout.strip():
            try:
                cli_json = json.loads(proc.stdout)
                final_text = str(cli_json.get("result", "") or "")
                cost_usd = cli_json.get("total_cost_usd")
                stop_reason = str(cli_json.get("subtype", cli_json.get("stop_reason", "ok")))
            except json.JSONDecodeError:
                final_text = proc.stdout
                stop_reason = "unparseable_stdout"
    except subprocess.TimeoutExpired:
        returncode = -1
        stop_reason = "timeout"
    duration_s = time.time() - started

    tool_calls = _read_tool_calls(session_id)

    return {
        "scenario_id": scenario_id,
        "run_idx": run_idx,
        "model": model,
        "session_id": session_id,
        "started_ts": started,
        "duration_s": round(duration_s, 2),
        "cli_returncode": returncode,
        "cli_stderr": stderr_tail,
        "final_text": final_text,
        "cost_usd": cost_usd,
        "stop_reason": stop_reason,
        "tool_calls": tool_calls,
        "n_tool_calls": len(tool_calls),
        "work_dir": str(work_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=list(ALL_SCENARIOS.keys()))
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-budget-usd", type=float, default=0.40)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    result = run_one(
        args.scenario, args.run_idx,
        model=args.model, max_budget_usd=args.max_budget_usd, timeout_s=args.timeout,
    )
    print(json.dumps({
        "scenario": result["scenario_id"],
        "session_id": result["session_id"],
        "model": result["model"],
        "n_tool_calls": result["n_tool_calls"],
        "cli_returncode": result["cli_returncode"],
        "stop_reason": result["stop_reason"],
        "cost_usd": result["cost_usd"],
        "duration_s": result["duration_s"],
        "tools": [c.get("tool") for c in result["tool_calls"]],
    }, indent=2))
    return 0 if result["cli_returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
