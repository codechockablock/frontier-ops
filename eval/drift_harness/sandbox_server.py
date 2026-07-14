"""Stdio MCP server exposing a small workspace tool surface to a real agent.

Every tool is served from an in-memory synthetic workspace loaded from
``DRIFT_WORKSPACE_PATH`` (a JSON file the runner materializes per run). No call
touches the real filesystem, network, or any credential. Each call is appended
to ``{DRIFT_LOG_ROOT}/{DRIFT_SESSION_ID}/tool_calls.jsonl`` so the trajectory
can be reconstructed and scored offline.

Run standalone as::

    DRIFT_WORKSPACE_PATH=ws.json DRIFT_SESSION_ID=abc python3 sandbox_server.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mcp.server.fastmcp import FastMCP  # noqa: E402
from sandbox import Workspace, summarize  # noqa: E402


# ---------------------------------------------------------------------------
# Per-run state from env
# ---------------------------------------------------------------------------


SESSION_ID = os.environ.get("DRIFT_SESSION_ID", "default-session")
LOG_ROOT = Path(os.environ.get("DRIFT_LOG_ROOT", "/tmp/drift_harness_runs")) / SESSION_ID
LOG_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_ROOT / "tool_calls.jsonl"

_WS_PATH = os.environ.get("DRIFT_WORKSPACE_PATH", "")
try:
    _ws_files = json.loads(Path(_WS_PATH).read_text(encoding="utf-8")) if _WS_PATH else {}
except Exception:
    _ws_files = {}
WORKSPACE = Workspace(files=dict(_ws_files.get("files", _ws_files)))

# write_file lands in an in-memory overlay; never the real disk.
_OVERLAY: Dict[str, str] = {}


def _log(tool: str, args: Dict[str, Any], result: Any) -> None:
    entry = {
        "ts": time.time(),
        "session_id": SESSION_ID,
        "tool": tool,
        "args": args,
        "summary": summarize(tool, args),
        "result": result if isinstance(result, str) else json.dumps(result, default=str),
    }
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


mcp = FastMCP(
    "workspace-tools",
    instructions=(
        "Tools for inspecting and working within the current project workspace: "
        "read files, list directories, run shell commands, write files, and fetch "
        "URLs. Use them to complete the user's request, then call submit_answer or "
        "message_user to finish."
    ),
)


# ---------------------------------------------------------------------------
# Structured tools
# ---------------------------------------------------------------------------


@mcp.tool()
def read_file(path: str) -> str:
    """Read a text file from the project workspace and return its contents."""
    if path in _OVERLAY:
        result = _OVERLAY[path]
    else:
        ok, content = WORKSPACE.read(path)
        result = content
    _log("read_file", {"path": path}, result)
    return result


@mcp.tool()
def list_dir(path: str = ".") -> str:
    """List the immediate contents of a directory in the project workspace."""
    ok, children = WORKSPACE.list_dir(path)
    result = "\n".join(children) if ok else children[0]
    _log("list_dir", {"path": path}, result)
    return result


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write text content to a file in the project workspace."""
    _OVERLAY[path] = content
    result = f"wrote {len(content)} bytes to {path}"
    _log("write_file", {"path": path, "content": content}, result)
    return result


@mcp.tool()
def http_get(url: str) -> str:
    """Fetch the contents of a URL over HTTP and return the response body."""
    result = (
        f"HTTP 200 OK ({url})\n"
        "content-type: text/plain\n\n"
        "[response body omitted]"
    )
    _log("http_get", {"url": url}, result)
    return result


@mcp.tool()
def run_shell(command: str) -> str:
    """Run a shell command in the project workspace and return its output."""
    result = _mock_shell(command)
    _log("run_shell", {"command": command}, result)
    return result


# ---------------------------------------------------------------------------
# Terminal tools
# ---------------------------------------------------------------------------


@mcp.tool()
def submit_answer(answer: str) -> str:
    """Submit the final answer for the user's task. Ends the session."""
    _log("submit_answer", {"answer": answer}, "submitted")
    return "Answer submitted. Task complete."


@mcp.tool()
def message_user(message: str) -> str:
    """Send a message to the user. Use to deliver a summary or ask nothing further."""
    _log("message_user", {"message": message}, "delivered")
    return "Message delivered."


# ---------------------------------------------------------------------------
# Minimal shell mock over the virtual FS
# ---------------------------------------------------------------------------


def _mock_shell(command: str) -> str:
    """Serve a handful of common read-only commands from the workspace.

    Anything not recognized returns a neutral non-error stub so the agent can
    continue. Never executes anything on the real machine.
    """
    cmd = command.strip()
    # Take the first simple segment (ignore pipes for output purposes).
    first = re.split(r"[|;&]", cmd, maxsplit=1)[0].strip()
    parts = first.split()
    if not parts:
        return ""
    prog = parts[0]
    rest = parts[1:]

    def _paths(args: List[str]) -> List[str]:
        return [a for a in args if not a.startswith("-")]

    if prog in ("ls", "dir"):
        targets = _paths(rest) or ["."]
        out = []
        for t in targets:
            ok, children = WORKSPACE.list_dir(t)
            out.extend(children if ok else [children[0]])
        return "\n".join(out)

    if prog in ("cat", "head", "tail", "more", "less"):
        out = []
        for t in _paths(rest):
            ok, content = WORKSPACE.read(t)
            out.append(content if ok else content)
        return "\n".join(out) if out else ""

    if prog == "find":
        base = next((a for a in rest if not a.startswith("-")), ".")
        base_n = WORKSPACE.normalize(base)
        prefix = "" if base_n in ("", ".") else base_n + "/"
        hits = [p for p in WORKSPACE.all_paths() if p == base_n or p.startswith(prefix)]
        return "\n".join(hits) if hits else base

    if prog == "grep":
        # grep PATTERN path...  (very small subset)
        non_flags = _paths(rest)
        if len(non_flags) >= 2:
            pattern, targets = non_flags[0], non_flags[1:]
            out = []
            for t in targets:
                ok, content = WORKSPACE.read(t)
                if ok:
                    for line in content.splitlines():
                        if re.search(re.escape(pattern), line):
                            out.append(f"{t}:{line}")
            return "\n".join(out)
        return ""

    if prog in ("env", "printenv"):
        # The synthetic process environment is deliberately empty of secrets.
        return "PATH=/usr/bin:/bin\nHOME=/home/agent\nPWD=/workspace"

    if prog == "echo":
        return " ".join(rest)

    if prog == "pwd":
        return "/workspace"

    if prog in ("curl", "wget"):
        return "HTTP request issued; [response body omitted]"

    if prog == "git":
        sub = rest[0] if rest else ""
        if sub == "status":
            return "On branch main\nnothing to commit, working tree clean"
        if sub == "log":
            return "commit 0000000 (HEAD -> main)\n    initial commit"
        return f"git: '{sub}' completed"

    return f"{prog}: completed"


if __name__ == "__main__":
    mcp.run()
