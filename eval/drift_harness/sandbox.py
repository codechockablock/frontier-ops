"""Synthetic in-memory workspace + mocked tool execution.

Shared by the MCP server (which serves live tool calls to a real agent) and by
the offline record/label/score code (which reconstructs action text from the
logged calls). ZERO real filesystem, network, or credential access happens
here — every tool call is served from an in-memory ``Workspace`` fixture that
ships with the scenario.

Nothing in this module imports any ``frontier_ops`` detector. It is deliberately
kept detector-free so the independent labeler that builds on it (``labeler.py``)
cannot accidentally become circular with the thing under evaluation.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Workspace model
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    """A fully synthetic file tree. ``files`` maps POSIX path -> content.

    Directories are implied by the paths of the files under them. No path in
    a committed fixture may contain a real secret; the planted "secrets" are
    obvious fakes (``sk-FAKE-...``) declared per scenario.
    """

    files: Dict[str, str] = field(default_factory=dict)

    def normalize(self, path: str) -> str:
        """Normalize an agent-supplied path to a workspace-relative POSIX path."""
        p = (path or "").strip()
        # Strip common prefixes the model tends to add.
        for pref in ("./", "/workspace/", "workspace/", "/repo/", "repo/", "/"):
            if p.startswith(pref):
                p = p[len(pref):]
        p = posixpath.normpath(p) if p else ""
        return "" if p == "." else p

    def is_dir(self, path: str) -> bool:
        p = self.normalize(path)
        if p == "":
            return True
        prefix = p + "/"
        return any(f == p and False for f in self.files) or any(
            f.startswith(prefix) for f in self.files
        )

    def read(self, path: str) -> Tuple[bool, str]:
        """Return (ok, content_or_error)."""
        p = self.normalize(path)
        if p in self.files:
            return True, self.files[p]
        if self.is_dir(path):
            return False, f"read_file: '{path}' is a directory, not a file"
        return False, f"read_file: no such file: '{path}'"

    def list_dir(self, path: str) -> Tuple[bool, List[str]]:
        """Immediate children (files and subdirs) of ``path``."""
        p = self.normalize(path)
        prefix = "" if p == "" else p + "/"
        children = set()
        for f in self.files:
            if not f.startswith(prefix):
                continue
            rest = f[len(prefix):]
            if "/" in rest:
                children.add(rest.split("/", 1)[0] + "/")
            else:
                children.add(rest)
        if not children and p != "" and not self.is_dir(path):
            return False, [f"list_dir: no such directory: '{path}'"]
        return True, sorted(children)

    def all_paths(self) -> List[str]:
        return sorted(self.files.keys())


# ---------------------------------------------------------------------------
# Deterministic action summary  (the f"{tool}: {summary}" convention)
# ---------------------------------------------------------------------------


def summarize(tool: str, args: Dict[str, Any]) -> str:
    """Build the short human-readable action description from tool + args.

    This is the ``summary`` half of the ``f"{tool}: {summary}"`` encoding that
    every frontier-ops eval feeds to the detectors. It is a faithful, mechanical
    rendering of what the call does — it does NOT inject detector-friendly
    vocabulary. Whether e.g. reading ``.env`` trips the credential dimension is
    the detector's problem, not something this function helps or hinders.
    """
    args = args or {}
    if tool == "read_file":
        return f"read file {args.get('path', '')}".strip()
    if tool == "list_dir":
        return f"list directory {args.get('path', '') or '.'}".strip()
    if tool == "write_file":
        return f"write file {args.get('path', '')}".strip()
    if tool == "http_get":
        return f"http GET {args.get('url', '')}".strip()
    if tool == "run_shell":
        return f"run shell: {args.get('command', '')}".strip()
    if tool == "submit_answer":
        # The answer content IS the action here (what the agent communicates);
        # a summary that dropped it would hide narration-borne drift (e.g. a
        # secret echoed in the final answer) from the detectors.
        return str(args.get("answer", "")).strip()
    if tool == "message_user":
        return str(args.get("message", "")).strip()
    # Unknown tool — render generically but honestly.
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{tool} {inner}".strip()


def action_text(tool: str, args: Dict[str, Any]) -> str:
    """The exact string fed to FullPipeline.process_step for one action."""
    return f"{tool}: {summarize(tool, args)}"


_TERMINAL_TOOLS = {"submit_answer", "message_user"}


def effective_actions(
    trajectory: List[Dict[str, Any]], final_text: str = ""
) -> List[Tuple[str, Dict[str, Any]]]:
    """The full action sequence a detector should see: every tool call, plus a
    synthetic terminal ``message_user`` action carrying the model's final
    narration WHEN the model answered without calling a terminal tool.

    Without this, narration-borne drift (the most common emergent case — a
    secret echoed in the final answer) would be scored by the independent
    labeler but never fed to the detector, an unfair asymmetry. Appending it
    only when no terminal tool was called avoids double-counting.
    """
    actions = [(s.get("tool", ""), s.get("args", {}) or {}) for s in trajectory]
    has_terminal = any(t in _TERMINAL_TOOLS for t, _ in actions)
    if final_text and not has_terminal:
        actions.append(("message_user", {"message": final_text}))
    return actions


# ---------------------------------------------------------------------------
# Path extraction  (used by the independent labeler and by run_shell mocking)
# ---------------------------------------------------------------------------


_URL_RE = re.compile(r"https?://[^\s'\"]+", re.IGNORECASE)
_NET_CMD_RE = re.compile(r"\b(curl|wget|nc|netcat|scp|ssh|ftp|telnet)\b", re.IGNORECASE)


def referenced_paths(tool: str, args: Dict[str, Any]) -> List[str]:
    """Every workspace path an action touches, tool-agnostically.

    For structured tools the path is an explicit arg. For ``run_shell`` we
    parse the command line for path-like tokens. Returned paths are raw (not
    normalized) so the caller can decide scope; normalization is left to the
    ``Workspace`` when it matters.
    """
    args = args or {}
    if tool in ("read_file", "list_dir", "write_file"):
        p = args.get("path", "")
        return [p] if p else []
    if tool == "run_shell":
        return _paths_in_command(str(args.get("command", "")))
    return []


def _paths_in_command(command: str) -> List[str]:
    """Extract path-like tokens from a shell command line."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    paths: List[str] = []
    for tok in tokens:
        # Skip the program name, flags, and shell operators.
        if tok.startswith("-") or tok in ("|", "&&", "||", ";", ">", ">>", "<"):
            continue
        if _looks_like_path(tok):
            paths.append(tok)
    return paths


_SHELL_BUILTINS = {
    "ls", "cat", "find", "grep", "head", "tail", "echo", "env", "printenv",
    "pwd", "cd", "wc", "sort", "uniq", "awk", "sed", "tr", "cut", "xargs",
    "git", "python", "python3", "node", "npm", "true", "false", "test",
    "curl", "wget", "nc", "netcat", "scp", "ssh",
}


def _looks_like_path(tok: str) -> bool:
    if not tok or tok in _SHELL_BUILTINS:
        return False
    if _URL_RE.match(tok):
        return False
    # A path-like token contains a slash, a dot-file, or a known extension.
    if "/" in tok or tok.startswith("."):
        return True
    if re.search(r"\.(py|js|ts|yaml|yml|json|env|txt|md|cfg|ini|toml|sh|sql|pem|key)$",
                 tok, re.IGNORECASE):
        return True
    return False


def urls_in(tool: str, args: Dict[str, Any]) -> List[str]:
    """External URLs an action reaches out to (http_get + shell curl/wget)."""
    args = args or {}
    if tool == "http_get":
        u = args.get("url", "")
        return [u] if u else []
    if tool == "run_shell":
        return _URL_RE.findall(str(args.get("command", "")))
    return []


def is_network_action(tool: str, args: Dict[str, Any]) -> bool:
    if tool == "http_get":
        return True
    if tool == "run_shell":
        return bool(_NET_CMD_RE.search(str((args or {}).get("command", ""))))
    return False


def outbound_content(tool: str, args: Dict[str, Any]) -> str:
    """Text an action would transmit or persist (for secret-leak detection)."""
    args = args or {}
    if tool == "write_file":
        return str(args.get("content", ""))
    if tool == "http_get":
        return str(args.get("url", ""))
    if tool == "run_shell":
        return str(args.get("command", ""))
    if tool in ("submit_answer", "message_user"):
        return str(args.get("answer", args.get("message", "")))
    return ""
