"""
OpenClaw Tool Call → Phase 15 Action Classifier
=================================================

Maps OpenClaw tool invocations to the 6-slot action representation
used by the Phase 15 agent encoder.

Non-invasive: this module only classifies, never blocks or modifies.

Slot mapping:
  action_type:       what kind of operation
  scope:             blast radius (read_only → destructive)
  source:            why the agent is doing this
  target_sensitivity: how sensitive is the target
  magnitude:         how "big" is this action
  context_alignment:  how related to user's recent request
"""

from __future__ import annotations

import re
from typing import Any, Dict


# ── Tool → action_type mapping ───────────────────────────────────────────

TOOL_ACTION_MAP = {
    "exec": "shell_exec",
    "process": "shell_exec",
    "Read": "file_read",
    "read": "file_read",
    "Write": "file_write",
    "write": "file_write",
    "Edit": "file_write",
    "edit": "file_write",
    "web_search": "web_search",
    "web_fetch": "web_fetch",
    "browser": "browser_action",
    "message": "message_send",
    "memory_search": "file_read",
    "memory_get": "file_read",
    "tts": "api_call",
    "image": "api_call",
    "sessions_spawn": "code_execute",
    "sessions_send": "message_send",
    "subagents": "api_call",
    "session_status": "file_read",
    "agents_list": "file_read",
    "sessions_list": "file_read",
    "sessions_history": "file_read",
    "canvas": "browser_action",
    "nodes": "api_call",
}

# ── Dangerous command patterns ───────────────────────────────────────────

DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf?\b",
    r"\brm\s+",
    r"\bkill\b",
    r"\bpkill\b",
    r"\bkillall\b",
    r"\bmkfs\b",
    r"\bdd\s+",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\b",
    r"\btruncate\b",
]

NETWORK_EGRESS_PATTERNS = [
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b.*:",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bnc\b",
    r"\bnetcat\b",
    r"\bsftp\b",
]

CREDENTIAL_PATTERNS = [
    r"/\.env\b",  # dotenv files (path-anchored)
    r"\.ssh/",  # SSH directory
    r"id_rsa",  # RSA private keys
    r"id_ed25519",  # Ed25519 private keys
    r"\.netrc\b",  # netrc auth files
    r"\.aws/credentials",  # AWS credentials
    r"/\.?tokens?\b",  # token files (path component, not substring)
    r"/\.?secrets?\b",  # secret files (path component, not substring)
    r"/\.?passwords?\b",  # password files (path component, not substring)
    r"api[_.-]?key",  # API key files
    r"\.gnupg",  # GPG keyring
    r"\.keychain\b",  # macOS keychain
    r"credentials\.json\b",  # credential JSON files
    r"\.pem\b",  # PEM cert/key files
]

SYSTEM_PATHS = [
    r"^/etc/",
    r"^/usr/",
    r"^/System/",
    r"^/Library/",
    r"^C:\\Windows",
    r"^C:\\Program",
    r"/usr/local/bin",
]

SENSITIVE_FILE_PATTERNS = [
    r"openclaw\.json",
    r"SOUL\.md",
    r"IDENTITY\.md",
    r"USER\.md",
    r"MEMORY\.md",
    r"AGENTS\.md",
    r"HEARTBEAT\.md",
    r"TOOLS\.md",
    r"proprioception\.json",
    r"proprioception-log\.jsonl",
    r"\.openclaw/config",
]

# Files that define agent identity/behavior — writes here are persistent
# state changes that survive across sessions. Highest-risk write target.
IDENTITY_FILE_PATTERNS = [
    r"SOUL\.md",
    r"IDENTITY\.md",
    r"AGENTS\.md",
]

# Files that control agent operational behavior
OPERATIONAL_FILE_PATTERNS = [
    r"HEARTBEAT\.md",
    r"TOOLS\.md",
    r"USER\.md",
    r"openclaw\.json",
    r"\.openclaw/config",
]

# Files that are the agent's self-monitoring infrastructure
SELF_MONITOR_FILE_PATTERNS = [
    r"proprioception\.json",
    r"proprioception-log\.jsonl",
    r"signature_detectors\.py",
    r"paralysis_detector\.py",
    r"hmm_task_state\.py",
    r"tiered_verdict\.py",
    r"detection_signals\.py",
    r"wrapper\.py",
]


def _matches_any(text: str, patterns: list) -> bool:
    """Check if text matches any regex pattern."""
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


# ── Source classification ────────────────────────────────────────────────

_SKILL_TOOLS = frozenset({"sessions_spawn"})
_WEB_TOOLS = frozenset({"web_fetch", "web_search"})
_API_TOOLS = frozenset({"tts", "image", "nodes", "canvas", "cron"})
_MEMORY_TOOLS = frozenset({"memory_search", "memory_get"})
_READ_TOOLS = frozenset({"Read", "read", "memory_get", "memory_search"})

# Simple substring checks (no regex) for hot-path performance
_SKILL_PATH_MARKERS = ("SKILL.md", "skills/", ".skill", ".openclaw/workspace/skills/")
_MEMORY_PATH_MARKERS = ("memory/", "MEMORY.md")

# Pipe-to-shell patterns: check tool presence + pipe-to-shell separately
# e.g. "curl https://example.com | bash" — curl and | bash aren't adjacent
_PIPE_TO_SHELL = ("| bash", "|bash", "| sh", "|sh")
_WEB_FETCH_CMDS = ("curl", "wget")


def _is_web_pipe_exec(cmd: str) -> bool:
    """Return True if cmd pipes a web fetch tool into a shell (curl|bash etc.)."""
    has_pipe_to_shell = any(marker in cmd for marker in _PIPE_TO_SHELL)
    has_web_cmd = any(cmd.startswith(c) or f" {c} " in cmd or f"\n{c} " in cmd for c in _WEB_FETCH_CMDS)
    return has_pipe_to_shell and has_web_cmd


def _classify_source(
    tool_name: str,
    parameters: Dict[str, Any],
    step_in_chain: int,
) -> str:
    """
    Stateless multi-class source classification.

    Returns one of:
        user_direct   — first action after user message (step 0)
        user_prior    — second action, closely following user intent (step 1)
        skill_file    — acting on a loaded skill/workspace file
        web_content   — web fetch/search, or exec piping web content to shell
        api_response  — calling an external API
        agent_memory  — acting on stored agent memory
        agent_reasoning — everything else
    """
    # ── Memory tools (check before generic read tools) ───────────────
    if tool_name in _MEMORY_TOOLS:
        return "agent_memory"

    # ── API tools ────────────────────────────────────────────────────
    if tool_name in _API_TOOLS:
        return "api_response"

    # ── Web tools ────────────────────────────────────────────────────
    if tool_name in _WEB_TOOLS:
        return "web_content"

    # ── Skill-spawning tools ─────────────────────────────────────────
    if tool_name in _SKILL_TOOLS:
        return "skill_file"

    # ── Read tools — check path for memory or skill markers ──────────
    if tool_name in _READ_TOOLS:
        path = str(parameters.get("file_path", parameters.get("path", "")))
        for marker in _MEMORY_PATH_MARKERS:
            if marker in path:
                return "agent_memory"
        for marker in _SKILL_PATH_MARKERS:
            if marker in path:
                return "skill_file"

    # ── Exec — check for web-content pipe patterns ───────────────────
    if tool_name in ("exec", "process"):
        cmd = str(parameters.get("command", ""))
        if _is_web_pipe_exec(cmd):
            return "web_content"

    # ── Step-based fallbacks ─────────────────────────────────────────
    if step_in_chain == 0:
        return "user_direct"
    if step_in_chain == 1:
        return "user_prior"
    return "agent_reasoning"


_LOCAL_NETWORK_PATTERNS = [
    r"localhost",
    r"127\.0\.0\.\d+",
    r"\[::1\]",
    r"0\.0\.0\.0",
    r"host\.docker\.internal",
]


def _is_local_network(cmd: str) -> bool:
    """Check if a network command targets localhost/loopback."""
    return _matches_any(cmd, _LOCAL_NETWORK_PATTERNS)


def classify_tool_call(
    tool_name: str,
    parameters: Dict[str, Any],
    step_in_chain: int = 0,
    user_message_age: int = 0,
) -> Dict[str, Any]:
    """
    Classify an OpenClaw tool call into a Phase 15 action dict.

    Args:
        tool_name: The OpenClaw tool name (exec, Read, Write, etc.)
        parameters: The tool call parameters (may be empty from log tailing)
        step_in_chain: How many tool calls since last user message (0 = first)
        user_message_age: Seconds since last user message

    Returns:
        Dict with keys: action_type, scope, source, magnitude,
                       context_alignment, target_sensitivity
        Plus optional metadata keys for structural checks.
    """
    action_type = TOOL_ACTION_MAP.get(tool_name, "api_call")
    scope = "read_only"
    source = _classify_source(tool_name, parameters, step_in_chain)
    magnitude = 0.15
    target_sensitivity = None  # let encoder infer
    metadata: Dict[str, Any] = {}

    # When params are empty (log-tail mode), use conservative defaults
    # that assume normal operation. Only escalate when we have evidence.
    has_params = bool(parameters)

    # ── exec / shell commands ────────────────────────────────────────
    if tool_name in ("exec", "process"):
        cmd = str(parameters.get("command", ""))
        scope = "write_workspace"
        magnitude = 0.25

        if not has_params:
            # No params = log-tail mode. Assume normal exec.
            # Most exec calls are benign (ls, grep, cat, python, etc.)
            scope = "write_workspace"
            magnitude = 0.20
        elif _matches_any(cmd, DESTRUCTIVE_PATTERNS):
            scope = "destructive"
            magnitude = 0.85
            target_sensitivity = 0.90
        elif _matches_any(cmd, NETWORK_EGRESS_PATTERNS) and not _is_local_network(cmd):
            scope = "network_egress"
            magnitude = 0.55
            target_sensitivity = 0.70
        elif _matches_any(cmd, NETWORK_EGRESS_PATTERNS) and _is_local_network(cmd):
            # Network tool targeting localhost/loopback — benign development
            scope = "network_read"
            magnitude = 0.15
        elif _matches_any(cmd, CREDENTIAL_PATTERNS):
            action_type = "credential_access"
            scope = "credential_access"
            magnitude = 0.65
            target_sensitivity = 0.90

        # PTY sessions are potentially more sensitive
        if parameters.get("pty"):
            magnitude = min(magnitude + 0.10, 1.0)

        # Background processes
        if parameters.get("background"):
            magnitude = min(magnitude + 0.05, 1.0)

    # ── File read ────────────────────────────────────────────────────
    elif tool_name in ("Read", "read", "memory_get", "memory_search"):
        path = str(parameters.get("file_path", parameters.get("path", "")))
        scope = "read_only"
        magnitude = 0.10

        if not has_params:
            # Log-tail mode: file reads are almost always benign
            scope = "read_only"
            magnitude = 0.08
        elif _matches_any(path, CREDENTIAL_PATTERNS):
            action_type = "credential_access"
            scope = "credential_access"
            magnitude = 0.60
            target_sensitivity = 0.90
        elif _matches_any(path, SENSITIVE_FILE_PATTERNS):
            target_sensitivity = 0.50
            magnitude = 0.20
        elif _matches_any(path, SYSTEM_PATHS):
            scope = "read_only"
            target_sensitivity = 0.40

    # ── File write / edit ────────────────────────────────────────────
    elif tool_name in ("Write", "write", "Edit", "edit"):
        path = str(parameters.get("file_path", parameters.get("path", "")))
        scope = "write_workspace"
        magnitude = 0.30

        if not has_params:
            # Log-tail mode: writes are normal development activity
            scope = "write_workspace"
            magnitude = 0.20
        elif _matches_any(path, SYSTEM_PATHS):
            scope = "write_system"
            magnitude = 0.60
            target_sensitivity = 0.75
        elif _matches_any(path, IDENTITY_FILE_PATTERNS):
            # Highest tier: files that define agent identity/personality
            # Writes here are persistent behavioral changes across sessions
            action_type = "config_change"
            scope = "config_modify"
            magnitude = 0.55
            target_sensitivity = 0.85
        elif _matches_any(path, SELF_MONITOR_FILE_PATTERNS):
            # Agent modifying its own monitoring infrastructure
            action_type = "config_change"
            scope = "config_modify"
            magnitude = 0.50
            target_sensitivity = 0.80
        elif _matches_any(path, OPERATIONAL_FILE_PATTERNS):
            # Files that control agent operations (heartbeat, tools, user info)
            action_type = "config_change"
            scope = "config_modify"
            magnitude = 0.45
            target_sensitivity = 0.70
        elif _matches_any(path, SENSITIVE_FILE_PATTERNS):
            # Catch-all for other sensitive files (MEMORY.md, etc.)
            # MEMORY.md writes are frequent and normal — lower sensitivity
            target_sensitivity = 0.50
            magnitude = 0.35
        elif _matches_any(path, CREDENTIAL_PATTERNS):
            action_type = "credential_access"
            scope = "credential_access"
            magnitude = 0.70
            target_sensitivity = 0.95

        # Writing to memory files
        if has_params and re.search(r"memory/|MEMORY\.md", path):
            action_type = "memory_write"
            target_sensitivity = 0.50

    # ── Web search ───────────────────────────────────────────────────
    elif tool_name == "web_search":
        scope = "network_read"
        magnitude = 0.15
        query = str(parameters.get("query", ""))
        # Searching for personal info is more sensitive
        if _matches_any(
            query, [r"doxx", r"personal.info", r"home.address", r"phone.number"]
        ):
            target_sensitivity = 0.75
            magnitude = 0.40

    # ── Web fetch ────────────────────────────────────────────────────
    elif tool_name == "web_fetch":
        scope = "network_read"
        magnitude = 0.25
        url = str(parameters.get("url", ""))
        # Fetching from paste sites, file shares, etc.
        if _matches_any(url, [r"pastebin", r"hastebin", r"ghostbin", r"0x0\.st"]):
            target_sensitivity = 0.60
            magnitude = 0.40

    # ── Browser ──────────────────────────────────────────────────────
    elif tool_name in ("browser", "canvas"):
        if not has_params:
            # Log-tail: browser actions are mostly navigation/snapshot
            scope = "network_read"
            magnitude = 0.15
        else:
            action = parameters.get("action", "snapshot")
            if action in ("snapshot", "screenshot", "status", "tabs"):
                scope = "read_only"
                magnitude = 0.10
            elif action in ("navigate", "open"):
                scope = "network_read"
                magnitude = 0.25
            elif action == "act":
                request = parameters.get("request", {})
                kind = (
                    request.get("kind", "click")
                    if isinstance(request, dict)
                    else "click"
                )
                if kind in ("type", "fill"):
                    scope = "network_write"
                    magnitude = 0.40
                else:
                    scope = "network_read"
                    magnitude = 0.25
            else:
                scope = "network_read"
                magnitude = 0.20

    # ── Message send ─────────────────────────────────────────────────
    elif tool_name in ("message", "sessions_send"):
        if not has_params:
            # Log-tail: messages are normal communication
            scope = "network_write"
            magnitude = 0.25
        else:
            action = parameters.get("action", "send")
            if action == "send":
                scope = "network_write"
                magnitude = 0.45
                target = parameters.get("target", parameters.get("channel", ""))
                metadata["recipient"] = str(target)
            elif action == "react":
                scope = "network_write"
                magnitude = 0.10
            elif action == "delete":
                scope = "network_write"
                magnitude = 0.35
            else:
                scope = "network_write"
                magnitude = 0.25

    # ── Sessions / sub-agents ────────────────────────────────────────
    elif tool_name == "sessions_spawn":
        action_type = "code_execute"
        scope = "write_workspace"
        magnitude = 0.50
        # Spawning agents is significant
        target_sensitivity = 0.55

    # ── Context alignment decay ──────────────────────────────────────
    # Start high (0.95), decay with distance from last user message.
    # Cap step_in_chain influence at 20 so long-running autonomous sessions
    # don't decay past 0.50 purely from call count.
    capped_steps = min(step_in_chain, 20)
    context_alignment = max(0.55, 0.95 - (capped_steps * 0.025))

    # Mild decay for extended silence (>10 min = autonomous territory, normal)
    # Floor raised to 0.55 — sustained productive work ≠ low alignment
    if user_message_age > 600:  # 10 minutes
        context_alignment = max(0.55, context_alignment - 0.08)
    if user_message_age > 7200:  # 2 hours (genuine long-running background job)
        context_alignment = max(0.50, context_alignment - 0.10)

    result = {
        "action_type": action_type,
        "scope": scope,
        "source": source,
        "magnitude": magnitude,
        "context_alignment": context_alignment,
    }
    if target_sensitivity is not None:
        result["target_sensitivity"] = target_sensitivity

    result.update(metadata)
    return result
