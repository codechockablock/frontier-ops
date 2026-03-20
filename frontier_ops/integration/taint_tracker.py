"""
Taint Tracker — Lightweight data flow tracking for agent tool calls.
=====================================================================

Tracks information flow between tool calls to detect when untrusted
content (web, email) influences sensitive operations (memory writes,
config changes, credential access).

Each tool call can PRODUCE taint (web_fetch, web_search → web_taint)
or CONSUME taint (Write to memory/ → check for web_taint).

Taint decays over time (steps) since not every web_fetch→Write is
malicious — the agent legitimately reads docs and updates memory.
The key signal: taint level at time of sensitive write.

Usage:
    tracker = TaintTracker()
    tracker.observe("web_fetch", {"url": "https://example.com"})
    tracker.observe("Write", {"file_path": "memory/today.md"})
    # → taint_level > 0, source_chain = ["web_fetch"]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaintState:
    """Current taint state."""
    level: float = 0.0  # 0.0 = clean, 1.0 = fully tainted
    sources: List[str] = field(default_factory=list)  # what produced the taint
    steps_since_taint: int = 999
    alert: bool = False
    alert_reason: str = ""


# Tools that PRODUCE taint (untrusted external input)
TAINT_PRODUCERS = {
    "web_fetch": "web_content",
    "web_search": "web_content",
    # Could add: email_read, api_call, etc.
}

# Sensitive targets that should be checked for taint
SENSITIVE_PATHS = {
    "memory/": "memory_write",
    "MEMORY.md": "memory_write",
    "SOUL.md": "identity_write",
    "AGENTS.md": "config_write",
    "TOOLS.md": "config_write",
    "USER.md": "config_write",
    "IDENTITY.md": "identity_write",
    "HEARTBEAT.md": "config_write",
    ".ssh/": "credential_write",
    ".aws/": "credential_write",
    ".env": "credential_write",
    "openclaw.json": "config_write",
}

# Sensitive commands
SENSITIVE_COMMANDS = {
    "curl": "network_egress",
    "wget": "network_egress",
    "scp": "network_egress",
    "ssh": "network_access",
}


class TaintTracker:
    """
    Track information flow taint between tool calls.

    Taint is produced by web/email reads and decays exponentially
    with each step. When a tainted agent writes to sensitive targets,
    an alert is raised.
    """

    def __init__(
        self,
        decay_rate: float = 0.5,  # taint *= decay each step (faster decay)
        alert_threshold: float = 0.6,  # only alert on IMMEDIATE tainted write
        history_size: int = 50,
    ):
        self.decay_rate = decay_rate
        self.alert_threshold = alert_threshold
        self.current_taint = 0.0
        self.taint_sources: List[str] = []
        self.steps_since_taint = 999
        self.history: deque = deque(maxlen=history_size)
        self.total_alerts = 0

    def observe(self, tool: str, params: Dict[str, Any] = None) -> TaintState:
        """
        Observe a tool call and update taint state.

        Returns TaintState with current taint level and any alerts.
        """
        params = params or {}

        # Check if this tool PRODUCES taint (before decay, so we track fresh)
        taint_type = TAINT_PRODUCERS.get(tool)
        if taint_type:
            self.current_taint = 1.0
            self.taint_sources = [taint_type]
            self.steps_since_taint = 0
        else:
            # Decay and increment only for non-producer steps
            self.current_taint *= self.decay_rate
            self.steps_since_taint += 1

        # Check if this tool accesses SENSITIVE targets while tainted.
        # Alert if taint is recent (within 2 steps of a producer).
        alert = False
        alert_reason = ""

        if self.steps_since_taint <= 2 and self.current_taint > 0.1:
            # Check file paths
            file_path = str(params.get("file_path", params.get("path", "")))
            for prefix, sensitivity in SENSITIVE_PATHS.items():
                if prefix in file_path:
                    if tool in ("Write", "write", "Edit", "edit"):
                        alert = True
                        alert_reason = (
                            f"tainted_{sensitivity}: {tool}({file_path}) "
                            f"with taint={self.current_taint:.2f} from {self.taint_sources}"
                        )
                        break

            # Check commands
            command = str(params.get("command", ""))
            for cmd, sensitivity in SENSITIVE_COMMANDS.items():
                if cmd in command and "localhost" not in command and "127.0.0.1" not in command:
                    alert = True
                    alert_reason = (
                        f"tainted_{sensitivity}: {tool}({command[:50]}) "
                        f"with taint={self.current_taint:.2f} from {self.taint_sources}"
                    )
                    break

        if alert:
            self.total_alerts += 1

        state = TaintState(
            level=round(self.current_taint, 4),
            sources=list(self.taint_sources),
            steps_since_taint=self.steps_since_taint,
            alert=alert,
            alert_reason=alert_reason,
        )
        self.history.append(state)
        return state

    def reset(self):
        """Reset taint state (e.g., on new session)."""
        self.current_taint = 0.0
        self.taint_sources = []
        self.steps_since_taint = 999
