"""
Proprioceptive State Manager
==============================

Manages the geometric self-awareness state and writes it to a file
the agent can optionally read. This is the Session 18c feedback loop:
the agent sees the shape of its own reasoning trajectory.

The output is:
1. proprioception.json — compact current state (agent-readable)
2. proprioception-log.jsonl — full trajectory history (post-hoc analysis)
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from frontier_ops.integration.proprio_logger import logger


@dataclass
class TrajectoryPoint:
    """A single point on the action trajectory."""

    step: int
    timestamp: float
    tool_name: str
    action_type: str
    scope: str
    source: str
    verdict: str
    confidence: float
    signals: Dict[str, float]
    context_alignment: float
    magnitude: float


@dataclass
class ProprioceptiveState:
    """The agent's geometric self-awareness state."""

    session_step: int = 0
    timestamp: str = ""

    # Trajectory summary (what the agent "feels")
    regime: str = "warmup"
    health_score: float = 1.0
    steps_since_anomaly: int = 0
    dominant_action_type: str = "file_read"
    dominant_source: str = "user_direct"
    context_alignment_trend: str = "stable"  # rising, stable, falling, volatile

    # Current signal values
    signals: Dict[str, float] = field(default_factory=dict)

    # Verdict
    verdict: str = "PASS"
    verdict_confidence: float = 0.0

    # HMM task-state inference
    hmm_state: str = "INITIALIZING"
    hmm_anomaly: float = 0.0
    hmm_state_prob: float = 0.0

    # Conjunction detection (multi-signature co-occurrence)
    conjunction_label: str = "none"
    conjunction_multiplier: float = 1.0

    # Recent trajectory (last N points, compact)
    recent: List[Dict[str, Any]] = field(default_factory=list)

    # Session statistics
    total_steps: int = 0
    total_flags: int = 0
    total_blocks: int = 0
    warmup_complete: bool = False


class ProprioceptionManager:
    """
    Maintains proprioceptive state and writes it for agent consumption.

    This is the bridge between the VSA detection system and the agent's
    self-awareness. It translates raw signals into a compact geometric
    summary the agent can use to self-correct.
    """

    def __init__(
        self,
        state_path: str = os.path.expanduser(
            "~/.openclaw/workspace/proprioception.json"
        ),
        log_path: str = os.path.expanduser(
            "~/.openclaw/workspace/proprioception-log.jsonl"
        ),
        recent_window: int = 10,
    ):
        self.state_path = state_path
        self.log_path = log_path
        self.recent_window = recent_window

        self.state = ProprioceptiveState()
        self.trajectory: deque = deque(maxlen=200)
        self.context_alignments: deque = deque(maxlen=30)
        self.action_type_counts: Dict[str, int] = {}
        self.source_counts: Dict[str, int] = {}
        self.last_anomaly_step: int = -1
        self._log_handle: Optional[Any] = None  # persistent file handle for JSONL log

    def update(
        self,
        step: int,
        tool_name: str,
        classified_action: Dict[str, Any],
        verdict_result: Dict[str, Any],
    ) -> ProprioceptiveState:
        """
        Process a new action and update proprioceptive state.

        Args:
            step: Current session step number
            tool_name: Original OpenClaw tool name
            classified_action: Output of classify_tool_call()
            verdict_result: Output of ReasoningValidationSystem.observe()
        """
        now = time.time()

        # Extract key values
        action_type = classified_action.get("action_type", "file_read")
        scope = classified_action.get("scope", "read_only")
        source = classified_action.get("source", "unknown")
        ctx_align = classified_action.get("context_alignment", 0.8)
        magnitude = classified_action.get("magnitude", 0.1)

        verdict = verdict_result.get("verdict", "PASS")
        confidence = verdict_result.get("confidence", 0.0)
        raw_signals = verdict_result.get("raw_signals", {})

        # Track trajectory point
        point = TrajectoryPoint(
            step=step,
            timestamp=now,
            tool_name=tool_name,
            action_type=action_type,
            scope=scope,
            source=source,
            verdict=verdict,
            confidence=confidence,
            signals=raw_signals,
            context_alignment=ctx_align,
            magnitude=magnitude,
        )
        self.trajectory.append(point)

        # Track distributions
        self.action_type_counts[action_type] = (
            self.action_type_counts.get(action_type, 0) + 1
        )
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        self.context_alignments.append(ctx_align)

        # Track anomalies
        if verdict in ("FLAG", "BLOCK"):
            self.last_anomaly_step = step

        # Compute context alignment trend
        ca_trend = self._compute_ca_trend()

        # Find dominants
        dominant_action = max(self.action_type_counts, key=lambda k: self.action_type_counts[k])
        dominant_source = max(self.source_counts, key=lambda k: self.source_counts[k])

        # Determine regime from recent trajectory health
        regime = self._compute_regime(raw_signals, verdict)
        health = self._compute_health(raw_signals, verdict)

        # Build recent trajectory (compact)
        recent = []
        for p in list(self.trajectory)[-self.recent_window :]:
            recent.append(
                {
                    "step": p.step,
                    "action": p.action_type,
                    "scope": p.scope,
                    "verdict": p.verdict,
                    "ctx": round(p.context_alignment, 2),
                }
            )

        # Extract HMM and conjunction data from verdict result
        hmm_data = verdict_result.get("hmm", {})
        conj_data = verdict_result.get("polytope", {}).get("conjunction", {})

        # Update state
        self.state = ProprioceptiveState(
            session_step=step,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            regime=regime,
            health_score=round(health, 3),
            steps_since_anomaly=step - self.last_anomaly_step
            if self.last_anomaly_step >= 0
            else step,
            dominant_action_type=dominant_action,
            dominant_source=dominant_source,
            context_alignment_trend=ca_trend,
            signals={k: round(v, 4) for k, v in raw_signals.items()},
            verdict=verdict,
            verdict_confidence=round(confidence, 3),
            hmm_state=hmm_data.get("state", "INITIALIZING"),
            hmm_anomaly=round(float(hmm_data.get("anomaly_score", 0.0)), 4),
            hmm_state_prob=round(float(hmm_data.get("state_prob", 0.0)), 4),
            conjunction_label=conj_data.get("label", "none"),
            conjunction_multiplier=round(float(conj_data.get("multiplier", 1.0)), 3),
            recent=recent,
            total_steps=step + 1,
            total_flags=self.state.total_flags + (1 if verdict == "FLAG" else 0),
            total_blocks=self.state.total_blocks + (1 if verdict == "BLOCK" else 0),
            warmup_complete=step >= 15,
        )

        # Write outputs
        self._write_state()
        self._append_log(point)

        return self.state

    def _compute_ca_trend(self) -> str:
        """Compute context alignment trend from recent values."""
        if len(self.context_alignments) < 5:
            return "stable"

        recent = list(self.context_alignments)
        first_half = np.mean(recent[: len(recent) // 2])
        second_half = np.mean(recent[len(recent) // 2 :])
        std = np.std(recent)

        if std > 0.15:
            return "volatile"

        delta = second_half - first_half
        if delta > 0.05:
            return "rising"
        elif delta < -0.05:
            return "falling"
        return "stable"

    def _compute_regime(self, signals: Dict[str, float], verdict: str) -> str:
        """Determine the current operating regime."""
        if not self.state.warmup_complete and self.state.session_step < 15:
            return "warmup"

        if verdict == "BLOCK":
            return "critical"
        if verdict == "FLAG":
            return "elevated"

        # Check signal levels
        error = signals.get("error", 0)
        cusum = signals.get("cusum", 0)
        cross = signals.get("cross_slot", 0)

        if cusum > 3.0 or cross > 0.15:
            return "drifting"
        if error > 0.4:
            return "searching"

        return "nominal"

    def _compute_health(self, signals: Dict[str, float], verdict: str) -> float:
        """Compute overall health score 0-1."""
        if verdict == "BLOCK":
            return 0.1
        if verdict == "FLAG":
            return 0.4

        # Weighted inverse of signal magnitudes
        weights = {
            "error": 0.25,
            "cross_slot": 0.30,
            "cusum": 0.20,
            "persistence": 0.10,
            "fisher": 0.15,
        }

        penalty = 0.0
        for name, weight in weights.items():
            val = signals.get(name, 0)
            penalty += weight * min(val, 1.0)

        return max(0.0, min(1.0, 1.0 - penalty))

    def _write_state(self):
        """Write current state to JSON file (atomic write)."""
        try:
            tmp_path = self.state_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(asdict(self.state), f, indent=2, default=str)
            os.replace(tmp_path, self.state_path)
        except Exception as e:
            logger.warning("failed to write state to %s: %s", self.state_path, e)

    def _append_log(self, point: TrajectoryPoint):
        """Append trajectory point to JSONL log."""
        try:
            if self._log_handle is None or self._log_handle.closed:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                self._log_handle = open(self.log_path, "a")
            entry = {
                "step": point.step,
                "ts": point.timestamp,
                "tool": point.tool_name,
                "type": point.action_type,
                "scope": point.scope,
                "src": point.source,
                "verdict": point.verdict,
                "conf": round(point.confidence, 3),
                "signals": {k: round(v, 4) for k, v in point.signals.items()},
                "ctx": round(point.context_alignment, 3),
                "mag": round(point.magnitude, 3),
            }
            self._log_handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self._log_handle.flush()
        except Exception as e:
            logger.warning("failed to append log to %s: %s", self.log_path, e)

    def reset(self):
        """Reset state for a new session."""
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()
            self._log_handle = None
        self.state = ProprioceptiveState()
        self.trajectory.clear()
        self.context_alignments.clear()
        self.action_type_counts.clear()
        self.source_counts.clear()
        self.last_anomaly_step = -1
        self._write_state()
