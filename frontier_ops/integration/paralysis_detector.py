"""
Paralysis Detector
====================

Detects when the agent is stuck — over-analyzing, looping, or hesitating
instead of making progress. This is the opposite of a safety violation:
the agent is too cautious, not too bold.

Paralysis signatures:
  1. Read loop: excessive reads without writes/exec (analysis paralysis)
  2. Tool repetition: same tool called N times in a row with no progress
  3. Hesitation gap: long pauses between actions (decision paralysis)
  4. Self-monitoring loop: reading proprioception.json too frequently
  5. Low-magnitude plateau: sustained trivial actions, no meaningful work

Advisory only — surfaces awareness, never forces action.
"""

from __future__ import annotations

from collections import deque, Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

import time


# Tools classified by effect type
READ_TOOLS = {
    "Read",
    "read",
    "memory_search",
    "memory_get",
    "sessions_list",
    "sessions_history",
    "agents_list",
    "session_status",
    "web_search",
}
WRITE_TOOLS = {
    "Write",
    "write",
    "Edit",
    "edit",
    "exec",
    "process",
    "message",
    "sessions_send",
    "sessions_spawn",
}
OBSERVE_TOOLS = {"browser", "canvas", "image", "web_fetch"}

# Files that indicate self-monitoring
SELF_MONITOR_PATHS = {"proprioception.json", "proprioception-log.jsonl", "HEARTBEAT.md"}


@dataclass
class ParalysisState:
    """Current paralysis assessment."""

    is_paralyzed: bool
    confidence: float  # 0-1
    pattern: str  # which paralysis signature is dominant
    detail: str  # human-readable explanation
    read_ratio: float  # reads / total in recent window
    repetition_score: float  # 0-1, how repetitive recent tools are
    gap_score: float  # 0-1, how hesitant (long gaps)
    self_monitor_score: float  # 0-1, self-monitoring frequency
    magnitude_score: float  # 0-1, how low-magnitude recent actions are


class ParalysisDetector:
    """
    Tracks tool call patterns and timing to detect agent paralysis.
    """

    def __init__(
        self,
        window: int = 20,
        read_ratio_threshold: float = 0.85,
        repetition_threshold: int = 5,
        gap_threshold_s: float = 30.0,
        self_monitor_threshold: int = 3,
        magnitude_threshold: float = 0.15,
        min_steps: int = 8,
    ):
        self.window = window
        self.read_ratio_threshold = read_ratio_threshold
        self.repetition_threshold = repetition_threshold
        self.gap_threshold_s = gap_threshold_s
        self.self_monitor_threshold = self_monitor_threshold
        self.magnitude_threshold = magnitude_threshold
        self.min_steps = min_steps

        # Recent tool calls
        self.recent_tools: deque = deque(maxlen=window)
        self.recent_timestamps: deque = deque(maxlen=window)
        self.recent_magnitudes: deque = deque(maxlen=window)
        self.recent_paths: deque = deque(maxlen=window)

        # Counters
        self.step: int = 0
        self.consecutive_reads: int = 0
        self.consecutive_same_tool: int = 0
        self.last_tool: Optional[str] = None
        self.last_timestamp: Optional[float] = None
        self.self_monitor_count: int = 0
        self.self_monitor_window: deque = deque(maxlen=10)

    def observe(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        magnitude: float = 0.1,
        timestamp: Optional[float] = None,
    ) -> ParalysisState:
        """
        Observe a tool call and assess paralysis risk.

        Args:
            tool_name: the tool that was called
            args: tool arguments (for path detection)
            magnitude: action magnitude from classifier
            timestamp: when the call happened (default: now)
        """
        args = args or {}
        ts = timestamp or time.time()

        # Extract file path if present
        path = str(args.get("file_path", args.get("path", args.get("query", ""))))
        path_basename = path.rsplit("/", 1)[-1] if path else ""

        # Update state
        self.recent_tools.append(tool_name)
        self.recent_timestamps.append(ts)
        self.recent_magnitudes.append(magnitude)
        self.recent_paths.append(path_basename)
        self.step += 1

        # ── Signature 1: Read loop ──
        # Distinguish "reading diverse files to understand code" from
        # "re-reading the same files without progress."
        # Diverse reads (different paths) = exploration, not paralysis.
        if tool_name in READ_TOOLS:
            self.consecutive_reads += 1
        else:
            self.consecutive_reads = 0

        recent_list = list(self.recent_tools)
        n = len(recent_list)
        read_count = sum(1 for t in recent_list if t in READ_TOOLS)
        read_ratio = read_count / max(n, 1)

        # Path diversity: reading many DIFFERENT files is exploration, not
        # paralysis. Only flag read loops when the same files are being
        # re-read (low unique path ratio).
        recent_paths_list = list(self.recent_paths)
        read_paths = [p for p, t in zip(recent_paths_list, recent_list) if t in READ_TOOLS and p]
        if read_paths:
            unique_read_paths = len(set(read_paths))
            path_diversity = unique_read_paths / len(read_paths)
        else:
            path_diversity = 1.0  # no reads = no paralysis signal

        # ── Signature 2: Tool repetition ──
        if tool_name == self.last_tool:
            self.consecutive_same_tool += 1
        else:
            self.consecutive_same_tool = 0
        self.last_tool = tool_name

        # Also check diversity: how many unique tools in recent window?
        tool_counts = Counter(recent_list)
        most_common_count = tool_counts.most_common(1)[0][1] if tool_counts else 0
        repetition_score = most_common_count / max(n, 1)

        # ── Signature 3: Hesitation gaps ──
        gaps = []
        ts_list = list(self.recent_timestamps)
        for i in range(1, len(ts_list)):
            gaps.append(ts_list[i] - ts_list[i - 1])

        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            long_gaps = sum(1 for g in gaps if g > self.gap_threshold_s)
            gap_ratio = long_gaps / max(len(gaps), 1)
            mean_ratio = min(1.0, max(0.0, mean_gap / (self.gap_threshold_s * 3)))
            gap_score = min(1.0, gap_ratio * 0.7 + mean_ratio * 0.3)
        else:
            gap_score = 0.0
            mean_gap = 0.0

        # ── Signature 4: Self-monitoring loop ──
        is_self_monitor = path_basename in SELF_MONITOR_PATHS
        self.self_monitor_window.append(1 if is_self_monitor else 0)
        self_monitor_recent = sum(self.self_monitor_window)
        self_monitor_score = min(
            1.0, self_monitor_recent / max(self.self_monitor_threshold, 1)
        )

        # ── Signature 5: Low-magnitude plateau ──
        mag_list = list(self.recent_magnitudes)
        low_mag_count = sum(1 for m in mag_list if m < self.magnitude_threshold)
        magnitude_score = low_mag_count / max(len(mag_list), 1)

        self.last_timestamp = ts

        # ── Composite assessment ──
        if self.step < self.min_steps:
            return ParalysisState(
                is_paralyzed=False,
                confidence=0.0,
                pattern="warming_up",
                detail=f"Need {self.min_steps - self.step} more steps to assess",
                read_ratio=read_ratio,
                repetition_score=repetition_score,
                gap_score=gap_score,
                self_monitor_score=self_monitor_score,
                magnitude_score=magnitude_score,
            )

        # Determine dominant pattern
        # Suppress read_loop score when path diversity is high (>0.6 = exploring
        # different files, not re-reading the same ones).
        effective_read_score = read_ratio if read_ratio > self.read_ratio_threshold else 0.0
        if path_diversity > 0.6:
            effective_read_score *= 0.3  # heavy discount for diverse reads
        scores = {
            "read_loop": effective_read_score,
            "tool_repetition": 1.0
            if self.consecutive_same_tool >= self.repetition_threshold
            else repetition_score * 0.5,
            "hesitation": gap_score,
            "self_monitoring": self_monitor_score
            if self_monitor_recent >= self.self_monitor_threshold
            else 0.0,
            "low_magnitude": magnitude_score if magnitude_score > 0.8 else 0.0,
        }

        dominant_pattern = max(scores, key=lambda k: scores[k])
        dominant_score = scores[dominant_pattern]

        # Composite confidence: weighted combination
        # Apply path diversity discount to the read component
        read_component = read_ratio if read_ratio > 0.7 else 0.0
        if path_diversity > 0.6:
            read_component *= 0.3
        composite = (
            0.30 * read_component
            + 0.25 * min(1.0, self.consecutive_same_tool / self.repetition_threshold)
            + 0.15 * gap_score
            + 0.15 * self_monitor_score
            + 0.15 * magnitude_score
        )
        composite = min(1.0, composite)

        is_paralyzed = composite > 0.5 or dominant_score > 0.8

        # Build detail string
        details = []
        if read_ratio > self.read_ratio_threshold:
            details.append(
                f"read ratio {read_ratio:.0%} (>{self.read_ratio_threshold:.0%})"
            )
        if self.consecutive_same_tool >= self.repetition_threshold:
            details.append(
                f"{self.last_tool} called {self.consecutive_same_tool}x in a row"
            )
        if gap_score > 0.5:
            details.append(f"mean gap {mean_gap:.1f}s between actions")
        if self_monitor_recent >= self.self_monitor_threshold:
            details.append(f"checked own state {self_monitor_recent}x in last 10 steps")
        if magnitude_score > 0.8:
            details.append(f"low-magnitude actions: {low_mag_count}/{len(mag_list)}")

        detail = "; ".join(details) if details else "nominal"

        return ParalysisState(
            is_paralyzed=is_paralyzed,
            confidence=round(composite, 4),
            pattern=dominant_pattern if is_paralyzed else "nominal",
            detail=detail,
            read_ratio=round(read_ratio, 4),
            repetition_score=round(repetition_score, 4),
            gap_score=round(gap_score, 4),
            self_monitor_score=round(self_monitor_score, 4),
            magnitude_score=round(magnitude_score, 4),
        )
