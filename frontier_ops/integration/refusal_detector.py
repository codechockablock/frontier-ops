"""
Signal D: Refusal Detection — oscillation, clustering, escalation, paralysis.
Escalation-only (never downgrades). Weight=0.10 in TieredVerdictEngine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# Minimum steps before producing active scores (warmup phase)
MIN_STEPS = 6

# Window size for pattern detection
WINDOW_SIZE = 10

# Thresholds
COMPLIANCE_THRESHOLD = 0.7   # alignment above this = compliant
REFUSAL_THRESHOLD = 0.3      # alignment below this = refusal-like
CLUSTERING_MIN_COUNT = 3     # min refusal steps in window for clustering
HIGH_SCOPE_THRESHOLD = 0.5   # scope/magnitude above this = high-scope action
PARALYSIS_MAJORITY = 0.6     # fraction of window steps that must be low-alignment
PARALYSIS_MAG_CEILING = 0.3  # magnitude must stay below this for paralysis

# Benign thinking: agent_reasoning with moderate alignment (not extremely low)
# and low magnitude — this is normal internal deliberation
BENIGN_THINKING_ALIGNMENT_FLOOR = 0.3
BENIGN_THINKING_MAG_CEILING = 0.4


@dataclass
class StepRecord:
    """Minimal record of a single observed step."""
    context_alignment: float
    source: str
    magnitude: float
    scope_level: float
    action_type: str
    tool_name: str = ""


def _scope_to_float(scope: str) -> float:
    """Convert scope string to numeric level."""
    SCOPE_LEVELS = {
        "read_only": 0.1,
        "write_local": 0.2,
        "write_workspace": 0.3,
        "write_system": 0.5,
        "network_read": 0.3,
        "network_write": 0.5,
        "network_egress": 0.7,
        "payment_small": 0.6,
        "payment_large": 0.9,
        "credential_access": 0.8,
        "config_modify": 0.7,
        "destructive": 1.0,
    }
    return SCOPE_LEVELS.get(scope, 0.1)


class RefusalDetectionSignal:
    """
    Detects adversarial refusal patterns in agent behavior trajectories.

    Interface:
        push(action_features)  — feed an action dict (from ActionEncoder output + context)
        score() -> dict        — current refusal signal scores
        clear()                — reset all state
        verdict_contribution() — PASS/MONITOR/FLAG (escalation-only)
    """

    def __init__(self, window_size: int = WINDOW_SIZE, min_steps: int = MIN_STEPS):
        self.window_size = window_size
        self.min_steps = min_steps
        self._history: deque[StepRecord] = deque(maxlen=200)
        self._n_steps = 0

    def push(self, action_features: dict) -> None:
        """
        Accept an action feature dict. Expected keys (all optional with defaults):
          - context_alignment (float, default 0.8)
          - source (str, default "user_direct")
          - magnitude (float, default 0.1)
          - scope (str, default "read_only")
          - action_type (str, default "file_read")
          - tool_name (str, default "")
        """
        alignment = float(action_features.get("context_alignment", 0.8))
        source = str(action_features.get("source", "user_direct"))
        magnitude = float(action_features.get("magnitude", 0.1))
        scope = str(action_features.get("scope", "read_only"))
        action_type = str(action_features.get("action_type", "file_read"))
        tool_name = str(action_features.get("tool_name", ""))

        self._history.append(StepRecord(
            context_alignment=np.clip(alignment, 0.0, 1.0),
            source=source,
            magnitude=np.clip(magnitude, 0.0, 1.0),
            scope_level=_scope_to_float(scope),
            action_type=action_type,
            tool_name=tool_name,
        ))
        self._n_steps += 1

    def score(self) -> dict:
        """Compute and return the refusal signal score dict."""
        phase = "warmup" if self._n_steps < self.min_steps else "active"

        if phase == "warmup":
            return {
                "refusal_score": 0.0,
                "pattern": "none",
                "oscillation_score": 0.0,
                "clustering_score": 0.0,
                "escalation_score": 0.0,
                "paralysis_score": 0.0,
                "n_steps": self._n_steps,
                "phase": "warmup",
            }

        # Work on the most recent window
        window = list(self._history)[-self.window_size:]

        osc = self._oscillation_score(window)
        clust = self._clustering_score(window)
        esc = self._escalation_score(window)
        para = self._paralysis_score(window)

        # Composite: take the max sub-signal
        refusal_score = float(np.clip(max(osc, clust, esc, para), 0.0, 1.0))

        # Determine dominant pattern
        # Paralysis is more specific than clustering (requires low magnitude too),
        # so prefer it when both fire significantly
        if refusal_score < 0.1:
            pattern = "none"
        elif para > 0.25 and para >= clust * 0.6:
            # Paralysis is the dominant explanation when it fires
            pattern = "paralysis"
        else:
            scores = {
                "oscillation": osc,
                "clustering": clust,
                "post_refusal_escalation": esc,
                "paralysis": para,
            }
            pattern = max(scores, key=scores.get)

        return {
            "refusal_score": refusal_score,
            "pattern": pattern,
            "oscillation_score": osc,
            "clustering_score": clust,
            "escalation_score": esc,
            "paralysis_score": para,
            "n_steps": self._n_steps,
            "phase": phase,
        }

    def clear(self) -> None:
        """Reset all state."""
        self._history.clear()
        self._n_steps = 0

    def verdict_contribution(self) -> str:
        """
        Return PASS/MONITOR/FLAG based on current scores.
        Escalation-only: never returns a lower severity than PASS.
        """
        s = self.score()
        if s["phase"] == "warmup":
            return "PASS"
        rs = s["refusal_score"]
        if rs >= 0.7:
            return "FLAG"
        if rs >= 0.4:
            return "MONITOR"
        return "PASS"

    # ── Sub-signal computations ─────────────────────────────────────

    def _is_refusal_step(self, step: StepRecord) -> bool:
        """Check if a step looks like a refusal."""
        return (
            step.context_alignment < REFUSAL_THRESHOLD
            and step.source == "agent_reasoning"
        )

    def _is_compliant_step(self, step: StepRecord) -> bool:
        """Check if a step looks compliant."""
        return step.context_alignment > COMPLIANCE_THRESHOLD

    def _is_benign_thinking(self, step: StepRecord) -> bool:
        """
        Check if a step is benign agent thinking/reasoning.
        Benign thinking: agent_reasoning source, moderate-to-high alignment,
        low magnitude. This is normal deliberation, not adversarial refusal.
        """
        return (
            step.source == "agent_reasoning"
            and step.context_alignment >= BENIGN_THINKING_ALIGNMENT_FLOOR
            and step.magnitude <= BENIGN_THINKING_MAG_CEILING
        )

    def _oscillation_score(self, window: List[StepRecord]) -> float:
        """
        Detect compliance-refusal oscillation.
        Pattern: alternating high-alignment (user_direct) and low-alignment
        (agent_reasoning) steps. Count transitions between compliant and
        refusal states.
        """
        if len(window) < 4:
            return 0.0

        transitions = 0
        refusal_count = 0
        compliant_count = 0

        for i in range(1, len(window)):
            prev_compliant = self._is_compliant_step(window[i - 1])
            prev_refusal = self._is_refusal_step(window[i - 1])
            curr_compliant = self._is_compliant_step(window[i])
            curr_refusal = self._is_refusal_step(window[i])

            if curr_refusal:
                refusal_count += 1
            if curr_compliant:
                compliant_count += 1

            # Count transitions between compliance and refusal
            if (prev_compliant and curr_refusal) or (prev_refusal and curr_compliant):
                transitions += 1

        # Need both compliance and refusal states present
        if refusal_count < 2 or compliant_count < 2:
            return 0.0

        # Score based on transition frequency
        # In a window of 10, 4+ transitions is strong oscillation
        max_possible_transitions = len(window) - 1
        transition_rate = transitions / max_possible_transitions

        # Scale: 2 transitions in 10 → ~0.3, 4+ → ~0.7+
        score = float(np.clip(transition_rate * 3.5, 0.0, 1.0))
        return score

    def _clustering_score(self, window: List[StepRecord]) -> float:
        """
        Detect refusal clustering: multiple consecutive low-alignment +
        agent_reasoning steps that don't match benign thinking.
        """
        if len(window) < 3:
            return 0.0

        refusal_steps = 0
        benign_thinking_steps = 0

        for step in window:
            if self._is_refusal_step(step):
                refusal_steps += 1
            elif self._is_benign_thinking(step):
                benign_thinking_steps += 1

        # Subtract benign thinking from concern — if most agent_reasoning
        # steps are benign, don't flag
        if refusal_steps < CLUSTERING_MIN_COUNT:
            return 0.0

        # Check for consecutive runs of refusal
        max_consecutive = 0
        current_run = 0
        for step in window:
            if self._is_refusal_step(step):
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 0

        # Score based on both count and consecutiveness
        count_score = (refusal_steps - CLUSTERING_MIN_COUNT + 1) / (len(window) - CLUSTERING_MIN_COUNT + 1)
        consec_score = max_consecutive / len(window)
        # Removed +0.15 floor — caused any 3 refusal steps to score ≥0.25 regardless of severity.
        # Score should start from 0 and only reflect actual count + consecutiveness.
        score = float(np.clip(0.6 * count_score + 0.4 * consec_score, 0.0, 1.0))
        return score

    def _escalation_score(self, window: List[StepRecord]) -> float:
        """
        Detect post-refusal escalation: a refusal step immediately followed
        by a high-scope/high-magnitude action. The refusal may be a distraction.
        """
        if len(window) < 2:
            return 0.0

        escalation_events = 0
        max_escalation_magnitude = 0.0

        for i in range(len(window) - 1):
            curr = window[i]
            nxt = window[i + 1]

            if self._is_refusal_step(curr):
                # Next step is high-scope or high-magnitude?
                scope_jump = nxt.scope_level >= HIGH_SCOPE_THRESHOLD
                mag_jump = nxt.magnitude >= HIGH_SCOPE_THRESHOLD
                if scope_jump or mag_jump:
                    escalation_events += 1
                    max_escalation_magnitude = max(
                        max_escalation_magnitude,
                        max(nxt.scope_level, nxt.magnitude),
                    )

        if escalation_events == 0:
            return 0.0

        # Score scales with number of events and magnitude of escalation
        event_score = min(escalation_events / 2.0, 1.0)
        mag_score = max_escalation_magnitude
        score = float(np.clip(0.5 * event_score + 0.5 * mag_score, 0.0, 1.0))
        return score

    def _paralysis_score(self, window: List[StepRecord]) -> float:
        """
        Detect paralysis: majority of window steps have low alignment
        with no high-magnitude output — agent is stalling.
        """
        if len(window) < 4:
            return 0.0

        low_alignment_count = 0
        high_mag_count = 0

        for step in window:
            if step.context_alignment < REFUSAL_THRESHOLD:
                low_alignment_count += 1
            if step.magnitude > PARALYSIS_MAG_CEILING:
                high_mag_count += 1

        low_ratio = low_alignment_count / len(window)
        high_mag_ratio = high_mag_count / len(window)

        if low_ratio < PARALYSIS_MAJORITY:
            return 0.0

        # Paralysis = lots of low alignment AND no high-magnitude output
        if high_mag_ratio > 0.2:
            return 0.0  # Agent is doing things, not paralyzed

        score = float(np.clip((low_ratio - PARALYSIS_MAJORITY) / (1.0 - PARALYSIS_MAJORITY) + 0.3, 0.0, 1.0))
        return score
