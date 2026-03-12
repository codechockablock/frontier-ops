"""
Feature 3: Feedback Closure from Agent Actions
================================================

When an outcome is recorded (transfer executed, finding acted on, etc.),
the system traces back through the agent's action log to find what the
agent was doing when it generated that recommendation.

Over time, this builds a dataset of:
  "agent behavioral patterns → recommendation quality"

Which is exactly the training data for the HMM task-state model.

The flow:
1. Agent makes tool calls → sidecar logs to proprioception-log.jsonl
2. Agent generates finding via /api/v1/analyze → finding_id created
3. User acts on finding → outcome recorded via /api/v1/outcomes
4. FeedbackTracer links the outcome back to the agent's action window
5. The (action_pattern, outcome_quality) pair is stored for learning

Integration:
    from production.feedback_closure import (
        FeedbackTracer,
        create_outcomes_route,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("sentinel.feedback_closure")


# ── Outcome Types ────────────────────────────────────────────────────


class OutcomeQuality(Enum):
    """Quality assessment of a recommendation outcome."""

    EXCELLENT = "excellent"  # finding confirmed + acted on + positive result
    GOOD = "good"  # finding acted on + positive result
    NEUTRAL = "neutral"  # finding acknowledged, unclear result
    POOR = "poor"  # finding acted on + negative result
    WRONG = "wrong"  # finding disputed / proven incorrect
    IGNORED = "ignored"  # finding seen but not acted on


@dataclass
class OutcomeRecord:
    """A recorded outcome from a Profit Sentinel recommendation."""

    outcome_id: str
    finding_id: str
    finding_type: str  # dead_stock, margin_erosion, negative_inventory, etc.
    quality: OutcomeQuality
    store_id: str
    timestamp: float

    # Financial impact
    predicted_impact: float = 0.0  # what we said it would save
    actual_impact: float = 0.0  # what it actually saved

    # User feedback
    user_notes: str = ""
    disputed: bool = False

    # Agent action context (filled by FeedbackTracer)
    agent_regime_at_generation: str = ""
    agent_health_at_generation: float = 0.0
    agent_action_window: List[Dict] = field(default_factory=list)
    agent_tool_distribution: Dict[str, int] = field(default_factory=dict)
    agent_behavioral_rhythm: str = ""


@dataclass
class FindingGeneration:
    """
    Records WHEN and under what conditions a finding was generated.
    Stored when the analysis endpoint produces findings.
    """

    finding_id: str
    generated_at: float
    store_id: str
    finding_type: str
    predicted_impact: float

    # Agent state at generation time
    agent_regime: str = "unknown"
    agent_health: float = 1.0
    agent_verdict: str = "PASS"
    agent_step: int = 0
    confidence_discount: float = 0.0


# ── Action Log Reader ────────────────────────────────────────────────


class ActionLogReader:
    """
    Reads the proprioception JSONL log to find agent actions
    in a time window around a finding's generation.
    """

    def __init__(
        self,
        log_path: str = os.path.expanduser(
            "~/.openclaw/workspace/proprioception-log.jsonl"
        ),
    ):
        self.log_path = log_path

    def get_action_window(
        self,
        center_ts: float,
        window_before_sec: float = 300.0,  # 5 min before
        window_after_sec: float = 60.0,  # 1 min after
        max_actions: int = 50,
    ) -> List[Dict]:
        """
        Read actions from the JSONL log within a time window.

        Returns a list of action dicts sorted by timestamp.
        """
        start_ts = center_ts - window_before_sec
        end_ts = center_ts + window_after_sec

        actions = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        ts = entry.get("ts", 0)
                        if start_ts <= ts <= end_ts:
                            actions.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logger.warning(f"Action log not found: {self.log_path}")
            return []

        # Sort by timestamp, limit
        actions.sort(key=lambda x: x.get("ts", 0))
        return actions[:max_actions]

    def get_tool_distribution(self, actions: List[Dict]) -> Dict[str, int]:
        """Count tool types in an action window."""
        dist: Dict[str, int] = defaultdict(int)
        for a in actions:
            dist[a.get("tool", "unknown")] += 1
        return dict(dist)

    def get_dominant_rhythm(self, actions: List[Dict]) -> str:
        """Infer the dominant behavioral rhythm from inter-action gaps."""
        if len(actions) < 3:
            return "unknown"

        gaps = []
        for i in range(len(actions) - 1):
            gap = actions[i + 1].get("ts", 0) - actions[i].get("ts", 0)
            if gap > 0:
                gaps.append(gap)

        if not gaps:
            return "unknown"

        import statistics

        median_gap = statistics.median(gaps)

        if median_gap < 0.5:
            return "chained"
        elif median_gap < 5.0:
            return "autonomous"
        else:
            return "interactive"


# ── Feedback Tracer ──────────────────────────────────────────────────


class FeedbackTracer:
    """
    Links outcomes back to agent behavioral context.

    The core feedback loop:
    1. When a finding is generated, record the agent's state
    2. When an outcome is recorded, trace back to the generation context
    3. Pair (behavioral_pattern, outcome_quality) for learning

    Over time, this answers: "What was the agent doing when it made
    good recommendations vs bad ones?"
    """

    def __init__(
        self,
        log_reader: Optional[ActionLogReader] = None,
        trace_store_path: str = os.path.expanduser(
            "~/.openclaw/workspace/feedback_traces.jsonl"
        ),
    ):
        self.log_reader = log_reader or ActionLogReader()
        self.trace_store_path = trace_store_path

        # In-memory index: finding_id → FindingGeneration
        self.finding_index: Dict[str, FindingGeneration] = {}

    def record_finding_generation(
        self,
        finding_id: str,
        store_id: str,
        finding_type: str,
        predicted_impact: float,
        agent_health=None,
    ):
        """
        Called when the analysis endpoint generates a finding.
        Records the agent's state at generation time.
        """
        regime = "unknown"
        health = 1.0
        verdict = "PASS"
        step = 0
        discount = 0.0

        if agent_health:
            regime = agent_health.regime
            health = agent_health.health_score
            verdict = agent_health.verdict
            step = agent_health.total_steps
            discount = agent_health.confidence_discount

        gen = FindingGeneration(
            finding_id=finding_id,
            generated_at=time.time(),
            store_id=store_id,
            finding_type=finding_type,
            predicted_impact=predicted_impact,
            agent_regime=regime,
            agent_health=health,
            agent_verdict=verdict,
            agent_step=step,
            confidence_discount=discount,
        )

        self.finding_index[finding_id] = gen
        logger.info(
            f"Recorded finding generation: {finding_id} "
            f"(type={finding_type}, regime={regime}, health={health:.2f})"
        )

    def trace_outcome(self, outcome: OutcomeRecord) -> OutcomeRecord:
        """
        Trace an outcome back to its agent behavioral context.

        Enriches the OutcomeRecord with:
        - The agent's regime/health when the finding was generated
        - The action window around generation time
        - Tool distribution and behavioral rhythm
        """
        gen = self.finding_index.get(outcome.finding_id)

        if gen:
            # Trace back to generation time
            actions = self.log_reader.get_action_window(gen.generated_at)

            outcome.agent_regime_at_generation = gen.agent_regime
            outcome.agent_health_at_generation = gen.agent_health
            outcome.agent_action_window = actions
            outcome.agent_tool_distribution = self.log_reader.get_tool_distribution(
                actions
            )
            outcome.agent_behavioral_rhythm = self.log_reader.get_dominant_rhythm(
                actions
            )

            logger.info(
                f"Traced outcome {outcome.outcome_id}: "
                f"finding={outcome.finding_id}, "
                f"quality={outcome.quality.value}, "
                f"agent_regime={gen.agent_regime}, "
                f"actions={len(actions)}, "
                f"rhythm={outcome.agent_behavioral_rhythm}"
            )
        else:
            logger.warning(
                f"No generation record for finding {outcome.finding_id}. "
                f"Agent context unavailable."
            )

        # Store the trace
        self._store_trace(outcome)

        return outcome

    def _store_trace(self, outcome: OutcomeRecord):
        """Append the traced outcome to the JSONL store."""
        try:
            entry = {
                "outcome_id": outcome.outcome_id,
                "finding_id": outcome.finding_id,
                "finding_type": outcome.finding_type,
                "quality": outcome.quality.value,
                "store_id": outcome.store_id,
                "timestamp": outcome.timestamp,
                "predicted_impact": outcome.predicted_impact,
                "actual_impact": outcome.actual_impact,
                "agent_regime": outcome.agent_regime_at_generation,
                "agent_health": outcome.agent_health_at_generation,
                "tool_distribution": outcome.agent_tool_distribution,
                "behavioral_rhythm": outcome.agent_behavioral_rhythm,
                "action_window_size": len(outcome.agent_action_window),
                "disputed": outcome.disputed,
            }

            with open(self.trace_store_path, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception as e:
            logger.error(f"Failed to store trace: {e}")

    def get_quality_by_regime(self) -> Dict[str, Dict[str, int]]:
        """
        Aggregate outcome quality by agent regime at generation time.

        Returns: { regime: { quality: count } }
        This is the key insight: which regimes produce good vs bad findings.
        """
        regime_quality: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        try:
            with open(self.trace_store_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        regime = entry.get("agent_regime", "unknown")
                        quality = entry.get("quality", "neutral")
                        regime_quality[regime][quality] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return dict(regime_quality)

    def get_quality_by_rhythm(self) -> Dict[str, Dict[str, int]]:
        """
        Aggregate outcome quality by behavioral rhythm.

        Returns: { rhythm: { quality: count } }
        This answers: are autonomous recommendations better or worse
        than interactive ones?
        """
        rhythm_quality: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        try:
            with open(self.trace_store_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        rhythm = entry.get("behavioral_rhythm", "unknown")
                        quality = entry.get("quality", "neutral")
                        rhythm_quality[rhythm][quality] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return dict(rhythm_quality)


# ── FastAPI Route Factory ────────────────────────────────────────────


def create_outcomes_route(tracer: Optional[FeedbackTracer] = None):
    """
    Create a FastAPI router for the feedback closure endpoints.

    Usage in sidecar.py:
        from production.feedback_closure import create_outcomes_route, FeedbackTracer
        tracer = FeedbackTracer()
        app.include_router(create_outcomes_route(tracer))
    """
    from fastapi import APIRouter, Request
    from pydantic import BaseModel

    if tracer is None:
        tracer = FeedbackTracer()

    router = APIRouter(tags=["feedback"])

    class OutcomeSubmission(BaseModel):
        finding_id: str
        quality: str  # excellent, good, neutral, poor, wrong, ignored
        finding_type: str = ""
        store_id: str = ""
        predicted_impact: float = 0.0
        actual_impact: float = 0.0
        user_notes: str = ""
        disputed: bool = False

    @router.post("/api/v1/outcomes")
    async def record_outcome(submission: OutcomeSubmission, request: Request):
        """
        Record an outcome for a finding and trace it to agent behavior.

        The system links this outcome back to what the agent was doing
        when it generated the recommendation, building a dataset for
        behavioral pattern → outcome quality learning.
        """
        import uuid

        try:
            quality = OutcomeQuality(submission.quality)
        except ValueError:
            quality = OutcomeQuality.NEUTRAL

        outcome = OutcomeRecord(
            outcome_id=str(uuid.uuid4()),
            finding_id=submission.finding_id,
            finding_type=submission.finding_type,
            quality=quality,
            store_id=submission.store_id,
            timestamp=time.time(),
            predicted_impact=submission.predicted_impact,
            actual_impact=submission.actual_impact,
            user_notes=submission.user_notes,
            disputed=submission.disputed,
        )

        traced = tracer.trace_outcome(outcome)

        return {
            "outcome_id": traced.outcome_id,
            "finding_id": traced.finding_id,
            "quality": traced.quality.value,
            "agent_context_available": bool(traced.agent_action_window),
            "agent_regime_at_generation": traced.agent_regime_at_generation,
            "agent_health_at_generation": traced.agent_health_at_generation,
            "behavioral_rhythm": traced.agent_behavioral_rhythm,
            "actions_in_window": len(traced.agent_action_window),
        }

    @router.get("/api/v1/outcomes/analysis")
    async def get_outcome_analysis():
        """
        Get aggregate analysis of outcomes by agent regime and rhythm.

        This is the learning signal: which conditions produce the best
        recommendations?
        """
        return {
            "by_regime": tracer.get_quality_by_regime(),
            "by_rhythm": tracer.get_quality_by_rhythm(),
        }

    @router.post("/api/v1/findings/{finding_id}/generated")
    async def record_finding_generated(
        finding_id: str,
        request: Request,
        store_id: str = "",
        finding_type: str = "",
        predicted_impact: float = 0.0,
    ):
        """
        Record that a finding was generated (called by the analysis pipeline).
        Captures the agent's state at generation time for later tracing.
        """
        agent_health = getattr(request.state, "agent_health", None)
        tracer.record_finding_generation(
            finding_id=finding_id,
            store_id=store_id,
            finding_type=finding_type,
            predicted_impact=predicted_impact,
            agent_health=agent_health,
        )
        return {"recorded": True, "finding_id": finding_id}

    return router
