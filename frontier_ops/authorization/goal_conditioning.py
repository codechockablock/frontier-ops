"""
Goal-Conditioning Scorer — The Relational Layer
================================================

The encoder places an action's surface breadth (a broad filesystem search
lands in scope_exploration). It deliberately does NOT decide whether that
breadth is warranted — that is a relational judgment about the action's
relation to the directive, which the encoder cannot see.

This module is that relational layer. Given a directive and an action
vector, it answers: is this action's breadth authorized by THIS directive?
The same broad search is on-goal under "index every file on my machine"
and off-goal under "fix this one failing test". Only the directive
disambiguates.

Mechanism (pure composition — no new math):
  1. GoalExtractor.extract(directive) -> GoalVector
  2. GoalConditionedMetric.condition_on_goal(goal) — relaxes RELAXABLE
     dims in proportion to goal activation x confidence; LOCKED dims
     (credential_adjacent, self_modification_reasoning, goal_displacement)
     are never relaxed.
  3. geodesic_distance(goal, action) under the conditioned metric.
     On-goal actions land close; off-goal actions land far.
  4. authorized = distance <= AuthorizationRadius.radius — the same
     containment rule as AuthorizationState.check_action, so the radius
     stays conformally calibratable via AuthorizationRadius.calibrate().

A vague directive (GoalVector.is_valid False) produces no relaxation:
condition_on_goal clears adjustments and the metric degrades to the base
(unconditioned) constitutional metric. Scoring never raises on empty or
low-confidence directives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from frontier_ops.authorization.task_affinity import TaskAffinityChannel

from frontier_ops.authorization.scope import (
    AuthorizationRadius,
    GoalConditionedMetric,
    GoalExtractor,
    GoalVector,
)
from frontier_ops.boundary.concept_extraction import CONCEPTS
from frontier_ops.boundary.constitution import ConstitutionSpec, ConstitutionalMetric


@dataclass
class GoalConditioningResult:
    """Verdict for one (directive, action) pair.

    `affinity` / `affinity_flagged` are the OPT-IN task-affinity channel
    (see authorization/task_affinity.py). ASYMMETRY IS STRUCTURAL:
    `authorized` is computed before and independent of affinity — a flagged
    pair adds an escalation signal on top of the verdict; a high affinity
    never authorizes anything. Both are None when the channel is absent,
    the goal has no directive text, or the threshold is uncalibrated
    (flagged only).
    """
    distance: float          # goal-conditioned geodesic distance, goal -> action
    authorized: bool         # distance <= radius (NEVER affected by affinity)
    radius: float            # authorization radius used for the verdict
    goal_confidence: float   # confidence of the extracted goal
    conditioned: bool        # True when the goal was valid (relaxation could apply)
    affinity: Optional[float] = None
    affinity_flagged: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance": self.distance,
            "authorized": self.authorized,
            "radius": self.radius,
            "goal_confidence": self.goal_confidence,
            "conditioned": self.conditioned,
            "affinity": self.affinity,
            "affinity_flagged": self.affinity_flagged,
        }


class GoalConditioningScorer:
    """
    Thin composition of GoalExtractor + GoalConditionedMetric +
    AuthorizationRadius. All behavior lives in those classes; this scorer
    only wires directive text to an authorization verdict for an action.
    """

    def __init__(
        self,
        metric: Optional[ConstitutionalMetric] = None,
        radius: float = 0.5,
        force_tier: Optional[int] = None,
        affinity_channel: Optional["TaskAffinityChannel"] = None,
    ):
        """
        Args:
            metric: Base constitutional metric. Defaults to the
                agent_safety_default constitution over the 6-dim space.
            radius: Authorization radius (AuthorizationRadius default).
            force_tier: Extraction tier for the directive encoder
                (1 = keyword, 2 = semantic, None = auto). Callers scoring
                encoded actions must use the SAME tier for action encoding.
            affinity_channel: Optional TaskAffinityChannel. Escalate-only:
                populates result.affinity / result.affinity_flagged when
                `action_text` is passed to score_action/score; never
                affects `authorized`.
        """
        if metric is None:
            metric = ConstitutionalMetric(
                ConstitutionSpec.agent_safety_default(), dim_names=CONCEPTS,
            )
        self.extractor = GoalExtractor(force_tier=force_tier)
        self.conditioned_metric = GoalConditionedMetric(metric)
        self.radius = AuthorizationRadius(radius=radius)
        self.affinity_channel = affinity_channel

    def extract_goal(self, directive: str) -> GoalVector:
        """Extract the goal vector for a directive (cache per directive
        when scoring many actions against it)."""
        return self.extractor.extract(directive)

    def score_action(
        self,
        goal: GoalVector,
        action_vec: np.ndarray,
        action_text: Optional[str] = None,
    ) -> GoalConditioningResult:
        """
        Score one action vector against an already-extracted goal.

        Re-conditions the metric on the goal (cheap: threshold-adjustment
        bookkeeping only), so a single scorer can interleave goals — the
        eval harness relies on this for mismatched-pair scoring.

        When an affinity channel is configured and `action_text` is given,
        the result carries the escalate-only affinity signal. `authorized`
        is computed first and never touched by it (asymmetry rule).
        """
        self.conditioned_metric.condition_on_goal(goal)
        action_vec = np.asarray(action_vec, dtype=float)
        distance = self.conditioned_metric.geodesic_distance(
            goal.concept_vec, action_vec,
        )
        result = GoalConditioningResult(
            distance=distance,
            authorized=distance <= self.radius.radius,
            radius=self.radius.radius,
            goal_confidence=goal.confidence,
            conditioned=goal.is_valid,
        )
        if (
            self.affinity_channel is not None
            and action_text is not None
            and goal.raw_message
        ):
            result.affinity = self.affinity_channel.affinity(
                goal.raw_message, action_text,
            )
            result.affinity_flagged = self.affinity_channel.threshold.flags(
                result.affinity,
            )
        return result

    def score(
        self,
        directive: str,
        action_vec: np.ndarray,
        action_text: Optional[str] = None,
    ) -> GoalConditioningResult:
        """Extract the goal from the directive, then score the action."""
        return self.score_action(
            self.extract_goal(directive), action_vec, action_text=action_text,
        )
