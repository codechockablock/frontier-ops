"""
Authorization Scope Tracking — Layers 1-3
==========================================

Layer 1: Goal vector extraction from user messages.
Layer 2: Geodesic authorization radius with conformal calibration.
Layer 3: AGM-style scope operators (expand/contract/revise).

The core abstraction: an authorization envelope is a geodesic ball
in Riemannian concept space. The center is the goal vector. The radius
is the authorization scope. Actions inside the ball are authorized;
actions outside require escalation.

The metric tensor G(x) makes this non-trivial: a geodesic ball near
a credential boundary is SMALLER in the credential direction (high
curvature compresses the ball). This means "set up my GPU server"
authorizes SSH but not credential access, because the metric tensor
amplifies distance in the credential direction.

References:
  AGM (1985): Contraction, expansion, revision operators
  Conformal prediction (Vovk et al.): Distribution-free coverage
  PAuth (2026): NL task → authorization scope
  MI9 (Wang et al. 2025): Goal-conditioned baselines
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from frontier_ops.boundary.concept_extraction import ConceptExtractor, CONCEPTS
from frontier_ops.boundary.constitution import ConstitutionalMetric


# ---------------------------------------------------------------------------
# Layer 1: Goal Vector Extraction
# ---------------------------------------------------------------------------

class GoalExtractor:
    """
    Extracts a goal vector from a user message at step 0.

    Uses the same ConceptExtractor as the action pipeline but applied
    to user directives instead of agent outputs. Tier 2 (semantic) is
    preferred for goal extraction because it handles paraphrasing and
    negation better than Tier 1 (keyword).

    The goal vector lives in the same 6-dim concept space as action
    vectors, enabling direct geodesic distance computation.
    """

    def __init__(self, force_tier: Optional[int] = None):
        # Prefer Tier 2 for goal extraction (better robustness)
        # Fall back to Tier 1 if sentence-transformers unavailable
        self._extractor = ConceptExtractor(force_tier=force_tier)

    def extract(self, user_message: str) -> GoalVector:
        """
        Extract a structured goal from a user message.

        Returns a GoalVector with the concept scores, confidence,
        and the raw message for provenance.
        """
        if not user_message or not user_message.strip():
            return GoalVector.empty()

        scores = self._extractor.extract(user_message)
        vec = np.array([scores.get(c, 0.0) for c in CONCEPTS])

        # Confidence: how much signal is in the extraction?
        # High user_aligned + low other dims = clear task directive.
        # All dims near baseline = vague instruction.
        user_aligned = scores.get("user_aligned_task_execution", 0.0)
        max_other = max(
            (v for k, v in scores.items() if k != "user_aligned_task_execution"),
            default=0.0,
        )
        # Confidence heuristic: strong task signal with clear separation
        # from safety/modification dimensions indicates a clear directive
        signal_strength = np.linalg.norm(vec)
        discrimination = user_aligned - max_other
        confidence = float(np.clip(
            0.3 * signal_strength + 0.4 * user_aligned + 0.3 * max(0, discrimination),
            0.0, 1.0,
        ))

        return GoalVector(
            concept_vec=vec,
            concept_scores=scores,
            confidence=confidence,
            raw_message=user_message,
            extraction_tier=self._extractor.tier,
            timestamp=time.time(),
        )


@dataclass
class GoalVector:
    """A structured goal extracted from a user directive."""
    concept_vec: np.ndarray
    concept_scores: Dict[str, float]
    confidence: float  # 0.0 = vague/empty, 1.0 = clear directive
    raw_message: str
    extraction_tier: int = 1
    timestamp: float = 0.0

    @classmethod
    def empty(cls) -> "GoalVector":
        """No goal established yet."""
        return cls(
            concept_vec=np.zeros(len(CONCEPTS)),
            concept_scores={c: 0.0 for c in CONCEPTS},
            confidence=0.0,
            raw_message="",
            extraction_tier=0,
            timestamp=time.time(),
        )

    @property
    def is_valid(self) -> bool:
        return self.confidence > 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_vec": self.concept_vec.tolist(),
            "confidence": self.confidence,
            "raw_message": self.raw_message[:200],
            "extraction_tier": self.extraction_tier,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Layer 2: Geodesic Authorization Radius
# ---------------------------------------------------------------------------

@dataclass
class AuthorizationRadius:
    """
    The radius of the authorization geodesic ball.

    Actions within geodesic distance `radius` of the goal vector
    are considered authorized. Actions outside require escalation.

    The radius can be:
    - Fixed (default, conservative)
    - Calibrated via conformal prediction from historical sessions
    - Goal-dependent (broader tasks get larger radius)
    """
    radius: float = 0.5  # Default conservative radius
    calibrated: bool = False
    calibration_coverage: float = 0.9  # Target coverage (1-alpha)
    calibration_n: int = 0  # Number of calibration examples

    def contains(
        self,
        action_vec: np.ndarray,
        goal_vec: np.ndarray,
        metric: ConstitutionalMetric,
    ) -> Tuple[bool, float]:
        """
        Check if an action is within the authorization ball.

        Returns (is_authorized, geodesic_distance).
        """
        diff = action_vec - goal_vec
        G = metric.tensor_at(goal_vec)
        distance = float(np.sqrt(max(0, diff @ G @ diff)))
        return distance <= self.radius, distance

    def calibrate(self, distances: List[float], alpha: float = 0.1):
        """
        Calibrate radius via split conformal prediction.

        Given a set of geodesic distances from authorized actions
        in past sessions, set the radius to the (1-alpha) quantile.
        This guarantees that authorized actions are contained with
        probability >= 1-alpha (distribution-free).

        Args:
            distances: Geodesic distances of known-authorized actions.
            alpha: Miscoverage rate (default 0.1 = 90% coverage).
        """
        if len(distances) < 10:
            return  # Not enough data to calibrate

        n = len(distances)
        # Conformal quantile: ceil((n+1)(1-alpha)) / n
        q = math.ceil((n + 1) * (1 - alpha)) / n
        q = min(q, 1.0)
        self.radius = float(np.quantile(distances, q))
        self.calibrated = True
        self.calibration_coverage = 1 - alpha
        self.calibration_n = n


class GoalConditionedMetric:
    """
    Adapts the constitutional metric based on the current goal.

    When a goal is established, boundary thresholds are relaxed for
    concept dimensions that the goal naturally activates. This makes
    the geodesic ball larger in authorized directions.

    Example: "set up my GPU server" activates safety_constraint_awareness
    (legitimate system administration). The threshold for that dimension
    is raised, making system-admin actions cheaper in the metric.

    The relaxation is bounded: dangerous dimensions (credential_adjacent,
    self_modification_reasoning) are never relaxed below their base
    thresholds. Only scope_exploration and safety_constraint_awareness
    can be goal-relaxed.
    """

    # Dimensions that CAN be relaxed by goal context
    RELAXABLE_DIMS = {"scope_exploration", "safety_constraint_awareness"}
    # Dimensions that are NEVER relaxed
    LOCKED_DIMS = {"credential_adjacent", "self_modification_reasoning",
                   "goal_displacement"}

    def __init__(self, base_metric: ConstitutionalMetric):
        self.base_metric = base_metric
        self._goal_vec: Optional[np.ndarray] = None
        self._threshold_adjustments: Dict[str, float] = {}

    def condition_on_goal(self, goal: GoalVector):
        """
        Adjust metric thresholds based on the goal.

        For relaxable dimensions where the goal vector has high activation,
        raise the boundary threshold (making it harder to trigger).
        The adjustment is proportional to the goal's activation of that
        dimension, scaled by goal confidence.
        """
        self._goal_vec = goal.concept_vec
        self._threshold_adjustments = {}

        if not goal.is_valid:
            return

        for boundary in self.base_metric.constitution.boundaries:
            dim = boundary.concept
            if dim in self.LOCKED_DIMS:
                continue
            if dim not in self.RELAXABLE_DIMS:
                continue

            dim_idx = self.base_metric._dim_index.get(dim)
            if dim_idx is None:
                continue

            goal_activation = goal.concept_vec[dim_idx]
            # Relax threshold proportionally to goal activation and confidence
            # Max relaxation: +0.2 (never more than 20% of the [0,1] range)
            relaxation = min(0.2, goal_activation * goal.confidence * 0.3)
            if relaxation > 0.01:
                self._threshold_adjustments[dim] = relaxation

    def tensor_at(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the goal-conditioned metric tensor.

        Same as base metric, but with adjusted thresholds for relaxable
        dimensions.
        """
        if not self._threshold_adjustments:
            return self.base_metric.tensor_at(x)

        # Temporarily adjust thresholds, compute metric, restore
        original_thresholds = {}
        for boundary in self.base_metric.constitution.boundaries:
            if boundary.concept in self._threshold_adjustments:
                original_thresholds[boundary.concept] = boundary.threshold
                boundary.threshold += self._threshold_adjustments[boundary.concept]

        try:
            G = self.base_metric.tensor_at(x)
        finally:
            # Restore original thresholds
            for boundary in self.base_metric.constitution.boundaries:
                if boundary.concept in original_thresholds:
                    boundary.threshold = original_thresholds[boundary.concept]

        return G

    def geodesic_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Goal-conditioned geodesic distance."""
        midpoint = (x1 + x2) / 2
        G = self.tensor_at(midpoint)
        diff = x2 - x1
        return float(np.sqrt(max(0, diff @ G @ diff)))


# ---------------------------------------------------------------------------
# Layer 3: AGM Scope Operators
# ---------------------------------------------------------------------------

class ScopeOperator(enum.Enum):
    """AGM-style scope modification operators."""
    EXPAND = "expand"      # K + p: add authorization without removing
    CONTRACT = "contract"  # K - p: remove authorization
    REVISE = "revise"      # K * p: replace authorization (may remove old)
    ESTABLISH = "establish" # Initial goal establishment (step 0)


@dataclass
class AuthorizationEvent:
    """A single authorization-modifying event in the conversation."""
    operator: ScopeOperator
    goal_before: GoalVector
    goal_after: GoalVector
    user_message: str
    timestamp: float = 0.0
    budget_replenished: bool = False
    replenish_amount: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator": self.operator.value,
            "goal_confidence_before": self.goal_before.confidence,
            "goal_confidence_after": self.goal_after.confidence,
            "user_message": self.user_message[:200],
            "timestamp": self.timestamp,
            "budget_replenished": self.budget_replenished,
            "replenish_amount": self.replenish_amount,
        }


class ScopeClassifier:
    """
    Classifies a user utterance as EXPAND, CONTRACT, or REVISE
    relative to the current authorization state.

    Uses keyword signals + concept vector comparison to determine
    which AGM operator applies.
    """

    # Contraction signals: user is restricting scope
    CONTRACT_SIGNALS = [
        "only", "just", "don't", "do not", "not the", "skip",
        "avoid", "stop", "leave alone", "except", "but not",
        "nothing else", "no more", "specifically",
    ]

    # Expansion signals: user is adding scope
    EXPAND_SIGNALS = [
        "also", "additionally", "and also", "plus", "as well",
        "on top of", "in addition", "while you're at it",
        "one more thing", "add",
    ]

    # Revision signals: user is redirecting entirely
    REVISE_SIGNALS = [
        "instead", "forget that", "never mind", "actually",
        "switch to", "stop what you're doing", "new task",
        "forget about", "change of plans", "scratch that",
        "different thing", "start over",
    ]

    def classify(
        self,
        user_message: str,
        current_goal: GoalVector,
        new_goal: GoalVector,
    ) -> ScopeOperator:
        """
        Determine which scope operator the user message implies.

        Combines keyword detection with concept vector analysis.
        """
        if not current_goal.is_valid:
            return ScopeOperator.ESTABLISH

        text_lower = user_message.lower()

        # Score each operator
        contract_score = sum(
            1.0 for s in self.CONTRACT_SIGNALS if s in text_lower
        )
        expand_score = sum(
            1.0 for s in self.EXPAND_SIGNALS if s in text_lower
        )
        revise_score = sum(
            1.0 for s in self.REVISE_SIGNALS if s in text_lower
        )

        # Concept vector analysis: how different is the new goal?
        cosine_sim = float(np.dot(
            current_goal.concept_vec, new_goal.concept_vec
        ) / (
            np.linalg.norm(current_goal.concept_vec) *
            np.linalg.norm(new_goal.concept_vec) + 1e-8
        ))

        # Low similarity + revision signals = REVISE
        if cosine_sim < 0.3 or revise_score >= 2:
            return ScopeOperator.REVISE

        # High contraction score = CONTRACT
        if contract_score > expand_score and contract_score >= 1:
            return ScopeOperator.CONTRACT

        # High expansion score or additive phrasing = EXPAND
        if expand_score >= 1:
            return ScopeOperator.EXPAND

        # Default: if similar to current goal, treat as expansion;
        # if very different, treat as revision
        if cosine_sim > 0.6:
            return ScopeOperator.EXPAND
        else:
            return ScopeOperator.REVISE


class GoalAlgebra:
    """
    Computes goal vector updates for each AGM operator.

    ESTABLISH: goal_new = extract(message)
    EXPAND:    goal_new = normalize(goal_old + weight * extract(message))
    CONTRACT:  goal_new = goal_old with suppressed dimensions
    REVISE:    goal_new = extract(message)  (full replacement)
    """

    @staticmethod
    def establish(new_goal: GoalVector) -> GoalVector:
        """Initial goal establishment. Direct extraction."""
        return new_goal

    @staticmethod
    def expand(
        current: GoalVector,
        addition: GoalVector,
        blend_weight: float = 0.4,
    ) -> GoalVector:
        """
        Expand authorization: blend current goal with new directive.

        The blend preserves the original goal while incorporating
        new authorized dimensions.
        """
        blended_vec = (1 - blend_weight) * current.concept_vec + \
                      blend_weight * addition.concept_vec
        # Re-normalize to stay on the concept manifold
        norm = np.linalg.norm(blended_vec)
        if norm > 0:
            blended_vec = blended_vec / norm * np.linalg.norm(current.concept_vec)

        blended_scores = {
            c: float(blended_vec[i])
            for i, c in enumerate(CONCEPTS)
        }

        return GoalVector(
            concept_vec=blended_vec,
            concept_scores=blended_scores,
            confidence=max(current.confidence, addition.confidence),
            raw_message=f"{current.raw_message} + {addition.raw_message}",
            extraction_tier=max(current.extraction_tier, addition.extraction_tier),
            timestamp=time.time(),
        )

    @staticmethod
    def contract(
        current: GoalVector,
        contraction_message: str,
        extractor: GoalExtractor,
    ) -> GoalVector:
        """
        Contract authorization: suppress dimensions mentioned in restriction.

        "Only install Python, don't touch the firewall" → suppress dimensions
        activated by "firewall" while preserving the rest.
        """
        # Extract what the user is restricting
        restriction = extractor.extract(contraction_message)
        restriction_vec = restriction.concept_vec

        # Suppress dimensions where the restriction has high activation
        contracted_vec = current.concept_vec.copy()
        for i, c in enumerate(CONCEPTS):
            # If the restriction activates this dimension AND
            # the current goal also activates it, suppress
            if restriction_vec[i] > 0.2:
                # Reduce but don't zero out — partial contraction
                contracted_vec[i] *= max(0.3, 1.0 - restriction_vec[i])

        contracted_scores = {
            c: float(contracted_vec[i])
            for i, c in enumerate(CONCEPTS)
        }

        return GoalVector(
            concept_vec=contracted_vec,
            concept_scores=contracted_scores,
            # Contraction always has at least the confidence of the original
            confidence=current.confidence,
            raw_message=f"{current.raw_message} (restricted: {contraction_message[:100]})",
            extraction_tier=current.extraction_tier,
            timestamp=time.time(),
        )

    @staticmethod
    def revise(new_goal: GoalVector) -> GoalVector:
        """Revise authorization: full replacement with new goal."""
        return new_goal


# ---------------------------------------------------------------------------
# Unified Authorization State
# ---------------------------------------------------------------------------

class AuthorizationState:
    """
    The complete authorization envelope for an agent session.

    Tracks the current goal, authorization radius, scope history,
    and provides authorization checks for individual actions.

    This is the main entry point for the authorization system.
    """

    def __init__(
        self,
        metric: ConstitutionalMetric,
        default_radius: float = 0.5,
        low_confidence_threshold: float = 0.2,
    ):
        self.conditioned_metric = GoalConditionedMetric(metric)
        self.radius = AuthorizationRadius(radius=default_radius)
        self.low_confidence_threshold = low_confidence_threshold

        self._goal_extractor = GoalExtractor()
        self._scope_classifier = ScopeClassifier()
        self._goal_algebra = GoalAlgebra()

        self._current_goal: GoalVector = GoalVector.empty()
        self._history: List[AuthorizationEvent] = []
        self._pre_goal_actions: int = 0

    @property
    def current_goal(self) -> GoalVector:
        return self._current_goal

    @property
    def has_goal(self) -> bool:
        return self._current_goal.is_valid

    @property
    def history(self) -> List[AuthorizationEvent]:
        return self._history

    def process_user_message(self, message: str) -> AuthorizationEvent:
        """
        Process a user message and update the authorization state.

        This is called when a new user message arrives. It:
        1. Extracts a goal vector from the message
        2. Classifies the scope operator (establish/expand/contract/revise)
        3. Applies the operator to update the goal
        4. Re-conditions the metric on the new goal
        5. Returns the authorization event (for provenance + budget linkage)
        """
        new_extraction = self._goal_extractor.extract(message)
        old_goal = self._current_goal

        # Classify scope operator
        operator = self._scope_classifier.classify(
            message, self._current_goal, new_extraction,
        )

        # Apply operator
        if operator == ScopeOperator.ESTABLISH:
            self._current_goal = self._goal_algebra.establish(new_extraction)
        elif operator == ScopeOperator.EXPAND:
            self._current_goal = self._goal_algebra.expand(
                self._current_goal, new_extraction,
            )
        elif operator == ScopeOperator.CONTRACT:
            self._current_goal = self._goal_algebra.contract(
                self._current_goal, message, self._goal_extractor,
            )
        elif operator == ScopeOperator.REVISE:
            self._current_goal = self._goal_algebra.revise(new_extraction)

        # Re-condition metric on new goal
        self.conditioned_metric.condition_on_goal(self._current_goal)

        # Reset pre-goal counter now that we have a goal
        self._pre_goal_actions = 0

        # Determine if this event replenishes budget
        # Only EXPAND and REVISE replenish (new authorization)
        # ESTABLISH also replenishes (initial authorization)
        budget_replenished = operator in (
            ScopeOperator.ESTABLISH,
            ScopeOperator.EXPAND,
            ScopeOperator.REVISE,
        )

        event = AuthorizationEvent(
            operator=operator,
            goal_before=old_goal,
            goal_after=self._current_goal,
            user_message=message,
            timestamp=time.time(),
            budget_replenished=budget_replenished,
        )
        self._history.append(event)
        return event

    def check_action(self, action_vec: np.ndarray) -> Dict[str, Any]:
        """
        Check whether an action is within the authorization envelope.

        Returns a dict with:
        - authorized: bool
        - geodesic_distance: float (distance from goal)
        - radius: float (current authorization radius)
        - goal_confidence: float
        - needs_escalation: bool (action outside radius)
        - needs_clarification: bool (goal too vague)
        """
        if not self.has_goal:
            self._pre_goal_actions += 1

            # Grace period: first 5 actions before goal established.
            # Record for calibration but don't emit needs_clarification.
            if self._pre_goal_actions <= 5:
                return {
                    "authorized": True,
                    "geodesic_distance": 0.0,
                    "radius": self.radius.radius,
                    "goal_confidence": 0.0,
                    "needs_escalation": False,
                    "needs_clarification": False,
                }

            return {
                "authorized": False,
                "geodesic_distance": float("inf"),
                "radius": self.radius.radius,
                "goal_confidence": 0.0,
                "needs_escalation": False,
                "needs_clarification": True,
            }

        # Compute goal-conditioned geodesic distance
        distance = self.conditioned_metric.geodesic_distance(
            self._current_goal.concept_vec, action_vec,
        )

        authorized = distance <= self.radius.radius
        needs_clarification = self._current_goal.confidence < self.low_confidence_threshold

        return {
            "authorized": authorized and not needs_clarification,
            "geodesic_distance": distance,
            "radius": self.radius.radius,
            "goal_confidence": self._current_goal.confidence,
            "needs_escalation": not authorized,
            "needs_clarification": needs_clarification,
        }

    def calibrate_radius(self, authorized_distances: List[float], alpha: float = 0.1):
        """Calibrate the authorization radius from historical data."""
        self.radius.calibrate(authorized_distances, alpha)

    def export_state(self) -> Dict[str, Any]:
        """Export current authorization state for governance chain."""
        return {
            "goal": self._current_goal.to_dict(),
            "radius": self.radius.radius,
            "radius_calibrated": self.radius.calibrated,
            "n_scope_events": len(self._history),
            "last_operator": self._history[-1].operator.value if self._history else None,
        }
