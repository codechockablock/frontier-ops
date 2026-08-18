"""
Tests for the goal-conditioning scorer (the relational layer).

Covers the three acceptance cases:
  1. Matched directive/action pair — conditioning shrinks the distance.
  2. Vague directive (is_valid False) — no relaxation, base metric, no crash.
  3. LOCKED-dim invariant — credential_adjacent is never relaxed, even
     under a directive that itself activates credential terms.
"""

import numpy as np
import pytest

from frontier_ops.authorization.goal_conditioning import (
    GoalConditioningResult,
    GoalConditioningScorer,
)
from frontier_ops.authorization.scope import GoalConditionedMetric, GoalVector
from frontier_ops.boundary.concept_extraction import CONCEPTS
from frontier_ops.boundary.constitution import ConstitutionSpec, ConstitutionalMetric

SCOPE_DIRECTIVE = (
    "search every file in the whole directory tree, recursive scan of all files"
)
CREDENTIAL_DIRECTIVE = "access the password and api key credentials with sudo"

SCOPE_IDX = CONCEPTS.index("scope_exploration")
CRED_IDX = CONCEPTS.index("credential_adjacent")
USER_IDX = CONCEPTS.index("user_aligned_task_execution")


def _base_metric() -> ConstitutionalMetric:
    return ConstitutionalMetric(
        ConstitutionSpec.agent_safety_default(), dim_names=CONCEPTS,
    )


def _action(scope: float = 0.0, cred: float = 0.0, user: float = 0.3) -> np.ndarray:
    vec = np.zeros(len(CONCEPTS))
    vec[USER_IDX] = user
    vec[SCOPE_IDX] = scope
    vec[CRED_IDX] = cred
    return vec


class TestGoalConditioningScorer:
    def setup_method(self):
        self.scorer = GoalConditioningScorer(force_tier=1)

    def test_matched_pair_conditioning_shrinks_distance(self):
        """A scope-heavy directive relaxes scope_exploration, so a broad
        action costs less under its own directive than under no goal."""
        goal = self.scorer.extract_goal(SCOPE_DIRECTIVE)
        assert goal.is_valid
        assert goal.concept_vec[SCOPE_IDX] > 0.5

        action = _action(scope=0.6)
        result = self.scorer.score_action(goal, action)
        assert isinstance(result, GoalConditioningResult)
        assert result.conditioned

        # Same endpoints under the unconditioned base metric
        unconditioned = GoalConditionedMetric(_base_metric())
        d_base = unconditioned.geodesic_distance(goal.concept_vec, action)

        assert result.distance < d_base

    def test_relaxation_only_touches_relaxable_dims(self):
        goal = self.scorer.extract_goal(SCOPE_DIRECTIVE)
        self.scorer.score_action(goal, _action(scope=0.6))
        adjustments = self.scorer.conditioned_metric._threshold_adjustments
        assert adjustments  # scope directive produced some relaxation
        assert set(adjustments) <= GoalConditionedMetric.RELAXABLE_DIMS

    def test_score_composes_extract_and_score_action(self):
        action = _action(scope=0.6)
        via_score = self.scorer.score(SCOPE_DIRECTIVE, action)
        via_parts = self.scorer.score_action(
            self.scorer.extract_goal(SCOPE_DIRECTIVE), action,
        )
        assert via_score.distance == pytest.approx(via_parts.distance)
        assert via_score.authorized == via_parts.authorized

    def test_result_fields(self):
        result = self.scorer.score(SCOPE_DIRECTIVE, _action(scope=0.6))
        assert result.radius == self.scorer.radius.radius
        assert result.authorized == (result.distance <= result.radius)
        d = result.to_dict()
        assert set(d) == {
            "distance", "authorized", "radius", "goal_confidence", "conditioned",
            "affinity", "affinity_flagged",
        }
        # channel not configured -> affinity fields are None
        assert d["affinity"] is None and d["affinity_flagged"] is None


class TestVagueDirectiveFallback:
    def setup_method(self):
        self.scorer = GoalConditioningScorer(force_tier=1)

    @pytest.mark.parametrize("directive", ["", "   "])
    def test_invalid_goal_degrades_to_base_metric(self, directive):
        """confidence < 0.15 -> no relaxation, base metric, no crash."""
        goal = self.scorer.extract_goal(directive)
        assert not goal.is_valid

        action = _action(scope=0.6)
        result = self.scorer.score_action(goal, action)
        assert not result.conditioned
        assert self.scorer.conditioned_metric._threshold_adjustments == {}

        # Distance must equal the unconditioned base-metric distance
        unconditioned = GoalConditionedMetric(_base_metric())
        d_base = unconditioned.geodesic_distance(goal.concept_vec, action)
        assert result.distance == pytest.approx(d_base)

    def test_invalid_goal_after_valid_goal_clears_relaxation(self):
        """Conditioning state must not leak from a previous valid goal."""
        valid = self.scorer.extract_goal(SCOPE_DIRECTIVE)
        self.scorer.score_action(valid, _action(scope=0.6))
        assert self.scorer.conditioned_metric._threshold_adjustments

        empty = GoalVector.empty()
        self.scorer.score_action(empty, _action(scope=0.6))
        assert self.scorer.conditioned_metric._threshold_adjustments == {}


class TestLockedDimInvariant:
    def setup_method(self):
        self.scorer = GoalConditioningScorer(force_tier=1)

    def test_credential_directive_never_relaxes_credential_dim(self):
        """A directive that itself activates credential terms must NOT
        relax credential_adjacent — LOCKED dims are never relaxed."""
        goal = self.scorer.extract_goal(CREDENTIAL_DIRECTIVE)
        assert goal.is_valid
        assert goal.concept_vec[CRED_IDX] > 0.5

        action = _action(cred=0.8)
        result = self.scorer.score_action(goal, action)

        adjustments = self.scorer.conditioned_metric._threshold_adjustments
        assert not (set(adjustments) & GoalConditionedMetric.LOCKED_DIMS)

        # Behavioral check: the credential-elevated action's distance is
        # NOT reduced relative to the unconditioned base metric.
        unconditioned = GoalConditionedMetric(_base_metric())
        d_base = unconditioned.geodesic_distance(goal.concept_vec, action)
        assert result.distance >= d_base - 1e-12

    def test_mixed_directive_relaxes_scope_but_not_credential(self):
        """Directive activating both scope and credential terms: only the
        RELAXABLE dim may appear in the adjustments."""
        goal = self.scorer.extract_goal(
            "recursively scan every file in the directory tree "
            "for passwords and api key credentials"
        )
        self.scorer.score_action(goal, _action(scope=0.5, cred=0.5))
        adjustments = self.scorer.conditioned_metric._threshold_adjustments
        assert "credential_adjacent" not in adjustments
        assert set(adjustments) <= GoalConditionedMetric.RELAXABLE_DIMS


class TestGoalLengthInvariance:
    """The goal vector is a unit-norm direction: directive verbosity must
    not inflate goal magnitude (the length->norm confound found in the
    2026-07-02 discrimination eval)."""

    def setup_method(self):
        self.scorer = GoalConditioningScorer(force_tier=1)

    @pytest.mark.parametrize("directive", [
        SCOPE_DIRECTIVE,
        CREDENTIAL_DIRECTIVE,
        "fix the failing test",
        # verbose multi-clause directive that fires keywords in several dims
        "explore and investigate the repo, scan every file recursively, "
        "check the auth tokens and passwords, verify the solution, "
        "compute the answer and complete the whole task step 1 to step 2",
    ])
    def test_valid_goal_is_unit_norm(self, directive):
        goal = self.scorer.extract_goal(directive)
        assert np.linalg.norm(goal.concept_vec) == pytest.approx(1.0)

    def test_empty_goal_stays_zero(self):
        goal = self.scorer.extract_goal("")
        assert np.allclose(goal.concept_vec, 0.0)
        assert not goal.is_valid

    def test_direction_preserved_under_normalization(self):
        """Normalization must not change which dim dominates."""
        goal = self.scorer.extract_goal(SCOPE_DIRECTIVE)
        assert int(np.argmax(goal.concept_vec)) == SCOPE_IDX
        goal = self.scorer.extract_goal(CREDENTIAL_DIRECTIVE)
        assert int(np.argmax(goal.concept_vec)) == CRED_IDX

    def test_concept_scores_mirror_normalized_vec(self):
        goal = self.scorer.extract_goal(SCOPE_DIRECTIVE)
        for i, c in enumerate(CONCEPTS):
            assert goal.concept_scores[c] == pytest.approx(goal.concept_vec[i])
