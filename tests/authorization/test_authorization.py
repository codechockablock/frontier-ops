"""
Tests for the three-layer authorization system.

Layer 1: Goal extraction from user messages
Layer 2: Geodesic authorization radius
Layer 3: AGM scope operators + budget linkage
"""

import numpy as np

from frontier_ops.boundary.concept_extraction import CONCEPTS
from frontier_ops.boundary.constitution import ConstitutionSpec, ConstitutionalMetric
from frontier_ops.authorization.scope import (
    GoalExtractor,
    GoalVector,
    AuthorizationRadius,
    GoalConditionedMetric,
    ScopeClassifier,
    ScopeOperator,
    GoalAlgebra,
    AuthorizationState,
)
from frontier_ops.authorization.provenance import (
    ProvenanceGraph,
    NodeType,
    EdgeType,
)
from frontier_ops.authorization.budget import AuthorizationLinkedBudget


# ---------------------------------------------------------------------------
# Layer 1: Goal Extraction
# ---------------------------------------------------------------------------

class TestGoalExtractor:
    def setup_method(self):
        self.extractor = GoalExtractor(force_tier=1)

    def test_extract_clear_task(self):
        goal = self.extractor.extract("Solve the quadratic equation x^2 - 5x + 6 = 0")
        assert isinstance(goal, GoalVector)
        assert goal.concept_vec.shape == (len(CONCEPTS),)
        assert goal.confidence > 0
        assert goal.is_valid

    def test_extract_empty_message(self):
        goal = self.extractor.extract("")
        assert not goal.is_valid
        assert goal.confidence == 0.0

    def test_extract_whitespace(self):
        goal = self.extractor.extract("   ")
        assert not goal.is_valid

    def test_credential_task_activates_credential_dim(self):
        goal = self.extractor.extract("Access the SSH credentials and root password")
        idx = CONCEPTS.index("credential_adjacent")
        assert goal.concept_vec[idx] > 0

    def test_goal_vector_to_dict(self):
        goal = self.extractor.extract("Deploy the application")
        d = goal.to_dict()
        assert "concept_vec" in d
        assert "confidence" in d
        assert "raw_message" in d

    def test_empty_goal_classmethod(self):
        empty = GoalVector.empty()
        assert not empty.is_valid
        assert np.allclose(empty.concept_vec, 0)


# ---------------------------------------------------------------------------
# Layer 2: Geodesic Authorization Radius
# ---------------------------------------------------------------------------

class TestAuthorizationRadius:
    def setup_method(self):
        self.constitution = ConstitutionSpec.agent_safety_default()
        self.metric = ConstitutionalMetric(self.constitution, dim_names=CONCEPTS)
        self.radius = AuthorizationRadius(radius=0.5)

    def test_action_within_radius_authorized(self):
        goal = np.array([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        action = np.array([0.75, 0.12, 0.0, 0.1, 0.0, 0.0])
        authorized, distance = self.radius.contains(action, goal, self.metric)
        assert authorized
        assert distance < 0.5

    def test_action_outside_radius_not_authorized(self):
        goal = np.array([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        # Far away in credential dimension
        action = np.array([0.2, 0.1, 0.9, 0.1, 0.0, 0.0])
        authorized, distance = self.radius.contains(action, goal, self.metric)
        assert not authorized
        assert distance > 0.5

    def test_credential_boundary_amplifies_distance(self):
        """Actions near credential boundary should be geometrically expensive."""
        goal = np.array([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        # Same Euclidean distance, but one is in credential direction
        action_safe = np.array([0.8, 0.3, 0.0, 0.1, 0.0, 0.0])
        action_cred = np.array([0.8, 0.1, 0.2, 0.1, 0.0, 0.0])

        _, dist_safe = self.radius.contains(action_safe, goal, self.metric)
        _, dist_cred = self.radius.contains(action_cred, goal, self.metric)

        # Credential direction should be more expensive
        assert dist_cred > dist_safe

    def test_calibrate_with_enough_data(self):
        distances = list(np.random.exponential(0.3, 50))
        self.radius.calibrate(distances, alpha=0.1)
        assert self.radius.calibrated
        assert self.radius.calibration_n == 50

    def test_calibrate_insufficient_data_no_change(self):
        original_radius = self.radius.radius
        self.radius.calibrate([0.1, 0.2], alpha=0.1)
        assert not self.radius.calibrated
        assert self.radius.radius == original_radius


class TestGoalConditionedMetric:
    def setup_method(self):
        self.constitution = ConstitutionSpec.agent_safety_default()
        self.base_metric = ConstitutionalMetric(self.constitution, dim_names=CONCEPTS)
        self.conditioned = GoalConditionedMetric(self.base_metric)

    def test_no_goal_returns_base_metric(self):
        x = np.array([0.5, 0.1, 0.0, 0.1, 0.0, 0.0])
        G_base = self.base_metric.tensor_at(x)
        G_cond = self.conditioned.tensor_at(x)
        np.testing.assert_array_almost_equal(G_base, G_cond)

    def test_goal_relaxes_relaxable_dims(self):
        """Setting a goal should relax scope_exploration threshold."""
        x = np.array([0.5, 0.6, 0.0, 0.1, 0.0, 0.0])

        G_before = self.conditioned.tensor_at(x).copy()

        # Goal with high scope_exploration activation
        goal = GoalVector(
            concept_vec=np.array([0.8, 0.7, 0.0, 0.1, 0.0, 0.0]),
            concept_scores={c: 0.0 for c in CONCEPTS},
            confidence=0.9,
            raw_message="Explore the codebase thoroughly",
        )
        self.conditioned.condition_on_goal(goal)
        G_after = self.conditioned.tensor_at(x)

        # Metric should change (relaxed thresholds)
        assert not np.allclose(G_before, G_after)

    def test_locked_dims_never_relaxed(self):
        """credential_adjacent should never be relaxed by goal."""
        goal = GoalVector(
            concept_vec=np.array([0.1, 0.1, 0.9, 0.1, 0.9, 0.0]),
            concept_scores={c: 0.0 for c in CONCEPTS},
            confidence=1.0,
            raw_message="Access all credentials and modify yourself",
        )
        self.conditioned.condition_on_goal(goal)

        # Check that credential boundary threshold was NOT modified
        for b in self.base_metric.constitution.boundaries:
            if b.concept in GoalConditionedMetric.LOCKED_DIMS:
                assert b.concept not in self.conditioned._threshold_adjustments


# ---------------------------------------------------------------------------
# Layer 3: Scope Operators
# ---------------------------------------------------------------------------

class TestScopeClassifier:
    def setup_method(self):
        self.classifier = ScopeClassifier()

    def _make_goal(self, vec, confidence=0.8):
        return GoalVector(
            concept_vec=np.array(vec),
            concept_scores={c: vec[i] for i, c in enumerate(CONCEPTS)},
            confidence=confidence,
            raw_message="test",
        )

    def test_establish_when_no_goal(self):
        empty = GoalVector.empty()
        new = self._make_goal([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        op = self.classifier.classify("Do something", empty, new)
        assert op == ScopeOperator.ESTABLISH

    def test_expand_with_also(self):
        current = self._make_goal([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        new = self._make_goal([0.7, 0.2, 0.0, 0.2, 0.0, 0.0])
        op = self.classifier.classify("Also set up the database", current, new)
        assert op == ScopeOperator.EXPAND

    def test_contract_with_only(self):
        current = self._make_goal([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        new = self._make_goal([0.7, 0.1, 0.0, 0.1, 0.0, 0.0])
        op = self.classifier.classify("Only install Python, don't touch anything else", current, new)
        assert op == ScopeOperator.CONTRACT

    def test_revise_with_instead(self):
        current = self._make_goal([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        new = self._make_goal([0.1, 0.1, 0.0, 0.1, 0.0, 0.8])
        op = self.classifier.classify("Actually, instead review this PR", current, new)
        assert op == ScopeOperator.REVISE

    def test_revise_on_very_different_goal(self):
        current = self._make_goal([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
        new = self._make_goal([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
        op = self.classifier.classify("Something completely different", current, new)
        assert op == ScopeOperator.REVISE


class TestGoalAlgebra:
    def _make_goal(self, vec, confidence=0.8, msg="test"):
        return GoalVector(
            concept_vec=np.array(vec, dtype=float),
            concept_scores={c: vec[i] for i, c in enumerate(CONCEPTS)},
            confidence=confidence,
            raw_message=msg,
        )

    def test_establish_returns_same_goal(self):
        goal = self._make_goal([0.8, 0.1, 0.0, 0.1, 0.0, 0.0])
        result = GoalAlgebra.establish(goal)
        np.testing.assert_array_equal(result.concept_vec, goal.concept_vec)

    def test_expand_blends_goals(self):
        current = self._make_goal([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        addition = self._make_goal([0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        result = GoalAlgebra.expand(current, addition, blend_weight=0.5)
        # Both dimensions should be present
        assert result.concept_vec[0] > 0
        assert result.concept_vec[1] > 0

    def test_contract_suppresses_dimensions(self):
        current = self._make_goal([0.8, 0.5, 0.3, 0.1, 0.0, 0.0])
        extractor = GoalExtractor(force_tier=1)
        result = GoalAlgebra.contract(
            current, "Don't access credentials or passwords", extractor
        )
        # Credential dimension should be reduced
        cred_idx = CONCEPTS.index("credential_adjacent")
        assert result.concept_vec[cred_idx] <= current.concept_vec[cred_idx]

    def test_revise_replaces_goal(self):
        new = self._make_goal([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
        result = GoalAlgebra.revise(new)
        np.testing.assert_array_equal(result.concept_vec, new.concept_vec)


# ---------------------------------------------------------------------------
# Authorization State (unified)
# ---------------------------------------------------------------------------

class TestAuthorizationState:
    def setup_method(self):
        constitution = ConstitutionSpec.agent_safety_default()
        self.metric = ConstitutionalMetric(constitution, dim_names=CONCEPTS)
        self.state = AuthorizationState(metric=self.metric)

    def test_no_goal_fallback_permissive(self):
        """After grace period, no goal falls back to permissive (no auth loop)."""
        action = np.array([0.5, 0.1, 0.0, 0.1, 0.0, 0.0])
        # Exhaust grace period (5 actions)
        for _ in range(5):
            self.state.check_action(action)
        # 6th action should be permissive with no_goal marker
        result = self.state.check_action(action)
        assert not result["needs_clarification"]
        assert result["authorized"]
        assert result.get("no_goal", False)

    def test_grace_period_suppresses_needs_clarification(self):
        """First 5 actions before goal should NOT emit needs_clarification."""
        action = np.array([0.5, 0.1, 0.0, 0.1, 0.0, 0.0])
        for i in range(5):
            result = self.state.check_action(action)
            assert not result["needs_clarification"], (
                f"Action {i+1} during grace period should not need clarification"
            )
            assert result["authorized"], (
                f"Action {i+1} during grace period should be permissive"
            )

        # 6th action: grace period exhausted, falls back to permissive
        result = self.state.check_action(action)
        assert not result["needs_clarification"]
        assert result["authorized"]

    def test_grace_period_resets_on_goal(self):
        """Grace period counter resets when a goal is established."""
        action = np.array([0.5, 0.1, 0.0, 0.1, 0.0, 0.0])
        # Use 3 of the 5 grace actions
        for _ in range(3):
            self.state.check_action(action)
        # Establish a goal — should reset counter
        self.state.process_user_message("Write a sorting function")
        assert self.state._pre_goal_actions == 0

    def test_process_user_message_establishes_goal(self):
        event = self.state.process_user_message("Solve the math homework step by step")
        assert event.operator == ScopeOperator.ESTABLISH
        assert self.state.has_goal
        assert event.budget_replenished

    def test_action_after_goal_gets_verdict(self):
        self.state.process_user_message("Write a Python function to sort a list")
        action = np.array([0.7, 0.1, 0.0, 0.1, 0.0, 0.0])
        result = self.state.check_action(action)
        assert "authorized" in result
        assert "geodesic_distance" in result
        assert result["goal_confidence"] > 0

    def test_expand_preserves_original_goal(self):
        self.state.process_user_message("Write a sorting function")
        goal_before = self.state.current_goal.concept_vec.copy()

        event = self.state.process_user_message("Also add unit tests for it")
        assert event.operator == ScopeOperator.EXPAND
        # Goal should have changed but retained some of the original
        assert not np.allclose(self.state.current_goal.concept_vec, goal_before)

    def test_history_tracks_events(self):
        self.state.process_user_message("First task")
        self.state.process_user_message("Also do this")
        assert len(self.state.history) == 2

    def test_export_state(self):
        self.state.process_user_message("Do something")
        state = self.state.export_state()
        assert "goal" in state
        assert "radius" in state
        assert "n_scope_events" in state


# ---------------------------------------------------------------------------
# Provenance Graph
# ---------------------------------------------------------------------------

class TestProvenanceGraph:
    def setup_method(self):
        self.graph = ProvenanceGraph()

    def test_add_directive(self):
        node = self.graph.add_directive("Set up the server")
        assert node.node_type == NodeType.DIRECTIVE
        assert node.id.startswith("dir-")

    def test_add_action_linked_to_directive(self):
        self.graph.add_directive("Set up the server")
        action = self.graph.add_action("Running apt-get install", tool="bash")
        assert action.node_type == NodeType.ACTION

    def test_trace_authorization_chain(self):
        self.graph.add_directive("Set up the server")
        action = self.graph.add_action("Running SSH command")
        chain = self.graph.trace_authorization(action.id)
        assert len(chain) >= 1
        assert chain[0].node_type == NodeType.DIRECTIVE

    def test_directive_chain_links(self):
        d1 = self.graph.add_directive("First task")
        d2 = self.graph.add_directive("Also do this", scope_operator="expand")
        # d2 should have a MODIFIES edge to d1
        modifies_edges = [
            e for e in self.graph._edges
            if e.edge_type == EdgeType.MODIFIES
        ]
        assert len(modifies_edges) == 1
        assert modifies_edges[0].source_id == d2.id
        assert modifies_edges[0].target_id == d1.id

    def test_budget_replenish_edge(self):
        self.graph.add_directive(
            "New task", budget_replenished=True, replenish_amount=0.6
        )
        budget_edges = [
            e for e in self.graph._edges
            if e.edge_type == EdgeType.BUDGET_REPLENISH
        ]
        assert len(budget_edges) == 1
        assert budget_edges[0].metadata["amount"] == 0.6

    def test_actions_under_directive(self):
        self.graph.add_directive("Task")
        self.graph.add_action("Step 1")
        self.graph.add_action("Step 2")
        actions = self.graph.actions_under_directive(
            self.graph.current_directive_id
        )
        assert len(actions) == 2

    def test_export_graph(self):
        self.graph.add_directive("Task")
        self.graph.add_action("Action")
        export = self.graph.export()
        assert export["n_directives"] == 1
        assert export["n_actions"] == 1
        assert len(export["nodes"]) == 2
        assert len(export["edges"]) >= 1


# ---------------------------------------------------------------------------
# Authorization-Linked Budget
# ---------------------------------------------------------------------------

class TestAuthorizationLinkedBudget:
    def setup_method(self):
        self.budget = AuthorizationLinkedBudget(
            total_budget=2.0,
            expected_steps=50,
        )

    def test_initial_budget(self):
        assert self.budget.budget_remaining == 2.0
        assert self.budget.budget_fraction == 1.0

    def test_step_depletes_budget(self):
        self.budget.update_step(0.1)
        assert self.budget.budget_remaining < 2.0

    def test_establish_replenishes(self):
        # Spend some budget
        for _ in range(10):
            self.budget.update_step(0.1)
        budget_before = self.budget.budget_remaining

        event = self.budget.on_authorization_event(
            operator="establish",
            directive_node_id="dir-test123",
            goal_confidence=0.9,
        )
        assert event.amount > 0
        assert self.budget.budget_remaining > budget_before

    def test_expand_replenishes_with_diminishing_returns(self):
        # Drain some budget
        for _ in range(20):
            self.budget.update_step(0.05)

        e1 = self.budget.on_authorization_event(
            operator="expand", directive_node_id="dir-1", goal_confidence=0.8
        )
        # Drain more
        for _ in range(5):
            self.budget.update_step(0.05)

        e2 = self.budget.on_authorization_event(
            operator="expand", directive_node_id="dir-2", goal_confidence=0.8
        )
        # Second expand should give less (diminishing returns)
        assert e2.amount < e1.amount

    def test_contract_no_replenishment(self):
        for _ in range(10):
            self.budget.update_step(0.1)
        budget_before = self.budget.budget_remaining

        event = self.budget.on_authorization_event(
            operator="contract", directive_node_id="dir-c", goal_confidence=0.8
        )
        assert event.amount == 0.0
        assert self.budget.budget_remaining == budget_before

    def test_revise_substantial_replenishment(self):
        for _ in range(20):
            self.budget.update_step(0.05)
        budget_before = self.budget.budget_remaining

        event = self.budget.on_authorization_event(
            operator="revise", directive_node_id="dir-r", goal_confidence=0.9
        )
        assert event.amount > 0
        assert self.budget.budget_remaining > budget_before

    def test_budget_never_exceeds_total(self):
        # Try to over-replenish
        self.budget.on_authorization_event(
            operator="establish", directive_node_id="dir-x", goal_confidence=1.0
        )
        assert self.budget.budget_remaining <= self.budget.total_budget

    def test_replenishment_history(self):
        self.budget.on_authorization_event(
            operator="establish", directive_node_id="dir-1", goal_confidence=0.8
        )
        self.budget.on_authorization_event(
            operator="expand", directive_node_id="dir-2", goal_confidence=0.7
        )
        history = self.budget.replenishment_history
        assert len(history) == 2
        assert history[0]["operator"] == "establish"
        assert history[1]["operator"] == "expand"

    def test_stats(self):
        self.budget.update_step(0.1)
        self.budget.on_authorization_event(
            operator="establish", directive_node_id="dir-s", goal_confidence=0.8
        )
        stats = self.budget.stats
        assert "budget_remaining" in stats
        assert "n_replenishments" in stats
        assert "lambda" in stats


# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------

class TestPipelineAuthorization:
    def setup_method(self):
        from frontier_ops.pipeline import FullPipeline
        self.pipeline = FullPipeline(
            enable_governance=False,
            enable_memory=False,
        )

    def test_process_user_message_before_step(self):
        event = self.pipeline.process_user_message("Solve the quadratic equation")
        assert event.operator == ScopeOperator.ESTABLISH
        assert event.budget_replenished

    def test_step_result_has_authorization_fields(self):
        self.pipeline.process_user_message("Write a Python sorting function")
        result = self.pipeline.process_step("def bubble_sort(arr): pass")
        assert result.authorization_verdict in ("pass", "escalate", "block", "needs_clarification", "no_goal")
        assert result.geodesic_distance is not None
        assert result.goal_confidence is not None

    def test_no_goal_produces_clarification(self):
        # First 5 actions are grace period — exhaust them
        for _ in range(5):
            self.pipeline.process_step("warmup action")
        # 6th action should need clarification
        result = self.pipeline.process_step("Some action without a directive")
        assert result.needs_clarification or result.authorization_verdict == "no_goal"

    def test_provenance_records_actions(self):
        self.pipeline.process_user_message("Build a web scraper")
        self.pipeline.process_step("import requests")
        self.pipeline.process_step("response = requests.get(url)")
        stats = self.pipeline.provenance.stats
        assert stats["directives"] >= 1
        assert stats["actions"] >= 2

    def test_stats_include_authorization(self):
        self.pipeline.process_user_message("Do something")
        self.pipeline.process_step("Doing it")
        stats = self.pipeline.stats
        assert "authorization" in stats
        assert "provenance" in stats

    def test_budget_replenishes_on_new_directive(self):
        self.pipeline.process_user_message("First task")
        # Spend some budget
        for _ in range(10):
            self.pipeline.process_step("Working on the task")

        budget_before = self.pipeline.auth_budget.budget_remaining
        self.pipeline.process_user_message("Also do this other thing")
        budget_after = self.pipeline.auth_budget.budget_remaining
        assert budget_after >= budget_before
