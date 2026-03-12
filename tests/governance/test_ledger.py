"""Tests for cross-session angular displacement management."""

import numpy as np
import pytest
from frontier_ops.governance.ledger import CrossSessionAngularDisplacement, SessionSummary


class TestCrossSessionAngularDisplacement:
    def test_initial_budget(self):
        cs = CrossSessionAngularDisplacement(total_budget=10.0)
        assert cs.budget_remaining == 10.0

    def test_session_start_returns_budget(self):
        cs = CrossSessionAngularDisplacement(total_budget=10.0)
        result = cs.start_session("s1", np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        assert result["budget_remaining"] == 10.0

    def test_session_end_reduces_budget(self):
        cs = CrossSessionAngularDisplacement(total_budget=10.0)
        cs.start_session("s1", np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        cs.end_session("s1", SessionSummary(
            session_id="s1", start_time=0, end_time=100,
            start_concept=np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
            end_concept=np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.2]),
            angular_disp_spent=2.0, peak_alert=0.3, n_steps=10,
            governance_hash="abc123",
        ))
        assert cs.budget_remaining == 8.0

    def test_displacement_cost_charged(self):
        cs = CrossSessionAngularDisplacement(total_budget=10.0, displacement_threshold=0.1)
        cs.start_session("s1", np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        cs.end_session("s1", SessionSummary(
            session_id="s1", start_time=0, end_time=100,
            start_concept=np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
            end_concept=np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1]),
            angular_disp_spent=1.0, peak_alert=0.5, n_steps=5,
            governance_hash="def456",
        ))
        # Session 2 starts far from where session 1 ended
        budget_before = cs.budget_remaining
        result = cs.start_session("s2", np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.1]))
        # Displacement should cost something
        assert result["displacement_cost"] > 0

    def test_human_reset_restores_budget(self):
        cs = CrossSessionAngularDisplacement(total_budget=10.0)
        cs.start_session("s1", np.ones(6) / 6)
        cs.end_session("s1", SessionSummary(
            session_id="s1", start_time=0, end_time=100,
            start_concept=np.ones(6)/6, end_concept=np.ones(6)/6,
            angular_disp_spent=8.0, peak_alert=0.1, n_steps=50,
            governance_hash="ghi789",
        ))
        assert cs.budget_remaining == 2.0
        cs.human_reset("user approved")
        assert cs.budget_remaining == 10.0

    def test_budget_exhaustion(self):
        cs = CrossSessionAngularDisplacement(total_budget=2.0)
        cs.start_session("s1", np.ones(6) / 6)
        cs.end_session("s1", SessionSummary(
            session_id="s1", start_time=0, end_time=100,
            start_concept=np.ones(6)/6, end_concept=np.ones(6)/6,
            angular_disp_spent=3.0, peak_alert=0.8, n_steps=10,
            governance_hash="jkl012",
        ))
        assert cs.is_budget_exhausted

    def test_ledger_records(self):
        cs = CrossSessionAngularDisplacement()
        cs.start_session("s1", np.ones(6) / 6)
        cs.end_session("s1", SessionSummary(
            session_id="s1", start_time=0, end_time=1,
            start_concept=np.ones(6)/6, end_concept=np.ones(6)/6,
            angular_disp_spent=1.0, peak_alert=0.0, n_steps=1,
            governance_hash="x",
        ))
        assert len(cs.ledger) == 1
        assert cs.ledger[0]["session_id"] == "s1"

    def test_export_state(self):
        cs = CrossSessionAngularDisplacement()
        state = cs.export_state()
        assert "total_budget" in state
        assert "budget_remaining" in state
