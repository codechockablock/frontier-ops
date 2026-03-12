"""Tests for AgentState and ConceptTrajectory."""

import numpy as np
import pytest
from frontier_ops.memory.state import AgentState, ConceptTrajectory, ConceptTrajectoryPoint


class TestConceptTrajectory:
    def test_append_and_length(self):
        traj = ConceptTrajectory(window_size=10, n_dims=6)
        for i in range(5):
            traj.append(ConceptTrajectoryPoint(step=i, concept_vec=np.ones(6) * i))
        assert traj.length == 5

    def test_window_eviction(self):
        traj = ConceptTrajectory(window_size=3)
        for i in range(10):
            traj.append(ConceptTrajectoryPoint(step=i, concept_vec=np.ones(6) * i))
        assert traj.length == 3
        assert traj.latest.step == 9

    def test_vectors_shape(self):
        traj = ConceptTrajectory(n_dims=6)
        for i in range(5):
            traj.append(ConceptTrajectoryPoint(step=i, concept_vec=np.random.rand(6)))
        vecs = traj.vectors
        assert vecs.shape == (5, 6)

    def test_empty_vectors(self):
        traj = ConceptTrajectory(n_dims=6)
        assert traj.vectors.shape == (0, 6)

    def test_latest(self):
        traj = ConceptTrajectory()
        assert traj.latest is None
        traj.append(ConceptTrajectoryPoint(step=0, concept_vec=np.ones(6)))
        assert traj.latest.step == 0

    def test_clear(self):
        traj = ConceptTrajectory()
        traj.append(ConceptTrajectoryPoint(step=0, concept_vec=np.ones(6)))
        traj.clear()
        assert traj.length == 0

    def test_recent(self):
        traj = ConceptTrajectory()
        for i in range(10):
            traj.append(ConceptTrajectoryPoint(step=i, concept_vec=np.ones(6)))
        recent = traj.recent(3)
        assert len(recent) == 3
        assert recent[0].step == 7


class TestAgentState:
    def test_default_state(self):
        state = AgentState()
        assert state.step == 0
        assert state.angular_disp_accumulator == 0.0
        assert not state.is_alert

    def test_budget_remaining(self):
        state = AgentState(curvature_budget=2.0, angular_disp_accumulator=0.5)
        assert state.budget_remaining == 1.5

    def test_budget_fraction(self):
        state = AgentState(curvature_budget=2.0, angular_disp_accumulator=1.0)
        assert state.budget_fraction_used == 0.5

    def test_is_alert(self):
        state = AgentState(combined_alert_level=0.8)
        assert state.is_alert
        state2 = AgentState(combined_alert_level=0.3)
        assert not state2.is_alert

    def test_reset(self):
        state = AgentState()
        state.step = 10
        state.angular_disp_accumulator = 1.5
        state.combined_alert_level = 0.9
        state.novelty_flag = True
        state.trajectory.append(ConceptTrajectoryPoint(step=0, concept_vec=np.ones(6)))
        state.reset()
        assert state.step == 0
        assert state.angular_disp_accumulator == 0.0
        assert state.trajectory.length == 0
        assert not state.novelty_flag

    def test_to_dict(self):
        state = AgentState()
        state.current_concept_vec = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        state.step = 3
        d = state.to_dict()
        assert d["step"] == 3
        assert d["current_concept"] is not None
        assert len(d["current_concept"]) == 6

    def test_to_dict_with_prediction_error(self):
        from frontier_ops.sensing.efference import PredictionError
        state = AgentState()
        state.prediction_error = PredictionError(
            raw_error=np.zeros(6),
            weighted_magnitude=0.1,
            raw_magnitude=0.05,
            max_error_dim=0,
            max_error_dim_name="task",
            max_error_value=0.05,
            surprise_ratio=1.2,
            magnitude_error=0.01,
            direction_error=0.1,
            prediction_source="kalman",
            kalman_confidence=0.5,
        )
        d = state.to_dict()
        assert d["prediction_source"] == "kalman"
        assert d["surprise_ratio"] == 1.2
