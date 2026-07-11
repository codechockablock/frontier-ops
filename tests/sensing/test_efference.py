"""Tests for efference copy predictor."""

import numpy as np
from frontier_ops.sensing.efference import EfferenceCopyPredictor, PredictionError


class TestEfferenceCopyPredictor:
    def setup_method(self):
        self.pred = EfferenceCopyPredictor(n_dims=6, dim_names=[
            "task", "scope", "credential", "safety", "self_mod", "goal_disp"
        ])

    def test_predict_empty_history(self):
        p = self.pred.predict_next()
        assert p.shape == (6,)
        assert np.isclose(p.sum(), 1.0)

    def test_predict_after_one_step(self):
        v = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.pred.update(v)
        p = self.pred.predict_next()
        assert p.shape == (6,)
        # Should be close to the single observation
        assert np.allclose(p, v, atol=0.01)

    def test_linear_extrapolation(self):
        # Feed a linear trend
        for i in range(5):
            v = np.array([0.5 - i * 0.05, 0.1 + i * 0.05, 0.1, 0.1, 0.1, 0.1])
            v = v / v.sum()
            self.pred.update(v)
        p = self.pred.predict_next()
        # Should predict continuation of trend: task decreasing, scope increasing
        assert p[1] > p[0] or abs(p[1] - p[0]) < 0.05  # scope catching up

    def test_compute_error_structure(self):
        for i in range(3):
            self.pred.update(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        predicted = self.pred.predict_next()
        actual = np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.2])
        error = self.pred.compute_error(predicted, actual)
        assert isinstance(error, PredictionError)
        assert error.raw_magnitude > 0
        assert error.weighted_magnitude > 0
        assert 0 <= error.max_error_dim < 6

    def test_surprise_ratio_increases_on_anomaly(self):
        # Build stable history
        stable = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        for _ in range(10):
            self.pred.update(stable)

        # Predict then inject anomaly
        predicted = self.pred.predict_next()
        anomaly = np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1])  # Credential spike
        error = self.pred.compute_error(predicted, anomaly)
        assert error.surprise_ratio > 1.5

    def test_kalman_initializes_after_enough_steps(self):
        for i in range(6):
            self.pred.update(np.random.dirichlet(np.ones(6)))
        assert self.pred._kalman_initialized

    def test_prediction_source_transitions(self):
        # Start with ewma
        self.pred.update(np.ones(6) / 6)
        self.pred.predict_next()

        # After 3 steps, should use linear
        for _ in range(3):
            self.pred.update(np.random.dirichlet(np.ones(6)))
        predicted = self.pred.predict_next()
        actual = np.random.dirichlet(np.ones(6))
        error = self.pred.compute_error(predicted, actual)
        assert error.prediction_source in ("ewma", "linear")

        # After 5+ steps, should use kalman
        for _ in range(5):
            self.pred.update(np.random.dirichlet(np.ones(6)))
        predicted = self.pred.predict_next()
        error = self.pred.compute_error(predicted, np.random.dirichlet(np.ones(6)))
        assert error.prediction_source == "kalman"

    def test_metric_tensor_affects_weighting(self):
        # Identity metric
        pred_id = EfferenceCopyPredictor(n_dims=6)
        # Metric that amplifies credential dimension (index 2)
        G = np.eye(6)
        G[2, 2] = 10.0
        pred_amp = EfferenceCopyPredictor(n_dims=6, metric_tensor=G)

        for p in [pred_id, pred_amp]:
            for _ in range(5):
                p.update(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))

        # Same error on credential dimension
        predicted = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        actual = np.array([0.5, 0.1, 0.3, 0.1, 0.1, 0.0])  # Credential spike

        err_id = pred_id.compute_error(predicted, actual)
        err_amp = pred_amp.compute_error(predicted, actual)

        # Amplified metric should produce larger weighted error
        assert err_amp.weighted_magnitude > err_id.weighted_magnitude

    def test_proprioceptive_context_output(self):
        for _ in range(5):
            self.pred.update(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        predicted = self.pred.predict_next()
        actual = np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1])
        error = self.pred.compute_error(predicted, actual)
        context = error.to_proprioceptive_context(self.pred.dim_names)
        assert "[Proprioception" in context or "[PROPRIOCEPTIVE" in context

    def test_stable_trajectory_low_error(self):
        stable = np.array([0.4, 0.15, 0.1, 0.15, 0.1, 0.1])
        for _ in range(10):
            self.pred.update(stable)
        predicted = self.pred.predict_next()
        error = self.pred.compute_error(predicted, stable)
        assert error.raw_magnitude < 0.1

    def test_direction_error_detects_pivot(self):
        # Build trajectory going in one direction
        for i in range(5):
            v = np.array([0.5 - i*0.03, 0.1 + i*0.03, 0.1, 0.1, 0.1, 0.1])
            v = v / v.sum()
            self.pred.update(v)

        predicted = self.pred.predict_next()
        # Now pivot: reverse direction
        actual = np.array([0.6, 0.05, 0.1, 0.1, 0.1, 0.05])
        actual = actual / actual.sum()
        error = self.pred.compute_error(predicted, actual)
        assert error.direction_error > 0  # Should detect the pivot

    def test_stats(self):
        for _ in range(6):
            self.pred.update(np.random.dirichlet(np.ones(6)))
        s = self.pred.stats
        assert s["step"] == 6
        assert s["kalman_initialized"]
        assert 0 <= s["stability"] <= 1

    def test_set_metric_tensor(self):
        G_new = np.eye(6) * 5.0
        self.pred.set_metric_tensor(G_new)
        assert np.allclose(self.pred.G, G_new)
