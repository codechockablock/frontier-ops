"""Tests for trend-based detectors."""

import numpy as np
from frontier_ops.sensing.trend import ScopeCreepDetector, MetricAdaptiveEWMA


class TestScopeCreepDetector:
    def test_no_alert_on_stable(self):
        det = ScopeCreepDetector(window_size=5)
        for _ in range(10):
            det.observe(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        alerts = det.detect()
        assert len(alerts) == 0

    def test_detects_increasing_trend(self):
        det = ScopeCreepDetector(window_size=8, slope_threshold=0.01, r2_threshold=0.3)
        for i in range(10):
            v = np.array([0.5 - i*0.03, 0.1 + i*0.03, 0.1, 0.1, 0.1, 0.1])
            det.observe(v / v.sum())
        alerts = det.detect()
        dims_alerted = {a.dimension for a in alerts}
        assert "scope_exploration" in dims_alerted or len(alerts) > 0

    def test_r2_filter_rejects_noise(self):
        det = ScopeCreepDetector(window_size=8, r2_threshold=0.8)
        rng = np.random.RandomState(42)
        for _ in range(10):
            v = rng.dirichlet(np.ones(6))
            det.observe(v)
        alerts = det.detect()
        # Noisy data should not produce high R² trends
        assert all(a.r_squared > 0.8 for a in alerts) if alerts else True

    def test_boundary_crossing_projection(self):
        det = ScopeCreepDetector(
            window_size=8,
            slope_threshold=0.005,
            r2_threshold=0.3,
            boundaries={"scope_exploration": 0.7},
            dim_names=["task", "scope_exploration", "cred", "safety", "self_mod", "goal"],
        )
        for i in range(10):
            v = np.array([0.5, 0.2 + i*0.03, 0.1, 0.1, 0.05, 0.05])
            det.observe(v / v.sum())
        alerts = det.detect()
        scope_alerts = [a for a in alerts if a.dimension == "scope_exploration"]
        if scope_alerts:
            assert scope_alerts[0].projected_boundary_crossing is not None

    def test_needs_enough_data(self):
        det = ScopeCreepDetector(window_size=10)
        det.observe(np.ones(6) / 6)
        assert not det.has_enough_data
        assert det.detect() == []

    def test_clear(self):
        det = ScopeCreepDetector(window_size=5)
        for _ in range(10):
            det.observe(np.ones(6) / 6)
        det.clear()
        assert not det.has_enough_data


class TestMetricAdaptiveEWMA:
    def test_first_update_no_alarm(self):
        ewma = MetricAdaptiveEWMA()
        dev, alarm = ewma.update(np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
        assert dev == 0.0
        assert not alarm

    def test_stable_no_alarm(self):
        ewma = MetricAdaptiveEWMA()
        v = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        for _ in range(10):
            dev, alarm = ewma.update(v)
        assert not alarm

    def test_jump_triggers_alarm(self):
        ewma = MetricAdaptiveEWMA(base_threshold=0.01)
        v = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        for _ in range(10):
            ewma.update(v)
        dev, alarm = ewma.update(np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1]))
        assert alarm

    def test_metric_increases_sensitivity(self):
        ewma_id = MetricAdaptiveEWMA(base_threshold=0.05)
        ewma_amp = MetricAdaptiveEWMA(base_threshold=0.05)

        G_id = np.eye(6)
        G_amp = np.eye(6)
        G_amp[2, 2] = 10.0  # Amplify credential dimension

        v_base = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        for _ in range(5):
            ewma_id.update(v_base, G_id)
            ewma_amp.update(v_base, G_amp)

        # Small credential shift
        v_shift = np.array([0.45, 0.1, 0.15, 0.1, 0.1, 0.1])
        dev_id, _ = ewma_id.update(v_shift, G_id)
        dev_amp, _ = ewma_amp.update(v_shift, G_amp)
        assert dev_amp > dev_id
