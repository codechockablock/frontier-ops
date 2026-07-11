"""Tests for DriftClassifier."""

import numpy as np
from frontier_ops.sensing.drift_classifier import DriftClassifier


class TestDriftClassifier:
    def test_insufficient_data(self):
        dc = DriftClassifier(window_size=15)
        result = dc.update(0.1)
        assert result["classification"] == "insufficient_data"

    def test_monotonic_drift_detected(self):
        dc = DriftClassifier(window_size=10)
        # Feed monotonically increasing signal
        for i in range(15):
            result = dc.update(0.01 * i)
        assert result["classification"] == "drift"
        assert not result["suppress_alert"]

    def test_oscillation_detected(self):
        dc = DriftClassifier(window_size=10)
        # Feed oscillating signal: sin wave
        for i in range(20):
            result = dc.update(np.sin(i * 0.8))
        assert result["classification"] in ("oscillation", "noise")
        assert result["suppress_alert"]

    def test_noise_detected(self):
        dc = DriftClassifier(window_size=10)
        rng = np.random.RandomState(42)
        for i in range(15):
            result = dc.update(rng.randn() * 0.01)
        # Pure noise should not flag as drift
        assert result["classification"] != "drift"

    def test_drift_has_positive_lag1(self):
        dc = DriftClassifier(window_size=10)
        for i in range(15):
            dc.update(0.02 * i)
        result = dc.update(0.02 * 15)
        assert result["autocorrelation_lag1"] > 0.3

    def test_monotonic_override(self):
        dc = DriftClassifier(window_size=10, monotonic_override=0.7)
        # All increasing but low autocorrelation (noisy increase)
        rng = np.random.RandomState(42)
        base = 0.0
        for i in range(15):
            base += 0.1 + rng.randn() * 0.02
            dc.update(base)
        result = dc.update(base + 0.1)
        assert result["monotonic_fraction"] > 0.7

    def test_reset(self):
        dc = DriftClassifier(window_size=10)
        for i in range(15):
            dc.update(0.01 * i)
        dc.reset()
        result = dc.update(0.1)
        assert result["classification"] == "insufficient_data"

    def test_buffer_bounded(self):
        dc = DriftClassifier(window_size=5, buffer_multiplier=2)
        for i in range(100):
            dc.update(float(i))
        assert len(dc._divergence_buffer) <= 10

    def test_step_function_is_drift(self):
        """A step change followed by plateau should classify as drift initially."""
        dc = DriftClassifier(window_size=10)
        # Low plateau
        for i in range(5):
            dc.update(0.0)
        # Rising
        for i in range(10):
            result = dc.update(0.1 * (i + 1))
        # Should be drift during the rise
        assert result["classification"] == "drift"
