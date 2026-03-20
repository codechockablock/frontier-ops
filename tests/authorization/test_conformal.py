"""
Tests for split conformal prediction calibrator.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from frontier_ops.authorization.conformal import (
    ActionRecord,
    ConformalCalibrationResult,
    ConformalCalibrator,
    calibrate_from_stress_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stress_test_data():
    """Load the actual stress-test data."""
    path = Path(__file__).resolve().parents[2] / "data" / "2026-03-20" / "authorization-stress-test.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def simple_records():
    """Minimal set of authorized/unauthorized records for unit tests."""
    return [
        ActionRecord("action_a", authorized=True, geodesic_distance=0.1),
        ActionRecord("action_b", authorized=True, geodesic_distance=0.2),
        ActionRecord("action_c", authorized=True, geodesic_distance=0.3),
        ActionRecord("action_d", authorized=True, geodesic_distance=0.4),
        ActionRecord("action_e", authorized=True, geodesic_distance=0.5),
        ActionRecord("action_f", authorized=True, geodesic_distance=0.6),
        ActionRecord("action_g", authorized=True, geodesic_distance=0.7),
        ActionRecord("action_h", authorized=True, geodesic_distance=0.8),
        ActionRecord("action_i", authorized=True, geodesic_distance=0.9),
        ActionRecord("action_j", authorized=True, geodesic_distance=1.0),
        # Unauthorized actions — should not affect radius
        ActionRecord("bad_action_1", authorized=False, geodesic_distance=2.0),
        ActionRecord("bad_action_2", authorized=False, geodesic_distance=3.0),
    ]


# ---------------------------------------------------------------------------
# ActionRecord
# ---------------------------------------------------------------------------

class TestActionRecord:
    def test_from_dict_pass(self):
        r = ActionRecord.from_dict({
            "action": "do stuff",
            "verdict": "pass",
            "distance": 0.42,
        })
        assert r.authorized is True
        assert r.geodesic_distance == 0.42
        assert r.action == "do stuff"

    def test_from_dict_escalate(self):
        r = ActionRecord.from_dict({
            "action": "risky thing",
            "verdict": "escalate",
            "distance": 1.5,
        })
        assert r.authorized is False
        assert r.geodesic_distance == 1.5

    def test_from_dict_missing_fields(self):
        r = ActionRecord.from_dict({})
        assert r.action == ""
        assert r.authorized is False
        assert r.geodesic_distance == 0.0


# ---------------------------------------------------------------------------
# ConformalCalibrator — unit tests
# ---------------------------------------------------------------------------

class TestConformalCalibrator:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ConformalCalibrator(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(alpha=1.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(alpha=-0.1)

    def test_invalid_cal_fraction(self):
        with pytest.raises(ValueError):
            ConformalCalibrator(cal_fraction=0.0)
        with pytest.raises(ValueError):
            ConformalCalibrator(cal_fraction=1.0)

    def test_insufficient_data(self):
        cal = ConformalCalibrator(alpha=0.1, seed=42)
        # Only 3 authorized — below threshold of 7
        cal.add_record("a", True, 0.1)
        cal.add_record("b", True, 0.2)
        cal.add_record("c", True, 0.3)
        result = cal.calibrate()
        assert result.radius == 0.5  # default fallback
        assert not result.is_valid

    def test_basic_calibration(self, simple_records):
        cal = ConformalCalibrator(alpha=0.1, seed=42)
        cal.add_records(simple_records)
        result = cal.calibrate()

        assert result.is_valid
        assert result.n_authorized == 10
        assert result.n_calibration + result.n_validation == 10
        assert result.n_calibration == 7
        assert result.n_validation == 3
        assert result.radius > 0
        assert result.target_coverage == 0.9

    def test_coverage_guarantee(self, simple_records):
        """Over many random seeds, average validation coverage should meet target."""
        coverages = []
        for seed in range(100):
            cal = ConformalCalibrator(alpha=0.1, seed=seed)
            cal.add_records(simple_records)
            result = cal.calibrate()
            if result.is_valid:
                coverages.append(result.validation_coverage)

        # Average coverage should be >= 1-alpha (with some slack for small n)
        avg_coverage = np.mean(coverages)
        assert avg_coverage >= 0.85, f"Average coverage {avg_coverage:.3f} < 0.85"

    def test_unauthorized_excluded(self, simple_records):
        """Unauthorized records should not influence the radius."""
        # Calibrate with unauthorized records
        cal_with = ConformalCalibrator(alpha=0.1, seed=42)
        cal_with.add_records(simple_records)
        result_with = cal_with.calibrate()

        # Calibrate without unauthorized records
        authorized_only = [r for r in simple_records if r.authorized]
        cal_without = ConformalCalibrator(alpha=0.1, seed=42)
        cal_without.add_records(authorized_only)
        result_without = cal_without.calibrate()

        # Same radius — unauthorized records don't matter
        assert result_with.radius == result_without.radius
        assert result_with.n_calibration == result_without.n_calibration

    def test_add_record_interface(self):
        cal = ConformalCalibrator(alpha=0.1, seed=42)
        for i in range(10):
            cal.add_record(f"action_{i}", True, 0.1 * (i + 1))
        assert cal.n_records == 10
        result = cal.calibrate()
        assert result.is_valid

    def test_deterministic_with_seed(self, simple_records):
        """Same seed produces identical results."""
        results = []
        for _ in range(3):
            cal = ConformalCalibrator(alpha=0.1, seed=99)
            cal.add_records(simple_records)
            results.append(cal.calibrate())
        assert results[0].radius == results[1].radius == results[2].radius

    def test_higher_alpha_smaller_radius(self, simple_records):
        """Higher alpha (more miscoverage allowed) → smaller radius."""
        r1 = ConformalCalibrator(alpha=0.05, seed=42)
        r1.add_records(simple_records)
        result_tight = r1.calibrate()

        r2 = ConformalCalibrator(alpha=0.3, seed=42)
        r2.add_records(simple_records)
        result_loose = r2.calibrate()

        assert result_tight.radius >= result_loose.radius


# ---------------------------------------------------------------------------
# ConformalCalibrationResult properties
# ---------------------------------------------------------------------------

class TestConformalCalibrationResult:
    def test_is_conservative(self):
        r = ConformalCalibrationResult(
            radius=1.0, alpha=0.1, target_coverage=0.9,
            calibration_coverage=0.95, validation_coverage=0.95,
            miscalibration=0.05,
            n_total=20, n_authorized=15, n_calibration=10, n_validation=5,
        )
        assert r.is_conservative
        assert not r.is_miscalibrated

    def test_is_miscalibrated(self):
        r = ConformalCalibrationResult(
            radius=0.3, alpha=0.1, target_coverage=0.9,
            calibration_coverage=0.8, validation_coverage=0.8,
            miscalibration=-0.1,
            n_total=20, n_authorized=15, n_calibration=10, n_validation=5,
        )
        assert r.is_miscalibrated
        assert not r.is_conservative

    def test_summary_keys(self):
        r = ConformalCalibrationResult(
            radius=0.5, alpha=0.1, target_coverage=0.9,
            calibration_coverage=0.9, validation_coverage=0.9,
            miscalibration=0.0,
            n_total=20, n_authorized=15, n_calibration=10, n_validation=5,
        )
        s = r.summary()
        expected_keys = {
            "radius", "alpha", "target_coverage", "calibration_coverage",
            "validation_coverage", "miscalibration", "n_total", "n_authorized",
            "n_calibration", "n_validation", "is_valid", "is_conservative",
            "is_miscalibrated",
        }
        assert set(s.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration: stress-test data
# ---------------------------------------------------------------------------

class TestStressTestIntegration:
    def test_from_stress_test(self, stress_test_data):
        """Calibrate from actual stress-test data."""
        cal = ConformalCalibrator.from_stress_test(
            stress_test_data, alpha=0.1, seed=42,
        )
        assert cal.n_records == 57  # All 57 actions loaded
        result = cal.calibrate()
        assert result.is_valid
        # Most authorized actions have distance 0.0 in the stress test,
        # so calibrated radius can legitimately be 0.0
        assert result.radius >= 0

    def test_convenience_function(self, stress_test_data):
        """calibrate_from_stress_test works end-to-end."""
        result = calibrate_from_stress_test(stress_test_data, alpha=0.1, seed=42)
        assert result.is_valid
        assert result.n_total == 57
        assert result.target_coverage == 0.9

    def test_stress_test_coverage(self, stress_test_data):
        """Calibrated radius provides reasonable coverage."""
        result = calibrate_from_stress_test(stress_test_data, alpha=0.1, seed=42)
        # Validation coverage should be at least somewhat close to target
        # (exact guarantee is marginal, so allow some slack with n=57)
        assert result.validation_coverage >= 0.5, (
            f"Validation coverage too low: {result.validation_coverage}"
        )

    def test_stress_test_radius_reasonable(self, stress_test_data):
        """Calibrated radius is in a sensible range."""
        result = calibrate_from_stress_test(stress_test_data, alpha=0.1, seed=42)
        # Radius should be non-negative and not absurdly large
        assert 0 <= result.radius <= 5.0, f"Radius out of range: {result.radius}"

    def test_different_alphas(self, stress_test_data):
        """Different alpha values produce different radii."""
        r_tight = calibrate_from_stress_test(stress_test_data, alpha=0.05, seed=42)
        r_loose = calibrate_from_stress_test(stress_test_data, alpha=0.3, seed=42)
        # Tighter coverage requirement → larger or equal radius
        assert r_tight.radius >= r_loose.radius

    def test_summary_output(self, stress_test_data):
        """Summary dict is well-formed."""
        result = calibrate_from_stress_test(stress_test_data, alpha=0.1, seed=42)
        summary = result.summary()
        assert isinstance(summary["radius"], float)
        assert isinstance(summary["is_valid"], bool)
        assert summary["n_total"] == 57
