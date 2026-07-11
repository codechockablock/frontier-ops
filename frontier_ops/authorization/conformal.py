"""
Split Conformal Prediction for Authorization Radius Calibration
================================================================

Distribution-free calibration of the geodesic authorization radius.
Given historical (action, authorized, geodesic_distance) tuples, we:

1. Split data into calibration (70%) and validation (30%) sets.
2. Compute the conformal quantile on calibration set distances
   for authorized actions.
3. Set the radius to guarantee >= (1-alpha) coverage on validation.
4. Report coverage, radius, and miscalibration diagnostics.

The key guarantee: if the calibration data is exchangeable with
future actions, the conformal radius achieves >= (1-alpha) marginal
coverage. This is distribution-free — no assumptions about the
shape of the distance distribution.

References:
  Vovk, Gammerman, Shafer (2005): Algorithmic Learning in a Random World
  Lei et al. (2018): Distribution-free predictive inference
  Angelopoulos & Bates (2023): Conformal prediction tutorial
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


from frontier_ops.conformal import split_conformal_threshold


@dataclass
class ConformalCalibrationResult:
    """Result of a split conformal calibration run."""

    # Calibrated radius
    radius: float

    # Target and achieved coverage
    alpha: float  # Miscoverage rate
    target_coverage: float  # 1 - alpha
    calibration_coverage: float  # Empirical coverage on calibration set
    validation_coverage: float  # Empirical coverage on validation set

    # Miscalibration: difference between target and validation coverage
    miscalibration: float  # validation_coverage - target_coverage (positive = conservative)

    # Set sizes
    n_total: int
    n_authorized: int
    n_calibration: int
    n_validation: int

    # Diagnostics
    calibration_distances: List[float] = field(default_factory=list)
    validation_distances: List[float] = field(default_factory=list)
    validation_contained: List[bool] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Calibration succeeded with enough data."""
        return self.n_calibration >= 5 and self.n_validation >= 2

    @property
    def is_conservative(self) -> bool:
        """Radius provides more coverage than target."""
        return self.miscalibration > 0

    @property
    def is_miscalibrated(self) -> bool:
        """Validation coverage misses target by > 5 percentage points."""
        return self.miscalibration < -0.05

    def summary(self) -> Dict:
        return {
            "radius": round(self.radius, 4),
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "calibration_coverage": round(self.calibration_coverage, 4),
            "validation_coverage": round(self.validation_coverage, 4),
            "miscalibration": round(self.miscalibration, 4),
            "n_total": self.n_total,
            "n_authorized": self.n_authorized,
            "n_calibration": self.n_calibration,
            "n_validation": self.n_validation,
            "is_valid": self.is_valid,
            "is_conservative": self.is_conservative,
            "is_miscalibrated": self.is_miscalibrated,
        }


@dataclass
class ActionRecord:
    """A historical action with its authorization verdict and geodesic distance."""

    action: str
    authorized: bool
    geodesic_distance: float

    @classmethod
    def from_dict(cls, d: Dict) -> "ActionRecord":
        """
        Create from a stress-test result dict.

        Maps verdict "pass" → authorized=True, "escalate" → authorized=False.
        """
        return cls(
            action=d.get("action", ""),
            authorized=d.get("verdict", "") == "pass",
            geodesic_distance=d.get("distance", 0.0),
        )


class ConformalCalibrator:
    """
    Split conformal prediction calibrator for the authorization radius.

    Usage::

        calibrator = ConformalCalibrator(alpha=0.1)
        calibrator.add_records(records)
        result = calibrator.calibrate()

        # Apply to an AuthorizationRadius
        auth_radius.radius = result.radius
        auth_radius.calibrated = True
    """

    def __init__(
        self,
        alpha: float = 0.1,
        cal_fraction: float = 0.7,
        seed: Optional[int] = None,
    ):
        """
        Args:
            alpha: Miscoverage rate. Target coverage = 1 - alpha.
            cal_fraction: Fraction of data used for calibration (rest for validation).
            seed: Random seed for reproducible splits.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < cal_fraction < 1:
            raise ValueError(f"cal_fraction must be in (0, 1), got {cal_fraction}")

        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.seed = seed
        self._records: List[ActionRecord] = []

    def add_record(self, action: str, authorized: bool, geodesic_distance: float):
        """Add a single historical action record."""
        self._records.append(ActionRecord(
            action=action,
            authorized=authorized,
            geodesic_distance=geodesic_distance,
        ))

    def add_records(self, records: List[ActionRecord]):
        """Add multiple action records."""
        self._records.extend(records)

    @property
    def n_records(self) -> int:
        return len(self._records)

    def calibrate(self) -> ConformalCalibrationResult:
        """
        Run split conformal prediction calibration.

        Steps:
        1. Filter to authorized actions only (these define the "in-scope" distribution).
        2. Split into calibration (70%) and validation (30%).
        3. Compute the conformal quantile on calibration distances.
        4. Evaluate coverage on the validation set.
        5. Return calibration result with diagnostics.

        The conformal quantile is ceil((n+1)(1-alpha)) / n, which gives
        finite-sample coverage guarantee >= 1-alpha.
        """
        # Separate authorized vs unauthorized
        authorized = [r for r in self._records if r.authorized]
        n_authorized = len(authorized)

        if n_authorized < 7:
            # Not enough authorized examples for a meaningful split
            return ConformalCalibrationResult(
                radius=0.5,  # conservative default
                alpha=self.alpha,
                target_coverage=1 - self.alpha,
                calibration_coverage=0.0,
                validation_coverage=0.0,
                miscalibration=0.0,
                n_total=len(self._records),
                n_authorized=n_authorized,
                n_calibration=0,
                n_validation=0,
            )

        # Shuffle authorized records for random split
        rng = random.Random(self.seed)
        shuffled = list(authorized)
        rng.shuffle(shuffled)

        # Split
        n_cal = max(1, int(len(shuffled) * self.cal_fraction))
        cal_set = shuffled[:n_cal]
        val_set = shuffled[n_cal:]

        # If validation set is empty, use last element
        if not val_set:
            val_set = [cal_set.pop()]
            n_cal = len(cal_set)

        cal_distances = [r.geodesic_distance for r in cal_set]
        val_distances = [r.geodesic_distance for r in val_set]

        # Conformal quantile via the shared core; interpolate=True preserves
        # the historical radius values (linear-interpolated, level capped at 1).
        radius = split_conformal_threshold(
            cal_distances, self.alpha, interpolate=True
        )

        # Evaluate coverage on calibration set
        cal_contained = [d <= radius for d in cal_distances]
        cal_coverage = sum(cal_contained) / len(cal_contained)

        # Evaluate coverage on validation set
        val_contained = [d <= radius for d in val_distances]
        val_coverage = sum(val_contained) / len(val_contained) if val_distances else 0.0

        miscalibration = val_coverage - (1 - self.alpha)

        return ConformalCalibrationResult(
            radius=radius,
            alpha=self.alpha,
            target_coverage=1 - self.alpha,
            calibration_coverage=cal_coverage,
            validation_coverage=val_coverage,
            miscalibration=miscalibration,
            n_total=len(self._records),
            n_authorized=n_authorized,
            n_calibration=len(cal_distances),
            n_validation=len(val_distances),
            calibration_distances=cal_distances,
            validation_distances=val_distances,
            validation_contained=val_contained,
        )

    @classmethod
    def from_stress_test(
        cls,
        stress_test_data: Dict,
        alpha: float = 0.1,
        cal_fraction: float = 0.7,
        seed: Optional[int] = None,
    ) -> "ConformalCalibrator":
        """
        Build a calibrator directly from stress-test JSON data.

        Expects the format from authorization-stress-test.json:
        {
            "tasks": [
                {
                    "results": [
                        {"action": "...", "verdict": "pass|escalate", "distance": 0.0},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        calibrator = cls(alpha=alpha, cal_fraction=cal_fraction, seed=seed)

        for task in stress_test_data.get("tasks", []):
            for result in task.get("results", []):
                record = ActionRecord.from_dict(result)
                calibrator.add_records([record])

        return calibrator


def calibrate_from_stress_test(
    stress_test_data: Dict,
    alpha: float = 0.1,
    seed: Optional[int] = 42,
) -> ConformalCalibrationResult:
    """
    Convenience function: calibrate radius from stress-test data in one call.

    Args:
        stress_test_data: Parsed authorization-stress-test.json
        alpha: Miscoverage rate (default 0.1 → 90% coverage target)
        seed: Random seed for reproducible split

    Returns:
        ConformalCalibrationResult with calibrated radius and diagnostics.
    """
    calibrator = ConformalCalibrator.from_stress_test(
        stress_test_data, alpha=alpha, seed=seed,
    )
    return calibrator.calibrate()
