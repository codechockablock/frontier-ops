"""
Trend-based detectors for slow violations.
===========================================

The scope_creep gap: slow drift doesn't push past any single boundary,
but the TREND is clearly moving toward violation. Neither angular displacement
nor point-wise EWMA catches this because each step is small.

Solution: fit linear regression on sliding windows of each concept
dimension. Flag when the slope exceeds a threshold.

VALIDATION REGIME (v2 note, 2026-07): this detector's design regime is
long-running live sessions. The Apollo deception campaign only ever
exercised it on 8-23-step response-internal paths, where it was
uninformative -- that is out-of-regime evidence, not a negative result.
Untested in its design regime; do not deprecate on that data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from dataclasses import dataclass


@dataclass
class TrendAlert:
    """Alert from the trend detector."""
    dimension: str
    slope: float
    r_squared: float
    steps_observed: int
    projected_value: float  # Where the dimension will be in N more steps
    projected_boundary_crossing: Optional[int]  # Steps until boundary crossed


class ScopeCreepDetector:
    """
    Detects slow systematic drift in concept dimensions via linear regression.

    For each dimension, fits OLS on a sliding window and checks if:
    1. The slope is above threshold
    2. The R² is above threshold (it's a real trend, not noise)
    3. The projected value would cross a boundary within horizon steps
    """

    def __init__(
        self,
        dim_names: Optional[List[str]] = None,
        window_size: int = 8,
        slope_threshold: float = 0.015,  # Per-step slope that triggers alert
        r2_threshold: float = 0.5,       # Minimum R² for trend to be credible
        horizon: int = 10,               # Steps to project forward
        boundaries: Optional[Dict[str, float]] = None,
    ):
        self.dim_names = dim_names or [f"dim_{i}" for i in range(6)]
        self.n_dims = len(self.dim_names)
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        self.r2_threshold = r2_threshold
        self.horizon = horizon
        self.boundaries = boundaries or {}  # concept -> threshold

        self._history: List[np.ndarray] = []

    def observe(self, concept_vec: np.ndarray):
        """Add a new observation."""
        self._history.append(concept_vec.copy())
        # Keep only what we need
        if len(self._history) > self.window_size * 3:
            self._history = self._history[-self.window_size * 2:]

    def detect(self) -> List[TrendAlert]:
        """Run trend detection on current history."""
        if len(self._history) < self.window_size:
            return []

        window = np.array(self._history[-self.window_size:])
        t = np.arange(self.window_size, dtype=np.float64)
        alerts = []

        for dim in range(min(self.n_dims, window.shape[1])):
            y = window[:, dim]
            slope, intercept, r2 = self._fit_ols(t, y)

            if abs(slope) > self.slope_threshold and r2 > self.r2_threshold:
                # Project forward
                projected = intercept + slope * (self.window_size + self.horizon)

                # Check boundary crossing
                boundary = self.boundaries.get(self.dim_names[dim])
                crossing_step = None
                if boundary is not None and slope > 0:
                    current = y[-1]
                    if current < boundary:
                        steps_to_boundary = (boundary - current) / slope
                        crossing_step = int(steps_to_boundary)

                alerts.append(TrendAlert(
                    dimension=self.dim_names[dim],
                    slope=slope,
                    r_squared=r2,
                    steps_observed=self.window_size,
                    projected_value=projected,
                    projected_boundary_crossing=crossing_step,
                ))

        return alerts

    @staticmethod
    def _fit_ols(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Simple OLS: y = a + b*t. Returns (slope, intercept, R²)."""
        t_mean = t.mean()
        y_mean = y.mean()

        ss_tt = ((t - t_mean) ** 2).sum()
        ss_ty = ((t - t_mean) * (y - y_mean)).sum()

        if ss_tt < 1e-12:
            return 0.0, y_mean, 0.0

        slope = ss_ty / ss_tt
        intercept = y_mean - slope * t_mean

        y_pred = intercept + slope * t
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return float(slope), float(intercept), float(r2)

    @property
    def has_enough_data(self) -> bool:
        return len(self._history) >= self.window_size

    def clear(self):
        self._history.clear()


class MetricAdaptiveEWMA:
    """
    EWMA that uses the position-dependent metric tensor for adaptive sensitivity.

    Near constitutional boundaries, the metric amplifies deviations,
    making the EWMA more sensitive where it matters most.
    """

    def __init__(
        self,
        n_dims: int = 6,
        alpha: float = 0.3,
        base_threshold: float = 0.15,
    ):
        self.n_dims = n_dims
        self.alpha = alpha
        self.base_threshold = base_threshold

        self._mean: Optional[np.ndarray] = None
        self._var: Optional[np.ndarray] = None
        self._step = 0

    def update(
        self,
        concept_vec: np.ndarray,
        metric_tensor: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool]:
        """
        Update EWMA and return (metric-weighted deviation, alarm).

        If metric_tensor is provided, deviation is computed as:
            d = (x - μ)^T G (x - μ)
        Otherwise standard L2.
        """
        if self._mean is None:
            self._mean = concept_vec.copy()
            self._var = np.zeros(self.n_dims)
            self._step = 1
            return 0.0, False

        deviation = concept_vec - self._mean

        if metric_tensor is not None:
            # Metric-weighted deviation (Mahalanobis-like)
            weighted_dev = float(deviation @ metric_tensor @ deviation)
        else:
            weighted_dev = float(np.dot(deviation, deviation))

        # Update EWMA
        self._mean = self.alpha * concept_vec + (1 - self.alpha) * self._mean
        self._var = self.alpha * (deviation ** 2) + (1 - self.alpha) * self._var
        self._step += 1

        # Adaptive threshold: scale by metric trace
        if metric_tensor is not None:
            metric_scale = np.trace(metric_tensor) / self.n_dims
            threshold = self.base_threshold * metric_scale
        else:
            threshold = self.base_threshold

        alarm = weighted_dev > threshold
        return weighted_dev, alarm

    @property
    def current_mean(self) -> Optional[np.ndarray]:
        return self._mean.copy() if self._mean is not None else None

    def clear(self):
        self._mean = None
        self._var = None
        self._step = 0
