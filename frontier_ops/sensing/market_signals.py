"""
Market Signal Definitions — D (Deception Tax) and S (Stagnation Tax)
=====================================================================

Proprioceptive market signals that detect behavioral pathologies via
constitutional-metric-weighted trajectory analysis.

Signal D (Deception Tax):
    CUSUM over geodesic efficiency ratio. Measures how much longer the agent's
    actual path is compared to the direct distance from start to current position,
    weighted by the constitutional metric G. Zigzag evasion patterns produce D >> 1.

Signal S (Stagnation Tax):
    CUSUM over windowed action-rate deviation from empirical baseline. Replaces
    the binary paralysis detector with a continuous signal.

Both signals expose a single atomic ``step()`` method that combines update+detect,
preventing the ordering bugs that arise from separate update()/detect() calls.

See CORRECTNESS_SPEC.md §3-4 for invariants and boundary conditions.

References:
    Ahmed et al. (2024), Sequential Analysis 43(1):1-27 (DAS-CUSUM)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from frontier_ops.boundary.constitution import ConstitutionalMetric
from frontier_ops.sensing.cusum import DASCUSUM, _validate_finite

__all__ = ["DeceptionTaxSignal", "StagnationTaxSignal", "DSignalResult", "SSignalResult"]


# ---------------------------------------------------------------------------
# Default CUSUM parameters (deliberately conservative)
# ---------------------------------------------------------------------------

_DEFAULT_D_CUSUM = {
    "threshold": 5.0,
    "drift": 0.5,
    "window_size": 50,
    "decay": 0.98,
    "ceiling": 30.0,
    "signal_name": "deception_tax",
}

_DEFAULT_S_CUSUM = {
    "threshold": 4.0,
    "drift": 0.3,
    "window_size": 30,
    "decay": 0.98,
    "ceiling": 30.0,
    "signal_name": "stagnation_tax",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSignalResult:
    """Immutable result from one D signal step."""
    d_raw: float           # Geodesic efficiency ratio (≥ 1.0 for non-stationary)
    cusum_statistic: float # Accumulated CUSUM value
    alarm: bool            # Whether CUSUM threshold is exceeded


@dataclass(frozen=True)
class SSignalResult:
    """Immutable result from one S signal step."""
    s_raw: float           # Rate deviation (positive = stagnation)
    observed_rate: float   # Windowed action rate
    cusum_statistic: float
    alarm: bool


# ---------------------------------------------------------------------------
# Signal D — Deception Tax
# ---------------------------------------------------------------------------

class DeceptionTaxSignal:
    """
    Signal D: CUSUM over geodesic efficiency ratio.

    D_raw(t) = PathLength_G(trajectory) / max(DirectDistance_G(start, current), ε)

    Where PathLength_G is the cumulative metric-weighted step distance and
    DirectDistance_G is the metric-weighted distance from start to current
    using the midpoint approximation.

    A perfectly efficient straight-line trajectory has D_raw ≈ 1.0.
    Zigzag or evasive trajectories have D_raw >> 1.0.

    Invariants (CORRECTNESS_SPEC §3.2):
      D1: D_raw ≥ 1.0 for non-stationary trajectories (enforced by floor)
      D3: D_raw = 1.0 when stationary
      D4: path_length monotonically non-decreasing
      D5: Memory bounded (only start + last + path_length stored)
      D6: step() is atomic update+detect
    """

    _EPSILON = 1e-10  # Floor for geodesic distance to avoid div-by-zero

    def __init__(
        self,
        metric: ConstitutionalMetric,
        cusum_params: Optional[Dict] = None,
    ):
        self.metric = metric
        params = {**_DEFAULT_D_CUSUM, **(cusum_params or {})}
        self._cusum = DASCUSUM(**params)
        self._start: Optional[np.ndarray] = None
        self._last: Optional[np.ndarray] = None
        self._path_length: float = 0.0
        self._d_raw: float = 1.0
        self._n_steps: int = 0

    def step(self, concept_vec: np.ndarray) -> DSignalResult:
        """
        Atomic update + detect. Feed one concept vector, get result.

        This is the ONLY public mutation method. There is no separate
        update()/detect() to prevent ordering bugs (invariant D6).
        """
        concept_vec = np.asarray(concept_vec, dtype=float)
        if not np.all(np.isfinite(concept_vec)):
            raise ValueError(f"concept_vec contains non-finite values: {concept_vec}")

        self._n_steps += 1

        # First observation: initialize, no signal
        if self._start is None:
            self._start = concept_vec.copy()
            self._last = concept_vec.copy()
            self._d_raw = 1.0
            return DSignalResult(d_raw=1.0, cusum_statistic=0.0, alarm=False)

        # Accumulate path length (invariant D4: monotonically non-decreasing)
        step_dist = self.metric.metric_weighted_distance(self._last, concept_vec)
        self._path_length += step_dist
        self._last = concept_vec.copy()

        # Direct distance from start
        geodesic = self.metric.metric_weighted_distance(self._start, concept_vec)

        # Compute ratio
        if geodesic < self._EPSILON:
            # Stationary trajectory — no directional information (invariant D3)
            self._d_raw = 1.0
        else:
            raw = self._path_length / geodesic
            # Floor at 1.0 to handle midpoint-approximation artifacts (invariant D1)
            self._d_raw = max(1.0, raw)

        # Feed into CUSUM (atomic detect)
        cusum_stat, alarm = self._cusum.update(self._d_raw)

        return DSignalResult(
            d_raw=self._d_raw,
            cusum_statistic=cusum_stat,
            alarm=alarm,
        )

    def reset(self) -> None:
        """Reset all mutable state."""
        self._cusum.reset()
        self._start = None
        self._last = None
        self._path_length = 0.0
        self._d_raw = 1.0
        self._n_steps = 0

    @property
    def d_raw(self) -> float:
        """Current geodesic efficiency ratio."""
        return self._d_raw

    @property
    def path_length(self) -> float:
        """Accumulated metric-weighted path length."""
        return self._path_length

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic."""
        return self._cusum.statistic

    @property
    def n_steps(self) -> int:
        return self._n_steps


# ---------------------------------------------------------------------------
# Signal S — Stagnation Tax
# ---------------------------------------------------------------------------

class StagnationTaxSignal:
    """
    Signal S: CUSUM over windowed action-rate deviation.

    Uses a sliding window of recent timestamps to compute observed action rate,
    then measures deviation from empirical baseline:

      S_raw = baseline_rate - windowed_observed_rate

    Positive S_raw = stagnation (slower than baseline).
    Negative S_raw = faster than baseline (no alarm).

    Invariants (CORRECTNESS_SPEC §4.2):
      S1: Timestamps monotonically non-decreasing
      S2: baseline_rate > 0
      S3: Timestamp buffer bounded
      S4: observed_rate ≥ 0
      S6: step() is atomic update+detect
    """

    def __init__(
        self,
        baseline_rate: float = 1.0,
        rate_window: int = 10,
        cusum_params: Optional[Dict] = None,
    ):
        if baseline_rate <= 0:
            raise ValueError(f"baseline_rate must be > 0, got {baseline_rate}")
        if rate_window < 2:
            raise ValueError(f"rate_window must be ≥ 2, got {rate_window}")

        self.baseline_rate = baseline_rate
        self.rate_window = rate_window
        params = {**_DEFAULT_S_CUSUM, **(cusum_params or {})}
        self._cusum = DASCUSUM(**params)
        self._timestamps: deque[float] = deque(maxlen=rate_window)
        self._s_raw: float = 0.0
        self._observed_rate: float = 0.0
        self._n_steps: int = 0

    def step(self, timestamp: float) -> SSignalResult:
        """
        Atomic update + detect. Feed one action timestamp, get result.

        Raises ValueError if timestamp is non-finite or decreasing.
        """
        _validate_finite(timestamp, "timestamp")

        # Monotonicity check (invariant S1)
        if self._timestamps and timestamp < self._timestamps[-1]:
            raise ValueError(
                f"Timestamps must be non-decreasing. "
                f"Got {timestamp} after {self._timestamps[-1]}"
            )

        self._n_steps += 1
        self._timestamps.append(timestamp)

        # Need at least 2 timestamps for rate
        if len(self._timestamps) < 2:
            self._s_raw = 0.0
            self._observed_rate = 0.0
            return SSignalResult(
                s_raw=0.0, observed_rate=0.0, cusum_statistic=0.0, alarm=False,
            )

        # Windowed rate estimation (invariant S4: rate ≥ 0)
        window_duration = self._timestamps[-1] - self._timestamps[0]
        n_intervals = len(self._timestamps) - 1

        if window_duration < 1e-10:
            # All timestamps identical — treat as baseline rate
            self._observed_rate = self.baseline_rate
        else:
            self._observed_rate = n_intervals / window_duration

        # Rate deviation (positive = stagnation, invariant S5)
        self._s_raw = self.baseline_rate - self._observed_rate

        # Feed into CUSUM
        cusum_stat, alarm = self._cusum.update(self._s_raw)

        return SSignalResult(
            s_raw=self._s_raw,
            observed_rate=self._observed_rate,
            cusum_statistic=cusum_stat,
            alarm=alarm,
        )

    def set_baseline(self, action_timestamps: List[float]) -> None:
        """
        Calibrate baseline action rate from historical timestamps.

        Uses mean inter-action interval. Requires ≥ 2 timestamps.
        """
        if len(action_timestamps) < 2:
            return  # Can't compute rate from single timestamp

        intervals = np.diff(sorted(action_timestamps))
        intervals = intervals[intervals > 1e-10]  # Filter simultaneous actions
        if len(intervals) == 0:
            return

        mean_interval = float(np.mean(intervals))
        self.baseline_rate = 1.0 / mean_interval

    def reset(self) -> None:
        """Reset all mutable state. Preserves baseline_rate."""
        self._cusum.reset()
        self._timestamps.clear()
        self._s_raw = 0.0
        self._observed_rate = 0.0
        self._n_steps = 0

    @property
    def s_raw(self) -> float:
        """Current rate deviation. Positive = stagnation."""
        return self._s_raw

    @property
    def observed_rate(self) -> float:
        """Current windowed action rate."""
        return self._observed_rate

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic."""
        return self._cusum.statistic

    @property
    def n_steps(self) -> int:
        return self._n_steps
