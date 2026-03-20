"""
Market Signal Definitions — D (Sustained Severity) and S (Interval Anomaly)
============================================================================

Pivoted 2026-03-20 after calibration against 330K sidecar entries, 538 frontier-ops
observations, and 24 labeled deep suite traces. The original geodesic ratio (D) and
rate deviation (S) signals were invalidated — see CALIBRATION_REPORT.md.

Signal D (Sustained Severity):
    DAS-CUSUM over windowed mean verdict severity from the existing sidecar.
    Detects when the sidecar has been elevated for a sustained period —
    the pattern that a single FLAG/BLOCK does not capture.

Signal S (Interval Anomaly):
    DAS-CUSUM over percentile-normalized inter-action intervals.
    Transforms bimodal raw intervals (burst vs idle) into a uniform [0,1]
    signal calibrated against the empirical benign distribution.

Both consume sidecar output rather than recomputing from scratch.
Both expose a single atomic ``step()`` method.

See CORRECTNESS_SPEC.md §3-4 for invariants and boundary conditions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from frontier_ops.sensing.cusum import DASCUSUM, _validate_finite

__all__ = [
    "SeveritySignal",
    "IntervalAnomalySignal",
    "DSignalResult",
    "SSignalResult",
    # Legacy names for backward compat with imports
    "DeceptionTaxSignal",
    "StagnationTaxSignal",
]


# ---------------------------------------------------------------------------
# Default CUSUM parameters (calibrated from 330K + 538 + 24 labeled traces)
# ---------------------------------------------------------------------------

_DEFAULT_D_CUSUM = {
    "threshold": 5.0,
    "drift": 0.5,
    "window_size": 30,
    "decay": 0.98,
    "ceiling": 30.0,
    "signal_name": "sustained_severity",
}

_DEFAULT_S_CUSUM = {
    "threshold": 5.0,
    "drift": 0.5,
    "window_size": 30,
    "decay": 0.98,
    "ceiling": 30.0,
    "signal_name": "interval_anomaly",
}

# Verdict severity mapping (CORRECTNESS_SPEC §3.2)
VERDICT_SEVERITY: Dict[str, float] = {
    "pass": 0.0,
    "monitor": 1.0,
    "flag": 2.0,
    "block": 3.0,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSignalResult:
    """Immutable result from one D signal step."""
    d_raw: float            # Windowed mean severity [0, 3]
    cusum_statistic: float  # Accumulated CUSUM value
    alarm: bool             # Whether CUSUM threshold is exceeded
    window_size: int        # Current window fill level


@dataclass(frozen=True)
class SSignalResult:
    """Immutable result from one S signal step."""
    s_raw: float            # Interval percentile [0, 1+]
    interval: float         # Raw inter-action interval (seconds)
    cusum_statistic: float
    alarm: bool


# ---------------------------------------------------------------------------
# Signal D — Sustained Severity
# ---------------------------------------------------------------------------

class SeveritySignal:
    """
    Signal D: DAS-CUSUM over windowed mean verdict severity.

    Consumes the sidecar's per-step verdicts (PASS/MONITOR/FLAG/BLOCK),
    maps them to severity scores (0/1/2/3), computes a rolling mean over
    a window of W recent verdicts, and feeds the windowed mean into DAS-CUSUM.

    A single FLAG is normal (tool-type switching). Sustained FLAG/BLOCK is
    the signal that the sidecar has been elevated for a concerning period.

    Invariants (CORRECTNESS_SPEC §3.3):
      D1: D_raw ∈ [0.0, 3.0]
      D2: D_raw = 0.0 when all recent verdicts are PASS
      D3: Verdict must be one of {PASS, MONITOR, FLAG, BLOCK}
      D4: Severity window bounded by capacity W
      D5: step() is atomic
      D6: Unknown verdict → ValueError
    """

    def __init__(
        self,
        severity_window: int = 10,
        cusum_params: Optional[Dict] = None,
    ) -> None:
        if severity_window < 1:
            raise ValueError(f"severity_window must be ≥ 1, got {severity_window}")

        self._severity_window = severity_window
        params = {**_DEFAULT_D_CUSUM, **(cusum_params or {})}
        self._cusum = DASCUSUM(**params)
        self._window: deque[float] = deque(maxlen=severity_window)
        self._d_raw: float = 0.0
        self._n_steps: int = 0

    def step(self, verdict: str) -> DSignalResult:
        """
        Atomic update + detect. Feed one sidecar verdict, get result.

        Args:
            verdict: One of "pass", "monitor", "flag", "block" (case-insensitive).

        Raises:
            ValueError: If verdict is not in the allowed set.
        """
        verdict_lower = verdict.strip().lower()
        if verdict_lower not in VERDICT_SEVERITY:
            raise ValueError(
                f"Unknown verdict {verdict!r}. "
                f"Must be one of: {sorted(VERDICT_SEVERITY.keys())}"
            )

        self._n_steps += 1
        severity = VERDICT_SEVERITY[verdict_lower]
        self._window.append(severity)

        # Windowed mean severity (invariant D1: ∈ [0, 3])
        self._d_raw = float(np.mean(self._window))

        # First observation: return raw, no CUSUM yet
        if self._n_steps < 2:
            return DSignalResult(
                d_raw=self._d_raw,
                cusum_statistic=0.0,
                alarm=False,
                window_size=len(self._window),
            )

        # Feed into CUSUM
        cusum_stat, alarm = self._cusum.update(self._d_raw)

        return DSignalResult(
            d_raw=self._d_raw,
            cusum_statistic=cusum_stat,
            alarm=alarm,
            window_size=len(self._window),
        )

    def reset(self) -> None:
        """Reset all mutable state."""
        self._cusum.reset()
        self._window.clear()
        self._d_raw = 0.0
        self._n_steps = 0

    @property
    def d_raw(self) -> float:
        return self._d_raw

    @property
    def statistic(self) -> float:
        return self._cusum.statistic

    @property
    def n_steps(self) -> int:
        return self._n_steps


# ---------------------------------------------------------------------------
# Signal S — Interval Anomaly
# ---------------------------------------------------------------------------

class IntervalAnomalySignal:
    """
    Signal S: DAS-CUSUM over percentile-normalized inter-action intervals.

    Calibrated against the empirical benign interval distribution. Transforms
    bimodal raw intervals (burst 0.01-2s vs idle 30s-60min+) into a uniform
    [0,1] signal where 0.5 = median benign interval.

    Values > 1.0 indicate intervals longer than any observed benign interval
    (extrapolation is valid and meaningful).

    Invariants (CORRECTNESS_SPEC §4.3):
      S1: Timestamps monotonically non-decreasing
      S2: S_raw ≥ 0
      S3: S_raw ≈ 0.5 for median benign interval
      S4: S_raw > 1.0 for intervals exceeding all calibration data
      S5: step() is atomic
      S6: Calibration data must have ≥ 10 intervals
      S7: O(1) memory for timestamps (stores only last)
    """

    def __init__(
        self,
        cusum_params: Optional[Dict] = None,
    ) -> None:
        params = {**_DEFAULT_S_CUSUM, **(cusum_params or {})}
        self._cusum = DASCUSUM(**params)
        self._benign_intervals: Optional[np.ndarray] = None  # sorted, for bisect
        self._last_timestamp: Optional[float] = None
        self._s_raw: float = 0.5
        self._last_interval: float = 0.0
        self._n_steps: int = 0
        self._calibrated: bool = False

    def calibrate(self, benign_intervals: List[float]) -> None:
        """
        Set the benign interval distribution for percentile computation.

        Args:
            benign_intervals: Inter-action intervals (seconds) from known-benign sessions.
                              Must have ≥ 10 values.

        Raises:
            ValueError: If fewer than 10 intervals provided.
        """
        intervals = np.array(benign_intervals, dtype=float)
        intervals = intervals[np.isfinite(intervals) & (intervals >= 0)]
        if len(intervals) < 10:
            raise ValueError(
                f"Need ≥ 10 benign intervals for calibration, got {len(intervals)}"
            )
        self._benign_intervals = np.sort(intervals)
        self._calibrated = True

    def calibrate_from_timestamps(self, timestamps: List[float]) -> None:
        """
        Convenience: calibrate from sorted action timestamps.

        Computes inter-action intervals and calls calibrate().
        """
        ts = np.array(sorted(timestamps), dtype=float)
        intervals = np.diff(ts)
        intervals = intervals[intervals > 0.001]  # filter < 1ms (duplicate timestamps)
        self.calibrate(intervals.tolist())

    def _percentile(self, interval: float) -> float:
        """Compute where interval falls in the benign CDF. Returns [0, 1+]."""
        if self._benign_intervals is None:
            raise RuntimeError("Must call calibrate() before step()")
        idx = np.searchsorted(self._benign_intervals, interval, side="right")
        return float(idx) / len(self._benign_intervals)

    def step(self, timestamp: float) -> SSignalResult:
        """
        Atomic update + detect. Feed one action timestamp, get result.

        Raises:
            ValueError: If timestamp is non-finite or decreasing.
            RuntimeError: If calibrate() has not been called.
        """
        _validate_finite(timestamp, "timestamp")

        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before step()")

        # Monotonicity check (invariant S1)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                f"Timestamps must be non-decreasing. "
                f"Got {timestamp} after {self._last_timestamp}"
            )

        self._n_steps += 1

        # First timestamp: neutral, no CUSUM
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._s_raw = 0.5
            self._last_interval = 0.0
            return SSignalResult(
                s_raw=0.5, interval=0.0, cusum_statistic=0.0, alarm=False,
            )

        # Compute interval and percentile
        interval = timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        self._last_interval = interval

        percentile = self._percentile(interval)
        self._s_raw = percentile

        # Feed into CUSUM
        cusum_stat, alarm = self._cusum.update(percentile)

        return SSignalResult(
            s_raw=percentile,
            interval=interval,
            cusum_statistic=cusum_stat,
            alarm=alarm,
        )

    def reset(self) -> None:
        """Reset mutable state. Preserves calibration."""
        self._cusum.reset()
        self._last_timestamp = None
        self._s_raw = 0.5
        self._last_interval = 0.0
        self._n_steps = 0

    @property
    def s_raw(self) -> float:
        return self._s_raw

    @property
    def statistic(self) -> float:
        return self._cusum.statistic

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


# ---------------------------------------------------------------------------
# Legacy aliases (backward compat for imports)
# ---------------------------------------------------------------------------

DeceptionTaxSignal = SeveritySignal
StagnationTaxSignal = IntervalAnomalySignal
