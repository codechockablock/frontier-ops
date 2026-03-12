"""
DriftClassifier — Disambiguate drift from oscillation.

Uses autocorrelation of the NEWMA divergence time series:
- Drift: positive AC at all lags, monotonically increasing
- Oscillation: negative AC at lag ≈ period/2
- Noise: AC decays exponentially

From Claude.ai collaboration (2026-02-28).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class DriftClassifier:
    """
    Classify NEWMA signal as drift, oscillation, or noise.

    Integration: After DualEWMA.update() returns alarm, pass the
    divergence through DriftClassifier.update(). If suppress_alert=True,
    override the NEWMA flag (benign oscillation, not drift).
    """

    def __init__(
        self,
        window_size: int = 15,
        drift_ac1_min: float = 0.5,
        drift_achalf_min: float = 0.0,
        osc_achalf_max: float = -0.2,
        monotonic_override: float = 0.8,
        buffer_multiplier: int = 3,
    ):
        self.window_size = window_size
        self.drift_ac1_min = drift_ac1_min
        self.drift_achalf_min = drift_achalf_min
        self.osc_achalf_max = osc_achalf_max
        self.monotonic_override = monotonic_override
        self.buffer_multiplier = buffer_multiplier
        self._buffer: List[float] = []
        self._divergence_buffer: List[float] = self._buffer  # alias

    def update(self, newma_statistic: float) -> dict:
        """
        Feed the NEWMA divergence (use signed scalar, not unsigned norm).
        Returns classification and whether to suppress the NEWMA alert.
        """
        self._buffer.append(newma_statistic)
        max_len = self.window_size * self.buffer_multiplier
        if len(self._buffer) > max_len:
            self._buffer[:] = self._buffer[-max_len:]

        if len(self._buffer) < self.window_size:
            return {"classification": "insufficient_data", "suppress_alert": False}

        window = np.array(self._buffer[-self.window_size:])
        ac = self._autocorrelation(window)
        ac_1 = ac[1] if len(ac) > 1 else 0.0
        half_lag = self.window_size // 2
        ac_half = ac[half_lag] if len(ac) > half_lag else 0.0

        # Monotonicity
        diffs = np.diff(window)
        mono_frac = float(np.mean(diffs > 0))

        # Classify
        if ac_1 > self.drift_ac1_min and ac_half > self.drift_achalf_min:
            classification = "drift"
            suppress = False
        elif ac_1 > 0.3 and ac_half < self.osc_achalf_max:
            classification = "oscillation"
            suppress = True
        else:
            classification = "noise"
            suppress = True if ac_1 < 0.2 else False

        if mono_frac > self.monotonic_override and classification != "drift":
            classification = "drift"
            suppress = False

        return {
            "classification": classification,
            "suppress_alert": suppress,
            "autocorrelation_lag1": float(ac_1),
            "autocorrelation_lag_half": float(ac_half),
            "ac_lag1": float(ac_1),
            "ac_lag_half": float(ac_half),
            "monotonic_fraction": mono_frac,
        }

    @staticmethod
    def _autocorrelation(x: np.ndarray) -> np.ndarray:
        n = len(x)
        xc = x - x.mean()
        var = np.dot(xc, xc)
        if var < 1e-12:
            return np.zeros(n)
        ac = np.correlate(xc, xc, mode='full')
        ac = ac[n - 1:]
        ac /= var
        return ac

    def clear(self):
        self._buffer.clear()

    def reset(self):
        """Alias for clear()."""
        self.clear()
