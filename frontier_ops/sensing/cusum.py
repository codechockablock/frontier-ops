"""
DAS-CUSUM Sequential Detector
==============================

Data-Adaptive Symmetric CUSUM (Ahmed et al. 2024, Sequential Analysis 43(1):1-27).

Standard CUSUM is asymmetric when both the mean and variance of the signal change
simultaneously. DAS-CUSUM uses a sliding reference window to adaptively estimate
the pre-change distribution, producing a symmetric statistic that detects shifts
in both directions with a single threshold.

The reference window EXCLUDES the current observation (no self-contamination).
The current observation is standardized against the reference, then appended.

Safety features (from 125x accumulation incident):
  - Per-step decay (default 0.98) prevents runaway accumulation
  - Hard ceiling (default 30.0) caps the statistic unconditionally
  - Both are structural: removal requires changing the code, not config

See CORRECTNESS_SPEC.md §1 for invariants, statistical properties, and boundary conditions.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["CUSUMAlert", "DASCUSUM", "SPRTDecision", "SPRTWrapper"]


def _validate_finite(value: float, name: str) -> None:
    """Raise ValueError if value is not a finite float."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CUSUMAlert:
    """Immutable alert from a CUSUM detector."""
    statistic: float
    threshold: float
    direction: str  # "up" | "down"
    run_length: int
    signal_name: str

    def __post_init__(self):
        assert self.direction in ("up", "down"), f"Bad direction: {self.direction}"


# ---------------------------------------------------------------------------
# DAS-CUSUM
# ---------------------------------------------------------------------------

class DASCUSUM:
    """
    Data-Adaptive Symmetric CUSUM detector.

    Maintains two one-sided statistics (S⁺ for upward shifts, S⁻ for downward)
    that accumulate standardized deviations from a sliding reference window.

    Invariants (see CORRECTNESS_SPEC.md §1.2):
      C1-C3: 0 ≤ S⁺, S⁻, statistic ≤ ceiling
      C5:    len(window) ≤ window_size
      C6-C7: alarm ⟺ statistic > threshold
      C9:    Reference window excludes current observation

    Parameters:
        threshold:   Detection threshold. alarm fires when statistic > threshold. (> 0)
        drift:       CUSUM allowance parameter (minimum shift to accumulate). (≥ 0)
        window_size: Sliding reference window capacity. (≥ 2)
        decay:       Per-step multiplicative decay on S⁺, S⁻. (0 < decay ≤ 1)
        ceiling:     Hard upper bound on S⁺, S⁻. (> 0)
        signal_name: Label for alerts.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        window_size: int = 50,
        decay: float = 0.98,
        ceiling: float = 30.0,
        signal_name: str = "cusum",
    ):
        # --- Parameter validation (CORRECTNESS_SPEC §1.4) ---
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if drift < 0:
            raise ValueError(f"drift must be ≥ 0, got {drift}")
        if window_size < 2:
            raise ValueError(f"window_size must be ≥ 2, got {window_size}")
        if not (0 < decay <= 1):
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if ceiling <= 0:
            raise ValueError(f"ceiling must be > 0, got {ceiling}")

        self.threshold = threshold
        self.drift = drift
        self.window_size = window_size
        self.decay = decay
        self.ceiling = ceiling
        self.signal_name = signal_name

        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._window: deque[float] = deque(maxlen=window_size)
        self._run_length: int = 0
        self._last_alarm: bool = False

    def update(self, value: float) -> Tuple[float, bool]:
        """
        Process one observation. Returns (statistic, alarm).

        The reference distribution is computed from the window BEFORE appending
        the current value (invariant C9).
        """
        _validate_finite(value, "value")
        self._run_length += 1

        # Compute reference from window BEFORE adding current observation (C9)
        if len(self._window) < 2:
            # Not enough reference data — append and return no-alarm
            self._window.append(value)
            self._last_alarm = False
            return 0.0, False

        # Adaptive reference: mean and std from existing window (excludes current)
        w = np.array(self._window)
        ref_mean = float(np.mean(w))
        ref_std = float(np.std(w, ddof=1))
        ref_std = max(ref_std, 1e-10)

        # Standardize current observation against reference
        z = (value - ref_mean) / ref_std

        # NOW append to window (after reference computation)
        self._window.append(value)

        # Decay existing statistics
        self._s_pos *= self.decay
        self._s_neg *= self.decay

        # CUSUM recursion
        self._s_pos = min(self.ceiling, max(0.0, self._s_pos + z - self.drift))
        self._s_neg = min(self.ceiling, max(0.0, self._s_neg - z - self.drift))

        # Symmetric statistic
        stat = max(self._s_pos, self._s_neg)
        alarm = stat > self.threshold
        self._last_alarm = alarm

        # Assert invariants C1-C3
        assert 0 <= self._s_pos <= self.ceiling
        assert 0 <= self._s_neg <= self.ceiling
        assert 0 <= stat <= self.ceiling

        return stat, alarm

    def get_alert(self) -> Optional[CUSUMAlert]:
        """Get structured alert if currently in alarm state. None otherwise."""
        if not self._last_alarm:
            return None
        return CUSUMAlert(
            statistic=max(self._s_pos, self._s_neg),
            threshold=self.threshold,
            direction="up" if self._s_pos >= self._s_neg else "down",
            run_length=self._run_length,
            signal_name=self.signal_name,
        )

    def reset(self) -> None:
        """Reset all mutable state. Preserves configuration."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._window.clear()
        self._run_length = 0
        self._last_alarm = False

    def calibrate_from_data(
        self,
        benign_values: np.ndarray,
        target_arl: int = 1000,
        n_trials: int = 10,
    ) -> float:
        """
        Calibrate threshold to achieve target ARL₀ on benign data.

        Uses bisection search over threshold values. For each candidate threshold,
        runs n_trials simulations (with shuffled benign data) and estimates
        the average ARL. Returns the calibrated threshold.

        Args:
            benign_values: 1D array of observations from known-benign behavior.
            target_arl:    Target average run length between false alarms.
            n_trials:      Number of simulation trials per candidate (for stability).

        Returns:
            The calibrated threshold value (also sets self.threshold).
        """
        benign_values = np.asarray(benign_values, dtype=float)
        if benign_values.ndim != 1 or len(benign_values) < 10:
            raise ValueError("Need at least 10 benign observations for calibration")

        rng = np.random.default_rng(42)

        def _estimate_arl(threshold: float) -> float:
            """Estimate ARL₀ at given threshold over n_trials shuffled runs."""
            all_arls = []
            for _ in range(n_trials):
                shuffled = rng.permutation(benign_values)
                self.threshold = threshold
                self.reset()
                steps_since_alarm = 0
                trial_arls = []
                for v in shuffled:
                    _, alarm = self.update(float(v))
                    steps_since_alarm += 1
                    if alarm:
                        trial_arls.append(steps_since_alarm)
                        steps_since_alarm = 0
                        self._s_pos = 0.0
                        self._s_neg = 0.0
                if not trial_arls:
                    # No alarms in entire run — ARL > len(data)
                    all_arls.append(float(len(shuffled)))
                else:
                    all_arls.append(float(np.mean(trial_arls)))
            return float(np.mean(all_arls))

        # Bisection search
        lo, hi = 0.5, 50.0
        for _ in range(20):  # ~20 iterations gives ~1e-6 precision
            mid = (lo + hi) / 2.0
            arl = _estimate_arl(mid)
            if arl < target_arl:
                lo = mid  # Threshold too low, raise it
            else:
                hi = mid  # Threshold sufficient, try lower
        best = (lo + hi) / 2.0

        self.threshold = best
        self.reset()
        return best

    @property
    def statistic(self) -> float:
        """Current symmetric CUSUM statistic. Invariant: 0 ≤ stat ≤ ceiling."""
        return max(self._s_pos, self._s_neg)

    @property
    def run_length(self) -> int:
        """Steps since last reset."""
        return self._run_length


# ---------------------------------------------------------------------------
# SPRT Wrapper
# ---------------------------------------------------------------------------

class SPRTDecision(Enum):
    """Terminal or continuing SPRT decision."""
    ACCEPT = "accept"
    REJECT = "reject"
    CONTINUE = "continue"


class SPRTWrapper:
    """
    Wald's Sequential Probability Ratio Test.

    Accumulates log-likelihood ratios and compares against Wald boundaries
    derived from Type I (α) and Type II (β) error rates.

    Once a terminal decision (accept/reject) is reached, the test is
    frozen — further updates raise RuntimeError. Call reset() to reuse.

    Invariants (CORRECTNESS_SPEC §2.2):
      P1: lower < 0 < upper for valid α, β
      P5: Decision is absorbing (terminal)

    Parameters:
        alpha: Type I error rate (false alarm). Must be in (0, 0.5).
        beta:  Type II error rate (missed detection). Must be in (0, 0.5).
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.10):
        if not (0 < alpha < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
        if not (0 < beta < 0.5):
            raise ValueError(f"beta must be in (0, 0.5), got {beta}")

        self.alpha = alpha
        self.beta = beta
        self._cumulative_llr: float = 0.0
        self._terminal: Optional[SPRTDecision] = None

        # Wald boundaries
        self._lower = math.log(beta / (1.0 - alpha))
        self._upper = math.log((1.0 - beta) / alpha)

        # Invariant P1: lower < 0 < upper
        assert self._lower < 0 < self._upper, \
            f"Boundary violation: lower={self._lower}, upper={self._upper}"

    def update(self, log_likelihood_ratio: float) -> SPRTDecision:
        """
        Update with one LLR observation. Returns decision.

        Raises RuntimeError if called after a terminal decision (P5).
        """
        if self._terminal is not None:
            raise RuntimeError(
                f"SPRT already reached terminal decision: {self._terminal.value}. "
                f"Call reset() to reuse."
            )
        _validate_finite(log_likelihood_ratio, "log_likelihood_ratio")

        self._cumulative_llr += log_likelihood_ratio

        if self._cumulative_llr >= self._upper:
            self._terminal = SPRTDecision.REJECT
            return SPRTDecision.REJECT
        elif self._cumulative_llr <= self._lower:
            self._terminal = SPRTDecision.ACCEPT
            return SPRTDecision.ACCEPT
        else:
            return SPRTDecision.CONTINUE

    def reset(self) -> None:
        """Reset to initial state. Clears terminal decision."""
        self._cumulative_llr = 0.0
        self._terminal = None

    @property
    def decision(self) -> Optional[SPRTDecision]:
        """Current decision, or None if no update has been called."""
        return self._terminal

    @property
    def cumulative_llr(self) -> float:
        return self._cumulative_llr

    @property
    def lower_boundary(self) -> float:
        return self._lower

    @property
    def upper_boundary(self) -> float:
        return self._upper

    @property
    def is_terminal(self) -> bool:
        return self._terminal is not None
