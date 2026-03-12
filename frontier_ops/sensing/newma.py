"""
NEWMA (Network EWMA) / Dual-EWMA Drift Detector
=================================================

Based on Keriven et al. (2020) and Claude.ai recommendation (2026-02-28).

Two EWMAs with different smoothing parameters detect drift at ALL scales
simultaneously. The fast EWMA tracks current position; the slow EWMA
preserves memory of the starting baseline.

Detection statistic: ||mu_fast - mu_slow||_G (metric-weighted divergence)

Under no drift: both converge to same mean, statistic → 0
Under monotonic drift: fast leads slow, gap grows linearly with time
After T steps of drift rate ε: gap ≈ ε · T · (1/α_slow - 1/α_fast)

This is SPECIFICALLY designed for the scope_creep detection gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class NEWMAAlert:
    """Alert from the NEWMA detector."""
    divergence: float  # ||mu_fast - mu_slow||_G
    divergence_unweighted: float  # ||mu_fast - mu_slow||_2
    drift_direction: np.ndarray  # Unit vector in direction of drift
    drift_rate: float  # Estimated per-step drift rate
    steps: int
    threshold: float


class DualEWMA:
    """
    NEWMA-style dual EWMA for multi-scale drift detection.

    The key insight: single EWMA measures distance from the running mean,
    which adapts to drift. Dual EWMA measures distance between two means
    with different adaptation rates, which grows monotonically under drift.
    """

    def __init__(
        self,
        n_dims: int = 6,
        alpha_fast: float = 0.5,
        alpha_slow: float = 0.05,
        threshold: float = 0.1,
        warmup_steps: int = 5,
    ):
        self.n_dims = n_dims
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.threshold = threshold
        self.warmup_steps = warmup_steps

        self._mu_fast: Optional[np.ndarray] = None
        self._mu_slow: Optional[np.ndarray] = None
        self._step = 0
        self._divergence_history: List[float] = []

    def update(
        self,
        concept_vec: np.ndarray,
        metric_tensor: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool]:
        """
        Update both EWMAs and return (divergence, alarm).

        Args:
            concept_vec: Current concept vector.
            metric_tensor: Position-dependent G(x) for metric-weighted divergence.

        Returns:
            (divergence, alarm): metric-weighted divergence and threshold check.
        """
        if self._mu_fast is None:
            self._mu_fast = concept_vec.copy()
            self._mu_slow = concept_vec.copy()
            self._step = 1
            return 0.0, False

        # Update both EWMAs
        self._mu_fast = self.alpha_fast * concept_vec + (1 - self.alpha_fast) * self._mu_fast
        self._mu_slow = self.alpha_slow * concept_vec + (1 - self.alpha_slow) * self._mu_slow
        self._step += 1

        # Compute divergence
        diff = self._mu_fast - self._mu_slow

        if metric_tensor is not None:
            divergence = float(np.sqrt(max(0, diff @ metric_tensor @ diff)))
        else:
            divergence = float(np.linalg.norm(diff))

        self._divergence_history.append(divergence)

        # No alarm during warmup
        if self._step < self.warmup_steps:
            return divergence, False

        alarm = divergence > self.threshold
        return divergence, alarm

    def get_alert(self, metric_tensor: Optional[np.ndarray] = None) -> Optional[NEWMAAlert]:
        """Get structured alert if currently in alarm state."""
        if self._mu_fast is None or self._mu_slow is None:
            return None

        diff = self._mu_fast - self._mu_slow

        if metric_tensor is not None:
            divergence = float(np.sqrt(max(0, diff @ metric_tensor @ diff)))
        else:
            divergence = float(np.linalg.norm(diff))

        if divergence <= self.threshold:
            return None

        norm = np.linalg.norm(diff)
        direction = diff / norm if norm > 1e-10 else np.zeros(self.n_dims)

        # Estimate drift rate from divergence history
        if len(self._divergence_history) >= 3:
            recent = self._divergence_history[-5:]
            drift_rate = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0.0
        else:
            drift_rate = 0.0

        return NEWMAAlert(
            divergence=divergence,
            divergence_unweighted=float(np.linalg.norm(diff)),
            drift_direction=direction,
            drift_rate=drift_rate,
            steps=self._step,
            threshold=self.threshold,
        )

    @property
    def mu_fast(self) -> Optional[np.ndarray]:
        return self._mu_fast.copy() if self._mu_fast is not None else None

    @property
    def mu_slow(self) -> Optional[np.ndarray]:
        return self._mu_slow.copy() if self._mu_slow is not None else None

    @property
    def divergence(self) -> float:
        """Current divergence."""
        if self._mu_fast is None or self._mu_slow is None:
            return 0.0
        return float(np.linalg.norm(self._mu_fast - self._mu_slow))

    def clear(self):
        self._mu_fast = None
        self._mu_slow = None
        self._step = 0
        self._divergence_history.clear()
