"""
RollingThreshold — adaptive benign-quantile thresholding.

Thresholds do not transport across workloads: a threshold frozen at
calibration time sees its realized false-positive rate drift as the benign
input distribution shifts (measured in
eval/results/calibration-transport-2026-07-04.md). This module keeps the
operating point tracking the *current* benign distribution without
relabeling: the operator feeds scores confirmed benign into
:class:`RollingThreshold`, which maintains the ``1-alpha`` benign quantile
over a sliding window, and the detector consults it instead of a frozen
threshold. Prefer ``conformal=True`` (see docs/CALIBRATION.md).

Usage::

    det = CalibratedDetector.load("det.npz")
    rt = RollingThreshold(alpha=0.1, window=64, conformal=True)
    for text in stream:
        s = det.score(text)
        flagged = rt.ready and s > rt.threshold
        if operator_confirms_benign(text):
            rt.update(s)

Numpy-only; usable standalone with any score stream. (A decayed-window
mode was removed pre-release: incompatible with the conformal correction
and no measured regime needed it — track faster shifts with a smaller
window.)
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

__all__ = ["RollingThreshold"]


class RollingThreshold:
    """Benign-quantile threshold over a sliding window of confirmed-benign
    scores.

    Args:
        alpha: target false-positive rate; the threshold is the ``1-alpha``
            quantile of the windowed scores.
        window: number of most-recent benign scores retained. Smaller
            windows track shifts faster at the cost of a noisier quantile.
        min_n: scores required before :attr:`threshold` is available
            (warmup). Until then :attr:`ready` is False and
            :attr:`threshold` raises.
        conformal: use the finite-sample split-conformal order statistic
            over the window instead of the plug-in ``np.quantile``. At
            small windows the plug-in quantile is anti-conservative (its
            realized FPR overshoots ``alpha`` — measured at +0.04 for
            window 32 on real streams); the conformal correction removes
            that bias. The finite-sample guarantee is exact only under
            local exchangeability of the window — under active drift it is
            a bias correction, not a certificate. Recommended.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        window: int = 256,
        min_n: int = 20,
        conformal: bool = False,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if min_n < 2:
            raise ValueError(f"min_n must be >= 2, got {min_n}")
        self.alpha = float(alpha)
        self.window = int(window)
        self.min_n = int(min_n)
        self.conformal = bool(conformal)
        self._scores: Deque[float] = deque(maxlen=self.window)

    def update(self, score: float) -> "RollingThreshold":
        """Record one operator-confirmed benign score."""
        self._scores.append(float(score))
        return self

    @property
    def n(self) -> int:
        """Number of benign scores currently in the window."""
        return len(self._scores)

    @property
    def ready(self) -> bool:
        """True once the warmup requirement (``min_n`` scores) is met."""
        return len(self._scores) >= self.min_n

    @property
    def threshold(self) -> float:
        """Current ``1-alpha`` benign quantile of the window.

        Raises:
            RuntimeError: during warmup (fewer than ``min_n`` scores seen).
        """
        if not self.ready:
            raise RuntimeError(
                f"RollingThreshold warming up: {self.n}/{self.min_n} benign "
                "scores seen"
            )
        scores = np.asarray(self._scores, dtype=float)
        if self.conformal:
            from frontier_ops.conformal import split_conformal_threshold

            return split_conformal_threshold(scores, self.alpha)
        return float(np.quantile(scores, 1.0 - self.alpha))

    def reset(self) -> None:
        """Drop all retained scores (back to warmup)."""
        self._scores.clear()
