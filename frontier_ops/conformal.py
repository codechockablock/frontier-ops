"""
Shared split-conformal quantile core.

One function implements the finite-sample split-conformal threshold used by
both the authorization-radius calibrator
(``frontier_ops.authorization.conformal``) and the detector's conformal
threshold mode (``CalibratedDetector.calibrate_conformal``). Given
calibration nonconformity scores and a target error rate ``alpha``, the
threshold is the ``ceil((n+1)*(1-alpha))``-th smallest score — the standard
finite-sample correction. Under exchangeability of the calibration scores
with a future score, ``P(new score > threshold) <= alpha`` (marginally).

References:
  Vovk, Gammerman, Shafer (2005): Algorithmic Learning in a Random World
  Lei et al. (2018): Distribution-free predictive inference
  Angelopoulos & Bates (2023): Conformal prediction tutorial
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["split_conformal_threshold"]


def split_conformal_threshold(
    scores: Sequence[float],
    alpha: float,
    *,
    interpolate: bool = False,
) -> float:
    """Finite-sample split-conformal threshold over calibration ``scores``.

    Computes ``k = ceil((n+1) * (1-alpha))`` and returns the ``k``-th
    smallest score (1-indexed). If ``k > n`` (``n`` too small for the
    requested ``alpha``), returns ``+inf`` — no finite threshold can honor
    the guarantee, so nothing is flagged.

    Args:
        scores: calibration nonconformity scores (higher = more nonconforming).
        alpha: target marginal error rate in (0, 1).
        interpolate: legacy mode used by the authorization-radius calibrator —
            evaluates ``np.quantile(scores, min(k/n, 1.0))`` with linear
            interpolation instead of the exact order statistic. Slightly
            anti-conservative; kept for back-compatibility of calibrated
            radii. New callers should leave this False.

    Returns:
        The threshold. With ``interpolate=False`` this carries the
        finite-sample guarantee ``P(new > threshold) <= alpha`` under
        exchangeability.

    Raises:
        ValueError: if ``alpha`` is outside (0, 1) or ``scores`` is empty.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(list(scores), dtype=float).ravel()
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    k = math.ceil((n + 1) * (1 - alpha))
    if interpolate:
        q_level = min(k / n, 1.0)
        return float(np.quantile(s, q_level))
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])
