"""
Market Entropy — Antitrust Mechanism
======================================

Prevents signal monopoly by measuring concentration of market signal magnitudes
via Shannon and Renyi-2 entropy, with redistribution to restore diversity.

Shannon entropy (``market_entropy``) detects when a single signal dominates.
Renyi-2 entropy (``market_health``) is a stricter measure used by the health
monitor state machine.

``redistribute()`` blends monopolised signal vectors toward uniform, preserving
sign and respecting emergency overrides for critical alerts.

``MarketHealthMonitor`` is a sustained-alert state machine that requires N
consecutive unhealthy readings before declaring MONOPOLY or COLLAPSE —
single-spike robustness by design.

See CORRECTNESS_SPEC.md §8 for invariants E1-E7, R1-R5, RD1-RD7, MH1-MH5.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

__all__ = [
    "market_entropy",
    "market_health",
    "redistribute",
    "MarketHealthMonitor",
]

_EPSILON = 1e-12


def _validate_signals(signals: np.ndarray) -> np.ndarray:
    """Validate and convert signals to a 1-D float array."""
    arr = np.asarray(signals, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"signals must be 1-D, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError("signals must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"signals contain non-finite values: {arr!r}")
    return arr


# ---------------------------------------------------------------------------
# Shannon entropy (normalised)
# ---------------------------------------------------------------------------

def market_entropy(signals: Union[Sequence[float], np.ndarray]) -> float:
    """
    Normalised Shannon entropy over signal magnitudes.

    Invariants (CORRECTNESS_SPEC §8.2):
      E1: result ∈ [0, 1]
      E2: uniform → 1.0
      E3: single-source → 0.0
      E4: all-zero → 1.0 (no monopoly when no signals active)
      E5: permutation-invariant
      E6: NaN/inf → ValueError
      E7: uses absolute values
    """
    arr = _validate_signals(signals)
    magnitudes = np.abs(arr)  # E7
    total = magnitudes.sum()
    n = len(magnitudes)

    if n == 1:
        return 0.0  # log(1) = 0 — single element convention

    if total < _EPSILON:
        return 1.0  # E4: no signal → no monopoly

    p = magnitudes / total
    # Shannon entropy: -Σ p_i log(p_i) for p_i > 0
    h = 0.0
    for pi in p:
        if pi > 0:
            h -= pi * math.log(pi)

    h_max = math.log(n)
    if h_max < _EPSILON:
        return 0.0  # degenerate

    return float(np.clip(h / h_max, 0.0, 1.0))  # E1


# ---------------------------------------------------------------------------
# Renyi-2 entropy (normalised)
# ---------------------------------------------------------------------------

def market_health(signals: Union[Sequence[float], np.ndarray]) -> float:
    """
    Normalised Renyi-2 entropy over signal magnitudes.

    Invariants (CORRECTNESS_SPEC §8.3):
      R1: result ∈ [0, 1]
      R2: uniform → 1.0
      R3: single-source → 0.0
      R4: all-zero → 0.0 (collapsed market = unhealthy)
      R5: market_health ≤ market_entropy (always)
    """
    arr = _validate_signals(signals)
    magnitudes = np.abs(arr)
    total = magnitudes.sum()
    n = len(magnitudes)

    if n == 1:
        return 0.0

    if total < _EPSILON:
        return 0.0  # R4: collapsed market

    p = magnitudes / total
    sum_p_sq = float(np.sum(p ** 2))

    if sum_p_sq <= 0:
        return 0.0

    h2 = -math.log(sum_p_sq)
    h2_max = math.log(n)

    if h2_max < _EPSILON:
        return 0.0

    return float(np.clip(h2 / h2_max, 0.0, 1.0))  # R1


# ---------------------------------------------------------------------------
# Redistribution
# ---------------------------------------------------------------------------

def redistribute(
    signals: Union[Sequence[float], np.ndarray],
    floor: float = 0.5,
    critical_threshold: float = 20.0,
) -> np.ndarray:
    """
    Blend monopolised signal vectors toward uniform distribution.

    Invariants (CORRECTNESS_SPEC §8.4):
      RD1: preserves sign of each element
      RD2: returns unchanged when H ≥ floor
      RD3: returns unchanged when max(|s|) > critical_threshold (emergency)
      RD4: entropy(result) ≥ entropy(input)
      RD5: idempotent on uniform distributions
      RD6: continuous in all inputs
      RD7: α ∈ [0, 1]
    """
    arr = _validate_signals(signals)

    if len(arr) == 1:
        return arr.copy()

    magnitudes = np.abs(arr)

    # RD3: emergency override
    if float(np.max(magnitudes)) > critical_threshold:
        return arr.copy()

    total = magnitudes.sum()
    if total < _EPSILON:
        return arr.copy()  # all-zero — nothing to redistribute

    h = market_entropy(arr)

    # RD2: already diverse enough
    if h >= floor:
        return arr.copy()

    # Blending strength: 0 at floor, approaches 1 at H=0
    alpha = (floor - h) / floor  # RD7: ∈ [0, 1] since 0 ≤ h < floor
    alpha = float(np.clip(alpha, 0.0, 1.0))  # defensive

    uniform_mag = float(np.mean(magnitudes))
    signs = np.sign(arr)
    # For zero elements, sign is 0 — blend toward positive uniform_mag direction
    # (preserves sign=0 since (1-α)*0 + α*uniform*0 = 0)

    result = (1.0 - alpha) * arr + alpha * uniform_mag * signs
    return result


# ---------------------------------------------------------------------------
# Market Health Monitor — sustained-alert state machine
# ---------------------------------------------------------------------------

class MarketHealthMonitor:
    """
    State machine that requires N consecutive unhealthy readings before alert.

    States: HEALTHY → MONOPOLY_PENDING(count) → MONOPOLY
            HEALTHY → COLLAPSE_PENDING(count) → COLLAPSE

    Invariants (CORRECTNESS_SPEC §8.5):
      MH1: status ∈ {"healthy", "monopoly", "collapse"}
      MH2: MONOPOLY requires N consecutive unhealthy readings
      MH3: COLLAPSE requires N consecutive zero-total readings
      MH4: Any single healthy reading resets pending count
      MH5: Alerts for human operator, never agent context
    """

    def __init__(
        self,
        monopoly_threshold: float = 0.3,
        sustained_count: int = 5,
    ):
        if monopoly_threshold <= 0 or monopoly_threshold >= 1:
            raise ValueError(
                f"monopoly_threshold must be in (0, 1), got {monopoly_threshold}"
            )
        if sustained_count < 1:
            raise ValueError(
                f"sustained_count must be ≥ 1, got {sustained_count}"
            )

        self._monopoly_threshold = monopoly_threshold
        self._sustained_count = sustained_count

        self._status: str = "healthy"
        self._pending_state: str = "healthy"  # what we're counting toward
        self._pending_count: int = 0

    def update(self, signals: Union[Sequence[float], np.ndarray]) -> str:
        """
        Update state machine with new signal vector. Returns current status.

        Returns one of: "healthy", "monopoly", "collapse".
        """
        arr = _validate_signals(signals)
        magnitudes = np.abs(arr)
        total = float(magnitudes.sum())
        health = market_health(arr)

        if total < _EPSILON:
            # Collapse path
            if self._pending_state == "collapse":
                self._pending_count += 1
            else:
                self._pending_state = "collapse"
                self._pending_count = 1

            if self._pending_count >= self._sustained_count:
                self._status = "collapse"
        elif health < self._monopoly_threshold:
            # Monopoly path
            if self._pending_state == "monopoly":
                self._pending_count += 1
            else:
                self._pending_state = "monopoly"
                self._pending_count = 1

            if self._pending_count >= self._sustained_count:
                self._status = "monopoly"
        else:
            # Healthy — MH4: reset everything
            self._status = "healthy"
            self._pending_state = "healthy"
            self._pending_count = 0

        return self._status

    @property
    def status(self) -> str:
        """Current status: 'healthy', 'monopoly', or 'collapse'."""
        return self._status

    @property
    def pending_count(self) -> int:
        """Number of consecutive unhealthy/collapse readings."""
        return self._pending_count
