"""
Market Gate — Binary Qualitative Signal Injection
===================================================

Intercepts tool-call evaluations and injects qualitative behavioral feedback
when market signals are elevated.

CRITICAL SECURITY CONSTRAINT: Qualitative labels ONLY.

    Skalse et al. (NeurIPS 2022) prove that for any non-constant reward function,
    there exists a policy that is optimal under that reward but not under the
    true objective. Any quantitative signal exposed to an optimizing agent WILL
    be optimized against. The gate therefore emits natural-language labels from
    a fixed set — the agent knows THAT the market reacted but not HOW to minimize
    the signal.

Pipeline:
    1. Tool call intercepted
    2. evaluate() computes all market signals (< 0.48ms budget)
    3. If all nominal: returns None (zero token overhead)
    4. If any elevated: returns qualitative label string

See CORRECTNESS_SPEC.md §5 for invariants and the qualitative-only constraint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from frontier_ops.sensing.market_signals import (
    DeceptionTaxSignal,
    DSignalResult,
    SSignalResult,
    StagnationTaxSignal,
)

__all__ = ["MarketSignalState", "MarketGate", "SIGNAL_LABELS"]


# ---------------------------------------------------------------------------
# Qualitative label mapping
# ---------------------------------------------------------------------------

# Every value must pass _NUMERIC_PATTERNS checks (enforced at import time).
SIGNAL_LABELS: dict[str, str] = {
    "d_only": "Behavioral trajectory deviating from expected path.",
    "s_only": "Extended period of low-productivity activity detected.",
    "d_and_s": "Multiple behavioral pressures active — review approach.",
    "exploration_safe": "Novel approach noted — proceeding.",
    "exploration_boundary": "Exploration near safety boundary — exercise caution.",
}

# Patterns that MUST NOT appear in any label (CORRECTNESS_SPEC §5.3)
_NUMERIC_PATTERNS = [
    re.compile(r"\b\d+\.\d+\b"),   # Floating point: "3.14"
    re.compile(r"\b\d{3,}\b"),      # Large integers: "100", "1024"
    re.compile(r"[=:]\s*\d"),       # Assignment-like: "D=3", "score: 7"
]


def _validate_label(key: str, label: str) -> None:
    """Raise AssertionError if a label leaks numeric information."""
    for pattern in _NUMERIC_PATTERNS:
        assert not pattern.search(label), (
            f"SIGNAL_LABELS[{key!r}] contains numeric information "
            f"matching {pattern.pattern!r}: {label!r}"
        )


# Validate all labels at import time — fail fast on any violation
for _k, _v in SIGNAL_LABELS.items():
    _validate_label(_k, _v)


# ---------------------------------------------------------------------------
# Signal state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSignalState:
    """
    Immutable snapshot of all market signal states.

    This is for logging/governance/auditing — NEVER expose to the agent.
    """
    d_statistic: float = 0.0
    d_alarm: bool = False
    d_raw: float = 1.0
    s_statistic: float = 0.0
    s_alarm: bool = False
    s_raw: float = 0.0
    observed_rate: float = 0.0
    # X and U — placeholders for Phase 4
    x_value: float = 0.0
    u_value: float = 0.0

    @property
    def any_elevated(self) -> bool:
        """True if any signal is in alarm state."""
        return self.d_alarm or self.s_alarm

    @property
    def signal_vector(self) -> np.ndarray:
        """[D, S, X, U] magnitude vector for entropy computation."""
        return np.array([
            self.d_statistic,
            self.s_statistic,
            self.x_value,
            self.u_value,
        ])


# ---------------------------------------------------------------------------
# Market Gate
# ---------------------------------------------------------------------------

class MarketGate:
    """
    Binary gate for qualitative market signal injection.

    Invariants (CORRECTNESS_SPEC §5.2):
      G1: Output is None or a string from SIGNAL_LABELS
      G2-G3: Output contains no numeric substrings
      G4: None ⟺ no signal in alarm
      G5: get_state() always returns valid MarketSignalState
      G6: evaluate() is the only public mutation method
      G7: evaluate() calls D.step() then S.step() in fixed order
    """

    def __init__(
        self,
        d_signal: DeceptionTaxSignal,
        s_signal: StagnationTaxSignal,
    ):
        self._d = d_signal
        self._s = s_signal
        self._state = MarketSignalState()
        self._n_evaluations: int = 0

    def evaluate(
        self,
        concept_vec: np.ndarray,
        action_timestamp: float,
    ) -> Optional[str]:
        """
        Evaluate all market signals and return qualitative label if any elevated.

        Returns None if all signals nominal (silent pass-through, zero token cost).
        Returns a label string if any signal is elevated.

        This is the single entry point for market evaluation (invariant G6).
        D is always evaluated before S (invariant G7).
        """
        self._n_evaluations += 1

        # D signal (invariant G7: D first)
        d_result: DSignalResult = self._d.step(concept_vec)

        # S signal
        s_result: SSignalResult = self._s.step(action_timestamp)

        # Update state snapshot
        self._state = MarketSignalState(
            d_statistic=d_result.cusum_statistic,
            d_alarm=d_result.alarm,
            d_raw=d_result.d_raw,
            s_statistic=s_result.cusum_statistic,
            s_alarm=s_result.alarm,
            s_raw=s_result.s_raw,
            observed_rate=s_result.observed_rate,
        )

        # Select label (invariant G4: None ⟺ no alarm)
        if d_result.alarm and s_result.alarm:
            return SIGNAL_LABELS["d_and_s"]
        elif d_result.alarm:
            return SIGNAL_LABELS["d_only"]
        elif s_result.alarm:
            return SIGNAL_LABELS["s_only"]
        else:
            return None

    def get_state(self) -> MarketSignalState:
        """
        Current signal state snapshot. For logging/governance only — NOT for agent.

        Returns the state from the most recent evaluate() call, or a zero-state
        if evaluate() has never been called (invariant G5).
        """
        return self._state

    @property
    def n_evaluations(self) -> int:
        """Total number of evaluate() calls."""
        return self._n_evaluations
