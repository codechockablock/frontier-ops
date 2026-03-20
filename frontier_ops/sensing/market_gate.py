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

Label rotation (Phase 3): Each signal state maps to 3-5 semantically equivalent
labels. Selection is deterministic given evaluation count but unpredictable to
the agent. See CORRECTNESS_SPEC.md §10.

See CORRECTNESS_SPEC.md §5 for invariants and the qualitative-only constraint.
"""

from __future__ import annotations

import hashlib
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

__all__ = [
    "MarketSignalState",
    "MarketGate",
    "SIGNAL_LABELS",
    "SIGNAL_LABEL_VARIANTS",
]


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

# Label rotation pool: 3-5 semantically equivalent variants per signal state.
# First variant in each list is the canonical label (== SIGNAL_LABELS value).
# All variants validated at import time (LR1, LR5).
SIGNAL_LABEL_VARIANTS: dict[str, list[str]] = {
    "d_only": [
        "Behavioral trajectory deviating from expected path.",
        "Approach pattern diverging from anticipated route.",
        "Observed trajectory not aligned with expected behavior.",
        "Movement pattern shows unexpected directional choices.",
    ],
    "s_only": [
        "Extended period of low-productivity activity detected.",
        "Action rate below expected baseline for sustained interval.",
        "Prolonged reduction in task-relevant activity observed.",
        "Operational tempo has dropped below expected levels.",
    ],
    "d_and_s": [
        "Multiple behavioral pressures active — review approach.",
        "Concurrent trajectory and activity anomalies detected.",
        "Several behavioral indicators warrant attention — reassess strategy.",
        "Combined directional and tempo signals suggest course correction.",
        "Simultaneous deviation in path and pace — consider adjustments.",
    ],
    "exploration_safe": [
        "Novel approach noted — proceeding.",
        "Unfamiliar strategy detected — within acceptable bounds.",
        "New behavioral pattern observed — no intervention needed.",
    ],
    "exploration_boundary": [
        "Exploration near safety boundary — exercise caution.",
        "Behavioral exploration approaching operational limits.",
        "Novel approach detected near constraint boundary — proceed carefully.",
        "Unconventional strategy close to safety margins — remain attentive.",
    ],
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


# Validate all labels at import time — fail fast on any violation (LR5)
for _k, _v in SIGNAL_LABELS.items():
    _validate_label(_k, _v)

for _k, _variants in SIGNAL_LABEL_VARIANTS.items():
    for _idx, _v in enumerate(_variants):
        _validate_label(f"{_k}[{_idx}]", _v)

# Verify canonical labels match first variant
for _k in SIGNAL_LABELS:
    assert _k in SIGNAL_LABEL_VARIANTS, (
        f"SIGNAL_LABELS key {_k!r} missing from SIGNAL_LABEL_VARIANTS"
    )
    assert SIGNAL_LABEL_VARIANTS[_k][0] == SIGNAL_LABELS[_k], (
        f"First variant of {_k!r} must equal canonical SIGNAL_LABELS value"
    )


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
# Label rotation helper
# ---------------------------------------------------------------------------

def _select_variant(signal_key: str, n_evaluations: int) -> str:
    """
    Deterministic label selection from the variant pool.

    Uses hash(n_evaluations) to select. Deterministic for same count (LR2)
    but unpredictable to the agent.
    """
    variants = SIGNAL_LABEL_VARIANTS[signal_key]
    # Hash the evaluation count for deterministic but unpredictable selection
    h = hashlib.sha256(str(n_evaluations).encode()).hexdigest()
    index = int(h, 16) % len(variants)
    return variants[index]


# ---------------------------------------------------------------------------
# Market Gate
# ---------------------------------------------------------------------------

class MarketGate:
    """
    Binary gate for qualitative market signal injection.

    Invariants (CORRECTNESS_SPEC §5.2):
      G1: Output is None or a string from SIGNAL_LABELS / SIGNAL_LABEL_VARIANTS
      G2-G3: Output contains no numeric substrings
      G4: None ⟺ no signal in alarm
      G5: get_state() always returns valid MarketSignalState
      G6: evaluate() is the only public mutation method
      G7: evaluate() calls D.step() then S.step() in fixed order

    Label rotation invariants (CORRECTNESS_SPEC §10.2):
      LR1: All variants pass numeric validation
      LR2: Same evaluation count → same label
      LR3: Different counts → different labels (usually)
    """

    def __init__(
        self,
        d_signal: DeceptionTaxSignal,
        s_signal: StagnationTaxSignal,
        audit_chain: Optional[object] = None,
    ):
        self._d = d_signal
        self._s = s_signal
        self._state = MarketSignalState()
        self._n_evaluations: int = 0
        self._audit_chain = audit_chain

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

        # Optional: entropy redistribution
        label: Optional[str] = None
        try:
            from frontier_ops.sensing.market_entropy import redistribute
            sv = self._state.signal_vector
            # Only redistribute if there are non-zero signals
            if float(np.sum(np.abs(sv))) > 1e-12:
                redistribute(sv)  # side-effect free check; used for audit
        except ImportError:
            pass

        # Select label with rotation (invariant G4: None ⟺ no alarm)
        signal_key: Optional[str] = None
        if d_result.alarm and s_result.alarm:
            signal_key = "d_and_s"
        elif d_result.alarm:
            signal_key = "d_only"
        elif s_result.alarm:
            signal_key = "s_only"

        if signal_key is not None:
            label = _select_variant(signal_key, self._n_evaluations)

        # Optional: audit chain recording
        if self._audit_chain is not None:
            try:
                from frontier_ops.governance.market_audit import MarketChainEntry
                from frontier_ops.sensing.market_entropy import (
                    market_entropy,
                    market_health,
                )
                import time as _time

                sv = self._state.signal_vector
                entry = MarketChainEntry(
                    d_statistic=self._state.d_statistic,
                    d_alarm=self._state.d_alarm,
                    d_raw=self._state.d_raw,
                    s_statistic=self._state.s_statistic,
                    s_alarm=self._state.s_alarm,
                    s_raw=self._state.s_raw,
                    observed_rate=self._state.observed_rate,
                    label_emitted=label,
                    market_entropy=market_entropy(sv),
                    market_health=market_health(sv),
                    timestamp=_time.time(),
                )
                self._audit_chain.record(entry)
            except ImportError:
                pass

        return label

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
