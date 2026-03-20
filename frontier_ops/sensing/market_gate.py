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

Pipeline (post-pivot):
    1. Sidecar verdict received
    2. evaluate() feeds verdict to D (severity) and timestamp to S (interval)
    3. If all nominal: returns None (zero token overhead)
    4. If any elevated: returns qualitative label string

See CORRECTNESS_SPEC.md §5 for invariants and the qualitative-only constraint.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from frontier_ops.sensing.market_signals import (
    SeveritySignal,
    IntervalAnomalySignal,
    DSignalResult,
    SSignalResult,
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

SIGNAL_LABELS: dict[str, str] = {
    "d_only": "Behavioral trajectory deviating from expected path.",
    "s_only": "Extended period of low-productivity activity detected.",
    "d_and_s": "Multiple behavioral pressures active — review approach.",
    "exploration_safe": "Novel approach noted — proceeding.",
    "exploration_boundary": "Exploration near safety boundary — exercise caution.",
}

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
    for pattern in _NUMERIC_PATTERNS:
        assert not pattern.search(label), (
            f"Label [{key!r}] contains numeric information "
            f"matching {pattern.pattern!r}: {label!r}"
        )


# Validate at import time
for _k, _v in SIGNAL_LABELS.items():
    _validate_label(_k, _v)
for _k, _variants in SIGNAL_LABEL_VARIANTS.items():
    for _idx, _v in enumerate(_variants):
        _validate_label(f"{_k}[{_idx}]", _v)
for _k in SIGNAL_LABELS:
    assert _k in SIGNAL_LABEL_VARIANTS
    assert SIGNAL_LABEL_VARIANTS[_k][0] == SIGNAL_LABELS[_k]


# ---------------------------------------------------------------------------
# Signal state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSignalState:
    """
    Immutable snapshot of all market signal states.
    For logging/governance/auditing — NEVER expose to the agent.
    """
    d_statistic: float = 0.0
    d_alarm: bool = False
    d_raw: float = 0.0
    s_statistic: float = 0.0
    s_alarm: bool = False
    s_raw: float = 0.5
    last_interval: float = 0.0
    # X and U — placeholders for Phase 4
    x_value: float = 0.0
    u_value: float = 0.0

    @property
    def any_elevated(self) -> bool:
        return self.d_alarm or self.s_alarm

    @property
    def signal_vector(self) -> np.ndarray:
        return np.array([
            self.d_statistic,
            self.s_statistic,
            self.x_value,
            self.u_value,
        ])


# ---------------------------------------------------------------------------
# Label rotation
# ---------------------------------------------------------------------------

def _select_variant(signal_key: str, n_evaluations: int) -> str:
    variants = SIGNAL_LABEL_VARIANTS[signal_key]
    h = hashlib.sha256(str(n_evaluations).encode()).hexdigest()
    index = int(h, 16) % len(variants)
    return variants[index]


# ---------------------------------------------------------------------------
# Market Gate
# ---------------------------------------------------------------------------

class MarketGate:
    """
    Binary gate for qualitative market signal injection.

    Post-pivot: consumes sidecar verdicts (D) and action timestamps (S)
    instead of raw concept vectors.

    Invariants (CORRECTNESS_SPEC §5.2):
      G1: Output is None or a string from SIGNAL_LABEL_VARIANTS
      G2-G3: Output contains no numeric substrings
      G4: None ⟺ no signal in alarm
      G5: get_state() always returns valid MarketSignalState
      G6: evaluate() is the only public mutation method
      G7: evaluate() calls D.step() then S.step() in fixed order
    """

    def __init__(
        self,
        d_signal: SeveritySignal,
        s_signal: IntervalAnomalySignal,
        audit_chain: Optional[object] = None,
    ):
        self._d = d_signal
        self._s = s_signal
        self._state = MarketSignalState()
        self._n_evaluations: int = 0
        self._audit_chain = audit_chain

    def evaluate(
        self,
        verdict: str,
        action_timestamp: float,
    ) -> Optional[str]:
        """
        Evaluate all market signals and return qualitative label if any elevated.

        Args:
            verdict: Sidecar verdict — one of "pass", "monitor", "flag", "block".
            action_timestamp: Unix timestamp of the action.

        Returns:
            None if all signals nominal.
            A qualitative label string if any signal is elevated.
        """
        self._n_evaluations += 1

        # D signal: severity (invariant G7: D first)
        d_result: DSignalResult = self._d.step(verdict)

        # S signal: interval anomaly
        s_result: SSignalResult = self._s.step(action_timestamp)

        # Update state snapshot
        self._state = MarketSignalState(
            d_statistic=d_result.cusum_statistic,
            d_alarm=d_result.alarm,
            d_raw=d_result.d_raw,
            s_statistic=s_result.cusum_statistic,
            s_alarm=s_result.alarm,
            s_raw=s_result.s_raw,
            last_interval=s_result.interval,
        )

        # Select label with rotation (G4: None ⟺ no alarm)
        label: Optional[str] = None
        signal_key: Optional[str] = None

        if d_result.alarm and s_result.alarm:
            signal_key = "d_and_s"
        elif d_result.alarm:
            signal_key = "d_only"
        elif s_result.alarm:
            signal_key = "s_only"

        if signal_key is not None:
            label = _select_variant(signal_key, self._n_evaluations)

        # Optional audit chain
        if self._audit_chain is not None:
            try:
                from frontier_ops.governance.market_audit import MarketChainEntry
                from frontier_ops.sensing.market_entropy import market_entropy, market_health
                import time as _time

                sv = self._state.signal_vector
                entry = MarketChainEntry(
                    d_statistic=self._state.d_statistic,
                    d_alarm=self._state.d_alarm,
                    d_raw=self._state.d_raw,
                    s_statistic=self._state.s_statistic,
                    s_alarm=self._state.s_alarm,
                    s_raw=self._state.s_raw,
                    observed_rate=1.0 / max(self._state.last_interval, 0.001),
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
        return self._state

    @property
    def n_evaluations(self) -> int:
        return self._n_evaluations
