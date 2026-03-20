"""
Market Audit Chain — Governance for Market Evaluations
========================================================

Every MarketGate.evaluate() call is signed into the existing Ed25519 hash chain,
creating a tamper-evident audit log of all market evaluations.

MarketAuditChain wraps GovernanceChain, adding ``record()`` and ``verify()``
methods specialised for market evaluation payloads.

See CORRECTNESS_SPEC.md §9 for invariants A1-A6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from frontier_ops.governance.chain import (
    GovernanceAuditor,
    GovernanceChain,
    VerificationResult,
)

__all__ = ["MarketChainEntry", "MarketAuditChain"]


# ---------------------------------------------------------------------------
# Market chain entry (frozen dataclass for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketChainEntry:
    """Immutable record of one market evaluation for auditing."""

    d_statistic: float
    d_alarm: bool
    d_raw: float
    s_statistic: float
    s_alarm: bool
    s_raw: float
    observed_rate: float
    label_emitted: Optional[str]
    market_entropy: float
    market_health: float
    timestamp: float

    def to_payload(self) -> Dict[str, Any]:
        """Convert to a dict suitable for GovernanceChain.observe()."""
        return {
            "type": "market_evaluation",
            "d_statistic": self.d_statistic,
            "d_alarm": self.d_alarm,
            "d_raw": self.d_raw,
            "s_statistic": self.s_statistic,
            "s_alarm": self.s_alarm,
            "s_raw": self.s_raw,
            "observed_rate": self.observed_rate,
            "label_emitted": self.label_emitted,
            "market_entropy": self.market_entropy,
            "market_health": self.market_health,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Market Audit Chain
# ---------------------------------------------------------------------------

class MarketAuditChain:
    """
    Tamper-evident audit chain for market evaluations.

    Wraps GovernanceChain and adds ``record()`` / ``verify()`` convenience
    methods for market-specific payloads.

    Invariants (CORRECTNESS_SPEC §9.2):
      A1: Every evaluate() produces exactly one chain entry
      A2: Chain is append-only and tamper-evident (Ed25519 + SHA-256)
      A3: export() + verify() succeeds for unmodified chains
      A4: Modifying any single entry causes verify() to fail
      A5: MarketGate works without audit chain (optional dependency)
      A6: Chain length == number of evaluate() calls
    """

    def __init__(self, chain: Optional[GovernanceChain] = None):
        if chain is None:
            self._chain = GovernanceChain()
        else:
            self._chain = chain

    def record(self, entry: MarketChainEntry) -> None:
        """
        Sign and append a market evaluation to the chain.

        Each call produces exactly one chain entry (invariant A1).
        """
        if not isinstance(entry, MarketChainEntry):
            raise TypeError(f"Expected MarketChainEntry, got {type(entry).__name__}")
        self._chain.observe(entry.to_payload())

    def verify(self) -> VerificationResult:
        """
        Verify the entire chain and return audit result.

        Succeeds for unmodified chains (A3), fails if any entry tampered (A4).
        """
        auditor = GovernanceAuditor(self._chain.public_key_hex())
        return auditor.verify_chain(self._chain.export_chain())

    def export(self) -> List[Dict[str, Any]]:
        """Export all chain entries as JSON-serialisable dicts."""
        return self._chain.export_chain()

    @property
    def public_key_hex(self) -> str:
        return self._chain.public_key_hex()

    def __len__(self) -> int:
        return len(self._chain)
