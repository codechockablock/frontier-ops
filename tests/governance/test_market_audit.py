"""
Tests for Market Audit Chain — Governance for Market Evaluations.

Test categories (per CORRECTNESS_SPEC.md §7):
  7.1 Invariant tests — A1-A6
  7.3 Boundary condition tests — empty chain, backward compat
  7.6 Integration tests — round-trip, tamper detection, 100-evaluation chain
"""

import time

import numpy as np
import pytest

from frontier_ops.governance.chain import GovernanceChain
from frontier_ops.governance.market_audit import MarketAuditChain, MarketChainEntry
from frontier_ops.sensing.market_gate import MarketGate, SIGNAL_LABELS
from frontier_ops.sensing.market_signals import SeveritySignal, IntervalAnomalySignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_calibrated_s():
    """Create a calibrated IntervalAnomalySignal for testing."""
    import numpy as np
    s = IntervalAnomalySignal()
    rng = np.random.default_rng(42)
    s.calibrate(sorted(rng.exponential(scale=10.0, size=200)))
    return s


def _make_entry(**overrides) -> MarketChainEntry:
    defaults = dict(
        d_statistic=1.5,
        d_alarm=False,
        d_raw=1.2,
        s_statistic=0.8,
        s_alarm=False,
        s_raw=0.1,
        observed_rate=0.9,
        label_emitted=None,
        market_entropy=0.85,
        market_health=0.75,
        timestamp=time.time(),
    )
    defaults.update(overrides)
    return MarketChainEntry(**defaults)


def _make_gate_with_audit():
    d = SeveritySignal()
    s = _make_calibrated_s()
    audit = MarketAuditChain()
    gate = MarketGate(d, s, audit_chain=audit)
    return gate, audit


# ===========================================================================
# §7.1 INVARIANT TESTS — A1-A6
# ===========================================================================

class TestAuditChainInvariants:
    """CORRECTNESS_SPEC §9.2: A1-A6."""

    def test_a1_every_evaluate_produces_one_entry(self):
        """A1: Every evaluate() call produces exactly one chain entry."""
        gate, audit = _make_gate_with_audit()
        for i in range(10):
            gate.evaluate("pass", float(i))
        assert len(audit) == 10

    def test_a2_chain_is_tamper_evident(self):
        """A2: Chain is append-only and tamper-evident."""
        audit = MarketAuditChain()
        for i in range(5):
            audit.record(_make_entry(d_statistic=float(i)))
        result = audit.verify()
        assert result.valid

    def test_a3_round_trip_integrity(self):
        """A3: export() + verify() succeeds for unmodified chains."""
        audit = MarketAuditChain()
        for i in range(20):
            audit.record(_make_entry(d_statistic=float(i)))
        result = audit.verify()
        assert result.valid
        assert result.entries_verified == 20

    def test_a4_tamper_detection(self):
        """A4: Modifying any single entry causes verify() to fail."""
        chain = GovernanceChain()
        audit = MarketAuditChain(chain)

        for i in range(10):
            audit.record(_make_entry(d_statistic=float(i)))

        # Export, tamper with one entry, verify via external auditor
        exported = audit.export()
        assert len(exported) == 10

        # Tamper with entry 5's payload
        exported[5]["payload"]["d_statistic"] = 999.0

        from frontier_ops.governance.chain import GovernanceAuditor
        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)
        assert not result.valid
        assert result.first_failure == 5

    def test_a5_gate_works_without_audit_chain(self):
        """A5: MarketGate operates normally without audit chain."""
        d = SeveritySignal()
        s = _make_calibrated_s()
        gate = MarketGate(d, s)  # No audit chain

        for i in range(10):
            result = gate.evaluate("pass", float(i))
        assert gate.n_evaluations == 10

    def test_a6_chain_length_equals_evaluations(self):
        """A6: Chain length == number of evaluate() calls."""
        gate, audit = _make_gate_with_audit()
        n = 25
        for i in range(n):
            gate.evaluate("pass", float(i))
        assert len(audit) == n
        assert gate.n_evaluations == n


# ===========================================================================
# INTEGRATION / STRESS TESTS
# ===========================================================================

class TestAuditChainIntegration:
    """Chain integrity under stress and various conditions."""

    def test_chain_integrity_after_100_evaluations(self):
        """Chain verifies correctly after 100 evaluations."""
        gate, audit = _make_gate_with_audit()
        rng = np.random.default_rng(42)
        verdicts = ["pass", "monitor", "flag", "block"]
        for i in range(100):
            gate.evaluate(verdicts[rng.integers(0, 4)], float(i))
        assert len(audit) == 100
        result = audit.verify()
        assert result.valid
        assert result.entries_verified == 100

    def test_round_trip_record_export_verify(self):
        """Record → export → verify round-trip."""
        audit = MarketAuditChain()
        entries = [_make_entry(d_statistic=float(i)) for i in range(15)]
        for e in entries:
            audit.record(e)

        exported = audit.export()
        assert len(exported) == 15

        result = audit.verify()
        assert result.valid

    def test_tamper_middle_entry(self):
        """Tamper with middle entry — verify detects."""
        chain = GovernanceChain()
        audit = MarketAuditChain(chain)
        for i in range(20):
            audit.record(_make_entry(d_statistic=float(i)))

        exported = audit.export()
        exported[10]["payload"]["s_alarm"] = True  # tamper

        from frontier_ops.governance.chain import GovernanceAuditor
        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)
        assert not result.valid
        assert result.first_failure == 10

    def test_frozen_dataclass(self):
        """MarketChainEntry is immutable."""
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.d_statistic = 999.0  # type: ignore[misc]

    def test_record_rejects_wrong_type(self):
        """record() rejects non-MarketChainEntry."""
        audit = MarketAuditChain()
        with pytest.raises(TypeError, match="MarketChainEntry"):
            audit.record({"not": "an entry"})  # type: ignore[arg-type]


# ===========================================================================
# BOUNDARY TESTS
# ===========================================================================

class TestAuditChainBoundary:
    def test_verify_empty_chain(self):
        """Verify empty chain succeeds."""
        audit = MarketAuditChain()
        result = audit.verify()
        assert result.valid
        assert result.entries_verified == 0
        assert result.chain_length == 0
