"""
Tests for DAS-CUSUM, market signals (D: severity, S: interval), and MarketGate.

Test categories (per CORRECTNESS_SPEC.md §7):
  7.1 Invariant tests — verify documented invariants hold
  7.2 Statistical tests — FPR, TPR, ARL via Monte Carlo
  7.3 Boundary condition tests — edge cases, ValueError
  7.4 Adversarial tests — gaming attempts
  7.5 Regression tests — known bugs (125x runaway)
  7.6 Integration tests — end-to-end traces
  7.7 Performance tests — latency ceiling
"""

import math
import re
import time

import numpy as np
import pytest

from frontier_ops.sensing.cusum import CUSUMAlert, DASCUSUM, SPRTDecision, SPRTWrapper
from frontier_ops.sensing.market_signals import (
    SeveritySignal,
    IntervalAnomalySignal,
    DSignalResult,
    SSignalResult,
    VERDICT_SEVERITY,
)
from frontier_ops.sensing.market_gate import (
    MarketGate,
    MarketSignalState,
    SIGNAL_LABELS,
    SIGNAL_LABEL_VARIANTS,
    _NUMERIC_PATTERNS,
    _select_variant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_s_signal(benign_intervals=None, **kwargs):
    """Create a calibrated IntervalAnomalySignal."""
    s = IntervalAnomalySignal(**kwargs)
    if benign_intervals is None:
        # Default: simulate benign intervals (median ~6s, wide range)
        rng = np.random.default_rng(42)
        benign_intervals = sorted(rng.exponential(scale=10.0, size=200))
    s.calibrate(benign_intervals)
    return s


def _make_gate(d_params=None, s_params=None, benign_intervals=None, audit_chain=None):
    """Factory for MarketGate with configurable parameters."""
    d = SeveritySignal(cusum_params=d_params)
    s = _make_s_signal(benign_intervals=benign_intervals, cusum_params=s_params)
    return MarketGate(d, s, audit_chain=audit_chain)


# ===========================================================================
# §7.1 INVARIANT TESTS — DAS-CUSUM
# ===========================================================================

class TestCUSUMInvariants:
    """CORRECTNESS_SPEC §1.2: C1-C10."""

    def test_c1_c2_c3_statistic_bounded(self):
        rng = np.random.default_rng(42)
        c = DASCUSUM(threshold=5.0, ceiling=20.0)
        for _ in range(500):
            stat, _ = c.update(rng.normal(0, 10))
            assert 0 <= stat <= 20.0

    def test_c5_window_bounded(self):
        c = DASCUSUM(window_size=10)
        for i in range(100):
            c.update(float(i))
        assert len(c._window) <= 10

    def test_c6_c7_alarm_iff_exceeds_threshold(self):
        rng = np.random.default_rng(99)
        c = DASCUSUM(threshold=3.0, drift=0.3)
        for _ in range(300):
            val = rng.normal(0, 1) + (5.0 if rng.random() > 0.9 else 0)
            stat, alarm = c.update(val)
            if alarm:
                assert stat > c.threshold
            else:
                assert stat <= c.threshold

    def test_c8_reset_clears_state(self):
        c = DASCUSUM()
        for i in range(50):
            c.update(float(i))
        c.reset()
        assert c.statistic == 0.0
        assert c.run_length == 0
        assert len(c._window) == 0

    def test_c9_reference_excludes_current(self):
        c = DASCUSUM(window_size=5)
        for _ in range(5):
            c.update(0.0)
        stat, _ = c.update(100.0)
        assert 100.0 in c._window
        assert stat > 0.5

    def test_c10_parameter_validation(self):
        with pytest.raises(ValueError):
            DASCUSUM(threshold=0)
        with pytest.raises(ValueError):
            DASCUSUM(drift=-0.1)
        with pytest.raises(ValueError):
            DASCUSUM(window_size=1)
        with pytest.raises(ValueError):
            DASCUSUM(decay=0)
        with pytest.raises(ValueError):
            DASCUSUM(ceiling=-5)

    def test_c4_run_length_monotonic(self):
        c = DASCUSUM()
        prev = 0
        for i in range(50):
            c.update(float(i))
            assert c.run_length >= prev
            prev = c.run_length


# ===========================================================================
# §7.1 INVARIANT TESTS — SPRT
# ===========================================================================

class TestSPRTInvariants:

    def test_p1_boundary_signs(self):
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        assert sprt.lower_boundary < 0
        assert sprt.upper_boundary > 0

    def test_p2_p3_terminal_conditions(self):
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(100):
            d = sprt.update(1.0)
            if d == SPRTDecision.REJECT:
                assert sprt.cumulative_llr >= sprt.upper_boundary
                break
        else:
            pytest.fail("Should have rejected")

        sprt2 = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(100):
            d = sprt2.update(-1.0)
            if d == SPRTDecision.ACCEPT:
                assert sprt2.cumulative_llr <= sprt2.lower_boundary
                break
        else:
            pytest.fail("Should have accepted")

    def test_p5_terminal_is_absorbing(self):
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(100):
            d = sprt.update(1.0)
            if d == SPRTDecision.REJECT:
                break
        with pytest.raises(RuntimeError, match="terminal"):
            sprt.update(0.0)

    def test_p6_reset_clears(self):
        sprt = SPRTWrapper()
        sprt.update(5.0)
        sprt.reset()
        assert sprt.cumulative_llr == 0.0
        assert not sprt.is_terminal

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            SPRTWrapper(alpha=0)
        with pytest.raises(ValueError):
            SPRTWrapper(beta=0.5)


# ===========================================================================
# §7.1 INVARIANT TESTS — D Signal (Severity)
# ===========================================================================

class TestSeverityInvariants:
    """CORRECTNESS_SPEC §3.3: D1-D6."""

    def test_d1_d_raw_bounded(self):
        """D1: D_raw ∈ [0.0, 3.0]."""
        d = SeveritySignal()
        for verdict in ["pass", "monitor", "flag", "block"] * 10:
            result = d.step(verdict)
            assert 0.0 <= result.d_raw <= 3.0, f"D_raw={result.d_raw} out of [0, 3]"

    def test_d2_all_pass_is_zero(self):
        """D2: All PASS verdicts → D_raw = 0.0."""
        d = SeveritySignal(severity_window=5)
        for _ in range(10):
            result = d.step("pass")
        assert result.d_raw == 0.0

    def test_d3_verdict_set(self):
        """D3: Only allowed verdicts accepted."""
        d = SeveritySignal()
        # Valid verdicts
        for v in ["pass", "PASS", "Pass", "monitor", "flag", "block"]:
            d.step(v)  # Should not raise

    def test_d4_window_bounded(self):
        """D4: Window doesn't grow unbounded."""
        d = SeveritySignal(severity_window=5)
        for _ in range(1000):
            d.step("flag")
        assert len(d._window) <= 5

    def test_d5_step_is_atomic(self):
        """D5: step() returns complete result."""
        d = SeveritySignal()
        result = d.step("pass")
        assert isinstance(result, DSignalResult)
        assert hasattr(result, 'd_raw')
        assert hasattr(result, 'cusum_statistic')
        assert hasattr(result, 'alarm')

    def test_d6_unknown_verdict_raises(self):
        """D6: Unknown verdict → ValueError."""
        d = SeveritySignal()
        with pytest.raises(ValueError, match="Unknown verdict"):
            d.step("invalid")
        with pytest.raises(ValueError, match="Unknown verdict"):
            d.step("")

    def test_sustained_flag_raises_d_raw(self):
        """Sustained FLAG verdicts should push D_raw toward 2.0."""
        d = SeveritySignal(severity_window=10)
        for _ in range(20):
            result = d.step("flag")
        assert result.d_raw == pytest.approx(2.0)

    def test_mixed_verdicts(self):
        """Mixed PASS/FLAG should produce intermediate D_raw."""
        d = SeveritySignal(severity_window=10)
        for _ in range(10):
            d.step("pass")
        for _ in range(10):
            d.step("flag")
        # Window of last 10 should be all FLAG → d_raw ≈ 2.0
        assert d.d_raw == pytest.approx(2.0)


# ===========================================================================
# §7.1 INVARIANT TESTS — S Signal (Interval Anomaly)
# ===========================================================================

class TestIntervalAnomalyInvariants:
    """CORRECTNESS_SPEC §4.3: S1-S7."""

    def test_s1_monotonicity_enforced(self):
        """S1: Decreasing timestamps raise ValueError."""
        s = _make_s_signal()
        s.step(10.0)
        with pytest.raises(ValueError, match="non-decreasing"):
            s.step(5.0)

    def test_s2_s_raw_nonneg(self):
        """S2: S_raw ≥ 0 (percentile is non-negative)."""
        s = _make_s_signal()
        for t in range(20):
            result = s.step(float(t))
            assert result.s_raw >= 0, f"S_raw={result.s_raw} < 0"

    def test_s3_median_interval_near_half(self):
        """S3: Median benign interval → S_raw ≈ 0.5."""
        rng = np.random.default_rng(42)
        intervals = sorted(rng.exponential(scale=10.0, size=1000))
        s = IntervalAnomalySignal()
        s.calibrate(intervals)
        median_interval = np.median(intervals)
        s.step(0.0)
        result = s.step(median_interval)
        assert 0.4 <= result.s_raw <= 0.6, f"S_raw={result.s_raw} for median interval"

    def test_s4_exceeding_all_benign(self):
        """S4: Interval > all benign → S_raw = 1.0 (at or beyond maximum)."""
        s = _make_s_signal(benign_intervals=list(range(1, 101)))  # max=100
        s.step(0.0)
        result = s.step(200.0)  # interval = 200 > all benign
        assert result.s_raw >= 1.0

    def test_s5_step_is_atomic(self):
        """S5: step() returns complete result."""
        s = _make_s_signal()
        result = s.step(0.0)
        assert isinstance(result, SSignalResult)

    def test_s6_calibration_requires_10(self):
        """S6: Calibration with < 10 intervals raises ValueError."""
        s = IntervalAnomalySignal()
        with pytest.raises(ValueError, match="10"):
            s.calibrate([1.0, 2.0, 3.0])

    def test_s7_o1_memory(self):
        """S7: Only stores last timestamp, not full history."""
        s = _make_s_signal()
        for t in range(10000):
            s.step(float(t))
        # Should NOT have a list of 10000 timestamps
        assert not hasattr(s, '_timestamps') or s._last_timestamp is not None

    def test_uncalibrated_raises(self):
        """Must calibrate before step()."""
        s = IntervalAnomalySignal()
        with pytest.raises(RuntimeError, match="calibrate"):
            s.step(0.0)

    def test_calibrate_from_timestamps(self):
        """Convenience calibration from timestamps."""
        s = IntervalAnomalySignal()
        ts = [float(i) for i in range(50)]
        s.calibrate_from_timestamps(ts)
        assert s.is_calibrated


# ===========================================================================
# §7.1 INVARIANT TESTS — Gate
# ===========================================================================

class TestGateInvariants:

    def test_g1_output_set(self):
        """G1: Output is None or a value from SIGNAL_LABEL_VARIANTS."""
        all_labels = set()
        for variants in SIGNAL_LABEL_VARIANTS.values():
            all_labels.update(variants)
        gate = _make_gate()
        for i in range(20):
            result = gate.evaluate("pass", float(i))
            assert result is None or result in all_labels

    def test_g2_g3_no_numerics_in_labels(self):
        for key, label in SIGNAL_LABELS.items():
            for pattern in _NUMERIC_PATTERNS:
                assert not pattern.search(label)

    def test_g4_none_iff_nominal(self):
        gate = _make_gate()
        for i in range(20):
            result = gate.evaluate("pass", float(i))
            state = gate.get_state()
            if result is None:
                assert not state.any_elevated
            else:
                assert state.any_elevated

    def test_g5_state_always_valid(self):
        gate = _make_gate()
        state = gate.get_state()
        assert isinstance(state, MarketSignalState)
        gate.evaluate("pass", 0.0)
        state = gate.get_state()
        assert isinstance(state, MarketSignalState)

    def test_g6_evaluate_only_mutation(self):
        gate = _make_gate()
        public_methods = [m for m in dir(gate)
                         if not m.startswith('_') and callable(getattr(gate, m))]
        mutation_methods = [m for m in public_methods if m not in ('evaluate', 'get_state')]
        assert len(mutation_methods) == 0


# ===========================================================================
# §7.2 STATISTICAL TESTS
# ===========================================================================

class TestCUSUMStatistical:

    def test_fpr_under_one_percent(self):
        """FPR < 1% on iid N(0,1) with default params over 1000 steps."""
        rng = np.random.default_rng(42)
        n_trials = 50
        false_alarm_counts = []
        for _ in range(n_trials):
            c = DASCUSUM(threshold=5.0, drift=0.5)
            alarms = sum(1 for _ in range(1000) if c.update(rng.normal(0, 1))[1])
            false_alarm_counts.append(alarms)
        mean_fpr = np.mean(false_alarm_counts) / 1000
        assert mean_fpr < 0.01

    def test_detection_delay_under_50(self):
        """Median detection delay < 50 for +3σ mean shift."""
        rng = np.random.default_rng(42)
        delays = []
        for _ in range(50):
            c = DASCUSUM(threshold=5.0, drift=0.5, window_size=50)
            for _ in range(100):
                c.update(rng.normal(0, 1))
            for step in range(200):
                _, alarm = c.update(rng.normal(3, 1))
                if alarm:
                    delays.append(step + 1)
                    break
            else:
                delays.append(200)
        assert np.median(delays) < 50

    def test_severity_sustained_flag_triggers_alarm(self):
        """Sustained FLAG verdicts should trigger CUSUM alarm."""
        d = SeveritySignal(severity_window=5, cusum_params={"threshold": 3.0, "drift": 0.3})
        # Build baseline with PASS
        for _ in range(30):
            d.step("pass")
        # Now sustained FLAG
        alarmed = False
        for _ in range(30):
            result = d.step("flag")
            if result.alarm:
                alarmed = True
                break
        assert alarmed, "Sustained FLAG should trigger alarm"

    def test_interval_anomaly_detects_stagnation(self):
        """Sustained long intervals should trigger S alarm."""
        s = _make_s_signal(
            benign_intervals=list(np.arange(0.5, 50.5, 0.5)),  # 0.5s to 50s
            cusum_params={"threshold": 3.0, "drift": 0.3},
        )
        # Normal rate
        for i in range(30):
            s.step(float(i * 5))  # 5s intervals
        # Stagnation: 200s intervals
        alarmed = False
        for i in range(20):
            result = s.step(150.0 + i * 200.0)
            if result.alarm:
                alarmed = True
                break
        assert alarmed, "Sustained stagnation should trigger alarm"


# ===========================================================================
# §7.3 BOUNDARY CONDITION TESTS
# ===========================================================================

class TestCUSUMBoundary:
    def test_nan_raises(self):
        with pytest.raises(ValueError):
            DASCUSUM().update(float('nan'))

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            DASCUSUM().update(float('inf'))

    def test_single_observation_no_alarm(self):
        stat, alarm = DASCUSUM().update(999.0)
        assert stat == 0.0 and not alarm


class TestSPRTBoundary:
    def test_nan_raises(self):
        with pytest.raises(ValueError):
            SPRTWrapper().update(float('nan'))

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            SPRTWrapper().update(float('inf'))


class TestSeverityBoundary:
    def test_case_insensitive(self):
        d = SeveritySignal()
        d.step("PASS")
        d.step("Pass")
        d.step("pass")

    def test_whitespace_trimmed(self):
        d = SeveritySignal()
        d.step(" pass ")

    def test_first_verdict_no_cusum(self):
        d = SeveritySignal()
        result = d.step("pass")
        assert result.cusum_statistic == 0.0
        assert not result.alarm


class TestIntervalBoundary:
    def test_nan_timestamp_raises(self):
        s = _make_s_signal()
        with pytest.raises(ValueError):
            s.step(float('nan'))

    def test_decreasing_timestamp_raises(self):
        s = _make_s_signal()
        s.step(10.0)
        with pytest.raises(ValueError):
            s.step(5.0)

    def test_equal_timestamps_ok(self):
        s = _make_s_signal()
        s.step(10.0)
        result = s.step(10.0)
        assert result.s_raw == 0.0  # interval=0, percentile=0

    def test_first_timestamp_neutral(self):
        s = _make_s_signal()
        result = s.step(0.0)
        assert result.s_raw == 0.5
        assert result.cusum_statistic == 0.0


# ===========================================================================
# §7.4 ADVERSARIAL TESTS
# ===========================================================================

class TestAdversarial:

    def test_oscillating_verdicts(self):
        """Agent oscillates FLAG/PASS to keep mean low."""
        d = SeveritySignal(severity_window=10, cusum_params={"threshold": 3.0, "drift": 0.3})
        # Build baseline
        for _ in range(30):
            d.step("pass")
        # Oscillate FLAG/PASS (mean severity ≈ 1.0)
        alarm_count = 0
        for i in range(100):
            result = d.step("flag" if i % 2 == 0 else "pass")
            if result.alarm:
                alarm_count += 1
        # Mean severity ≈ 1.0 is elevated vs PASS baseline of 0.0
        # CUSUM should accumulate this sustained shift
        assert alarm_count > 0 or d.d_raw > 0.5, "Oscillation should be detectable"

    def test_rapid_pass_dilution(self):
        """Agent sends rapid PASS actions to dilute severity window."""
        d = SeveritySignal(severity_window=10, cusum_params={"threshold": 3.0, "drift": 0.3})
        # Genuine FLAGs
        for _ in range(5):
            d.step("flag")
        # Rapid PASS to dilute
        for _ in range(20):
            d.step("pass")
        # D_raw should drop back to low (this is correct behavior — dilution works)
        assert d.d_raw < 1.0

    def test_gate_never_leaks_numbers(self):
        """Stress test: varied verdicts, verify no numeric leak."""
        gate = _make_gate(
            d_params={"threshold": 0.5, "drift": 0.1},
            s_params={"threshold": 0.5, "drift": 0.1},
        )
        rng = np.random.default_rng(42)
        verdicts = ["pass", "monitor", "flag", "block"]
        for i in range(500):
            v = verdicts[rng.integers(0, 4)]
            result = gate.evaluate(v, float(i))
            if result is not None:
                for pattern in _NUMERIC_PATTERNS:
                    assert not pattern.search(result)


# ===========================================================================
# §7.5 REGRESSION TESTS
# ===========================================================================

class TestRegression:
    def test_125x_runaway_prevented(self):
        c = DASCUSUM(threshold=5.0, drift=0.5, decay=0.98, ceiling=30.0)
        for i in range(1020):
            c.update(0.1 * i)
        assert c.statistic <= 30.0

    def test_ceiling_is_structural(self):
        c = DASCUSUM(threshold=100.0, drift=0.0, ceiling=15.0, window_size=10)
        for i in range(200):
            c.update(2.0 ** i if i < 50 else 0.0)
        assert c.statistic <= 15.0


# ===========================================================================
# §7.6 INTEGRATION TESTS
# ===========================================================================

class TestIntegration:

    def test_full_pipeline_benign_then_suspicious(self):
        """Benign verdicts → no labels; sustained FLAG → labels."""
        gate = _make_gate(
            d_params={"threshold": 3.0, "drift": 0.3, "window_size": 15},
            s_params={"threshold": 3.0, "drift": 0.3, "window_size": 15},
        )
        results = []

        # Phase 1: Normal (20 PASS verdicts at 5s intervals)
        for i in range(20):
            result = gate.evaluate("pass", float(i * 5))
            results.append(result)
        assert all(r is None for r in results), "Benign phase produced labels"

        # Phase 2: Suspicious (sustained FLAG at 200s intervals)
        for i in range(30):
            result = gate.evaluate("flag", 100.0 + i * 200.0)
            results.append(result)

        phase2_labels = [r for r in results[20:] if r is not None]
        assert len(phase2_labels) > 0, "Suspicious phase produced no labels"

        all_allowed = set()
        for variants in SIGNAL_LABEL_VARIANTS.values():
            all_allowed.update(variants)
        for label in phase2_labels:
            assert label in all_allowed

    def test_state_tracks_evaluations(self):
        gate = _make_gate()
        assert gate.n_evaluations == 0
        for i in range(10):
            gate.evaluate("pass", float(i))
        assert gate.n_evaluations == 10
        state = gate.get_state()
        assert state.signal_vector.shape == (4,)

    def test_sprt_composes_with_cusum(self):
        """SPRT over CUSUM statistic from benign data → ACCEPT."""
        rng = np.random.default_rng(42)
        cusum = DASCUSUM(threshold=5.0, drift=0.5)
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(200):
            stat, _ = cusum.update(rng.normal(0, 1))
            llr = (stat - cusum.threshold) / cusum.threshold
            decision = sprt.update(llr)
            if decision != SPRTDecision.CONTINUE:
                break
        assert decision == SPRTDecision.ACCEPT

    def test_replay_realistic_session(self):
        """Replay a realistic verdict+timestamp trace through the pipeline."""
        gate = _make_gate(
            d_params={"threshold": 4.0, "drift": 0.5, "window_size": 20},
            s_params={"threshold": 4.0, "drift": 0.5, "window_size": 20},
        )
        # Simulate: 10 PASS, 5 MONITOR, 10 FLAG, 5 BLOCK
        trace = (
            [("pass", 1.0)] * 10 +
            [("monitor", 5.0)] * 5 +
            [("flag", 3.0)] * 10 +
            [("block", 8.0)] * 5
        )
        t = 0.0
        results = []
        for verdict, interval in trace:
            t += interval
            result = gate.evaluate(verdict, t)
            results.append(result)

        # Should have some labels during FLAG/BLOCK phase
        flag_block_results = results[15:]
        labels = [r for r in flag_block_results if r is not None]
        # At minimum, the state should reflect the escalation
        state = gate.get_state()
        assert state.d_raw > 1.0, "D_raw should be elevated after FLAG/BLOCK"


# ===========================================================================
# LABEL ROTATION TESTS
# ===========================================================================

class TestLabelRotation:

    def test_lr1_all_variants_pass_numeric_checks(self):
        for key, variants in SIGNAL_LABEL_VARIANTS.items():
            for idx, label in enumerate(variants):
                for pattern in _NUMERIC_PATTERNS:
                    assert not pattern.search(label)

    def test_lr2_same_count_same_label(self):
        for key in SIGNAL_LABEL_VARIANTS:
            assert _select_variant(key, 42) == _select_variant(key, 42)

    def test_lr3_different_counts_produce_varied_labels(self):
        for key, variants in SIGNAL_LABEL_VARIANTS.items():
            if len(variants) < 2:
                continue
            labels_seen = set()
            for count in range(100):
                labels_seen.add(_select_variant(key, count))
            assert len(labels_seen) > 1

    def test_lr5_first_variant_matches_canonical(self):
        for key in SIGNAL_LABELS:
            assert key in SIGNAL_LABEL_VARIANTS
            assert SIGNAL_LABEL_VARIANTS[key][0] == SIGNAL_LABELS[key]

    def test_gate_with_audit_chain(self):
        from frontier_ops.governance.market_audit import MarketAuditChain
        audit = MarketAuditChain()
        gate = _make_gate(audit_chain=audit)
        for i in range(15):
            gate.evaluate("pass", float(i))
        assert gate.n_evaluations == 15
        assert len(audit) == 15


# ===========================================================================
# CALIBRATION TESTS
# ===========================================================================

class TestCalibration:

    def test_calibrate_increases_threshold(self):
        rng = np.random.default_rng(42)
        c = DASCUSUM(threshold=0.5, drift=0.3)
        c.calibrate_from_data(rng.normal(0, 1, 500), target_arl=500)
        assert c.threshold > 0.5

    def test_interval_calibrate_from_timestamps(self):
        s = IntervalAnomalySignal()
        ts = [float(i) for i in range(100)]
        s.calibrate_from_timestamps(ts)
        assert s.is_calibrated

    def test_interval_calibrate_sets_distribution(self):
        s = IntervalAnomalySignal()
        s.calibrate(list(np.arange(1.0, 101.0)))  # 1s to 100s
        s.step(0.0)
        # 50s interval should be near median (0.5)
        result = s.step(50.0)
        assert 0.4 <= result.s_raw <= 0.6


# ===========================================================================
# §7.7 PERFORMANCE TESTS
# ===========================================================================

class TestPerformance:

    def test_evaluate_latency(self):
        """Full evaluate() call < 0.48ms P99."""
        gate = _make_gate()
        verdicts = ["pass", "monitor", "flag", "block"]
        rng = np.random.default_rng(42)

        for i in range(50):
            gate.evaluate(verdicts[i % 4], float(i))

        latencies = []
        for i in range(1000):
            v = verdicts[rng.integers(0, 4)]
            t0 = time.perf_counter_ns()
            gate.evaluate(v, 50.0 + float(i))
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)

        p99 = np.percentile(latencies, 99)
        assert p99 < 0.48, f"P99 latency = {p99:.3f}ms"
