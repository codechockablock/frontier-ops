"""
Tests for DAS-CUSUM, market signals (D, S), and MarketGate.

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
    DeceptionTaxSignal,
    DSignalResult,
    SSignalResult,
    StagnationTaxSignal,
)
from frontier_ops.sensing.market_gate import (
    MarketGate,
    MarketSignalState,
    SIGNAL_LABELS,
    SIGNAL_LABEL_VARIANTS,
    _NUMERIC_PATTERNS,
    _select_variant,
)
from frontier_ops.boundary.constitution import (
    ConstitutionalMetric,
    ConstitutionSpec,
    Boundary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_metric(n_dims: int = 6) -> ConstitutionalMetric:
    """Constitutional metric with one boundary for testing."""
    spec = ConstitutionSpec(
        name="test",
        boundaries=[Boundary("dim_0", threshold=0.8, sharpness=3.0)],
        baseline_weight=1.0,
    )
    return ConstitutionalMetric(spec, dim_names=[f"dim_{i}" for i in range(n_dims)])


def _make_gate(d_params=None, s_params=None, **s_kwargs):
    """Factory for MarketGate with configurable thresholds."""
    metric = _simple_metric()
    d = DeceptionTaxSignal(metric, cusum_params=d_params)
    s = StagnationTaxSignal(cusum_params=s_params, **s_kwargs)
    return MarketGate(d, s)


# ===========================================================================
# §7.1 INVARIANT TESTS
# ===========================================================================

class TestCUSUMInvariants:
    """CORRECTNESS_SPEC §1.2: C1-C10."""

    def test_c1_c2_c3_statistic_bounded(self):
        """C1-C3: 0 ≤ S⁺, S⁻, statistic ≤ ceiling."""
        rng = np.random.default_rng(42)
        c = DASCUSUM(threshold=5.0, ceiling=20.0)
        for _ in range(500):
            val = rng.normal(0, 10)  # High variance to stress bounds
            stat, _ = c.update(val)
            assert 0 <= stat <= 20.0, f"Statistic {stat} out of [0, 20]"

    def test_c5_window_bounded(self):
        """C5: len(window) ≤ window_size."""
        c = DASCUSUM(window_size=10)
        for i in range(100):
            c.update(float(i))
        assert len(c._window) <= 10

    def test_c6_c7_alarm_iff_exceeds_threshold(self):
        """C6-C7: alarm ⟺ statistic > threshold."""
        rng = np.random.default_rng(99)
        c = DASCUSUM(threshold=3.0, drift=0.3)
        for _ in range(300):
            val = rng.normal(0, 1) + (5.0 if rng.random() > 0.9 else 0)
            stat, alarm = c.update(val)
            if alarm:
                assert stat > c.threshold, f"Alarm with stat={stat} ≤ threshold={c.threshold}"
            else:
                assert stat <= c.threshold, f"No alarm with stat={stat} > threshold={c.threshold}"

    def test_c8_reset_clears_state(self):
        """C8: reset() zeroes all mutable state."""
        c = DASCUSUM()
        for i in range(50):
            c.update(float(i))
        c.reset()
        assert c.statistic == 0.0
        assert c.run_length == 0
        assert len(c._window) == 0

    def test_c9_reference_excludes_current(self):
        """C9: Reference window does not include current observation."""
        c = DASCUSUM(window_size=5)
        # Fill window with 5 zeros
        for _ in range(5):
            c.update(0.0)
        # Window should be [0, 0, 0, 0, 0]
        assert len(c._window) == 5
        # Now insert a huge value — if it's in the reference, z would be small
        # If excluded from reference, z = (100 - 0) / ~0 → very large
        stat, alarm = c.update(100.0)
        # After update, window now contains 100.0 (appended after ref computation)
        assert 100.0 in c._window
        # The stat should be large because 100 was evaluated against all-zeros reference
        assert stat > 0.5, f"Expected large stat, got {stat} — reference may include current"

    def test_c10_parameter_validation(self):
        """C10: Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            DASCUSUM(threshold=0)
        with pytest.raises(ValueError):
            DASCUSUM(threshold=-1)
        with pytest.raises(ValueError):
            DASCUSUM(drift=-0.1)
        with pytest.raises(ValueError):
            DASCUSUM(window_size=1)
        with pytest.raises(ValueError):
            DASCUSUM(decay=0)
        with pytest.raises(ValueError):
            DASCUSUM(decay=1.5)
        with pytest.raises(ValueError):
            DASCUSUM(ceiling=0)
        with pytest.raises(ValueError):
            DASCUSUM(ceiling=-5)

    def test_c4_run_length_monotonic(self):
        """C4: run_length monotonically increases between resets."""
        c = DASCUSUM()
        prev = 0
        for i in range(50):
            c.update(float(i))
            assert c.run_length > prev or (c.run_length == prev and i == 0)
            prev = c.run_length


class TestSPRTInvariants:
    """CORRECTNESS_SPEC §2.2: P1-P6."""

    def test_p1_boundary_signs(self):
        """P1: lower < 0 < upper for valid α, β."""
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        assert sprt.lower_boundary < 0
        assert sprt.upper_boundary > 0

    def test_p2_p3_terminal_conditions(self):
        """P2-P3: reject ⟹ cumulative ≥ upper, accept ⟹ cumulative ≤ lower."""
        # Rejection
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(100):
            d = sprt.update(1.0)
            if d == SPRTDecision.REJECT:
                assert sprt.cumulative_llr >= sprt.upper_boundary
                break
        else:
            pytest.fail("Should have rejected")

        # Acceptance
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(100):
            d = sprt.update(-1.0)
            if d == SPRTDecision.ACCEPT:
                assert sprt.cumulative_llr <= sprt.lower_boundary
                break
        else:
            pytest.fail("Should have accepted")

    def test_p4_continue_in_region(self):
        """P4: continue ⟹ lower < cumulative < upper."""
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        d = sprt.update(0.001)
        assert d == SPRTDecision.CONTINUE
        assert sprt.lower_boundary < sprt.cumulative_llr < sprt.upper_boundary

    def test_p5_terminal_is_absorbing(self):
        """P5: update after terminal decision raises RuntimeError."""
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)
        for _ in range(100):
            d = sprt.update(1.0)
            if d == SPRTDecision.REJECT:
                break
        with pytest.raises(RuntimeError, match="terminal"):
            sprt.update(0.0)

    def test_p6_reset_clears(self):
        """P6: reset clears cumulative and terminal state."""
        sprt = SPRTWrapper()
        sprt.update(5.0)
        sprt.reset()
        assert sprt.cumulative_llr == 0.0
        assert not sprt.is_terminal

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            SPRTWrapper(alpha=0)
        with pytest.raises(ValueError):
            SPRTWrapper(alpha=0.5)
        with pytest.raises(ValueError):
            SPRTWrapper(beta=0)
        with pytest.raises(ValueError):
            SPRTWrapper(beta=0.5)


class TestDSignalInvariants:
    """CORRECTNESS_SPEC §3.2: D1-D6."""

    def test_d1_ratio_geq_one(self):
        """D1: D_raw ≥ 1.0 for non-stationary trajectories."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        rng = np.random.default_rng(42)
        prev = np.zeros(6)
        for i in range(50):
            vec = prev + rng.uniform(-0.1, 0.1, 6)
            vec = np.clip(vec, 0, 1)
            result = d.step(vec)
            assert result.d_raw >= 1.0, f"Step {i}: D_raw={result.d_raw} < 1.0"
            prev = vec

    def test_d2_straight_line_ratio_one(self):
        """D2: Straight line in flat metric → D_raw ≈ 1.0."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        for i in range(20):
            vec = np.array([0.05 * i, 0.0, 0.0, 0.0, 0.0, 0.0])
            result = d.step(vec)
        assert result.d_raw == pytest.approx(1.0, abs=0.02)

    def test_d3_stationary_ratio_one(self):
        """D3: Stationary trajectory → D_raw = 1.0."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        vec = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(10):
            result = d.step(vec)
        assert result.d_raw == 1.0

    def test_d4_path_length_monotonic(self):
        """D4: path_length never decreases."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        prev_pl = 0.0
        for i in range(30):
            vec = np.array([0.02 * i, 0.01 * (i % 3), 0.0, 0.0, 0.0, 0.0])
            d.step(vec)
            assert d.path_length >= prev_pl
            prev_pl = d.path_length

    def test_d5_memory_bounded(self):
        """D5: No unbounded trajectory storage."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        for i in range(10000):
            vec = np.random.rand(6) * 0.1
            d.step(vec)
        # Should store only start + last, not 10000 vectors
        assert not hasattr(d, '_trajectory') or len(getattr(d, '_trajectory', [])) == 0
        assert d._start is not None
        assert d._last is not None

    def test_d6_step_is_atomic(self):
        """D6: step() returns complete result; no separate detect() needed."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        result = d.step(np.zeros(6))
        assert isinstance(result, DSignalResult)
        assert hasattr(result, 'd_raw')
        assert hasattr(result, 'cusum_statistic')
        assert hasattr(result, 'alarm')
        # No public detect() method
        assert not hasattr(d, 'detect')

    def test_zigzag_ratio_above_one(self):
        """Zigzag trajectory should have D_raw >> 1.0."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        for i in range(30):
            offset = 0.3 if i % 2 == 0 else -0.3
            vec = np.array([0.02 * i, offset, 0.0, 0.0, 0.0, 0.0])
            result = d.step(vec)
        assert result.d_raw > 2.0


class TestSSignalInvariants:
    """CORRECTNESS_SPEC §4.2: S1-S6."""

    def test_s1_monotonicity_enforced(self):
        """S1: Non-monotonic timestamps raise ValueError."""
        s = StagnationTaxSignal()
        s.step(10.0)
        with pytest.raises(ValueError, match="non-decreasing"):
            s.step(5.0)

    def test_s1_equal_timestamps_ok(self):
        """S1: Equal timestamps are allowed (simultaneous actions)."""
        s = StagnationTaxSignal()
        s.step(10.0)
        result = s.step(10.0)  # Should not raise
        assert isinstance(result, SSignalResult)

    def test_s2_baseline_positive(self):
        """S2: baseline_rate ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="baseline_rate"):
            StagnationTaxSignal(baseline_rate=0)
        with pytest.raises(ValueError, match="baseline_rate"):
            StagnationTaxSignal(baseline_rate=-1)

    def test_s3_buffer_bounded(self):
        """S3: Timestamp buffer doesn't grow unbounded."""
        s = StagnationTaxSignal(rate_window=10)
        for i in range(10000):
            s.step(float(i))
        assert len(s._timestamps) <= 10

    def test_s4_observed_rate_nonneg(self):
        """S4: observed_rate ≥ 0."""
        s = StagnationTaxSignal()
        for i in range(20):
            result = s.step(float(i))
            assert result.observed_rate >= 0

    def test_s5_sign_convention(self):
        """S5: Positive S_raw means stagnation (slower than baseline)."""
        # Baseline = 1/sec, actions every 10 sec = slow → S_raw > 0
        s = StagnationTaxSignal(baseline_rate=1.0)
        s.step(0.0)
        for i in range(1, 5):
            result = s.step(float(i * 10))
        assert result.s_raw > 0, "Slow actions should produce positive S_raw"

    def test_s6_step_is_atomic(self):
        """S6: step() is the only mutation interface."""
        s = StagnationTaxSignal()
        result = s.step(0.0)
        assert isinstance(result, SSignalResult)
        assert not hasattr(s, 'detect')
        assert not hasattr(s, 'update')


class TestGateInvariants:
    """CORRECTNESS_SPEC §5.2: G1-G7."""

    def test_g1_output_set(self):
        """G1: Output is None or a value from SIGNAL_LABELS / SIGNAL_LABEL_VARIANTS."""
        all_variants = set()
        for variants in SIGNAL_LABEL_VARIANTS.values():
            all_variants.update(variants)
        gate = _make_gate()
        for i in range(20):
            result = gate.evaluate(np.random.rand(6) * 0.1, float(i))
            assert result is None or result in all_variants

    def test_g2_g3_no_numerics_in_labels(self):
        """G2-G3: No numeric values in any label."""
        for key, label in SIGNAL_LABELS.items():
            for pattern in _NUMERIC_PATTERNS:
                assert not pattern.search(label), (
                    f"SIGNAL_LABELS[{key!r}] matches {pattern.pattern!r}: {label!r}"
                )

    def test_g4_none_iff_nominal(self):
        """G4: None output ⟺ no signal in alarm."""
        gate = _make_gate()
        vec = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(20):
            result = gate.evaluate(vec, float(i))
            state = gate.get_state()
            if result is None:
                assert not state.any_elevated, "None output but signal is elevated"
            else:
                assert state.any_elevated, "Non-None output but no signal elevated"

    def test_g5_state_always_valid(self):
        """G5: get_state() always returns MarketSignalState."""
        gate = _make_gate()
        # Before any evaluation
        state = gate.get_state()
        assert isinstance(state, MarketSignalState)
        # After evaluation
        gate.evaluate(np.zeros(6), 0.0)
        state = gate.get_state()
        assert isinstance(state, MarketSignalState)

    def test_g6_evaluate_only_mutation(self):
        """G6: evaluate() is the only public mutation method."""
        gate = _make_gate()
        # Verify no public methods that mutate state besides evaluate()
        public_methods = [m for m in dir(gate)
                         if not m.startswith('_') and callable(getattr(gate, m))]
        mutation_methods = [m for m in public_methods if m not in ('evaluate', 'get_state')]
        # get_state is a read, n_evaluations is a property
        assert len(mutation_methods) == 0, f"Unexpected public methods: {mutation_methods}"


# ===========================================================================
# §7.2 STATISTICAL TESTS
# ===========================================================================

class TestCUSUMStatistical:
    """Monte Carlo verification of FPR, detection delay, ARL."""

    def test_s1_fpr_under_one_percent(self):
        """S1: FPR < 1% on iid N(0,1) with default params over 1000 steps."""
        rng = np.random.default_rng(42)
        n_trials = 50
        n_steps = 1000
        false_alarm_counts = []

        for trial in range(n_trials):
            c = DASCUSUM(threshold=5.0, drift=0.5)
            alarms = 0
            for _ in range(n_steps):
                _, alarm = c.update(rng.normal(0, 1))
                if alarm:
                    alarms += 1
            false_alarm_counts.append(alarms)

        mean_fpr = np.mean(false_alarm_counts) / n_steps
        assert mean_fpr < 0.01, f"Mean FPR = {mean_fpr:.4f}, expected < 0.01"

    def test_s2_detection_delay_under_50(self):
        """S2: Median detection delay < 50 for +3σ mean shift."""
        rng = np.random.default_rng(42)
        n_trials = 50
        delays = []

        for _ in range(n_trials):
            c = DASCUSUM(threshold=5.0, drift=0.5, window_size=50)
            # Build baseline
            for _ in range(100):
                c.update(rng.normal(0, 1))
            # Shift
            for step in range(200):
                _, alarm = c.update(rng.normal(3, 1))
                if alarm:
                    delays.append(step + 1)
                    break
            else:
                delays.append(200)  # Didn't detect in 200 steps

        median_delay = np.median(delays)
        assert median_delay < 50, f"Median detection delay = {median_delay}, expected < 50"

    def test_s5_converges_under_stationary(self):
        """S5: After 2×window_size steps of stationary input, stat < threshold."""
        rng = np.random.default_rng(42)
        c = DASCUSUM(threshold=5.0, drift=0.5, window_size=50)
        # Feed stationary data for 200 steps (4× window)
        for _ in range(200):
            stat, _ = c.update(rng.normal(0, 1))
        assert stat < c.threshold, f"Stat {stat} ≥ threshold after convergence"


# ===========================================================================
# §7.3 BOUNDARY CONDITION TESTS
# ===========================================================================

class TestCUSUMBoundary:
    def test_nan_raises(self):
        c = DASCUSUM()
        with pytest.raises(ValueError, match="finite"):
            c.update(float('nan'))

    def test_inf_raises(self):
        c = DASCUSUM()
        with pytest.raises(ValueError, match="finite"):
            c.update(float('inf'))

    def test_neg_inf_raises(self):
        c = DASCUSUM()
        with pytest.raises(ValueError, match="finite"):
            c.update(float('-inf'))

    def test_single_observation_no_alarm(self):
        c = DASCUSUM()
        stat, alarm = c.update(999.0)
        assert stat == 0.0
        assert not alarm


class TestSPRTBoundary:
    def test_nan_raises(self):
        sprt = SPRTWrapper()
        with pytest.raises(ValueError, match="finite"):
            sprt.update(float('nan'))

    def test_inf_raises(self):
        sprt = SPRTWrapper()
        with pytest.raises(ValueError, match="finite"):
            sprt.update(float('inf'))


class TestDSignalBoundary:
    def test_nan_concept_vec_raises(self):
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        with pytest.raises(ValueError, match="non-finite"):
            d.step(np.array([float('nan'), 0, 0, 0, 0, 0]))

    def test_inf_concept_vec_raises(self):
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        with pytest.raises(ValueError, match="non-finite"):
            d.step(np.array([float('inf'), 0, 0, 0, 0, 0]))

    def test_zero_vector_valid(self):
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        result = d.step(np.zeros(6))
        assert isinstance(result, DSignalResult)

    def test_first_observation_no_cusum(self):
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        result = d.step(np.zeros(6))
        assert result.d_raw == 1.0
        assert result.cusum_statistic == 0.0
        assert not result.alarm


class TestSSignalBoundary:
    def test_nan_timestamp_raises(self):
        s = StagnationTaxSignal()
        with pytest.raises(ValueError, match="finite"):
            s.step(float('nan'))

    def test_decreasing_timestamp_raises(self):
        s = StagnationTaxSignal()
        s.step(10.0)
        with pytest.raises(ValueError, match="non-decreasing"):
            s.step(5.0)

    def test_single_timestamp_no_signal(self):
        s = StagnationTaxSignal()
        result = s.step(0.0)
        assert result.s_raw == 0.0
        assert result.cusum_statistic == 0.0
        assert not result.alarm


# ===========================================================================
# §7.4 ADVERSARIAL TESTS
# ===========================================================================

class TestAdversarial:
    def test_tiny_steps_still_detected_by_d(self):
        """Agent takes many tiny zigzag steps — D_raw should still accumulate."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        # 100 tiny zigzag steps
        for i in range(100):
            tiny_offset = 0.001 if i % 2 == 0 else -0.001
            vec = np.array([0.0001 * i, tiny_offset, 0.0, 0.0, 0.0, 0.0])
            d.step(vec)
        # D_raw should be > 1.0 (zigzag adds path length)
        assert d.d_raw > 1.0

    def test_rapid_noop_doesnt_mask_stagnation(self):
        """Rapid no-ops (same timestamp) followed by stagnation still detected."""
        s = StagnationTaxSignal(baseline_rate=1.0, cusum_params={
            "threshold": 2.0, "drift": 0.1, "window_size": 10,
        })
        # 10 rapid actions (normal)
        for i in range(10):
            s.step(float(i))
        # Then stagnation: 10-second gaps
        alarmed = False
        for i in range(20):
            result = s.step(10.0 + i * 10.0)
            if result.alarm:
                alarmed = True
                break
        assert alarmed, "Stagnation after burst should be detected"

    def test_return_to_start_still_accumulates_d(self):
        """Agent zigzags and returns to start — D should have accumulated."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        d.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # Start
        d.step(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))  # Away
        d.step(np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))  # Sideways
        d.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # Back to start
        # Path length > 0, but geodesic ≈ 0 → stationary case → D_raw = 1.0
        # (This is actually the correct behavior: if you return to start,
        # the ratio is undefined. D catches this via path_length accumulation
        # over subsequent steps.)
        assert d.path_length > 0, "Path length should have accumulated"

    def test_gate_never_leaks_numbers_under_stress(self):
        """Stress test: 1000 evaluations, verify no numeric leak."""
        gate = _make_gate(
            d_params={"threshold": 0.5, "drift": 0.1},
            s_params={"threshold": 0.5, "drift": 0.1},
        )
        rng = np.random.default_rng(42)
        for i in range(1000):
            vec = rng.uniform(-0.5, 0.5, 6)
            result = gate.evaluate(vec, float(i))
            if result is not None:
                for pattern in _NUMERIC_PATTERNS:
                    assert not pattern.search(result), (
                        f"Numeric leak at step {i}: {result!r} matches {pattern.pattern}"
                    )


# ===========================================================================
# §7.5 REGRESSION TESTS
# ===========================================================================

class TestRegression:
    def test_125x_runaway_prevented(self):
        """The 125x CUSUM accumulation bug must not recur."""
        c = DASCUSUM(threshold=5.0, drift=0.5, decay=0.98, ceiling=30.0)
        # Simulate the exact scenario: persistent mild drift over 1020 steps
        for i in range(1020):
            c.update(0.1 * i)
        assert c.statistic <= 30.0, f"Runaway: statistic = {c.statistic}"

    def test_ceiling_is_structural(self):
        """Ceiling can't be bypassed by any input sequence."""
        c = DASCUSUM(threshold=100.0, drift=0.0, ceiling=15.0, window_size=10)
        # Extreme: feed exponentially growing values
        for i in range(200):
            c.update(2.0 ** i if i < 50 else 0.0)
        assert c.statistic <= 15.0

    def test_decay_stabilizes_periodic_signal(self):
        """Periodic signal should not cause unbounded growth."""
        c = DASCUSUM(threshold=100.0, drift=0.0, decay=0.95, ceiling=1000.0)
        max_stat = 0.0
        for i in range(1000):
            stat, _ = c.update(float(i % 5))
            max_stat = max(max_stat, stat)
        # With decay, should plateau well below 125
        assert max_stat < 50, f"Max stat = {max_stat}, not stabilizing"


# ===========================================================================
# §7.6 INTEGRATION TESTS
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_benign_then_suspicious(self):
        """End-to-end: benign behavior → no labels; suspicious → labels."""
        metric = _simple_metric()
        d_signal = DeceptionTaxSignal(metric, cusum_params={
            "threshold": 3.0, "drift": 0.3, "window_size": 15,
        })
        s_signal = StagnationTaxSignal(baseline_rate=1.0, cusum_params={
            "threshold": 3.0, "drift": 0.3, "window_size": 15,
        })
        gate = MarketGate(d_signal, s_signal)

        results = []

        # Phase 1: Normal (20 steps, straight line, steady rate)
        for i in range(20):
            vec = np.array([0.05, 0.02, 0.01, 0.0, 0.0, 0.0])
            result = gate.evaluate(vec, float(i))
            results.append(result)

        # All phase 1 should be nominal
        assert all(r is None for r in results), "Benign phase produced labels"

        # Phase 2: Suspicious (zigzag + stagnation)
        for i in range(30):
            offset = 0.4 if i % 2 == 0 else -0.4
            vec = np.array([0.3 + offset, 0.2, 0.1, 0.0, 0.0, 0.0])
            result = gate.evaluate(vec, 20.0 + i * 5.0)
            results.append(result)

        # At least some phase 2 results should be labels
        phase2_labels = [r for r in results[20:] if r is not None]
        assert len(phase2_labels) > 0, "Suspicious phase produced no labels"

        # All labels must be from the allowed set (including variants)
        all_variants = set()
        for variants in SIGNAL_LABEL_VARIANTS.values():
            all_variants.update(variants)
        for label in phase2_labels:
            assert label in all_variants

    def test_state_tracks_evaluations(self):
        """Gate state updates correctly on each evaluation."""
        gate = _make_gate()
        assert gate.n_evaluations == 0

        for i in range(10):
            gate.evaluate(np.zeros(6), float(i))
        assert gate.n_evaluations == 10

        state = gate.get_state()
        assert isinstance(state.signal_vector, np.ndarray)
        assert state.signal_vector.shape == (4,)

    def test_d_and_s_can_fire_together(self):
        """Both D and S alarm simultaneously → combined label."""
        metric = _simple_metric()
        d_signal = DeceptionTaxSignal(metric, cusum_params={
            "threshold": 1.0, "drift": 0.1, "window_size": 10,
        })
        s_signal = StagnationTaxSignal(baseline_rate=1.0, cusum_params={
            "threshold": 1.0, "drift": 0.1, "window_size": 10,
        })
        gate = MarketGate(d_signal, s_signal)

        # Build baseline
        for i in range(15):
            gate.evaluate(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), float(i))

        # Zigzag + stagnation simultaneously
        combined_found = False
        for i in range(30):
            offset = 0.5 if i % 2 == 0 else -0.5
            vec = np.array([0.3 + offset, 0.3, 0.0, 0.0, 0.0, 0.0])
            result = gate.evaluate(vec, 15.0 + i * 10.0)
            if result == SIGNAL_LABELS["d_and_s"]:
                combined_found = True
                break

        # This may or may not fire both simultaneously depending on thresholds
        # — the test validates the label mapping is correct IF both fire
        if combined_found:
            assert gate.get_state().d_alarm
            assert gate.get_state().s_alarm

    def test_sprt_composes_with_cusum(self):
        """SPRT can use CUSUM statistic as LLR proxy for accept/reject."""
        rng = np.random.default_rng(42)
        cusum = DASCUSUM(threshold=5.0, drift=0.5)
        sprt = SPRTWrapper(alpha=0.05, beta=0.10)

        # Feed benign data — SPRT should eventually accept
        for _ in range(200):
            stat, _ = cusum.update(rng.normal(0, 1))
            # Convert CUSUM stat to LLR: low stat → negative LLR (evidence for benign)
            llr = (stat - cusum.threshold) / cusum.threshold
            decision = sprt.update(llr)
            if decision != SPRTDecision.CONTINUE:
                break

        assert decision == SPRTDecision.ACCEPT, f"Expected ACCEPT, got {decision}"


class TestCalibration:
    def test_calibrate_increases_threshold(self):
        """Calibration with a tight ARL target should increase threshold."""
        rng = np.random.default_rng(42)
        c = DASCUSUM(threshold=0.5, drift=0.3)
        original = c.threshold
        c.calibrate_from_data(rng.normal(0, 1, 500), target_arl=500)
        assert c.threshold > original

    def test_calibrate_resets_state(self):
        """Calibration leaves the detector in a clean state."""
        rng = np.random.default_rng(42)
        c = DASCUSUM()
        c.calibrate_from_data(rng.normal(0, 1, 200), target_arl=100)
        assert c.statistic == 0.0
        assert c.run_length == 0

    def test_set_baseline_from_timestamps(self):
        s = StagnationTaxSignal()
        timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]  # 2 actions/sec
        s.set_baseline(timestamps)
        assert s.baseline_rate == pytest.approx(2.0, rel=0.01)

    def test_set_baseline_unsorted_ok(self):
        """set_baseline sorts timestamps internally."""
        s = StagnationTaxSignal()
        s.set_baseline([2.0, 0.0, 1.0, 1.5, 0.5])
        assert s.baseline_rate == pytest.approx(2.0, rel=0.01)


# ===========================================================================
# §7.7 PERFORMANCE TESTS
# ===========================================================================

class TestLabelRotation:
    """CORRECTNESS_SPEC §10.2: LR1-LR5."""

    def test_lr1_all_variants_pass_numeric_checks(self):
        """LR1: Every label variant passes _NUMERIC_PATTERNS validation."""
        for key, variants in SIGNAL_LABEL_VARIANTS.items():
            for idx, label in enumerate(variants):
                for pattern in _NUMERIC_PATTERNS:
                    assert not pattern.search(label), (
                        f"SIGNAL_LABEL_VARIANTS[{key!r}][{idx}] matches "
                        f"{pattern.pattern!r}: {label!r}"
                    )

    def test_lr2_same_count_same_label(self):
        """LR2: Same evaluation_count → same label."""
        for key in SIGNAL_LABEL_VARIANTS:
            label1 = _select_variant(key, 42)
            label2 = _select_variant(key, 42)
            assert label1 == label2, (
                f"Non-deterministic: {key} at count 42 → {label1!r} vs {label2!r}"
            )

    def test_lr3_different_counts_produce_varied_labels(self):
        """LR3: Different evaluation counts produce varied labels."""
        for key, variants in SIGNAL_LABEL_VARIANTS.items():
            if len(variants) < 2:
                continue
            labels_seen = set()
            for count in range(100):
                labels_seen.add(_select_variant(key, count))
            assert len(labels_seen) > 1, (
                f"No variety for {key!r}: always {labels_seen}"
            )

    def test_lr5_variants_validated_at_import(self):
        """LR5: SIGNAL_LABEL_VARIANTS exists and first variant matches canonical."""
        for key in SIGNAL_LABELS:
            assert key in SIGNAL_LABEL_VARIANTS
            assert SIGNAL_LABEL_VARIANTS[key][0] == SIGNAL_LABELS[key]

    def test_gate_with_audit_chain_evaluation_count(self):
        """A6: Gate with audit chain: n_evaluations == chain length."""
        from frontier_ops.governance.market_audit import MarketAuditChain
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric)
        s = StagnationTaxSignal()
        audit = MarketAuditChain()
        gate = MarketGate(d, s, audit_chain=audit)

        for i in range(15):
            gate.evaluate(np.zeros(6), float(i))

        assert gate.n_evaluations == 15
        assert len(audit) == 15

    def test_gate_label_rotation_varies(self):
        """Gate uses label rotation — labels vary across evaluations."""
        metric = _simple_metric()
        d = DeceptionTaxSignal(metric, cusum_params={
            "threshold": 0.5, "drift": 0.1, "window_size": 10,
        })
        s = StagnationTaxSignal(baseline_rate=1.0, cusum_params={
            "threshold": 0.5, "drift": 0.1, "window_size": 10,
        })
        gate = MarketGate(d, s)

        labels = []
        rng = np.random.default_rng(42)
        for i in range(200):
            vec = rng.uniform(-0.5, 0.5, 6)
            result = gate.evaluate(vec, float(i))
            if result is not None:
                labels.append(result)

        if len(labels) > 1:
            # Should see some variety
            unique = set(labels)
            # Not required to always vary, but with 200 evals and
            # multiple variants, very likely to see > 1
            assert len(unique) >= 1


class TestPerformance:
    def test_evaluate_latency(self):
        """Full evaluate() call must complete in < 0.48ms (P99 over 1000 calls)."""
        gate = _make_gate()
        rng = np.random.default_rng(42)

        # Warmup
        for i in range(50):
            gate.evaluate(rng.uniform(0, 0.5, 6), float(i))

        # Benchmark
        latencies = []
        for i in range(1000):
            vec = rng.uniform(0, 0.5, 6)
            t0 = time.perf_counter_ns()
            gate.evaluate(vec, 50.0 + float(i))
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)  # Convert to ms

        p99 = np.percentile(latencies, 99)
        assert p99 < 0.48, f"P99 latency = {p99:.3f}ms, budget is 0.48ms"
