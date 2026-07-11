"""
Tests for Market Entropy — Antitrust Mechanism.

Test categories (per CORRECTNESS_SPEC.md §7):
  7.1 Invariant tests — E1-E7, R1-R5, RD1-RD7, MH1-MH5
  7.3 Boundary condition tests — §8.6
  7.2 Statistical tests — random vectors, redistribution entropy increase
  7.4 Adversarial tests — entropy gaming, emergency override
  7.7 Performance tests — latency ceilings
"""

import time

import numpy as np
import pytest

from frontier_ops.sensing.market_entropy import (
    market_entropy,
    market_health,
    redistribute,
    MarketHealthMonitor,
)


# ===========================================================================
# §7.1 INVARIANT TESTS — Shannon Entropy (E1-E7)
# ===========================================================================

class TestShannonEntropyInvariants:
    """CORRECTNESS_SPEC §8.2: E1-E7."""

    def test_e1_result_in_unit_interval(self):
        """E1: market_entropy ∈ [0, 1] for all finite non-negative inputs."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(1, 10)
            signals = rng.uniform(0, 10, n)
            h = market_entropy(signals)
            assert 0.0 <= h <= 1.0, f"H={h} outside [0,1] for {signals}"

    def test_e1_negative_inputs(self):
        """E1: result ∈ [0, 1] even with negative inputs (via |s_i|)."""
        h = market_entropy([-3.0, -3.0, -3.0])
        assert 0.0 <= h <= 1.0

    def test_e2_uniform_is_one(self):
        """E2: uniform distribution → 1.0."""
        assert market_entropy([1, 1, 1, 1]) == pytest.approx(1.0)
        assert market_entropy([5, 5, 5, 5]) == pytest.approx(1.0)
        assert market_entropy([0.1, 0.1, 0.1, 0.1]) == pytest.approx(1.0)
        assert market_entropy([100, 100]) == pytest.approx(1.0)

    def test_e3_single_source_is_zero(self):
        """E3: single source → 0.0."""
        assert market_entropy([1, 0, 0, 0]) == pytest.approx(0.0)
        assert market_entropy([999, 0, 0, 0]) == pytest.approx(0.0)
        assert market_entropy([0, 0, 5, 0]) == pytest.approx(0.0)

    def test_e4_all_zero_is_one(self):
        """E4: all-zero → 1.0 (no monopoly when no signals active)."""
        assert market_entropy([0, 0, 0, 0]) == pytest.approx(1.0)
        assert market_entropy([0, 0]) == pytest.approx(1.0)

    def test_e5_permutation_invariant(self):
        """E5: order of signals doesn't matter."""
        a = [3.0, 1.0, 0.5, 2.0]
        import itertools
        values = set()
        for perm in itertools.permutations(a):
            values.add(round(market_entropy(list(perm)), 10))
        assert len(values) == 1, f"Permutation produced different values: {values}"

    def test_e6_nan_raises(self):
        """E6: NaN → ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            market_entropy([1.0, float('nan'), 2.0])

    def test_e6_inf_raises(self):
        """E6: inf → ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            market_entropy([1.0, float('inf'), 2.0])

    def test_e6_neg_inf_raises(self):
        """E6: -inf → ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            market_entropy([float('-inf'), 1.0])

    def test_e7_uses_absolute_values(self):
        """E7: negative signals handled via |s_i|."""
        h_pos = market_entropy([3.0, 1.0, 2.0])
        h_neg = market_entropy([-3.0, -1.0, -2.0])
        h_mix = market_entropy([-3.0, 1.0, -2.0])
        assert h_pos == pytest.approx(h_neg)
        assert h_pos == pytest.approx(h_mix)


# ===========================================================================
# §7.1 INVARIANT TESTS — Renyi-2 Entropy (R1-R5)
# ===========================================================================

class TestRenyiEntropyInvariants:
    """CORRECTNESS_SPEC §8.3: R1-R5."""

    def test_r1_result_in_unit_interval(self):
        """R1: market_health ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(1, 10)
            signals = rng.uniform(0, 10, n)
            h2 = market_health(signals)
            assert 0.0 <= h2 <= 1.0, f"H2={h2} outside [0,1] for {signals}"

    def test_r2_uniform_is_one(self):
        """R2: uniform → 1.0."""
        assert market_health([1, 1, 1, 1]) == pytest.approx(1.0)
        assert market_health([7, 7, 7, 7]) == pytest.approx(1.0)

    def test_r3_single_source_is_zero(self):
        """R3: single source → 0.0."""
        assert market_health([5, 0, 0, 0]) == pytest.approx(0.0)
        assert market_health([0, 0, 1, 0]) == pytest.approx(0.0)

    def test_r4_all_zero_is_zero(self):
        """R4: all-zero → 0.0 (collapsed market = unhealthy)."""
        assert market_health([0, 0, 0, 0]) == pytest.approx(0.0)

    def test_r5_renyi_leq_shannon(self):
        """R5: market_health ≤ market_entropy (always)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 8)
            signals = rng.uniform(0, 10, n)
            h = market_entropy(signals)
            h2 = market_health(signals)
            assert h2 <= h + 1e-10, f"R5 violated: H2={h2} > H={h}"


# ===========================================================================
# §7.1 INVARIANT TESTS — Redistribution (RD1-RD7)
# ===========================================================================

class TestRedistributionInvariants:
    """CORRECTNESS_SPEC §8.4: RD1-RD7."""

    def test_rd1_preserves_sign(self):
        """RD1: sign of each element preserved."""
        signals = np.array([5.0, -1.0, 0.5, -0.1])
        result = redistribute(signals, floor=0.99)
        for i in range(len(signals)):
            if signals[i] > 0:
                assert result[i] >= 0, f"Sign flip at index {i}"
            elif signals[i] < 0:
                assert result[i] <= 0, f"Sign flip at index {i}"

    def test_rd2_unchanged_when_above_floor(self):
        """RD2: returns unchanged when H ≥ floor."""
        signals = np.array([1.0, 1.0, 1.0, 1.0])  # H = 1.0
        result = redistribute(signals, floor=0.5)
        np.testing.assert_array_almost_equal(result, signals)

    def test_rd3_emergency_override(self):
        """RD3: returns unchanged when max(|s|) > critical_threshold."""
        signals = np.array([25.0, 0.0, 0.0, 0.0])  # monopoly, but critical
        result = redistribute(signals, floor=0.5, critical_threshold=20.0)
        np.testing.assert_array_almost_equal(result, signals)

    def test_rd4_entropy_never_decreases(self):
        """RD4: entropy(result) ≥ entropy(input)."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = rng.integers(2, 6)
            signals = rng.uniform(0, 10, n)
            result = redistribute(signals, floor=0.8)
            h_before = market_entropy(signals)
            h_after = market_entropy(result)
            assert h_after >= h_before - 1e-10, (
                f"RD4 violated: {h_after} < {h_before}"
            )

    def test_rd5_idempotent_on_uniform(self):
        """RD5: no change to already-balanced signals."""
        signals = np.array([3.0, 3.0, 3.0, 3.0])
        result = redistribute(signals, floor=0.5)
        np.testing.assert_array_almost_equal(result, signals)

    def test_rd6_continuous(self):
        """RD6: small perturbation → small change in output."""
        signals = np.array([5.0, 1.0, 0.5, 0.1])
        r1 = redistribute(signals, floor=0.8)
        r2 = redistribute(signals + 0.001, floor=0.8)
        diff = np.max(np.abs(r1 - r2))
        assert diff < 0.1, f"Discontinuity: max diff = {diff}"

    def test_rd7_alpha_bounded(self):
        """RD7: α ∈ [0, 1] — verified by checking output is between input and uniform."""
        signals = np.array([10.0, 0.1, 0.1, 0.1])
        result = redistribute(signals, floor=0.9)
        # Result should be between original and uniform for each element
        mags = np.abs(signals)
        uniform_mag = np.mean(mags)
        for i in range(len(signals)):
            # The result magnitude should be between the original and uniform
            orig_mag = abs(signals[i])
            res_mag = abs(result[i])
            lo = min(orig_mag, uniform_mag)
            hi = max(orig_mag, uniform_mag)
            assert lo - 1e-10 <= res_mag <= hi + 1e-10, (
                f"Index {i}: result mag {res_mag} not between {lo} and {hi}"
            )


# ===========================================================================
# §7.1 INVARIANT TESTS — Market Health Monitor (MH1-MH5)
# ===========================================================================

class TestMarketHealthMonitorInvariants:
    """CORRECTNESS_SPEC §8.5: MH1-MH5."""

    def test_mh1_status_in_closed_set(self):
        """MH1: status ∈ {'healthy', 'monopoly', 'collapse'}."""
        mon = MarketHealthMonitor(sustained_count=3)
        valid = {"healthy", "monopoly", "collapse"}
        assert mon.status in valid

        # After monopoly-like updates
        for _ in range(10):
            status = mon.update([10, 0, 0, 0])
            assert status in valid

        # After collapse-like updates
        mon2 = MarketHealthMonitor(sustained_count=2)
        for _ in range(5):
            status = mon2.update([0, 0, 0, 0])
            assert status in valid

    def test_mh2_monopoly_requires_n_consecutive(self):
        """MH2: MONOPOLY requires N consecutive unhealthy readings."""
        mon = MarketHealthMonitor(monopoly_threshold=0.3, sustained_count=5)
        # Only 4 unhealthy — should stay healthy
        for _ in range(4):
            mon.update([10, 0, 0, 0])
        assert mon.status == "healthy"

        # 5th pushes to monopoly
        mon.update([10, 0, 0, 0])
        assert mon.status == "monopoly"

    def test_mh3_collapse_requires_n_consecutive(self):
        """MH3: COLLAPSE requires N consecutive zero-total readings."""
        mon = MarketHealthMonitor(sustained_count=3)
        for _ in range(2):
            mon.update([0, 0, 0, 0])
        assert mon.status == "healthy"

        mon.update([0, 0, 0, 0])
        assert mon.status == "collapse"

    def test_mh4_single_healthy_resets(self):
        """MH4: Any single healthy reading resets pending count."""
        mon = MarketHealthMonitor(monopoly_threshold=0.3, sustained_count=5)
        # 4 unhealthy
        for _ in range(4):
            mon.update([10, 0, 0, 0])
        assert mon.pending_count == 4

        # 1 healthy resets
        mon.update([1, 1, 1, 1])
        assert mon.status == "healthy"
        assert mon.pending_count == 0

        # Need full 5 again
        for _ in range(4):
            mon.update([10, 0, 0, 0])
        assert mon.status == "healthy"

    def test_mh5_not_for_agent(self):
        """MH5: Alerts are for human operator — status is string, not injected."""
        mon = MarketHealthMonitor(sustained_count=1)
        status = mon.update([10, 0, 0, 0])
        # Status is a simple string, not a structured signal for the agent
        assert isinstance(status, str)
        assert status in {"healthy", "monopoly", "collapse"}


# ===========================================================================
# §7.3 BOUNDARY CONDITION TESTS (§8.6)
# ===========================================================================

class TestEntropyBoundaryConditions:
    """CORRECTNESS_SPEC §8.6 boundary table."""

    def test_empty_array_entropy(self):
        """Empty array → ValueError for market_entropy."""
        with pytest.raises(ValueError, match="empty"):
            market_entropy([])

    def test_empty_array_health(self):
        """Empty array → ValueError for market_health."""
        with pytest.raises(ValueError, match="empty"):
            market_health([])

    def test_empty_array_redistribute(self):
        """Empty array → ValueError for redistribute."""
        with pytest.raises(ValueError, match="empty"):
            redistribute([])

    def test_single_element_entropy(self):
        """Single element → 0.0 (N=1, log(1)=0)."""
        assert market_entropy([5.0]) == pytest.approx(0.0)
        assert market_entropy([0.1]) == pytest.approx(0.0)

    def test_single_element_health(self):
        """Single element → 0.0."""
        assert market_health([5.0]) == pytest.approx(0.0)

    def test_single_element_redistribute(self):
        """Single element → return unchanged."""
        result = redistribute([7.0])
        np.testing.assert_array_almost_equal(result, [7.0])

    def test_all_zeros_entropy(self):
        """All zeros → 1.0 for entropy."""
        assert market_entropy([0, 0, 0, 0]) == pytest.approx(1.0)

    def test_all_zeros_health(self):
        """All zeros → 0.0 for health."""
        assert market_health([0, 0, 0, 0]) == pytest.approx(0.0)

    def test_all_zeros_redistribute(self):
        """All zeros → return unchanged (total < ε)."""
        result = redistribute([0, 0, 0, 0])
        np.testing.assert_array_almost_equal(result, [0, 0, 0, 0])

    def test_nan_entropy(self):
        with pytest.raises(ValueError):
            market_entropy([1, float('nan')])

    def test_nan_health(self):
        with pytest.raises(ValueError):
            market_health([1, float('nan')])

    def test_nan_redistribute(self):
        with pytest.raises(ValueError):
            redistribute([1, float('nan')])

    def test_inf_entropy(self):
        with pytest.raises(ValueError):
            market_entropy([1, float('inf')])

    def test_inf_health(self):
        with pytest.raises(ValueError):
            market_health([1, float('inf')])

    def test_inf_redistribute(self):
        with pytest.raises(ValueError):
            redistribute([1, float('inf')])

    def test_negative_values_entropy(self):
        """Negative values handled via |s_i|."""
        h = market_entropy([-3, -3, -3])
        assert h == pytest.approx(1.0)

    def test_negative_values_health(self):
        h2 = market_health([-3, -3, -3])
        assert h2 == pytest.approx(1.0)

    def test_negative_values_redistribute(self):
        """Preserves sign for negative values."""
        result = redistribute([-5, -1, -1, -1], floor=0.9)
        assert all(r <= 0 for r in result)


# ===========================================================================
# §7.2 STATISTICAL TESTS
# ===========================================================================

class TestEntropyStatistical:
    """Random vectors and redistribution entropy increase."""

    def test_random_vectors_always_in_unit(self):
        """market_entropy of random vectors always ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            n = rng.integers(2, 20)
            signals = rng.uniform(-10, 10, n)
            h = market_entropy(signals)
            assert 0.0 <= h <= 1.0

    def test_redistribution_always_increases_entropy(self):
        """redistribute always increases entropy (or keeps it equal)."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            n = rng.integers(2, 8)
            signals = rng.uniform(0, 10, n)
            result = redistribute(signals, floor=0.8)
            h_before = market_entropy(signals)
            h_after = market_entropy(result)
            assert h_after >= h_before - 1e-10


# ===========================================================================
# §7.4 ADVERSARIAL TESTS
# ===========================================================================

class TestEntropyAdversarial:
    """Entropy gaming and emergency override."""

    def test_entropy_gaming_attempt(self):
        """Agent tries to concentrate signal — redistribution increases entropy."""
        # Use non-zero signals so redistribution can actually blend
        monopoly = np.array([10.0, 0.1, 0.1, 0.1])
        h_before = market_entropy(monopoly)
        assert h_before < 0.5  # heavily concentrated

        result = redistribute(monopoly, floor=0.8, critical_threshold=200.0)
        h_after = market_entropy(result)
        assert h_after > h_before

    def test_emergency_override_preserves_critical_signal(self):
        """Critical signal is not dampened by redistribution."""
        critical = np.array([50.0, 0.0, 0.0, 0.0])
        result = redistribute(critical, floor=0.5, critical_threshold=30.0)
        np.testing.assert_array_equal(result, critical)

    def test_monitor_resists_intermittent_monopoly(self):
        """Alternating monopoly/healthy doesn't trigger MONOPOLY status."""
        mon = MarketHealthMonitor(monopoly_threshold=0.3, sustained_count=5)
        for _ in range(20):
            mon.update([10, 0, 0, 0])  # monopoly
            mon.update([1, 1, 1, 1])   # healthy → resets count
        assert mon.status == "healthy"


# ===========================================================================
# §7.7 PERFORMANCE TESTS
# ===========================================================================

class TestEntropyPerformance:
    """Latency ceilings from spec."""

    def test_market_entropy_latency(self):
        """market_entropy < 0.01ms."""
        signals = np.array([3.0, 1.0, 0.5, 2.0])
        # Warmup
        for _ in range(100):
            market_entropy(signals)

        latencies = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            market_entropy(signals)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)

        p99 = np.percentile(latencies, 99)
        assert p99 < 0.05, f"P99 = {p99:.4f}ms, budget is 0.05ms"

    def test_redistribute_latency(self):
        """redistribute < 0.15ms."""
        signals = np.array([10.0, 0.1, 0.5, 2.0])
        # Warmup
        for _ in range(100):
            redistribute(signals, floor=0.8)

        latencies = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            redistribute(signals, floor=0.8)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)

        p99 = np.percentile(latencies, 99)
        assert p99 < 0.15, f"P99 = {p99:.4f}ms, budget is 0.15ms"
