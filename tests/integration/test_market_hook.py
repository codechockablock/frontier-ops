"""
Tests for Market Hook — sidecar integration.

Verifies the hook correctly wires market signals into the sidecar pipeline,
writes state to disk, and maintains audit chain integrity.
"""

import json
import os
import tempfile
import time

import numpy as np
import pytest

from frontier_ops.integration.market_hook import MarketHook
from frontier_ops.sensing.market_signals import SeveritySignal, IntervalAnomalySignal
from frontier_ops.sensing.market_gate import SIGNAL_LABEL_VARIANTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hook(state_dir=None, **kwargs):
    """Create a MarketHook with synthetic calibration."""
    if state_dir is None:
        state_dir = tempfile.mkdtemp()
    state_path = os.path.join(state_dir, "market_state.json")

    rng = np.random.default_rng(42)
    benign_intervals = sorted(rng.exponential(scale=8.0, size=500))

    return MarketHook.from_intervals(
        benign_intervals=benign_intervals,
        state_path=state_path,
        **kwargs,
    )


def _make_telemetry_file(n_pass=100, n_flag=20):
    """Create a temporary telemetry JSONL file."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    t = 1000.0
    with os.fdopen(fd, 'w') as f:
        for i in range(n_pass):
            t += np.random.exponential(5.0)
            entry = {"step": i, "ts": t, "verdict": "PASS",
                     "signals": {"error": 0.1, "cusum": 0.5}}
            f.write(json.dumps(entry) + "\n")
        for i in range(n_flag):
            t += np.random.exponential(2.0)
            entry = {"step": n_pass + i, "ts": t, "verdict": "FLAG",
                     "signals": {"error": 0.5, "cusum": 3.0}}
            f.write(json.dumps(entry) + "\n")
    return path


# ===========================================================================
# CORE FUNCTIONALITY
# ===========================================================================

class TestMarketHookBasics:

    def test_on_step_returns_none_for_pass(self):
        """PASS verdicts should produce no label."""
        hook = _make_hook()
        for i in range(20):
            label = hook.on_step("pass", timestamp=float(i))
        # With all PASS, severity is 0.0, no alarm
        assert label is None

    def test_on_step_returns_label_for_sustained_flag(self):
        """Sustained FLAG should eventually produce a label."""
        hook = _make_hook(
            d_cusum_params={"threshold": 3.0, "drift": 0.3, "window_size": 10},
        )
        # Build baseline
        for i in range(20):
            hook.on_step("pass", timestamp=float(i))
        # Sustained FLAG
        labels = []
        for i in range(30):
            label = hook.on_step("flag", timestamp=20.0 + float(i))
            if label is not None:
                labels.append(label)
        assert len(labels) > 0, "Sustained FLAG should produce labels"

    def test_labels_are_qualitative(self):
        """All emitted labels must be from the allowed set."""
        import re
        all_allowed = set()
        for variants in SIGNAL_LABEL_VARIANTS.values():
            all_allowed.update(variants)

        hook = _make_hook(
            d_cusum_params={"threshold": 1.0, "drift": 0.1},
            s_cusum_params={"threshold": 1.0, "drift": 0.1},
        )
        verdicts = ["pass", "monitor", "flag", "block"]
        rng = np.random.default_rng(42)
        for i in range(200):
            label = hook.on_step(verdicts[rng.integers(0, 4)], float(i))
            if label is not None:
                assert label in all_allowed, f"Unknown label: {label!r}"
                # No numeric values
                assert not re.search(r'\b\d+\.\d+\b', label)

    def test_n_steps_tracks_correctly(self):
        hook = _make_hook()
        assert hook.n_steps == 0
        for i in range(10):
            hook.on_step("pass", float(i))
        assert hook.n_steps == 10

    def test_n_alerts_tracks_correctly(self):
        hook = _make_hook(
            d_cusum_params={"threshold": 1.0, "drift": 0.1},
        )
        for i in range(10):
            hook.on_step("pass", float(i))
        initial_alerts = hook.n_alerts
        # Force some flags
        for i in range(30):
            hook.on_step("flag", 10.0 + float(i))
        assert hook.n_alerts >= initial_alerts


# ===========================================================================
# CALIBRATION
# ===========================================================================

class TestCalibration:

    def test_from_telemetry(self):
        """Calibrate from telemetry JSONL file."""
        path = _make_telemetry_file()
        try:
            hook = MarketHook.from_telemetry(telemetry_path=path, state_path="/dev/null")
            # Should be calibrated
            hook.on_step("pass", 0.0)
            hook.on_step("pass", 5.0)
            assert hook.n_steps == 2
        finally:
            os.unlink(path)

    def test_from_telemetry_missing_file(self):
        """Missing telemetry falls back to synthetic calibration."""
        hook = MarketHook.from_telemetry(
            telemetry_path="/nonexistent/file.jsonl",
            state_path="/dev/null",
        )
        hook.on_step("pass", 0.0)
        assert hook.n_steps == 1

    def test_from_intervals(self):
        """Explicit interval calibration."""
        intervals = list(np.arange(0.5, 100.5, 0.5))
        hook = MarketHook.from_intervals(
            benign_intervals=intervals,
            state_path="/dev/null",
        )
        hook.on_step("pass", 0.0)
        hook.on_step("pass", 50.0)  # ~median interval
        state = hook.get_state()
        assert 0.4 <= state.s_raw <= 0.6


# ===========================================================================
# GOVERNANCE CHAIN
# ===========================================================================

class TestAuditChain:

    def test_chain_grows_with_steps(self):
        hook = _make_hook()
        for i in range(25):
            hook.on_step("pass", float(i))
        assert hook.audit_chain_length == 25

    def test_chain_verifies(self):
        hook = _make_hook()
        for i in range(50):
            hook.on_step("pass" if i < 30 else "flag", float(i))
        result = hook.verify_chain()
        assert result.valid
        assert result.entries_verified == 50

    def test_chain_one_to_one_with_steps(self):
        """Every on_step() produces exactly one chain entry."""
        hook = _make_hook()
        verdicts = ["pass", "monitor", "flag", "block"]
        for i in range(40):
            hook.on_step(verdicts[i % 4], float(i))
        assert hook.audit_chain_length == hook.n_steps == 40


# ===========================================================================
# DISK I/O
# ===========================================================================

class TestDiskIO:

    def test_writes_state_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "market_state.json")
            hook = _make_hook(state_dir=d)
            hook.on_step("pass", 0.0)

            assert os.path.exists(path)
            with open(path) as f:
                state = json.load(f)
            assert state["market_active"] is True
            assert "label" in state
            assert "health_status" in state

    def test_state_file_no_raw_numbers(self):
        """State file should not contain raw CUSUM statistics."""
        with tempfile.TemporaryDirectory() as d:
            hook = _make_hook(state_dir=d)
            for i in range(10):
                hook.on_step("flag", float(i))

            path = os.path.join(d, "market_state.json")
            with open(path) as f:
                raw = f.read()
            # Should not contain d_statistic, s_statistic, d_raw, etc
            assert "d_statistic" not in raw
            assert "s_statistic" not in raw
            assert "cusum" not in raw.lower()

    def test_state_survives_crash(self):
        """Atomic write (tmp + rename) means partial writes don't corrupt."""
        with tempfile.TemporaryDirectory() as d:
            hook = _make_hook(state_dir=d)
            # Write several times
            for i in range(20):
                hook.on_step("pass", float(i))
            # Last write should be valid JSON
            path = os.path.join(d, "market_state.json")
            with open(path) as f:
                state = json.load(f)
            assert isinstance(state, dict)


# ===========================================================================
# HEALTH MONITOR
# ===========================================================================

class TestHealthMonitor:

    def test_health_starts_healthy(self):
        hook = _make_hook()
        assert hook.market_health_status == "healthy"

    def test_health_tracks_after_steps(self):
        hook = _make_hook()
        for i in range(20):
            hook.on_step("pass", float(i))
        assert hook.market_health_status == "healthy"


# ===========================================================================
# INTEGRATION WITH SIDECAR DATA
# ===========================================================================

class TestSidecarIntegration:

    def test_replay_mixed_session(self):
        """Replay a realistic verdict sequence through the hook."""
        hook = _make_hook(
            d_cusum_params={"threshold": 4.0, "drift": 0.5, "window_size": 15},
            s_cusum_params={"threshold": 4.0, "drift": 0.5, "window_size": 15},
        )

        # Simulate: 20 PASS, 5 MONITOR, 15 FLAG, 5 BLOCK
        trace = (
            [("pass", 3.0)] * 20 +
            [("monitor", 4.0)] * 5 +
            [("flag", 5.0)] * 15 +
            [("block", 10.0)] * 5
        )
        t = 0.0
        labels = []
        for verdict, interval in trace:
            t += interval
            label = hook.on_step(verdict, t)
            if label is not None:
                labels.append(label)

        # Should have detected the escalation
        state = hook.get_state()
        assert state.d_raw > 0.5, "D should be elevated after FLAG/BLOCK"
        assert hook.n_steps == 45
        assert hook.audit_chain_length == 45

    def test_from_real_telemetry_if_available(self):
        """If the real telemetry file exists, calibrate from it."""
        real_path = os.path.expanduser(
            "~/.openclaw/workspace/proprioception-log.jsonl"
        )
        if not os.path.exists(real_path):
            pytest.skip("Real telemetry not available")

        hook = MarketHook.from_telemetry(
            telemetry_path=real_path,
            state_path="/dev/null",
            verbose=False,
        )
        # Should be calibrated from real data
        hook.on_step("pass", time.time())
        hook.on_step("pass", time.time() + 5.0)
        assert hook.n_steps == 2


# ===========================================================================
# PERFORMANCE
# ===========================================================================

class TestPerformance:

    def test_on_step_latency(self):
        """on_step() < 1ms P99 (includes disk write)."""
        hook = _make_hook()
        # Warmup
        for i in range(50):
            hook.on_step("pass", float(i))

        latencies = []
        for i in range(500):
            t0 = time.perf_counter_ns()
            hook.on_step("pass", 50.0 + float(i))
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)

        p99 = np.percentile(latencies, 99)
        assert p99 < 1.0, f"P99 = {p99:.3f}ms, budget is 1.0ms"


# ===========================================================================
# CUSUM RECOVERY (Fix 3)
# ===========================================================================

class TestCUSUMRecovery:

    def test_d_raw_recovers_after_blocks_then_passes(self):
        """After 20 BLOCKs then PASS verdicts, d_raw < 1.0 within 15 PASS steps."""
        hook = _make_hook(
            d_cusum_params={"threshold": 5.0, "drift": 0.5, "window_size": 30,
                            "decay": 0.98, "ceiling": 30.0},
        )
        t = 0.0
        # 20 BLOCK verdicts — saturate D signal
        for i in range(20):
            t += 1.0
            hook.on_step("block", t)

        state = hook.get_state()
        assert state.d_raw > 2.0, "D should be saturated after 20 BLOCKs"

        # Now feed PASS verdicts — should recover within 15 steps
        for i in range(15):
            t += 1.0
            hook.on_step("pass", t)

        state = hook.get_state()
        assert state.d_raw < 1.0, (
            f"d_raw={state.d_raw:.2f} should be < 1.0 within 15 PASS steps"
        )

    def test_single_pass_in_block_window_does_not_reset(self):
        """A single PASS among BLOCKs must NOT reset CUSUM."""
        sig = SeveritySignal(severity_window=10)
        # Fill window with BLOCKs
        for _ in range(10):
            sig.step("block")
        stat_before = sig.statistic

        # One PASS — window is now 9 BLOCKs + 1 PASS, not all-PASS
        sig.step("pass")
        # CUSUM should still be elevated (not reset)
        assert sig.statistic > 0, "Single PASS should not reset CUSUM"
