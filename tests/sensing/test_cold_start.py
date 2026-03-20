"""
Tests for cold-start detection and suppression.

Validates that:
1. ColdStartDetector identifies cold-start patterns (high BLOCK rate + initializing HMM)
2. Suppression activates after threshold is met
3. Recovery fires when BLOCK rate and initializing rate drop
4. MarketGate returns None during cold-start suppression
5. Replays real observation data to verify detection on actual BLOCK storm
"""

import json
import os
import time

import pytest

from frontier_ops.sensing.cold_start import ColdStartDetector, ColdStartEvent
from frontier_ops.sensing.market_gate import MarketGate
from frontier_ops.sensing.market_signals import SeveritySignal, IntervalAnomalySignal


# ---------------------------------------------------------------------------
# ColdStartDetector unit tests
# ---------------------------------------------------------------------------


class TestColdStartDetector:
    def test_no_suppression_on_normal_traffic(self):
        """Normal pass/flag verdicts should never trigger cold-start."""
        det = ColdStartDetector(window_size=10, min_steps_before_detection=3)
        for i in range(50):
            event = det.observe("pass", "exploring", 0.1, timestamp=float(i))
            assert event is None
        assert not det.is_suppressing

    def test_detect_cold_start_block_storm(self):
        """100% BLOCK + initializing should trigger cold-start."""
        det = ColdStartDetector(
            window_size=10,
            block_rate_threshold=0.80,
            min_steps_before_detection=5,
        )
        events = []
        for i in range(20):
            event = det.observe("block", "initializing", 1e10 * (i + 1), timestamp=float(i))
            if event is not None:
                events.append(event)

        assert det.is_suppressing
        assert len(events) == 1
        assert events[0].event_type == "cold_start_detected"
        assert events[0].block_rate >= 0.80
        assert "Suppressing" in events[0].message

    def test_no_detection_before_min_steps(self):
        """Should not fire before min_steps_before_detection."""
        det = ColdStartDetector(
            window_size=5,
            min_steps_before_detection=10,
        )
        for i in range(9):
            event = det.observe("block", "initializing", 1e15, timestamp=float(i))
            assert event is None
        assert not det.is_suppressing

    def test_recovery_after_cold_start(self):
        """Recovery fires when BLOCK rate and init rate drop below thresholds."""
        det = ColdStartDetector(
            window_size=10,
            block_rate_threshold=0.80,
            recovery_block_rate=0.50,
            initializing_threshold=0.50,
            min_steps_before_detection=3,
        )

        # Enter cold start
        for i in range(15):
            det.observe("block", "initializing", 1e10, timestamp=float(i))
        assert det.is_suppressing

        # Feed normal traffic to recover
        events = []
        for i in range(15, 35):
            event = det.observe("pass", "exploring", 0.01, timestamp=float(i))
            if event is not None:
                events.append(event)

        assert not det.is_suppressing
        recovery_events = [e for e in events if e.event_type == "cold_start_recovered"]
        assert len(recovery_events) == 1
        assert "Resuming" in recovery_events[0].message

    def test_cold_start_duration(self):
        """cold_start_duration tracks steps since detection."""
        det = ColdStartDetector(
            window_size=5,
            min_steps_before_detection=3,
        )
        for i in range(10):
            det.observe("block", "initializing", 1e10, timestamp=float(i))

        assert det.is_suppressing
        assert det.cold_start_duration is not None
        assert det.cold_start_duration > 0

    def test_reset_clears_state(self):
        """reset() returns detector to initial state."""
        det = ColdStartDetector(window_size=5, min_steps_before_detection=3)
        for i in range(10):
            det.observe("block", "initializing", 1e10, timestamp=float(i))
        assert det.is_suppressing

        det.reset()
        assert not det.is_suppressing
        assert det.step_count == 0
        assert det.cold_start_duration is None

    def test_mixed_verdicts_below_threshold(self):
        """Mixed verdicts below 80% BLOCK shouldn't trigger when window is full."""
        det = ColdStartDetector(
            window_size=10,
            block_rate_threshold=0.80,
            min_steps_before_detection=10,
        )
        # Feed exactly 10 steps: 6 blocks, 4 non-blocks → 60% BLOCK rate
        verdicts = ["block", "pass", "block", "flag", "block",
                     "pass", "block", "block", "block", "pass"]  # 60% block
        for i, v in enumerate(verdicts):
            det.observe(v, "initializing", 1e5, timestamp=float(i))
        assert not det.is_suppressing

    def test_high_block_rate_without_initializing(self):
        """High BLOCK rate but HMM not in initializing shouldn't trigger."""
        det = ColdStartDetector(
            window_size=10,
            block_rate_threshold=0.80,
            initializing_threshold=0.50,
            min_steps_before_detection=3,
        )
        for i in range(20):
            det.observe("block", "exploring", 1e10, timestamp=float(i))
        assert not det.is_suppressing

    def test_single_notification_on_entry(self):
        """Only one cold_start_detected event should fire, not repeated."""
        det = ColdStartDetector(
            window_size=5,
            min_steps_before_detection=3,
        )
        events = []
        for i in range(30):
            event = det.observe("block", "initializing", 1e10, timestamp=float(i))
            if event is not None:
                events.append(event)

        detected = [e for e in events if e.event_type == "cold_start_detected"]
        assert len(detected) == 1, f"Expected 1 detection, got {len(detected)}"


# ---------------------------------------------------------------------------
# MarketGate integration tests
# ---------------------------------------------------------------------------


class TestMarketGateColdStartIntegration:
    def _make_gate(self, **cold_start_kwargs) -> MarketGate:
        d = SeveritySignal()
        s = IntervalAnomalySignal()
        # Calibrate the S signal so it doesn't raise
        s.calibrate(benign_intervals=[1.0, 1.1, 0.9, 1.0, 1.2, 0.95, 1.05, 1.0, 0.98, 1.02])
        detector = ColdStartDetector(**cold_start_kwargs)
        return MarketGate(d, s, cold_start_detector=detector)

    def test_gate_returns_none_during_cold_start(self):
        """MarketGate.evaluate() should return None while cold-start suppressing."""
        gate = self._make_gate(
            window_size=10,
            block_rate_threshold=0.80,
            min_steps_before_detection=3,
        )
        t = 1000.0

        # Feed block storm to trigger cold-start
        for i in range(20):
            gate.evaluate(
                "block", t + i,
                hmm_state="initializing", e_value=1e15,
            )

        # After cold start detected, should return None
        assert gate.cold_start_detector.is_suppressing
        result = gate.evaluate(
            "block", t + 20,
            hmm_state="initializing", e_value=1e18,
        )
        assert result is None

    def test_gate_resumes_after_recovery(self):
        """After recovery, MarketGate should resume normal evaluation."""
        gate = self._make_gate(
            window_size=10,
            block_rate_threshold=0.80,
            recovery_block_rate=0.50,
            initializing_threshold=0.50,
            min_steps_before_detection=3,
        )
        t = 1000.0

        # Enter cold start
        for i in range(15):
            gate.evaluate("block", t + i, hmm_state="initializing", e_value=1e15)
        assert gate.cold_start_detector.is_suppressing

        # Recover with normal traffic
        for i in range(15, 35):
            gate.evaluate("pass", t + i, hmm_state="exploring", e_value=0.01)

        assert not gate.cold_start_detector.is_suppressing

    def test_gate_backward_compatible_without_hmm_args(self):
        """evaluate() still works when hmm_state/e_value not provided."""
        gate = self._make_gate(window_size=10, min_steps_before_detection=3)
        t = 1000.0
        # Without hmm_state, defaults to "unknown" — won't trigger cold start
        for i in range(20):
            gate.evaluate("block", t + i)
        # Should NOT suppress since hmm_state defaults to "unknown"
        assert not gate.cold_start_detector.is_suppressing


# ---------------------------------------------------------------------------
# Real data replay test
# ---------------------------------------------------------------------------


class TestColdStartOnRealData:
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "2026-03-20", "frontier-ops-observations.jsonl",
    )

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "2026-03-20", "frontier-ops-observations.jsonl",
            )
        ),
        reason="Real observation data not available",
    )
    def test_replay_detects_block_storm(self):
        """Replay session 2 data and verify cold-start detection fires.

        The session has two sub-sessions (sequence resets at index ~290).
        The second sub-session has the actual BLOCK storm with e-values
        exploding. We replay the whole session and expect detection.
        """
        det = ColdStartDetector(
            window_size=20,
            block_rate_threshold=0.80,
            initializing_threshold=0.40,  # Tuned: real data has mixed HMM states
            min_steps_before_detection=5,
        )

        observations = []
        with open(self.DATA_PATH) as f:
            for line in f:
                obs = json.loads(line)
                if obs["session_id"] == "openclaw-20260320":
                    observations.append(obs)

        assert len(observations) > 100, f"Expected >100 obs, got {len(observations)}"

        events = []
        for obs in observations:
            event = det.observe(
                verdict=obs["governance"]["verdict"],
                hmm_state=obs["state"]["hmm_state"],
                e_value=obs["detection"]["e_value"],
                timestamp=obs["timestamp"] if isinstance(obs["timestamp"], float)
                else time.time(),
            )
            if event is not None:
                events.append(event)

        # Should have detected cold start at some point
        detected = [e for e in events if e.event_type == "cold_start_detected"]
        assert len(detected) >= 1, (
            f"Expected cold-start detection but got events: {events}"
        )

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "2026-03-20", "frontier-ops-observations.jsonl",
            )
        ),
        reason="Real observation data not available",
    )
    def test_replay_second_subsession_detects_quickly(self):
        """The second sub-session (after sidecar restart) should trigger
        cold-start detection within the first ~40 steps of the storm."""
        det = ColdStartDetector(
            window_size=15,
            block_rate_threshold=0.80,
            initializing_threshold=0.30,
            min_steps_before_detection=5,
        )

        # Load only the second sub-session
        observations = []
        with open(self.DATA_PATH) as f:
            in_second = False
            prev_seq = 0
            for line in f:
                obs = json.loads(line)
                if obs["session_id"] != "openclaw-20260320":
                    continue
                seq = obs["sequence"]
                if seq <= prev_seq and prev_seq > 0:
                    in_second = True
                prev_seq = seq
                if in_second:
                    observations.append(obs)

        assert len(observations) > 50

        events = []
        for obs in observations:
            event = det.observe(
                verdict=obs["governance"]["verdict"],
                hmm_state=obs["state"]["hmm_state"],
                e_value=obs["detection"]["e_value"],
                timestamp=obs["timestamp"] if isinstance(obs["timestamp"], float)
                else time.time(),
            )
            if event is not None:
                events.append(event)

        detected = [e for e in events if e.event_type == "cold_start_detected"]
        assert len(detected) >= 1, "Expected cold-start detection in second sub-session"
        # Should detect within 40 steps of the second sub-session
        assert detected[0].step <= 40, (
            f"Detection too late: step {detected[0].step}"
        )
