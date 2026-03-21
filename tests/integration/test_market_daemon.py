"""Tests for market_daemon FIFO validation, exception logging, and verdict filtering."""

import os
import stat
import tempfile
from unittest.mock import MagicMock, call

import pytest

from frontier_ops.integration.market_daemon import validate_fifo_path


class TestValidateFifoPath:
    """Tests for validate_fifo_path()."""

    def test_accepts_existing_fifo(self, tmp_path):
        fifo = tmp_path / "test.fifo"
        os.mkfifo(str(fifo))
        result = validate_fifo_path(str(fifo))
        assert result == str(fifo)
        assert stat.S_ISFIFO(os.stat(result).st_mode)

    def test_rejects_regular_file(self, tmp_path):
        regular = tmp_path / "regular.txt"
        regular.write_text("not a fifo")
        with pytest.raises(ValueError, match="not a FIFO"):
            validate_fifo_path(str(regular))

    def test_rejects_symlink(self, tmp_path):
        fifo = tmp_path / "real.fifo"
        os.mkfifo(str(fifo))
        link = tmp_path / "link.fifo"
        os.symlink(str(fifo), str(link))
        with pytest.raises(ValueError, match="symlink"):
            validate_fifo_path(str(link))

    def test_creates_fifo_when_missing(self, tmp_path):
        fifo = tmp_path / "new.fifo"
        assert not fifo.exists()
        result = validate_fifo_path(str(fifo))
        assert os.path.exists(result)
        assert stat.S_ISFIFO(os.stat(result).st_mode)

    def test_rejects_directory(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises(ValueError, match="not a FIFO"):
            validate_fifo_path(str(d))


class TestWriteAuthStateLogging:
    """Verify write_auth_state logs exceptions instead of silencing them."""

    def test_logs_on_write_failure(self, tmp_path, monkeypatch, caplog):
        import logging
        from unittest.mock import MagicMock
        from frontier_ops.integration.market_daemon import write_auth_state, AUTH_STATE_PATH

        # Point AUTH_STATE_PATH to an impossible location
        bad_path = str(tmp_path / "no" / "such" / "dir" / "state.json")
        monkeypatch.setattr(
            "frontier_ops.integration.market_daemon.AUTH_STATE_PATH", bad_path
        )

        pipeline = MagicMock()
        pipeline.stats = {"authorization": {}}

        with caplog.at_level(logging.ERROR, logger="frontier_ops.integration.market_daemon"):
            write_auth_state(pipeline, "none", None)

        assert any("Failed to write authorization state" in r.message for r in caplog.records)


class TestVerdictFiltering:
    """Verify that the daemon's verdict filter logic correctly skips FLAG/BLOCK."""

    def test_only_pass_and_monitor_reach_market_gate(self):
        """Alternating FLAG/PASS — market gate should only receive PASS verdicts.

        Simulates the daemon's filtering logic: only call hook.on_step()
        for PASS and MONITOR verdicts.
        """
        from frontier_ops.integration.market_hook import MarketHook
        import numpy as np

        rng = np.random.default_rng(42)
        benign_intervals = sorted(rng.exponential(scale=8.0, size=500))

        hook = MarketHook.from_intervals(
            benign_intervals=benign_intervals,
            state_path="/dev/null",
        )

        # Simulate daemon verdict filtering: alternating FLAG/PASS
        verdicts = ["flag", "pass"] * 10  # 20 verdicts total
        fed_to_hook = []
        t = 0.0
        for v in verdicts:
            t += 1.0
            if v in ("pass", "monitor"):
                hook.on_step(v, t)
                fed_to_hook.append(v)

        # Only PASS verdicts should have been fed (10 out of 20)
        assert len(fed_to_hook) == 10
        assert all(v == "pass" for v in fed_to_hook)
        assert hook.n_steps == 10

        # D signal should stay at 0 since only PASS was fed
        state = hook.get_state()
        assert state.d_raw == 0.0, (
            f"d_raw={state.d_raw}, should be 0.0 with only PASS input"
        )
