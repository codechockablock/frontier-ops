"""Tests for market_daemon FIFO validation and exception logging."""

import os
import stat
import tempfile

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
