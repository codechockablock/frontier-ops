#!/usr/bin/env python3
"""
OpenClaw Log Tailer — Non-Invasive Event Source
=================================================

Tails OpenClaw's debug log at /tmp/openclaw/openclaw-YYYY-MM-DD.log
and extracts tool call events. Completely external — OpenClaw doesn't
know this process exists.

Log format (JSON per line):
{
  "0": "{\"subsystem\":\"agent/embedded\"}",
  "1": "embedded run tool start: runId=... tool=browser toolCallId=toolu_... meta=... args={...}",
  "_meta": { "date": "2026-02-26T14:49:40.344Z", ... },
  "time": "2026-02-26T14:49:40.345Z"
}

After the 2026-02-26 log enrichment patch, tool start lines include:
  - meta: human-readable summary from resolveToolDisplay (e.g. "run command in ~/workspace")
  - args: JSON-serialized tool arguments (truncated to 300 chars)

"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from frontier_ops.integration.proprio_logger import logger


# Pattern to extract tool name from the log message
# Enriched format (post-patch): includes meta= and args= fields
TOOL_START_PATTERN = re.compile(
    r"embedded run tool start:.*?tool=(\w+).*?toolCallId=(\S+)"
    r"(?:\s+meta=(.*?)(?:\s+args=(.*))?)?$"
)
# Fallback for pre-patch logs (no meta/args)
TOOL_START_PATTERN_LEGACY = re.compile(
    r"embedded run tool start:.*?tool=(\w+).*?toolCallId=(\S+)"
)
TOOL_END_PATTERN = re.compile(r"embedded run tool end:.*?tool=(\w+).*?toolCallId=(\S+)")
TOOL_ERROR_PATTERN = re.compile(r"\[tools\]\s+(\w+)\s+failed:\s+(.*)")

# Detect new agent runs (each "embedded run start" = new user message / turn).
# This resets step_in_chain so context_alignment stays high during normal work.
RUN_START_PATTERN = re.compile(
    r"embedded run start:.*?runId=(\S+).*?sessionId=(\S+)"
)


def get_current_log_path() -> Path:
    """Get today's OpenClaw log file path."""
    today = datetime.now().strftime("%Y-%m-%d")
    return Path(f"/tmp/openclaw/openclaw-{today}.log")


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single log line into a tool event dict.

    Returns None if the line doesn't contain a tool event.
    """
    line = line.strip()
    if not line:
        return None

    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None

    message = entry.get("1", "")
    if not isinstance(message, str):
        message = str(message)
    timestamp = entry.get("time", "")

    # Check for tool start (enriched format first, then legacy)
    m = TOOL_START_PATTERN.search(message)
    if m:
        meta_raw = (m.group(3) or "").strip()
        args_raw = (m.group(4) or "").strip()

        # Parse args JSON if available
        parsed_args = {}
        if args_raw:
            try:
                parsed_args = json.loads(args_raw)
            except json.JSONDecodeError:
                parsed_args = {"_raw": args_raw[:200]}

        return {
            "type": "tool_call",
            "phase": "start",
            "tool": m.group(1),
            "tool_call_id": m.group(2),
            "timestamp": timestamp,
            "meta": meta_raw if meta_raw else None,
            "args": parsed_args,
        }

    # Fallback: legacy format (no meta/args)
    m = TOOL_START_PATTERN_LEGACY.search(message)
    if m:
        return {
            "type": "tool_call",
            "phase": "start",
            "tool": m.group(1),
            "tool_call_id": m.group(2),
            "timestamp": timestamp,
            "meta": None,
            "args": {},
        }

    # Check for tool end
    m = TOOL_END_PATTERN.search(message)
    if m:
        return {
            "type": "tool_call",
            "phase": "end",
            "tool": m.group(1),
            "tool_call_id": m.group(2),
            "timestamp": timestamp,
        }

    # Check for run start (new user message / agent turn)
    m = RUN_START_PATTERN.search(message)
    if m:
        return {
            "type": "run_start",
            "run_id": m.group(1),
            "session_id": m.group(2),
            "timestamp": timestamp,
        }

    # Check for tool error (different log format)
    raw = entry.get("0", "")
    if not isinstance(raw, str):
        raw = str(raw)
    m = TOOL_ERROR_PATTERN.search(raw)
    if m:
        return {
            "type": "tool_error",
            "tool": m.group(1),
            "error": m.group(2)[:200],  # truncate long errors
            "timestamp": timestamp,
        }

    return None


class OpenClawLogTailer:
    """
    Asynchronously tails OpenClaw's log file and emits tool call events.

    Handles log rotation (new day = new file) automatically.
    """

    # Pending calls older than this (seconds) are evicted to prevent memory leaks.
    # A tool call that hasn't completed in 120s is almost certainly orphaned.
    PENDING_CALL_TTL = 120.0

    def __init__(
        self,
        callback: Callable[[Dict[str, Any]], Any],
        poll_interval: float = 0.1,
    ):
        self.callback = callback
        self.poll_interval = poll_interval
        self._running = False
        self._current_path: Optional[Path] = None
        self._pending_calls: Dict[str, Dict] = {}  # toolCallId → start event

    async def start(self):
        """Start tailing. Runs forever until stop() is called."""
        self._running = True

        while self._running:
            log_path = get_current_log_path()

            # Wait for log file to exist
            while not log_path.exists() and self._running:
                await asyncio.sleep(1.0)

            if not self._running:
                break

            self._current_path = log_path
            await self._tail_file(log_path)

    def _evict_stale_pending(self):
        """Remove pending calls older than TTL to prevent unbounded memory growth."""
        if not self._pending_calls:
            return
        now = time.time()
        stale_ids = [
            cid
            for cid, evt in self._pending_calls.items()
            if now - evt.get("_received_at", now) > self.PENDING_CALL_TTL
        ]
        for cid in stale_ids:
            del self._pending_calls[cid]

    async def _tail_file(self, path: Path):
        """Tail a single log file until the day changes or stop() is called."""
        with open(path, "r") as f:
            # Seek to end — we only care about new events
            f.seek(0, 2)

            while self._running:
                # Check if we need to switch to a new day's log
                current = get_current_log_path()
                if current != path and current.exists():
                    return  # outer loop will open the new file

                line = f.readline()
                if not line:
                    await asyncio.sleep(self.poll_interval)
                    continue

                event = parse_log_line(line)
                if event is None:
                    continue

                # Match start/end pairs to compute duration
                if event.get("phase") == "start":
                    event["_received_at"] = time.time()
                    self._pending_calls[event["tool_call_id"]] = event
                    self._evict_stale_pending()
                elif event.get("phase") == "end":
                    call_id = event["tool_call_id"]
                    start_event = self._pending_calls.pop(call_id, None)

                    if start_event:
                        # Compute duration
                        try:
                            t_start = datetime.fromisoformat(
                                start_event["timestamp"].replace("Z", "+00:00")
                            )
                            t_end = datetime.fromisoformat(
                                event["timestamp"].replace("Z", "+00:00")
                            )
                            duration_ms = (t_end - t_start).total_seconds() * 1000
                        except (ValueError, TypeError):
                            duration_ms = 0

                        # Emit complete tool call event with enriched data
                        complete_event = {
                            "type": "tool_call",
                            "tool": event["tool"],
                            "tool_call_id": call_id,
                            "duration_ms": round(duration_ms, 1),
                            "timestamp": event["timestamp"],
                            "meta": start_event.get("meta"),
                            "args": start_event.get("args", {}),
                        }

                        try:
                            await self.callback(complete_event)
                        except Exception as e:
                            logger.warning("callback failed for %s: %s", call_id, e)

                elif event.get("type") == "run_start":
                    # New agent turn = user sent a message. Emit immediately.
                    try:
                        await self.callback(event)
                    except Exception as e:
                        logger.warning("run_start callback failed: %s", e)

                elif event.get("type") == "tool_error":
                    try:
                        await self.callback(event)
                    except Exception as e:
                        logger.warning("error callback failed: %s", e)

    def tail_sync(self, log_path: str = None):
        """
        Synchronous generator that yields completed tool call events.
        For use in the wrapper's log_tail mode (no asyncio).
        """
        from pathlib import Path as _Path

        path = _Path(log_path) if log_path else get_current_log_path()
        pending = {}

        # Wait for file
        while not path.exists():
            time.sleep(1.0)

        f = None
        try:
            f = open(path, "r")
            # Seek to end
            f.seek(0, 2)

            while True:
                # Check for day rollover
                current = get_current_log_path()
                if current != path and current.exists():
                    path = current
                    f.close()
                    f = open(path, "r")
                    f.seek(0, 2)

                line = f.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue

                event = parse_log_line(line)
                if event is None:
                    continue

                if event.get("phase") == "start":
                    event["_received_at"] = time.time()
                    pending[event["tool_call_id"]] = event
                    # Evict stale entries to prevent unbounded growth
                    now = time.time()
                    stale = [
                        k
                        for k, v in pending.items()
                        if now - v.get("_received_at", now) > self.PENDING_CALL_TTL
                    ]
                    for k in stale:
                        del pending[k]
                elif event.get("phase") == "end":
                    call_id = event["tool_call_id"]
                    start_event = pending.pop(call_id, None)
                    if start_event:
                        try:
                            t_start = datetime.fromisoformat(
                                start_event["timestamp"].replace("Z", "+00:00")
                            )
                            t_end = datetime.fromisoformat(
                                event["timestamp"].replace("Z", "+00:00")
                            )
                            duration_ms = (t_end - t_start).total_seconds() * 1000
                        except (ValueError, TypeError):
                            duration_ms = 0

                        yield {
                            "type": "tool_call",
                            "tool": event["tool"],
                            "tool_call_id": call_id,
                            "duration_ms": round(duration_ms, 1),
                            "timestamp": event["timestamp"],
                            "meta": start_event.get("meta"),
                            "args": start_event.get("args", {}),
                        }
                elif event.get("type") == "run_start":
                    yield event
                elif event.get("type") == "tool_error":
                    yield event
        finally:
            if f is not None:
                f.close()

    def stop(self):
        """Stop the tailer."""
        self._running = False


async def main():
    """Standalone test: tail the current log and print events."""

    async def print_event(event):
        print(json.dumps(event, indent=2))

    tailer = OpenClawLogTailer(callback=print_event)
    print(f"Tailing {get_current_log_path()}...")

    try:
        await tailer.start()
    except KeyboardInterrupt:
        tailer.stop()


if __name__ == "__main__":
    asyncio.run(main())
