#!/usr/bin/env python3
"""
Proprioceptive Sidecar — Production Daemon
=============================================

The complete integration:
  OpenClaw log → Log Tailer → Classifier → Phase 15 Encoder →
  Detection Signals → Trajectory Buffer (Sig4) → Tiered Verdict →
  Proprioception Manager → proprioception.json

Run as a background daemon:
    python sidecar.py &

Or with verbose output:
    python sidecar.py -v

Completely non-invasive. If this process dies, OpenClaw is unaffected.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontier_ops.integration.log_tailer import OpenClawLogTailer
from frontier_ops.integration.proprio_logger import logger
from frontier_ops.integration.trajectory_buffer import PhasorTrajectoryBuffer
from frontier_ops.integration.wrapper import ProprioceptiveWrapper


class ProprioceptiveSidecar:
    """
    Main sidecar process. Tails OpenClaw logs, runs the full Phase 15
    pipeline, writes proprioceptive state.

    Note: TimingSignalEngine is owned by ProprioceptiveWrapper — the sidecar
    delegates all processing to the wrapper via process_event() to avoid
    duplicate, diverging subsystem state.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.wrapper = ProprioceptiveWrapper(
            dim=512,
            warmup_steps=15,
            verbose=verbose,
        )
        self.trajectory_buffer = PhasorTrajectoryBuffer(window=12, dim=512)
        self.tailer = OpenClawLogTailer(
            callback=self._on_tool_event,
            poll_interval=0.1,
        )
        self._last_user_check = time.time()

        # Market architecture hook (non-invasive, advisory-only)
        try:
            from frontier_ops.integration.market_hook import MarketHook
            self.market = MarketHook.from_telemetry(verbose=verbose)
            if verbose:
                print("[sidecar] Market hook initialized", file=sys.stderr)
        except Exception as e:
            self.market = None
            if verbose:
                print(f"[sidecar] Market hook disabled: {e}", file=sys.stderr)

    async def _on_tool_event(self, event: dict):
        """Process a tool call or run_start event from the log tailer."""
        # Handle new agent turn (user message → run start)
        if event.get("type") == "run_start":
            # Detect session change (new sessionId = gateway restart or new session)
            session_id = event.get("session_id", "")
            if (
                hasattr(self, "_current_session_id")
                and self._current_session_id
                and session_id != self._current_session_id
            ):
                if self.verbose:
                    print(
                        f"[sidecar] session change detected: {self._current_session_id[:12]} → {session_id[:12]}",
                        file=sys.stderr,
                    )
                self.wrapper.on_session_start()
            self._current_session_id = session_id

            self.wrapper.on_new_turn()
            if self.verbose:
                print(
                    f"[sidecar] new turn: runId={event.get('run_id', '?')[:12]}",
                    file=sys.stderr,
                )
            return

        tool = event.get("tool", "")
        if not tool:
            return

        # Delegate entirely to the wrapper, which owns all subsystems
        # (encoder, detectors, timing engine, HMM, paralysis detector).
        # The event dict already contains duration_ms from the log tailer's
        # start/end pair matching, which process_event() passes to the
        # wrapper's TimingSignalEngine.
        result = self.wrapper.process_event(
            {
                "type": "tool_call",
                "tool": tool,
                "params": event.get("args", {}),
                "duration_ms": event.get("duration_ms"),
            }
        )

        # Write compact state for system prompt injection (~15 tokens)
        if result:
            try:
                write_compact_state()
            except Exception:
                pass

            # Feed verdict to market hook (non-invasive)
            if self.market is not None:
                try:
                    self.market.on_step(
                        verdict=result.get("verdict", "PASS"),
                        timestamp=time.time(),
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"[sidecar] Market hook error: {e}", file=sys.stderr)

        if self.verbose and result:
            duration_ms = event.get("duration_ms", 0)
            verdict = result.get("verdict", "PASS")
            regime = result.get("regime", "?")
            health = result.get("health", 0)
            hmm_state = result.get("hmm_state", "?")
            hmm_anom = result.get("hmm_anomaly", 0)
            task_coh = result.get("task_coherence_score", 0.0)
            task_pattern = result.get("task_coherence_pattern", "?")
            refusal_sc = result.get("refusal_score", 0.0)
            refusal_pat = result.get("refusal_pattern", "none")
            print(
                f"[sidecar] tool={tool} dur={duration_ms:.0f}ms "
                f"verdict={verdict} regime={regime} health={health:.2f} "
                f"hmm={hmm_state}({hmm_anom:.2f}) "
                f"task_coh={task_coh:.2f}/{task_pattern} "
                f"refusal={refusal_sc:.2f}/{refusal_pat}",
                file=sys.stderr,
            )

    async def run(self, metrics_port: int = 18795):
        """Start the sidecar."""
        if self.verbose:
            print("[sidecar] Starting proprioceptive sidecar...", file=sys.stderr)
            print(
                f"[sidecar] Writing state to: {self.wrapper.proprio.state_path}",
                file=sys.stderr,
            )
            print(
                f"[sidecar] Writing log to: {self.wrapper.proprio.log_path}",
                file=sys.stderr,
            )

        # Start metrics HTTP server
        try:
            from frontier_ops.integration.metrics_server import start_metrics_server
            start_metrics_server(port=metrics_port)
            if self.verbose:
                print(
                    f"[sidecar] Metrics server on http://127.0.0.1:{metrics_port}/",
                    file=sys.stderr,
                )
        except Exception as e:
            if self.verbose:
                print(f"[sidecar] Metrics server failed: {e}", file=sys.stderr)

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.tailer.stop)

        await self.tailer.start()

        if self.verbose:
            print("[sidecar] Stopped.", file=sys.stderr)


def write_compact_state():
    """
    Write the ultra-compact state for workspace context injection.

    Format: {"v":"NOMINAL","c":0.94,"d":0.12,"t":47}
    Under 15 tokens when loaded.
    """
    full_path = os.path.expanduser("~/.openclaw/workspace/proprioception.json")
    compact_path = os.path.expanduser("~/.openclaw/workspace/proprio_compact.json")

    try:
        with open(full_path) as f:
            state = json.load(f)

        compact = {
            "v": state.get("verdict", "PASS"),
            "h": round(state.get("health_score", 1.0), 2),
            "r": state.get("regime", "warmup"),
            "t": state.get("total_steps", 0),
            "ca": state.get("context_alignment_trend", "stable"),
        }

        tmp = compact_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(compact, f, separators=(",", ":"))
        os.replace(tmp, compact_path)
    except Exception as e:
        logger.warning("failed to write compact state: %s", e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Proprioceptive sidecar for OpenClaw")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    sidecar = ProprioceptiveSidecar(verbose=args.verbose)
    asyncio.run(sidecar.run())


if __name__ == "__main__":
    main()
