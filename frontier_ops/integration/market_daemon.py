#!/usr/bin/env python3
"""
Market Daemon — Persistent process for proprioceptive market evaluation.

Reads verdicts from a FIFO (named pipe), maintains stateful DAS-CUSUM
accumulators and governance chain, writes market_state.json.

Run:
    python market_daemon.py --fifo /path/to/fifo

The daemon auto-calibrates from the telemetry log on startup, then
processes one verdict per line from the FIFO.
"""

import argparse
import json
import os
import signal
import sys
import time

from frontier_ops.integration.market_hook import MarketHook


def main():
    parser = argparse.ArgumentParser(description="Market evaluation daemon")
    parser.add_argument("--fifo", required=True, help="Path to verdict FIFO")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Initialize hook (auto-calibrate from telemetry)
    hook = MarketHook.from_telemetry(verbose=args.verbose)

    if args.verbose:
        print(f"[market-daemon] Listening on {args.fifo}", file=sys.stderr)

    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False
        if args.verbose:
            print(f"[market-daemon] Shutting down (signal {sig})", file=sys.stderr)
            result = hook.verify_chain()
            print(
                f"[market-daemon] Chain: {hook.audit_chain_length} entries, "
                f"valid={result.valid}",
                file=sys.stderr,
            )

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while running:
        try:
            # Open FIFO for reading (blocks until writer connects)
            with open(args.fifo, 'r') as fifo:
                for line in fifo:
                    if not running:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        verdict = msg.get("verdict", "pass")
                        timestamp = msg.get("timestamp", time.time())
                        label = hook.on_step(verdict, timestamp)
                        if label and args.verbose:
                            print(
                                f"[market-daemon] 📊 {label}",
                                file=sys.stderr,
                            )
                    except (json.JSONDecodeError, Exception) as e:
                        if args.verbose:
                            print(
                                f"[market-daemon] Error: {e}",
                                file=sys.stderr,
                            )
        except OSError:
            # FIFO may be removed during shutdown
            if running:
                time.sleep(0.5)

    if args.verbose:
        print(
            f"[market-daemon] Stopped. {hook.n_steps} steps, "
            f"{hook.n_alerts} alerts.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
