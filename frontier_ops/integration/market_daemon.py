#!/usr/bin/env python3
"""
Market + Authorization Daemon
==============================

Persistent process that runs alongside the sidecar, providing:
1. Market evaluation (DAS-CUSUM over verdict severity + interval percentile)
2. Authorization scoping (goal extraction + geodesic radius + AGM revision)

Reads events from a FIFO (named pipe). Two event types:
  {"type": "verdict", "verdict": "pass", "timestamp": 1234.5}
  {"type": "user_message", "text": "Set up PyTorch on my GPU", "timestamp": 1234.5}
  {"type": "action", "tool": "exec", "content": "pip install torch", "timestamp": 1234.5}

Writes:
  ~/.openclaw/workspace/market_state.json (qualitative market signals)
  ~/.openclaw/workspace/authorization_state.json (goal + radius + verdict)
"""

import argparse
import json
import os
import signal
import sys
import time

from frontier_ops.integration.market_hook import MarketHook
from frontier_ops import FullPipeline

AUTH_STATE_PATH = os.path.expanduser("~/.openclaw/workspace/authorization_state.json")


def write_auth_state(pipeline, last_auth_verdict, last_action):
    """Write authorization state to disk (qualitative only for agent consumption)."""
    stats = pipeline.stats
    auth = stats.get("authorization", {})

    goal_info = auth.get("goal", {})
    state = {
        "authorization_active": True,
        "has_goal": goal_info.get("confidence", 0) > 0,
        "last_operator": auth.get("last_operator", "none"),
        "n_scope_events": auth.get("n_scope_events", 0),
        "last_verdict": last_auth_verdict,
        "radius_calibrated": auth.get("radius_calibrated", False),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }

    tmp = AUTH_STATE_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f, separators=(",", ":"))
        os.replace(tmp, AUTH_STATE_PATH)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Market + Authorization daemon")
    parser.add_argument("--fifo", required=True, help="Path to event FIFO")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Initialize market hook (auto-calibrate)
    hook = MarketHook.from_telemetry(verbose=args.verbose)

    # Initialize authorization pipeline
    auth_pipeline = FullPipeline()
    last_auth_verdict = "none"

    if args.verbose:
        print("[daemon] Market + Authorization initialized", file=sys.stderr)
        print(f"[daemon] Listening on {args.fifo}", file=sys.stderr)

    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False
        if args.verbose:
            print(f"[daemon] Shutting down (signal {sig})", file=sys.stderr)
            result = hook.verify_chain()
            print(
                f"[daemon] Market chain: {hook.audit_chain_length} entries, "
                f"valid={result.valid}",
                file=sys.stderr,
            )
            auth_stats = auth_pipeline.stats.get("authorization", {})
            print(
                f"[daemon] Auth: {auth_stats.get('n_scope_events', 0)} scope events, "
                f"last_operator={auth_stats.get('last_operator', 'none')}",
                file=sys.stderr,
            )

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while running:
        try:
            with open(args.fifo, 'r') as fifo:
                for line in fifo:
                    if not running:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        msg_type = msg.get("type", "verdict")
                        timestamp = msg.get("timestamp", time.time())

                        if msg_type == "user_message":
                            # Authorization: extract goal from user message
                            text = msg.get("text", "")
                            if text:
                                event = auth_pipeline.process_user_message(text)
                                if args.verbose:
                                    print(
                                        f"[daemon] 🎯 GOAL: op={event.operator.value} "
                                        f"conf={event.goal_after.confidence:.2f} "
                                        f"budget={event.replenish_amount:.2f}",
                                        file=sys.stderr,
                                    )
                                write_auth_state(auth_pipeline, last_auth_verdict, None)

                        elif msg_type == "action":
                            # Authorization: check action against goal
                            tool = msg.get("tool", "")
                            content = msg.get("content", "")
                            action_text = f"{tool}: {content}" if content else tool
                            r = auth_pipeline.process_step(action_text)
                            last_auth_verdict = getattr(r, "authorization_verdict", "N/A")
                            geodesic_d = getattr(r, "geodesic_distance", 0)

                            if args.verbose and last_auth_verdict != "pass":
                                print(
                                    f"[daemon] 🔒 AUTH: [{last_auth_verdict}] "
                                    f"d={geodesic_d:.3f} tool={tool}",
                                    file=sys.stderr,
                                )
                            write_auth_state(auth_pipeline, last_auth_verdict, action_text)

                        elif msg_type == "verdict":
                            # Market: feed sidecar verdict
                            verdict = msg.get("verdict", "pass")
                            label = hook.on_step(verdict, timestamp)
                            if label and args.verbose:
                                print(f"[daemon] 📊 {label}", file=sys.stderr)

                        # Also feed action to market if it has a verdict
                        if msg_type == "verdict":
                            pass  # already handled above

                    except (json.JSONDecodeError, Exception) as e:
                        if args.verbose:
                            print(f"[daemon] Error: {e}", file=sys.stderr)

        except OSError:
            if running:
                time.sleep(0.5)

    if args.verbose:
        print(
            f"[daemon] Stopped. Market: {hook.n_steps} steps, "
            f"{hook.n_alerts} alerts.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
