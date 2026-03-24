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
import logging
import os
import signal
import stat
import sys
import time

logger = logging.getLogger(__name__)

from frontier_ops.integration.market_hook import MarketHook
from frontier_ops import FullPipeline

AUTH_STATE_PATH = os.path.expanduser("~/.openclaw/workspace/authorization_state.json")


def validate_fifo_path(path: str) -> str:
    """Validate that *path* is a FIFO (named pipe).

    - If the path exists, it must be a FIFO — regular files and symlinks are
      rejected with ``ValueError``.
    - If the path does not exist, a new FIFO is created via ``os.mkfifo``.

    Returns the validated (absolute) path.
    """
    path = os.path.abspath(path)

    if os.path.exists(path):
        # lstat so we don't follow symlinks
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"FIFO path is a symlink (refusing to follow): {path}"
            )
        if not stat.S_ISFIFO(mode):
            raise ValueError(
                f"FIFO path exists but is not a FIFO (mode={oct(mode)}): {path}"
            )
        logger.info("Using existing FIFO: %s", path)
    else:
        os.mkfifo(path)
        logger.info("Created new FIFO: %s", path)

    return path


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
        logger.exception("Failed to write authorization state to %s", AUTH_STATE_PATH)


def main():
    parser = argparse.ArgumentParser(description="Market + Authorization daemon")
    parser.add_argument("--fifo", required=True, help="Path to event FIFO")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Validate FIFO path before anything else
    args.fifo = validate_fifo_path(args.fifo)

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
                            # Market: feed sidecar verdict + cold-start signals
                            verdict = msg.get("verdict", "pass")
                            hmm_state = msg.get("hmm_state", "unknown")
                            e_value = float(msg.get("e_value", 0.0))

                            # Feed all verdicts to market gate. Previously FLAG/BLOCK
                            # were skipped to avoid amplification cascades, but this
                            # caused the market to go stale when the sidecar had
                            # elevated flag rates. The market's own DAS-CUSUM handles
                            # severity properly — it doesn't need pre-filtering.
                            label = hook.on_step(verdict, timestamp, hmm_state=hmm_state, e_value=e_value)
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
