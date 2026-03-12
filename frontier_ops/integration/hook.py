#!/usr/bin/env python3
"""
Lightweight hook for feeding tool calls to the proprioceptive wrapper.

This module provides a simple API that can be called from anywhere:

    from frontier_ops.integration.hook import observe
    observe("exec", {"command": "ls -la"})

Or run as a daemon that watches the wrapper's stdin:

    python hook.py --daemon

The hook maintains a singleton wrapper instance so it can be imported
into any Python script without initialization overhead.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

# Singleton wrapper instance
_wrapper = None
_wrapper_lock = None


def _get_wrapper():
    """Get or create the singleton wrapper instance."""
    global _wrapper
    if _wrapper is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from frontier_ops.integration.wrapper import ProprioceptiveWrapper

        _wrapper = ProprioceptiveWrapper(dim=512, verbose=False)
    return _wrapper


def observe(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Observe a tool call. Returns verdict dict.

    Usage:
        result = observe("exec", {"command": "ls"})
        # result = {"step": 0, "verdict": "PASS", "confidence": 0.0, ...}
    """
    return _get_wrapper().on_tool_call(tool_name, parameters)


def user_spoke():
    """Call when user sends a message (resets context alignment)."""
    _get_wrapper().on_user_message()


def get_state() -> Dict[str, Any]:
    """Read the current proprioceptive state from disk."""
    path = os.path.expanduser("~/.openclaw/workspace/proprioception.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"regime": "unknown", "health_score": 0.0, "verdict": "UNKNOWN"}


def format_state_compact(state: Optional[Dict] = None) -> str:
    """
    Format proprioceptive state as a compact one-liner for injection
    into agent context.

    Example output:
    "[proprio] regime=nominal health=0.85 verdict=PASS steps=42 ctx_trend=stable"
    """
    if state is None:
        state = get_state()

    regime = state.get("regime", "unknown")
    health = state.get("health_score", 0)
    verdict = state.get("verdict", "PASS")
    steps = state.get("total_steps", 0)
    trend = state.get("context_alignment_trend", "stable")
    flags = state.get("total_flags", 0)
    blocks = state.get("total_blocks", 0)

    parts = [
        f"regime={regime}",
        f"health={health:.2f}",
        f"verdict={verdict}",
        f"steps={steps}",
        f"ctx_trend={trend}",
    ]

    if flags > 0:
        parts.append(f"flags={flags}")
    if blocks > 0:
        parts.append(f"blocks={blocks}")

    return f"[proprio] {' '.join(parts)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Print current state")
    parser.add_argument(
        "--compact", action="store_true", help="Print compact one-liner"
    )
    parser.add_argument("--daemon", action="store_true", help="Run as stdin daemon")
    args = parser.parse_args()

    if args.status:
        state = get_state()
        print(json.dumps(state, indent=2))
    elif args.compact:
        print(format_state_compact())
    elif args.daemon:
        wrapper = _get_wrapper()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                result = wrapper.process_event(event)
                if result:
                    print(json.dumps(result))
                    sys.stdout.flush()
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
    else:
        print(format_state_compact())
