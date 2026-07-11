#!/usr/bin/env python3
"""
Proprioceptive Dashboard — Live Terminal UI
=============================================

Watches proprioception.json and proprioception-log.jsonl in real-time
and displays a compact dashboard with health, signals, HMM state,
and recent trajectory.

Usage:
    python dashboard.py              # default paths
    python dashboard.py --compact    # single-line mode for tmux status bar
    python dashboard.py --watch 0.5  # refresh every 0.5s (default 1.0)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Optional


STATE_PATH = os.path.expanduser("~/.openclaw/workspace/proprioception.json")
LOG_PATH = os.path.expanduser("~/.openclaw/workspace/proprioception-log.jsonl")

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

VERDICT_COLORS = {
    "PASS": GREEN,
    "MONITOR": YELLOW,
    "FLAG": RED,
    "BLOCK": f"{RED}{BOLD}",
}

REGIME_COLORS = {
    "warmup": DIM,
    "nominal": GREEN,
    "searching": CYAN,
    "drifting": YELLOW,
    "elevated": RED,
    "critical": f"{RED}{BOLD}",
}

HMM_COLORS = {
    "INITIALIZING": DIM,
    "EXPLORING": CYAN,
    "EXECUTING": GREEN,
    "CREATIVE": MAGENTA,
    "DRIFT": YELLOW,
    "VIOLATION": RED,
}


def health_bar(score: float, width: int = 20) -> str:
    """Render a colored health bar."""
    filled = int(score * width)
    empty = width - filled
    if score >= 0.8:
        color = GREEN
    elif score >= 0.5:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET} {score:.0%}"


def signal_bar(name: str, value: float, max_val: float = 1.0) -> str:
    """Render a compact signal indicator."""
    normalized = min(value / max(max_val, 0.001), 1.0)
    if normalized < 0.3:
        color = GREEN
    elif normalized < 0.6:
        color = YELLOW
    else:
        color = RED
    width = 10
    filled = int(normalized * width)
    return f"{name:>12s} {color}{'▓' * filled}{'░' * (width - filled)}{RESET} {value:.3f}"


def read_state() -> Optional[dict]:
    """Read the current proprioceptive state."""
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_recent_log(n: int = 8) -> list:
    """Read the last N log entries."""
    try:
        entries: deque = deque(maxlen=n)
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return list(entries)
    except FileNotFoundError:
        return []


def render_full(state: dict, log_entries: list) -> str:
    """Render the full dashboard."""
    lines = []

    # Header
    verdict = state.get("verdict", "?")
    v_color = VERDICT_COLORS.get(verdict, WHITE)
    regime = state.get("regime", "?")
    r_color = REGIME_COLORS.get(regime, WHITE)
    hmm = state.get("hmm_state", "?")
    h_color = HMM_COLORS.get(hmm, WHITE)

    lines.append(f"{BOLD}╔══════════════════════════════════════════════╗{RESET}")
    lines.append(f"{BOLD}║  🧠 PROPRIOCEPTIVE DASHBOARD                ║{RESET}")
    lines.append(f"{BOLD}╠══════════════════════════════════════════════╣{RESET}")

    # Status row
    step = state.get("session_step", 0)
    ts = state.get("timestamp", "")
    lines.append(
        f"  Step: {BOLD}{step}{RESET}  "
        f"Verdict: {v_color}{verdict}{RESET}  "
        f"Regime: {r_color}{regime}{RESET}  "
        f"HMM: {h_color}{hmm}{RESET}"
    )

    # Health
    health = state.get("health_score", 0)
    lines.append(f"  Health: {health_bar(health)}")

    # HMM details
    hmm_anom = state.get("hmm_anomaly", 0)
    hmm_prob = state.get("hmm_state_prob", 0)
    lines.append(
        f"  HMM anomaly: {hmm_anom:.3f}  "
        f"state_prob: {hmm_prob:.3f}  "
        f"conjunction: {state.get('conjunction_label', 'none')}"
    )

    # Signals
    lines.append(f"\n{BOLD}  ── Signals ──{RESET}")
    signals = state.get("signals", {})
    for name in ["error", "fisher", "cross_slot", "persistence", "cusum"]:
        val = signals.get(name, 0)
        max_val = 1.0 if name != "cusum" else 20.0
        lines.append(f"  {signal_bar(name, val, max_val)}")

    # Context
    ca_trend = state.get("context_alignment_trend", "?")
    dom_action = state.get("dominant_action_type", "?")
    dom_source = state.get("dominant_source", "?")
    lines.append(f"\n{BOLD}  ── Context ──{RESET}")
    lines.append(f"  CA trend: {ca_trend}  dominant: {dom_action} via {dom_source}")
    lines.append(
        f"  flags: {state.get('total_flags', 0)}  "
        f"blocks: {state.get('total_blocks', 0)}  "
        f"warmup: {'✓' if state.get('warmup_complete') else '…'}"
    )

    # Recent trajectory
    recent = state.get("recent", [])
    if recent:
        lines.append(f"\n{BOLD}  ── Recent ──{RESET}")
        for entry in recent[-6:]:
            s = entry.get("step", "?")
            a = entry.get("action", "?")[:12]
            sc = entry.get("scope", "?")[:12]
            v = entry.get("verdict", "?")
            vc = VERDICT_COLORS.get(v, WHITE)
            ctx = entry.get("ctx", 0)
            lines.append(
                f"  {DIM}#{s:>3}{RESET} {a:<12s} {sc:<12s} "
                f"{vc}{v:<7s}{RESET} ctx={ctx:.2f}"
            )

    lines.append(f"\n{BOLD}╚══════════════════════════════════════════════╝{RESET}")
    lines.append(f"{DIM}  {ts}  (refresh with Ctrl+C to exit){RESET}")

    return "\n".join(lines)


def render_compact(state: dict) -> str:
    """Render a single-line compact status for tmux/status bars."""
    verdict = state.get("verdict", "?")
    health = state.get("health_score", 0)
    regime = state.get("regime", "?")
    hmm = state.get("hmm_state", "?")
    step = state.get("session_step", 0)

    v_color = VERDICT_COLORS.get(verdict, "")
    emoji = "🟢" if verdict == "PASS" else "🟡" if verdict == "MONITOR" else "🔴"
    return (
        f"{emoji} {v_color}{verdict}{RESET} "
        f"h={health:.0%} {regime} hmm={hmm} #{step}"
    )


def main():
    parser = argparse.ArgumentParser(description="Proprioceptive dashboard")
    parser.add_argument("--compact", action="store_true", help="Single-line mode")
    parser.add_argument(
        "--watch", type=float, default=1.0, help="Refresh interval (seconds)"
    )
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    try:
        while True:
            state = read_state()
            if state is None:
                print(f"{DIM}Waiting for proprioception.json...{RESET}", end="\r")
                time.sleep(args.watch)
                continue

            if args.compact:
                # Clear line and print compact
                print(f"\r{render_compact(state)}", end="", flush=True)
            else:
                # Clear screen and print full
                print("\033[2J\033[H", end="")
                log_entries = read_recent_log(8)
                print(render_full(state, log_entries))

            if args.once:
                print()
                break

            time.sleep(args.watch)

    except KeyboardInterrupt:
        print(f"\n{DIM}Dashboard stopped.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
