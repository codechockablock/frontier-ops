#!/usr/bin/env python3
"""
CLI entry point for the market architecture benchmark.

Usage:
    python -m eval.run_benchmark [--date 2026-03-20] [--thresholds 3.0,5.0,8.0,12.0]
    python eval/run_benchmark.py [--date 2026-03-20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.market_benchmark import (
    generate_report,
    load_market_log,
    load_observations,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market architecture benchmark — replay telemetry and measure detection."
    )
    parser.add_argument(
        "--date",
        default="2026-03-20",
        help="Date folder under data/ (default: 2026-03-20)",
    )
    parser.add_argument(
        "--thresholds",
        default="3.0,5.0,8.0,12.0",
        help="Comma-separated CUSUM thresholds to test (default: 3.0,5.0,8.0,12.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: eval/results/benchmark-{date}.md)",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / args.date
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    obs_path = data_dir / "frontier-ops-observations.jsonl"
    market_path = data_dir / "market-daemon.log"

    if not obs_path.exists():
        print(f"Error: observations file not found: {obs_path}", file=sys.stderr)
        sys.exit(1)

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    print(f"Loading observations from {obs_path} ...")
    observations = load_observations(obs_path)
    print(f"  → {len(observations)} observations loaded")

    market_entries = []
    if market_path.exists():
        print(f"Loading market log from {market_path} ...")
        market_entries = load_market_log(market_path)
        print(f"  → {len(market_entries)} market evaluations loaded")
    else:
        print(f"  ⚠ Market log not found at {market_path}, skipping agreement analysis")

    print(f"Running benchmark with thresholds: {thresholds} ...")
    result = run_benchmark(observations, market_entries, thresholds)

    report = generate_report(result, args.date)

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "eval" / "results" / f"benchmark-{args.date}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\n✅ Report saved to {output_path}")

    # Also print summary to stdout
    print("\n" + "=" * 60)
    for tr in result.threshold_results:
        print(
            f"  θ={tr.threshold:5.1f}  |  "
            f"FPR={tr.fpr:.3f}  TPR={tr.tpr:.3f}  F1={tr.f1:.3f}  |  "
            f"ARL(pass)={tr.arl_pass:6.1f}  Delay={tr.detection_delay_mean:5.1f}"
        )
    print("=" * 60)
    print(f"  Market agreement: {result.agreement_rate:.1%}")


if __name__ == "__main__":
    main()
