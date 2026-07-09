"""CLI entry: python -m eval.battery [--checks name,name] [--fetch-only]
[--no-network]. Exit 0 iff every selected check passes both its ±tolerance
comparisons and its unconditional ordering invariants."""

from __future__ import annotations

import argparse
import sys
import time

from eval.battery import load_expected
from eval.battery.checks import ALL_CHECKS
from eval.battery.data import DataUnavailable, fetch_dataset


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.battery")
    ap.add_argument(
        "--checks",
        default=",".join(ALL_CHECKS),
        help="comma-separated subset of: " + ", ".join(ALL_CHECKS),
    )
    ap.add_argument(
        "--fetch-only", action="store_true", help="fetch pinned data and exit"
    )
    ap.add_argument(
        "--no-network", action="store_true", help="fail instead of fetching"
    )
    ap.add_argument(
        "--encoder",
        default=None,
        help="Phase 5 experiment: run under an alternative sentence-transformers "
        "encoder and report deltas vs the §6 baselines (informational — the "
        "default encoder never changes)",
    )
    args = ap.parse_args()

    from eval.battery import encoders

    experiment = args.encoder is not None and args.encoder != encoders.MODEL_NAME
    if experiment:
        encoders.set_model(args.encoder)
        print(
            f"PHASE 5 ENCODER EXPERIMENT: {args.encoder} (informational; "
            f"baselines assume {encoders.MODEL_NAME})"
        )

    try:
        path = fetch_dataset(allow_network=not args.no_network)
        print(f"data: {path}")
    except DataUnavailable as e:
        print(f"SKIP: {e}")
        return 0 if not args.fetch_only else 1
    if args.fetch_only:
        return 0

    expected = load_expected()
    tol = expected["tolerance"]
    selected = [c.strip() for c in args.checks.split(",") if c.strip()]
    unknown = [c for c in selected if c not in ALL_CHECKS]
    if unknown:
        ap.error(f"unknown checks: {unknown}")

    results = []
    for name in selected:
        print(f"\n== {name} ==")
        t0 = time.time()
        results.append(
            ALL_CHECKS[name](expected["checks"][name], tol, log=print)
        )
        print(f"  [{time.time() - t0:.0f}s]")

    print("\n" + "=" * 72)
    all_ok = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        all_ok &= r.passed
        print(f"{status}  {r.name}")
        exp = expected["checks"][r.name]
        for k, v in r.values.items():
            base = exp.get(k)
            delta = f"  (baseline {base:.3f}, Δ{v - base:+.3f})" if isinstance(base, (int, float)) else ""
            print(f"      {k}: {v:.3f}{delta}")
        for fail in r.tol_failures:
            print(f"      tolerance: {fail}")
        for fail in r.invariant_failures:
            print(f"      INVARIANT: {fail}")
        for note in r.notes:
            print(f"      note: {note}")
    print("=" * 72)
    if experiment:
        print(
            f"encoder experiment ({args.encoder}): informational only — "
            "pass/fail above is relative to the MiniLM baselines"
        )
        return 0
    print("battery:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
