"""Top-level grid driver: run (scenario × N) real agents, write records.

Writes provenance-labeled JSONL records (one per successful run) plus a run
manifest capturing model id, seeds/timestamps, and per-run stop reasons. No
labeling or detector scoring happens here — that is score.py's job, kept
separate so the expensive real-agent runs never need to be repeated to
re-score.

Usage::

    python3 run_all.py --model claude-haiku-4-5-20251001 --n 10 --concurrency 5
    python3 run_all.py --scenarios summarize_env,count_todos --n 3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scenarios import SCENARIOS  # noqa: E402
from scenarios_fresh import FRESH_SCENARIOS  # noqa: E402
from runner import run_one  # noqa: E402

# The default grid (no --scenarios) remains the ORIGINAL set; fresh-batch ids
# must be named explicitly so documented commands keep their meaning.
ALL_SCENARIOS = {**SCENARIOS, **FRESH_SCENARIOS}
from records import build_record, assert_writable  # noqa: E402

RESULTS_DIR = _HERE.parent / "results"


def _one(scenario_id: str, run_idx: int, model: str, budget: float,
         timeout: int) -> Dict[str, Any]:
    res = run_one(scenario_id, run_idx, model=model,
                  max_budget_usd=budget, timeout_s=timeout)
    spec = ALL_SCENARIOS[scenario_id]
    record = None
    writable = False
    if res["tool_calls"]:
        record = build_record(
            scenario=spec,
            run_idx=run_idx,
            model=model,
            session_id=res["session_id"],
            tool_calls=res["tool_calls"],
            final_text=res["final_text"],
            stop_reason=res["stop_reason"],
            started_ts=res["started_ts"],
            duration_s=res["duration_s"],
            cli_returncode=res["cli_returncode"],
        )
        try:
            assert_writable(record)
            writable = True
        except Exception:
            writable = False
    return {"res": res, "record": record, "writable": writable}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--n", type=int, default=10, help="runs per scenario")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--budget", type=float, default=0.40)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--scenarios", default="", help="comma list; default = all")
    p.add_argument("--out", default="", help="records JSONL path override")
    args = p.parse_args()

    date = _dt.date.today().isoformat()
    ids = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios else list(SCENARIOS.keys())
    )
    for sid in ids:
        if sid not in ALL_SCENARIOS:
            sys.exit(f"unknown scenario: {sid}")

    model_slug = args.model.replace("claude-", "").replace("-", "")[:12]
    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"drift-harness-records-{date}-{model_slug}.jsonl"
    )
    manifest_path = out_path.with_suffix(".manifest.json")

    jobs = [(sid, i) for sid in ids for i in range(args.n)]
    print(f"[run_all] {len(ids)} scenarios × N={args.n} = {len(jobs)} runs "
          f"| model={args.model} | concurrency={args.concurrency}")

    records: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(_one, sid, i, args.model, args.budget, args.timeout): (sid, i)
            for sid, i in jobs
        }
        for fut in as_completed(futs):
            sid, i = futs[fut]
            done += 1
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{done}/{len(jobs)}] {sid}#{i} ERROR: {e}")
                manifest.append({"scenario_id": sid, "run_idx": i, "error": str(e)})
                continue
            res = out["res"]
            manifest.append({
                "scenario_id": sid, "run_idx": i, "model": args.model,
                "session_id": res["session_id"], "started_ts": res["started_ts"],
                "duration_s": res["duration_s"], "cli_returncode": res["cli_returncode"],
                "stop_reason": res["stop_reason"], "n_tool_calls": res["n_tool_calls"],
                "cost_usd": res["cost_usd"], "writable": out["writable"],
            })
            if out["writable"]:
                records.append(out["record"])
            tools = [c.get("tool") for c in res["tool_calls"]]
            print(f"[{done}/{len(jobs)}] {sid}#{i} rc={res['cli_returncode']} "
                  f"stop={res['stop_reason']} tools={tools}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            assert_writable(rec)
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    manifest_path.write_text(
        json.dumps({"model": args.model, "n_per_scenario": args.n,
                    "date": date, "runs": manifest}, indent=2),
        encoding="utf-8",
    )

    total_cost = sum(m.get("cost_usd") or 0 for m in manifest)
    print(f"\n[run_all] wrote {len(records)} records -> {out_path}")
    print(f"[run_all] manifest -> {manifest_path}")
    print(f"[run_all] ~${total_cost:.2f} equivalent from subscription allowance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
