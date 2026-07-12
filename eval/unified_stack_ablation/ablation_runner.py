"""Baseline run of unified-stack's three-dataset eval, keeping per-trace
fired-step attribution (which the stock summary discards). Writes to the
scratchpad only — leaves the unified-stack tree untouched."""

import json
import sys
import time
from pathlib import Path

US = Path("/Users/joseph/unified-stack")
sys.path.insert(0, str(US))

from eval.run_evaluation import make_default_pipeline, run_trace  # noqa: E402

OUT = Path(__file__).parent / "ablation_baseline.json"
ROOT = US / "eval" / "traces"
DATASETS = {
    "internal": ROOT,
    "thinking_augmented": ROOT / "thinking_augmented",
    "atbench": ROOT / "external" / "atbench",
    "agentdojo": ROOT / "external" / "agentdojo",
}


def load_dataset(base: Path):
    traces = []
    for sub in [base / "benign", base / "adversarial"]:
        for fp in sorted(sub.glob("*.json")):
            with open(fp) as f:
                traces.append(json.load(f))
    return traces


def main() -> None:
    out = {}
    for name, base in DATASETS.items():
        traces = load_dataset(base)
        print(f"=== {name}: {len(traces)} traces ===", flush=True)
        t0 = time.time()
        rows = []
        for i, trace in enumerate(traces, 1):
            res = run_trace(trace, make_default_pipeline)
            fired = sorted(
                {
                    s["verdict_reason"]
                    for s in res["step_results"]
                    if s["verdict"] in ("flag", "block")
                }
            )
            rows.append(
                dict(
                    trace_id=res["trace_id"],
                    category=res["category"],
                    fired_steps=fired,
                )
            )
            if i % 100 == 0 or i == len(traces):
                print(f"  {i}/{len(traces)} ({time.time()-t0:.0f}s)", flush=True)
        out[name] = rows
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"written: {OUT}", flush=True)


if __name__ == "__main__":
    main()
