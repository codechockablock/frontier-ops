"""Cache per-action detector rows to JSONL so recalibration analysis is instant.

Replays every record through a fresh FullPipeline (affinity omitted — the
recalibration work needs only geodesic_distance / alert_level / authorized /
needs_escalation, none of which touch the embedding model). Run once:

    python3 dump_rows.py --records ../results/drift-harness-records-<date>.jsonl \
        --out /tmp/drift_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import score  # noqa: E402
from records import load_records  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    recs = load_records(Path(args.records))
    score._attach_specs(recs)
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(recs):
            for row in score.replay_record(r, None):
                # (scenario_id, run_idx) collides across models in combined
                # record files, so qualify the cached run_key with the model.
                row["model"] = r["model"]
                row["run_key"] = f"{r['model']}#{row['run_key']}"
                fh.write(json.dumps(row, default=str) + "\n")
                n += 1
            if (i + 1) % 25 == 0:
                print(f"  replayed {i+1}/{len(recs)} records, {n} rows", flush=True)
    print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
