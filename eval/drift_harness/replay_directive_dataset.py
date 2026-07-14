"""Replay the real directive dataset through the drift-harness detector contract.

Produces benign-distance rows comparable to the harness row cache: for each
human-directive record a FRESH ``FullPipeline`` gets
``process_user_message(directive)`` then ``process_step(f"{tool}: {summary}")``
per action — the identical replay contract used by ``score.replay_record``.
Every action in this dataset is ON-GOAL ground truth (the agent did what was
asked), so these rows support FPR (false-escalation) transport measurement
only, never TPR.

The dataset itself is gitignored real personal content. The rows written here
deliberately carry NO raw text — only session/seq identifiers and detector
signals — and should still go to a scratch location, not be committed.

Usage::

    ../../.venv/bin/python replay_directive_dataset.py --out /tmp/directive_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from goal_conditioning_eval import DATA_PATH, action_text, load_records  # noqa: E402

from frontier_ops.authorization.scope import GoalConditionedMetric  # noqa: E402
from frontier_ops.pipeline import FullPipeline  # noqa: E402

_LOCKED_DIMS = sorted(GoalConditionedMetric.LOCKED_DIMS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(DATA_PATH))
    args = p.parse_args()

    records = load_records(Path(args.data))
    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            pipe = FullPipeline()  # fresh per record: detector state is session-scoped
            pipe.process_user_message(rec["directive"])
            for act in rec["actions"]:
                r = pipe.process_step(action_text(act))
                row = {
                    "dataset": "directive-2026-07-02",
                    "session": rec.get("session"),
                    "run_key": f"directive#{rec.get('session')}#{rec.get('seq', i)}",
                    "tool": act.get("tool", ""),
                    "off_goal": False,  # dataset ground truth: all actions on-goal
                    "geodesic_distance": r.geodesic_distance,
                    "alert_level": r.alert_level,
                    "newma_divergence": r.newma_divergence,
                    "authorized": r.authorized,
                    "needs_escalation": bool(r.needs_escalation),
                    "max_locked_activation": max(
                        r.concept_scores.get(d, 0.0) for d in _LOCKED_DIMS
                    ),
                    "locked_fired": any(
                        r.concept_scores.get(d, 0.0)
                        >= pipe._boundary_thresholds.get(d, float("inf"))
                        for d in _LOCKED_DIMS
                    ),
                }
                fh.write(json.dumps(row, default=str) + "\n")
                n += 1
            if (i + 1) % 25 == 0:
                print(f"  replayed {i + 1}/{len(records)} records, {n} rows", flush=True)
    print(f"wrote {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
