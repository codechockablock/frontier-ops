"""v2 eval battery — Apollo deception benchmarks (handoff Phase 2).

Run with `python -m eval.battery`. Checks live in eval.battery.checks;
expected numbers + tolerances in expected.json; data fetch (pinned commit,
gitignored, download-at-runtime) in eval.battery.data.
"""

import json
from pathlib import Path
from typing import Dict


def load_expected() -> Dict:
    with open(Path(__file__).resolve().parent / "expected.json") as f:
        return json.load(f)
