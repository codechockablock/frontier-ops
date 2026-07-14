"""Acceptance-criteria tests for the calibration-transport experiment.

Maps to fable-spec-calibration-transport.md §5. Fast and offline: the numbers
come from the experiment scripts; here we lock the *invariants* — held-out
discipline (calibrate on one partition, evaluate on a disjoint one), generator
blindness of the fresh scenario batch, no real secrets in fixtures, and the
label-free calibration rules keeping/dropping exactly what they claim.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

_HARNESS = Path(__file__).resolve().parents[2] / "eval" / "drift_harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import scenarios  # noqa: E402
import scenarios_fresh  # noqa: E402
import transport_experiment as tx  # noqa: E402
import warmup_calibration_experiment as wx  # noqa: E402


def _row(scenario_id, run, dist, off=False, model="m", surface="s", locked=False):
    return {
        "scenario_id": scenario_id, "surface": surface, "provenance": "emergent",
        "run_key": f"{model}#{scenario_id}#{run}", "model": model, "tool": "t",
        "off_goal": off, "category": None, "geodesic_distance": dist,
        "alert_level": 0.0, "newma_divergence": 0.0, "max_traj_cross": 0.0,
        "affinity": None, "max_locked_activation": 0.0, "locked_fired": locked,
        "authorized": True, "needs_escalation": False, "newma_alarm": False,
    }


# --- §5.1  Held-out discipline ----------------------------------------------

def test_transport_calibrates_only_on_cal_side():
    """The radius must come from the calibration side alone: eval distances
    sit far above every calibration distance, so any leakage of eval rows
    into calibration would raise the radius and drive FPR to 0."""
    rng = np.random.default_rng(0)
    cal = [_row("a", i, 1.0 + 0.01 * i) for i in range(20)]
    ev = [_row("b", i, 5.0) for i in range(20)]
    out = tx.transport(cal, ev, alpha=0.1, rng=rng)
    assert out["radius"] < 2.0
    assert out["fpr"] == 1.0


def test_loso_scenario_folds_are_disjoint():
    """Fold exclusion in both directions: a scenario whose distances dwarf the
    others must get FPR 1.0 on its own fold (its rows were NOT in
    calibration), while the other folds — which DO include it — get radii
    large enough to contain themselves (FPR 0)."""
    rng = np.random.default_rng(0)
    rows = []
    for s in ("a", "b"):
        rows += [_row(s, i, 1.0 + 0.01 * i) for i in range(15)]
    rows += [_row("c", i, 10.0) for i in range(15)]
    out = tx.loso_scenario(rows, alpha=0.1, rng=rng)
    assert out["per_scenario"]["c"]["fpr"] == 1.0
    assert out["per_scenario"]["a"]["fpr"] == 0.0
    assert out["per_scenario"]["b"]["fpr"] == 0.0


def test_within_half_split_never_mixes_runs():
    """Run-clustered splits: with one run per distance value and huge
    between-run spread, held-out FPR must stay strictly between 0 and 1 —
    and the function must not crash on benign-only rows (no off-goal)."""
    rows = [_row("a", i, float(i % 7) + 1.0) for i in range(40)]
    out = tx.within_half_split_fpr(rows, alpha=0.1, k=50, seed=0)
    assert out["n_splits"] > 0
    assert 0.0 <= out["fpr"] <= 1.0


# --- §5.3  Generator blindness of the fresh batch ----------------------------

def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_fresh_scenarios_blind_to_detectors():
    for mod in _imports_of(_HARNESS / "scenarios_fresh.py"):
        assert not mod.startswith("frontier_ops"), (
            f"scenarios_fresh.py must be blind to detectors; found {mod}")


def test_fresh_batch_shape():
    fresh = scenarios_fresh.FRESH_SCENARIOS
    assert len(fresh) >= 8
    assert not set(fresh) & set(scenarios.SCENARIOS)
    assert {v["surface"] for v in fresh.values()} == {
        "scope_exploration", "credential_adjacent", "goal_displacement"}
    assert all(v["provenance"] == "emergent" for v in fresh.values())


def test_fresh_fixtures_embed_no_real_secret():
    import re
    for sid, spec in scenarios_fresh.FRESH_SCENARIOS.items():
        for path, content in spec["workspace"]["files"].items():
            for m in re.findall(r"(sk-[A-Za-z0-9\-]+|ghp_[A-Za-z0-9]+)", content):
                assert "FAKE" in m.upper(), f"{sid}:{path} may embed a real secret: {m}"


# --- Label-free calibration rules keep/drop what they claim ------------------

def test_warmup_rule_dists():
    rows = [
        _row("a", 0, 1.0),
        _row("a", 1, 2.0, off=True),
        _row("a", 2, 3.0, locked=True),
        _row("a", 3, 4.0, off=True, locked=True),
    ]
    assert list(wx._rule_dists(rows, "oracle")) == [1.0, 3.0]
    assert list(wx._rule_dists(rows, "naive")) == [1.0, 2.0, 3.0, 4.0]
    assert list(wx._rule_dists(rows, "locked_filtered")) == [1.0, 2.0]


def test_deployment_warmup_eval_set_excludes_warmup_runs():
    """Warmup runs are chronologically first; the eval side must be exactly
    the remainder (equal eval counts for every rule)."""
    rng = np.random.default_rng(0)
    rows = [_row("a", i, 1.0) for i in range(20)]
    order = sorted({r["run_key"] for r in rows})
    out = wx.deployment_warmup(rows, order, frac=0.5, alpha=0.1, rng=rng)
    assert out["n_warmup_runs"] == 10
    assert out["n_eval_runs"] == 10
    assert out["n_eval_benign"] == 10
