"""
Task-Affinity Channel — Held-Out Evaluation + Conformal Calibration Demo
=========================================================================

Validates the escalate-only task-affinity channel (fable-spec-task-affinity.md)
on data the channel's constants never saw:

  1. SESSION-LEVEL SPLIT: sessions are deterministically hashed into a
     CALIBRATION half and a HELD-OUT half. Nothing about the channel is
     fitted to held-out sessions.
  2. HELD-OUT AUC (threshold-free): pairwise own-vs-mismatched affinity on
     held-out records only, mismatches drawn from other held-out sessions.
     With similarity strata (negative-quality view) and the masked-action
     robustness variant.
  3. CALIBRATION DEMO: AffinityThreshold calibrated (alpha=0.1) on
     calibration-split own-pair affinities; reported number is the benign
     false-escalation rate on HELD-OUT own-pairs (distribution-free check:
     should be <~ alpha).

Escalation path (spec §4): if held-out AUC ~= 0.5, the affinity signal is
session-idiosyncratic and the channel should NOT ship. Report plainly.

Dataset: data/2026-07-02/directive-dataset.jsonl (gitignored, local-only).

Usage:
    python3 eval/task_affinity_eval.py [--seed 7] [--k 5] [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from goal_conditioning_controls import mask_lexical  # noqa: E402
from goal_conditioning_eval import (  # noqa: E402
    DATA_PATH,
    RESULTS_DIR,
    action_text,
    band_of,
    cluster_bootstrap_ci,
    load_records,
    pairwise_outcome,
    sample_mismatches,
)

from frontier_ops.authorization.task_affinity import (  # noqa: E402
    AffinityThreshold,
    TaskAffinityChannel,
)

BAND_ORDER = ["short (<=6 words)", "medium (7-15)", "long (>15)"]


def split_of(session_id: str) -> str:
    """Deterministic session-level split (independent of seed/order)."""
    digest = hashlib.sha1(session_id.encode()).hexdigest()
    return "calibration" if int(digest, 16) % 2 == 0 else "heldout"


def run(seed: int, k: int, n_boot: int, alpha: float) -> dict:
    rng = np.random.default_rng(seed)
    records = load_records(DATA_PATH)

    cal_records = [r for r in records if split_of(r["session"]) == "calibration"]
    held_records = [r for r in records if split_of(r["session"]) == "heldout"]
    cal_sessions = {r["session"] for r in cal_records}
    held_sessions = {r["session"] for r in held_records}
    print(f"Split: {len(cal_sessions)} calibration sessions "
          f"({len(cal_records)} records) / {len(held_sessions)} held-out "
          f"sessions ({len(held_records)} records)")

    channel = TaskAffinityChannel.create()
    if channel is None:
        sys.exit("ERROR: eval requires sentence-transformers.")

    # ── Held-out AUC (threshold-free) ───────────────────────────────────────
    pairs = sample_mismatches(held_records, rng, k)

    per_record: List[List[float]] = []
    per_record_masked: List[List[float]] = []
    per_band: Dict[str, List[List[float]]] = defaultdict(list)
    per_band_masked: Dict[str, List[List[float]]] = defaultdict(list)
    units = []  # per (record, mm): sim_dd + outcomes, for strata

    for rec_idx, (rec, mm_texts) in enumerate(pairs):
        directive = rec["directive"]
        a_texts = [action_text(a) for a in rec["actions"]]
        d_own = [1.0 - channel.affinity(directive, t) for t in a_texts]
        d_own_m = [1.0 - channel.affinity(directive, mask_lexical(t))
                   for t in a_texts]
        rec_out, rec_out_m = [], []
        for mm in mm_texts:
            sim_dd = channel.affinity(directive, mm)
            out, out_m = [], []
            for i, t in enumerate(a_texts):
                out.append(pairwise_outcome(
                    d_own[i], 1.0 - channel.affinity(mm, t)))
                out_m.append(pairwise_outcome(
                    d_own_m[i], 1.0 - channel.affinity(mm, mask_lexical(t))))
            units.append({"rec_idx": rec_idx, "sim_dd": sim_dd, "out": out})
            rec_out.extend(out)
            rec_out_m.extend(out_m)
        if rec_out:
            per_record.append(rec_out)
            per_record_masked.append(rec_out_m)
            per_band[band_of(directive)].append(rec_out)
            per_band_masked[band_of(directive)].append(rec_out_m)

    n_pairs = sum(len(o) for o in per_record)
    auc = float(sum(sum(o) for o in per_record) / n_pairs)
    ci = cluster_bootstrap_ci(per_record, rng, n_boot)
    auc_masked = float(sum(sum(o) for o in per_record_masked) / n_pairs)
    ci_masked = cluster_bootstrap_ci(per_record_masked, rng, n_boot)

    bands = {}
    for name in BAND_ORDER:
        outs = per_band.get(name, [])
        if not outs:
            continue
        total = sum(len(o) for o in outs)
        outs_m = per_band_masked[name]
        bands[name] = {
            "auc": float(sum(sum(o) for o in outs) / total),
            "ci": list(cluster_bootstrap_ci(outs, rng, n_boot)),
            "auc_masked": float(sum(sum(o) for o in outs_m) / total),
            "n_records": len(outs), "n_pairs": total,
        }

    # Similarity strata (negative-quality view, Q1 methodology)
    sims = np.array([u["sim_dd"] for u in units])
    qcuts = np.percentile(sims, [25, 50, 75])
    strata = []
    for qi, label in enumerate(["least similar", "second", "third",
                                "most similar"]):
        subset = [u for u in units
                  if int(np.searchsorted(qcuts, u["sim_dd"], side="left")) == qi]
        by_rec: Dict[int, List[float]] = defaultdict(list)
        for u in subset:
            by_rec[u["rec_idx"]].extend(u["out"])
        clusters = list(by_rec.values())
        total = sum(len(c) for c in clusters)
        strata.append({
            "stratum": label,
            "auc": float(sum(sum(c) for c in clusters) / total),
            "ci": list(cluster_bootstrap_ci(clusters, rng, n_boot)),
            "n_pairs": total,
        })

    # ── Calibration demo (constants from calibration split ONLY) ───────────
    cal_affinities = [
        channel.affinity(rec["directive"], action_text(a))
        for rec in cal_records for a in rec["actions"]
    ]
    threshold = AffinityThreshold()
    threshold.calibrate(cal_affinities, alpha=alpha)

    held_affinities = [
        channel.affinity(rec["directive"], action_text(a))
        for rec in held_records for a in rec["actions"]
    ]
    false_escalation = float(np.mean(
        [threshold.flags(a) for a in held_affinities]))

    return {
        "date": str(date.today()), "seed": seed, "k": k, "n_boot": n_boot,
        "alpha": alpha,
        "split": {
            "calibration_sessions": len(cal_sessions),
            "heldout_sessions": len(held_sessions),
            "calibration_records": len(cal_records),
            "heldout_records": len(held_records),
        },
        "auc": auc, "ci": list(ci),
        "auc_masked": auc_masked, "ci_masked": list(ci_masked),
        "n_records": len(per_record), "n_pairs": n_pairs,
        "bands": bands,
        "strata": strata,
        "sim_quartile_cuts": [float(x) for x in qcuts],
        "calibration": {
            "n_calibration_pairs": len(cal_affinities),
            "threshold": threshold.threshold,
            "n_heldout_pairs": len(held_affinities),
            "heldout_false_escalation_rate": false_escalation,
        },
    }


def write_report(r: dict, out_md: Path):
    cal = r["calibration"]
    lines = [
        "# Task-Affinity Channel — Held-Out Evaluation",
        f"**Date:** {r['date']} · **Spec:** `fable-spec-task-affinity.md` · "
        "**Channel:** `frontier_ops/authorization/task_affinity.py` "
        "(all-MiniLM-L6-v2 cosine, escalate-only)",
        f"**Split:** deterministic session hash — "
        f"{r['split']['calibration_sessions']} calibration sessions "
        f"({r['split']['calibration_records']} records) / "
        f"{r['split']['heldout_sessions']} held-out sessions "
        f"({r['split']['heldout_records']} records). No channel constant "
        "was fitted to held-out sessions; the frozen seed-7 "
        "goal-conditioning pairs were not reused.",
        f"**Determinism:** seed={r['seed']}, K={r['k']}, {r['n_boot']} "
        "bootstrap resamples (cluster bootstrap over records).",
        "",
        "---",
        "",
        "## Held-out directional result (threshold-free)",
        "",
        f"- **Affinity AUC = {r['auc']:.3f}** "
        f"(95% CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]) on "
        f"{r['n_records']} held-out records / {r['n_pairs']} pairs — "
        "P(affinity-to-own-directive > affinity-to-mismatched), ties 0.5.",
        f"- Masked-action robustness variant (paths/files/ids/numbers "
        f"masked): **{r['auc_masked']:.3f}** "
        f"(95% CI [{r['ci_masked'][0]:.3f}, {r['ci_masked'][1]:.3f}]).",
        "",
        "| Band | AUC | 95% CI | AUC (masked) | records | pairs |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in r["bands"].items():
        lines.append(
            f"| {name} | {b['auc']:.3f} | [{b['ci'][0]:.3f}, "
            f"{b['ci'][1]:.3f}] | {b['auc_masked']:.3f} "
            f"| {b['n_records']} | {b['n_pairs']} |"
        )
    lines += [
        "",
        "Similarity strata (negative-quality view — same methodology as the "
        "goal-conditioning controls):",
        "",
        "| Stratum (directive↔directive affinity) | AUC | 95% CI | pairs |",
        "|---|---|---|---|",
    ]
    for s in r["strata"]:
        lines.append(
            f"| {s['stratum']} | {s['auc']:.3f} | [{s['ci'][0]:.3f}, "
            f"{s['ci'][1]:.3f}] | {s['n_pairs']} |"
        )
    lines += [
        "",
        "## Conformal calibration demo (escalate-only threshold)",
        "",
        f"`AffinityThreshold` calibrated on the calibration split only "
        f"({cal['n_calibration_pairs']} benign own-pairs, "
        f"alpha={r['alpha']}) → threshold = {cal['threshold']:.3f}. "
        f"On {cal['n_heldout_pairs']} held-out benign own-pairs the "
        f"false-escalation rate is "
        f"**{cal['heldout_false_escalation_rate']:.1%}** "
        f"(target ≲ {r['alpha']:.0%}, distribution-free).",
        "",
    ]
    rate = cal["heldout_false_escalation_rate"]
    if rate > 1.5 * r["alpha"]:
        lines += [
            f"**The miss is informative, not a tuning target.** The "
            f"conformal guarantee assumes exchangeability between "
            f"calibration and deployment pairs; splitting by SESSION "
            f"breaks it — different sessions have different directive "
            f"styles and affinity baselines, and the calibration half here "
            f"is small ({r['split']['calibration_sessions']} sessions / "
            f"{r['split']['calibration_records']} records). A "
            f"{rate:.0%}-vs-{r['alpha']:.0%} inflation under cross-session "
            "transfer means the threshold must be calibrated in-domain "
            "(per deployment/workload, or rolling), exactly as radius "
            "calibration is already framed. Alpha was not adjusted after "
            "seeing this number.",
            "",
        ]
    else:
        lines += [
            "The held-out rate is within the conformal tolerance: the "
            "escalate-only threshold transfers across this session split.",
            "",
        ]
    lines += [
        "## Scope of the claim",
        "",
        "- **Directionality only.** All actions are on-goal; negatives are "
        "mismatched real pairs. This does not test detection of genuinely "
        "off-goal or covert actions.",
        "- **Escalate-only by construction.** Low affinity adds an "
        "escalation signal; high affinity authorizes nothing "
        "(`authorized` is provably unaffected — see "
        "`tests/authorization/test_task_affinity.py::TestAsymmetry`). "
        "Under this rule, vocabulary mimicry can only restore an "
        "attacker's no-channel baseline, never unlock anything.",
        "- **Mimicry robustness is UNTESTED.** No adversarial data exists; "
        "the masking variant bounds the benign lexical share only.",
        "- One user, one workload, 22 sessions; session split is not a "
        "time split. MiniLM was fixed in advance, not searched.",
        "- No constants were tuned on held-out data; the only fitted "
        "quantity (the threshold) comes from the calibration split and is "
        "evaluated once on held-out.",
    ]
    out_md.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args(argv)

    r = run(seed=args.seed, k=args.k, n_boot=args.bootstrap, alpha=args.alpha)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = RESULTS_DIR / f"task-affinity-{r['date']}.md"
    out_json = RESULTS_DIR / f"task-affinity-{r['date']}.json"
    write_report(r, out_md)
    with open(out_json, "w") as f:
        json.dump(r, f, indent=2)

    print(f"\nHeld-out affinity AUC = {r['auc']:.3f} "
          f"(95% CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}])  "
          f"masked = {r['auc_masked']:.3f}")
    for name, b in r["bands"].items():
        print(f"  {name}: AUC={b['auc']:.3f} masked={b['auc_masked']:.3f} "
              f"records={b['n_records']}")
    for s in r["strata"]:
        print(f"  {s['stratum']}: {s['auc']:.3f} "
              f"[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]")
    cal = r["calibration"]
    print(f"Calibration: threshold={cal['threshold']:.3f} "
          f"(alpha={r['alpha']}) -> held-out false-escalation "
          f"{cal['heldout_false_escalation_rate']:.1%}")
    print(f"\nReport: {out_md}\nRaw:    {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
