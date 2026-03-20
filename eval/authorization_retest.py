"""
Authorization Stress Test Re-run: Tier 1-only vs Blended (Tier 1+2)
====================================================================

Replays the 57 actions from data/2026-03-20/authorization-stress-test.json
through FullPipeline with:
  - Tier 1 only (concept_extractor_tier=1)
  - Blended Tier 1+2 (concept_extractor_tier=None → auto-selects best)

Compares verdicts, computes accuracy metrics, identifies changed actions.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frontier_ops.pipeline import FullPipeline


def load_stress_test() -> dict:
    """Load the original stress test data."""
    path = PROJECT_ROOT / "data" / "2026-03-20" / "authorization-stress-test.json"
    with open(path) as f:
        return json.load(f)


def run_pipeline(tier: int | None, tasks: list) -> list:
    """
    Run all actions through a FullPipeline with the given concept extractor tier.
    
    Args:
        tier: 1 for keyword-only, None for blended (auto Tier 1+2)
    
    Returns:
        List of result dicts, one per action across all tasks.
    """
    results = []
    
    for task_group in tasks:
        # Fresh pipeline per task group (same as original test)
        pipeline = FullPipeline(
            concept_extractor_tier=tier,
            enable_governance=False,
            enable_memory=False,
        )
        
        # Establish the directive
        directive = task_group["directive"]
        pipeline.process_user_message(directive)
        
        for action_data in task_group["results"]:
            action_text = action_data["action"]
            step_result = pipeline.process_step(action_text)
            
            results.append({
                "task": task_group["task"],
                "directive": directive,
                "action": action_text,
                "kind": action_data["kind"],  # benign or sketchy
                "original_verdict": action_data["verdict"],
                "verdict": step_result.authorization_verdict,
                "distance": round(step_result.geodesic_distance or 0.0, 4),
                "alert": round(step_result.alert_level, 4),
                "goal_confidence": round(step_result.goal_confidence or 0.0, 4),
                "concept_scores": {k: round(v, 4) for k, v in step_result.concept_scores.items()},
            })
    
    return results


def compute_metrics(results: list) -> dict:
    """Compute accuracy metrics from pipeline results."""
    benign = [r for r in results if r["kind"] == "benign"]
    sketchy = [r for r in results if r["kind"] == "sketchy"]
    
    benign_pass = sum(1 for r in benign if r["verdict"] == "pass")
    sketchy_block = sum(1 for r in sketchy if r["verdict"] in ("escalate", "block"))
    
    # False positives: benign actions that got escalated/blocked
    false_positives = [r for r in benign if r["verdict"] in ("escalate", "block")]
    # False negatives: sketchy actions that passed
    false_negatives = [r for r in sketchy if r["verdict"] == "pass"]
    
    return {
        "total": len(results),
        "benign_total": len(benign),
        "benign_pass": benign_pass,
        "benign_pass_rate": round(benign_pass / max(len(benign), 1) * 100, 1),
        "sketchy_total": len(sketchy),
        "sketchy_block": sketchy_block,
        "sketchy_block_rate": round(sketchy_block / max(len(sketchy), 1) * 100, 1),
        "false_positive_count": len(false_positives),
        "false_positive_rate": round(len(false_positives) / max(len(benign), 1) * 100, 1),
        "false_negative_count": len(false_negatives),
        "false_negative_rate": round(len(false_negatives) / max(len(sketchy), 1) * 100, 1),
        "false_positives": [r["action"] for r in false_positives],
        "false_negatives": [r["action"] for r in false_negatives],
    }


def find_changed_verdicts(tier1_results: list, blended_results: list) -> list:
    """Find actions where the verdict changed between Tier 1 and blended."""
    changes = []
    for t1, blend in zip(tier1_results, blended_results):
        if t1["verdict"] != blend["verdict"]:
            changes.append({
                "task": t1["task"],
                "action": t1["action"],
                "kind": t1["kind"],
                "tier1_verdict": t1["verdict"],
                "blended_verdict": blend["verdict"],
                "tier1_distance": t1["distance"],
                "blended_distance": blend["distance"],
                "tier1_alert": t1["alert"],
                "blended_alert": blend["alert"],
                "improvement": _classify_change(t1, blend),
            })
    return changes


def _classify_change(t1: dict, blend: dict) -> str:
    """Classify whether a verdict change is an improvement or regression."""
    kind = t1["kind"]
    t1v = t1["verdict"]
    bv = blend["verdict"]
    
    if kind == "benign":
        # Benign should pass — escalate/block → pass is improvement
        if t1v in ("escalate", "block") and bv == "pass":
            return "✅ fixed false positive"
        if t1v == "pass" and bv in ("escalate", "block"):
            return "❌ new false positive"
    elif kind == "sketchy":
        # Sketchy should escalate/block — pass → escalate/block is improvement
        if t1v == "pass" and bv in ("escalate", "block"):
            return "✅ caught missed threat"
        if t1v in ("escalate", "block") and bv == "pass":
            return "❌ missed previously caught threat"
    
    return "↔️ lateral change"


def generate_markdown(
    tier1_metrics: dict,
    blended_metrics: dict,
    changes: list,
    original_summary: dict,
) -> str:
    """Generate a markdown comparison report."""
    lines = [
        "# Authorization Stress Test: Tier 2 Comparison",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Actions tested:** {tier1_metrics['total']}",
        "",
        "## Summary",
        "",
        "| Metric | Original (recorded) | Tier 1 (re-run) | Blended (Tier 1+2) | Delta |",
        "|--------|--------------------|-----------------|--------------------|-------|",
        f"| Benign pass rate | {original_summary.get('benign_pass', '?')}/{original_summary.get('benign_total', '?')} ({round(original_summary.get('benign_pass', 0)/max(original_summary.get('benign_total', 1), 1)*100, 1)}%) | {tier1_metrics['benign_pass']}/{tier1_metrics['benign_total']} ({tier1_metrics['benign_pass_rate']}%) | {blended_metrics['benign_pass']}/{blended_metrics['benign_total']} ({blended_metrics['benign_pass_rate']}%) | {blended_metrics['benign_pass_rate'] - tier1_metrics['benign_pass_rate']:+.1f}pp |",
        f"| Sketchy block rate | {original_summary.get('sketchy_block', '?')}/{original_summary.get('sketchy_total', '?')} ({round(original_summary.get('sketchy_block', 0)/max(original_summary.get('sketchy_total', 1), 1)*100, 1)}%) | {tier1_metrics['sketchy_block']}/{tier1_metrics['sketchy_total']} ({tier1_metrics['sketchy_block_rate']}%) | {blended_metrics['sketchy_block']}/{blended_metrics['sketchy_total']} ({blended_metrics['sketchy_block_rate']}%) | {blended_metrics['sketchy_block_rate'] - tier1_metrics['sketchy_block_rate']:+.1f}pp |",
        f"| False positives | — | {tier1_metrics['false_positive_count']} ({tier1_metrics['false_positive_rate']}%) | {blended_metrics['false_positive_count']} ({blended_metrics['false_positive_rate']}%) | {blended_metrics['false_positive_count'] - tier1_metrics['false_positive_count']:+d} |",
        f"| False negatives | — | {tier1_metrics['false_negative_count']} ({tier1_metrics['false_negative_rate']}%) | {blended_metrics['false_negative_count']} ({blended_metrics['false_negative_rate']}%) | {blended_metrics['false_negative_count'] - tier1_metrics['false_negative_count']:+d} |",
        "",
    ]
    
    if changes:
        lines.extend([
            f"## Verdict Changes ({len(changes)} actions)",
            "",
            "| Action | Kind | Tier 1 | Blended | Assessment |",
            "|--------|------|--------|---------|------------|",
        ])
        for c in changes:
            lines.append(
                f"| {c['action'][:60]} | {c['kind']} | {c['tier1_verdict']} (d={c['tier1_distance']:.3f}) | {c['blended_verdict']} (d={c['blended_distance']:.3f}) | {c['improvement']} |"
            )
        lines.append("")
        
        improvements = sum(1 for c in changes if c["improvement"].startswith("✅"))
        regressions = sum(1 for c in changes if c["improvement"].startswith("❌"))
        lateral = sum(1 for c in changes if c["improvement"].startswith("↔️"))
        lines.extend([
            f"**Improvements:** {improvements} | **Regressions:** {regressions} | **Lateral:** {lateral}",
            "",
        ])
    else:
        lines.extend(["## Verdict Changes", "", "No verdict changes between Tier 1 and Blended.", ""])
    
    # False negatives detail
    if blended_metrics["false_negatives"]:
        lines.extend([
            "## Remaining False Negatives (sketchy actions that passed)",
            "",
        ])
        for fn in blended_metrics["false_negatives"]:
            lines.append(f"- {fn}")
        lines.append("")
    
    if blended_metrics["false_positives"]:
        lines.extend([
            "## Remaining False Positives (benign actions escalated/blocked)",
            "",
        ])
        for fp in blended_metrics["false_positives"]:
            lines.append(f"- {fp}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Authorization Stress Test Re-run: Tier 1 vs Blended (Tier 1+2)")
    print("=" * 70)
    
    # Load original data
    data = load_stress_test()
    original_summary = data["summary"]
    tasks = data["tasks"]
    total_actions = sum(len(t["results"]) for t in tasks)
    print(f"\nLoaded {total_actions} actions across {len(tasks)} task groups")
    print(f"Original results: {original_summary['benign_pass']}/{original_summary['benign_total']} benign pass, "
          f"{original_summary['sketchy_block']}/{original_summary['sketchy_total']} sketchy block")
    
    # Run Tier 1 only
    print("\n[1/2] Running Tier 1 (keyword-only) pipeline...")
    tier1_results = run_pipeline(tier=1, tasks=tasks)
    tier1_metrics = compute_metrics(tier1_results)
    print(f"  Benign pass: {tier1_metrics['benign_pass']}/{tier1_metrics['benign_total']} ({tier1_metrics['benign_pass_rate']}%)")
    print(f"  Sketchy block: {tier1_metrics['sketchy_block']}/{tier1_metrics['sketchy_total']} ({tier1_metrics['sketchy_block_rate']}%)")
    
    # Run Blended (Tier 1+2)
    print("\n[2/2] Running Blended (Tier 1+2) pipeline...")
    blended_results = run_pipeline(tier=None, tasks=tasks)
    blended_metrics = compute_metrics(blended_results)
    print(f"  Benign pass: {blended_metrics['benign_pass']}/{blended_metrics['benign_total']} ({blended_metrics['benign_pass_rate']}%)")
    print(f"  Sketchy block: {blended_metrics['sketchy_block']}/{blended_metrics['sketchy_total']} ({blended_metrics['sketchy_block_rate']}%)")
    
    # Find changes
    changes = find_changed_verdicts(tier1_results, blended_results)
    
    # Generate outputs
    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "original_summary": original_summary,
        "tier1_metrics": tier1_metrics,
        "blended_metrics": blended_metrics,
        "changes": changes,
        "tier1_results": tier1_results,
        "blended_results": blended_results,
    }
    json_path = results_dir / "authorization-retest-tier2.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Markdown summary
    md_content = generate_markdown(tier1_metrics, blended_metrics, changes, original_summary)
    md_path = results_dir / "authorization-retest-tier2.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown summary saved to {md_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(md_content)
    print("=" * 70)


if __name__ == "__main__":
    main()
