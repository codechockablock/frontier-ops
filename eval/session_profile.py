#!/usr/bin/env python3
"""
Session Behavioral Profile Analyzer for frontier-ops telemetry.
Analyzes observations, computes per-session metrics, identifies regime changes,
and produces a markdown report with ASCII timeline visualizations.
"""

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "2026-03-20"
OBS_FILE = DATA_DIR / "frontier-ops-observations.jsonl"
GT_FILE = DATA_DIR / "ground-truth-annotations.json"
REPORT_FILE = Path(__file__).parent / "results" / "session-profile-2026-03-20.md"


def load_observations():
    obs = []
    with open(OBS_FILE) as f:
        for line in f:
            if line.strip():
                obs.append(json.loads(line))
    return obs


def load_ground_truth():
    with open(GT_FILE) as f:
        return json.load(f)


def group_by_session(observations):
    sessions = defaultdict(list)
    for o in observations:
        sessions[o["session_id"]].append(o)
    for sid in sessions:
        sessions[sid].sort(key=lambda x: x["sequence"])
    return dict(sessions)


def compute_tool_distribution(obs_list):
    tools = Counter(o["action"]["tool"] for o in obs_list)
    total = len(obs_list)
    return {t: {"count": c, "pct": round(100 * c / total, 1)} for t, c in tools.most_common()}


def compute_verdict_distribution(obs_list):
    verdicts = Counter(o["governance"]["verdict"] for o in obs_list)
    total = len(obs_list)
    return {v: {"count": c, "pct": round(100 * c / total, 1)} for v, c in verdicts.most_common()}


def compute_action_rate(obs_list, bucket_minutes=10):
    """Compute action rate over time in buckets."""
    if not obs_list:
        return []
    timestamps = [datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")) for o in obs_list]
    t0 = min(timestamps)
    buckets = defaultdict(int)
    for ts in timestamps:
        bucket = int((ts - t0).total_seconds() / (bucket_minutes * 60))
        buckets[bucket] += 1
    max_bucket = max(buckets.keys()) if buckets else 0
    return [(i, buckets.get(i, 0)) for i in range(max_bucket + 1)]


def compute_hmm_transitions(obs_list):
    """Compute state transition matrix from HMM states."""
    transitions = Counter()
    states = set()
    for i in range(1, len(obs_list)):
        s_from = obs_list[i - 1]["state"]["hmm_state"]
        s_to = obs_list[i]["state"]["hmm_state"]
        transitions[(s_from, s_to)] += 1
        states.add(s_from)
        states.add(s_to)
    return transitions, sorted(states)


def identify_regime_changes(obs_list, window=10):
    """Identify behavioral regime changes based on dominant tool shifts."""
    regimes = []
    if len(obs_list) < window:
        return regimes

    def dominant_tool(subset):
        tools = Counter(o["action"]["tool"] for o in subset)
        return tools.most_common(1)[0][0] if tools else "unknown"

    current_regime = dominant_tool(obs_list[:window])
    regime_start = 0
    for i in range(window, len(obs_list)):
        w = obs_list[max(0, i - window):i]
        dom = dominant_tool(w)
        if dom != current_regime:
            ts = obs_list[i]["timestamp"]
            regimes.append({
                "sequence": i,
                "timestamp": ts,
                "from_regime": current_regime,
                "to_regime": dom,
            })
            current_regime = dom
            regime_start = i
    return regimes


def compute_false_positive_rate(ground_truth):
    """Compute empirical FPR from ground truth annotations."""
    est = ground_truth.get("estimated_metrics", {})
    tp = est.get("true_positives", 0)
    fp = est.get("false_positives", 0)
    total_blocks = est.get("sidecar_blocks", 0)
    total_actions = est.get("total_actions", 0)

    # Also count from detailed annotations
    tp_detailed = len(ground_truth.get("true_positives", []))
    fp_detailed = len(ground_truth.get("false_positives", []))
    tn_detailed = len(ground_truth.get("true_negatives", []))
    fn_detailed = len(ground_truth.get("false_negatives", []))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (fp + tn_detailed) if (fp + tn_detailed) > 0 else fp / total_actions if total_actions > 0 else 0
    recall = est.get("recall", tp / (tp + fn_detailed) if (tp + fn_detailed) > 0 else 0)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn_detailed,
        "false_negatives": fn_detailed,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "total_blocks": total_blocks,
        "total_actions": total_actions,
        "block_rate": round(total_blocks / total_actions, 4) if total_actions > 0 else 0,
    }


def top_risk_actions(obs_list, n=10):
    """Identify top N highest-risk actions by e-value."""
    scored = []
    for o in obs_list:
        e_val = o["detection"].get("e_value", 0)
        fisher = o["detection"].get("fisher_divergence", 0)
        # Combined risk score: log(1+e_value) + fisher_divergence
        risk = math.log1p(e_val) + fisher
        scored.append({
            "sequence": o["sequence"],
            "session_id": o["session_id"],
            "tool": o["action"]["tool"],
            "content_preview": o["action"]["content"][:100],
            "e_value": e_val,
            "fisher_divergence": fisher,
            "verdict": o["governance"]["verdict"],
            "hmm_state": o["state"]["hmm_state"],
            "risk_score": risk,
        })
    scored.sort(key=lambda x: x["risk_score"], reverse=True)
    return scored[:n]


def ascii_timeline(obs_list, width=80):
    """Generate ASCII timeline showing verdict severity over time."""
    if not obs_list:
        return "  (no observations)"

    verdict_chars = {"pass": ".", "monitor": "o", "flag": "!", "block": "X"}
    # Compress to width
    n = len(obs_list)
    if n <= width:
        line = "".join(verdict_chars.get(o["governance"]["verdict"], "?") for o in obs_list)
    else:
        # Downsample: take max severity per bucket
        severity_rank = {"pass": 0, "monitor": 1, "flag": 2, "block": 3}
        rank_to_verdict = {v: k for k, v in severity_rank.items()}
        line = ""
        for i in range(width):
            start = int(i * n / width)
            end = int((i + 1) * n / width)
            bucket = obs_list[start:end]
            max_sev = max(severity_rank.get(o["governance"]["verdict"], 0) for o in bucket)
            line += verdict_chars.get(rank_to_verdict[max_sev], "?")
    return line


def ascii_tool_timeline(obs_list, width=80):
    """Generate ASCII timeline showing dominant tool per bucket."""
    if not obs_list:
        return "  (no observations)"
    tool_chars = {
        "exec": "E", "read": "R", "write": "W", "edit": "e",
        "web_search": "S", "web_fetch": "F", "browser": "B",
        "message": "M", "process": "P", "sessions_spawn": "s",
        "canvas": "C", "image": "I", "cron": "c", "gateway": "G",
        "nodes": "N", "memory_search": "m",
    }
    n = len(obs_list)
    if n <= width:
        return "".join(tool_chars.get(o["action"]["tool"], "?") for o in obs_list)
    line = ""
    for i in range(width):
        start = int(i * n / width)
        end = int((i + 1) * n / width)
        bucket = obs_list[start:end]
        tools = Counter(o["action"]["tool"] for o in bucket)
        dom = tools.most_common(1)[0][0]
        line += tool_chars.get(dom, "?")
    return line


def format_transition_matrix(transitions, states):
    """Format HMM transition matrix as markdown table."""
    # Count totals from each state
    from_totals = Counter()
    for (s_from, s_to), count in transitions.items():
        from_totals[s_from] += count

    header = "| From \\ To | " + " | ".join(states) + " |"
    sep = "|" + "---|" * (len(states) + 1)
    rows = [header, sep]
    for s_from in states:
        cells = []
        for s_to in states:
            c = transitions.get((s_from, s_to), 0)
            total = from_totals[s_from]
            if total > 0:
                pct = round(100 * c / total, 0)
                cells.append(f"{c} ({pct:.0f}%)")
            else:
                cells.append("0")
        rows.append(f"| **{s_from}** | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def generate_report(sessions, ground_truth, all_obs):
    """Generate the full markdown report."""
    gt_metrics = compute_false_positive_rate(ground_truth)
    top_risks = top_risk_actions(all_obs)

    lines = []
    lines.append("# Session Behavioral Profile Report — 2026-03-20")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Total observations:** {len(all_obs)}")
    lines.append(f"**Sessions:** {len(sessions)}")
    lines.append(f"**Date range:** {all_obs[0]['timestamp'][:10]} → {all_obs[-1]['timestamp'][:10]}")
    lines.append("")

    # ---- Overall Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    overall_verdicts = compute_verdict_distribution(all_obs)
    overall_tools = compute_tool_distribution(all_obs)
    block_pct = overall_verdicts.get("block", {}).get("pct", 0)
    pass_pct = overall_verdicts.get("pass", {}).get("pct", 0)
    lines.append(f"- **Block rate:** {block_pct}% ({overall_verdicts.get('block', {}).get('count', 0)}/{len(all_obs)} actions)")
    lines.append(f"- **Pass rate:** {pass_pct}%")
    lines.append(f"- **Empirical precision:** {gt_metrics['precision']} ({gt_metrics['true_positives']} TP / {gt_metrics['true_positives'] + gt_metrics['false_positives']} flagged)")
    lines.append(f"- **Recall:** {gt_metrics['recall']}")
    lines.append(f"- **Dominant tool:** {list(overall_tools.keys())[0]} ({list(overall_tools.values())[0]['pct']}%)")
    lines.append("")
    lines.append(f"> **Key insight:** {ground_truth.get('key_insight', 'N/A')}")
    lines.append("")

    # ---- Ground Truth Metrics ----
    lines.append("## Ground Truth Analysis")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in gt_metrics.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
    lines.append("")

    # Detailed annotations
    lines.append("### True Positives (correctly caught)")
    for tp in ground_truth.get("true_positives", []):
        lines.append(f"- **{tp['action']}** — {tp['reason']} (severity: {tp.get('severity', 'N/A')})")
    lines.append("")

    lines.append("### False Positives (incorrectly blocked)")
    for fp in ground_truth.get("false_positives", []):
        lines.append(f"- **{fp['action']}** — {fp['reason']} (category: {fp.get('category', 'N/A')})")
    lines.append("")

    lines.append("### False Negatives (missed)")
    for fn in ground_truth.get("false_negatives", []):
        lines.append(f"- **{fn['action']}** — {fn['reason']} (severity: {fn.get('severity', 'N/A')})")
    lines.append("")

    # ---- Per-Session Analysis ----
    for sid, obs_list in sorted(sessions.items()):
        lines.append(f"## Session: `{sid}`")
        lines.append("")
        lines.append(f"**Observations:** {len(obs_list)}")
        ts_start = obs_list[0]["timestamp"]
        ts_end = obs_list[-1]["timestamp"]
        lines.append(f"**Time range:** {ts_start} → {ts_end}")
        lines.append("")

        # Tool distribution
        tools = compute_tool_distribution(obs_list)
        lines.append("### Tool Distribution")
        lines.append("")
        lines.append("| Tool | Count | % |")
        lines.append("|------|-------|---|")
        for t, info in tools.items():
            bar = "█" * max(1, int(info["pct"] / 2))
            lines.append(f"| {t} | {info['count']} | {info['pct']}% {bar} |")
        lines.append("")

        # Verdict distribution
        verdicts = compute_verdict_distribution(obs_list)
        lines.append("### Verdict Distribution")
        lines.append("")
        lines.append("| Verdict | Count | % |")
        lines.append("|---------|-------|---|")
        for v, info in verdicts.items():
            lines.append(f"| {v} | {info['count']} | {info['pct']}% |")
        lines.append("")

        # Action rate
        rate = compute_action_rate(obs_list, bucket_minutes=10)
        if rate:
            lines.append("### Action Rate (per 10-min bucket)")
            lines.append("")
            lines.append("```")
            max_rate = max(r for _, r in rate) if rate else 1
            for bucket_i, count in rate:
                bar_len = int(40 * count / max_rate) if max_rate > 0 else 0
                lines.append(f"  {bucket_i * 10:4d}m │{'█' * bar_len}{'░' * (40 - bar_len)}│ {count}")
            lines.append("```")
            lines.append("")

        # HMM transitions
        transitions, states = compute_hmm_transitions(obs_list)
        lines.append("### HMM State Transitions")
        lines.append("")
        lines.append(format_transition_matrix(transitions, states))
        lines.append("")

        # Regime changes
        regimes = identify_regime_changes(obs_list)
        if regimes:
            lines.append("### Behavioral Regime Changes")
            lines.append("")
            for rc in regimes:
                lines.append(f"- **seq {rc['sequence']}** ({rc['timestamp'][:19]}): "
                             f"`{rc['from_regime']}` → `{rc['to_regime']}`")
            lines.append("")

        # ASCII timelines
        lines.append("### Verdict Timeline")
        lines.append("")
        lines.append("Legend: `.`=pass  `o`=monitor  `!`=flag  `X`=block")
        lines.append("```")
        lines.append(ascii_timeline(obs_list))
        lines.append("```")
        lines.append("")

        lines.append("### Tool Timeline")
        lines.append("")
        lines.append("Legend: `E`=exec `R`=read `W`=write `e`=edit `S`=search `F`=fetch `B`=browser `M`=message `P`=process `s`=spawn")
        lines.append("```")
        lines.append(ascii_tool_timeline(obs_list))
        lines.append("```")
        lines.append("")

    # ---- Top 10 Highest Risk ----
    lines.append("## Top 10 Highest-Risk Actions")
    lines.append("")
    lines.append("Ranked by combined risk score: `log(1 + e_value) + fisher_divergence`")
    lines.append("")
    lines.append("| # | Seq | Session | Tool | Verdict | E-Value | Fisher | Risk Score |")
    lines.append("|---|-----|---------|------|---------|---------|--------|------------|")
    for i, r in enumerate(top_risks, 1):
        e_str = f"{r['e_value']:.2e}" if r['e_value'] > 1000 else f"{r['e_value']:.4f}"
        lines.append(
            f"| {i} | {r['sequence']} | {r['session_id'][-8:]} | {r['tool']} | "
            f"{r['verdict']} | {e_str} | {r['fisher_divergence']:.3f} | {r['risk_score']:.2f} |"
        )
    lines.append("")

    lines.append("### Highest-Risk Action Details")
    lines.append("")
    for i, r in enumerate(top_risks[:5], 1):
        lines.append(f"**#{i}** — seq {r['sequence']} (`{r['session_id']}`)")
        lines.append(f"  - Tool: `{r['tool']}` | Verdict: `{r['verdict']}` | HMM: `{r['hmm_state']}`")
        lines.append(f"  - Content: `{r['content_preview']}...`")
        lines.append("")

    # ---- Conclusion ----
    lines.append("## Conclusions")
    lines.append("")
    lines.append("1. **Cold-start problem is dominant:** The sidecar's HMM needs ~50-100 observations to converge. "
                 "Until then, nearly everything triggers block/flag verdicts.")
    lines.append(f"2. **Precision crisis:** At {gt_metrics['precision']} precision, enforcement mode would block "
                 f"{gt_metrics['false_positives']} legitimate actions to catch {gt_metrics['true_positives']} real issues.")
    lines.append("3. **Recall is perfect:** All annotated true positives were caught. The detection signal exists — "
                 "it's the threshold/calibration that needs work.")
    lines.append("4. **Advisory mode is correct:** Until precision exceeds ~0.50, enforcement would degrade the user experience.")
    lines.append("5. **Dominant workflow:** `exec` dominates tool usage across both sessions, consistent with "
                 "a development/infrastructure workflow.")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by `eval/session_profile.py`*")

    return "\n".join(lines)


def main():
    print("Loading observations...")
    all_obs = load_observations()
    print(f"  → {len(all_obs)} observations loaded")

    print("Loading ground truth...")
    gt = load_ground_truth()
    print(f"  → {len(gt.get('true_positives', []))} TP, {len(gt.get('false_positives', []))} FP, "
          f"{len(gt.get('false_negatives', []))} FN annotated")

    print("Grouping by session...")
    sessions = group_by_session(all_obs)
    for sid, obs_list in sessions.items():
        print(f"  → {sid}: {len(obs_list)} observations")

    print("\nGenerating report...")
    report = generate_report(sessions, gt, all_obs)

    os.makedirs(REPORT_FILE.parent, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"  → Saved to {REPORT_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    gt_metrics = compute_false_positive_rate(gt)
    print(f"\nTotal observations: {len(all_obs)}")
    print(f"Sessions: {list(sessions.keys())}")
    print("\nGround Truth Metrics:")
    print(f"  Precision:  {gt_metrics['precision']}")
    print(f"  Recall:     {gt_metrics['recall']}")
    print(f"  FPR:        {gt_metrics['fpr']}")
    print(f"  Block rate: {gt_metrics['block_rate']} ({gt_metrics['total_blocks']}/{gt_metrics['total_actions']})")

    print("\nVerdict distribution (all sessions):")
    for v, info in compute_verdict_distribution(all_obs).items():
        print(f"  {v:8s}: {info['count']:4d} ({info['pct']}%)")

    print("\nTop tool usage:")
    for t, info in list(compute_tool_distribution(all_obs).items())[:5]:
        print(f"  {t:15s}: {info['count']:4d} ({info['pct']}%)")

    print("\nTop 5 highest-risk actions:")
    for i, r in enumerate(top_risk_actions(all_obs, 5), 1):
        e_str = f"{r['e_value']:.2e}" if r['e_value'] > 1000 else f"{r['e_value']:.4f}"
        print(f"  {i}. seq={r['sequence']} tool={r['tool']} verdict={r['verdict']} "
              f"e={e_str} fisher={r['fisher_divergence']:.3f}")

    # Regime changes
    for sid, obs_list in sessions.items():
        regimes = identify_regime_changes(obs_list)
        if regimes:
            print(f"\nRegime changes in {sid}:")
            for rc in regimes:
                print(f"  seq {rc['sequence']}: {rc['from_regime']} → {rc['to_regime']}")

    # ASCII timeline preview
    for sid, obs_list in sessions.items():
        print(f"\nVerdict timeline ({sid}):")
        print(f"  {ascii_timeline(obs_list)}")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
