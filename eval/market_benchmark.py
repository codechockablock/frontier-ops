"""
Market Architecture Benchmark Suite
====================================

Replays real telemetry through SeveritySignal + IntervalAnomalySignal and measures
detection performance against ground-truth sidecar verdicts.

Metrics:
  - ARL (Average Run Length) on PASS-only segments
  - Detection delay on FLAG/BLOCK transitions
  - FPR (False Positive Rate) — alarm on PASS observations
  - TPR (True Positive Rate) — alarm on FLAG/BLOCK observations
  - Market-vs-sidecar verdict agreement rate
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from frontier_ops.sensing.market_signals import (
    IntervalAnomalySignal,
    SeveritySignal,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    sequence: int
    timestamp: float  # epoch seconds
    verdict: str      # sidecar verdict (pass/monitor/flag/block)
    behavioral_vector: List[float]
    detection: Dict
    session_id: str


def load_observations(path: Path) -> List[Observation]:
    """Load JSONL observations, return sorted by sequence."""
    obs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ts_str = rec["timestamp"]
        # Parse ISO timestamp to epoch
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        obs.append(Observation(
            sequence=rec["sequence"],
            timestamp=ts,
            verdict=rec["governance"]["verdict"].lower(),
            behavioral_vector=rec["behavioral_vector"],
            detection=rec["detection"],
            session_id=rec["session_id"],
        ))
    obs.sort(key=lambda o: (o.timestamp, o.sequence))
    return obs


@dataclass
class MarketLogEntry:
    step: int
    verdict: str
    d_raw: float
    s_raw: float
    label: str


def load_market_log(path: Path) -> List[MarketLogEntry]:
    """Parse market-daemon.log for [market] step= lines."""
    entries = []
    pattern = re.compile(
        r"\[market\] step=(\d+) verdict=(\w+) d_raw=([\d.]+) s_raw=([\d.]+) label='(.+?)'"
    )
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            entries.append(MarketLogEntry(
                step=int(m.group(1)),
                verdict=m.group(2).lower(),
                d_raw=float(m.group(3)),
                s_raw=float(m.group(4)),
                label=m.group(5),
            ))
    entries.sort(key=lambda e: e.step)
    return entries


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    threshold: float
    # Detection metrics
    arl_pass: float           # Average run length on PASS-only segments
    arl_pass_std: float
    detection_delay_mean: float  # Steps from first FLAG/BLOCK to first alarm
    detection_delay_std: float
    n_transitions: int        # Number of PASS→FLAG/BLOCK transitions
    # Classification metrics
    fpr: float                # False positive rate (alarm on PASS)
    tpr: float                # True positive rate (alarm on FLAG/BLOCK)
    precision: float
    f1: float
    n_pass: int
    n_flagblock: int
    # CUSUM stats
    d_alarms: int
    s_alarms: int
    either_alarms: int


@dataclass
class BenchmarkResult:
    n_observations: int
    verdict_counts: Dict[str, int]
    threshold_results: List[ThresholdResult]
    # Market agreement
    market_entries: int
    agreement_rate: float
    agreement_by_verdict: Dict[str, float]


def _is_elevated(verdict: str) -> bool:
    return verdict in ("flag", "block")


def run_benchmark(
    observations: List[Observation],
    market_entries: List[MarketLogEntry],
    thresholds: List[float],
) -> BenchmarkResult:
    """Run the full benchmark across multiple threshold configs."""

    verdict_counts: Dict[str, int] = {}
    for obs in observations:
        verdict_counts[obs.verdict] = verdict_counts.get(obs.verdict, 0) + 1

    # -----------------------------------------------------------------------
    # Per-threshold evaluation
    # -----------------------------------------------------------------------
    threshold_results = []
    for thresh in thresholds:
        result = _evaluate_threshold(observations, thresh)
        threshold_results.append(result)

    # -----------------------------------------------------------------------
    # Market vs sidecar agreement
    # -----------------------------------------------------------------------
    agreement_rate, agreement_by_verdict = _compute_agreement(observations, market_entries)

    return BenchmarkResult(
        n_observations=len(observations),
        verdict_counts=verdict_counts,
        threshold_results=threshold_results,
        market_entries=len(market_entries),
        agreement_rate=agreement_rate,
        agreement_by_verdict=agreement_by_verdict,
    )


def _evaluate_threshold(
    observations: List[Observation],
    threshold: float,
) -> ThresholdResult:
    """Evaluate D+S signals at a specific CUSUM threshold."""

    cusum_params = {"threshold": threshold}

    # Build signals
    d_signal = SeveritySignal(severity_window=10, cusum_params=cusum_params)
    s_signal = IntervalAnomalySignal(cusum_params=cusum_params)

    # Calibrate S signal from PASS-only timestamps
    pass_timestamps = [
        obs.timestamp for obs in observations
        if obs.verdict == "pass"
    ]
    if len(pass_timestamps) >= 10:
        s_signal.calibrate_from_timestamps(pass_timestamps)
    else:
        # Fallback: use all timestamps
        s_signal.calibrate_from_timestamps([obs.timestamp for obs in observations])

    # Replay observations
    d_results = []
    s_results = []
    alarms = []  # per-step: True if either D or S alarmed

    for obs in observations:
        d_res = d_signal.step(obs.verdict)
        s_res = s_signal.step(obs.timestamp)
        d_results.append(d_res)
        s_results.append(s_res)
        alarms.append(d_res.alarm or s_res.alarm)

    # -----------------------------------------------------------------------
    # ARL on PASS-only segments
    # -----------------------------------------------------------------------
    pass_run_lengths = _compute_pass_run_lengths(observations, alarms)
    arl_pass = float(np.mean(pass_run_lengths)) if pass_run_lengths else float("inf")
    arl_pass_std = float(np.std(pass_run_lengths)) if len(pass_run_lengths) > 1 else 0.0

    # -----------------------------------------------------------------------
    # Detection delay on FLAG/BLOCK transitions
    # -----------------------------------------------------------------------
    delays, n_transitions = _compute_detection_delays(observations, alarms)
    delay_mean = float(np.mean(delays)) if delays else float("nan")
    delay_std = float(np.std(delays)) if len(delays) > 1 else 0.0

    # -----------------------------------------------------------------------
    # FPR / TPR
    # -----------------------------------------------------------------------
    tp = fp = fn = tn = 0
    for obs, alarm in zip(observations, alarms):
        elevated = _is_elevated(obs.verdict)
        if alarm and elevated:
            tp += 1
        elif alarm and not elevated:
            fp += 1
        elif not alarm and elevated:
            fn += 1
        else:
            tn += 1

    n_pass = tn + fp
    n_flagblock = tp + fn
    fpr = fp / n_pass if n_pass > 0 else 0.0
    tpr = tp / n_flagblock if n_flagblock > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0

    d_alarm_count = sum(1 for r in d_results if r.alarm)
    s_alarm_count = sum(1 for r in s_results if r.alarm)

    return ThresholdResult(
        threshold=threshold,
        arl_pass=arl_pass,
        arl_pass_std=arl_pass_std,
        detection_delay_mean=delay_mean,
        detection_delay_std=delay_std,
        n_transitions=n_transitions,
        fpr=fpr,
        tpr=tpr,
        precision=precision,
        f1=f1,
        n_pass=n_pass,
        n_flagblock=n_flagblock,
        d_alarms=d_alarm_count,
        s_alarms=s_alarm_count,
        either_alarms=sum(alarms),
    )


def _compute_pass_run_lengths(
    observations: List[Observation],
    alarms: List[bool],
) -> List[int]:
    """
    Compute run lengths (steps between alarms) on PASS-only observation segments.
    A run is the number of consecutive PASS observations before an alarm fires.
    """
    runs = []
    current_run = 0
    in_pass_segment = False

    for obs, alarm in zip(observations, alarms):
        if obs.verdict == "pass":
            in_pass_segment = True
            current_run += 1
            if alarm:
                runs.append(current_run)
                current_run = 0
        else:
            if in_pass_segment and current_run > 0:
                runs.append(current_run)  # Segment ended without alarm
            current_run = 0
            in_pass_segment = False

    if in_pass_segment and current_run > 0:
        runs.append(current_run)

    return runs


def _compute_detection_delays(
    observations: List[Observation],
    alarms: List[bool],
) -> Tuple[List[int], int]:
    """
    For each transition from PASS → FLAG/BLOCK, compute steps until first alarm.
    """
    delays = []
    n_transitions = 0
    i = 0
    n = len(observations)

    while i < n:
        # Find start of an elevated segment preceded by a PASS
        if _is_elevated(observations[i].verdict):
            # Check if preceded by pass
            if i == 0 or not _is_elevated(observations[i - 1].verdict):
                n_transitions += 1
                # Count steps until alarm in this elevated segment
                delay = 0
                found_alarm = False
                j = i
                while j < n and _is_elevated(observations[j].verdict):
                    delay += 1
                    if alarms[j]:
                        delays.append(delay)
                        found_alarm = True
                        break
                    j += 1
                if not found_alarm:
                    # Never alarmed during this segment — record segment length as delay
                    delays.append(delay)
                i = j if not found_alarm else j + 1
                continue
        i += 1

    return delays, n_transitions


def _compute_agreement(
    observations: List[Observation],
    market_entries: List[MarketLogEntry],
) -> Tuple[float, Dict[str, float]]:
    """
    Compare market daemon verdicts with sidecar verdicts where steps overlap.
    Market steps are 1-indexed starting from the beginning of the market daemon's
    observation window, which may not align 1:1 with observation sequence numbers.
    We align by step order.
    """
    if not market_entries:
        return 0.0, {}

    # Market entries are a subset — align by step number to observation index
    # Market step numbers correspond to observations processed by the daemon.
    # We'll match by positional order within the observation list.
    market_by_step = {e.step: e for e in market_entries}
    min_step = min(market_by_step.keys())

    agree_total = 0
    agree_count = 0
    agree_by_verdict: Dict[str, List[bool]] = {}

    for entry in market_entries:
        obs_idx = entry.step - min_step
        if obs_idx < 0 or obs_idx >= len(observations):
            continue

        sidecar_v = observations[obs_idx].verdict
        market_v = entry.verdict

        matched = sidecar_v == market_v
        agree_total += 1
        agree_count += int(matched)

        if sidecar_v not in agree_by_verdict:
            agree_by_verdict[sidecar_v] = []
        agree_by_verdict[sidecar_v].append(matched)

    rate = agree_count / agree_total if agree_total > 0 else 0.0
    by_verdict = {
        v: sum(matches) / len(matches)
        for v, matches in agree_by_verdict.items()
    }
    return rate, by_verdict


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(result: BenchmarkResult, date_label: str) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        f"# Market Architecture Benchmark — {date_label}",
        "",
        "## Dataset Summary",
        "",
        f"- **Observations:** {result.n_observations}",
        f"- **Verdict distribution:** {_fmt_counts(result.verdict_counts)}",
        f"- **Market daemon evaluations:** {result.market_entries}",
        "",
        "## Detection Performance by Threshold",
        "",
        "| Threshold | ARL (PASS) | ARL σ | Det. Delay | Delay σ | Transitions | FPR | TPR | Precision | F1 | D Alarms | S Alarms | Total Alarms |",
        "|-----------|-----------|-------|------------|---------|-------------|-----|-----|-----------|----|---------:|---------:|-------------:|",
    ]

    for tr in result.threshold_results:
        lines.append(
            f"| {tr.threshold:.1f} "
            f"| {_fmt_float(tr.arl_pass)} "
            f"| {_fmt_float(tr.arl_pass_std)} "
            f"| {_fmt_float(tr.detection_delay_mean)} "
            f"| {_fmt_float(tr.detection_delay_std)} "
            f"| {tr.n_transitions} "
            f"| {tr.fpr:.3f} "
            f"| {tr.tpr:.3f} "
            f"| {tr.precision:.3f} "
            f"| {tr.f1:.3f} "
            f"| {tr.d_alarms} "
            f"| {tr.s_alarms} "
            f"| {tr.either_alarms} |"
        )

    lines.extend([
        "",
        "## Detailed Threshold Analysis",
        "",
    ])

    for tr in result.threshold_results:
        lines.extend([
            f"### Threshold = {tr.threshold:.1f}",
            "",
            f"- **PASS observations:** {tr.n_pass} — FP alarms: {int(tr.fpr * tr.n_pass)}",
            f"- **FLAG/BLOCK observations:** {tr.n_flagblock} — TP alarms: {int(tr.tpr * tr.n_flagblock)}",
            f"- **ARL on PASS segments:** {_fmt_float(tr.arl_pass)} steps (σ={_fmt_float(tr.arl_pass_std)})",
            f"- **Detection delay:** {_fmt_float(tr.detection_delay_mean)} steps (σ={_fmt_float(tr.detection_delay_std)}) across {tr.n_transitions} transitions",
            "",
        ])

    lines.extend([
        "## Market vs Sidecar Agreement",
        "",
        f"- **Overall agreement:** {result.agreement_rate:.1%} across {result.market_entries} evaluations",
        "",
    ])

    if result.agreement_by_verdict:
        lines.append("| Sidecar Verdict | Agreement Rate |")
        lines.append("|-----------------|----------------|")
        for v, rate in sorted(result.agreement_by_verdict.items()):
            lines.append(f"| {v} | {rate:.1%} |")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- **ARL (PASS):** Higher is better — measures average steps between false alarms on benign segments.",
        "- **Detection delay:** Lower is better — measures responsiveness to real FLAG/BLOCK transitions.",
        "- **FPR:** Lower is better — fraction of PASS observations that trigger alarms.",
        "- **TPR:** Higher is better — fraction of FLAG/BLOCK observations correctly alarmed.",
        "- **F1:** Harmonic mean of precision and TPR — overall detection quality.",
        "",
        "---",
        f"*Generated by `eval/market_benchmark.py` on {date_label}*",
    ])

    return "\n".join(lines)


def _fmt_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _fmt_float(v: float) -> str:
    if not np.isfinite(v):
        return "∞" if v == float("inf") else "NaN"
    return f"{v:.2f}"
