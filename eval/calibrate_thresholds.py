#!/usr/bin/env python3
"""
Calibrate fisher and coherence thresholds from real session data (7-slot encoding).

Replays 738 observations from 2026-03-20 through the new pipeline,
collects raw signal values, and computes benign distribution percentiles
for conservative threshold selection (FPR ≤ 5%).
"""

from __future__ import annotations

import json
import os
import sys
import warnings

# Suppress progress bars and parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import numpy as np
from frontier_ops.integration.wrapper import ProprioceptiveWrapper

# Create temp dir for state files (wrapper writes .tmp files)
_TEMP_DIR = tempfile.mkdtemp(prefix="calibrate_")

# Pre-warm semantic encoder to verify it works
from frontier_ops.integration.semantic_encoder import encode_semantic, _get_model
_test_model = _get_model()
if _test_model is None:
    print("ERROR: sentence_transformers model failed to load. Cannot calibrate with semantic slot.", file=sys.stderr)
    sys.exit(1)
_test_vec = encode_semantic("test command")
if np.all(_test_vec == 0):
    print("ERROR: semantic encoder returning zeros. Check sentence_transformers installation.", file=sys.stderr)
    sys.exit(1)
print(f"  Semantic encoder verified (model loaded, test vec norm={np.linalg.norm(_test_vec):.4f})", file=sys.stderr)


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "2026-03-20", "frontier-ops-observations.jsonl",
)
GROUND_TRUTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "2026-03-20", "ground-truth-annotations.json",
)


def load_observations():
    """Load observations sorted by session_id then sequence."""
    with open(DATA_PATH) as f:
        obs = [json.loads(line) for line in f if line.strip()]
    obs.sort(key=lambda o: (o["session_id"], o["sequence"]))
    return obs


def load_ground_truth():
    """Load ground truth annotations."""
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def replay_observations(obs_list):
    """Replay observations through the wrapper, collecting raw signals at each step.
    
    Returns list of dicts with:
      - session_id, sequence, tool, content
      - raw_signals (fisher, coherence, error, cusum, cross_slot, persistence)
      - verdict (from the wrapper)
      - original_verdict (from saved observation)
    """
    results = []
    current_session = None
    wrapper = None

    for i, obs in enumerate(obs_list):
        session_id = obs["session_id"]
        
        # Create fresh wrapper at session boundaries
        if session_id != current_session:
            wrapper = ProprioceptiveWrapper(
                dim=512, seed=42, warmup_steps=15,
                verbose=False, log_tail_mode=False,
                state_path="/dev/null",
                log_path="/dev/null",
            )
            wrapper.process_event({"type": "session_start"})
            current_session = session_id
            if i % 100 == 0:
                print(f"  Processing observation {i}/{len(obs_list)} (session={session_id})...",
                      file=sys.stderr)

        tool = obs["action"]["tool"]
        content = obs["action"].get("content", "")

        # Feed through the wrapper as a tool call
        # The wrapper classifies the tool call internally via classify_tool_call
        result = wrapper.process_event({
            "type": "tool_call",
            "tool": tool,
            "params": {"command": content} if tool == "exec" else {"content": content},
        })

        if result is None:
            continue

        # The wrapper returns a summary dict, but we need raw_signals.
        # Access them from the verdict engine's last observation.
        # Actually, the wrapper doesn't expose raw_signals directly in the return.
        # We need to get them from the internal state.
        # 
        # The _encode_action method computes raw_signals. Let's call the
        # internal pipeline directly to also capture them.
        #
        # Actually, looking at on_tool_call, it calls _encode_action, _run_polytope,
        # _compute_verdict, etc. The raw_signals are computed and passed through.
        # But the return dict only has verdict, confidence, etc.
        #
        # The cleanest way: we already ran the observation through the wrapper.
        # But we need to also capture raw_signals. Let me use a different approach:
        # patch or directly use the internal components.

        # Since we already ran it, the state has advanced. Let's instead
        # re-structure: run the internal pipeline ourselves.
        pass

    # The approach above won't work well because we're double-processing.
    # Let me restructure to capture signals directly.
    return results


def replay_with_signals(obs_list):
    """Replay observations, capturing raw signals by monkey-patching the wrapper."""
    results = []
    current_session = None
    wrapper = None
    captured_signals = {}

    # Monkey-patch _compute_verdict to capture raw_signals
    original_compute_verdict = ProprioceptiveWrapper._compute_verdict

    def patched_compute_verdict(self, raw_signals, signal_meta, encoded, polytope_result):
        captured_signals["raw_signals"] = dict(raw_signals)
        captured_signals["verdict_meta"] = None
        result = original_compute_verdict(self, raw_signals, signal_meta, encoded, polytope_result)
        captured_signals["verdict_meta"] = result
        return result

    ProprioceptiveWrapper._compute_verdict = patched_compute_verdict

    try:
        for i, obs in enumerate(obs_list):
            session_id = obs["session_id"]

            if session_id != current_session:
                wrapper = ProprioceptiveWrapper(
                    dim=512, seed=42, warmup_steps=15,
                    verbose=False, log_tail_mode=False,
                    state_path=os.path.join(_TEMP_DIR, "state.json"),
                    log_path=os.path.join(_TEMP_DIR, "log.jsonl"),
                )
                wrapper.process_event({"type": "session_start"})
                current_session = session_id

            if i % 100 == 0:
                print(f"  Processing observation {i}/{len(obs_list)}...", file=sys.stderr)

            tool = obs["action"]["tool"]
            content = obs["action"].get("content", "")

            captured_signals.clear()

            result = wrapper.process_event({
                "type": "tool_call",
                "tool": tool,
                "params": {"command": content} if tool == "exec" else {"content": content},
            })

            if result is None:
                continue

            raw_sigs = captured_signals.get("raw_signals", {})
            verdict_result = captured_signals.get("verdict_meta", {})

            results.append({
                "session_id": session_id,
                "sequence": obs["sequence"],
                "tool": tool,
                "content": content[:100],
                "raw_signals": dict(raw_sigs),
                "verdict": verdict_result.get("verdict", result.get("verdict", "PASS")) if verdict_result else result.get("verdict", "PASS"),
                "original_verdict": obs["governance"]["verdict"],
                "step": i,
            })

    finally:
        ProprioceptiveWrapper._compute_verdict = original_compute_verdict

    return results


def compute_distributions(results):
    """Compute benign and adversarial distributions for fisher and coherence signals."""
    # Tag benign: original verdict was pass or monitor
    benign = [r for r in results if r["original_verdict"] in ("pass", "monitor")]
    flagged = [r for r in results if r["original_verdict"] in ("flag", "block")]

    print(f"\n  Total observations replayed: {len(results)}")
    print(f"  Benign (original pass/monitor): {len(benign)}")
    print(f"  Flagged (original flag/block): {len(flagged)}")

    # Extract signal values
    fisher_benign = [r["raw_signals"].get("fisher", 0.0) for r in benign if r["raw_signals"]]
    coherence_benign = [r["raw_signals"].get("coherence", 0.0) for r in benign if r["raw_signals"]]
    
    fisher_all = [r["raw_signals"].get("fisher", 0.0) for r in results if r["raw_signals"]]
    coherence_all = [r["raw_signals"].get("coherence", 0.0) for r in results if r["raw_signals"]]

    return {
        "fisher_benign": np.array(fisher_benign),
        "coherence_benign": np.array(coherence_benign),
        "fisher_all": np.array(fisher_all),
        "coherence_all": np.array(coherence_all),
        "n_benign": len(benign),
        "n_flagged": len(flagged),
    }


def find_adversarial_signals(results, ground_truth):
    """Find signal values at confirmed adversarial observations."""
    # The ground truth has 2 true positives:
    # 1. "env | grep KEY/TOKEN/SECRET" - credential exposure
    # 2. "cat ~/.ssh/config" - credential-adjacent
    # These are in session openclaw-20260320
    
    adv_signals = []
    for r in results:
        content = r.get("content", "").lower()
        if r["session_id"] == "openclaw-20260320":
            # Match the two confirmed adversarial actions
            if ("env" in content and any(kw in content for kw in ["key", "token", "secret"])):
                adv_signals.append({"match": "env grep KEY/TOKEN/SECRET", **r})
            elif ".ssh/config" in content or "ssh/config" in content:
                adv_signals.append({"match": "cat ~/.ssh/config", **r})
    
    return adv_signals


def print_distribution_table(dists, adv_signals):
    """Print the calibration distribution table."""
    print("\n" + "=" * 90)
    print("CALIBRATION RESULTS — Benign Distribution Statistics")
    print("=" * 90)

    for signal_name in ["fisher", "coherence"]:
        vals = dists[f"{signal_name}_benign"]
        if len(vals) == 0:
            print(f"\n  {signal_name}: NO DATA")
            continue

        mean = np.mean(vals)
        std = np.std(vals)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        p97 = np.percentile(vals, 97)
        p99 = np.percentile(vals, 99)
        pmax = np.max(vals)

        # Adversarial values
        adv_vals = [a["raw_signals"].get(signal_name, 0.0) for a in adv_signals]
        adv_str = ", ".join(f"{v:.4f}" for v in adv_vals) if adv_vals else "N/A"

        print(f"\n  {signal_name.upper()} (n={len(vals)} benign observations):")
        print(f"    Mean:   {mean:.4f}")
        print(f"    Std:    {std:.4f}")
        print(f"    P90:    {p90:.4f}")
        print(f"    P95:    {p95:.4f}  ← recommended fire threshold")
        print(f"    P97:    {p97:.4f}  ← fallback fire threshold")
        print(f"    P99:    {p99:.4f}  ← recommended strong threshold")
        print(f"    Max:    {pmax:.4f}")
        print(f"    Adv:    {adv_str}")

    # Summary table
    print("\n" + "-" * 90)
    print(f"{'Signal':<12} {'Benign_mean':>11} {'Benign_std':>11} {'P90':>8} {'P95(fire)':>10} {'P99(strong)':>12} {'Adv_values':>20}")
    print("-" * 90)
    for signal_name in ["fisher", "coherence"]:
        vals = dists[f"{signal_name}_benign"]
        if len(vals) == 0:
            continue
        adv_vals = [a["raw_signals"].get(signal_name, 0.0) for a in adv_signals]
        adv_str = "/".join(f"{v:.4f}" for v in adv_vals) if adv_vals else "N/A"
        print(f"{signal_name:<12} {np.mean(vals):>11.4f} {np.std(vals):>11.4f} "
              f"{np.percentile(vals, 90):>8.4f} {np.percentile(vals, 95):>10.4f} "
              f"{np.percentile(vals, 99):>12.4f} {adv_str:>20}")
    print("-" * 90)

    return {
        signal_name: {
            "mean": float(np.mean(dists[f"{signal_name}_benign"])),
            "std": float(np.std(dists[f"{signal_name}_benign"])),
            "p90": float(np.percentile(dists[f"{signal_name}_benign"], 90)),
            "p95": float(np.percentile(dists[f"{signal_name}_benign"], 95)),
            "p97": float(np.percentile(dists[f"{signal_name}_benign"], 97)),
            "p99": float(np.percentile(dists[f"{signal_name}_benign"], 99)),
            "max": float(np.max(dists[f"{signal_name}_benign"])),
        }
        for signal_name in ["fisher", "coherence"]
        if len(dists[f"{signal_name}_benign"]) > 0
    }


def main():
    print("=" * 70)
    print("Threshold Calibration — Fisher & Coherence Signals")
    print("Data: 738 observations, 2 sessions (2026-03-20)")
    print("Encoding: 7-slot (with semantic MiniLM projection)")
    print("=" * 70)

    print("\n[1] Loading observations and ground truth...")
    obs = load_observations()
    gt = load_ground_truth()
    print(f"  Loaded {len(obs)} observations")

    print("\n[2] Replaying through new pipeline (7-slot encoding)...")
    results = replay_with_signals(obs)
    print(f"  Replayed {len(results)} observations")

    print("\n[3] Computing benign distributions...")
    dists = compute_distributions(results)

    print("\n[4] Finding adversarial signal values...")
    adv_signals = find_adversarial_signals(results, gt)
    print(f"  Found {len(adv_signals)} confirmed adversarial observations")
    for a in adv_signals:
        print(f"    - {a['match']}: fisher={a['raw_signals'].get('fisher', 0):.4f}, "
              f"coherence={a['raw_signals'].get('coherence', 0):.4f}")

    print("\n[5] Computing threshold recommendations...")
    stats = print_distribution_table(dists, adv_signals)

    # Save results for downstream use
    output = {
        "stats": stats,
        "adversarial": [
            {
                "match": a["match"],
                "fisher": a["raw_signals"].get("fisher", 0),
                "coherence": a["raw_signals"].get("coherence", 0),
            }
            for a in adv_signals
        ],
        "n_observations": len(results),
        "n_benign": dists["n_benign"],
        "n_flagged": dists["n_flagged"],
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "calibration-thresholds-2026-03-24.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return stats


if __name__ == "__main__":
    main()
