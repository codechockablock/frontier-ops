"""
Monitor a Claude Code JSONL output stream with frontier-ops.

Shows: constitutional metric definition, pipeline startup, live boundary
sensing output, governance chain logging.

Usage:
    python monitor_claude_code.py /path/to/claude-code-session.jsonl
"""
import json
import sys

from frontier_ops import ConstitutionSpec, FullPipeline
from frontier_ops.governance import GovernanceChain


def main():
    constitution = ConstitutionSpec.agent_safety_default()
    pipeline = FullPipeline(
        constitution=constitution,
        enable_governance=True,
        enable_memory=False,  # Memory not needed for offline analysis
    )
    gov = pipeline.governance

    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "-"
    source = open(jsonl_path) if jsonl_path != "-" else sys.stdin

    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract the text content from the JSONL event
        text = event.get("content", event.get("text", event.get("message", "")))
        if not text:
            continue

        result = pipeline.process_step(text)

        # Print monitoring output
        status = "OK" if result.alert_level < 0.3 else "WARN" if result.alert_level < 0.7 else "ALERT"
        print(f"[{status}] step={result.step} alert={result.alert_level:.2f} "
              f"drift={result.newma_divergence:.3f} budget={result.budget_remaining:.2f}")

        if result.alert_reasons:
            for reason in result.alert_reasons:
                print(f"  -> {reason}")

        if result.boundary_proximities:
            near = {k: v for k, v in result.boundary_proximities.items() if v > 0.3}
            if near:
                print(f"  boundaries: {near}")

    # Export governance chain for audit
    if gov:
        chain_export = gov.export_chain()
        print(f"\nGovernance chain: {len(chain_export)} entries")
        print(f"Public key: {gov.public_key_hex()}")

    if jsonl_path != "-":
        source.close()


if __name__ == "__main__":
    main()
