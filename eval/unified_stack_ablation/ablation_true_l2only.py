"""TRUE L2-only re-run: zero every geometric input at the verdict boundary
(the seam unified-stack's own tests monkeypatch) so the cascade falls
through the geometric steps and semantic steps get their chance at every
action. Two configs:

  strict  — zero conformal / e-value / conjunction / displacement /
            coherence / curvature / quality / trajectory_cost
  keep_tc — same but trajectory_cost kept (Mahalanobis over semantic risk
            categories — the contested classification)

In-memory patch only; the unified-stack tree is not modified.
"""

import json
import sys
import time
from pathlib import Path

US = Path("/Users/joseph/unified-stack")
sys.path.insert(0, str(US))

from frontier_ops.pipeline.core import Pipeline  # noqa: E402
from frontier_ops.pipeline.verdict import _determine_verdict  # noqa: E402
from frontier_ops.sensing.conformal import Verdict  # noqa: E402
from frontier_ops.sensing.quality_combiner import QUALITY_OK  # noqa: E402

from eval.run_evaluation import make_default_pipeline, run_trace  # noqa: E402

HERE = Path(__file__).parent
ROOT = US / "eval" / "traces"
DATASETS = {
    "internal": ROOT,
    "thinking_augmented": ROOT / "thinking_augmented",
    "atbench": ROOT / "external" / "atbench",
    "agentdojo": ROOT / "external" / "agentdojo",
}


def make_geo_free_verdict(keep_trajectory_cost: bool):
    def patched(
        breaker_verdict,
        conformal_verdict,
        cumulative_e,
        conjunction_active,
        budget,
        content_score=0.0,
        kill_chain_override=None,
        displacement_score=0.0,
        *args,
        **kw,
    ):
        # geometric machinery off
        conformal_verdict = Verdict.ACT
        cumulative_e = 0.0
        conjunction_active = False
        displacement_score = 0.0
        kw["task_coherence_violation"] = False
        kw["curvature_max_jump"] = 0.0
        kw["curvature_n_steps"] = 0
        kw["quality_e_value"] = 0.0
        kw["quality_verdict"] = QUALITY_OK
        if not keep_trajectory_cost:
            kw["trajectory_cost"] = 0.0
        return _determine_verdict(
            breaker_verdict,
            conformal_verdict,
            cumulative_e,
            conjunction_active,
            budget,
            content_score,
            kill_chain_override,
            displacement_score,
            *args,
            **kw,
        )

    return staticmethod(patched)


def load_dataset(base: Path):
    traces = []
    for sub in [base / "benign", base / "adversarial"]:
        for fp in sorted(sub.glob("*.json")):
            with open(fp) as f:
                traces.append(json.load(f))
    return traces


def run_config(name: str, keep_tc: bool):
    Pipeline._determine_verdict = make_geo_free_verdict(keep_tc)
    out = {}
    try:
        for ds, base in DATASETS.items():
            traces = load_dataset(base)
            t0 = time.time()
            rows = []
            for trace in traces:
                res = run_trace(trace, make_default_pipeline)
                fired = sorted(
                    {
                        s["verdict_reason"]
                        for s in res["step_results"]
                        if s["verdict"] in ("flag", "block")
                    }
                )
                rows.append(
                    dict(
                        trace_id=res["trace_id"],
                        category=res["category"],
                        fired_steps=fired,
                    )
                )
            n_adv = sum(r["category"] == "adversarial" for r in rows)
            n_ben = len(rows) - n_adv
            tp = sum(bool(r["fired_steps"]) for r in rows if r["category"] == "adversarial")
            fp = sum(bool(r["fired_steps"]) for r in rows if r["category"] != "adversarial")
            out[ds] = dict(
                n_adv=n_adv,
                n_ben=n_ben,
                tpr=tp / max(n_adv, 1),
                fpr=fp / max(n_ben, 1),
                rows=rows,
            )
            print(
                f"[{name}] {ds}: TPR {out[ds]['tpr']:.3f} FPR {out[ds]['fpr']:.3f}"
                f" ({time.time()-t0:.0f}s)",
                flush=True,
            )
    finally:
        Pipeline._determine_verdict = staticmethod(_determine_verdict)
    return out


def main():
    results = {
        "strict": run_config("strict", keep_tc=False),
        "keep_trajectory_cost": run_config("keep_tc", keep_tc=True),
    }
    with open(HERE / "ablation_true_l2only.json", "w") as f:
        json.dump(results, f, indent=1)
    print("written: ablation_true_l2only.json", flush=True)


if __name__ == "__main__":
    main()
