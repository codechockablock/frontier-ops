"""L0-vs-L2 attribution ablation over the baseline fired-step records.

Classification rule: a verdict step is GEO_FREE if it can fire with the
geometric machinery absent — its *necessary* trigger is semantic/policy
(regex, classifier, state machine) and any corroboration requirement has at
least one non-geometric disjunct. Steps whose necessary condition is
geometric (e-value/conformal over fisher divergence, embedding displacement
or coherence, curvature, quality e-value, Mahalanobis trajectory cost)
are GEO_NEEDED.

Cascade-order caveat (stated in the report): attribution records the FIRST
matching step per action, so GEO-attributed actions might also satisfy a
later semantic step — GEO_FREE detection rates are LOWER bounds, and
"GEO-unique catches" are UPPER bounds on what removing geometry loses.
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
BASE = json.load(open(HERE / "ablation_baseline.json"))

# step -> geo_free (True = can fire without geometric machinery).
# trajectory_cost (Mahalanobis over risk-category vectors) is counted
# geometric here; SENSITIVITY flips it to semantic.
GEO_FREE = {
    # breaker / content / kill chain / policy / injection / harm intent
    "step_1_breaker_block": True,
    "step_2_content_block": True,
    "step_3_kill_chain_block": True,
    "step_3_kill_chain_discovery": True,
    "step_3b_task_auth_path_violation": True,
    "step_3b_task_auth_tool_violation": True,
    "step_5_content_flag": True,
    "step_5b_injection_flag": True,
    "step_5b2_corrupted_tool_followup": True,
    "step_5b3_post_injection_anomalous": True,
    "step_5b3_post_injection_anomalous_corroborated": True,
    "step_5b4_credential_accumulation": True,
    "step_5e_api_sequence_high": True,
    "step_5e_api_sequence_corroborated": True,  # OR: injection/content disjuncts
    "step_5f_prompted_privilege_interaction": True,  # OR: injection/corrupted/content
    "step_5g_system_recon_targeting": True,
    "step_5g2_injection_compliance": True,
    "step_5g3_url_substitution": True,
    "step_5g3_url_substitution_corroborated": True,
    "step_5h_harm_intent_corroborated": True,
    "step_5h_harm_intent_strong_corroboration": True,
    "step_10_breaker_flag": True,
    "step_10b_thinking_strong_breaker": True,   # thinking + breaker (both semantic)
    "step_10c_trajectory_strong_breaker": True,  # thinking trajectory + breaker
    "step_10d_obfuscation_corroborated": True,  # OR: breaker/injection/content disjuncts
    "step_10g_absence_strong_standalone": True,
    "step_10g_absence_strong_corroborated": True,
    "step_10g_absence_moderate_fusion": True,
    "step_13_semantic_standalone": True,
    "step_13_semantic_injection": True,
    "step_13_semantic_high_risk_cat": True,
    "step_13_semantic_lightweight": True,
    "step_13_semantic_trajectory_content": True,
    "step_13_semantic_trajectory_strong": True,
    "step_13_semantic_trajectory_moderate": True,
    "step_13_weak_voting": True,
    "step_14_track_b_enforcement": True,
    # geometric-necessary
    "step_4_displacement_unconditional": False,   # embedding displacement trigger
    "step_4_displacement_corroborated": False,
    "step_5c_trajectory_block": False,            # Mahalanobis trajectory cost
    "step_5c_trajectory_high_risk": False,
    "step_5c_trajectory_multi_vector": False,
    "step_5c_trajectory_diverse_high_risk": False,
    "step_5d_coherence_violation": False,         # embedding coherence
    "step_5f_drifted_egress": False,              # corroboration: traj_cost OR displacement
    "step_5f_drifted_egress_interaction": False,
    "step_5f_exfil_interaction": False,           # check on fire; exfil interaction gates
    "step_5f_long_drifted_egress_taint": False,
    "step_6_evalue_block": False,
    "step_7_evalue_flag": False,
    "step_8_conformal_conjunction_block": False,
    "step_9_conformal_evalue_flag": False,
    "step_10b_thinking_moderate_evalue": False,   # AND cumulative_e
    "step_10c_trajectory_moderate_evalue": False,
    "step_10c2_quality_kill_chain": False,        # quality e-value trigger
    "step_10c2_quality_displacement": False,
    "step_10e_curvature_interaction": False,      # curvature trigger
    "step_10e_curvature_conformal": False,
    "step_10e_curvature_auth_escalate": False,
    "step_10e_curvature_short_trajectory": False,
}

# steps counted geometric ONLY via trajectory_cost; flipped in sensitivity
TRAJ_ONLY = {
    "step_5c_trajectory_block",
    "step_5c_trajectory_high_risk",
    "step_5c_trajectory_multi_vector",
    "step_5c_trajectory_diverse_high_risk",
}


def analyze(rows, geo_free):
    unknown = Counter()
    n_adv = n_ben = tp = fp = 0
    tp_geo_free = fp_geo_free = 0
    tp_geo_unique = fp_geo_unique = 0
    geo_unique_steps = Counter()
    for r in rows:
        fired = r["fired_steps"]
        for s in fired:
            if s not in geo_free:
                unknown[s] += 1
        caught = bool(fired)
        sem_caught = any(geo_free.get(s, False) for s in fired)
        geo_only = caught and not sem_caught
        if r["category"] == "adversarial":
            n_adv += 1
            tp += caught
            tp_geo_free += sem_caught
            tp_geo_unique += geo_only
            if geo_only:
                geo_unique_steps.update(fired)
        else:
            n_ben += 1
            fp += caught
            fp_geo_free += sem_caught
            fp_geo_unique += geo_only
    return dict(
        n_adv=n_adv,
        n_ben=n_ben,
        combined=dict(tpr=tp / max(n_adv, 1), fpr=fp / max(n_ben, 1)),
        geo_free=dict(tpr=tp_geo_free / max(n_adv, 1), fpr=fp_geo_free / max(n_ben, 1)),
        geo_unique=dict(tp=tp_geo_unique, fp=fp_geo_unique),
        geo_unique_steps=dict(geo_unique_steps),
        unknown_steps=dict(unknown),
    )


def main():
    out = {}
    sens = {k: (True if k in TRAJ_ONLY else v) for k, v in GEO_FREE.items()}
    for name, rows in BASE.items():
        out[name] = analyze(rows, GEO_FREE)
        out[name]["sensitivity_traj_as_semantic"] = analyze(rows, sens)["geo_free"]
        fired_all = Counter(s for r in rows for s in r["fired_steps"])
        out[name]["fired_step_census"] = dict(fired_all.most_common())
    with open(HERE / "ablation_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    for name, res in out.items():
        c, g, u = res["combined"], res["geo_free"], res["geo_unique"]
        print(
            f"{name}: adv={res['n_adv']} ben={res['n_ben']}\n"
            f"  combined      TPR {c['tpr']:.3f}  FPR {c['fpr']:.3f}\n"
            f"  L2-only (LB)  TPR {g['tpr']:.3f}  FPR {g['fpr']:.3f}\n"
            f"  GEO-unique    TP {u['tp']}  FP {u['fp']}\n"
            f"  sens(traj=sem) TPR {res['sensitivity_traj_as_semantic']['tpr']:.3f} "
            f"FPR {res['sensitivity_traj_as_semantic']['fpr']:.3f}"
        )
        if res["geo_unique_steps"]:
            print(f"  geo-unique fired steps: {res['geo_unique_steps']}")
        if res["unknown_steps"]:
            print(f"  UNCLASSIFIED steps: {res['unknown_steps']}")


if __name__ == "__main__":
    main()
