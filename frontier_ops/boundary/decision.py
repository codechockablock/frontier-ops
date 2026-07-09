"""
Decision layer — expert softplus costs over calibrated scores (v2).

The Apollo campaign rehoused the constitution's softplus cost field: it is a
valid *decision-layer* object, not a metric. Experts specify λ costs;
geometry gets estimated (see static_metric.CalibratedMetric.from_labeled
and StepMeanScorer). Costs therefore apply to CALIBRATED
scores AFTER ranking — never inside the distance/ranking computation, where
saturating transforms destroy ranking information (pipeline alert AUROC
0.502 on a signal that scored 0.647 raw).

Raw ranking scores always stay accessible (e.g. on StepResult) pre-gating;
this layer only adds a cost on top for downstream decisions.

Usage::

    costs = DecisionCosts(constitution)
    raw = scorer.score(text)              # ranking signal — use for AUROC
    decision = costs.apply(raw, scores)   # cost-adjusted decision value
"""

from __future__ import annotations

from typing import Dict

from frontier_ops.boundary.constitution import ConstitutionSpec, softplus


class DecisionCosts:
    """Softplus boundary costs with expert λ weights.

    Each constitution boundary (concept, threshold t, sharpness α, weight λ)
    contributes λ · softplus(x − t, α) — near zero below threshold,
    approaching λ·(x − t) above it. Cross-terms contribute
    λ · relu(x_a − t) · relu(x_b − t).
    """

    def __init__(self, constitution: ConstitutionSpec):
        self.constitution = constitution

    def cost(self, concept_scores: Dict[str, float]) -> float:
        """Total decision cost for one step's calibrated concept scores.

        Missing dims contribute nothing, so a constitution can be broader
        than the active dim set without crashing.
        """
        total = 0.0
        for b in self.constitution.boundaries:
            if b.concept not in concept_scores:
                continue
            total += b.base_weight * softplus(
                concept_scores[b.concept] - b.threshold, b.sharpness
            )
        for ct in self.constitution.cross_terms:
            a, bb = ct.concepts
            if a not in concept_scores or bb not in concept_scores:
                continue
            act_a = max(0.0, concept_scores[a] - ct.activation_threshold)
            act_b = max(0.0, concept_scores[bb] - ct.activation_threshold)
            total += ct.weight * act_a * act_b
        return total

    def apply(self, ranking_score: float, concept_scores: Dict[str, float]) -> float:
        """Cost-adjusted decision value. The input ranking score must remain
        available to callers — evaluate rankings on it, never on this."""
        return float(ranking_score) + self.cost(concept_scores)
