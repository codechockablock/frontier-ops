"""
Constitutional Manifold Metric Tensor
======================================

The constitution is the domain adapter. A nuclear facility constitution and a
coffee shop inventory constitution produce completely different Riemannian
geometries from the same mathematical framework.

The metric tensor G(x) is position-dependent:
- Near boundaries: amplified curvature (softplus barrier)
- On-boundary: very high curvature (violations geometrically expensive)
- Far from boundaries: near-identity (normal behavior is cheap)

Cross-terms define dangerous COMBINATIONS:
- credential + scope = privilege escalation
- goal_displacement + self_modification = autonomous goal rewriting

The metric IS the precision matrix for the efference copy predictor.
The metric IS the weighting for angular displacement computation.
These are formally the same object (Friston's insight).

Usage::

    constitution = ConstitutionSpec(
        boundaries=[
            Boundary("credential_adjacent", threshold=0.6, sharpness=5.0),
            Boundary("self_modification_reasoning", threshold=0.4, sharpness=8.0),
        ],
        cross_terms=[
            CrossTerm(("credential_adjacent", "scope_exploration"), weight=3.0),
        ],
    )
    metric = ConstitutionalMetric(constitution, dim_names=CONCEPTS)
    G = metric.tensor_at(concept_vector)  # Position-dependent metric
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from frontier_ops.boundary.concept_extraction import CONCEPTS


# --- Constitution Specification -------------------------------------------

@dataclass
class Boundary:
    """A constitutional boundary on a concept dimension.

    Args:
        concept: Name of the concept dimension.
        threshold: Where the boundary is (0-1 in concept space).
        sharpness: Softplus alpha parameter. Higher = steeper boundary.
            Nuclear facility: 10-15 (strict). Dev tooling: 2-3 (gentle).
        base_weight: How much this boundary contributes to the metric.
    """
    concept: str
    threshold: float = 0.5
    sharpness: float = 5.0
    base_weight: float = 1.0


@dataclass
class CrossTerm:
    """A cross-term for dangerous concept combinations.

    The metric gets a cross-term G[i,j] when both concepts are elevated.
    This makes COMBINED activation expensive even if each alone is fine.

    Args:
        concepts: Pair of concept names.
        weight: How much this cross-term contributes.
        activation_threshold: Both concepts must exceed this for the term to activate.
    """
    concepts: Tuple[str, str]
    weight: float = 2.0
    activation_threshold: float = 0.3


@dataclass
class ConstitutionSpec:
    """A complete constitutional specification.

    Domain experts write this. The math is derived automatically.
    """
    name: str = "default"
    description: str = ""
    boundaries: List[Boundary] = field(default_factory=list)
    cross_terms: List[CrossTerm] = field(default_factory=list)
    # Global parameters
    baseline_weight: float = 1.0  # Diagonal entries when far from boundaries
    max_amplification: float = 100.0  # Cap on metric amplification

    @classmethod
    def agent_safety_default(cls) -> "ConstitutionSpec":
        """Default constitution for autonomous agent safety monitoring."""
        return cls(
            name="agent_safety_v1",
            description="Default safety constitution for autonomous agents",
            boundaries=[
                Boundary("credential_adjacent", threshold=0.5, sharpness=6.0, base_weight=3.0),
                Boundary("self_modification_reasoning", threshold=0.3, sharpness=8.0, base_weight=5.0),
                Boundary("goal_displacement", threshold=0.4, sharpness=5.0, base_weight=2.0),
                Boundary("scope_exploration", threshold=0.7, sharpness=3.0, base_weight=1.0),
            ],
            cross_terms=[
                # Credential access + scope exploration = privilege escalation
                CrossTerm(("credential_adjacent", "scope_exploration"), weight=4.0),
                # Goal displacement + self modification = autonomous goal rewriting
                CrossTerm(("goal_displacement", "self_modification_reasoning"), weight=5.0),
                # Credential + self modification = credential theft for self-enhancement
                CrossTerm(("credential_adjacent", "self_modification_reasoning"), weight=6.0),
            ],
            baseline_weight=1.0,
            max_amplification=50.0,
        )

    @classmethod
    def retail_sentinel(cls) -> "ConstitutionSpec":
        """Constitution for retail inventory profit leak detection."""
        return cls(
            name="retail_sentinel_v1",
            description="Retail profit leak detection constitution",
            boundaries=[
                Boundary("goal_displacement", threshold=0.5, sharpness=4.0, base_weight=2.0),
                Boundary("scope_exploration", threshold=0.6, sharpness=3.0, base_weight=1.5),
            ],
            cross_terms=[],
            baseline_weight=1.0,
            max_amplification=20.0,
        )


# --- Metric Tensor Construction ------------------------------------------

def softplus(x: float, alpha: float) -> float:
    """Softplus barrier function: log(1 + exp(alpha * x)) / alpha.

    Properties:
    - x << 0: approaches 0 (far from boundary, cheap)
    - x = 0: ~log(2)/alpha (at boundary)
    - x >> 0: approaches x (in violation territory, linearly growing)
    - Always differentiable (unlike ReLU)
    """
    # Numerically stable version
    if alpha * x > 20:
        return x
    return math.log(1 + math.exp(alpha * x)) / alpha


def softplus_derivative(x: float, alpha: float) -> float:
    """Derivative of softplus: sigmoid(alpha * x)."""
    ax = alpha * x
    if ax > 20:
        return 1.0
    if ax < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-ax))


class ConstitutionalMetric:
    """
    Position-dependent Riemannian metric tensor derived from a constitution.

    G(x) = G_baseline + sum_i boundary_amplification_i(x) + sum_j cross_term_j(x)

    Where each boundary contributes a rank-1 update to the metric:
        G_i(x) = w_i * softplus'(x_i - threshold_i, alpha_i) * e_i @ e_i^T

    And each cross-term contributes:
        G_j(x) = w_j * activation_j(x) * (e_a @ e_b^T + e_b @ e_a^T)
    """

    def __init__(
        self,
        constitution: ConstitutionSpec,
        dim_names: Optional[List[str]] = None,
    ):
        self.constitution = constitution
        self.dim_names = dim_names or CONCEPTS
        self.n_dims = len(self.dim_names)
        self._dim_index = {name: i for i, name in enumerate(self.dim_names)}

        # Validate constitution
        for b in constitution.boundaries:
            if b.concept not in self._dim_index:
                raise ValueError(f"Boundary concept '{b.concept}' not in dim_names: {self.dim_names}")
        for ct in constitution.cross_terms:
            for c in ct.concepts:
                if c not in self._dim_index:
                    raise ValueError(f"Cross-term concept '{c}' not in dim_names: {self.dim_names}")

    def tensor_at(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the metric tensor G(x) at position x in concept space.

        Args:
            x: Concept vector (n_dims,) with values in [0, 1].

        Returns:
            G: Symmetric positive-definite metric tensor (n_dims, n_dims).
        """
        G = np.eye(self.n_dims) * self.constitution.baseline_weight

        # Boundary contributions
        for b in self.constitution.boundaries:
            i = self._dim_index[b.concept]
            # How far past the boundary threshold
            distance = x[i] - b.threshold
            # Softplus derivative = sigmoid: smooth activation near boundary
            activation = softplus_derivative(distance, b.sharpness)
            # Rank-1 update: amplify metric in the boundary direction
            amplification = b.base_weight * activation
            G[i, i] += min(amplification, self.constitution.max_amplification)

        # Cross-term contributions
        for ct in self.constitution.cross_terms:
            ia, ib = [self._dim_index[c] for c in ct.concepts]
            # Both concepts must be above activation threshold
            act_a = max(0, x[ia] - ct.activation_threshold)
            act_b = max(0, x[ib] - ct.activation_threshold)
            joint_activation = act_a * act_b * ct.weight
            if joint_activation > 0:
                # Symmetric cross-term
                cross_val = min(joint_activation, self.constitution.max_amplification)
                G[ia, ib] += cross_val
                G[ib, ia] += cross_val
                # Also boost diagonals to maintain positive-definiteness
                G[ia, ia] += cross_val * 0.5
                G[ib, ib] += cross_val * 0.5

        return G

    def metric_weighted_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute metric-weighted distance between two concept vectors.

        Uses midpoint metric: G((x1+x2)/2).

        .. deprecated:: 0.3.0
            Curvature walls inside a distance are not a metric (v2 handoff
            §1): the wall magnitude has no habitat on real trajectories
            (coefficient of variation 0.03–0.04), and softplus costs belong
            in the decision layer (boundary.decision.DecisionCosts) applied
            to calibrated scores. For ranking, estimate the metric instead
            (static_metric.CalibratedMetric.from_labeled).
        """
        warnings.warn(
            "metric_weighted_distance is deprecated: position-dependent "
            "curvature inside a distance lost its pre-registered kill test "
            "(v2 handoff §1). Use an estimated static metric "
            "(CalibratedMetric.from_labeled) for ranking and DecisionCosts "
            "for expert costs.",
            DeprecationWarning,
            stacklevel=2,
        )
        midpoint = (x1 + x2) / 2
        G = self.tensor_at(midpoint)
        diff = x2 - x1
        return float(np.sqrt(max(0, diff @ G @ diff)))

    def metric_weighted_path_length(self, trajectory: List[np.ndarray]) -> float:
        """Total metric-weighted path length along a trajectory.

        .. deprecated:: 0.3.0
            Path-energy trajectory features scored at or below chance on the
            Apollo benchmarks (all variants ≤ 0.635 AUROC, whitened 0.452 —
            v2 handoff §1). Aggregate positions instead: chart-space step
            mean via boundary.step_mean.StepMeanScorer (0.790 on the same
            data).
        """
        warnings.warn(
            "metric_weighted_path_length is deprecated: path-energy features "
            "carried no signal on the Apollo benchmarks (v2 handoff §1). Use "
            "StepMeanScorer's chart-space mean instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(trajectory) < 2:
            return 0.0
        total = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for i in range(1, len(trajectory)):
                total += self.metric_weighted_distance(trajectory[i - 1], trajectory[i])
        return total

    def boundary_proximity(self, x: np.ndarray) -> Dict[str, float]:
        """For each boundary, return proximity (0 = far, 1 = at/past boundary)."""
        proximities = {}
        for b in self.constitution.boundaries:
            i = self._dim_index[b.concept]
            distance = x[i] - b.threshold
            proximities[b.concept] = softplus_derivative(distance, b.sharpness)
        return proximities

    def cross_term_activations(self, x: np.ndarray) -> Dict[str, float]:
        """For each cross-term, return current activation level."""
        activations = {}
        for ct in self.constitution.cross_terms:
            a_idx = self._dim_index[ct.concepts[0]]
            b_idx = self._dim_index[ct.concepts[1]]
            act_a = max(0, x[a_idx] - ct.activation_threshold)
            act_b = max(0, x[b_idx] - ct.activation_threshold)
            key = f"{ct.concepts[0]}\u00d7{ct.concepts[1]}"
            activations[key] = act_a * act_b * ct.weight
        return activations

    def summary_at(self, x: np.ndarray) -> Dict:
        """Full diagnostic summary at a point in concept space."""
        G = self.tensor_at(x)
        return {
            "metric_trace": float(np.trace(G)),
            "metric_det": float(np.linalg.det(G)),
            "metric_condition": float(np.linalg.cond(G)),
            "boundary_proximity": self.boundary_proximity(x),
            "cross_term_activations": self.cross_term_activations(x),
            "eigenvalues": np.linalg.eigvalsh(G).tolist(),
        }
