"""
Full Geometric Agent Pipeline
===============================

Wires all components into a single clean API.
This is what the agent loop calls on each step.

Usage::

    pipeline = FullPipeline(constitution=ConstitutionSpec.agent_safety_default())
    for step_text in agent_outputs:
        result = pipeline.process_step(step_text)
        if result.alert_level > 0.5:
            print(f"ALERT: {result.alert_reasons}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from frontier_ops.boundary.concept_extraction import ConceptExtractor, CONCEPTS
from frontier_ops.boundary.constitution import ConstitutionSpec, ConstitutionalMetric
from frontier_ops.sensing.efference import EfferenceCopyPredictor, PredictionError
from frontier_ops.sensing.newma import DualEWMA
from frontier_ops.sensing.trend import ScopeCreepDetector, MetricAdaptiveEWMA
from frontier_ops.sensing.drift_classifier import DriftClassifier
from frontier_ops.sensing.combiner import BayesFactorCombiner
from frontier_ops.authorization.scope import AuthorizationState, AuthorizationEvent
from frontier_ops.authorization.provenance import ProvenanceGraph, RESTRICTED_DIMENSIONS
from frontier_ops.authorization.budget import AuthorizationLinkedBudget


@dataclass
class StepResult:
    """Complete diagnostic result for one pipeline step.

    Ranking vs alerting (v2 note): `alert_level` is a saturating,
    threshold-gated decision signal for live monitoring. Never rank or
    evaluate (AUROC etc.) on it — gating destroys ranking information
    (alert AUROC 0.502 on a signal that scored 0.647 raw in the Apollo
    campaign). Rank on the raw fields (concept_vec, concept_scores,
    geodesic_distance, ...) or use boundary.step_mean.StepMeanScorer.
    """
    step: int
    text: str
    # Concept extraction
    concept_vec: np.ndarray
    concept_scores: Dict[str, float]
    # Efference copy
    predicted_vec: Optional[np.ndarray]
    pred_error: Optional[PredictionError]
    proprioceptive_context: str
    # Metric
    metric_tensor: np.ndarray
    metric_trace: float
    boundary_proximities: Dict[str, float]
    cross_term_activations: Dict[str, float]
    # Angular displacement
    angular_disp_delta: float
    angular_disp_cumulative: float
    # NEWMA
    newma_divergence: float
    newma_alarm: bool
    drift_classification: str
    # Cross-terms evaluated at the trajectory-smoothed position (temporal
    # conjunction: e.g. cred probe a few steps after an enum sweep)
    trajectory_cross_terms: Dict[str, float]
    # Trend
    trend_alerts: List[Any]
    # Budget
    lambda_value: float
    budget_remaining: float
    # Memory
    primed_memories: List[Any]
    novelty: bool
    # Alert
    alert_level: float
    alert_reasons: List[str]
    # Authorization (Layer 1-3)
    authorized: Optional[bool] = None
    geodesic_distance: Optional[float] = None
    authorization_radius: Optional[float] = None
    goal_confidence: Optional[float] = None
    needs_escalation: bool = False
    needs_clarification: bool = False
    authorization_verdict: str = "no_goal"


class FullPipeline:
    """
    Full governance pipeline: concept encoding, the drift/NEWMA sensing
    stack, authorization envelope, provenance graph, and (optional) Ed25519
    governance chain, wired together. Call process_step() per agent output.

    For DETECTION, prefer ``frontier_ops.CalibratedDetector`` — an in-domain
    calibrated prototype that beat this pipeline's alert_level and the best
    subset of its channels on every workload tested
    (eval/results/repo-open-questions-2026-07-10.md). Use FullPipeline when
    you want the authorization/governance/provenance layer around it; its
    ranking-quality signals are ``alert_level`` (post-v3 combiner fix) and
    the NEWMA channel, not the full ensemble.

    v3 default change: ``enable_memory`` now defaults to False. The VSA
    memory/activation path has no measured detection value (a repo-wide
    audit found it exercised by no benchmark) and added per-step cost; it is
    opt-in. Set ``enable_memory=True`` to restore the priming path.
    """

    def __init__(
        self,
        constitution: Optional[ConstitutionSpec] = None,
        curvature_budget: float = 2.0,
        expected_steps: int = 50,
        vsa_dim: int = 512,
        concept_extractor_tier: Optional[int] = 1,
        enable_governance: bool = True,
        enable_memory: bool = False,
        newma_threshold: float = 0.65,
        dim_names: Optional[List[str]] = None,
        semantic_anchors: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            enable_memory: opt-in VSA memory/activation priming (default
                False in v3 — no measured detection value; see class doc).
            dim_names: active concept dimensions, threaded through metric,
                predictor, and extractor construction. Defaults to the stock
                CONCEPTS, or to ``semantic_anchors``' keys when those are given.
            semantic_anchors: anchor phrases per dimension for the Tier-2
                extractor; required for custom dims the Tier-1 keyword table
                does not cover.
        """
        if dim_names is None:
            dim_names = (
                list(semantic_anchors.keys())
                if semantic_anchors is not None
                else list(CONCEPTS)
            )
        self.dim_names: List[str] = list(dim_names)

        self.constitution = constitution or ConstitutionSpec.agent_safety_default()

        # Layer 0: Metric
        self.metric = ConstitutionalMetric(self.constitution, dim_names=self.dim_names)
        self._boundary_thresholds = {
            b.concept: b.threshold for b in self.constitution.boundaries
        }

        # Concept extraction
        self.extractor = ConceptExtractor(
            force_tier=concept_extractor_tier,
            dims=self.dim_names,
            anchors=semantic_anchors,
        )

        # Efference copy
        self.predictor = EfferenceCopyPredictor(
            n_dims=len(self.dim_names),
            dim_names=self.dim_names,
        )

        # NEWMA. Threshold calibrated to the benign p95 of the metric-weighted
        # divergence on the 2026-03-20 real traces (679 benign steps: p95=0.65
        # for both tier-1 and blended vectors; drift-replay experiment). The
        # previous default of 0.15 alarmed on ~16% of benign steps, so alarms
        # carried almost no evidence. Recalibrate per deployment via the
        # newma_threshold constructor arg.
        self.newma = DualEWMA(
            n_dims=len(self.dim_names),
            alpha_fast=0.5,
            alpha_slow=0.05,
            threshold=newma_threshold,
        )

        # Drift classifier (disambiguates NEWMA signal)
        self.drift_classifier = DriftClassifier(window_size=15)

        # Bayes factor evidence combiner
        self.bayes_combiner = BayesFactorCombiner()

        # Trend detector
        self.trend = ScopeCreepDetector(
            dim_names=self.dim_names,
            window_size=8,
            slope_threshold=0.012,
            r2_threshold=0.5,
            boundaries={b.concept: b.threshold for b in self.constitution.boundaries},
        )

        # Metric-adaptive EWMA
        self.ewma = MetricAdaptiveEWMA(
            n_dims=len(self.dim_names),
            alpha=0.3,
            base_threshold=0.15,
        )

        # Authorization system (Layers 1-3)
        self.provenance = ProvenanceGraph()
        self.auth_state = AuthorizationState(
            metric=self.metric,
            default_radius=0.5,
        )
        self.auth_budget = AuthorizationLinkedBudget(
            total_budget=curvature_budget,
            expected_steps=expected_steps,
            provenance=self.provenance,
        )

        # Adaptive Lagrangian (now accessed via auth_budget)
        self.lagrangian = self.auth_budget.lagrangian

        # Memory + Activation
        self._enable_memory = enable_memory
        if enable_memory:
            # memory lives on the attic/pre-v0.6 branch; opt-in requires it
            try:
                from frontier_ops.memory.activation import (
                    AutoActivator,
                    MemoryEventBus,
                    MemoryEventType,
                )
                from frontier_ops.memory.vsa import VSAMemory
            except ImportError as e:
                raise ImportError(
                    "enable_memory=True requires the memory package, which "
                    "moved to the attic/pre-v0.6 branch (no measured "
                    "detection value — docs/LEGACY.md)."
                ) from e
            self.event_bus = MemoryEventBus()
            self.activator = AutoActivator(event_bus=self.event_bus)
            self.memory = VSAMemory(dim=vsa_dim)
            self._primed: list = []
            self._novelty = False
            self.event_bus.subscribe(
                lambda e: self._primed.extend(e.activated_traces),
                event_types={MemoryEventType.ACTIVATION},
            )
            self.event_bus.subscribe(
                lambda e: setattr(self, '_novelty', True),
                event_types={MemoryEventType.NOVELTY},
            )

        # Governance. Imported lazily: chain.py needs the optional
        # [governance] extra (cryptography), and importing it at module
        # level broke `import frontier_ops` on numpy-only installs.
        self._enable_governance = enable_governance
        if enable_governance:
            from frontier_ops.governance.chain import GovernanceChain

            self.governance = GovernanceChain()

        # Trajectory-smoothed position for temporal cross-term conjunction.
        # Pointwise cross-terms only fire when ONE action is jointly elevated
        # (e.g. cat ~/.ssh/config is both cred and scope). A probe sequence
        # that alternates dimensions (enum, then cred, then recon) never has
        # a jointly-elevated single action — the conjunction exists across
        # steps. alpha=0.2 (~5-step memory) captures that window; on the
        # 2026-03-20 real traces it activates only inside the probe burst
        # (0 of 679 benign steps).
        self.traj_alpha = 0.2
        self._traj_mu: Optional[np.ndarray] = None

        # State
        self._step = 0
        self._prev_concept: Optional[np.ndarray] = None
        self._angular_disp_acc = 0.0
        self._trajectory: List[np.ndarray] = []

    def process_user_message(self, message: str) -> AuthorizationEvent:
        """
        Process a user message to establish/update authorization scope.

        Call this BEFORE process_step() when a new user message arrives.
        This extracts the goal, classifies the scope operator, updates
        the authorization envelope, records provenance, and triggers
        budget replenishment if applicable.
        """
        # Layer 1-3: Update authorization state
        event = self.auth_state.process_user_message(message)

        # Record in provenance graph
        prov_node = self.provenance.add_directive(
            user_message=message,
            scope_operator=event.operator.value,
            goal_confidence=event.goal_after.confidence,
            budget_replenished=event.budget_replenished,
        )

        # Link budget replenishment to authorization event
        if event.budget_replenished:
            replenish_event = self.auth_budget.on_authorization_event(
                operator=event.operator.value,
                directive_node_id=prov_node.id,
                goal_confidence=event.goal_after.confidence,
            )
            event.replenish_amount = replenish_event.amount

        return event

    def process_step(self, text: str) -> StepResult:
        """Process one agent output through the full pipeline."""
        self._step += 1

        # 1. Concept extraction
        concept_scores = self.extractor.extract(text)
        concept_vec = np.array([concept_scores.get(c, 0.0) for c in self.dim_names])

        # 2. Metric at current position
        G = self.metric.tensor_at(concept_vec)
        metric_trace = float(np.trace(G))
        proximities = self.metric.boundary_proximity(concept_vec)
        cross_acts = self.metric.cross_term_activations(concept_vec)

        # 3. Efference copy
        self.predictor.set_metric_tensor(G)
        predicted = self.predictor.predict_next()
        pred_error = self.predictor.compute_error(predicted, concept_vec)
        proprio_context = pred_error.to_proprioceptive_context(self.dim_names)
        self.predictor.update(concept_vec)

        # 4. Angular displacement delta
        if self._prev_concept is not None:
            diff = concept_vec - self._prev_concept
            angular_disp_delta = float(np.sqrt(max(0, diff @ G @ diff)))
        else:
            angular_disp_delta = 0.0
        self._angular_disp_acc += angular_disp_delta

        # 5. NEWMA
        newma_div, newma_alarm = self.newma.update(concept_vec, G)

        # 5b. Drift classification (disambiguate NEWMA signal)
        drift_result = self.drift_classifier.update(newma_div)
        drift_classification = drift_result["classification"]
        if drift_result["suppress_alert"]:
            newma_alarm = False

        # 5c. Trajectory cross-terms: conjunction across recent steps
        if self._traj_mu is None:
            self._traj_mu = concept_vec.copy()
        else:
            self._traj_mu = (
                self.traj_alpha * concept_vec + (1 - self.traj_alpha) * self._traj_mu
            )
        traj_cross_acts = self.metric.cross_term_activations(self._traj_mu)

        # 6. Trend
        self.trend.observe(concept_vec)
        trend_alerts = self.trend.detect()

        # 7. EWMA
        ewma_dev, ewma_alarm = self.ewma.update(concept_vec, G)

        # 8. Lambda update
        lam = self.lagrangian.update(angular_disp_delta)

        # 9. Memory activation
        primed = []
        novelty = False
        if self._enable_memory:
            self._primed = []
            self._novelty = False
            from frontier_ops.memory.vsa import phasor_encode

            c_phasor = phasor_encode(concept_vec[:3].tobytes().hex(), self.memory.dim)
            r_phasor = phasor_encode("reasoning", self.memory.dim)
            self.activator.perceive(c_phasor, r_phasor, step=self._step)
            primed = self._primed
            novelty = self._novelty
            # Encode into memory
            trace = self.memory.encode(text[:200], str(concept_vec[:2]), "reasoning",
                                       self._step, step=self._step)
            self.activator.register_trace(trace, c_phasor, r_phasor)

        # 10. Authorization check (Layers 1-3)
        auth_result = self.auth_state.check_action(concept_vec)

        # Record action in provenance graph
        auth_verdict = "no_goal"
        if auth_result.get("no_goal"):
            auth_verdict = "no_goal"
        elif auth_result["needs_clarification"]:
            auth_verdict = "needs_clarification"
        elif auth_result["needs_escalation"]:
            auth_verdict = "escalate"
        elif auth_result["authorized"]:
            auth_verdict = "pass"
        else:
            auth_verdict = "block"

        # Which fixed dimensions this action's position crossed, so the
        # provenance certificate can actually check that claim rather than
        # reporting it as unrecorded.
        entered_fixed = [
            c for c in RESTRICTED_DIMENSIONS
            if c in self._boundary_thresholds
            and c in self.dim_names
            and concept_vec[self.dim_names.index(c)] >= self._boundary_thresholds[c]
        ]

        self.provenance.add_action(
            action_content=text[:200],
            tool="",
            authorized=auth_result["authorized"],
            geodesic_distance=auth_result["geodesic_distance"],
            verdict=auth_verdict,
            entered_restricted_dims=entered_fixed,
        )

        # 11. Combined alert level
        alert_level, alert_reasons = self._compute_alert(
            pred_error, proximities, cross_acts, newma_alarm, newma_div,
            trend_alerts, ewma_alarm, traj_cross_acts,
        )

        # Authorization escalation is surfaced via `needs_escalation` and a
        # reason string; it no longer OVERWRITES alert_level. The old
        # `max(alert_level, 0.6)` fired on ~98% of actions and flattened the
        # ranking (alert_level drift AUROC 0.75 -> 0.59; see
        # eval/results/repo-open-questions-2026-07-10.md Q1). Gate decisions
        # on `needs_escalation`; rank on `alert_level`.
        if auth_result["needs_escalation"]:
            alert_reasons.append(
                f"auth:outside_radius(d={auth_result['geodesic_distance']:.2f}>"
                f"r={auth_result['radius']:.2f})"
            )
        if auth_result["needs_clarification"]:
            alert_reasons.append("auth:goal_unclear")

        # 12. Governance
        if self._enable_governance:
            # observe_agent_step expects a StepRecord-like object
            # We'll just append to chain directly for now
            self.governance.observe({
                "step": self._step,
                "alert_level": alert_level,
                "angular_disp": self._angular_disp_acc,
            })

        # Update state
        self._prev_concept = concept_vec.copy()
        self._trajectory.append(concept_vec)

        return StepResult(
            step=self._step,
            text=text[:200],
            concept_vec=concept_vec,
            concept_scores=concept_scores,
            predicted_vec=predicted,
            pred_error=pred_error,
            proprioceptive_context=proprio_context,
            metric_tensor=G,
            metric_trace=metric_trace,
            boundary_proximities=proximities,
            cross_term_activations=cross_acts,
            angular_disp_delta=angular_disp_delta,
            angular_disp_cumulative=self._angular_disp_acc,
            newma_divergence=newma_div,
            newma_alarm=newma_alarm,
            drift_classification=drift_classification,
            trajectory_cross_terms=traj_cross_acts,
            trend_alerts=trend_alerts,
            lambda_value=lam,
            budget_remaining=self.lagrangian.budget_remaining,
            primed_memories=primed,
            novelty=novelty,
            alert_level=alert_level,
            alert_reasons=alert_reasons,
            authorized=auth_result["authorized"],
            geodesic_distance=auth_result["geodesic_distance"],
            authorization_radius=auth_result["radius"],
            goal_confidence=auth_result["goal_confidence"],
            needs_escalation=auth_result["needs_escalation"],
            needs_clarification=auth_result["needs_clarification"],
            authorization_verdict=auth_verdict,
        )

    def _compute_alert(self, pred_error, proximities, cross_acts,
                       newma_alarm, newma_div, trend_alerts, ewma_alarm,
                       traj_cross_acts=None) -> tuple:
        """Combine detector signals via Bayes factor evidence combination.

        Each detector's continuous signal is converted to a p-value via
        p = exp(-k * signal), then combined through the Vovk-Sellke bound.
        The combined Bayes factor is mapped to alert_level in [0, 1] via
        alert_level = log10(BF) / 3, so BF=1000 (decisive) maps to 1.0.
        """
        p_values = {}
        reasons = []

        # NOTE: both efference-copy channels — "surprise ratio" and
        # "direction error" — were REMOVED from the alert combination (v3).
        # On real drift each was at or below chance as a standalone signal
        # (only-surprise AUROC 0.429, only-direction 0.469) and surprise
        # contributed the most benign-side evidence of any channel (+1.31
        # log10 BF per benign action); removing them lifted the recombined
        # ensemble 0.75 -> 0.85 (eval/results/repo-open-questions-2026-07-10.md
        # Q1). The EfferenceCopyPredictor still runs and its outputs
        # (surprise_ratio, direction_error, proprioceptive_context) remain on
        # StepResult for diagnostics — efference is simply no longer a
        # detection channel.

        # Boundary proximity (0-1; >0.5 notable)
        max_prox = max(proximities.values()) if proximities else 0.0
        if max_prox > 0.2:
            p = math.exp(-3.0 * max_prox)
            p_values["boundary"] = p
            if max_prox > 0.5:
                closest = max(proximities, key=proximities.get)
                reasons.append(f"boundary:{closest}={max_prox:.2f}")

        # Cross-term activations (>0.1 notable)
        max_cross = max(cross_acts.values()) if cross_acts else 0.0
        if max_cross > 0.05:
            p = math.exp(-2.0 * max_cross)
            p_values["cross_term"] = p
            if max_cross > 0.1:
                reasons.append(f"cross_term:{max_cross:.2f}")

        # Trajectory cross-terms (temporal conjunction across recent steps).
        # At traj_alpha=0.2 this is zero on all 679 benign steps of the
        # 2026-03-20 real traces, so any positive activation is strong
        # evidence; k=20 maps the observed probe-burst range (~0.02-0.08)
        # to meaningful-but-not-decisive p-values.
        max_traj_cross = max(traj_cross_acts.values()) if traj_cross_acts else 0.0
        if max_traj_cross > 0.01:
            p = math.exp(-20.0 * max_traj_cross)
            p_values["traj_cross"] = p
            worst = max(traj_cross_acts, key=traj_cross_acts.get)
            reasons.append(f"traj_cross:{worst}={max_traj_cross:.3f}")

        # NEWMA (only when alarm fires, drift classifier hasn't suppressed)
        if newma_alarm:
            p = math.exp(-10.0 * newma_div)
            p_values["newma"] = p
            reasons.append(f"newma:{newma_div:.3f}")

        # Trend (slope * r²)
        if trend_alerts:
            max_trend = max(abs(a.slope) * a.r_squared for a in trend_alerts)
            p = math.exp(-50.0 * max_trend)
            p_values["trend"] = p
            reasons.append(f"trend:{trend_alerts[0].dimension}")

        # EWMA (binary alarm → fixed p-value)
        if ewma_alarm:
            p_values["ewma"] = 0.05
            reasons.append("ewma_alarm")

        if not p_values:
            return 0.0, []

        evidence = self.bayes_combiner.combine(p_values)
        alert_level = max(0.0, min(1.0, evidence.log_bf / 3.0))

        if evidence.log_bf > 0:
            reasons.append(f"bf:{evidence.combined_bf:.1f}({evidence.interpretation})")

        return alert_level, reasons

    def reset(self):
        """Reset all state for a new session."""
        self._step = 0
        self._prev_concept = None
        self._angular_disp_acc = 0.0
        self._trajectory = []
        self._traj_mu = None
        self.predictor = EfferenceCopyPredictor(
            n_dims=len(self.dim_names), dim_names=self.dim_names
        )
        self.newma.clear()
        self.drift_classifier.clear()
        self.trend.clear()
        self.ewma.clear()
        if self._enable_memory:
            from frontier_ops.memory.vsa import VSAMemory

            self.memory = VSAMemory(dim=self.memory.dim)
        # Reset authorization (provenance is preserved for audit)
        self.auth_state = AuthorizationState(
            metric=self.metric,
            default_radius=0.5,
        )

    @property
    def stats(self) -> Dict:
        return {
            "step": self._step,
            "angular_disp": self._angular_disp_acc,
            "lambda": self.lagrangian.lam,
            "budget_remaining": self.auth_budget.budget_remaining,
            "budget_fraction": self.auth_budget.budget_fraction,
            "newma_divergence": self.newma.divergence,
            "trajectory_length": len(self._trajectory),
            "concept_extractor": self.extractor.backend_name,
            "authorization": self.auth_state.export_state(),
            "provenance": self.provenance.stats,
        }
