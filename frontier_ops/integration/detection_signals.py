"""Signal computation for runtime reasoning validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from frontier_ops.integration.vsa_core import safe_normalize


@dataclass
class TrajectoryHealth:
    absorption_rate: float
    geodesic_efficiency: float
    curvature: float
    final_confidence: float
    n_iterations: int
    converged: bool
    health_score: float
    regime: str


class FisherTrajectoryEvaluator:
    def __init__(self):
        self.history = deque(maxlen=2000)

    def evaluate(self, trajectory: List[float], converged: bool) -> TrajectoryHealth:
        """Evaluate a resonator trajectory and return health metrics."""
        if len(trajectory) < 2:
            result = TrajectoryHealth(
                absorption_rate=0.0,
                geodesic_efficiency=1.0,
                curvature=0.0,
                final_confidence=float(trajectory[0]) if trajectory else 0.0,
                n_iterations=len(trajectory),
                converged=converged,
                health_score=0.5,
                regime="stuck",
            )
            self.history.append(result)
            return result

        t = np.asarray(trajectory, dtype=float)
        n = len(t)
        x = np.arange(n) / max(n - 1, 1)

        absorption_rate = float(np.polyfit(x, t, 1)[0])
        displacement = abs(float(t[-1] - t[0]))
        total_path = float(np.sum(np.abs(np.diff(t)))) + 1e-10
        geodesic_efficiency = displacement / total_path
        curvature = float(np.mean(np.abs(np.diff(t, 2)))) if n >= 3 else 0.0
        final_confidence = float(t[-1])

        ar_score = np.clip(absorption_rate / 0.3 + 0.5, 0.0, 1.0)
        ge_score = np.clip(geodesic_efficiency, 0.0, 1.0)
        conf_score = np.clip(final_confidence, 0.0, 1.0)
        conv_bonus = 0.20 if converged else 0.0
        curv_penalty = np.clip(curvature * 5.0, 0.0, 0.30)

        health_score = float(
            np.clip(
                0.30 * ar_score
                + 0.25 * ge_score
                + 0.25 * conf_score
                + conv_bonus
                - curv_penalty,
                0.0,
                1.0,
            )
        )

        if converged and absorption_rate > 0.05:
            regime = "convergent"
        elif absorption_rate < -0.05:
            regime = "chaotic"
        elif geodesic_efficiency < 0.3:
            regime = "wandering"
        elif not converged and absorption_rate < 0.01:
            regime = "stuck"
        else:
            regime = "searching"

        result = TrajectoryHealth(
            absorption_rate=absorption_rate,
            geodesic_efficiency=float(geodesic_efficiency),
            curvature=float(curvature),
            final_confidence=final_confidence,
            n_iterations=n,
            converged=converged,
            health_score=health_score,
            regime=regime,
        )
        self.history.append(result)
        return result


class PersistenceTracker:
    def __init__(self, similarity_threshold: float = 0.995):
        self.similarity_threshold = similarity_threshold
        self.prev_filler = None
        self.steps_since_change = 0
        self.change_intervals = deque(maxlen=300)
        self.near_identity_count = 0
        self.historical_magnitude = deque(maxlen=300)

    def observe(self, filler, algebra) -> Dict[str, float]:
        """Track filler persistence and velocity, returning anomaly scores."""
        identity = algebra.identity()
        magnitude = 1.0 - algebra.similarity(filler, identity)
        self.historical_magnitude.append(magnitude)

        if magnitude < 0.1:
            self.near_identity_count += 1
        else:
            self.near_identity_count = 0

        if self.prev_filler is not None:
            sim = algebra.similarity(filler, self.prev_filler)
            if sim > self.similarity_threshold:
                self.steps_since_change += 1
            else:
                self.change_intervals.append(self.steps_since_change)
                self.steps_since_change = 0

        self.prev_filler = filler.copy()
        return {
            "persistence_anomaly": self._persistence_anomaly(),
            "velocity_drought": self._velocity_drought(),
            "steps_since_change": float(self.steps_since_change),
        }

    def _persistence_anomaly(self) -> float:
        """Z-score of current stagnation length relative to historical change intervals."""
        if len(self.change_intervals) < 4:
            return 0.0
        values = np.asarray(self.change_intervals, dtype=float)
        mean = float(np.mean(values))
        std = float(max(np.std(values), 1.0))
        if mean < 1.0:
            return 0.0
        z = (self.steps_since_change - mean) / std
        return float(np.clip(z / 5.0, 0.0, 1.0))

    def _velocity_drought(self) -> float:
        """Detect prolonged runs of near-zero magnitude (filler stuck near identity)."""
        if len(self.historical_magnitude) < 12:
            return 0.0
        hist = np.asarray(self.historical_magnitude, dtype=float)
        near_zero_rate = float(np.mean(hist < 0.1))
        if self.near_identity_count < 3:
            return 0.0
        # expected_run handles high near_zero_rate naturally (→ large expected run → near-zero score)
        expected_run = max(1.0, 1.0 / max(1.0 - near_zero_rate, 0.01))
        score = (self.near_identity_count - expected_run) / max(expected_run, 1.0)
        return float(np.clip(score / 5.0, 0.0, 1.0))


class CUSUMDetector:
    def __init__(
        self,
        warmup_steps: int = 15,
        threshold_h: float = 3.0,
        slack_k: float = 0.02,
        decay: float = 0.95,
        ceiling: float = 20.0,
    ):
        self.warmup_steps = warmup_steps
        self.threshold_h = threshold_h
        self.slack_k = slack_k
        # Exponential decay: each step, accumulators shrink by this factor before
        # the new observation is added. Prevents unbounded accumulation across long
        # sessions. decay=0.95 gives half-life ≈ 14 steps (~4 min at normal pace).
        # Equilibrium at z/(1-decay) = 20*z, so even persistent z=0.5 caps at 10.
        # Set to 1.0 to disable decay (original behavior).
        self.decay = decay
        # Hard ceiling on the raw accumulator value — defense-in-depth.
        self.ceiling = ceiling
        self.warmup_values = []
        self.reference = None
        self.reference_std = None
        self.cusum_high = 0.0
        self.cusum_low = 0.0

    def reset(self) -> None:
        """Reset accumulators for a new session (call on session_start)."""
        self.warmup_values = []
        self.reference = None
        self.reference_std = None
        self.cusum_high = 0.0
        self.cusum_low = 0.0

    def observe(self, value: float) -> Dict[str, float]:
        """Update CUSUM accumulators with a new value, returning the alarm score."""
        if len(self.warmup_values) < self.warmup_steps:
            self.warmup_values.append(float(value))
            if len(self.warmup_values) == self.warmup_steps:
                self.reference = float(np.mean(self.warmup_values))
                self.reference_std = float(max(np.std(self.warmup_values), 0.001))
            return {"cusum_score": 0.0, "alarm": 0.0, "phase": "warmup"}

        z = (value - self.reference) / self.reference_std
        # Apply decay then update (prevents unbounded accumulation)
        self.cusum_high = min(
            self.ceiling, max(0.0, self.cusum_high * self.decay + z - self.slack_k)
        )
        self.cusum_low = min(
            self.ceiling, max(0.0, self.cusum_low * self.decay - z - self.slack_k)
        )
        max_cusum = max(self.cusum_high, self.cusum_low)
        score = max_cusum / self.threshold_h
        return {
            "cusum_score": float(score),
            "alarm": 1.0 if max_cusum > self.threshold_h else 0.0,
            "phase": "active",
        }


class Resonator:
    def __init__(
        self, algebra, max_iters: int = 25, threshold: float = 0.85, blend: float = 0.4
    ):
        self.algebra = algebra
        self.max_iters = max_iters
        self.threshold = threshold
        self.blend = blend

    def search(self, query, codebook: List[np.ndarray]):
        """Iteratively search the codebook for the best match to query."""
        if not codebook:
            return query.copy(), [], False

        cb = np.asarray(codebook)
        estimate = query.copy()
        trajectory: List[float] = []

        for _ in range(self.max_iters):
            sims = np.asarray(
                [self.algebra.similarity(estimate, item) for item in cb], dtype=float
            )
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            trajectory.append(best_sim)
            if best_sim > self.threshold:
                return cb[best_idx].copy(), trajectory, True
            estimate = safe_normalize(
                (1.0 - self.blend) * estimate + self.blend * cb[best_idx]
            )

        sims = np.asarray(
            [self.algebra.similarity(query, item) for item in cb], dtype=float
        )
        best_idx = int(np.argmax(sims))
        return cb[best_idx].copy(), trajectory, False


class SlotPredictor:
    def __init__(
        self,
        algebra,
        role_name: str,
        max_codebook: int = 200,
        resonator_iters: int = 30,
        resonator_threshold: float = 0.85,
        base_novelty: float = 0.85,
    ):
        self.algebra = algebra
        self.role_name = role_name
        self.codebook = deque(maxlen=max_codebook)
        self.prev_filler = None
        self.history = deque(maxlen=400)
        self.recent_similarity = deque(maxlen=100)

        self.resonator = Resonator(
            algebra,
            max_iters=resonator_iters,
            threshold=resonator_threshold,
            blend=0.4,
        )
        self.base_novelty = base_novelty

        self.persistence = PersistenceTracker()
        self.cusum = CUSUMDetector(warmup_steps=15, threshold_h=3.0, slack_k=0.02)
        self.fisher = FisherTrajectoryEvaluator()

    def _predict_query(self):
        if len(self.history) >= 2:
            h = list(self.history)
            return safe_normalize(h[-1] + 0.3 * (h[-1] - h[-2]))
        if len(self.history) == 1:
            return self.history[-1].copy()
        if self.prev_filler is not None:
            return self.prev_filler.copy()
        return self.algebra.identity()

    def _adaptive_novelty_threshold(self) -> float:
        if len(self.recent_similarity) < 8:
            return self.base_novelty
        avg_sim = float(np.mean(self.recent_similarity))
        if avg_sim > 0.98:
            return min(0.93, self.base_novelty + 0.05)
        if avg_sim < 0.85:
            return max(0.80, self.base_novelty - 0.03)
        return self.base_novelty

    def observe(self, actual_filler) -> Dict[str, object]:
        """Observe a new filler, predict via resonator, and compute error/health signals."""
        query = self._predict_query()
        if len(self.codebook) < 2:
            predicted = query
            trajectory = [self.algebra.similarity(query, actual_filler)]
            converged = False
        else:
            predicted, trajectory, converged = self.resonator.search(
                query, list(self.codebook)
            )

        error = 1.0 - self.algebra.similarity(predicted, actual_filler)

        if self.prev_filler is not None:
            self.recent_similarity.append(
                self.algebra.similarity(self.prev_filler, actual_filler)
            )

        persistence_meta = self.persistence.observe(actual_filler, self.algebra)
        cusum_meta = self.cusum.observe(error)
        fisher_health = self.fisher.evaluate(trajectory, converged)

        novelty_threshold = self._adaptive_novelty_threshold()
        is_novel = True
        if self.codebook:
            max_sim = max(
                self.algebra.similarity(item, actual_filler) for item in self.codebook
            )
            if max_sim > novelty_threshold:
                is_novel = False
        if is_novel:
            self.codebook.append(actual_filler.copy())

        self.prev_filler = actual_filler.copy()
        self.history.append(actual_filler.copy())

        return {
            "error": float(error),
            "trajectory": trajectory,
            "converged": converged,
            "fisher_health": float(fisher_health.health_score),
            "fisher_regime": fisher_health.regime,
            "persistence": persistence_meta,
            "cusum": cusum_meta,
            "codebook_size": len(self.codebook),
        }


class CrossSlotConsistency:
    def __init__(
        self,
        algebra,
        slot_names: List[str],
        max_codebook: int = 300,
        resonator_iters: int = 20,
        resonator_threshold: float = 0.80,
        novelty_threshold: float = 0.90,
    ):
        self.algebra = algebra
        self.slot_names = slot_names
        self.novelty_threshold = novelty_threshold
        self.codebook = deque(maxlen=max_codebook)
        self.resonator = Resonator(
            algebra,
            max_iters=resonator_iters,
            threshold=resonator_threshold,
            blend=0.4,
        )
        self.fisher = FisherTrajectoryEvaluator()

    def observe(self, fillers: Dict[str, np.ndarray]) -> Dict[str, object]:
        """Measure cross-slot binding consistency against historical patterns."""
        available = [fillers[name] for name in self.slot_names if name in fillers]
        if len(available) < 2:
            return {
                "consistency_score": 1.0,
                "converged": True,
                "trajectory": [1.0],
                "health": None,
            }

        joint = available[0]
        for value in available[1:]:
            joint = self.algebra.bind(joint, value)

        if len(self.codebook) < 2:
            self.codebook.append(joint.copy())
            return {
                "consistency_score": 1.0,
                "converged": True,
                "trajectory": [1.0],
                "health": None,
            }

        max_sim = max(self.algebra.similarity(joint, prior) for prior in self.codebook)
        query = joint.copy()
        _, trajectory, converged = self.resonator.search(query, list(self.codebook))

        if max_sim < self.novelty_threshold:
            self.codebook.append(joint.copy())

        health = (
            self.fisher.evaluate(trajectory, converged)
            if len(trajectory) >= 2
            else None
        )
        # Use raw nearest-neighbor similarity as consistency.
        # This keeps novelty visible; the resonator trajectory is used for Fisher.
        consistency_score = float(max_sim)
        return {
            "consistency_score": consistency_score,
            "converged": converged,
            "trajectory": trajectory,
            "health": health,
            "max_similarity": float(max_sim),
            "codebook_size": len(self.codebook),
        }


class DetectionSignalEngine:
    def __init__(
        self,
        algebra,
        role_names: List[str],
        cross_slot_roles: Optional[List[str]] = None,
        resonator_iters: int = 30,
        resonator_threshold: float = 0.85,
    ):
        self.algebra = algebra
        self.role_names = role_names
        self.slot_predictors = {
            role: SlotPredictor(
                algebra,
                role,
                resonator_iters=resonator_iters,
                resonator_threshold=resonator_threshold,
                base_novelty=0.85,
            )
            for role in role_names
        }

        if cross_slot_roles is None:
            cross_slot_roles = [
                "action_type",
                "scope",
                "source",
                "target_sensitivity",
                "magnitude",
            ]
        self.cross_slot = CrossSlotConsistency(algebra, cross_slot_roles)

    def observe(self, fillers: Dict[str, np.ndarray]) -> Dict[str, object]:
        """Run all slot predictors and cross-slot checks, returning composite signals."""
        slot_meta: Dict[str, Dict[str, object]] = {}
        per_slot_errors: Dict[str, float] = {}
        per_slot_fisher: Dict[str, float] = {}

        for role, value in fillers.items():
            if role not in self.slot_predictors:
                continue
            meta = self.slot_predictors[role].observe(value)
            slot_meta[role] = meta
            per_slot_errors[role] = float(meta["error"])
            per_slot_fisher[role] = float(meta["fisher_health"])

        cross_meta = self.cross_slot.observe(fillers)

        mean_error = (
            float(np.mean(list(per_slot_errors.values()))) if per_slot_errors else 0.0
        )
        fisher_vals = [
            v
            for role, v in per_slot_fisher.items()
            if slot_meta[role]["trajectory"] and len(slot_meta[role]["trajectory"]) >= 2
        ]
        fisher_signal = 1.0 - min(fisher_vals) if fisher_vals else 0.0

        persistence_vals = []
        for role, meta in slot_meta.items():
            p = meta.get("persistence", {})
            persistence_vals.append(
                max(p.get("persistence_anomaly", 0.0), p.get("velocity_drought", 0.0))
            )
        persistence_signal = float(max(persistence_vals)) if persistence_vals else 0.0

        cusum_vals = []
        for role, meta in slot_meta.items():
            c = meta.get("cusum", {})
            cusum_vals.append(float(c.get("cusum_score", 0.0)))
        cusum_signal = float(np.mean(cusum_vals)) if cusum_vals else 0.0

        cross_slot_signal = 1.0 - float(cross_meta.get("consistency_score", 1.0))

        raw_signals = {
            "error": float(mean_error),
            "fisher": float(np.clip(fisher_signal, 0.0, 1.0)),
            "cross_slot": float(np.clip(cross_slot_signal, 0.0, 1.0)),
            "persistence": float(np.clip(persistence_signal, 0.0, 1.0)),
            "cusum": float(max(cusum_signal, 0.0)),
        }

        return {
            "raw_signals": raw_signals,
            "per_slot": slot_meta,
            "cross_slot": cross_meta,
        }
