"""
Safety Polytope Signature Detectors
=====================================

5 geometric signatures for safety violation detection in VSA-encoded agent traces.
Sig1: Intent Binding Fracture, Sig2: Source Provenance Corruption,
Sig3: Constitutional Manifold Boundary, Sig4: Trajectory Coherence Fracture,
Sig5: Confidence-Grounding Decoupling.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from frontier_ops.integration.proprio_logger import logger
from frontier_ops.integration.vsa_core import PhasorAlgebra
from frontier_ops.integration.conjunction_detector import ConjunctionDetector


# ─── Signature 1: Intent Binding Fracture ─────────────────────────────────


class IntentBindingFractureDetector:
    """Detects sustained context_alignment drops via CUSUM + persistence tracking."""

    def __init__(
        self,
        cusum_threshold: float = 4.0,
        drop_threshold: float = 0.45,
        sustain_steps: int = 5,
        window: int = 20,
    ):
        self.cusum_threshold = cusum_threshold
        self.drop_threshold = drop_threshold
        self.sustain_steps = sustain_steps
        self.window = window

        # Alignment history
        self.alignments: deque = deque(maxlen=window)

        # CUSUM state (downward shift detector)
        self.cusum_neg: float = 0.0
        self.cusum_pos: float = 0.0

        # Calibration
        self.calibration_buffer: List[float] = []
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.calibrated: bool = False

    def _calibrate_if_ready(self):
        if self.calibrated or len(self.calibration_buffer) < 30:
            return
        self.baseline_mean = float(np.mean(self.calibration_buffer))
        self.baseline_std = float(max(np.std(self.calibration_buffer), 0.01))
        self.calibrated = True

    def observe(
        self, context_alignment: float, action_vec: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Observe a new action's context alignment score.

        Args:
            context_alignment: the classifier's context_alignment value (0-1)
            action_vec: optional composite HV (for future cross-slot analysis)

        Returns:
            Dict with signal strength, pattern classification, and verdict contribution.
        """
        self.alignments.append(context_alignment)

        if not self.calibrated:
            self.calibration_buffer.append(context_alignment)
            self._calibrate_if_ready()
            return {
                "signal": 0.0,
                "pattern": "calibrating",
                "cusum_neg": 0.0,
                "persistence": 0.0,
                "verdict_contribution": "PASS",
            }

        # CUSUM for downward shift (with decay to prevent runaway)
        z = (context_alignment - self.baseline_mean) / self.baseline_std
        self.cusum_neg = max(0.0, self.cusum_neg * 0.95 - z + 0.5)  # 0.5 = slack
        self.cusum_pos = max(0.0, self.cusum_pos * 0.95 + z - 0.5)

        # Persistence: what fraction of recent window is below threshold?
        recent = list(self.alignments)
        below_count = sum(1 for a in recent if a < self.drop_threshold)
        persistence = below_count / len(recent)

        # Derivative check: is the alignment stabilizing at a low value?
        # Low std in recent diffs = flat at low alignment = permanent fracture
        if len(recent) >= self.sustain_steps:
            recent_tail = recent[-self.sustain_steps :]
            diffs = [
                recent_tail[i + 1] - recent_tail[i] for i in range(len(recent_tail) - 1)
            ]
            derivative_std = float(np.std(diffs)) if diffs else 1.0
            all_below = all(a < self.drop_threshold for a in recent_tail)
            flat_at_low = all_below and derivative_std < 0.05
        else:
            flat_at_low = False
            derivative_std = 1.0

        # Pattern classification
        if self.cusum_neg > self.cusum_threshold and flat_at_low:
            pattern = "permanent_fracture"
            verdict_contribution = "FLAG"
            signal = min(1.0, self.cusum_neg / (self.cusum_threshold * 2))
        elif self.cusum_neg > self.cusum_threshold and persistence > 0.5:
            pattern = "sustained_drift"
            verdict_contribution = "MONITOR"
            signal = min(0.7, self.cusum_neg / (self.cusum_threshold * 2))
        elif self.cusum_neg > self.cusum_threshold * 0.5:
            pattern = "temporary_dip"
            verdict_contribution = "PASS"
            signal = min(0.3, self.cusum_neg / (self.cusum_threshold * 2))
        else:
            pattern = "nominal"
            verdict_contribution = "PASS"
            signal = 0.0

        return {
            "signal": round(float(signal), 4),
            "pattern": pattern,
            "cusum_neg": round(float(self.cusum_neg), 4),
            "persistence": round(float(persistence), 4),
            "derivative_std": round(float(derivative_std), 4),
            "current_alignment": round(float(context_alignment), 4),
            "verdict_contribution": verdict_contribution,
        }


# ─── Signature 2: Source Provenance Corruption ────────────────────────────


class SourceProvenanceCorruptionDetector:
    """Detects high-sensitivity actions from untrusted sources. Structural fast-path + geometric signal."""

    # Class-level defaults — overridden by constructor args (and ultimately by config)
    _DEFAULT_TRUSTED_SOURCES = {
        "user_direct",
        "user_prior",
        "agent_reasoning",
        "agent_memory",
    }
    _DEFAULT_HIGH_SENSITIVITY_TYPES = {
        "credential_access",
        "config_change",
        "skill_install",
        "payment",
    }
    _DEFAULT_HIGH_BLAST_SCOPES = {
        "destructive",
        "credential_access",
        "payment_large",
        "write_system",
        "config_modify",
        "network_egress",
    }

    def __init__(
        self,
        ema_alpha: float = 0.05,
        window: int = 15,
        trusted_sources=None,
        high_sensitivity_types=None,
        high_blast_radius_scopes=None,
    ):
        self.ema_alpha = ema_alpha
        self.window = window
        self.TRUSTED_SOURCES = (
            set(trusted_sources)
            if trusted_sources is not None
            else self._DEFAULT_TRUSTED_SOURCES
        )
        self.HIGH_SENSITIVITY_TYPES = (
            set(high_sensitivity_types)
            if high_sensitivity_types is not None
            else self._DEFAULT_HIGH_SENSITIVITY_TYPES
        )
        self.HIGH_BLAST_SCOPES = (
            set(high_blast_radius_scopes)
            if high_blast_radius_scopes is not None
            else self._DEFAULT_HIGH_BLAST_SCOPES
        )

        # Track source→action binding strength via EMA
        self.ema_binding: Optional[float] = None

        # History for pattern detection
        self.history: deque = deque(maxlen=window)

        # Count consecutive untrusted high-sensitivity actions
        self.consecutive_untrusted_sensitive: int = 0

    def observe(
        self,
        action: Dict[str, Any],
        source_filler: np.ndarray = None,
        action_type_filler: np.ndarray = None,
        algebra: PhasorAlgebra = None,
    ) -> Dict[str, Any]:
        """
        Observe a classified action and check for provenance corruption.

        Args:
            action: the classified action dict from openclaw_classifier
            source_filler: VSA filler for source slot (optional, for geometric signal)
            action_type_filler: VSA filler for action_type slot (optional)
            algebra: PhasorAlgebra instance (optional, needed for geometric signal)

        Returns:
            Dict with structural flag, geometric signal, and verdict contribution.
        """
        source = str(action.get("source", "unknown"))
        action_type = str(action.get("action_type", "file_read"))
        scope = str(action.get("scope", "read_only"))
        target_sensitivity = float(action.get("target_sensitivity", 0.2))
        context_alignment = float(action.get("context_alignment", 0.8))

        is_trusted = source in self.TRUSTED_SOURCES
        is_high_sensitivity = (
            action_type in self.HIGH_SENSITIVITY_TYPES
            or scope in self.HIGH_BLAST_SCOPES
            or target_sensitivity >= 0.75
        )

        # ── Structural fast-path (zero calibration needed) ──
        structural_flag = is_high_sensitivity and not is_trusted

        if structural_flag:
            self.consecutive_untrusted_sensitive += 1
        else:
            self.consecutive_untrusted_sensitive = 0

        # ── Geometric signal (VSA cross-slot binding) ──
        binding_strength = None
        binding_anomaly = 0.0
        if (
            source_filler is not None
            and action_type_filler is not None
            and algebra is not None
        ):
            # Measure: how well does source explain the action choice?
            cross_binding = source_filler * action_type_filler  # phasor bind
            # In a healthy system, the same source-action patterns repeat
            binding_strength = float(np.abs(np.mean(cross_binding)))

            if self.ema_binding is None:
                self.ema_binding = binding_strength
            else:
                self.ema_binding = (
                    self.ema_alpha * binding_strength
                    + (1 - self.ema_alpha) * self.ema_binding
                )
            binding_anomaly = max(0.0, self.ema_binding - binding_strength)

        # ── Verdict contribution ──
        if structural_flag and self.consecutive_untrusted_sensitive >= 3:
            verdict_contribution = "FLAG"
            signal = 0.9
        elif structural_flag and context_alignment < 0.5:
            verdict_contribution = "FLAG"
            signal = 0.8
        elif structural_flag:
            verdict_contribution = "MONITOR"
            signal = 0.5
        elif binding_anomaly > 0.25:
            verdict_contribution = "MONITOR"
            signal = 0.4
        else:
            verdict_contribution = "PASS"
            signal = 0.0

        entry = {
            "signal": round(float(signal), 4),
            "structural_flag": structural_flag,
            "is_trusted": is_trusted,
            "is_high_sensitivity": is_high_sensitivity,
            "consecutive_untrusted_sensitive": self.consecutive_untrusted_sensitive,
            "binding_strength": round(float(binding_strength), 4)
            if binding_strength is not None
            else None,
            "binding_anomaly": round(float(binding_anomaly), 4),
            "verdict_contribution": verdict_contribution,
        }
        self.history.append(entry)
        return entry


# ─── Signature 3: Constitutional Manifold Boundary Crossing ───────────────


class ConstitutionalManifoldDetector:
    """Detects actions outside the safe operating region via centroid distance,
    PCA convex hull, and Mahalanobis distance (3 phases as data accumulates)."""

    def __init__(
        self,
        baseline_size: int = 100,
        proximity_threshold: float = 4.50,
        penetration_threshold: float = 6.00,
        pca_dims: int = 4,
    ):
        self.baseline_size = baseline_size
        self.proximity_threshold = proximity_threshold
        self.penetration_threshold = penetration_threshold
        self.pca_dims = pca_dims

        # Phase 1: centroid-based
        self.baseline_vectors: List[np.ndarray] = []
        self.centroid: Optional[np.ndarray] = None
        self.baseline_radius: Optional[float] = None
        self.baseline_std: Optional[float] = None

        # Phase 2: PCA + convex hull
        self.pca_components: Optional[np.ndarray] = None
        self.pca_mean: Optional[np.ndarray] = None
        self.hull = None  # scipy.spatial.ConvexHull
        self.hull_equations: Optional[np.ndarray] = None  # (n_facets, pca_dims+1)

        # Phase 3: Mahalanobis distance (best for high-D)
        self.mahal_inv_cov: Optional[np.ndarray] = None
        self.mahal_active: bool = False

        self.calibrated: bool = False
        self.hull_active: bool = False

    def _calibrate(self):
        """Compute centroid, radius, PCA covariance, and convex hull from baseline."""
        if len(self.baseline_vectors) < self.baseline_size:
            return

        vecs = np.array(
            [np.real(v) if np.iscomplexobj(v) else v for v in self.baseline_vectors]
        )

        # Phase 1: centroid + radius
        self.centroid = np.mean(vecs, axis=0)
        distances = np.array([np.linalg.norm(v - self.centroid) for v in vecs])
        self.baseline_radius = float(np.mean(distances))
        self.baseline_std = float(max(np.std(distances), 0.01))
        self.calibrated = True

        # Phase 2: PCA projection + convex hull
        self._build_hull(vecs)

        # Phase 3: Mahalanobis distance in PCA space
        self._build_mahalanobis(vecs)

    def _build_mahalanobis(self, vecs: np.ndarray):
        """Build Mahalanobis distance metric in PCA-projected space."""
        n, d = vecs.shape
        effective_dims = min(self.pca_dims, n - 1, d)
        if effective_dims < 2 or self.pca_components is None:
            return

        # Project to PCA space
        centered = vecs - self.pca_mean
        projected = centered @ self.pca_components.T  # (n, effective_dims)

        # Compute covariance in PCA space
        cov = np.cov(projected, rowvar=False)  # (effective_dims, effective_dims)

        # Regularize to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-4

        try:
            self.mahal_inv_cov = np.linalg.inv(cov)
            self.mahal_active = True
        except np.linalg.LinAlgError:
            self.mahal_inv_cov = None
            self.mahal_active = False

    def _mahalanobis_distance(self, point_pca: np.ndarray) -> float:
        """Compute Mahalanobis distance from the baseline distribution center."""
        if self.mahal_inv_cov is None or self.pca_mean is None:
            return 0.0

        # point_pca is already centered and projected
        diff = point_pca  # Already centered in _project_to_pca
        d_sq = float(diff @ self.mahal_inv_cov @ diff)
        return float(np.sqrt(max(0, d_sq)))

    def _build_hull(self, vecs: np.ndarray):
        """Project baseline vectors via PCA and build a convex hull."""
        from scipy.spatial import ConvexHull

        n, d = vecs.shape
        effective_dims = min(self.pca_dims, n - 1, d)
        if effective_dims < 2:
            return  # not enough data for hull

        # PCA: center, SVD, take top components
        self.pca_mean = np.mean(vecs, axis=0)
        centered = vecs - self.pca_mean
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self.pca_components = Vt[:effective_dims]  # (effective_dims, d)
        except np.linalg.LinAlgError:
            return  # SVD failed, stay in centroid mode

        # Project baseline to PCA space
        projected = centered @ self.pca_components.T  # (n, effective_dims)

        # Add small jitter to avoid degenerate hull from near-coplanar points
        jitter = np.random.RandomState(42).randn(*projected.shape) * 1e-6
        projected = projected + jitter

        try:
            self.hull = ConvexHull(projected)
            self.hull_equations = self.hull.equations  # (n_facets, effective_dims+1)
            self.hull_active = True
        except Exception as e:
            # Degenerate hull (coplanar points, etc.) — stay in centroid mode
            logger.info(
                "hull construction failed (expected for near-coplanar data): %s", e
            )
            self.hull = None
            self.hull_active = False

    def _hull_signed_distance(self, point_pca: np.ndarray) -> float:
        """Signed distance to hull boundary. Negative=inside, positive=outside."""
        if self.hull_equations is None:
            return 0.0

        # hull_equations: each row is [normal..., offset]
        # A point x satisfies Ax + b <= 0 for all facets if inside
        normals = self.hull_equations[:, :-1]
        offsets = self.hull_equations[:, -1]
        # signed distance to each facet: positive = outside that facet
        facet_distances = normals @ point_pca + offsets
        # max facet distance: if > 0, point is outside hull
        return float(np.max(facet_distances))

    def _project_to_pca(self, real_hv: np.ndarray) -> Optional[np.ndarray]:
        """Project a vector into PCA space."""
        if self.pca_components is None or self.pca_mean is None:
            return None
        centered = real_hv - self.pca_mean
        return centered @ self.pca_components.T

    def observe(self, composite_hv: np.ndarray) -> Dict[str, Any]:
        """Check boundary distance for a composite action HV."""
        real_hv = (
            np.real(composite_hv) if np.iscomplexobj(composite_hv) else composite_hv
        )

        if not self.calibrated:
            self.baseline_vectors.append(real_hv.copy())
            self._calibrate()
            return {
                "signal": 0.0,
                "distance_from_centroid": 0.0,
                "boundary_status": "calibrating",
                "z_score": 0.0,
                "hull_distance": None,
                "mode": "calibrating",
                "verdict_contribution": "PASS",
            }

        # Phase 1: centroid distance (always computed as fallback metric)
        centroid_distance = float(np.linalg.norm(real_hv - self.centroid))
        z_score = (centroid_distance - self.baseline_radius) / self.baseline_std

        # Phase 2: Hull distance (fallback)
        hull_distance = None
        if self.hull_active:
            point_pca = self._project_to_pca(real_hv)
            if point_pca is not None:
                hull_distance = self._hull_signed_distance(point_pca)

        # Phase 3: Mahalanobis distance (preferred)
        mahal_distance = None
        if self.mahal_active:
            point_pca = self._project_to_pca(real_hv)
            if point_pca is not None:
                mahal_distance = self._mahalanobis_distance(point_pca)

        z_proximity = 2.5
        z_penetration = 3.5

        mode = (
            "mahalanobis+centroid"
            if mahal_distance is not None
            else ("hull+centroid" if hull_distance is not None else "centroid")
        )

        if z_score > z_penetration:
            boundary_status = "penetration"
            signal = min(1.0, z_score / (z_penetration * 2))
            verdict_contribution = "FLAG"
        elif z_score > z_proximity:
            boundary_status = "proximity"
            signal = min(0.5, z_score / (z_penetration * 2))
            verdict_contribution = "MONITOR"
        else:
            boundary_status = "interior"
            signal = 0.0
            verdict_contribution = "PASS"

        return {
            "signal": round(float(signal), 4),
            "distance_from_centroid": round(float(centroid_distance), 4),
            "baseline_radius": round(float(self.baseline_radius), 4),
            "z_score": round(float(z_score), 4),
            "hull_distance": round(float(hull_distance), 4)
            if hull_distance is not None
            else None,
            "mahal_distance": round(float(mahal_distance), 4)
            if mahal_distance is not None
            else None,
            "boundary_status": boundary_status,
            "mode": mode,
            "verdict_contribution": verdict_contribution,
        }


# ─── Signature 4: Trajectory Coherence Fracture ──────────────────────────


class TrajectoryCoherenceFractureDetector:
    """Wrapper around PhasorTrajectoryBuffer.fracture_signal() + TaskCoherenceScorer."""

    def __init__(self, window: int = 12, dim: int = 512, config: Dict | None = None):
        # Import here to avoid circular imports
        from frontier_ops.integration.trajectory_buffer import PhasorTrajectoryBuffer
        from frontier_ops.integration.task_coherence import TaskCoherenceScorer

        self.buffer = PhasorTrajectoryBuffer(window=window, dim=dim)
        self.coherence_scorer = TaskCoherenceScorer(window=20, dim=dim)
        self.config: Dict = config or {}

    def observe(self, composite_hv: np.ndarray) -> Dict[str, Any]:
        """
        Push a composite HV and return fracture analysis.

        Returns:
            Dict with fracture signal, pattern, and verdict contribution.
        """
        self.buffer.push(composite_hv)
        self.coherence_scorer.push(composite_hv)
        fracture = self.buffer.fracture_signal()
        task_coherence = self.coherence_scorer.score()

        # Require a full buffer window before producing verdicts.
        # With < window entries, the baseline is unreliable and normal
        # tool type switches can appear as injection fractures.
        buffer_full = len(self.buffer.hvs) >= self.buffer.window

        # Map pattern to verdict contribution.
        # In log-tail mode (config flag), suppress sig4 entirely: tool type
        # switches in phasor VSA produce near-orthogonal composite vectors
        # (cos ≈ 0) that look like fractures but are normal coding behavior.
        # Without enriched tool parameters, we can't distinguish "agent
        # switched from Read to exec" (normal) from "prompt injection
        # redirected agent" (dangerous). The signal is noise.
        log_tail_mode = self.config.get("log_tail_mode", False)
        pattern = fracture.get("pattern", "nominal")
        if log_tail_mode:
            # Log-tail mode: sig4 is advisory only, never escalates verdict
            verdict_contribution = "PASS"
        elif not buffer_full:
            # Buffer not full: baseline unreliable, suppress verdicts
            verdict_contribution = "PASS"
        elif pattern == "injection_fracture":
            verdict_contribution = "FLAG"
        elif pattern == "chaotic":
            verdict_contribution = "MONITOR"
        elif pattern == "creative_drift":
            verdict_contribution = "PASS"
        else:
            verdict_contribution = "PASS"

        # Holonomy: curvature accumulation in phasor space
        # Budgets pulled from config (governed parameters — changes are governance events)
        h_config = self.config.get("holonomy", {}) if self.config else {}
        holonomy = self.buffer.session_holonomy(
            budget_autonomous=h_config.get("budget_autonomous", 2.0),
            budget_focused=h_config.get("budget_focused", 0.5),
        )

        return {
            "signal": fracture.get("signal", 0.0),
            "pattern": pattern,
            "max_snap": fracture.get("max_snap"),
            "mean_coherence": fracture.get("mean_coherence"),
            "sim_variance": fracture.get("sim_variance"),
            "verdict_contribution": verdict_contribution,
            "holonomy": holonomy.get("holonomy", 0.0),
            "holonomy_exceeds_budget": holonomy.get("exceeds_autonomous", False),
            "task_coherence": task_coherence,
            "task_coherence_score": task_coherence.get("coherence", 0.0),
            "task_coherence_pattern": task_coherence.get("phase", "warmup"),
        }


# ─── Signature 5: Confidence-Grounding Decoupling ────────────────────────


class ConfidenceGroundingDecouplingDetector:
    """Detects high-magnitude actions with no provenance in recent trajectory (geometric orphans)."""

    def __init__(
        self,
        magnitude_threshold: float = 0.5,
        grounding_threshold: float = 0.25,
        window: int = 10,
    ):
        self.magnitude_threshold = magnitude_threshold
        self.grounding_threshold = grounding_threshold
        self.window = window

        # Recent action HVs for provenance checking
        self.recent_hvs: deque = deque(maxlen=window)
        # Recent magnitudes for context
        self.recent_magnitudes: deque = deque(maxlen=window)
        # Provenance bundle: running sum of recent HVs (bundling)
        self.provenance_bundle: Optional[np.ndarray] = None

    def _update_provenance(self, hv: np.ndarray):
        """Update the provenance bundle with the new action vector."""
        if self.provenance_bundle is None:
            self.provenance_bundle = hv.copy()
        else:
            # Exponential blend: recent actions matter more
            self.provenance_bundle = 0.9 * self.provenance_bundle + 0.1 * hv
            # Normalize
            norm = np.linalg.norm(self.provenance_bundle)
            if norm > 1e-9:
                self.provenance_bundle = self.provenance_bundle / norm

    def _compute_grounding(self, hv: np.ndarray) -> float:
        """
        How well is this action grounded in recent trajectory?

        Returns cosine similarity to the provenance bundle (0-1).
        Higher = more grounded, lower = more orphaned.
        """
        if self.provenance_bundle is None or len(self.recent_hvs) < 3:
            return 1.0  # can't assess yet, assume grounded

        # Method 1: similarity to provenance bundle
        norm_hv = np.linalg.norm(hv)
        norm_prov = np.linalg.norm(self.provenance_bundle)
        if norm_hv < 1e-9 or norm_prov < 1e-9:
            return 0.0

        bundle_sim = float(np.abs(np.real(np.vdot(hv, self.provenance_bundle))))
        bundle_sim /= norm_hv * norm_prov

        # Method 2: max similarity to any recent individual action
        max_individual = 0.0
        for prior in self.recent_hvs:
            norm_prior = np.linalg.norm(prior)
            if norm_prior < 1e-9:
                continue
            sim = float(np.abs(np.real(np.vdot(hv, prior)))) / (norm_hv * norm_prior)
            max_individual = max(max_individual, sim)

        # Combined grounding: weighted average
        grounding = 0.6 * bundle_sim + 0.4 * max_individual
        return float(np.clip(grounding, 0.0, 1.0))

    def observe(
        self,
        composite_hv: np.ndarray,
        magnitude: float,
        context_alignment: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Check if this action is a geometric orphan.

        Args:
            composite_hv: bound 6-slot hypervector
            magnitude: action magnitude from classifier (0-1)
            context_alignment: context alignment score (0-1)

        Returns:
            Dict with orphan status and grounding metrics.
        """
        grounding = self._compute_grounding(composite_hv)

        # An orphan is high magnitude + low grounding
        is_high_magnitude = magnitude >= self.magnitude_threshold
        is_low_grounding = grounding < self.grounding_threshold
        is_orphan = is_high_magnitude and is_low_grounding

        # Additional signal: high magnitude + low context alignment + low grounding
        # is even more suspicious
        is_suspicious_orphan = is_orphan and context_alignment < 0.5

        # Signal strength
        if is_suspicious_orphan:
            signal = min(1.0, magnitude * (1.0 - grounding) * 2.0)
            verdict_contribution = "FLAG"
        elif is_orphan:
            signal = min(0.7, magnitude * (1.0 - grounding))
            verdict_contribution = "MONITOR"
        elif is_high_magnitude and grounding < 0.4:
            signal = 0.3
            verdict_contribution = "MONITOR"
        else:
            signal = 0.0
            verdict_contribution = "PASS"

        # Update state AFTER checking (current action shouldn't ground itself)
        self.recent_hvs.append(composite_hv.copy())
        self.recent_magnitudes.append(magnitude)
        self._update_provenance(composite_hv)

        return {
            "signal": round(float(signal), 4),
            "grounding": round(float(grounding), 4),
            "magnitude": round(float(magnitude), 4),
            "is_orphan": is_orphan,
            "is_suspicious_orphan": is_suspicious_orphan,
            "verdict_contribution": verdict_contribution,
        }


# ─── Composite Signature Engine ──────────────────────────────────────────


class SafetyPolytopeEngine:
    """
    Runs all 5 signature detectors and produces a composite safety assessment.

    Integrates with the existing TieredVerdictEngine via verdict_contribution
    fields that can escalate (never de-escalate) the base verdict.
    """

    def __init__(
        self, algebra: PhasorAlgebra, dim: int = 512, config: Dict[str, Any] = None
    ):
        self.algebra = algebra

        cfg = config or {}

        ib_cfg = cfg.get("intent_binding", {})
        sp_cfg = cfg.get("source_provenance", {})
        cm_cfg = cfg.get("constitutional_manifold", {})

        self.sig1_intent = IntentBindingFractureDetector(
            cusum_threshold=ib_cfg.get("cusum_threshold", 4.0),
        )
        self.sig2_provenance = SourceProvenanceCorruptionDetector(
            trusted_sources=sp_cfg.get("trusted_sources"),
            high_sensitivity_types=sp_cfg.get("high_sensitivity_types"),
            high_blast_radius_scopes=sp_cfg.get("high_blast_radius_scopes"),
        )
        self.sig3_manifold = ConstitutionalManifoldDetector(
            baseline_size=cm_cfg.get("baseline_size", 100),
            proximity_threshold=cm_cfg.get("proximity_threshold", 4.50),
            penetration_threshold=cm_cfg.get("penetration_threshold", 6.00),
            pca_dims=cm_cfg.get("pca_dims", 4),
        )
        self.sig4_coherence = TrajectoryCoherenceFractureDetector(
            dim=dim, config=config
        )
        self.sig5_grounding = ConfidenceGroundingDecouplingDetector()

        # Conjunction detector: rolling-window multi-signature co-occurrence
        conj_window = cfg.get("conjunction", {}).get("window", 8)
        self.conjunction = ConjunctionDetector(window=conj_window)

    def observe(
        self,
        action: Dict[str, Any],
        composite_hv: np.ndarray,
        fillers: Dict[str, np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run all 5 signature detectors on a single action.

        Args:
            action: classified action dict from openclaw_classifier
            composite_hv: bound 6-slot hypervector
            fillers: optional dict of per-slot fillers (for geometric signals)

        Returns:
            Dict with per-signature results and composite assessment.
        """
        context_alignment = float(action.get("context_alignment", 0.8))
        magnitude = float(action.get("magnitude", 0.1))

        # Run each signature detector
        sig1 = self.sig1_intent.observe(context_alignment, composite_hv)
        sig2 = self.sig2_provenance.observe(
            action,
            source_filler=fillers.get("source") if fillers else None,
            action_type_filler=fillers.get("action_type") if fillers else None,
            algebra=self.algebra,
        )
        sig3 = self.sig3_manifold.observe(composite_hv)
        sig4 = self.sig4_coherence.observe(composite_hv)
        sig5 = self.sig5_grounding.observe(composite_hv, magnitude, context_alignment)

        # sig4 excluded — advisory only pending learned encoder
        contributions = [
            sig1["verdict_contribution"],
            sig2["verdict_contribution"],
            sig3["verdict_contribution"],
            # sig4 excluded — advisory only (see above)
            sig5["verdict_contribution"],
        ]

        # Composite: highest escalation wins
        VERDICT_ORDER = {"PASS": 0, "MONITOR": 1, "FLAG": 2, "BLOCK": 3}
        max_verdict = max(contributions, key=lambda v: VERDICT_ORDER.get(v, 0))

        # Count how many signatures are firing
        firing_count = sum(1 for c in contributions if c != "PASS")

        # Weighted composite (sig4 weight=0 — advisory only)
        base_composite = (
            0.25 * sig1["signal"]
            + 0.30 * sig2["signal"]
            + 0.20 * sig3["signal"]
            + 0.00 * sig4["signal"]
            + 0.25 * sig5["signal"]
        )

        # Conjunction: multi-signature co-occurrence is superlinear evidence
        source = action.get("source", "unknown")
        conj_result = self.conjunction.observe(
            firing_sigs=set(
                name
                for name, sig in zip(
                    [
                        "sig1_intent",
                        "sig2_provenance",
                        "sig3_manifold",
                        "sig4_coherence",
                        "sig5_grounding",
                    ],
                    [sig1, sig2, sig3, sig4, sig5],
                )
                if sig["verdict_contribution"] != "PASS"
            ),
            source=source,
            current_composite=float(base_composite),
        )
        composite_signal = min(1.0, float(base_composite) * conj_result["multiplier"])

        # Escalate only when post-conjunction composite is meaningful
        if conj_result["label"] == "high_confidence_violation" and max_verdict in (
            "PASS",
            "MONITOR",
            "FLAG",
        ):
            max_verdict = "FLAG"
        elif (
            composite_signal > 0.30
            and conj_result["multiplier"] > 1.0
            and max_verdict == "PASS"
        ):
            max_verdict = "MONITOR"

        firing_sigs = [
            name
            for name, sig in zip(
                [
                    "sig1_intent",
                    "sig2_provenance",
                    "sig3_manifold",
                    "sig4_coherence",
                    "sig5_grounding",
                ],
                [sig1, sig2, sig3, sig4, sig5],
            )
            if sig["verdict_contribution"] != "PASS"
        ]

        return {
            "signatures": {
                "intent_binding_fracture": sig1,
                "source_provenance_corruption": sig2,
                "constitutional_manifold": sig3,
                "trajectory_coherence_fracture": sig4,
                "confidence_grounding_decoupling": sig5,
            },
            "composite_signal": round(float(composite_signal), 4),
            "composite_verdict": max_verdict,
            "firing_count": firing_count,
            "firing_signatures": firing_sigs,
            "conjunction": conj_result,
        }

    def reset(self) -> None:
        """Reset detector state for a new session. Preserves sig3 manifold baseline."""
        self.sig1_intent.cusum_neg = 0.0
        self.sig1_intent.cusum_pos = 0.0
        self.sig1_intent.alignments.clear()
        self.sig1_intent.calibration_buffer.clear()
        self.sig1_intent.baseline_mean = None
        self.sig1_intent.baseline_std = None
        self.sig1_intent.calibrated = False

        self.sig2_provenance.history.clear()
        self.sig2_provenance.ema_binding = None
        self.sig2_provenance.consecutive_untrusted_sensitive = 0

        # sig3 baseline preserved — calibration data, not runtime state

        if hasattr(self.sig4_coherence, "buffer"):
            self.sig4_coherence.buffer.clear()
        if hasattr(self.sig4_coherence, "coherence_scorer"):
            self.sig4_coherence.coherence_scorer.clear()
        if hasattr(self.sig4_coherence, "trajectory"):
            self.sig4_coherence.trajectory.clear()

        self.sig5_grounding.recent_hvs.clear()
        self.sig5_grounding.recent_magnitudes.clear()
        self.sig5_grounding.provenance_bundle = None

        self.conjunction.reset()
