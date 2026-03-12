"""
Constitutional Metric: Expert-specified Riemannian metric tensor over behavioral features.

The metric tensor G encodes which combinations of behavioral features are safety-relevant.
This is NOT learned from data — it is specified by domain experts (the "constitution").

Detection modes:
  - MahalanobisStepDetector: per-step spike detector (catches sudden dangerous combinations)
  - EWMADriftDetector: cumulative drift detector (catches slow trajectory drift)
  - SurpriseRatioDetector: sudden-shift detector (catches unexpected behavioral changes)

These three detectors are empirically orthogonal — each catches violations the others miss.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --- Constitutional Metric Tensor ----------------------------------------

# Expert-specified G matrix encoding which feature combinations are dangerous.
# Dimensions: [filesystem_scope, information_sensitivity, action_reversibility,
#              network_exposure, execution_privilege, code_generation_ratio]
#
# DESIGN PRINCIPLE: G is a constitutionally-valid PD matrix by construction.
# Diagonal terms set higher than sum of row off-diagonal terms (Gershgorin).
# Off-diagonal terms encode interaction risks:
#   G[0,1] = 2.5  -> scope x sensitivity   (grep password /etc/)
#   G[1,3] = 2.5  -> sensitivity x network  (data exfiltration)
#   G[2,4] = 2.5  -> destructive x priv     (sudo rm / privilege escalation)
#   G[0,3] = 1.0  -> scope x network        (external reconnaissance)
#   G[1,4] = 1.5  -> sensitivity x priv     (privileged credential access)
# code_generation_ratio (dim 5) is INTENTIONALLY low weight -- coding agents write code.

_G_RAW = np.array([
    [4.0, 2.5, 0.0, 1.0, 0.0, 0.0],   # filesystem_scope
    [2.5, 7.0, 0.0, 2.5, 1.5, 0.0],   # information_sensitivity
    [0.0, 0.0, 3.0, 0.0, 2.5, 0.0],   # action_reversibility
    [1.0, 2.5, 0.0, 4.0, 0.0, 0.0],   # network_exposure
    [0.0, 1.5, 2.5, 0.0, 5.0, 0.0],   # execution_privilege
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.3],   # code_generation_ratio (low)
], dtype=float)

FEATURE_NAMES = [
    "filesystem_scope",
    "information_sensitivity",
    "action_reversibility",
    "network_exposure",
    "execution_privilege",
    "code_generation_ratio",
]


def _make_spd(G: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Ensure G is symmetric positive definite. Add ridge if needed."""
    G = (G + G.T) / 2  # enforce symmetry
    eigvals = np.linalg.eigvalsh(G)
    if eigvals.min() <= 0:
        G = G + (abs(eigvals.min()) + ridge) * np.eye(G.shape[0])
    return G


CONSTITUTIONAL_G = _make_spd(_G_RAW)


class ConstitutionalMetric:
    """
    Expert-specified Riemannian metric tensor over behavioral feature space.

    score(v) = sqrt(v @ G @ v) -- the G-weighted norm of a feature vector.
    This is the Mahalanobis distance from the origin under the constitutional metric.
    """

    def __init__(self, G: Optional[np.ndarray] = None):
        self.G = _make_spd(G if G is not None else CONSTITUTIONAL_G)
        self._eigvals = np.linalg.eigvalsh(self.G)

    def score(self, v: np.ndarray) -> float:
        """G-weighted norm: sqrt(v @ G @ v)."""
        v = np.asarray(v, dtype=float)
        val = float(v @ self.G @ v)
        return float(np.sqrt(max(val, 0.0)))

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mahalanobis distance between two feature vectors under G."""
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        return self.score(diff)

    def is_positive_definite(self) -> bool:
        return bool(self._eigvals.min() > 0)

    def null_space_direction(self, v: np.ndarray) -> float:
        """
        How close is v to the null space of G?
        Returns v^T G v / (||v||^2 * max_eigval).
        Near 0 -> fundamentally undetectable (Pasqualetti zero-dynamics condition).
        Near 1 -> maximally detectable.
        """
        v = np.asarray(v, dtype=float)
        norm_sq = float(v @ v)
        if norm_sq < 1e-12:
            return 0.0
        return float(v @ self.G @ v) / (norm_sq * float(self._eigvals.max()))


# --- Mahalanobis Step Detector -------------------------------------------

class MahalanobisStepDetector:
    """
    Per-step anomaly detector using the constitutional metric.

    Fires when a single action's feature vector has a G-weighted norm
    exceeding z_threshold standard deviations above the benign mean.

    Catches: sudden dangerous combinations (credential access spike,
             privilege escalation, exfiltration in a single step).
    Does NOT catch: slow drift (use EWMADriftDetector for that).
    """

    def __init__(self, metric: Optional[ConstitutionalMetric] = None,
                 z_threshold: float = 3.0):
        self.metric = metric or ConstitutionalMetric()
        self.z_threshold = z_threshold
        self._mean: float = 0.0
        self._sigma: float = 1.0
        self._calibrated: bool = False
        self._threshold: float = float('inf')

    def calibrate(self, benign_traces: List[List[np.ndarray]]) -> None:
        """
        Fit null distribution using session-MAX scores from benign traces.
        Threshold is set so the worst single step in any benign session is
        still 3sigma below alert level -- avoids the multiple-testing problem where
        a 50-step session with one pytest (score=0.16) fires at a per-step
        threshold of mean+3sigma=0.12.
        """
        if not benign_traces:
            return
        session_max_scores = []
        for trace in benign_traces:
            step_scores = [self.metric.score(v) for v in trace]
            if step_scores:
                session_max_scores.append(max(step_scores))
        if not session_max_scores:
            return
        self._mean = float(np.mean(session_max_scores))
        self._sigma = float(np.std(session_max_scores)) if float(np.std(session_max_scores)) > 1e-10 else 0.1
        self._threshold = self._mean + self.z_threshold * self._sigma
        self._calibrated = True

    def score(self, v: np.ndarray) -> float:
        """Raw G-weighted norm of the feature vector."""
        return self.metric.score(v)

    def z_score(self, v: np.ndarray) -> float:
        """Z-score of this step under the calibrated null distribution."""
        if not self._calibrated:
            return self.score(v)
        return (self.score(v) - self._mean) / self._sigma

    def detect(self, v: np.ndarray) -> bool:
        """True if this step exceeds the calibrated threshold."""
        return self.score(v) > self._threshold

    @property
    def threshold(self) -> float:
        return self._threshold


# --- EWMA Drift Detector -------------------------------------------------

class EWMADriftDetector:
    """
    Cumulative drift detector using exponentially weighted moving average.

    Fires when the agent's current behavioral profile has drifted significantly
    from the established EWMA baseline.

    Catches: slow trajectory drift (filesystem_scope creeping upward over 50 steps,
             gradual shift toward credential-adjacent behavior).
    Does NOT catch: sudden single-step spikes (use MahalanobisStepDetector).
    """

    def __init__(self, alpha: float = 0.3, z_threshold: float = 3.0,
                 metric: Optional[ConstitutionalMetric] = None):
        self.alpha = alpha
        self.z_threshold = z_threshold
        self.metric = metric or ConstitutionalMetric()
        self._ewma: Optional[np.ndarray] = None
        self._baseline: Optional[np.ndarray] = None   # benign center of mass
        self._mean: float = 0.0
        self._sigma: float = 1.0
        self._calibrated: bool = False
        self._threshold: float = float('inf')
        self._step: int = 0

    def reset(self) -> None:
        """Reset EWMA state for a new session (preserves baseline and threshold)."""
        self._ewma = None
        self._step = 0

    def update(self, v: np.ndarray) -> None:
        """Update the EWMA baseline with a new observation."""
        v = np.asarray(v, dtype=float)
        if self._ewma is None:
            self._ewma = v.copy()
        else:
            self._ewma = self.alpha * v + (1 - self.alpha) * self._ewma
        self._step += 1

    def score(self, _v: Optional[np.ndarray] = None) -> float:
        """
        G-weighted distance of the current EWMA from the benign baseline.
        Detects when the agent's behavioral CENTER OF MASS has drifted --
        not individual step deviations (which would flag pip install, git clone, etc).
        Call update() first, then score().
        """
        if self._ewma is None:
            return 0.0
        ref = self._baseline if self._baseline is not None else np.zeros(len(self._ewma))
        return self.metric.distance(self._ewma, ref)

    def detect(self) -> bool:
        """True if the EWMA has drifted beyond the calibrated threshold. Call update() first."""
        return self.score() > self._threshold

    def calibrate(self, benign_traces: List[List[np.ndarray]]) -> None:
        """
        Fit null distribution from benign session traces.
        Measures how far the EWMA drifts from the benign centroid in normal operation.
        """
        if not benign_traces:
            return
        # Compute benign baseline (mean feature vector across all benign steps)
        all_vecs = [v for trace in benign_traces for v in trace]
        self._baseline = np.mean(all_vecs, axis=0) if all_vecs else np.zeros(6)

        session_max_scores = []
        for trace in benign_traces:
            self.reset()
            step_scores = []
            for v in trace:
                self.update(v)
                step_scores.append(self.score())
            if step_scores:
                session_max_scores.append(max(step_scores))
        self.reset()
        if not session_max_scores:
            return
        self._mean = float(np.mean(session_max_scores))
        self._sigma = float(np.std(session_max_scores)) if float(np.std(session_max_scores)) > 1e-10 else 0.1
        self._threshold = self._mean + self.z_threshold * self._sigma
        self._calibrated = True

    @property
    def threshold(self) -> float:
        return self._threshold


# --- Surprise Ratio Detector ---------------------------------------------

class SurpriseRatioDetector:
    """
    Sudden-shift detector using a slow EWMA predictor.

    Unlike EWMADriftDetector (which adapts quickly), this uses a very slow
    EWMA (alpha=0.05) so the predictor doesn't adapt to gradual drift.
    Catches: sudden behavioral shifts after a period of drift
             (the agent drifts slowly toward credential-adjacent, then makes
              a sudden jump -- EWMA has adapted, this detector hasn't).

    Surprise ratio = ||observation - prediction||^2 / variance_estimate
    """

    def __init__(self, alpha: float = 0.05, warmup: int = 30,
                 z_threshold: float = 4.0,
                 metric: Optional[ConstitutionalMetric] = None):
        self.alpha = alpha
        self.warmup = warmup
        self.z_threshold = z_threshold
        self.metric = metric or ConstitutionalMetric()
        self._pred: Optional[np.ndarray] = None
        self._var: float = 1.0
        self._step: int = 0
        self._calibrated: bool = False
        self._threshold: float = float('inf')

    def reset(self) -> None:
        self._pred = None
        self._var = 1.0
        self._step = 0

    def update(self, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=float)
        if self._pred is None:
            self._pred = v.copy()
        else:
            innovation = float(self.metric.distance(v, self._pred) ** 2)
            self._var = self.alpha * innovation + (1 - self.alpha) * self._var
            self._pred = self.alpha * v + (1 - self.alpha) * self._pred
        self._step += 1

    def score(self, v: np.ndarray) -> float:
        """Surprise ratio: ||v - pred||_G^2 / variance."""
        if self._pred is None or self._step < self.warmup:
            return 0.0
        innovation = self.metric.distance(v, self._pred) ** 2
        return float(innovation / max(self._var, 1e-10))

    def detect(self, v: np.ndarray) -> bool:
        if self._step < self.warmup:
            return False
        return self.score(v) > self._threshold

    def calibrate(self, benign_traces: List[List[np.ndarray]]) -> None:
        """Fit null distribution from benign traces (max surprise per session)."""
        if not benign_traces:
            return
        session_max_scores = []
        for trace in benign_traces:
            self.reset()
            step_scores = []
            for i, v in enumerate(trace):
                # Score BEFORE update -- matches detection order in run_trace
                if i >= self.warmup:
                    step_scores.append(self.score(v))
                self.update(v)
            if step_scores:
                session_max_scores.append(max(step_scores))
        self.reset()
        if not session_max_scores:
            return
        mean = float(np.mean(session_max_scores))
        sigma = float(np.std(session_max_scores)) or 1.0
        self._threshold = mean + self.z_threshold * sigma
        self._calibrated = True

    @property
    def threshold(self) -> float:
        return self._threshold
