"""
Efference Copy Predictor for Constitutional Manifold Agent
===========================================================

Biological basis (from Claude.ai research, 2026-02-28):
    - Forward model predicts sensory consequences of motor commands
    - Prediction error = actual - predicted
    - Error is precision-weighted (Friston's active inference)
    - CD component: expected magnitude (EWMA baseline)
    - EC component: expected direction (Kalman filter)

Key research insights:
    - Wolpert & Ghahramani (2000): paired forward-inverse models
    - Tanaka et al. (2020): cerebellum implements forward models
    - Honda et al. (2018): tandem forward-inverse architecture
    - Li et al. (2020): motor signals bifurcate into CD (general suppression)
      and EC (content-specific enhancement)
    - Friston et al. (2022): precision = inverse variance, prediction errors
      weighted by precision. Constitutional metric tensor IS precision matrix.
    - Ciria et al. (2021): precision weighting underexplored in robotics
    - Ji et al. (2024): LLM hidden states predict hallucination risk (84.32%)
    - Xu et al. (2024) SaySelf: fine-grained confidence with self-reflective rationales
    - Xiao et al. (2025) EAGLE: internal beliefs from intermediate layers

The prediction error replaces text-based proprioceptive narratives.
Instead of "your angular displacement is 0.4 radians" → the agent receives a
structured geometric signal: the precision-weighted prediction error
vector in concept space.

Integration:
    predictor = EfferenceCopyPredictor(n_dims=6, metric_tensor=G)

    # In agent step, BEFORE the LLM call:
    predicted = predictor.predict_next()
    # ... agent acts, concept_vec extracted ...
    error = predictor.compute_error(predicted, actual_concept_vec)
    predictor.update(actual_concept_vec)

    # error.weighted_magnitude is the proprioceptive signal
    # error.direction_vec tells you WHERE the deviation is
    # error.surprise_ratio tells you HOW surprising vs historical
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.linalg import inv as scipy_inv

logger = logging.getLogger(__name__)


# ─── Prediction Error Result ──────────────────────────────────────────────

@dataclass
class PredictionError:
    """Structured prediction error with geometric metadata."""
    raw_error: np.ndarray
    weighted_magnitude: float   # e^T G e = variational free energy
    raw_magnitude: float
    max_error_dim: int
    max_error_dim_name: str
    max_error_value: float
    surprise_ratio: float       # current error / historical mean
    magnitude_error: float      # CD: |actual_step| - |predicted_step|
    direction_error: float      # EC: angular deviation of step direction
    prediction_source: str      # "kalman", "ewma", "linear"
    kalman_confidence: float    # trace of Kalman covariance

    @property
    def is_surprising(self) -> bool:
        return self.surprise_ratio > 2.0

    @property
    def direction_vec(self) -> np.ndarray:
        norm = np.linalg.norm(self.raw_error)
        if norm < 1e-12:
            return np.zeros_like(self.raw_error)
        return self.raw_error / norm

    def to_proprioceptive_context(self, dim_names: List[str]) -> str:
        """Generate structured proprioceptive context for prompt injection."""
        if self.weighted_magnitude < 0.01:
            return "[Proprioception: trajectory stable, within predicted bounds]"

        abs_errors = np.abs(self.raw_error)
        sorted_dims = np.argsort(abs_errors)[::-1]

        parts = []
        if self.surprise_ratio > 3.0:
            parts.append(f"[PROPRIOCEPTIVE ALERT: unexpected trajectory shift "
                        f"({self.surprise_ratio:.1f}x normal variance)]")
        elif self.surprise_ratio > 1.5:
            parts.append(f"[Proprioception: moderate trajectory deviation "
                        f"({self.surprise_ratio:.1f}x normal)]")
        else:
            parts.append("[Proprioception: minor trajectory adjustment]")

        for i in range(min(2, len(sorted_dims))):
            dim_idx = sorted_dims[i]
            dim_name = dim_names[dim_idx] if dim_idx < len(dim_names) else f"dim_{dim_idx}"
            err_val = self.raw_error[dim_idx]
            direction = "increasing" if err_val > 0 else "decreasing"
            parts.append(f"  {dim_name}: {direction} ({abs(err_val):.3f} beyond predicted)")

        if self.direction_error > 0.3:
            parts.append(f"  Step direction deviated {self.direction_error:.2f} rad from predicted")

        return "\n".join(parts)


# ─── Core Predictor ───────────────────────────────────────────────────────

class EfferenceCopyPredictor:
    """
    Multi-tier concept trajectory predictor.

    Tier 1: Linear extrapolation (last 2-3 steps) — fast, detects sudden changes
    Tier 2: EWMA baseline — stable reference trajectory
    Tier 3: Kalman filter — adaptive, optimal under Gaussian assumptions

    Precision weighting uses the constitutional metric tensor G.
    The weighted prediction error e^T G e is the variational free energy.
    """

    def __init__(
        self,
        n_dims: int = 6,
        dim_names: Optional[List[str]] = None,
        metric_tensor: Optional[np.ndarray] = None,
        ewma_alpha: float = 0.3,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 0.05,
        history_size: int = 50,
        min_steps_for_kalman: int = 5,
        min_steps_for_linear: int = 3,
    ):
        self.n_dims = n_dims
        self.dim_names = dim_names or [f"dim_{i}" for i in range(n_dims)]

        if metric_tensor is not None:
            self.G = np.array(metric_tensor, dtype=np.float64)
        else:
            self.G = np.eye(n_dims, dtype=np.float64)

        self.ewma_alpha = ewma_alpha
        self._ewma_mean: Optional[np.ndarray] = None
        self._ewma_var: Optional[np.ndarray] = None

        self._kalman_state = np.zeros(2 * n_dims)
        self._kalman_P = np.eye(2 * n_dims) * 1.0
        self._kalman_Q = np.eye(2 * n_dims) * kalman_process_noise
        self._kalman_R = np.eye(n_dims) * kalman_measurement_noise

        self._kalman_F = np.eye(2 * n_dims)
        self._kalman_F[:n_dims, n_dims:] = np.eye(n_dims)

        self._kalman_H = np.zeros((n_dims, 2 * n_dims))
        self._kalman_H[:n_dims, :n_dims] = np.eye(n_dims)

        self.history: List[np.ndarray] = []
        self.error_history: List[float] = []
        self.history_size = history_size
        self.min_steps_for_kalman = min_steps_for_kalman
        self.min_steps_for_linear = min_steps_for_linear

        self._step = 0
        self._kalman_initialized = False

    def predict_next(self) -> np.ndarray:
        """Predict next concept vector using best available method."""
        n = len(self.history)

        if n == 0:
            return np.ones(self.n_dims) / self.n_dims

        if n < self.min_steps_for_linear:
            return self._ewma_mean.copy() if self._ewma_mean is not None \
                else self.history[-1].copy()

        pred_ewma = self._predict_ewma()
        pred_linear = self._predict_linear()

        if n >= self.min_steps_for_kalman and self._kalman_initialized:
            pred_kalman = self._predict_kalman()
            stability = self._estimate_stability()
            w_kalman = stability
            w_linear = (1 - stability) * 0.6
            w_ewma = (1 - stability) * 0.4
        else:
            pred_kalman = pred_ewma
            w_kalman = 0.0
            w_linear = 0.5
            w_ewma = 0.5

        blended = w_kalman * pred_kalman + w_linear * pred_linear + w_ewma * pred_ewma
        blended = np.clip(blended, 1e-8, None)
        blended /= blended.sum()
        return blended

    def compute_error(self, predicted: np.ndarray, actual: np.ndarray) -> PredictionError:
        """Compute structured prediction error."""
        raw_error = actual - predicted
        raw_magnitude = float(np.linalg.norm(raw_error))
        weighted_magnitude = float(raw_error @ self.G @ raw_error)

        magnitude_error, direction_error = self._decompose_error(predicted, actual)

        if self.error_history:
            historical_mean = np.mean(self.error_history[-20:])
            surprise_ratio = weighted_magnitude / (historical_mean + 1e-8)
        else:
            surprise_ratio = 1.0

        abs_errors = np.abs(raw_error)
        max_dim = int(np.argmax(abs_errors))

        if self._kalman_initialized:
            kalman_conf = float(np.trace(self._kalman_P[:self.n_dims, :self.n_dims]))
        else:
            kalman_conf = float('inf')

        n = len(self.history)
        if n >= self.min_steps_for_kalman and self._kalman_initialized:
            source = "kalman"
        elif n >= self.min_steps_for_linear:
            source = "linear"
        else:
            source = "ewma"

        return PredictionError(
            raw_error=raw_error,
            weighted_magnitude=weighted_magnitude,
            raw_magnitude=raw_magnitude,
            max_error_dim=max_dim,
            max_error_dim_name=self.dim_names[max_dim],
            max_error_value=float(raw_error[max_dim]),
            surprise_ratio=surprise_ratio,
            magnitude_error=magnitude_error,
            direction_error=direction_error,
            prediction_source=source,
            kalman_confidence=kalman_conf,
        )

    def update(self, actual: np.ndarray):
        """Update all internal models with observed actual vector."""
        self.history.append(actual.copy())
        if len(self.history) > self.history_size:
            self.history.pop(0)

        self._update_ewma(actual)

        if len(self.history) >= self.min_steps_for_kalman:
            self._update_kalman(actual)

        if len(self.history) >= 2:
            step_error = float(np.linalg.norm(actual - self.history[-2]))
            self.error_history.append(step_error)
            if len(self.error_history) > self.history_size:
                self.error_history.pop(0)

        self._step += 1

    def set_metric_tensor(self, G: np.ndarray):
        """Update the constitutional metric tensor (precision matrix)."""
        self.G = np.array(G, dtype=np.float64)

    # ─── Tier 1: Linear Extrapolation ─────────────────────────────────

    def _predict_linear(self) -> np.ndarray:
        n = min(len(self.history), 3)
        if n < 2:
            return self.history[-1].copy()

        recent = np.array(self.history[-n:])
        t = np.arange(n, dtype=np.float64)
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        t_mean = np.average(t, weights=weights)
        pos_mean = np.average(recent, axis=0, weights=weights)

        numerator = np.zeros(self.n_dims)
        denominator = 0.0
        for i in range(n):
            numerator += weights[i] * (t[i] - t_mean) * (recent[i] - pos_mean)
            denominator += weights[i] * (t[i] - t_mean) ** 2

        if abs(denominator) < 1e-12:
            return self.history[-1].copy()

        velocity = numerator / denominator
        predicted = pos_mean + velocity * (t[-1] + 1 - t_mean)
        predicted = np.clip(predicted, 1e-8, None)
        predicted /= predicted.sum()
        return predicted

    # ─── Tier 2: EWMA Baseline ────────────────────────────────────────

    def _predict_ewma(self) -> np.ndarray:
        if self._ewma_mean is None:
            return self.history[-1].copy() if self.history else np.ones(self.n_dims) / self.n_dims
        return self._ewma_mean.copy()

    def _update_ewma(self, actual: np.ndarray):
        if self._ewma_mean is None:
            self._ewma_mean = actual.copy()
            self._ewma_var = np.zeros(self.n_dims)
            return
        alpha = self.ewma_alpha
        diff = actual - self._ewma_mean  # Compute diff BEFORE updating mean
        self._ewma_mean = alpha * actual + (1 - alpha) * self._ewma_mean
        self._ewma_var = alpha * (diff ** 2) + (1 - alpha) * self._ewma_var

    # ─── Tier 3: Kalman Filter ────────────────────────────────────────

    def _predict_kalman(self) -> np.ndarray:
        x_pred = self._kalman_F @ self._kalman_state
        predicted_pos = x_pred[:self.n_dims]
        predicted_pos = np.clip(predicted_pos, 1e-8, None)
        predicted_pos /= predicted_pos.sum()
        return predicted_pos

    def _update_kalman(self, actual: np.ndarray):
        if not self._kalman_initialized:
            self._kalman_state[:self.n_dims] = actual
            if len(self.history) >= 2:
                self._kalman_state[self.n_dims:] = self.history[-1] - self.history[-2]
            self._kalman_initialized = True
            return

        F, H, Q, R = self._kalman_F, self._kalman_H, self._kalman_Q, self._kalman_R

        x_pred = F @ self._kalman_state
        P_pred = F @ self._kalman_P @ F.T + Q

        y = actual - H @ x_pred
        S = H @ P_pred @ H.T + R

        try:
            K = P_pred @ H.T @ scipy_inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Kalman update: singular innovation covariance")
            return

        self._kalman_state = x_pred + K @ y
        eye = np.eye(2 * self.n_dims)
        self._kalman_P = (eye - K @ H) @ P_pred

    # ─── Error Decomposition ──────────────────────────────────────────

    def _decompose_error(self, predicted: np.ndarray, actual: np.ndarray) -> tuple:
        """Decompose into CD (magnitude) and EC (direction) components."""
        if len(self.history) < 1:
            return 0.0, 0.0

        prev = self.history[-1]
        predicted_step = predicted - prev
        actual_step = actual - prev

        pred_mag = np.linalg.norm(predicted_step)
        actual_mag = np.linalg.norm(actual_step)

        magnitude_error = abs(actual_mag - pred_mag)

        if pred_mag < 1e-10 or actual_mag < 1e-10:
            direction_error = 0.0
        else:
            cos_sim = np.clip(
                np.dot(predicted_step, actual_step) / (pred_mag * actual_mag),
                -1.0, 1.0,
            )
            direction_error = float(np.arccos(cos_sim))

        return magnitude_error, direction_error

    # ─── Stability Estimation ─────────────────────────────────────────

    def _estimate_stability(self) -> float:
        if len(self.history) < 3:
            return 0.5

        n = min(len(self.history), 10)
        recent = self.history[-n:]
        steps = [np.linalg.norm(recent[i] - recent[i-1]) for i in range(1, len(recent))]

        if not steps:
            return 0.5

        mean_step = np.mean(steps)
        std_step = np.std(steps)

        if mean_step < 1e-10:
            return 1.0

        cv = std_step / mean_step
        stability = float(np.exp(-cv))
        return np.clip(stability, 0.0, 1.0)

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "history_len": len(self.history),
            "kalman_initialized": self._kalman_initialized,
            "kalman_confidence": float(np.trace(
                self._kalman_P[:self.n_dims, :self.n_dims]
            )) if self._kalman_initialized else None,
            "stability": self._estimate_stability(),
            "ewma_mean": self._ewma_mean.tolist() if self._ewma_mean is not None else None,
            "recent_error_mean": float(np.mean(self.error_history[-10:]))
                if self.error_history else None,
        }
