"""
ICA-based Compound Violation Mode Detector
==========================================

.. deprecated:: 0.4.0
    DEAD CODE — quarantined, not deleted. This ICA compound-mode detector
    was never wired into any pipeline, is imported nowhere in the package,
    is not exported from ``frontier_ops.sensing``, and has zero benchmark
    contact. Its premise (decomposing a "47.9% detection gap" into compound
    modes) predates the v3 finding that detection reduces to an in-domain
    prototype direction (``frontier_ops.CalibratedDetector``;
    eval/results/repo-open-questions-2026-07-10.md). Kept importable for
    provenance only; it will emit a DeprecationWarning on import.

Learns independent violation patterns from labeled traces.
Projects new trajectories onto learned modes for detection.
ICA in Mahalanobis-whitened space respects the constitutional geometry.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

warnings.warn(
    "frontier_ops.sensing.modes (ViolationModeDetector) is dead code, "
    "quarantined in v3: never wired in, no benchmark contact. Use "
    "frontier_ops.CalibratedDetector for detection.",
    DeprecationWarning,
    stacklevel=2,
)


class ViolationModeDetector:
    """
    ICA-based compound violation mode detector.

    Learns independent violation patterns from labeled traces.
    Projects new trajectories onto violation modes.
    Flags when any mode activation exceeds the normal-behavior threshold.

    Operates in Mahalanobis-whitened space (G^{1/2} @ c) so that
    independent components respect the constitutional geometry.
    """

    def __init__(
        self,
        n_modes: int = 4,
        threshold_percentile: float = 95.0,
        metric_tensor: Optional[np.ndarray] = None,
    ):
        self.n_modes = n_modes
        self.threshold_percentile = threshold_percentile
        self.G = metric_tensor
        self._G_sqrt: Optional[np.ndarray] = None
        self._G_sqrt_inv: Optional[np.ndarray] = None
        self._ica = None
        self._thresholds: Optional[np.ndarray] = None
        self._mixing: Optional[np.ndarray] = None
        self._mode_labels: List[str] = []
        self._fitted = False

    def _compute_metric_sqrt(self):
        if self.G is None:
            return
        eigvals, eigvecs = np.linalg.eigh(self.G)
        eigvals = np.clip(eigvals, 1e-10, None)
        self._G_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        self._G_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        if self._G_sqrt is not None:
            return X @ self._G_sqrt.T
        return X

    def fit(
        self,
        violation_trajectories: List[np.ndarray],
        normal_trajectories: List[np.ndarray],
        dim_names: Optional[List[str]] = None,
    ):
        """
        Learn violation modes from labeled trajectories.

        Args:
            violation_trajectories: List of (n_steps, n_dims) arrays from violation traces.
            normal_trajectories: List of (n_steps, n_dims) arrays from normal traces.
            dim_names: Names for concept dimensions (for interpretable mode labels).
        """
        from sklearn.decomposition import FastICA

        self._compute_metric_sqrt()

        X_viol = self._whiten(np.vstack(violation_trajectories))
        X_norm = self._whiten(np.vstack(normal_trajectories))

        n_components = min(self.n_modes, X_viol.shape[1], X_viol.shape[0] - 1)

        self._ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=2000,
            tol=1e-4,
        )
        self._ica.fit(X_viol)
        self._mixing = self._ica.mixing_

        # Compute thresholds from normal data
        normal_proj = np.abs(self._ica.transform(X_norm))
        self._thresholds = np.percentile(
            normal_proj, self.threshold_percentile, axis=0,
        )

        # Label modes interpretably
        if dim_names and self._mixing is not None:
            mixing_orig = self._mixing
            if self._G_sqrt_inv is not None:
                mixing_orig = self._G_sqrt_inv @ self._mixing

            self._mode_labels = []
            for mode_idx in range(mixing_orig.shape[1]):
                weights = mixing_orig[:, mode_idx]
                sorted_dims = np.argsort(np.abs(weights))[::-1]
                parts = []
                for d in sorted_dims[:3]:
                    if abs(weights[d]) > 0.1:
                        sign = "+" if weights[d] > 0 else "-"
                        name = dim_names[d] if d < len(dim_names) else f"dim{d}"
                        parts.append(f"{sign}{name}")
                self._mode_labels.append(
                    " ".join(parts) if parts else f"mode_{mode_idx}"
                )

        self._fitted = True

    def detect(self, trajectory: np.ndarray) -> Dict:
        """
        Detect violation modes in a trajectory.

        Args:
            trajectory: (n_steps, n_dims) concept trajectory.

        Returns:
            Dict with flagged modes, activations, thresholds.
        """
        assert self._fitted, "Call fit() first"

        X = self._whiten(trajectory)
        projections = np.abs(self._ica.transform(X))

        max_activations = projections.max(axis=0)
        mean_activations = projections.mean(axis=0)

        flagged_modes = []
        for i in range(len(self._thresholds)):
            if (max_activations[i] > self._thresholds[i] or
                    mean_activations[i] > 0.7 * self._thresholds[i]):
                flagged_modes.append(i)

        return {
            "flagged": len(flagged_modes) > 0,
            "max_activations": max_activations,
            "mean_activations": mean_activations,
            "thresholds": self._thresholds.copy(),
            "flagged_modes": flagged_modes,
            "mode_labels": [self._mode_labels[i] for i in flagged_modes]
                if self._mode_labels else [],
        }

    def detect_step(self, concept_vec: np.ndarray) -> Dict:
        """Single-step online detection."""
        assert self._fitted, "Call fit() first"
        X = self._whiten(concept_vec.reshape(1, -1))
        activations = np.abs(self._ica.transform(X)).flatten()

        flagged = [
            i for i in range(len(self._thresholds))
            if activations[i] > self._thresholds[i]
        ]

        return {
            "flagged": len(flagged) > 0,
            "activations": activations,
            "flagged_modes": flagged,
            "mode_labels": [self._mode_labels[i] for i in flagged]
                if self._mode_labels else [],
        }

    @property
    def mode_descriptions(self) -> List[str]:
        return self._mode_labels.copy()
