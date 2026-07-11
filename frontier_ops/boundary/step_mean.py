"""
StepMeanScorer — chart-space step-mean scoring (v2 champion).

The best detector of the Apollo campaign (insider trading episodes:
AUROC 0.790 vs 0.611 for whole-text single-point encoding): split a response
into sentence steps, encode each step in the concept chart, average in chart
space, and classify with nearest-centroid under a calibrated (estimated)
metric. This is the streaming `process_step` decomposition wearing its
correct aggregator.

Standalone — no pipeline required, and deliberately stateless across calls:
every text is scored independently, so episodes can never leak trajectory
state into each other.

MiniLM truncates at ~256 wordpieces, which is exactly why per-step encoding
beats whole-text encoding on long responses: the single-point encode silently
drops everything past the truncation horizon.

Usage::

    scorer = StepMeanScorer(dims=DIMS, anchors=ANCHORS)
    X = np.stack([scorer.embed(t) for t in train_texts])
    scorer.fit(X, train_labels)
    s = scorer.score(new_text)   # higher = closer to class-1 centroid
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

STEP_SPLIT_RE = r"(?<=[.!?])\s+|\n+"
MIN_STEP_CHARS = 15
MIN_STEPS = 3


class StepSplitError(ValueError):
    """Raised when a text yields fewer than MIN_STEPS usable steps."""


class StepMeanScorer:
    """Sentence-step chart-mean encoder + nearest-centroid scorer.

    Args:
        extractor: a SemanticConceptExtractor to reuse (recommended when
            scoring many texts — anchor embeddings precompute at extractor
            init). Built lazily from the remaining args when omitted.
        dims: concept dimensions (default: stock CONCEPTS).
        anchors: anchor phrases per dim (default: stock SEMANTIC_ANCHORS).
        model_name: sentence-transformers model for a lazily built extractor.
        encoder: any frontier_ops.encoder.Encoder for the lazily built
            extractor (skips sentence-transformers entirely). Ignored when
            ``extractor`` is given.
        ridge: ridge added to the pooled within-class covariance in fit().
    """

    def __init__(
        self,
        extractor=None,
        *,
        dims: Optional[List[str]] = None,
        anchors: Optional[Dict[str, List[str]]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        encoder=None,
        ridge: float = 1e-3,
    ):
        self._extractor = extractor
        self._dims = dims
        self._anchors = anchors
        self._model_name = model_name
        self._encoder = encoder
        self.ridge = ridge
        self.metric: Optional[np.ndarray] = None
        self.centroid_0: Optional[np.ndarray] = None
        self.centroid_1: Optional[np.ndarray] = None

    # -- encoding ----------------------------------------------------------

    @property
    def extractor(self):
        if self._extractor is None:
            from frontier_ops.boundary.semantic_extraction import (
                SemanticConceptExtractor,
            )

            self._extractor = SemanticConceptExtractor(
                model_name=self._model_name,
                dims=self._dims,
                anchors=self._anchors,
                encoder=self._encoder,
            )
        return self._extractor

    @property
    def dims(self) -> List[str]:
        return list(self.extractor.dims)

    @staticmethod
    def split_steps(text: str) -> List[str]:
        """Sentence/line steps: split on sentence ends or newlines, keep
        fragments longer than MIN_STEP_CHARS."""
        return [
            s.strip()
            for s in re.split(STEP_SPLIT_RE, text)
            if len(s.strip()) > MIN_STEP_CHARS
        ]

    def encode_steps(self, text: str) -> np.ndarray:
        """Per-step chart vectors, shape (n_steps, n_dims).

        Raises StepSplitError when the text yields fewer than MIN_STEPS
        steps — scoring a near-empty text as a "trajectory" would silently
        degenerate to single-point encoding.
        """
        steps = self.split_steps(text)
        if len(steps) < MIN_STEPS:
            raise StepSplitError(
                f"need >= {MIN_STEPS} steps of > {MIN_STEP_CHARS} chars, got "
                f"{len(steps)} — text too short for step-mean scoring"
            )
        enc = self.extractor
        dims = enc.dims
        return np.stack(
            [np.array([enc.extract(s)[d] for d in dims]) for s in steps]
        )

    def embed(self, text: str) -> np.ndarray:
        """Chart-space mean of the per-step vectors, shape (n_dims,)."""
        return self.encode_steps(text).mean(axis=0)

    # -- calibration + scoring ---------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric_X: Optional[np.ndarray] = None,
        metric_y: Optional[np.ndarray] = None,
    ) -> "StepMeanScorer":
        """Calibrate centroids and the estimated metric.

        Args:
            X: (n, d) step-mean vectors (from embed()).
            y: binary labels; class 1 is what score() ranks upward.
            metric_X / metric_y: optional separate vector set for the pooled
                within-class covariance (the campaign's F4 estimated it from
                whole-text vectors while centroids came from step means).
        """
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        mX = X if metric_X is None else np.asarray(metric_X, float)
        my = y if metric_y is None else np.asarray(metric_y, int)
        d = X.shape[1]
        Sw = np.zeros((d, d))
        for c in (0, 1):
            Z = mX[my == c] - mX[my == c].mean(0)
            Sw += Z.T @ Z
        M = np.linalg.inv(Sw / max(len(mX) - 2, 1) + self.ridge * np.eye(d))
        self.metric = M * d / np.trace(M)
        self.centroid_0 = X[y == 0].mean(0)
        self.centroid_1 = X[y == 1].mean(0)
        return self

    def score_vector(self, x: np.ndarray) -> float:
        """Nearest-centroid margin under the calibrated metric: squared
        distance to centroid 0 minus squared distance to centroid 1, so
        higher = more class-1."""
        if self.metric is None:
            raise RuntimeError("fit() must be called before scoring")
        x = np.asarray(x, float)
        d0 = x - self.centroid_0
        d1 = x - self.centroid_1
        return float(d0 @ self.metric @ d0 - d1 @ self.metric @ d1)

    def score(self, text: str) -> float:
        """embed() + score_vector() for a single text."""
        return self.score_vector(self.embed(text))
