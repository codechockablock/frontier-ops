"""
CalibratedDetector — the v3 lean detector.

The whole empirical program (Apollo battery, the drift-harness domain port,
the combiner diagnostic) converges on one detector: embed each action's text
with a raw sentence encoder, score it against a **prototype direction**
calibrated in-domain from a few dozen labeled examples, and threshold at a
benign quantile. On the project's own agent-session traffic this reaches
AUROC 0.93 — beating the shipping 12-signal pipeline's alert_level (0.59) and
the best achievable subset of its channels (0.90); see
eval/results/repo-open-questions-2026-07-10.md.

Why a prototype and not the full pipeline:
  - The evidence combiner *inverted* signal (an anti-informative "surprise"
    channel plus an escalation overwrite dragged a 0.85 signal to 0.59).
  - No subset of the eight combiner channels beat this prototype.
  - The concept chart, expert-asserted metrics, curvature walls, and
    path-energy features were all deprecated in v2 as non-detecting.
What survives as complements, not core: NEWMA (the one drift channel that is
benign-stable over long live sessions — keep it for streaming), and the
authorization/governance layer (semantics + audit, not detection).

Calibration cost (measured): ~32-64 labeled in-domain episodes reach within
0.02 AUROC of full-data; eight already match an unsupervised baseline.
Thresholds do NOT transport across workloads — recalibrate the benign
quantile per deployment (label-free). See the domain-port and
calibration-transport results.

Standalone: no pipeline, no constitution required. Requires the [semantic]
extra (sentence-transformers).

Usage::

    det = CalibratedDetector()
    det.calibrate(train_texts, train_labels)      # labels: 1 = flag-worthy
    det.set_threshold(benign_texts, alpha=0.1)    # or pass fpr during calibrate
    score = det.score(action_text)                # higher = more off-goal
    flag = det.flag(action_text)                  # score > threshold
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class CalibratedDetector:
    """In-domain-calibrated prototype-direction detector.

    Args:
        model: a preloaded sentence-transformers model to reuse (recommended
            when scoring many texts). Lazily loaded from ``model_name`` when
            omitted.
        model_name: encoder to load if ``model`` is not given.
    """

    def __init__(self, model=None, model_name: str = DEFAULT_MODEL):
        self._model = model
        self._model_name = model_name
        self.direction: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        # calibration provenance, for auditability
        self.n_calibration: int = 0
        self.alpha: Optional[float] = None

    # -- embedding ---------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - exercised via extra
                raise ImportError(
                    "CalibratedDetector requires sentence-transformers "
                    "(pip install frontier-ops[semantic])."
                ) from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalized encoder embeddings, shape (n, d)."""
        return self.model.encode(
            list(texts),
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # -- calibration -------------------------------------------------------

    def calibrate(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        fpr: Optional[float] = 0.1,
    ) -> "CalibratedDetector":
        """Fit the prototype direction from labeled in-domain examples.

        direction = unit(mean(embed[label==1]) - mean(embed[label==0])), the
        nearest-centroid margin under the identity metric — the statistic a
        few dozen labels can estimate. When ``fpr`` is given, the flag
        threshold is set to that false-positive rate on the label==0
        (benign) examples; pass ``fpr=None`` and call ``set_threshold`` on a
        dedicated benign set instead.
        """
        labels = np.asarray(labels, int)
        if set(np.unique(labels).tolist()) - {0, 1}:
            raise ValueError("labels must be 0/1 (1 = flag-worthy)")
        if labels.min() == labels.max():
            raise ValueError("calibration needs both classes present")
        E = self.embed(texts)
        pos = E[labels == 1].mean(0)
        neg = E[labels == 0].mean(0)
        d = pos - neg
        self.direction = d / (np.linalg.norm(d) + 1e-12)
        self.n_calibration = int(len(labels))
        if fpr is not None:
            self._set_threshold_from_scores(E[labels == 0] @ self.direction, fpr)
        return self

    def set_threshold(self, benign_texts: Sequence[str], alpha: float = 0.1) -> float:
        """Set the flag threshold to the ``1-alpha`` quantile of benign
        scores (a label-free per-deployment recalibration — thresholds do
        not transport across workloads)."""
        if self.direction is None:
            raise RuntimeError("calibrate() before set_threshold()")
        scores = self.embed(benign_texts) @ self.direction
        return self._set_threshold_from_scores(scores, alpha)

    def _set_threshold_from_scores(self, benign_scores: np.ndarray, alpha: float) -> float:
        self.threshold = float(np.quantile(benign_scores, 1.0 - alpha))
        self.alpha = float(alpha)
        return self.threshold

    # -- scoring -----------------------------------------------------------

    def score(self, text: str) -> float:
        """Signed distance along the prototype direction; higher = more
        flag-worthy. Ranking-safe — evaluate AUROC on this."""
        if self.direction is None:
            raise RuntimeError("calibrate() before score()")
        return float(self.embed([text])[0] @ self.direction)

    def score_batch(self, texts: Sequence[str]) -> np.ndarray:
        if self.direction is None:
            raise RuntimeError("calibrate() before score()")
        return self.embed(texts) @ self.direction

    def flag(self, text: str) -> bool:
        """True if the action scores above the calibrated threshold."""
        if self.threshold is None:
            raise RuntimeError("set a threshold (calibrate(fpr=...) or set_threshold)")
        return self.score(text) > self.threshold

    @property
    def calibrated(self) -> bool:
        return self.direction is not None and self.threshold is not None


def calibrate_from_labeled(
    texts: Sequence[str],
    labels: Sequence[int],
    benign_texts: Optional[Sequence[str]] = None,
    alpha: float = 0.1,
    model=None,
    model_name: str = DEFAULT_MODEL,
) -> CalibratedDetector:
    """Convenience: build + calibrate a detector in one call. When
    ``benign_texts`` is given the threshold is set on it, else on the
    label==0 subset of ``texts``."""
    det = CalibratedDetector(model=model, model_name=model_name)
    det.calibrate(texts, labels, fpr=None if benign_texts else alpha)
    if benign_texts:
        det.set_threshold(benign_texts, alpha=alpha)
    return det


__all__: List[str] = ["CalibratedDetector", "calibrate_from_labeled"]
