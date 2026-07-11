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

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from frontier_ops.conformal import split_conformal_threshold

DEFAULT_MODEL = "all-MiniLM-L6-v2"

#: Serialization format version written by :meth:`CalibratedDetector.save`.
FORMAT_VERSION = 1


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
        # held-out benign scores from calibrate_conformal, kept so the
        # threshold can be re-derived at another alpha (persisted by save)
        self.conformal_scores: Optional[np.ndarray] = None

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

    def calibrate_conformal(
        self, benign_texts: Sequence[str], alpha: float = 0.1
    ) -> float:
        """Set the flag threshold by split-conformal calibration.

        Scores a *held-out* benign set and thresholds at the
        ``ceil((n+1)*(1-alpha))``-th smallest score. Under exchangeability of
        the calibration set with future benign traffic, the marginal
        false-positive rate of :meth:`flag` is guaranteed ≤ ``alpha``
        (finite-sample, distribution-free) — unlike :meth:`set_threshold`,
        whose plug-in quantile has no such guarantee at small ``n``.

        The benign scores are retained on ``conformal_scores`` (and included
        by :meth:`save`) so the threshold can be re-derived at a different
        ``alpha`` without re-embedding.

        Args:
            benign_texts: held-out benign examples, exchangeable with the
                deployment's benign traffic. Must not reuse the texts that
                fitted ``direction``.
            alpha: target marginal false-positive rate in (0, 1).

        Returns:
            The calibrated threshold (``+inf`` when ``len(benign_texts)`` is
            too small for the requested ``alpha``; nothing is flagged then).
        """
        if self.direction is None:
            raise RuntimeError("calibrate() before calibrate_conformal()")
        scores = np.asarray(self.embed(benign_texts) @ self.direction, dtype=float)
        self.conformal_scores = scores
        self.threshold = split_conformal_threshold(scores, alpha)
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

    # -- persistence ---------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the calibrated state to a single ``.npz`` file at ``path``.

        Writes the prototype ``direction`` array plus JSON metadata
        (``model_name``, ``threshold``, ``alpha``, ``n_calibration``, format
        version). The encoder itself is NOT serialized — only its name — so
        the file is small and the encoder is re-resolved lazily on load.

        Raises:
            RuntimeError: if called before :meth:`calibrate`.
        """
        if self.direction is None:
            raise RuntimeError("calibrate() before save()")
        meta = {
            "format_version": FORMAT_VERSION,
            "model_name": self._model_name,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
        }
        arrays = {
            "direction": self.direction,
            "meta": np.array(json.dumps(meta)),
        }
        if self.conformal_scores is not None:
            arrays["conformal_scores"] = self.conformal_scores
        # Writing through an open handle keeps np.savez from appending a
        # second ".npz" suffix, so save/load round-trip on the exact path.
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model=None,
        model_name: Optional[str] = None,
    ) -> "CalibratedDetector":
        """Restore a detector saved with :meth:`save`.

        The encoder is not loaded here — scoring stays lazy, so ``load`` works
        without sentence-transformers installed until the first
        ``score``/``embed`` call.

        Args:
            path: file written by :meth:`save`.
            model: a preloaded encoder to reuse (skips the lazy load).
            model_name: if given, must match the ``model_name`` recorded in
                the file — scores from a different encoder are not comparable.

        Raises:
            ValueError: on a ``model_name`` mismatch or an unknown (newer)
                format version.
        """
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            direction = np.array(data["direction"])
            conformal_scores = (
                np.array(data["conformal_scores"])
                if "conformal_scores" in data
                else None
            )
        version = int(meta.get("format_version", -1))
        if version > FORMAT_VERSION:
            raise ValueError(
                f"detector file {path!s} has format version {version}, newer "
                f"than this library supports ({FORMAT_VERSION}); upgrade "
                "frontier-ops to load it"
            )
        saved_name = meta["model_name"]
        if model_name is not None and model_name != saved_name:
            raise ValueError(
                f"detector file {path!s} was calibrated with encoder "
                f"'{saved_name}' but '{model_name}' was requested; scores "
                "from different encoders are not comparable — load with the "
                "matching model_name or recalibrate"
            )
        det = cls(model=model, model_name=saved_name)
        det.direction = direction
        det.threshold = meta["threshold"]
        det.alpha = meta["alpha"]
        det.n_calibration = int(meta["n_calibration"])
        det.conformal_scores = conformal_scores
        return det


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
