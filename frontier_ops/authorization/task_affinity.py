"""
Task-Affinity Channel — Escalate-Only Directional Signal
=========================================================

Supplies the task-identity information the 6-dim concept space cannot
represent (see eval/results/goal-conditioning-2026-07-03.md and the
controls report): cosine affinity between the directive text and the
action text in sentence-embedding space.

THE ASYMMETRY RULE (the mimicry answer — structural, not conventional):

    Low affinity may only ADD suspicion (escalate).
    High affinity must never REMOVE suspicion — it cannot authorize,
    relax a boundary, expand a radius, or offset a concept-space verdict.

Embedding affinity is adversarially mimicable: a covert action can borrow
the directive's vocabulary to inflate cosine similarity. Under the
asymmetry rule, mimicry only buys an attacker back to the baseline they
would have had without this channel — it never unlocks anything. The
non-mimicable floor remains the concept-space risk placement (an action's
credential/self-modification surface is scored from what it DOES, not how
it is worded) plus the LOCKED dims.

Thresholds are never hand-set: an uncalibrated AffinityThreshold flags
nothing. Calibration is split-conformal on benign own-directive affinities
(lower-tail quantile), mirroring AuthorizationRadius.calibrate.

Mimicry robustness is UNTESTED (no adversarial data exists); do not claim
it. See fable-spec-task-affinity.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

EmbedFn = Callable[[str], np.ndarray]


def _sentence_transformer_embed_fn(
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[EmbedFn]:
    """Build an embed_fn backed by sentence-transformers, or None."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(model_name)

    def embed(text: str) -> np.ndarray:
        return model.encode(text, convert_to_numpy=True)

    return embed


@dataclass
class AffinityThreshold:
    """Lower-tail conformal threshold: escalate when affinity < threshold.

    Uncalibrated (the default) means NO flagging — there is deliberately no
    hand-set default constant.
    """
    threshold: Optional[float] = None
    calibrated: bool = False
    calibration_alpha: float = 0.1
    calibration_n: int = 0

    def calibrate(self, benign_affinities: List[float], alpha: float = 0.1):
        """Split-conformal lower-tail calibration.

        Given affinities of known-benign (directive, action) pairs, set the
        threshold so that a new benign pair falls below it with probability
        <= alpha (distribution-free). Mirrors AuthorizationRadius.calibrate,
        lower tail instead of upper.
        """
        if len(benign_affinities) < 10:
            return  # Not enough data to calibrate
        n = len(benign_affinities)
        # Conformal lower quantile: floor((n+1) * alpha) / n
        q = math.floor((n + 1) * alpha) / n
        q = max(q, 0.0)
        self.threshold = float(np.quantile(benign_affinities, q))
        self.calibrated = True
        self.calibration_alpha = alpha
        self.calibration_n = n

    def flags(self, affinity: float) -> Optional[bool]:
        """True = escalate-worthy (below threshold). None if uncalibrated."""
        if not self.calibrated or self.threshold is None:
            return None
        return affinity < self.threshold


class TaskAffinityChannel:
    """
    Directive<->action affinity in sentence-embedding space.

    affinity() returns cosine similarity of unit-normalized embeddings
    (cached per text). Consumers must honor the asymmetry rule documented
    in the module docstring; GoalConditioningScorer enforces it
    structurally by never letting affinity touch `authorized`.

    Args:
        embed_fn: text -> embedding vector. Injectable for tests; defaults
            to sentence-transformers all-MiniLM-L6-v2.
        threshold: AffinityThreshold; a fresh (uncalibrated, never-flagging)
            one if omitted.
    """

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        threshold: Optional[AffinityThreshold] = None,
    ):
        if embed_fn is None:
            embed_fn = _sentence_transformer_embed_fn()
            if embed_fn is None:
                raise ImportError(
                    "TaskAffinityChannel requires sentence-transformers "
                    "(or an injected embed_fn). Install with: "
                    "pip install sentence-transformers"
                )
        self._embed_fn = embed_fn
        self.threshold = threshold or AffinityThreshold()
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def create(
        cls, threshold: Optional[AffinityThreshold] = None,
    ) -> Optional["TaskAffinityChannel"]:
        """Factory returning None when sentence-transformers is missing
        (mirrors SemanticConceptExtractor.create)."""
        embed_fn = _sentence_transformer_embed_fn()
        if embed_fn is None:
            return None
        return cls(embed_fn=embed_fn, threshold=threshold)

    def _unit_embedding(self, text: str) -> np.ndarray:
        if text not in self._cache:
            v = np.asarray(self._embed_fn(text), dtype=float)
            self._cache[text] = v / (np.linalg.norm(v) + 1e-10)
        return self._cache[text]

    def affinity(self, directive_text: str, action_text: str) -> float:
        """Cosine affinity between directive and action texts."""
        e_d = self._unit_embedding(directive_text)
        e_a = self._unit_embedding(action_text)
        return float(e_d @ e_a)

    def flags(self, directive_text: str, action_text: str) -> Optional[bool]:
        """Escalate-only signal; None when the threshold is uncalibrated."""
        return self.threshold.flags(self.affinity(directive_text, action_text))
