"""
Encoder protocol — the minimal text-embedding contract.

Every semantic component (:class:`~frontier_ops.detector.CalibratedDetector`,
``SemanticConceptExtractor``, ``StepMeanScorer``) embeds text through one
duck-typed operation. This module names that contract so any embedding
backend — sentence-transformers (the default ``all-MiniLM-L6-v2``), an API
client, a cached lookup table, or a deterministic test stub — can be
injected via the components' constructors without a sentence-transformers
install.

Contract:

- ``encode(texts)`` takes a sequence of strings and returns a float
  ``np.ndarray`` of shape ``(len(texts), d)``.
- Rows should be L2-normalized. Components normalize defensively where the
  math requires unit vectors, but conforming encoders should not rely on it.
- Extra keyword arguments (``batch_size``, ``convert_to_numpy``,
  ``normalize_embeddings``) MAY be accepted (sentence-transformers models
  do); callers fall back to the bare single-argument form when they are not.

Usage::

    class StubEncoder:
        def encode(self, texts):
            out = np.stack([my_embedding(t) for t in texts])
            return out / np.linalg.norm(out, axis=1, keepdims=True)

    det = CalibratedDetector(model=StubEncoder())
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["Encoder", "encode_batch"]


@runtime_checkable
class Encoder(Protocol):
    """Anything with ``encode(texts) -> (n, d)`` float array."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``; one L2-normalized row per input string."""
        ...


def encode_batch(encoder, texts: Sequence[str], **st_kwargs) -> np.ndarray:
    """Call ``encoder.encode`` with sentence-transformers kwargs when
    supported, falling back to the bare :class:`Encoder` form.

    Args:
        encoder: any :class:`Encoder`-conforming object.
        texts: strings to embed.
        **st_kwargs: sentence-transformers options (``batch_size``,
            ``convert_to_numpy``, ``normalize_embeddings``) forwarded on the
            rich path and dropped on the bare path.

    Returns:
        ``(len(texts), d)`` float array.
    """
    try:
        return np.asarray(encoder.encode(list(texts), **st_kwargs))
    except TypeError:
        out = np.asarray(encoder.encode(list(texts)))
        if st_kwargs.get("normalize_embeddings"):
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-12)
        return out
