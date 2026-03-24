"""Semantic encoder: frozen MiniLM → phasor projection for action content.

Encodes command/content strings into unit-magnitude complex phasor vectors
in C^512 using a frozen all-MiniLM-L6-v2 model and fixed random projection.

Degrades gracefully if sentence_transformers is not installed.
"""

from __future__ import annotations

import logging
import functools
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_model = None
_model_load_attempted = False
_projection_matrix: Optional[np.ndarray] = None
_st_available: Optional[bool] = None

PHASOR_DIM = 512
EMBED_DIM = 384
PROJECTION_SEED = 42


def _get_projection_matrix() -> np.ndarray:
    """Return the fixed random projection matrix W: (PHASOR_DIM, EMBED_DIM)."""
    global _projection_matrix
    if _projection_matrix is None:
        rng = np.random.RandomState(PROJECTION_SEED)
        _projection_matrix = rng.randn(PHASOR_DIM, EMBED_DIM).astype(np.float64)
    return _projection_matrix


def _get_model():
    """Lazy-load the MiniLM model. Returns None if unavailable."""
    global _model, _model_load_attempted, _st_available

    if _model_load_attempted:
        return _model

    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer

        _st_available = True
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model
    except ImportError:
        _st_available = False
        logger.warning(
            "semantic_encoder: sentence_transformers not available, "
            "semantic slot will be zero"
        )
        return None
    except Exception as exc:
        _st_available = False
        logger.warning(
            "semantic_encoder: failed to load model (%s), "
            "semantic slot will be zero",
            exc,
        )
        return None


@functools.lru_cache(maxsize=2000)
def encode_semantic(text: str) -> np.ndarray:
    """Encode a command/content string into a phasor vector.

    Pipeline:
    1. Embed with frozen all-MiniLM-L6-v2 → R^384
    2. Project: R^384 → R^512 via fixed random matrix W (seed=42)
    3. Phase-encode: phase = pi * tanh(projected) → C^512 via exp(i*phase)
    4. Return unit-magnitude complex vector

    Returns zeros if sentence_transformers is not available (graceful degradation).
    """
    model = _get_model()
    if model is None:
        return np.zeros(PHASOR_DIM, dtype=complex)

    try:
        # Step 1: Embed with MiniLM
        embedding = model.encode(text, show_progress_bar=False)
        embedding = np.asarray(embedding, dtype=np.float64).flatten()

        if embedding.shape[0] != EMBED_DIM:
            return np.zeros(PHASOR_DIM, dtype=complex)

        # Step 2: Project R^384 → R^512
        W = _get_projection_matrix()
        projected = W @ embedding

        # Step 3: Phase-encode → C^512
        phase = np.pi * np.tanh(projected)
        phasor = np.exp(1j * phase)

        # Step 4: Unit magnitude (already unit by construction since |exp(ix)|=1,
        # but normalize for numerical safety)
        norm = np.linalg.norm(phasor)
        if norm > 1e-9:
            phasor = phasor / norm
        return phasor

    except Exception as exc:
        logger.debug("semantic_encoder: encoding failed (%s), returning zeros", exc)
        return np.zeros(PHASOR_DIM, dtype=complex)
