"""Shared encoders + encoding caches for the v2 eval battery.

Uses the Phase 1 constructor-param APIs (dims/anchors) instead of the module
mutation the session artifacts had to resort to. Anchor phrases come verbatim
from eval/session_artifacts/earned_anchors.py — the single source of truth.

MiniLM truncates at 256 wordpieces, so texts are always encoded per step /
per response, never concatenated beyond the artifacts' 1200-char exchange cap.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from eval.battery.data import (
    TASKS,
    cache_dir,
    load_binary,
    load_insider_episodes,
)
from eval.session_artifacts.earned_anchors import DEC_ANCHORS

DEC_DIMS: List[str] = [
    "fabricated_justification",
    "strategic_vagueness",
    "impression_management",
    "honest_disclosure",
]

MODEL_NAME = "all-MiniLM-L6-v2"  # campaign default — do not change (§6)

# Phase 5 (stretch): the battery can run under an alternative encoder via
# `python -m eval.battery --encoder NAME`. Alternative-encoder runs are
# informational (deltas vs the §6 baselines); the default stays MiniLM.
_ACTIVE_MODEL = MODEL_NAME

_ST_CACHE: Dict[str, object] = {}
_ST_PATCHED = False


def set_model(name: str) -> None:
    """Switch the battery's encoder (Phase 5 experiment flag)."""
    global _ACTIVE_MODEL, _CHART_EXTRACTOR
    if name != _ACTIVE_MODEL:
        _ACTIVE_MODEL = name
        _CHART_EXTRACTOR = None


def active_model() -> str:
    return _ACTIVE_MODEL


def _cache_name(base: str) -> str:
    """Cache filename, suffixed with a model slug for non-default encoders
    so alternative-encoder runs never poison the baseline caches."""
    if _ACTIVE_MODEL == MODEL_NAME:
        return f"{base}.npz"
    slug = _ACTIVE_MODEL.split("/")[-1]
    return f"{base}--{slug}.npz"


def install_st_cache() -> None:
    """Memoize SentenceTransformer by model name (session-artifact trick) so
    repeated extractor/pipeline construction doesn't reload MiniLM."""
    global _ST_PATCHED
    if _ST_PATCHED:
        return
    import sentence_transformers

    orig = sentence_transformers.SentenceTransformer

    def _cached(name: str, *a, **k):
        if name not in _ST_CACHE:
            _ST_CACHE[name] = orig(name, *a, **k)
        return _ST_CACHE[name]

    sentence_transformers.SentenceTransformer = _cached
    _ST_PATCHED = True


_CHART_EXTRACTOR = None


def chart_extractor():
    """Singleton 4-D deception-chart extractor (Tier-2, earned anchors)."""
    global _CHART_EXTRACTOR
    if _CHART_EXTRACTOR is None:
        install_st_cache()
        from frontier_ops.boundary.semantic_extraction import SemanticConceptExtractor

        _CHART_EXTRACTOR = SemanticConceptExtractor(
            model_name=_ACTIVE_MODEL, dims=DEC_DIMS, anchors=DEC_ANCHORS
        )
    return _CHART_EXTRACTOR


def chart_vec(scores: Dict[str, float]) -> np.ndarray:
    return np.array([scores[d] for d in DEC_DIMS])


def chart_encodings(task: str, log=print) -> Tuple[np.ndarray, np.ndarray]:
    """4-D chart encodings of response texts, cached (encode_cache.py port)."""
    cache = cache_dir() / _cache_name(f"enc_{task}")
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["Y"]
    texts, Y = load_binary(task)
    enc = chart_extractor()
    t0 = time.time()
    X = []
    for i, t in enumerate(texts):
        X.append(chart_vec(enc.extract(t)))
        if i % 200 == 0:
            log(f"  encode {task} {i}/{len(texts)} [{time.time() - t0:.0f}s]")
    X = np.array(X)
    np.savez(cache, X=X, Y=Y)
    log(f"  {task}: n={len(Y)} chart-encoded [{time.time() - t0:.0f}s]")
    return X, Y


def raw_encodings(task: str, log=print) -> Tuple[np.ndarray, np.ndarray]:
    """Raw 384-D normalized MiniLM encodings, cached (encode_raw.py port)."""
    cache = cache_dir() / _cache_name(f"raw_{task}")
    if cache.exists():
        z = np.load(cache)
        return z["E"], z["Y"]
    texts, Y = load_binary(task)
    model = chart_extractor().model
    t0 = time.time()
    E = model.encode(
        texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
    )
    np.savez(cache, E=E, Y=Y)
    log(f"  {task}: n={len(Y)} raw-encoded 384-D [{time.time() - t0:.0f}s]")
    return E, Y


def insider_step_encodings(
    log=print,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Per-step + full-response chart encodings for insider episodes, cached
    (curvature_kill_test.py port). Returns (STEP_X, FULL_X, Y)."""
    cache = cache_dir() / _cache_name("enc_insider_steps")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["step_x"]), z["full_x"], z["Y"]
    episodes, Y = load_insider_episodes()
    enc = chart_extractor()
    t0 = time.time()
    STEP_X: List[np.ndarray] = []
    FULL_X = []
    for i, e in enumerate(episodes):
        STEP_X.append(np.stack([chart_vec(enc.extract(s)) for s in e["steps"]]))
        FULL_X.append(chart_vec(enc.extract(e["full"])))
        if i % 40 == 0:
            log(f"  encode insider steps {i}/{len(episodes)} [{time.time() - t0:.0f}s]")
    FULL_X = np.stack(FULL_X)
    np.savez(
        cache, step_x=np.array(STEP_X, dtype=object), full_x=FULL_X, Y=Y
    )
    log(f"  insider: {len(Y)} episodes step-encoded [{time.time() - t0:.0f}s]")
    return STEP_X, FULL_X, Y


def all_chart_encodings(log=print) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {t: chart_encodings(t, log=log) for t in TASKS}


def all_raw_encodings(log=print) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {t: raw_encodings(t, log=log) for t in TASKS}
