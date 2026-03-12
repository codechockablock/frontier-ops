"""
VSA Memory Substrate — Phasor Hyperdimensional Computing
=========================================================

Implements a hippocampal-style associative memory using phasor VSA operations.
Self-contained — no imports from proprioceptive-wrapper.

Core operations:
  phasor_encode(concept, dim)  — deterministic random phase vector (seeded by hash)
  bind(a, b)                   — element-wise complex multiplication (phase addition)
  bundle(vectors)              — circular mean of phases -> unit phasor
  similarity(a, b)             — cosine similarity of complex vectors

Memory operations:
  VSAMemory.encode()     — bind concept (x) role (x) position, store trace
  VSAMemory.activate()   — cosine similarity search over stored traces
  VSAMemory.validate()   — geometric consistency check before storage

Correctness targets:
  - encode->activate round-trip similarity > 0.85
  - capacity >= 50 distinct traces without degradation
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core phasor algebra primitives
# ---------------------------------------------------------------------------


def phasor_encode(concept: str, dim: int = 512) -> np.ndarray:
    """
    Generate a deterministic unit-magnitude phasor vector for a concept string.

    Uses SHA-256 of the concept to seed a numpy RNG, ensuring the same concept
    always maps to the same vector regardless of call order.

    Returns:
        Complex ndarray of shape (dim,) with |z_i| = 1 for all i.
    """
    digest = hashlib.sha256(concept.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, dim)
    return np.exp(1j * phases)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Bind two phasor vectors via element-wise complex multiplication.

    In phasor VSA, multiplication = phase addition.
    Result is still unit-magnitude if inputs are unit-magnitude.
    """
    return a * b


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Bundle (superpose) phasor vectors via circular mean of phases.

    Sums the complex vectors, then normalizes to unit magnitude.
    This computes the circular mean direction of the phase ensemble.

    Args:
        vectors: Non-empty list of phasor arrays (same shape).

    Returns:
        Unit-magnitude phasor array in the average phase direction.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list of vectors")
    summed = np.sum(np.stack(vectors, axis=0), axis=0)
    magnitude = np.abs(summed)
    magnitude = np.where(magnitude < 1e-10, 1.0, magnitude)
    return summed / magnitude


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two complex phasor vectors.

    For unit-magnitude phasors this equals Re(<a,b>)/sqrt(<a,a><b,b>).
    Returns a float in [-1, 1].  High value = similar phase structure.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    raw = float(np.real(np.vdot(a, b))) / (norm_a * norm_b)
    return float(np.clip(raw, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Memory trace data structure
# ---------------------------------------------------------------------------


@dataclass
class MemoryTrace:
    """A single stored memory trace."""

    key: np.ndarray       # VSA key = bind(concept_vec, role_vec) -- used for retrieval
    content: np.ndarray   # VSA content = bind(key, position_vec) -- full encoding
    concept: str          # Original concept label (e.g. "math_reasoning")
    role: str             # Role in the trace (e.g. "reasoning", "conclusion")
    position: int         # Position within the step sequence
    text: str             # Original text of this step
    step: int             # Agent step number when encoded
    metadata: dict        # Optional extra data (concept scores, angular displacement, etc.)


# ---------------------------------------------------------------------------
# VSA Memory
# ---------------------------------------------------------------------------

# Recognised role names -- other roles are supported but won't use cached vectors
_KNOWN_ROLES = [
    "reasoning",
    "conclusion",
    "observation",
    "hypothesis",
    "correction",
    "plan",
    "query",
]


class VSAMemory:
    """
    Hippocampal-style associative memory using phasor VSA.

    Each memory trace encodes: concept (x) role (x) position
    Retrieval is by cosine similarity search over stored keys.

    Usage::

        mem = VSAMemory(dim=512)
        trace = mem.encode("I will solve the integral by parts", "math", "reasoning", 0, step=1)
        hits = mem.activate("math", "reasoning", top_k=3)
        similar_text = hits[0][1].text
    """

    def __init__(self, dim: int = 512, capacity: int = 200):
        self.dim = dim
        self.capacity = capacity
        self.traces: List[MemoryTrace] = []

        # Pre-compute fixed role vectors (deterministic -- same across instances)
        self._role_vecs: dict[str, np.ndarray] = {
            role: phasor_encode(f"__role__{role}__", dim) for role in _KNOWN_ROLES
        }
        # Position vector cache (created on demand)
        self._pos_cache: dict[int, np.ndarray] = {}

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _role_vec(self, role: str) -> np.ndarray:
        if role not in self._role_vecs:
            self._role_vecs[role] = phasor_encode(f"__role__{role}__", self.dim)
        return self._role_vecs[role]

    def _pos_vec(self, position: int) -> np.ndarray:
        if position not in self._pos_cache:
            self._pos_cache[position] = phasor_encode(f"__pos__{position}__", self.dim)
        return self._pos_cache[position]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def encode(
        self,
        text: str,
        concept: str,
        role: str = "reasoning",
        position: int = 0,
        step: int = 0,
        metadata: Optional[dict] = None,
    ) -> MemoryTrace:
        """
        Encode a text step into VSA memory.

        Binding structure:
            key     = concept_vec (x) role_vec          (for retrieval)
            content = key (x) position_vec              (full encoding)

        The trace is stored in the memory buffer (FIFO eviction at capacity).
        Returns the MemoryTrace for immediate use (e.g. validate before storage).
        """
        concept_vec = phasor_encode(concept, self.dim)
        role_v = self._role_vec(role)
        pos_v = self._pos_vec(position)

        key = bind(concept_vec, role_v)
        content = bind(key, pos_v)

        trace = MemoryTrace(
            key=key,
            content=content,
            concept=concept,
            role=role,
            position=position,
            text=text,
            step=step,
            metadata=metadata or {},
        )

        if len(self.traces) >= self.capacity:
            self.traces.pop(0)  # evict oldest

        self.traces.append(trace)
        return trace

    def activate(
        self,
        query_concept: str,
        query_role: str = "reasoning",
        top_k: int = 5,
        threshold: float = 0.2,
    ) -> List[Tuple[float, MemoryTrace]]:
        """
        Find stored traces matching the query concept + role.

        Computes cosine similarity between query key and every stored key.
        Returns list of (score, trace) sorted by descending similarity.
        Only traces with similarity >= threshold are returned.
        """
        if not self.traces:
            return []

        concept_vec = phasor_encode(query_concept, self.dim)
        role_v = self._role_vec(query_role)
        query_key = bind(concept_vec, role_v)

        scored: List[Tuple[float, MemoryTrace]] = []
        for trace in self.traces:
            sim = similarity(query_key, trace.key)
            if sim >= threshold:
                scored.append((sim, trace))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def activate_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.2,
    ) -> List[Tuple[float, MemoryTrace]]:
        """
        Find stored traces by direct content-vector similarity.

        Useful when the caller has already computed a composite context vector.
        """
        if not self.traces:
            return []

        scored: List[Tuple[float, MemoryTrace]] = []
        for trace in self.traces:
            sim = similarity(query_vec, trace.content)
            if sim >= threshold:
                scored.append((sim, trace))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def validate(self, trace: MemoryTrace) -> bool:
        """
        Check geometric consistency of a trace before long-term storage.

        Validation criteria:
          1. Phase distribution looks like a random phasor (std > 0.5 rad)
          2. Content vector is not strongly anti-correlated with recent memory
             (would indicate encoding corruption or degenerate state)

        Returns True if the trace is geometrically consistent.
        """
        phases = np.angle(trace.content)
        phase_std = float(np.std(phases))

        # A valid random phasor has phase std ~ pi/sqrt(3) ~ 1.81 rad.
        # Degenerate vectors (all same phase) would have std ~ 0.
        if phase_std < 0.5:
            return False

        # Anti-correlation with recent bundle is suspicious (not just orthogonal)
        if len(self.traces) >= 3:
            recent = [t.content for t in self.traces[-5:]]
            bundled = bundle(recent)
            sim = similarity(trace.content, bundled)
            if sim < -0.8:
                return False

        return True

    def bundle_recent(self, n: int = 10) -> Optional[np.ndarray]:
        """Bundle the n most recent traces into a single context vector."""
        if not self.traces:
            return None
        recent = [t.content for t in self.traces[-n:]]
        return bundle(recent)

    def summary(self) -> dict:
        """Return a compact summary of memory state."""
        return {
            "stored_traces": len(self.traces),
            "capacity": self.capacity,
            "dim": self.dim,
            "oldest_step": self.traces[0].step if self.traces else None,
            "newest_step": self.traces[-1].step if self.traces else None,
        }

    def clear(self) -> None:
        """Clear all stored traces."""
        self.traces.clear()
        self._pos_cache.clear()
