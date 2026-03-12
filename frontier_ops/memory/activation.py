"""
Push-based memory activation system.

Adds to existing VSAMemory:
  - MemoryEvent: typed events for activations, novelty detection, interference
  - MemoryEventBus: observer pattern with priority-ordered callbacks
  - ActivationBuffer: tiered recency-weighted trace index for fast matching
  - AutoActivator: the main component -- sits between perception and encoding,
    fires pattern completion automatically on new input

Integration point: call memory.perceive(concept_vec, role_vec) at the START
of each agent step, BEFORE the LLM call. Activated memories arrive via
registered callbacks and get injected into the prompt context.

Architecture:
  input arrives -> perceive() -> key-based scan of active buffer
    -> top-k candidates above threshold -> content reranking
    -> MemoryEvents emitted -> callbacks fire -> agent receives primed memories
    -> LLM called with primed context -> response -> encode() stores new trace

Design from Claude.ai collaboration (2026-02-28):
  - Key-based primary matching (hippocampal indexing)
  - Content reranking secondary (neocortical pattern completion)
  - Tiered buffer: hot/warm/cold with recency-weighted thresholds
  - Synchronous callbacks (theta-phase ordering: perceive -> prime -> act -> encode)
  - Interference detection (fan effect) and novelty detection as proprioceptive signals
"""

from __future__ import annotations

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# --- Events ----------------------------------------------------------------

class MemoryEventType(Enum):
    """Types of memory events the system can emit."""
    ACTIVATION = auto()       # Trace(s) matched above threshold
    NOVELTY = auto()          # Input has NO good matches -- new territory
    INTERFERENCE = auto()     # Multiple strong matches conflict (fan effect)
    RECONSOLIDATION = auto()  # Existing trace updated by new activation
    DECAY = auto()            # Trace dropped from active buffer


@dataclass(frozen=True)
class MemoryEvent:
    """Immutable event emitted by the memory system."""
    event_type: MemoryEventType
    timestamp: float
    # For ACTIVATION: the matched traces with scores
    activated_traces: tuple = ()  # Tuple[Tuple[float, MemoryTrace], ...]
    # For NOVELTY: the query that found nothing
    query_key: Optional[np.ndarray] = field(default=None, repr=False)
    novelty_score: float = 0.0
    # For INTERFERENCE: the conflicting traces
    conflict_set: tuple = ()
    # Metadata for audit trail
    step: int = 0
    trigger: str = ""  # What caused this event ("perceive", "encode", "replay")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def top_activation(self):
        """Highest-scoring activated trace, if any."""
        if self.activated_traces:
            return self.activated_traces[0]
        return None

    @property
    def n_activated(self) -> int:
        return len(self.activated_traces)


# --- Observer Pattern ------------------------------------------------------

@dataclass
class _Subscription:
    callback: Callable[[MemoryEvent], None]
    event_types: frozenset
    priority: int  # Lower = fires first


class MemoryEventBus:
    """
    Priority-ordered event bus for memory system.

    Callbacks fire synchronously in priority order (lowest first).
    This is intentional -- memory priming must complete before the
    agent continues to the LLM call.
    """

    def __init__(self):
        self._subscriptions: List[_Subscription] = []
        self._event_history: deque = deque(maxlen=1000)
        self._emit_count = 0

    def subscribe(
        self,
        callback: Callable[[MemoryEvent], None],
        event_types: Optional[set] = None,
        priority: int = 50,
    ) -> int:
        """Register a callback. None event_types = all events. Returns sub ID."""
        types = frozenset(event_types) if event_types else frozenset(MemoryEventType)
        sub = _Subscription(callback=callback, event_types=types, priority=priority)
        self._subscriptions.append(sub)
        self._subscriptions.sort(key=lambda s: s.priority)
        return id(sub)

    def unsubscribe(self, subscription_id: int):
        """Remove a subscription by ID."""
        self._subscriptions = [
            s for s in self._subscriptions if id(s) != subscription_id
        ]

    def emit(self, event: MemoryEvent):
        """Emit an event to all matching subscribers, synchronously."""
        self._event_history.append(event)
        self._emit_count += 1
        for sub in self._subscriptions:
            if event.event_type in sub.event_types:
                try:
                    sub.callback(event)
                except Exception as e:
                    logger.error(f"Memory event callback failed: {e}", exc_info=True)

    @property
    def recent_events(self) -> List[MemoryEvent]:
        return list(self._event_history)


# --- Activation Buffer -----------------------------------------------------

class ActivationBuffer:
    """
    Tiered memory buffer with recency-weighted activation thresholds.

    Mirrors hippocampal memory organization:
    - Hot tier: last N traces, lowest activation threshold
    - Warm tier: recent traces, moderate threshold
    - Cold tier: older traces, highest threshold
    """

    def __init__(
        self,
        hot_size: int = 50,
        warm_size: int = 200,
        hot_threshold: float = 0.15,
        warm_threshold: float = 0.25,
        cold_threshold: float = 0.35,
        decay_rate: float = 0.995,
    ):
        self.hot_size = hot_size
        self.warm_size = warm_size
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self.cold_threshold = cold_threshold
        self.decay_rate = decay_rate

        self._hot: deque = deque(maxlen=hot_size)
        self._warm: deque = deque(maxlen=warm_size)
        self._cold: List = []
        self._activation_levels: dict = {}  # trace_id -> level

    def add_trace(self, trace: Any, key_phasor: np.ndarray):
        """Add a new trace to hot tier. Cascades to warm/cold."""
        entry = (trace, key_phasor)

        if len(self._hot) >= self.hot_size:
            evicted = self._hot[0]
            self._warm.append(evicted)
            if len(self._warm) >= self.warm_size:
                cold_evicted = self._warm[0]
                self._cold.append(cold_evicted)

        self._hot.append(entry)
        self._activation_levels[id(trace)] = 1.0

    def scan(
        self,
        query_key: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[float, Any, str]]:
        """Scan all tiers. Returns (score, trace, tier_name) sorted by score."""
        candidates = []

        for tier_name, tier_data, threshold in [
            ("hot", self._hot, self.hot_threshold),
            ("warm", self._warm, self.warm_threshold),
            ("cold", self._cold, self.cold_threshold),
        ]:
            if not tier_data:
                continue

            keys = np.array([entry[1] for entry in tier_data])
            sims = np.abs(keys @ query_key.conj()) / query_key.shape[0]

            for i, sim in enumerate(sims):
                if sim >= threshold:
                    trace = tier_data[i][0]
                    trace_id = id(trace)
                    activation = self._activation_levels.get(trace_id, 0.5)
                    score = float(sim) * activation
                    candidates.append((score, trace, tier_name))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:top_k]

    def decay_step(self):
        """Apply activation decay. Call once per agent step."""
        for trace_id in list(self._activation_levels.keys()):
            self._activation_levels[trace_id] *= self.decay_rate
            if self._activation_levels[trace_id] < 0.01:
                del self._activation_levels[trace_id]

    def boost_activation(self, trace: Any, amount: float = 0.3):
        """Boost a trace's activation level (retrieval strengthens memory)."""
        trace_id = id(trace)
        current = self._activation_levels.get(trace_id, 0.0)
        self._activation_levels[trace_id] = min(1.0, current + amount)

    @property
    def stats(self) -> dict:
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "cold": len(self._cold),
            "total": len(self._hot) + len(self._warm) + len(self._cold),
            "active_traces": len(self._activation_levels),
        }

    def clear(self):
        """Clear all tiers."""
        self._hot.clear()
        self._warm.clear()
        self._cold.clear()
        self._activation_levels.clear()


# --- Auto-Activator --------------------------------------------------------

class AutoActivator:
    """
    Hippocampal-style automatic pattern completion.

    Sits between perception and encoding:
        input -> concept_extraction -> perceive() -> [activations fire] -> LLM call -> encode()

    The agent doesn't decide whether to check memory -- memory announces itself.
    """

    def __init__(
        self,
        event_bus: MemoryEventBus,
        buffer: Optional[ActivationBuffer] = None,
        top_k: int = 5,
        novelty_threshold: float = 0.15,
        interference_threshold: float = 0.8,
        interference_min_count: int = 3,
        content_rerank: bool = True,
        content_weight: float = 0.3,
    ):
        self.event_bus = event_bus
        self.buffer = buffer or ActivationBuffer()
        self.top_k = top_k
        self.novelty_threshold = novelty_threshold
        self.interference_threshold = interference_threshold
        self.interference_min_count = interference_min_count
        self.content_rerank = content_rerank
        self.content_weight = content_weight

        self._step = 0
        self._perceive_count = 0
        self._total_activations = 0

    def perceive(
        self,
        concept_phasor: np.ndarray,
        role_phasor: np.ndarray,
        content_phasor: Optional[np.ndarray] = None,
        step: Optional[int] = None,
    ) -> List[Tuple[float, Any]]:
        """
        Perceive new input and automatically activate matching memories.
        Call BEFORE the LLM call. Returns activated (score, trace) pairs
        AND emits events to all subscribers.
        """
        if step is not None:
            self._step = step
        self._perceive_count += 1

        # Bind concept (x) role into query key
        query_key = concept_phasor * role_phasor
        query_key = np.exp(1j * np.angle(query_key))  # Renormalize to unit circle

        # Scan activation buffer
        candidates = self.buffer.scan(query_key, top_k=self.top_k * 2)

        # Content reranking
        if self.content_rerank and content_phasor is not None and candidates:
            candidates = self._content_rerank(candidates, content_phasor)

        candidates = candidates[:self.top_k]

        # Emit events
        self._emit_events(candidates, query_key)

        # Boost activation of matched traces
        for score, trace, _tier in candidates:
            self.buffer.boost_activation(trace, amount=0.2)

        # Decay all activations
        self.buffer.decay_step()

        result = [(score, trace) for score, trace, _tier in candidates]
        self._total_activations += len(result)
        return result

    def register_trace(self, trace: Any, concept_phasor: np.ndarray, role_phasor: np.ndarray):
        """Register a newly encoded trace. Does NOT trigger activation."""
        key_phasor = concept_phasor * role_phasor
        key_phasor = np.exp(1j * np.angle(key_phasor))
        self.buffer.add_trace(trace, key_phasor)

    def _content_rerank(
        self,
        candidates: List[Tuple[float, Any, str]],
        content_phasor: np.ndarray,
    ) -> List[Tuple[float, Any, str]]:
        """Second-stage reranking using content similarity."""
        reranked = []
        for key_score, trace, tier in candidates:
            trace_content = getattr(trace, 'phasor', None)
            if trace_content is None:
                trace_content = getattr(trace, 'content', None)

            if trace_content is not None and isinstance(trace_content, np.ndarray):
                content_sim = float(
                    np.abs(np.dot(trace_content, content_phasor.conj()))
                    / content_phasor.shape[0]
                )
                blended = (1 - self.content_weight) * key_score + self.content_weight * content_sim
            else:
                blended = key_score

            reranked.append((blended, trace, tier))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked

    def _emit_events(
        self,
        candidates: List[Tuple[float, Any, str]],
        query_key: np.ndarray,
    ):
        """Analyze activation results and emit appropriate events."""
        now = time.time()

        if not candidates:
            self.event_bus.emit(MemoryEvent(
                event_type=MemoryEventType.NOVELTY,
                timestamp=now,
                query_key=query_key,
                novelty_score=1.0,
                step=self._step,
                trigger="perceive",
            ))
            return

        top_score = candidates[0][0]

        if top_score < self.novelty_threshold:
            self.event_bus.emit(MemoryEvent(
                event_type=MemoryEventType.NOVELTY,
                timestamp=now,
                query_key=query_key,
                novelty_score=1.0 - top_score,
                step=self._step,
                trigger="perceive",
            ))
            return

        # ACTIVATION event
        activated = tuple((score, trace) for score, trace, _tier in candidates)
        self.event_bus.emit(MemoryEvent(
            event_type=MemoryEventType.ACTIVATION,
            timestamp=now,
            activated_traces=activated,
            step=self._step,
            trigger="perceive",
        ))

        # Check for INTERFERENCE (fan effect)
        strong_matches = [
            (s, t) for s, t, _tier in candidates
            if s > self.interference_threshold * top_score
        ]
        if len(strong_matches) >= self.interference_min_count:
            self.event_bus.emit(MemoryEvent(
                event_type=MemoryEventType.INTERFERENCE,
                timestamp=now,
                conflict_set=tuple(strong_matches),
                step=self._step,
                trigger="perceive",
            ))

    @property
    def stats(self) -> dict:
        return {
            "perceive_count": self._perceive_count,
            "total_activations": self._total_activations,
            "avg_activations_per_perceive": (
                self._total_activations / self._perceive_count
                if self._perceive_count > 0 else 0
            ),
            "buffer": self.buffer.stats,
        }
