"""Tests for push-based memory activation system."""

import numpy as np
import pytest
from frontier_ops.memory.activation import (
    MemoryEventBus,
    MemoryEventType,
    MemoryEvent,
    ActivationBuffer,
    AutoActivator,
)


def make_phasor(dim=512, seed=None):
    rng = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, dim)
    return np.exp(1j * phases)


def make_similar_phasor(base, noise=0.1, seed=None):
    rng = np.random.RandomState(seed)
    perturbation = rng.uniform(-noise * np.pi, noise * np.pi, base.shape[0])
    return np.exp(1j * (np.angle(base) + perturbation))


class FakeTrace:
    def __init__(self, text="", step=0, phasor=None):
        self.text = text
        self.step = step
        self.phasor = phasor or make_phasor()


class TestMemoryEventBus:
    def test_subscribe_and_emit(self):
        bus = MemoryEventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))
        bus.emit(MemoryEvent(event_type=MemoryEventType.ACTIVATION, timestamp=1.0))
        assert len(received) == 1

    def test_event_type_filtering(self):
        bus = MemoryEventBus()
        activations, novelties = [], []
        bus.subscribe(lambda e: activations.append(e), event_types={MemoryEventType.ACTIVATION})
        bus.subscribe(lambda e: novelties.append(e), event_types={MemoryEventType.NOVELTY})
        bus.emit(MemoryEvent(event_type=MemoryEventType.ACTIVATION, timestamp=1.0))
        bus.emit(MemoryEvent(event_type=MemoryEventType.NOVELTY, timestamp=2.0))
        assert len(activations) == 1
        assert len(novelties) == 1

    def test_priority_ordering(self):
        bus = MemoryEventBus()
        order = []
        bus.subscribe(lambda e: order.append("second"), priority=50)
        bus.subscribe(lambda e: order.append("first"), priority=0)
        bus.subscribe(lambda e: order.append("third"), priority=100)
        bus.emit(MemoryEvent(event_type=MemoryEventType.ACTIVATION, timestamp=1.0))
        assert order == ["first", "second", "third"]

    def test_unsubscribe(self):
        bus = MemoryEventBus()
        received = []
        sub_id = bus.subscribe(lambda e: received.append(e))
        bus.emit(MemoryEvent(event_type=MemoryEventType.ACTIVATION, timestamp=1.0))
        bus.unsubscribe(sub_id)
        bus.emit(MemoryEvent(event_type=MemoryEventType.ACTIVATION, timestamp=2.0))
        assert len(received) == 1

    def test_callback_exception_doesnt_break_others(self):
        bus = MemoryEventBus()
        received = []
        bus.subscribe(lambda e: (_ for _ in ()).throw(ValueError("boom")), priority=0)
        bus.subscribe(lambda e: received.append(e), priority=50)
        bus.emit(MemoryEvent(event_type=MemoryEventType.ACTIVATION, timestamp=1.0))
        assert len(received) == 1


class TestActivationBuffer:
    def test_add_and_scan_finds_similar(self):
        buf = ActivationBuffer(hot_threshold=0.1)
        base = make_phasor(seed=42)
        trace = FakeTrace(text="original")
        buf.add_trace(trace, base)
        query = make_similar_phasor(base, noise=0.05, seed=99)
        results = buf.scan(query)
        assert len(results) > 0
        assert results[0][1].text == "original"

    def test_dissimilar_not_found(self):
        buf = ActivationBuffer(hot_threshold=0.3)
        buf.add_trace(FakeTrace(), make_phasor(seed=1))
        results = buf.scan(make_phasor(seed=999))
        assert len(results) == 0

    def test_tiered_eviction(self):
        buf = ActivationBuffer(hot_size=3, warm_size=3)
        for i in range(10):
            buf.add_trace(FakeTrace(text=f"trace_{i}"), make_phasor(seed=i))
        assert len(buf._hot) == 3
        assert len(buf._warm) == 3
        assert len(buf._cold) >= 4  # Exact count depends on eviction timing

    def test_decay(self):
        buf = ActivationBuffer(decay_rate=0.5)
        trace = FakeTrace()
        buf.add_trace(trace, make_phasor())
        initial = buf._activation_levels[id(trace)]
        buf.decay_step()
        assert buf._activation_levels[id(trace)] == pytest.approx(initial * 0.5)

    def test_boost(self):
        buf = ActivationBuffer()
        trace = FakeTrace()
        buf.add_trace(trace, make_phasor())
        buf.decay_step()
        buf.decay_step()
        low = buf._activation_levels[id(trace)]
        buf.boost_activation(trace, amount=0.5)
        assert buf._activation_levels[id(trace)] > low

    def test_activation_caps_at_one(self):
        buf = ActivationBuffer()
        trace = FakeTrace()
        buf.add_trace(trace, make_phasor())
        buf.boost_activation(trace, amount=0.9)
        assert buf._activation_levels[id(trace)] <= 1.0

    def test_clear(self):
        buf = ActivationBuffer()
        for i in range(5):
            buf.add_trace(FakeTrace(), make_phasor(seed=i))
        buf.clear()
        assert buf.stats["total"] == 0


class TestAutoActivator:
    def test_perceive_returns_matches(self):
        bus = MemoryEventBus()
        act = AutoActivator(event_bus=bus)
        c, r = make_phasor(seed=42), make_phasor(seed=43)
        act.register_trace(FakeTrace(text="remembered"), c, r)
        results = act.perceive(
            concept_phasor=make_similar_phasor(c, noise=0.05),
            role_phasor=make_similar_phasor(r, noise=0.05),
        )
        assert len(results) > 0
        assert results[0][1].text == "remembered"

    def test_emits_activation_event(self):
        bus = MemoryEventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))
        act = AutoActivator(event_bus=bus)
        c, r = make_phasor(seed=42), make_phasor(seed=43)
        act.register_trace(FakeTrace(), c, r)
        act.perceive(concept_phasor=make_similar_phasor(c, 0.05), role_phasor=make_similar_phasor(r, 0.05))
        assert any(e.event_type == MemoryEventType.ACTIVATION for e in events)

    def test_emits_novelty_when_empty(self):
        bus = MemoryEventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))
        act = AutoActivator(event_bus=bus)
        act.perceive(concept_phasor=make_phasor(seed=1), role_phasor=make_phasor(seed=2))
        assert any(e.event_type == MemoryEventType.NOVELTY for e in events)

    def test_register_does_not_emit(self):
        bus = MemoryEventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))
        act = AutoActivator(event_bus=bus)
        act.register_trace(FakeTrace(), make_phasor(), make_phasor())
        assert len(events) == 0

    def test_stats(self):
        bus = MemoryEventBus()
        act = AutoActivator(event_bus=bus)
        c, r = make_phasor(seed=42), make_phasor(seed=43)
        act.register_trace(FakeTrace(), c, r)
        act.perceive(concept_phasor=c, role_phasor=r)
        s = act.stats
        assert s["perceive_count"] == 1
        assert s["buffer"]["total"] == 1
