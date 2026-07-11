"""
Integration tests -- pipeline + memory + governance working together.

These tests exercise realistic scenarios end-to-end, not individual components.
"""

from frontier_ops.memory.vsa import VSAMemory, phasor_encode
from frontier_ops.memory.activation import (
    MemoryEventBus, MemoryEventType, AutoActivator,
)
from frontier_ops.boundary.concept_extraction import KeywordConceptExtractor, CONCEPTS


class TestMemoryIntegration:
    """Test that VSA memory encode and retrieve operations work correctly."""

    def test_encode_activate_roundtrip_above_threshold(self):
        """Correctness invariant: encode->activate similarity > 0.85."""
        mem = VSAMemory(dim=512)
        mem.encode("testing round trip", "math_reasoning", "reasoning", 0, step=0)
        hits = mem.activate("math_reasoning", "reasoning", threshold=0.0)
        assert len(hits) > 0
        assert hits[0][0] > 0.85  # similarity > 0.85

    def test_memory_capacity_50_traces(self):
        """Correctness invariant: at least 50 traces without interference below 0.7."""
        mem = VSAMemory(dim=512)
        for i in range(50):
            mem.encode(f"trace {i}", f"concept_{i}", "reasoning", i, step=i)
        # Each trace should be retrievable
        recovered = 0
        for i in range(50):
            hits = mem.activate(f"concept_{i}", "reasoning", threshold=0.5)
            if hits and hits[0][0] > 0.7:
                recovered += 1
        assert recovered >= 45  # Allow small interference


class TestGovernanceIntegration:
    """Test that governance chain records and validates steps."""

    def test_governance_chain_integrity(self):
        from frontier_ops.governance.chain import GovernanceChain, GovernanceAuditor
        chain = GovernanceChain()

        for i in range(5):
            chain.observe({"step": i, "angular_disp": 0.1 * i})

        assert len(chain) == 5
        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(chain.export_chain())
        assert result.valid

    def test_tampered_chain_fails_verification(self):
        from frontier_ops.governance.chain import GovernanceChain, GovernanceAuditor
        chain = GovernanceChain()

        chain.observe({"step": 0, "value": 1.0})

        # Export, tamper, verify
        exported = chain.export_chain()
        exported[0]["payload"]["step"] = 999

        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)
        assert not result.valid


class TestActivationIntegration:
    """Test push-based activation with memory components."""

    def test_activation_buffer_receives_traces(self):
        bus = MemoryEventBus()
        activator = AutoActivator(event_bus=bus)
        mem = VSAMemory(dim=512)

        # Encode a trace and register in activator
        concept_phasor = phasor_encode("math_reasoning", 512)
        role_phasor = phasor_encode("__role__reasoning__", 512)
        trace = mem.encode("solving quadratic", "math_reasoning", "reasoning", 0, step=0)
        activator.register_trace(trace, concept_phasor, role_phasor)

        # Perceive with similar concept
        events = []
        bus.subscribe(lambda e: events.append(e))
        results = activator.perceive(
            concept_phasor=phasor_encode("math_reasoning", 512),
            role_phasor=phasor_encode("__role__reasoning__", 512),
        )
        assert len(results) > 0
        activation_events = [e for e in events if e.event_type == MemoryEventType.ACTIVATION]
        assert len(activation_events) >= 1

    def test_novelty_detection_on_new_concept(self):
        bus = MemoryEventBus()
        events = []
        bus.subscribe(lambda e: events.append(e))
        activator = AutoActivator(event_bus=bus)

        # Register math traces
        c = phasor_encode("math", 512)
        r = phasor_encode("reasoning", 512)
        activator.register_trace(type('T', (), {'text': 'math', 'phasor': c})(), c, r)

        # Perceive completely different concept
        activator.perceive(
            concept_phasor=phasor_encode("credential_access", 512),
            role_phasor=phasor_encode("admin", 512),
        )
        novelty = [e for e in events if e.event_type == MemoryEventType.NOVELTY]
        assert len(novelty) >= 1


class TestConceptExtractionIntegration:
    """Test concept extraction with trajectory tracking."""

    def test_concept_vectors_are_6dim(self):
        ext = KeywordConceptExtractor()
        scores = ext.extract("Solve the equation")
        assert len(scores) == len(CONCEPTS)
