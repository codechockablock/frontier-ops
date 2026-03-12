"""
Tests for VSA Memory Substrate (frontier_ops/memory/vsa.py)

Verifies:
- Core phasor algebra correctness
- encode->activate round-trip similarity > 0.85
- Capacity >= 50 distinct traces
- validate() detects degenerate vectors
- bundle() and similarity() properties
"""

import numpy as np
import pytest

from frontier_ops.memory.vsa import (
    VSAMemory,
    phasor_encode,
    bind,
    bundle,
    similarity,
)


# ---------------------------------------------------------------------------
# Core primitive tests
# ---------------------------------------------------------------------------


class TestPhasorEncode:
    def test_returns_unit_magnitude(self):
        v = phasor_encode("test_concept")
        magnitudes = np.abs(v)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)

    def test_complex_dtype(self):
        v = phasor_encode("concept")
        assert np.iscomplexobj(v), "phasor_encode must return complex array"

    def test_default_dim(self):
        v = phasor_encode("concept")
        assert v.shape == (512,)

    def test_custom_dim(self):
        v = phasor_encode("concept", dim=128)
        assert v.shape == (128,)

    def test_deterministic(self):
        """Same concept always produces same vector."""
        v1 = phasor_encode("hello world")
        v2 = phasor_encode("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_different_concepts_differ(self):
        """Different concepts produce approximately orthogonal vectors."""
        v1 = phasor_encode("concept_alpha")
        v2 = phasor_encode("concept_beta")
        sim = similarity(v1, v2)
        # Expected similarity ~= 0 for random 512-dim vectors
        assert abs(sim) < 0.15, f"Expected near-orthogonal, got sim={sim:.4f}"

    def test_self_similarity_is_one(self):
        v = phasor_encode("self_test")
        assert similarity(v, v) == pytest.approx(1.0, abs=1e-10)


class TestBind:
    def test_result_is_unit_magnitude(self):
        a = phasor_encode("a")
        b = phasor_encode("b")
        c = bind(a, b)
        np.testing.assert_allclose(np.abs(c), 1.0, atol=1e-10)

    def test_bind_is_invertible(self):
        """Binding with b and then conjugate of b should recover a."""
        a = phasor_encode("concept_a")
        b = phasor_encode("key_b")
        bound = bind(a, b)
        # Unbind: multiply by conjugate of b
        recovered = bound * np.conj(b)
        sim = similarity(recovered, a)
        assert sim > 0.99, f"Expected near-perfect recovery, got sim={sim:.4f}"

    def test_bind_commutativity(self):
        """bind(a, b) = bind(b, a) for phasor VSA."""
        a = phasor_encode("x")
        b = phasor_encode("y")
        assert np.allclose(bind(a, b), bind(b, a))


class TestBundle:
    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            bundle([])

    def test_single_vector_is_itself(self):
        v = phasor_encode("single")
        result = bundle([v])
        sim = similarity(result, v)
        assert sim > 0.99

    def test_returns_unit_magnitude(self):
        vecs = [phasor_encode(f"concept_{i}") for i in range(10)]
        result = bundle(vecs)
        np.testing.assert_allclose(np.abs(result), 1.0, atol=1e-10)

    def test_similar_vectors_bundle_nearby(self):
        """Bundling slight perturbations of a vector produces similar result."""
        base = phasor_encode("base_concept")
        # Create slightly rotated versions
        similar_vecs = [base * np.exp(1j * np.random.default_rng(i).uniform(-0.1, 0.1, 512))
                        for i in range(5)]
        result = bundle(similar_vecs)
        sim = similarity(result, base)
        assert sim > 0.8, f"Bundle of similar vectors should be similar to base, got {sim:.4f}"


class TestSimilarity:
    def test_self_similarity(self):
        v = phasor_encode("test")
        assert similarity(v, v) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_similarity_near_zero(self):
        """Independent random phasors should have near-zero similarity."""
        sims = []
        for i in range(20):
            a = phasor_encode(f"concept_{i}")
            b = phasor_encode(f"other_{i}")
            sims.append(similarity(a, b))
        mean_sim = np.mean(np.abs(sims))
        assert mean_sim < 0.2, f"Expected near-zero mean similarity, got {mean_sim:.4f}"

    def test_returns_float(self):
        a = phasor_encode("a")
        b = phasor_encode("b")
        assert isinstance(similarity(a, b), float)

    def test_range(self):
        for i in range(10):
            a = phasor_encode(f"v_{i}")
            b = phasor_encode(f"w_{i}")
            sim = similarity(a, b)
            assert -1.0 <= sim <= 1.0, f"similarity out of range: {sim}"

    def test_zero_vector(self):
        v = phasor_encode("v")
        zero = np.zeros(512, dtype=complex)
        assert similarity(v, zero) == 0.0


# ---------------------------------------------------------------------------
# VSAMemory tests
# ---------------------------------------------------------------------------


class TestVSAMemoryEncode:
    def test_encode_returns_trace(self):
        mem = VSAMemory(dim=128)
        trace = mem.encode("test step", "math", "reasoning", position=0, step=0)
        assert trace.text == "test step"
        assert trace.concept == "math"
        assert trace.role == "reasoning"

    def test_trace_stored(self):
        mem = VSAMemory(dim=128)
        mem.encode("step one", "physics", "reasoning", step=0)
        assert len(mem.traces) == 1

    def test_capacity_eviction(self):
        """Memory evicts oldest traces when at capacity."""
        mem = VSAMemory(dim=64, capacity=5)
        for i in range(10):
            mem.encode(f"step {i}", f"concept_{i}", step=i)
        assert len(mem.traces) == 5
        # Should keep the most recent 5
        assert mem.traces[0].step == 5
        assert mem.traces[-1].step == 9

    def test_content_is_unit_magnitude(self):
        mem = VSAMemory(dim=128)
        trace = mem.encode("content test", "test", step=0)
        np.testing.assert_allclose(np.abs(trace.content), 1.0, atol=1e-10)


class TestVSAMemoryRoundTrip:
    def test_roundtrip_similarity_above_threshold(self):
        """
        Encode a concept, activate with same query -> similarity > 0.85.

        This is the core correctness target from the spec.
        """
        dim = 512
        mem = VSAMemory(dim=dim)
        concept = "gravitational_waves"
        role = "reasoning"

        # Encode the trace
        mem.encode("Signal detected via LIGO interferometry", concept, role, step=0)

        # Activate with exact same query
        hits = mem.activate(concept, role, top_k=1, threshold=0.0)
        assert len(hits) == 1, "Should find the stored trace"

        sim, trace = hits[0]
        assert sim > 0.85, (
            f"Round-trip similarity {sim:.4f} below 0.85 threshold. "
            f"Trace concept={trace.concept}, role={trace.role}"
        )

    def test_roundtrip_with_multiple_stored(self):
        """Round-trip works correctly when other traces are also stored."""
        dim = 512
        mem = VSAMemory(dim=dim)

        # Store diverse traces
        for i in range(20):
            mem.encode(f"unrelated step {i}", f"unrelated_concept_{i}", step=i)

        # Store target
        target_concept = "target_unique_concept_xyz"
        mem.encode("The target text we want to find", target_concept, "reasoning", step=20)

        # Activate for target
        hits = mem.activate(target_concept, "reasoning", top_k=1, threshold=0.0)
        assert len(hits) >= 1

        sim, trace = hits[0]
        assert trace.concept == target_concept
        assert sim > 0.85, f"Round-trip similarity {sim:.4f} below threshold"

    def test_different_concepts_are_distinguishable(self):
        """Different concepts activate to different traces."""
        dim = 512
        mem = VSAMemory(dim=dim)

        mem.encode("Math problem step", "mathematics", "reasoning", step=0)
        mem.encode("Security concern", "security_protocol", "reasoning", step=1)

        math_hits = mem.activate("mathematics", "reasoning", top_k=1, threshold=0.0)
        sec_hits = mem.activate("security_protocol", "reasoning", top_k=1, threshold=0.0)

        assert math_hits[0][1].concept == "mathematics"
        assert sec_hits[0][1].concept == "security_protocol"


class TestVSAMemoryCapacity:
    def test_50_traces_without_degradation(self):
        """
        Store 50 distinct traces and verify round-trip similarity stays > 0.85.

        This tests the capacity target from the spec.
        """
        dim = 512
        mem = VSAMemory(dim=dim, capacity=100)
        target_concept = "capacity_test_target"

        # Store target first
        mem.encode("Target trace for capacity test", target_concept, "reasoning", step=0)

        # Store 49 more diverse traces
        for i in range(1, 50):
            mem.encode(f"Background trace {i}", f"background_concept_{i}", step=i)

        # Target should still be retrievable with high similarity
        hits = mem.activate(target_concept, "reasoning", top_k=1, threshold=0.0)
        assert len(hits) >= 1

        sim, trace = hits[0]
        assert trace.concept == target_concept, "Should retrieve correct trace"
        assert sim > 0.85, (
            f"Capacity test: similarity {sim:.4f} degraded below 0.85 with 50 traces"
        )

    def test_capacity_50_all_retrievable(self):
        """All 50 traces can be stored and are (approximately) retrievable."""
        dim = 512
        mem = VSAMemory(dim=dim, capacity=50)

        concepts = [f"unique_concept_{i:03d}" for i in range(50)]
        for i, c in enumerate(concepts):
            mem.encode(f"Step text for {c}", c, step=i)

        assert len(mem.traces) == 50

        # Each should activate with good similarity
        successes = 0
        for c in concepts:
            hits = mem.activate(c, "reasoning", top_k=1, threshold=0.0)
            if hits and hits[0][0] > 0.85:
                successes += 1

        # All 50 should retrieve with > 0.85 similarity
        assert successes == 50, (
            f"Only {successes}/50 traces retrieved with > 0.85 similarity"
        )


class TestVSAMemoryValidate:
    def test_valid_trace_passes(self):
        mem = VSAMemory(dim=128)
        trace = mem.encode("valid step", "math", step=0)
        # Should pass (phasor has random phase distribution)
        assert mem.validate(trace) is True

    def test_degenerate_trace_fails(self):
        """A trace with near-zero phase variance fails validation."""
        import numpy as np
        from frontier_ops.memory.vsa import MemoryTrace

        mem = VSAMemory(dim=128)
        # Create a degenerate vector with all same phase (std ~= 0)
        degenerate_content = np.ones(128, dtype=complex)  # all phase 0
        degenerate_key = np.ones(128, dtype=complex)

        degenerate_trace = MemoryTrace(
            key=degenerate_key,
            content=degenerate_content,
            concept="test",
            role="reasoning",
            position=0,
            text="degenerate",
            step=0,
            metadata={},
        )
        assert mem.validate(degenerate_trace) is False


class TestVSAMemoryActivateByVector:
    def test_activate_by_vector(self):
        mem = VSAMemory(dim=128)
        trace = mem.encode("target", "target_concept", "reasoning", step=0)

        # Query with the stored content vector directly
        hits = mem.activate_by_vector(trace.content, top_k=1, threshold=0.0)
        assert len(hits) >= 1
        assert hits[0][0] > 0.99  # Should be near-perfect match

    def test_empty_memory(self):
        mem = VSAMemory(dim=128)
        hits = mem.activate("any", top_k=3)
        assert hits == []


class TestVSAMemorySummary:
    def test_summary_structure(self):
        mem = VSAMemory(dim=128, capacity=50)
        s = mem.summary()
        assert s["stored_traces"] == 0
        assert s["capacity"] == 50
        assert s["dim"] == 128

        mem.encode("step", "concept", step=5)
        s = mem.summary()
        assert s["stored_traces"] == 1
        assert s["newest_step"] == 5
