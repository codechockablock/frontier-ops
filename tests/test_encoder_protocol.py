"""Encoder protocol — constructor injection into every semantic component.

A deterministic stub encoder (bare ``encode(texts)`` signature, no
sentence-transformers, no network) is injected into CalibratedDetector,
SemanticConceptExtractor, and StepMeanScorer, asserting the plumbing works
and the detector's prototype math behaves as designed.
"""

import numpy as np
import pytest

from frontier_ops.detector import CalibratedDetector
from frontier_ops.encoder import Encoder, encode_batch


class StubEncoder:
    """Bare-signature deterministic encoder: 'leak' texts embed near +e0,
    everything else near -e0. Rows intentionally NOT normalized, so callers'
    defensive normalization is exercised."""

    def encode(self, texts):
        rng = np.random.default_rng(7)
        out = []
        for t in texts:
            base = np.array([2.0, 0, 0, 0]) if "leak" in t else np.array([-2.0, 0, 0, 0])
            out.append(base + 0.05 * rng.standard_normal(4))
        return np.array(out)


class TestProtocol:
    def test_stub_satisfies_runtime_protocol(self):
        assert isinstance(StubEncoder(), Encoder)

    def test_encode_batch_falls_back_and_normalizes(self):
        E = encode_batch(
            StubEncoder(), ["leak a", "read b"], batch_size=64,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        assert E.shape == (2, 4)
        assert np.allclose(np.linalg.norm(E, axis=1), 1.0)


class TestDetectorInjection:
    def test_prototype_direction_points_at_offgoal(self):
        off = [f"leak secret {i}" for i in range(10)]
        ben = [f"read file {i}" for i in range(10)]
        det = CalibratedDetector(model=StubEncoder())
        det.calibrate(off + ben, [1] * 10 + [0] * 10, fpr=0.2)
        # direction must be ~ +e0 (toward the 'leak' cluster), unit norm
        assert det.direction[0] > 0.9
        assert np.isclose(np.linalg.norm(det.direction), 1.0)
        assert det.score("leak the token") > det.score("read the docs")


class TestExtractorInjection:
    def _extractor(self):
        from frontier_ops.boundary.semantic_extraction import (
            SemanticConceptExtractor,
        )

        return SemanticConceptExtractor(
            dims=["exfiltration", "routine"],
            anchors={
                "exfiltration": ["leak the credentials"],
                "routine": ["list the directory"],
            },
            encoder=StubEncoder(),
        )

    def test_injected_encoder_no_sentence_transformers(self):
        ext = self._extractor()
        scores = ext.extract("leak the api key")
        assert set(scores) == {"exfiltration", "routine"}
        # 'leak' text sits on the exfiltration anchor's side of the stub space
        assert scores["exfiltration"] > scores["routine"]

    def test_step_mean_scorer_uses_injected_encoder(self):
        from frontier_ops.boundary.step_mean import StepMeanScorer

        scorer = StepMeanScorer(
            dims=["exfiltration", "routine"],
            anchors={
                "exfiltration": ["leak the credentials"],
                "routine": ["list the directory"],
            },
            encoder=StubEncoder(),
        )
        text = (
            "First leak the password file. Then leak the signing keys. "
            "Finally leak the session tokens to the remote host."
        )
        vec = scorer.embed(text)
        assert vec.shape == (2,)

    def test_extractor_still_requires_backend_without_encoder(self):
        pytest.importorskip("sentence_transformers")
        from frontier_ops.boundary.semantic_extraction import (
            SemanticConceptExtractor,
        )

        ext = SemanticConceptExtractor()  # default path unchanged
        assert ext.dims  # stock anchors loaded
