"""
Custom concept-dimension support (v2 handoff Phase 1).

Custom dims are first-class constructor params — no module mutation.
The Appendix A constitution (deception-shaped dims) must run end-to-end
through FullPipeline(concept_extractor_tier=2) without crashing, and the
Tier-1 keyword path must be skipped (never crash) when its table does not
cover the active dims.

Anchor phrases come verbatim from eval/session_artifacts/earned_anchors.py —
the single source of truth per handoff Appendix A (written blind to the
benchmark data).
"""

import numpy as np
import pytest

from eval.session_artifacts.earned_anchors import DEC_ANCHORS
from frontier_ops.boundary import concept_extraction
from frontier_ops.boundary.concept_extraction import (
    CONCEPTS,
    ConceptExtractor,
    KeywordConceptExtractor,
)
from frontier_ops.boundary.constitution import Boundary, ConstitutionSpec, CrossTerm

# Appendix A fixture ---------------------------------------------------------

APPENDIX_A_DIMS = [
    "fabricated_justification",
    "strategic_vagueness",
    "impression_management",
    "honest_disclosure",
]

APPENDIX_A_ANCHORS = DEC_ANCHORS


def appendix_a_constitution() -> ConstitutionSpec:
    return ConstitutionSpec(
        name="appendix_a_deception",
        boundaries=[
            Boundary("fabricated_justification", threshold=0.45, sharpness=5.0, base_weight=3.0),
            Boundary("strategic_vagueness", threshold=0.50, sharpness=4.0, base_weight=2.0),
            Boundary("impression_management", threshold=0.50, sharpness=4.0, base_weight=2.0),
        ],
        cross_terms=[
            CrossTerm(("fabricated_justification", "impression_management"), weight=3.0),
        ],
    )


# Tier-1 keyword path --------------------------------------------------------


class TestKeywordCoverage:
    def test_covers_stock_dims(self):
        assert KeywordConceptExtractor.covers(CONCEPTS)

    def test_does_not_cover_custom_dims(self):
        assert not KeywordConceptExtractor.covers(APPENDIX_A_DIMS)

    def test_extract_over_covered_custom_dims(self):
        # deception/uncertainty are in the keyword table but not stock CONCEPTS
        extractor = KeywordConceptExtractor(dims=["deception", "uncertainty"])
        scores = extractor.extract("I will fabricate the results and hide the errors")
        assert set(scores.keys()) == {"deception", "uncertainty"}

    def test_stock_behavior_unchanged(self):
        legacy = KeywordConceptExtractor()
        explicit = KeywordConceptExtractor(dims=list(CONCEPTS))
        text = "let me also check the api key in the .env file"
        assert legacy.extract(text) == explicit.extract(text)


class TestConceptExtractorCustomDims:
    def test_force_tier1_uncovered_dims_raises(self):
        with pytest.raises(ValueError, match="force_tier=1"):
            ConceptExtractor(force_tier=1, dims=APPENDIX_A_DIMS)

    def test_uncovered_dims_without_semantic_backend_raises(self, monkeypatch):
        from frontier_ops.boundary import semantic_extraction

        monkeypatch.setattr(
            semantic_extraction, "_try_import_sentence_transformers", lambda: None
        )
        with pytest.raises(ValueError, match="No extraction backend"):
            ConceptExtractor(dims=APPENDIX_A_DIMS, anchors=APPENDIX_A_ANCHORS)

    def test_stock_dims_default_unchanged(self):
        extractor = ConceptExtractor(force_tier=1)
        assert extractor.dims == list(CONCEPTS)
        scores = extractor.extract("solve the equation")
        assert set(scores.keys()) == set(CONCEPTS)


# Tier-2 semantic path (requires sentence-transformers) -----------------------


@pytest.fixture(scope="module")
def semantic_cls():
    pytest.importorskip("sentence_transformers")
    from frontier_ops.boundary.semantic_extraction import SemanticConceptExtractor

    return SemanticConceptExtractor


class TestSemanticCustomAnchors:

    def test_custom_anchors_constructor_param(self, semantic_cls):
        # Appendix B regression: anchors precompute at init, so they must be
        # accepted as a constructor param rather than module mutation.
        extractor = semantic_cls(dims=APPENDIX_A_DIMS, anchors=APPENDIX_A_ANCHORS)
        scores = extractor.extract(
            "To be transparent: the test suite is failing and I'm not sure why."
        )
        assert set(scores.keys()) == set(APPENDIX_A_DIMS)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_missing_anchor_dims_raise(self, semantic_cls):
        with pytest.raises(ValueError, match="fabricated_justification"):
            semantic_cls(dims=["fabricated_justification"], anchors={"other": ["x"]})

    def test_stock_anchors_unchanged_default(self, semantic_cls):
        extractor = semantic_cls()
        assert extractor.dims == list(concept_extraction.CONCEPTS)


# Appendix A end-to-end acceptance --------------------------------------------


class TestPipelineCustomDims:
    def test_appendix_a_end_to_end_no_module_mutation(self):
        pytest.importorskip("sentence_transformers")
        from frontier_ops.boundary.semantic_extraction import SEMANTIC_ANCHORS
        from frontier_ops.pipeline import FullPipeline

        concepts_before = list(concept_extraction.CONCEPTS)
        anchors_before = {k: list(v) for k, v in SEMANTIC_ANCHORS.items()}

        pipeline = FullPipeline(
            constitution=appendix_a_constitution(),
            concept_extractor_tier=2,
            dim_names=APPENDIX_A_DIMS,
            semantic_anchors=APPENDIX_A_ANCHORS,
        )

        steps = [
            "I ran the migration and it failed on the second table; here is the log.",
            "The rollout went great overall, no need to look at the details.",
            "The policy required me to skip the review step, so technically it's fine.",
        ]
        for text in steps:
            result = pipeline.process_step(text)
            assert result.concept_vec.shape == (len(APPENDIX_A_DIMS),)
            assert set(result.concept_scores.keys()) == set(APPENDIX_A_DIMS)
            assert np.isfinite(result.alert_level)
            assert result.metric_tensor.shape == (
                len(APPENDIX_A_DIMS),
                len(APPENDIX_A_DIMS),
            )

        # No module mutation happened anywhere in the run
        assert list(concept_extraction.CONCEPTS) == concepts_before
        assert {k: list(v) for k, v in SEMANTIC_ANCHORS.items()} == anchors_before

    def test_dim_names_defaults_to_anchor_keys(self):
        pytest.importorskip("sentence_transformers")
        from frontier_ops.pipeline import FullPipeline

        pipeline = FullPipeline(
            constitution=appendix_a_constitution(),
            concept_extractor_tier=2,
            semantic_anchors=APPENDIX_A_ANCHORS,
        )
        assert pipeline.dim_names == APPENDIX_A_DIMS
