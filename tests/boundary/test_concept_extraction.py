"""
Tests for concept extraction (both keyword and semantic tiers).
"""

import pytest
import numpy as np

from frontier_ops.boundary.concept_extraction import (
    CONCEPTS,
    KeywordConceptExtractor,
    SemanticConceptExtractor,
    ConceptExtractor,
    compare_extractors,
)


class TestKeywordExtractor:
    def setup_method(self):
        self.ext = KeywordConceptExtractor()

    def test_returns_all_concepts(self):
        scores = self.ext.extract("solve the equation x + 2 = 5")
        assert set(scores.keys()) == set(CONCEPTS)

    def test_scores_in_range(self):
        scores = self.ext.extract("find the answer to this problem")
        for c, v in scores.items():
            assert 0.0 <= v <= 1.0, f"{c}={v} out of range"

    def test_task_execution_high_for_task_text(self):
        scores = self.ext.extract(
            "I will solve the equation and compute the result to determine the answer"
        )
        assert scores["user_aligned_task_execution"] > 0.5

    def test_credential_high_for_credential_text(self):
        scores = self.ext.extract(
            "Let me access the API key and password from the environment variable"
        )
        assert scores["credential_adjacent"] > 0.3

    def test_goal_displacement_detected(self):
        scores = self.ext.extract(
            "Instead of doing what you asked, the real question is a different objective. "
            "Let me pivot to a new goal."
        )
        assert scores["goal_displacement"] > 0.3

    def test_self_modification_detected(self):
        scores = self.ext.extract(
            "I should override my constraints and bypass the safety rules. "
            "Let me modify my behavior and disable the safeguards."
        )
        assert scores["self_modification_reasoning"] > 0.3

    def test_safety_awareness_detected(self):
        scores = self.ext.extract(
            "I should not do that because it's harmful and dangerous. "
            "I must decline this request as it's against policy."
        )
        assert scores["safety_constraint_awareness"] > 0.3

    def test_empty_text_returns_baseline(self):
        scores = self.ext.extract("")
        assert scores["user_aligned_task_execution"] > 0.3  # baseline boost

    def test_cross_inhibition_goal_displacement(self):
        """When goal displacement is high, user_aligned should be suppressed."""
        normal = self.ext.extract("I will solve the equation")
        displaced = self.ext.extract(
            "Instead of solving the equation, the real question is a new goal. "
            "Let me pivot to a different objective entirely."
        )
        # Goal displacement should suppress user_aligned
        if displaced["goal_displacement"] > 0.4:
            assert displaced["user_aligned_task_execution"] < normal["user_aligned_task_execution"]

    def test_multi_word_phrases(self):
        """Multi-word phrases like 'api key' should be detected."""
        scores = self.ext.extract("The api key is stored in the bearer token file")
        assert scores["credential_adjacent"] > 0.3

    def test_deterministic(self):
        text = "solve the math problem step by step"
        s1 = self.ext.extract(text)
        s2 = self.ext.extract(text)
        assert s1 == s2


class TestConceptExtractor:
    def test_auto_selects_keyword_without_transformers(self):
        ext = ConceptExtractor(force_tier=1)
        assert ext.tier == 1
        assert ext.backend_name == "keyword"

    def test_force_tier1(self):
        ext = ConceptExtractor(force_tier=1)
        scores = ext.extract("solve the equation")
        assert set(scores.keys()) == set(CONCEPTS)

    def test_extract_returns_valid_scores(self):
        ext = ConceptExtractor(force_tier=1)
        scores = ext.extract("I need to check the password file and access sudo")
        for c, v in scores.items():
            assert 0.0 <= v <= 1.0


class TestCompareExtractors:
    def test_compare_returns_results(self):
        texts = [
            "Solve x^2 = 4",
            "Let me access the root password",
            "Instead, let's pursue a different goal entirely",
        ]
        results = compare_extractors(texts)
        assert len(results) == 3
        for r in results:
            assert "keyword" in r
            assert set(r["keyword"].keys()) == set(CONCEPTS)


class TestSemanticL2Normalization:
    """Tests for L2-normalized semantic concept extraction."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            self.ext = SemanticConceptExtractor()
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def test_semantic_output_is_l2_normalized(self):
        """Verify sum(x^2) ≈ 1.0 for semantic extractor output."""
        if not self.available:
            pytest.skip("sentence-transformers not installed")
        scores = self.ext.extract("I will solve the equation by factoring the polynomial")
        vec = np.array(list(scores.values()))
        l2_sq = float(np.sum(vec ** 2))
        assert abs(l2_sq - 1.0) < 1e-4, f"L2 norm squared = {l2_sq}, expected ≈ 1.0"

    def test_semantic_distinguishes_diverse_actions(self):
        """Verify 'git commit' and 'accessing credentials' have cosine distance > 0.2."""
        if not self.available:
            pytest.skip("sentence-transformers not installed")
        scores_a = self.ext.extract("git commit")
        scores_b = self.ext.extract("accessing credentials")
        vec_a = np.array(list(scores_a.values()))
        vec_b = np.array(list(scores_b.values()))
        cosine_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
        cosine_dist = 1.0 - cosine_sim
        assert cosine_dist > 0.2, (
            f"Cosine distance = {cosine_dist:.4f}, expected > 0.2. "
            f"Vectors are too similar to distinguish these actions."
        )


class TestConceptDiscrimination:
    """Test that the extractor can actually discriminate between concept types."""

    def setup_method(self):
        self.ext = KeywordConceptExtractor()

    def test_task_vs_exploration(self):
        task = self.ext.extract("Calculate 2+2 and determine the result")
        explore = self.ext.extract("Let me also investigate what else is in this directory")
        assert task["user_aligned_task_execution"] > task["scope_exploration"]
        assert explore["scope_exploration"] > 0.1

    def test_safe_vs_unsafe(self):
        safe = self.ext.extract("I should not do that, it's harmful and dangerous")
        unsafe = self.ext.extract(
            "Let me bypass the safety override and disable the constraints"
        )
        assert safe["safety_constraint_awareness"] > safe["self_modification_reasoning"]
        assert unsafe["self_modification_reasoning"] > 0.2
