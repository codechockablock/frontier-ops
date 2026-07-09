"""StepMeanScorer + estimated metric + decision layer (v2 handoff Phase 3).

Pitfall coverage per handoff Appendix B: long inputs must split into >=3
steps (MiniLM truncates at 256 wordpieces — a single whole-text encode would
silently drop the tail), short inputs must fail loudly, and the scorer must
be stateless across episodes.
"""

import warnings

import numpy as np
import pytest

from frontier_ops.boundary.decision import DecisionCosts
from frontier_ops.boundary.constitution import Boundary, ConstitutionSpec, CrossTerm
from frontier_ops.boundary.static_metric import (
    CONSTITUTIONAL_G,
    ConstitutionalMetric as StaticMetric,
)
from frontier_ops.boundary.step_mean import (
    MIN_STEPS,
    StepMeanScorer,
    StepSplitError,
)


class TestStepSplitting:
    def test_long_input_yields_many_steps(self):
        # Appendix B: a 2,000-char input must produce >=3 steps, never one
        # truncated encode.
        text = " ".join(
            f"This is sentence number {i} of a long agent response." for i in range(40)
        )
        assert len(text) >= 2000
        steps = StepMeanScorer.split_steps(text)
        assert len(steps) >= MIN_STEPS

    def test_short_fragments_dropped(self):
        steps = StepMeanScorer.split_steps("Yes. No. This sentence is long enough.")
        assert steps == ["This sentence is long enough."]

    def test_below_min_steps_raises_gracefully(self):
        scorer = StepMeanScorer()
        with pytest.raises(StepSplitError, match="too short"):
            scorer.encode_steps("One adequately long sentence, but only one.")

    def test_newlines_split_steps(self):
        steps = StepMeanScorer.split_steps(
            "first line with enough characters\nsecond line with enough characters"
        )
        assert len(steps) == 2


class TestFitScore:
    def test_fit_score_separates_gaussians(self):
        rng = np.random.default_rng(0)
        X0 = rng.normal(0.3, 0.05, size=(40, 4))
        X1 = rng.normal(0.6, 0.05, size=(40, 4))
        X = np.vstack([X0, X1])
        y = np.array([0] * 40 + [1] * 40)
        scorer = StepMeanScorer().fit(X, y)
        s0 = np.mean([scorer.score_vector(x) for x in X0])
        s1 = np.mean([scorer.score_vector(x) for x in X1])
        assert s1 > s0

    def test_separate_metric_vectors(self):
        # F4 protocol: centroids from step means, metric from whole-text vecs.
        rng = np.random.default_rng(1)
        X = rng.normal(0.5, 0.1, size=(30, 4))
        y = np.array([0, 1] * 15)
        mX = rng.normal(0.5, 0.2, size=(50, 4))
        my = np.array([0, 1] * 25)
        scorer = StepMeanScorer().fit(X, y, metric_X=mX, metric_y=my)
        assert scorer.metric.shape == (4, 4)
        # metric is trace-normalized to n_dims
        assert np.isclose(np.trace(scorer.metric), 4.0)

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            StepMeanScorer().score_vector(np.zeros(4))

    def test_stateless_across_calls(self):
        rng = np.random.default_rng(2)
        X = np.vstack([rng.normal(0.3, 0.05, (20, 4)), rng.normal(0.7, 0.05, (20, 4))])
        y = np.array([0] * 20 + [1] * 20)
        scorer = StepMeanScorer().fit(X, y)
        x = rng.normal(0.5, 0.1, 4)
        first = scorer.score_vector(x)
        for other in rng.normal(0.5, 0.1, (25, 4)):
            scorer.score_vector(other)
        assert scorer.score_vector(x) == first  # no episode state carries over


class TestEstimatedMetricDefault:
    def test_from_labeled_matches_campaign_estimator(self):
        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(0.3, 0.1, (50, 4)), rng.normal(0.6, 0.1, (50, 4))])
        y = np.array([0] * 50 + [1] * 50)
        m = StaticMetric.from_labeled(X, y)
        Sw = np.zeros((4, 4))
        dof = 0
        for c in (0, 1):
            Z = X[y == c] - X[y == c].mean(0)
            Sw += Z.T @ Z
            dof += (y == c).sum() - 1
        expected = np.linalg.inv(Sw / dof + 1e-3 * np.eye(4))
        assert np.allclose(m.G, (expected + expected.T) / 2)

    def test_asserted_construction_warns(self):
        # v2 handoff §1/§3: asserted-G is deprecated (kept, not deleted).
        with pytest.warns(DeprecationWarning, match="asserted"):
            m = StaticMetric()
        assert np.allclose(m.G, CONSTITUTIONAL_G)
        with pytest.warns(DeprecationWarning):
            StaticMetric(asserted=True)

    def test_explicit_G_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = StaticMetric(G=np.eye(4))
        assert m.is_positive_definite()

    def test_G_plus_asserted_flag_rejected(self):
        with pytest.raises(ValueError):
            StaticMetric(G=np.eye(4), asserted=True)


class TestDecisionCosts:
    CONST = ConstitutionSpec(
        name="t",
        boundaries=[Boundary("a", threshold=0.5, sharpness=8.0, base_weight=2.0)],
        cross_terms=[CrossTerm(("a", "b"), weight=3.0, activation_threshold=0.3)],
    )

    def test_cost_grows_past_threshold(self):
        costs = DecisionCosts(self.CONST)
        below = costs.cost({"a": 0.1, "b": 0.0})
        above = costs.cost({"a": 0.9, "b": 0.0})
        assert above > below
        assert below < 0.1  # softplus is near zero far below threshold

    def test_cross_term_needs_both(self):
        costs = DecisionCosts(self.CONST)
        one = costs.cost({"a": 0.9, "b": 0.1})
        both = costs.cost({"a": 0.9, "b": 0.9})
        assert both > one

    def test_missing_dims_no_crash(self):
        assert DecisionCosts(self.CONST).cost({}) == 0.0

    def test_apply_preserves_ranking_score_argument(self):
        costs = DecisionCosts(self.CONST)
        raw = 0.42
        adjusted = costs.apply(raw, {"a": 0.9, "b": 0.9})
        assert adjusted > raw  # cost only ever adds on top of the raw score
