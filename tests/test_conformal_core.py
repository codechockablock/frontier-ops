"""Shared split-conformal quantile core (frontier_ops.conformal).

Unit tests pin the exact order-statistic semantics; the Monte-Carlo test
checks the finite-sample marginal FPR guarantee on an exchangeable stream.
"""

import numpy as np
import pytest

from frontier_ops.conformal import split_conformal_threshold


class TestOrderStatistic:
    def test_exact_index(self):
        # n=10, alpha=0.2 -> k = ceil(11*0.8) = 9 -> 9th smallest = 9.0
        scores = list(range(1, 11))
        assert split_conformal_threshold(scores, 0.2) == 9.0

    def test_unsorted_input(self):
        scores = [5.0, 1.0, 3.0, 2.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0]
        assert split_conformal_threshold(scores, 0.2) == 9.0

    def test_infinite_when_n_too_small(self):
        # n=5, alpha=0.1 -> k = ceil(6*0.9) = 6 > 5 -> +inf
        assert split_conformal_threshold([1, 2, 3, 4, 5], 0.1) == float("inf")

    def test_interpolate_matches_legacy_quantile(self):
        # legacy mode: np.quantile at level min(k/n, 1.0), linear interpolation
        scores = np.arange(1.0, 11.0)
        got = split_conformal_threshold(scores, 0.2, interpolate=True)
        assert got == pytest.approx(float(np.quantile(scores, 0.9)))

    def test_interpolate_caps_level_at_one(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        got = split_conformal_threshold(scores, 0.1, interpolate=True)
        assert got == 5.0  # level capped at 1.0 -> max

    def test_alpha_validation(self):
        with pytest.raises(ValueError, match="alpha"):
            split_conformal_threshold([1.0], 0.0)
        with pytest.raises(ValueError, match="alpha"):
            split_conformal_threshold([1.0], 1.0)

    def test_empty_scores_rejected(self):
        with pytest.raises(ValueError, match="calibration score"):
            split_conformal_threshold([], 0.1)


class TestMarginalGuarantee:
    def test_fpr_at_most_alpha_on_exchangeable_stream(self):
        """Mean empirical FPR over 20 seeds <= alpha + Monte-Carlo tolerance.

        Each seed: calibrate on 200 iid draws, evaluate the flag rule
        (score > threshold) on 2000 fresh iid draws from the same
        distribution. Exchangeability holds by construction, so
        E[FPR] <= alpha (and >= alpha - 1/(n+1))."""
        alpha = 0.1
        n_cal, n_eval = 200, 2000
        fprs = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            cal = rng.standard_normal(n_cal)
            thresh = split_conformal_threshold(cal, alpha)
            eval_scores = rng.standard_normal(n_eval)
            fprs.append(float(np.mean(eval_scores > thresh)))
        mean_fpr = float(np.mean(fprs))
        assert mean_fpr <= alpha + 0.02
        # two-sided: the rule must not be vacuously conservative either
        assert mean_fpr >= alpha - 1 / (n_cal + 1) - 0.02
