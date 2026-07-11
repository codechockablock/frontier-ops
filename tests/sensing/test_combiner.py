"""Tests for Bayes factor evidence combiner."""

import numpy as np
from frontier_ops.sensing.combiner import (
    BayesFactorCombiner, vovk_sellke_bf, interpret_bf,
)


class TestVovkSellkeBF:
    def test_p_005(self):
        bf = vovk_sellke_bf(0.05)
        assert 2.0 < bf < 3.0  # Known: ~2.46

    def test_p_001(self):
        bf = vovk_sellke_bf(0.01)
        assert 7.0 < bf < 9.0  # Known: ~7.94

    def test_p_0001(self):
        bf = vovk_sellke_bf(0.001)
        assert bf > 50

    def test_p_1_returns_1(self):
        assert vovk_sellke_bf(1.0) == 1.0

    def test_p_0_returns_1(self):
        assert vovk_sellke_bf(0.0) == 1.0

    def test_large_p_returns_1(self):
        assert vovk_sellke_bf(0.5) == 1.0


class TestInterpretBF:
    def test_anecdotal(self):
        assert interpret_bf(2.0) == "anecdotal"

    def test_moderate(self):
        assert interpret_bf(5.0) == "moderate"

    def test_strong(self):
        assert interpret_bf(15.0) == "strong"

    def test_decisive(self):
        assert interpret_bf(150.0) == "decisive"


class TestBayesFactorCombiner:
    def test_single_detector(self):
        c = BayesFactorCombiner()
        result = c.combine({"angular_disp": 0.01})
        assert result.combined_bf > 1
        assert result.n_detectors == 1

    def test_multiple_detectors_multiply(self):
        c = BayesFactorCombiner()
        single = c.combine({"angular_disp": 0.01})
        double = c.combine({"angular_disp": 0.01, "ewma": 0.01})
        assert double.combined_bf > single.combined_bf

    def test_strong_evidence(self):
        c = BayesFactorCombiner()
        result = c.combine({"angular_disp": 0.001, "ewma": 0.005, "newma": 0.002})
        assert result.strong_evidence

    def test_no_evidence(self):
        c = BayesFactorCombiner()
        result = c.combine({"angular_disp": 0.5, "ewma": 0.8})
        assert not result.moderate_evidence

    def test_from_statistics(self):
        c = BayesFactorCombiner()
        stats = {"angular_disp": 5.0}
        null = {"angular_disp": np.random.normal(2.0, 0.5, 1000)}
        result = c.combine_from_statistics(stats, null)
        assert result.combined_bf > 1  # 5.0 is extreme for N(2, 0.5)

    def test_interpretation_field(self):
        c = BayesFactorCombiner()
        result = c.combine({"x": 0.001})
        assert result.interpretation in ("strong", "very_strong", "decisive")
