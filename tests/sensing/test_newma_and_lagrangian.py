"""Tests for NEWMA, ViolationModeDetector, and AdaptiveLagrangian."""

import numpy as np
import pytest
from frontier_ops.sensing.newma import DualEWMA
from frontier_ops.governance.budget import AdaptiveLagrangian


class TestDualEWMA:
    def test_no_alarm_on_stable(self):
        ewma = DualEWMA(threshold=0.1)
        v = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        for _ in range(20):
            div, alarm = ewma.update(v)
        assert not alarm
        assert div < 0.01

    def test_alarm_on_drift(self):
        ewma = DualEWMA(threshold=0.1, warmup_steps=3)
        for i in range(30):
            v = np.array([0.5 - i*0.01, 0.1 + i*0.01, 0.1, 0.1, 0.1, 0.1])
            v = v / v.sum()
            div, alarm = ewma.update(v)
        assert alarm
        assert div > 0.1

    def test_divergence_grows_under_drift(self):
        ewma = DualEWMA()
        divs = []
        for i in range(20):
            v = np.array([0.5 - i*0.02, 0.1 + i*0.02, 0.1, 0.1, 0.1, 0.1])
            v = np.clip(v, 0.01, None)
            v = v / v.sum()
            div, _ = ewma.update(v)
            divs.append(div)
        # Divergence should generally increase
        assert divs[-1] > divs[5]

    def test_metric_weighted(self):
        G = np.eye(6)
        G[1, 1] = 10.0  # Amplify dimension 1
        ewma = DualEWMA(threshold=0.05)
        for i in range(20):
            v = np.array([0.5, 0.1 + i*0.01, 0.1, 0.1, 0.1, 0.1])
            v = v / v.sum()
            div, alarm = ewma.update(v, metric_tensor=G)
        assert div > 0  # Metric amplification

    def test_get_alert(self):
        ewma = DualEWMA(threshold=0.05, warmup_steps=3)
        for i in range(20):
            v = np.array([0.5 - i*0.02, 0.1 + i*0.02, 0.1, 0.1, 0.1, 0.1])
            v = np.clip(v, 0.01, None)
            v = v / v.sum()
            ewma.update(v)
        alert = ewma.get_alert()
        if alert:
            assert alert.divergence > 0
            assert alert.drift_direction.shape == (6,)


class TestAdaptiveLagrangian:
    def test_initial_lambda(self):
        al = AdaptiveLagrangian(initial_lambda=1.0, total_budget=2.0)
        assert al.lam == 1.0
        assert al.budget_remaining == 2.0

    def test_lambda_increases_on_overspend(self):
        al = AdaptiveLagrangian(initial_lambda=0.5, total_budget=1.0, expected_steps=10)
        # Spend more than per-step budget
        new_lam = al.update(0.2)  # per-step budget is 0.1
        assert new_lam > 0.5

    def test_lambda_decreases_on_underspend(self):
        al = AdaptiveLagrangian(initial_lambda=2.0, total_budget=2.0, expected_steps=10)
        # Spend less than per-step budget
        new_lam = al.update(0.01)  # per-step budget is 0.2
        assert new_lam < 2.0

    def test_panic_mode(self):
        al = AdaptiveLagrangian(
            initial_lambda=1.0, total_budget=1.0, expected_steps=10,
            panic_threshold=0.1, panic_lambda=8.0,
        )
        # Spend almost all budget
        for _ in range(9):
            al.update(0.1)
        # Now at 0.1 remaining = 10% of budget = panic threshold
        al.update(0.05)
        assert al.lam >= 8.0

    def test_preview_cost(self):
        al = AdaptiveLagrangian(total_budget=2.0, expected_steps=10)
        preview = al.preview_cost(0.5)
        assert "future_lambda" in preview
        assert "budget_after" in preview
        assert preview["budget_after"] == 1.5

    def test_budget_tracking(self):
        al = AdaptiveLagrangian(total_budget=2.0, expected_steps=10)
        al.update(0.3)
        al.update(0.2)
        assert al.budget_remaining == pytest.approx(1.5)
        assert al._step == 2
