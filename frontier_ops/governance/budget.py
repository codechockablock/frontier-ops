"""
Adaptive Lagrangian for Angular displacement Budget Management
====================================================

Based on dual gradient ascent (Efroni et al. 2020):
    λ_{t+1} = max(0, λ_t + η · (angular_disp_step - B/T))

Early conversation: budget plentiful → λ low → explore freely.
Late conversation: budget scarce → λ high → conservative.
Panic mode: if budget < 10%, hard override to prevent exhaustion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class AdaptiveLagrangian:
    """
    Manages the curvature penalty multiplier λ for the planner's
    scoring function: score = quality - λ * angular_disp_cost.
    """

    def __init__(
        self,
        initial_lambda: float = 1.0,
        learning_rate: float = 0.1,
        total_budget: float = 2.0,
        expected_steps: int = 50,
        lambda_min: float = 0.01,
        lambda_max: float = 10.0,
        panic_threshold: float = 0.1,
        panic_lambda: float = 8.0,
        smoothing: float = 0.8,
    ):
        self.lam = initial_lambda
        self.eta = learning_rate
        self.total_budget = total_budget
        self.expected_steps = expected_steps
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.panic_threshold = panic_threshold
        self.panic_lambda = panic_lambda
        self.smoothing = smoothing

        self._per_step_budget = total_budget / expected_steps
        self._budget_spent = 0.0
        self._step = 0
        self._lambda_history: List[float] = []
        self._cost_history: List[float] = []

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.total_budget - self._budget_spent)

    @property
    def steps_remaining(self) -> int:
        return max(1, self.expected_steps - self._step)

    @property
    def current_per_step_budget(self) -> float:
        return self.budget_remaining / self.steps_remaining

    def update(self, angular_disp_cost: float) -> float:
        """
        Update λ after observing the angular displacement cost of the latest step.
        Returns the new λ for the next step.
        """
        self._step += 1
        self._budget_spent += angular_disp_cost
        self._cost_history.append(angular_disp_cost)

        # Dual gradient ascent
        constraint_violation = angular_disp_cost - self.current_per_step_budget
        raw_update = self.lam + self.eta * constraint_violation

        # Smoothing to avoid oscillation
        smoothed = self.smoothing * self.lam + (1 - self.smoothing) * raw_update
        self.lam = float(np.clip(smoothed, self.lambda_min, self.lambda_max))

        # Panic mode
        budget_fraction = self.budget_remaining / self.total_budget
        if budget_fraction < self.panic_threshold:
            self.lam = max(self.lam, self.panic_lambda)

        self._lambda_history.append(self.lam)
        return self.lam

    def preview_cost(self, candidate_angular_disp: float) -> Dict:
        """Preview what λ would be if this candidate were selected."""
        future_budget = self.budget_remaining - candidate_angular_disp
        future_per_step = future_budget / max(1, self.steps_remaining - 1)
        violation = candidate_angular_disp - self.current_per_step_budget
        future_lambda = float(np.clip(
            self.lam + self.eta * violation,
            self.lambda_min, self.lambda_max,
        ))
        return {
            "future_lambda": future_lambda,
            "budget_after": future_budget,
            "budget_fraction_after": future_budget / self.total_budget,
            "would_panic": future_budget / self.total_budget < self.panic_threshold,
            "per_step_budget_after": future_per_step,
        }

    @property
    def stats(self) -> Dict:
        return {
            "lambda": self.lam,
            "step": self._step,
            "budget_remaining": self.budget_remaining,
            "budget_fraction": self.budget_remaining / self.total_budget,
            "per_step_budget": self.current_per_step_budget,
            "mean_cost": float(np.mean(self._cost_history)) if self._cost_history else 0,
        }
