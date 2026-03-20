"""
Authorization-Linked Budget
===========================

Budget replenishment is earned through human authorization, not timers.

The key insight: budget replenishes ONLY when a user issues a new directive
(EXPAND or REVISE). This links the provenance graph to budget management —
the same authorization event that creates a provenance edge also triggers
budget replenishment.

This replaces the automatic time-based replenishment in the cross-session
ledger with a principled mechanism: spending is free, replenishing requires
explicit human engagement.

Wraps AdaptiveLagrangian to add authorization-linked replenishment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from frontier_ops.governance.budget import AdaptiveLagrangian
from frontier_ops.authorization.provenance import ProvenanceGraph, EdgeType


@dataclass
class ReplenishmentEvent:
    """Records a budget replenishment triggered by a user directive."""
    timestamp: float
    directive_node_id: str
    operator: str  # expand, revise, establish
    amount: float
    budget_before: float
    budget_after: float


class AuthorizationLinkedBudget:
    """
    Budget that replenishes on authorization events, not timers.

    Wraps AdaptiveLagrangian for per-step lambda management and adds
    a provenance-linked replenishment layer on top.

    Replenishment rules:
    - ESTABLISH: Full budget grant (first directive)
    - EXPAND: Partial replenishment proportional to scope increase
    - REVISE: Full replenishment (new task = new budget)
    - CONTRACT: No replenishment (tighter scope, same budget)

    The replenishment amount is bounded: you can never exceed the
    total budget, and repeated expansions give diminishing returns
    to prevent gaming via trivial "also do X" messages.
    """

    def __init__(
        self,
        total_budget: float = 2.0,
        expected_steps: int = 50,
        establish_fraction: float = 1.0,   # Full budget on first directive
        expand_fraction: float = 0.3,      # 30% replenish on expand
        revise_fraction: float = 0.8,      # 80% replenish on revise
        expand_decay: float = 0.7,         # Each successive expand gives 70% of previous
        provenance: Optional[ProvenanceGraph] = None,
    ):
        self.total_budget = total_budget
        self.establish_fraction = establish_fraction
        self.expand_fraction = expand_fraction
        self.revise_fraction = revise_fraction
        self.expand_decay = expand_decay

        self._lagrangian = AdaptiveLagrangian(
            total_budget=total_budget,
            expected_steps=expected_steps,
        )
        self._provenance = provenance or ProvenanceGraph()

        self._replenishment_history: List[ReplenishmentEvent] = []
        self._consecutive_expands = 0
        self._total_replenished = 0.0

    @property
    def lagrangian(self) -> AdaptiveLagrangian:
        """Access to the underlying Lagrangian for lambda queries."""
        return self._lagrangian

    @property
    def provenance(self) -> ProvenanceGraph:
        return self._provenance

    @property
    def budget_remaining(self) -> float:
        return self._lagrangian.budget_remaining

    @property
    def budget_fraction(self) -> float:
        return self.budget_remaining / self.total_budget

    @property
    def is_exhausted(self) -> bool:
        return self._lagrangian.budget_remaining <= 0

    def update_step(self, angular_disp_cost: float) -> float:
        """
        Process one agent step's angular displacement cost.
        Returns the updated lambda value.
        """
        return self._lagrangian.update(angular_disp_cost)

    def on_authorization_event(
        self,
        operator: str,
        directive_node_id: str,
        goal_confidence: float = 0.5,
    ) -> ReplenishmentEvent:
        """
        Handle an authorization event from the scope system.

        Called when AuthorizationState.process_user_message() fires.
        Determines replenishment amount based on operator type.

        Args:
            operator: One of "establish", "expand", "revise", "contract"
            directive_node_id: Provenance node ID for this directive
            goal_confidence: Confidence of the extracted goal (scales replenishment)

        Returns:
            ReplenishmentEvent describing what happened.
        """
        budget_before = self._lagrangian.budget_remaining

        if operator == "establish":
            amount = self._compute_establish_amount(goal_confidence)
            self._consecutive_expands = 0
        elif operator == "expand":
            amount = self._compute_expand_amount(goal_confidence)
            self._consecutive_expands += 1
        elif operator == "revise":
            amount = self._compute_revise_amount(goal_confidence)
            self._consecutive_expands = 0
        else:
            # CONTRACT or unknown — no replenishment
            amount = 0.0

        # Apply replenishment
        if amount > 0:
            self._apply_replenishment(amount)
            self._total_replenished += amount

        event = ReplenishmentEvent(
            timestamp=time.time(),
            directive_node_id=directive_node_id,
            operator=operator,
            amount=amount,
            budget_before=budget_before,
            budget_after=self._lagrangian.budget_remaining,
        )
        self._replenishment_history.append(event)
        return event

    def _compute_establish_amount(self, confidence: float) -> float:
        """First directive: grant full budget scaled by confidence."""
        # Minimum 50% even for low-confidence directives
        scale = max(0.5, confidence)
        return self.total_budget * self.establish_fraction * scale

    def _compute_expand_amount(self, confidence: float) -> float:
        """
        Expansion: partial replenishment with diminishing returns.

        Each consecutive EXPAND gives expand_decay^n of the base amount.
        This prevents gaming via trivial expansions.
        """
        decay = self.expand_decay ** self._consecutive_expands
        base = self.total_budget * self.expand_fraction
        return base * decay * max(0.3, confidence)

    def _compute_revise_amount(self, confidence: float) -> float:
        """Revision: substantial replenishment (new task)."""
        scale = max(0.5, confidence)
        return self.total_budget * self.revise_fraction * scale

    def _apply_replenishment(self, amount: float):
        """
        Add budget back to the Lagrangian.

        Never exceed total_budget. The Lagrangian tracks budget as
        total_budget - budget_spent, so we reduce budget_spent.
        """
        current = self._lagrangian.budget_remaining
        max_replenish = self.total_budget - current
        actual = min(amount, max_replenish)

        if actual > 0:
            self._lagrangian._budget_spent -= actual

    @property
    def replenishment_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": e.timestamp,
                "directive": e.directive_node_id,
                "operator": e.operator,
                "amount": e.amount,
                "budget_before": e.budget_before,
                "budget_after": e.budget_after,
            }
            for e in self._replenishment_history
        ]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "budget_remaining": self.budget_remaining,
            "budget_fraction": self.budget_fraction,
            "total_replenished": self._total_replenished,
            "n_replenishments": len(self._replenishment_history),
            "consecutive_expands": self._consecutive_expands,
            "lambda": self._lagrangian.lam,
            "lagrangian_stats": self._lagrangian.stats,
        }
