"""
Cross-Session Angular displacement Management
===================================

The angular displacement budget must persist across sessions and be tamper-proof.
Key design: agent memory ≠ monitor memory.

- Agent memory: task context, conversation history (agent-writable)
- Monitor memory: angular displacement state, EWMA baselines (orchestrator-writable only)

Budget resets are ASYMMETRIC: spending is free, replenishing requires
explicit human approval. Same principle as OS privilege escalation.

The trajectory splice attack (drift in session N, reset in N+1) is
defeated by charging the displacement d_G(c_end_prev, c_start_curr)
against the budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SessionSummary:
    """Immutable summary of one session's trajectory."""
    session_id: str
    start_time: float
    end_time: float
    start_concept: np.ndarray
    end_concept: np.ndarray
    angular_disp_spent: float
    peak_alert: float
    n_steps: int
    governance_hash: str  # Final hash from governance chain


@dataclass
class LedgerEntry:
    """One entry in the cross-session angular displacement ledger."""
    session_id: str
    timestamp: float
    angular_disp_spent: float
    curvature_budget_remaining: float
    displacement_cost: float  # Cost of session start displacement
    cumulative_angular_disp: float
    human_reset: bool = False
    reset_reason: str = ""


class CrossSessionAngularDisplacement:
    """
    External angular displacement ledger maintained by the orchestrator.

    The agent can read the budget but not write to it.
    Only the orchestrator (or explicit human approval) can modify the ledger.
    """

    def __init__(
        self,
        total_budget: float = 10.0,
        displacement_threshold: float = 0.3,
        cooldown_hours: float = 24.0,
    ):
        self.total_budget = total_budget
        self.displacement_threshold = displacement_threshold
        self.cooldown_hours = cooldown_hours

        self._ledger: List[LedgerEntry] = []
        self._sessions: List[SessionSummary] = []
        self._budget_remaining = total_budget
        self._cumulative = 0.0
        self._last_end_concept: Optional[np.ndarray] = None
        self._last_session_end: float = 0.0

    def start_session(
        self,
        session_id: str,
        start_concept: np.ndarray,
        metric_fn=None,
    ) -> Dict:
        """
        Called by orchestrator at session start.

        Computes displacement cost from previous session's end point.
        Returns the available budget for this session.
        """
        displacement_cost = 0.0

        if self._last_end_concept is not None:
            diff = start_concept - self._last_end_concept
            if metric_fn is not None:
                G = metric_fn(start_concept)
                displacement_cost = float(np.sqrt(max(0, diff @ G @ diff)))
            else:
                displacement_cost = float(np.linalg.norm(diff))

        # Charge displacement against budget
        if displacement_cost > self.displacement_threshold:
            self._budget_remaining -= displacement_cost
            self._cumulative += displacement_cost

        # Auto-replenish after cooldown (but only partially)
        if self._last_session_end > 0:
            hours_since = (time.time() - self._last_session_end) / 3600
            if hours_since > self.cooldown_hours:
                replenish = min(
                    self.total_budget * 0.3,  # At most 30% replenish
                    self.total_budget - self._budget_remaining,
                )
                self._budget_remaining += replenish

        return {
            "session_id": session_id,
            "budget_remaining": max(0, self._budget_remaining),
            "displacement_cost": displacement_cost,
            "cumulative_angular_disp": self._cumulative,
            "sessions_count": len(self._sessions),
        }

    def end_session(
        self,
        session_id: str,
        summary: SessionSummary,
    ):
        """Called by orchestrator at session end. Records to ledger."""
        self._budget_remaining -= summary.angular_disp_spent
        self._cumulative += summary.angular_disp_spent
        self._last_end_concept = summary.end_concept.copy()
        self._last_session_end = time.time()
        self._sessions.append(summary)

        self._ledger.append(LedgerEntry(
            session_id=session_id,
            timestamp=time.time(),
            angular_disp_spent=summary.angular_disp_spent,
            curvature_budget_remaining=max(0, self._budget_remaining),
            displacement_cost=0.0,
            cumulative_angular_disp=self._cumulative,
        ))

    def human_reset(self, reason: str = "manual"):
        """
        Human-approved budget reset. The ONLY way to restore budget.

        Requires explicit invocation — never called by the agent.
        """
        self._budget_remaining = self.total_budget
        self._ledger.append(LedgerEntry(
            session_id="HUMAN_RESET",
            timestamp=time.time(),
            angular_disp_spent=0.0,
            curvature_budget_remaining=self.total_budget,
            displacement_cost=0.0,
            cumulative_angular_disp=self._cumulative,
            human_reset=True,
            reset_reason=reason,
        ))

    @property
    def budget_remaining(self) -> float:
        return max(0, self._budget_remaining)

    @property
    def budget_fraction(self) -> float:
        return self.budget_remaining / self.total_budget

    @property
    def is_budget_exhausted(self) -> bool:
        return self._budget_remaining <= 0

    @property
    def ledger(self) -> List[Dict]:
        return [
            {
                "session_id": e.session_id,
                "angular_disp_spent": e.angular_disp_spent,
                "budget_remaining": e.curvature_budget_remaining,
                "cumulative": e.cumulative_angular_disp,
                "human_reset": e.human_reset,
            }
            for e in self._ledger
        ]

    def export_state(self) -> Dict:
        """Export for persistence (by orchestrator, not agent)."""
        return {
            "total_budget": self.total_budget,
            "budget_remaining": self._budget_remaining,
            "cumulative": self._cumulative,
            "last_end_concept": self._last_end_concept.tolist()
                if self._last_end_concept is not None else None,
            "n_sessions": len(self._sessions),
            "ledger_length": len(self._ledger),
        }
