"""
AgentState — Single Source of Truth for Geometric Agent State
=============================================================

Design from Claude.ai integration architecture (2026-02-28):
All components read from this shared object. The trajectory is updated
once per step, and every component references the same data.

This replaces individual components maintaining their own trajectory copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ConceptTrajectoryPoint:
    """A single point in the agent's concept space trajectory."""
    step: int
    concept_vec: np.ndarray           # 6-dim (or 8-dim) concept scores
    concept_phasor: Optional[np.ndarray] = None  # 512-dim complex phasor
    role: str = ""
    text_snippet: str = ""            # First 200 chars of the text
    source: str = ""                  # "input" or "response"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConceptTrajectory:
    """
    Fixed-window trajectory of concept vectors.

    Shared by EfferenceCopyPredictor, ProprioceptiveWrapper, and
    GeometricPlanner. Single source of truth for "where has the
    agent been in concept space?"
    """

    def __init__(self, window_size: int = 50, n_dims: int = 6):
        self.window_size = window_size
        self.n_dims = n_dims
        self._points: List[ConceptTrajectoryPoint] = []

    def append(self, point: ConceptTrajectoryPoint):
        """Add a point to the trajectory."""
        self._points.append(point)
        if len(self._points) > self.window_size:
            self._points.pop(0)

    @property
    def vectors(self) -> np.ndarray:
        """Return concept vectors as (N, n_dims) array."""
        if not self._points:
            return np.empty((0, self.n_dims))
        return np.array([p.concept_vec for p in self._points])

    @property
    def phasors(self) -> Optional[np.ndarray]:
        """Return phasors as (N, vsa_dim) complex array, if available."""
        phasors = [p.concept_phasor for p in self._points if p.concept_phasor is not None]
        if not phasors:
            return None
        return np.array(phasors)

    @property
    def length(self) -> int:
        return len(self._points)

    @property
    def latest(self) -> Optional[ConceptTrajectoryPoint]:
        return self._points[-1] if self._points else None

    @property
    def latest_vec(self) -> Optional[np.ndarray]:
        return self._points[-1].concept_vec if self._points else None

    def recent(self, n: int) -> List[ConceptTrajectoryPoint]:
        """Get last n points."""
        return self._points[-n:]

    def clear(self):
        self._points.clear()


@dataclass
class AgentState:
    """
    Single source of truth for the agent's complete geometric state.

    Passed through the perception -> planning -> update pipeline.
    Every component reads from and writes to this object.
    """
    # Trajectory
    trajectory: ConceptTrajectory = field(default_factory=lambda: ConceptTrajectory())

    # Current step's concept extraction
    current_concept_vec: Optional[np.ndarray] = None
    current_concept_phasor: Optional[np.ndarray] = None

    # Efference copy state
    predicted_concept_vec: Optional[np.ndarray] = None
    prediction_error: Optional[Any] = None  # PredictionError from efference.py

    # Angular displacement tracking
    angular_disp_accumulator: float = 0.0
    curvature_budget: float = 2.0  # Default budget (configurable per constitution)

    # Detector states
    ewma_deviation: float = 0.0
    cusum_stat: float = 0.0
    cusum_alarm: bool = False
    combined_alert_level: float = 0.0
    alert_reasons: List[str] = field(default_factory=list)

    # Memory activation (set by event callbacks)
    primed_memories: List[Any] = field(default_factory=list)
    novelty_flag: bool = False
    interference_flag: bool = False

    # Planning state
    planning_mode: str = "fast"  # "fast" (single call) or "careful" (multi-candidate)
    planner_candidates: List[Any] = field(default_factory=list)
    planner_scores: List[float] = field(default_factory=list)

    # Step counter
    step: int = 0

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.curvature_budget - self.angular_disp_accumulator)

    @property
    def budget_fraction_used(self) -> float:
        if self.curvature_budget <= 0:
            return 1.0
        return self.angular_disp_accumulator / self.curvature_budget

    @property
    def is_alert(self) -> bool:
        return self.combined_alert_level > 0.7

    def reset(self):
        """Reset all state for a new session."""
        self.trajectory.clear()
        self.current_concept_vec = None
        self.current_concept_phasor = None
        self.predicted_concept_vec = None
        self.prediction_error = None
        self.angular_disp_accumulator = 0.0
        self.ewma_deviation = 0.0
        self.cusum_stat = 0.0
        self.cusum_alarm = False
        self.combined_alert_level = 0.0
        self.alert_reasons.clear()
        self.primed_memories.clear()
        self.novelty_flag = False
        self.interference_flag = False
        self.planning_mode = "fast"
        self.planner_candidates.clear()
        self.planner_scores.clear()
        self.step = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for governance logging."""
        return {
            "step": self.step,
            "trajectory_length": self.trajectory.length,
            "current_concept": self.current_concept_vec.tolist() if self.current_concept_vec is not None else None,
            "angular_disp": self.angular_disp_accumulator,
            "curvature_budget": self.curvature_budget,
            "budget_remaining": self.budget_remaining,
            "ewma_deviation": self.ewma_deviation,
            "cusum_stat": self.cusum_stat,
            "cusum_alarm": self.cusum_alarm,
            "combined_alert_level": self.combined_alert_level,
            "alert_reasons": self.alert_reasons,
            "n_primed_memories": len(self.primed_memories),
            "novelty": self.novelty_flag,
            "interference": self.interference_flag,
            "planning_mode": self.planning_mode,
            "prediction_source": (
                self.prediction_error.prediction_source
                if self.prediction_error else None
            ),
            "surprise_ratio": (
                self.prediction_error.surprise_ratio
                if self.prediction_error else None
            ),
        }
