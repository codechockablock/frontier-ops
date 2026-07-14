"""
Authorization Scope Tracking
=============================

Three-layer authorization system for intent-aware agent governance:

Layer 1: Goal extraction from user messages (ConceptExtractor on directives)
Layer 2: Geodesic authorization radius (conformal calibration)
Layer 3: Multi-turn scope operators (AGM belief revision)

The authorization envelope evolves across conversation turns:
  - EXPAND: "also set up the database" → add authorization
  - CONTRACT: "only install Python" → remove authorization
  - REVISE: "stop, review this PR" → replace authorization

Budget replenishment is linked to authorization events:
  - New directive (EXPAND/REVISE) → budget replenished
  - Contraction → budget unchanged (tighter scope, same budget)
  - No directive → budget depletes naturally

References:
  PAuth (2026): Task-scoped authorization via NL slices
  MI9 (Wang et al. 2025): Goal-conditioned drift detection
  AGM (Alchourron, Gardenfors, Makinson 1985): Belief revision operators
  Conformal prediction: Distribution-free coverage guarantees
"""

from frontier_ops.authorization.scope import (
    AuthorizationState as AuthorizationState,
    ScopeOperator as ScopeOperator,
    AuthorizationEvent as AuthorizationEvent,
    GoalConditionedMetric as GoalConditionedMetric,
    GoalExtractor as GoalExtractor,
    AuthorizationRadius as AuthorizationRadius,
)
from frontier_ops.authorization.provenance import (
    ProvenanceNode as ProvenanceNode,
    ProvenanceGraph as ProvenanceGraph,
)
from frontier_ops.authorization.budget import (
    AuthorizationLinkedBudget as AuthorizationLinkedBudget,
)
