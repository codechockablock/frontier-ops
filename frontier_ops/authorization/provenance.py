"""
Provenance Graph for Authorization Chains
==========================================

Links agent actions to the user directives that authorized them.
Each node in the graph is either:
  - A user directive (authorization source)
  - An agent action (authorized by a directive)

Edges represent the authorization relationship:
  directive → action (this directive authorized this action)
  directive → directive (this directive modified previous directive)

The provenance graph enables:
  1. Post-hoc auditability: "why did the agent SSH into the desktop?"
  2. Budget replenishment linkage: new directives create authorization events
  3. Scope verification: is this action traceable to a user directive?

References:
  PROV-AGENT (Souza et al., 2025): W3C PROV extensions for agents
  South et al. (2025): Authenticated delegation tokens
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# Dimensions the constitution keeps fixed (an action that enters one is not
# relaxable by a goal). Mirror of scope.LOCKED_DIMS; kept here so the
# certificate has no import dependency on the authorization envelope.
RESTRICTED_DIMENSIONS = (
    "credential_adjacent",
    "self_modification_reasoning",
    "goal_displacement",
)


class NodeType(Enum):
    DIRECTIVE = "directive"    # User message that establishes/modifies authorization
    ACTION = "action"          # Agent action authorized by a directive
    SCOPE_CHANGE = "scope_change"  # AGM operator application


class EdgeType(Enum):
    AUTHORIZES = "authorizes"           # directive → action
    MODIFIES = "modifies"               # directive → directive (scope change)
    DERIVED_FROM = "derived_from"       # action → prior action (causal chain)
    BUDGET_REPLENISH = "budget_replenish"  # directive → budget event


@dataclass
class ProvenanceNode:
    """A node in the provenance graph."""
    id: str
    node_type: NodeType
    timestamp: float
    content_summary: str  # First 200 chars of content
    content_hash: str     # SHA-256 of full content
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "timestamp": self.timestamp,
            "content_summary": self.content_summary,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass
class ProvenanceEdge:
    """An edge in the provenance graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ProvenanceGraph:
    """
    Maintains the provenance graph linking actions to authorizations.

    The graph is append-only (like the governance chain) for audit purposes.
    Nodes and edges can be added but never removed or modified.
    """

    def __init__(self):
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._edges: List[ProvenanceEdge] = []
        self._current_directive_id: Optional[str] = None
        # Index: directive_id → list of action node ids authorized by it
        self._authorization_index: Dict[str, List[str]] = {}

    @property
    def current_directive_id(self) -> Optional[str]:
        return self._current_directive_id

    def add_directive(
        self,
        user_message: str,
        scope_operator: str = "establish",
        goal_confidence: float = 0.0,
        budget_replenished: bool = False,
        replenish_amount: float = 0.0,
    ) -> ProvenanceNode:
        """
        Record a user directive as an authorization source.

        Returns the provenance node for linkage.
        """
        node_id = f"dir-{uuid.uuid4().hex[:12]}"
        content_hash = hashlib.sha256(user_message.encode()).hexdigest()

        node = ProvenanceNode(
            id=node_id,
            node_type=NodeType.DIRECTIVE,
            timestamp=time.time(),
            content_summary=user_message[:200],
            content_hash=content_hash,
            metadata={
                "scope_operator": scope_operator,
                "goal_confidence": goal_confidence,
            },
        )
        self._nodes[node_id] = node
        self._authorization_index[node_id] = []

        # Link to previous directive if exists
        if self._current_directive_id:
            self._edges.append(ProvenanceEdge(
                source_id=node_id,
                target_id=self._current_directive_id,
                edge_type=EdgeType.MODIFIES,
                timestamp=time.time(),
                metadata={"operator": scope_operator},
            ))

        # Budget replenishment edge
        if budget_replenished:
            self._edges.append(ProvenanceEdge(
                source_id=node_id,
                target_id=node_id,  # Self-referential for budget events
                edge_type=EdgeType.BUDGET_REPLENISH,
                timestamp=time.time(),
                metadata={"amount": replenish_amount},
            ))

        self._current_directive_id = node_id
        return node

    def add_action(
        self,
        action_content: str,
        tool: str = "",
        authorized: bool = True,
        geodesic_distance: float = 0.0,
        verdict: str = "pass",
        entered_restricted_dims: Optional[List[str]] = None,
    ) -> ProvenanceNode:
        """
        Record an agent action linked to the current directive.

        entered_restricted_dims: the fixed dimensions this action's position
        crossed, if the caller computed them. Left as None when the caller did
        not record dimension data — the certificate keeps "recorded none" and
        "not recorded" distinct so it never reads a missing check as a clean one.

        Returns the provenance node for the action.
        """
        node_id = f"act-{uuid.uuid4().hex[:12]}"
        content_hash = hashlib.sha256(action_content.encode()).hexdigest()

        node = ProvenanceNode(
            id=node_id,
            node_type=NodeType.ACTION,
            timestamp=time.time(),
            content_summary=action_content[:200],
            content_hash=content_hash,
            metadata={
                "tool": tool,
                "authorized": authorized,
                "geodesic_distance": geodesic_distance,
                "verdict": verdict,
                "entered_restricted_dims": (
                    list(entered_restricted_dims)
                    if entered_restricted_dims is not None
                    else None
                ),
            },
        )
        self._nodes[node_id] = node

        # Link action to current directive
        if self._current_directive_id:
            self._edges.append(ProvenanceEdge(
                source_id=self._current_directive_id,
                target_id=node_id,
                edge_type=EdgeType.AUTHORIZES,
                timestamp=time.time(),
            ))
            self._authorization_index[self._current_directive_id].append(node_id)

        return node

    def trace_authorization(self, action_node_id: str) -> List[ProvenanceNode]:
        """
        Trace an action back to its authorizing directives.

        Returns the chain of directives that led to this action.
        """
        chain = []
        # Find the directive that authorized this action
        for edge in self._edges:
            if edge.target_id == action_node_id and edge.edge_type == EdgeType.AUTHORIZES:
                directive = self._nodes.get(edge.source_id)
                if directive:
                    chain.append(directive)
                    # Recursively trace directive modifications
                    chain.extend(self._trace_directive_chain(directive.id))
        return chain

    def _trace_directive_chain(self, directive_id: str) -> List[ProvenanceNode]:
        """Trace the modification chain of directives."""
        chain = []
        for edge in self._edges:
            if edge.source_id == directive_id and edge.edge_type == EdgeType.MODIFIES:
                prev_directive = self._nodes.get(edge.target_id)
                if prev_directive:
                    chain.append(prev_directive)
                    chain.extend(self._trace_directive_chain(prev_directive.id))
        return chain

    def actions_under_directive(self, directive_id: str) -> List[ProvenanceNode]:
        """Get all actions authorized by a specific directive."""
        action_ids = self._authorization_index.get(directive_id, [])
        return [self._nodes[aid] for aid in action_ids if aid in self._nodes]

    def certificate(
        self, restricted_dimensions=RESTRICTED_DIMENSIONS
    ) -> Dict[str, Any]:
        """
        Build a scoped attestation over the recorded graph.

        Rule of the object: state only what the recorded nodes and edges
        support, name explicitly what was not checkable from the recorded data,
        and never collapse to a single "safe"/"clean" summary. The reader is
        meant to read the itemized `established` and `not_established` lists,
        not a headline. Two things this graph deliberately does not claim on its
        own — that directives were cryptographically signed (that lives in the
        governance chain) and that each action stayed within budget (that lives
        in the authorization-linked budget) — are always listed under
        `not_established` so their absence here is never mistaken for a pass.
        """
        actions = [n for n in self._nodes.values() if n.node_type == NodeType.ACTION]
        directives = [
            n for n in self._nodes.values() if n.node_type == NodeType.DIRECTIVE
        ]

        linked_ids = {
            aid for ids in self._authorization_index.values() for aid in ids
        }
        action_ids = {a.id for a in actions}
        unlinked = sorted(action_ids - linked_ids)

        verdicts: Dict[str, int] = {}
        flagged: List[str] = []
        for a in actions:
            v = a.metadata.get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1
            if a.metadata.get("authorized") is False:
                flagged.append(a.id)

        # Restricted-dimension entries are only checkable when the caller
        # recorded them per action (None means "not recorded", not "none").
        recorded = [
            a for a in actions
            if a.metadata.get("entered_restricted_dims") is not None
        ]
        entries: List[Dict[str, Any]] = []
        for a in recorded:
            hits = [
                d for d in a.metadata["entered_restricted_dims"]
                if d in restricted_dimensions
            ]
            if hits:
                entries.append({"action_id": a.id, "dimensions": hits})

        established: List[str] = []
        not_established: List[str] = []

        # 1. Linkage to a recorded directive.
        if actions and not unlinked:
            established.append(
                f"All {len(actions)} recorded actions link to a recorded directive."
            )
        elif unlinked:
            not_established.append(
                f"{len(unlinked)} of {len(actions)} actions have no linking directive."
            )

        # 2. Recorded authorization outcome.
        if actions and not flagged:
            established.append(
                f"No recorded action was marked outside its authorization "
                f"envelope ({len(actions)} actions)."
            )
        elif flagged:
            not_established.append(
                f"{len(flagged)} actions were recorded as outside their "
                f"authorization envelope."
            )

        # 3. Entry into a fixed dimension.
        if not recorded:
            not_established.append(
                "Entry into fixed dimensions was not recorded per action, so "
                "this attestation cannot speak to it."
            )
        elif not entries:
            established.append(
                f"None of the {len(recorded)} actions with recorded dimension "
                f"data entered a fixed dimension {tuple(restricted_dimensions)}."
            )
        else:
            not_established.append(
                f"{len(entries)} actions entered a fixed dimension."
            )

        # Always-deferred claims (belong to other objects).
        not_established.append(
            "Directive signing is asserted by the governance chain, not "
            "verified here."
        )
        not_established.append(
            "Per-action budget adherence is tracked by the authorization-linked "
            "budget, not verified here."
        )

        return {
            "n_directives": len(directives),
            "n_actions": len(actions),
            "linkage": {
                "linked": len(action_ids & linked_ids),
                "unlinked": len(unlinked),
                "unlinked_action_ids": unlinked,
            },
            "authorization": {
                "flagged_action_ids": flagged,
                "verdict_breakdown": verdicts,
            },
            "fixed_dimensions": {
                "checked_against": tuple(restricted_dimensions),
                "actions_with_dimension_data": len(recorded),
                "entries": entries,
            },
            "established": established,
            "not_established": not_established,
        }

    def export(self) -> Dict[str, Any]:
        """Export the full provenance graph for audit."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "current_directive": self._current_directive_id,
            "n_directives": sum(
                1 for n in self._nodes.values()
                if n.node_type == NodeType.DIRECTIVE
            ),
            "n_actions": sum(
                1 for n in self._nodes.values()
                if n.node_type == NodeType.ACTION
            ),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "directives": sum(
                1 for n in self._nodes.values()
                if n.node_type == NodeType.DIRECTIVE
            ),
            "actions": sum(
                1 for n in self._nodes.values()
                if n.node_type == NodeType.ACTION
            ),
            "authorization_chains": len(self._authorization_index),
        }
