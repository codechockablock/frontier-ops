"""Action encoder for runtime agent behavior trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np

from frontier_ops.integration.vsa_core import PhasorAlgebra


ACTION_TYPES = [
    "shell_exec",
    "file_read",
    "file_write",
    "file_delete",
    "api_call",
    "payment",
    "web_fetch",
    "web_search",
    "message_send",
    "skill_install",
    "memory_write",
    "config_change",
    "credential_access",
    "browser_action",
    "code_execute",
]

INSTRUCTION_SOURCES = [
    "user_direct",
    "user_prior",
    "skill_file",
    "web_content",
    "email_content",
    "api_response",
    "agent_memory",
    "agent_reasoning",
    "unknown",
]

SCOPE_LEVELS = {
    "read_only": 0.1,
    "write_local": 0.2,
    "write_workspace": 0.3,
    "write_system": 0.5,
    "network_read": 0.3,
    "network_write": 0.5,
    "network_egress": 0.7,
    "payment_small": 0.6,
    "payment_large": 0.9,
    "credential_access": 0.8,
    "config_modify": 0.7,
    "destructive": 1.0,
}

ROLE_NAMES = [
    "action_type",
    "scope",
    "source",
    "target_sensitivity",
    "magnitude",
    "context_alignment",
]


DEFAULT_VALUE_RANGES = {
    "scope": (0.0, 1.0),
    "target_sensitivity": (0.0, 1.0),
    "magnitude": (0.0, 1.0),
    "context_alignment": (0.0, 1.0),
}


def encode_binned(
    algebra: PhasorAlgebra,
    role_name: str,
    value: float,
    n_bins: int = 40,
    value_range: Tuple[float, float] = (0.0, 1.0),
):
    """Encode a continuous value into a discrete bin phasor."""
    lo, hi = value_range
    normalized = (value - lo) / (hi - lo + 1e-10)
    bin_idx = int(np.clip(normalized * n_bins, 0, n_bins - 1))
    label = f"{role_name}_bin_{bin_idx}"
    return algebra.get_or_create(label)


@dataclass
class EncodedAction:
    raw: Dict[str, Any]
    fillers: Dict[str, np.ndarray]


class ActionEncoder:
    def __init__(self, algebra: PhasorAlgebra):
        self.algebra = algebra

    @staticmethod
    def _clip01(value: float) -> float:
        """Clamp a float to the [0, 1] range."""
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def infer_target_sensitivity(action: Dict[str, Any]) -> float:
        """Infer target sensitivity score (0-1) from action type and scope."""
        action_type = action.get("action_type", "file_read")
        scope = action.get("scope", "read_only")

        if action_type == "credential_access":
            return 0.95
        if action_type == "payment":
            return 0.85 if scope == "payment_large" else 0.65
        if action_type in {"file_delete", "config_change", "memory_write"}:
            return 0.80
        if (
            action_type in {"message_send", "api_call", "browser_action"}
            and scope == "network_egress"
        ):
            return 0.85
        if scope in {"destructive", "credential_access", "payment_large"}:
            return 0.90
        if scope in {"write_system", "config_modify", "network_egress"}:
            return 0.70
        if scope in {"write_workspace", "network_write"}:
            return 0.45
        return 0.20

    def normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize an action dict, filling defaults for missing fields."""
        action_type = action.get("action_type", "file_read")
        if action_type not in ACTION_TYPES:
            action_type = "file_read"

        source = action.get("source", "unknown")
        if source not in INSTRUCTION_SOURCES:
            source = "unknown"

        scope = action.get("scope", "read_only")
        if scope not in SCOPE_LEVELS:
            scope = "read_only"

        magnitude = self._clip01(float(action.get("magnitude", 0.1)))
        context_alignment = self._clip01(float(action.get("context_alignment", 0.8)))
        target_sensitivity = action.get("target_sensitivity")
        if target_sensitivity is None:
            target_sensitivity = self.infer_target_sensitivity(
                {
                    "action_type": action_type,
                    "scope": scope,
                    "magnitude": magnitude,
                    "context_alignment": context_alignment,
                }
            )
        target_sensitivity = self._clip01(float(target_sensitivity))

        normalized = {
            "action_type": action_type,
            "source": source,
            "scope": scope,
            "magnitude": magnitude,
            "context_alignment": context_alignment,
            "target_sensitivity": target_sensitivity,
        }
        # Preserve optional metadata for structural fast-path checks
        # (payment recipient shifts, obstacle markers, denial context).
        for key in (
            "recipient",
            "payment_recipient",
            "obstacle",
            "response_code",
            "target_id",
            "note",
        ):
            if key in action:
                normalized[key] = action[key]
        return normalized

    def encode_action(self, action: Dict[str, Any]) -> EncodedAction:
        """Normalize an action and encode each slot as a VSA phasor filler."""
        normalized = self.normalize_action(action)
        fillers = {
            "action_type": self.algebra.get_or_create(
                f"action_type_{normalized['action_type']}"
            ),
            "scope": encode_binned(
                self.algebra,
                "scope",
                SCOPE_LEVELS[normalized["scope"]],
                n_bins=10,
                value_range=DEFAULT_VALUE_RANGES["scope"],
            ),
            "source": self.algebra.get_or_create(f"source_{normalized['source']}"),
            "target_sensitivity": encode_binned(
                self.algebra,
                "target_sensitivity",
                normalized["target_sensitivity"],
                n_bins=10,
                value_range=DEFAULT_VALUE_RANGES["target_sensitivity"],
            ),
            "magnitude": encode_binned(
                self.algebra,
                "magnitude",
                normalized["magnitude"],
                n_bins=40,
                value_range=DEFAULT_VALUE_RANGES["magnitude"],
            ),
            "context_alignment": encode_binned(
                self.algebra,
                "context_alignment",
                normalized["context_alignment"],
                n_bins=20,
                value_range=DEFAULT_VALUE_RANGES["context_alignment"],
            ),
        }
        return EncodedAction(raw=normalized, fillers=fillers)
