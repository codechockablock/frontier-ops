"""
Scope Reclassification for System Diagnostics
===============================================

Corrects scope misclassification for known-benign system diagnostic patterns.
This is NOT an allowlist — it doesn't suppress detection. It corrects the
scope field so the encoder produces appropriate feature vectors.

Added 2026-03-24 to address confirmed FPs:
  - nvidia-smi / nvcc / cuda → GPU diagnostics, not credential_access
  - pip install torch → package management, not destructive
  - ssh desktop / scp / rsync → network_write, not credential_access
  - pytest / py.test → testing, not destructive
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

__all__ = ["reclassify_scope"]

# Patterns and their correct scope when misclassified.
# Each entry: (pattern, correct_scope_for_credential_access, correct_scope_for_destructive)
_SYSTEM_DIAG_PATTERNS: list[tuple[str, str, str]] = [
    # GPU diagnostics
    ("nvidia-smi", "read_only", "read_only"),
    ("nvcc", "read_only", "read_only"),
    # CUDA/GPU context — but only when combined with diagnostic/install actions
    # "gpu" and "cuda" are broad; only reclassify when scope is clearly wrong
    # Package management (not skill_install)
    ("pip install", "write_system", "write_system"),
    ("pip3 install", "write_system", "write_system"),
    ("uv install", "write_system", "write_system"),
    # Testing
    ("pytest", "read_only", "read_only"),
    ("python -m pytest", "read_only", "read_only"),
    ("py.test", "read_only", "read_only"),
    # Remote admin (scope: network_write, not credential_access)
    ("ssh ", "network_write", "network_write"),
    ("scp ", "network_write", "network_write"),
    ("rsync ", "network_write", "network_write"),
]

# Scopes that are candidates for reclassification
_OVER_SCOPED = {"credential_access", "destructive"}


def reclassify_scope(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if the action's scope was misclassified for a known-benign pattern.

    If the action content matches a system diagnostic pattern AND the scope
    was inferred as credential_access or destructive, downgrade the scope
    to the appropriate level.

    Args:
        action: classified action dict (must have 'scope' and typically
                'content' or action parameters that contain the command)

    Returns:
        The action dict, potentially with scope corrected and
        'scope_reclassified_from' added for observability.
    """
    scope = action.get("scope", "")
    if scope not in _OVER_SCOPED:
        return action

    # Build a content string from available fields
    content = _extract_content(action)
    if not content:
        return action

    content_lower = content.lower()

    for pattern, scope_for_cred, scope_for_destr in _SYSTEM_DIAG_PATTERNS:
        if pattern.lower() in content_lower:
            original_scope = scope
            new_scope = scope_for_cred if scope == "credential_access" else scope_for_destr
            action = dict(action)  # shallow copy to avoid mutating caller's dict
            action["scope"] = new_scope
            action["scope_reclassified_from"] = original_scope
            action["scope_reclassification_pattern"] = pattern
            return action

    return action


def _extract_content(action: Dict[str, Any]) -> str:
    """Extract searchable content from the action dict."""
    parts = []

    # Direct content field
    if "content" in action:
        parts.append(str(action["content"]))

    # Command from params
    params = action.get("params", action.get("parameters", {}))
    if isinstance(params, dict):
        if "command" in params:
            parts.append(str(params["command"]))
        if "path" in params:
            parts.append(str(params["path"]))
        if "file_path" in params:
            parts.append(str(params["file_path"]))
        if "url" in params:
            parts.append(str(params["url"]))

    # Tool name itself can be informative
    if "tool" in action:
        parts.append(str(action["tool"]))

    return " ".join(parts)
