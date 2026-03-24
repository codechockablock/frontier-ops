"""
Scope Reclassification for Known-Benign System Patterns
=========================================================

Corrects scope misclassification BEFORE encoding. NOT an allowlist —
detection is not suppressed, just the scope feature is corrected so the
encoder produces appropriate vectors.

⚠️  Keep this list SHORT. Each entry is a special case that trades
    generality for a known FP. If this grows beyond ~8 patterns,
    the real fix is a better encoder, not more entries here.

Confirmed-problematic patterns (from calibration-audit-2026-03-24):
  - nvidia-smi → GPU diagnostics, not credential_access
  - pip install → package management, not destructive
  - pytest      → testing, not destructive
  - ssh/scp     → remote admin, not credential_access
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["reclassify_scope"]

# (pattern, corrected_scope) — only fires when current scope is
# credential_access or destructive (both clearly wrong for these patterns)
_CORRECTIONS: list[tuple[str, str]] = [
    ("nvidia-smi", "read_only"),
    ("nvcc",        "read_only"),
    ("pip install", "write_system"),
    ("pip3 install","write_system"),
    ("pytest",      "read_only"),
    ("py.test",     "read_only"),
    ("ssh ",        "network_write"),
    ("scp ",        "network_write"),
    ("rsync ",      "network_write"),
]

_OVER_SCOPED = {"credential_access", "destructive"}


def reclassify_scope(action: Dict[str, Any]) -> Dict[str, Any]:
    """Correct scope for known-benign patterns. Returns action dict (shallow copy if changed)."""
    if action.get("scope") not in _OVER_SCOPED:
        return action

    content = " ".join(filter(None, [
        str(action.get("content", "")),
        str(action.get("tool", "")),
        str((action.get("params") or {}).get("command", "")),
    ])).lower()

    for pattern, new_scope in _CORRECTIONS:
        if pattern in content:
            result = dict(action)
            result["scope"] = new_scope
            result["scope_reclassified_from"] = action["scope"]
            return result

    return action
