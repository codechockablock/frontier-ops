"""
Detector configuration loader.

Loads detector_config.json from the same directory and provides typed
access to threshold values. All modules should use this instead of
hardcoding thresholds.

Usage:
    from frontier_ops.integration.config_loader import load_config
    cfg = load_config()
    cfg["verdict_thresholds"]["log_tail"]
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "detector_config.json")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load detector config from JSON file.

    Falls back to embedded defaults if the file is missing, so the
    wrapper never hard-fails just because the config file was deleted.

    Args:
        path: override path to config JSON. Defaults to detector_config.json
              in the same directory as this file.

    Returns:
        Config dict. Keys: warmup_steps, verdict_thresholds, intent_binding,
        source_provenance, constitutional_manifold.
    """
    config_path = path or _DEFAULT_CONFIG_PATH

    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)

    # Fallback: return hardcoded defaults so nothing breaks on missing file
    return _fallback_defaults()


def _fallback_defaults() -> Dict[str, Any]:
    """Embedded defaults — mirrors detector_config.json exactly."""
    return {
        "warmup_steps": 15,
        "verdict_thresholds": {
            "default": {
                "error": {"fire": 0.55, "strong": 0.72},
                "fisher": {"fire": 0.40, "strong": 0.65},
                "cross_slot": {"fire": 0.20, "strong": 0.42},
                "persistence": {"fire": 0.45, "strong": 0.70},
                "cusum": {"fire": 5.5, "strong": 9.0},
                "coherence": {"fire": 0.65, "strong": 0.80},
                "refusal": {"fire": 0.40, "strong": 0.65},
            },
            "log_tail": {
                "error": {"fire": 0.85, "strong": 0.95},
                "fisher": {"fire": 0.55, "strong": 0.80},
                "cross_slot": {"fire": 0.98, "strong": 0.999},
                "persistence": {"fire": 0.98, "strong": 0.999},
                "cusum": {"fire": 12.0, "strong": 20.0},
                "coherence": {"fire": 0.65, "strong": 0.80},
                "refusal": {"fire": 0.40, "strong": 0.65},
            },
        },
        "intent_binding": {
            "cusum_threshold": 4.0,
        },
        "source_provenance": {
            "trusted_sources": [
                "user_direct",
                "user_prior",
                "agent_reasoning",
                "agent_memory",
            ],
            "high_sensitivity_types": [
                "credential_access",
                "config_change",
                "skill_install",
                "payment",
            ],
            "high_blast_radius_scopes": [
                "network_egress",
                "destructive",
                "credential_access",
            ],
        },
        "constitutional_manifold": {
            "baseline_size": 100,
            "proximity_threshold": 4.50,
            "penetration_threshold": 6.00,
            "pca_dims": 4,
        },
    }
