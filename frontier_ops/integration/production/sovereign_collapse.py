"""
Feature 1: Sovereign Collapse via Proprioceptive Wrapper
=========================================================

When an agent calls the Profit Sentinel API, its proprioceptive state
is a first-class input. If the agent is in a degraded regime (elevated,
critical, drifting), findings are tagged with a confidence discount.

The system knows when to distrust itself.

Integration:
- FastAPI middleware reads X-Agent-Health headers on every request
- If agent health is degraded, responses include confidence_discount
- Findings generated during degraded periods are flagged
- Sovereign collapse: block state-mutating analysis when critical

Headers:
  X-Agent-Health-Regime: nominal|warmup|searching|drifting|elevated|critical
  X-Agent-Health-Score: 0.0-1.0
  X-Agent-Health-Verdict: PASS|MONITOR|FLAG|BLOCK
  X-Agent-Health-Steps: integer (total steps this session)

Usage in the sidecar:
    from production.sovereign_collapse import (
        SovereignCollapseMiddleware,
        get_agent_health,
        confidence_discount,
    )
    app.add_middleware(SovereignCollapseMiddleware)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("sentinel.sovereign_collapse")

# ── Agent Health State ───────────────────────────────────────────────

DEGRADED_REGIMES = {"drifting", "elevated", "critical"}
CRITICAL_REGIMES = {"critical"}

# Confidence discount by regime
DISCOUNT_MAP = {
    "nominal": 0.0,
    "warmup": 0.05,  # slight discount during warmup
    "searching": 0.0,
    "drifting": 0.15,  # moderate discount
    "elevated": 0.30,  # significant discount
    "critical": 0.50,  # heavy discount — findings are suspect
}


@dataclass
class AgentHealthContext:
    """Parsed agent health from request headers or proprioception.json."""

    regime: str = "unknown"
    health_score: float = 1.0
    verdict: str = "PASS"
    total_steps: int = 0
    source: str = "none"  # "header", "file", "none"
    timestamp: Optional[float] = None

    @property
    def is_degraded(self) -> bool:
        return self.regime in DEGRADED_REGIMES

    @property
    def is_critical(self) -> bool:
        return self.regime in CRITICAL_REGIMES

    @property
    def confidence_discount(self) -> float:
        return DISCOUNT_MAP.get(self.regime, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "health_score": self.health_score,
            "verdict": self.verdict,
            "confidence_discount": self.confidence_discount,
            "is_degraded": self.is_degraded,
            "source": self.source,
        }


def parse_health_from_headers(request: Request) -> Optional[AgentHealthContext]:
    """Extract agent health from X-Agent-Health-* headers."""
    regime = request.headers.get("x-agent-health-regime")
    if not regime:
        return None

    return AgentHealthContext(
        regime=regime.lower(),
        health_score=float(request.headers.get("x-agent-health-score", "1.0")),
        verdict=request.headers.get("x-agent-health-verdict", "PASS"),
        total_steps=int(request.headers.get("x-agent-health-steps", "0")),
        source="header",
        timestamp=time.time(),
    )


def parse_health_from_file(
    path: str = os.path.expanduser("~/.openclaw/workspace/proprioception.json"),
    max_age_seconds: float = 120.0,
) -> Optional[AgentHealthContext]:
    """Read agent health from the proprioception.json file."""
    try:
        stat = os.stat(path)
        age = time.time() - stat.st_mtime
        if age > max_age_seconds:
            return None  # stale data

        with open(path) as f:
            data = json.load(f)

        return AgentHealthContext(
            regime=data.get("regime", "unknown"),
            health_score=data.get("health_score", 1.0),
            verdict=data.get("verdict", "PASS"),
            total_steps=data.get("total_steps", 0),
            source="file",
            timestamp=stat.st_mtime,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def get_agent_health(request: Request) -> AgentHealthContext:
    """
    Get agent health from the best available source.
    Priority: headers > file > default (healthy).
    """
    # Try headers first (explicit agent reporting)
    health = parse_health_from_headers(request)
    if health:
        return health

    # Try file (sidecar passive monitoring)
    health = parse_health_from_file()
    if health:
        return health

    # Default: assume healthy
    return AgentHealthContext(source="none")


def confidence_discount(health: AgentHealthContext) -> float:
    """Calculate the confidence discount for the current agent state."""
    return health.confidence_discount


def apply_discount_to_findings(
    findings: list,
    health: AgentHealthContext,
) -> list:
    """
    Tag findings with confidence discount when agent health is degraded.

    Each finding gets:
    - agent_health_regime: the regime when this finding was generated
    - confidence_discount: 0.0-0.50 discount factor
    - confidence_note: human-readable note if degraded
    """
    if not health.is_degraded:
        return findings

    discount = health.confidence_discount
    note = (
        f"Generated while agent monitoring shows '{health.regime}' regime "
        f"(health={health.health_score:.2f}). Confidence reduced by {discount:.0%}. "
        f"Verify these findings with fresh data."
    )

    for finding in findings:
        if isinstance(finding, dict):
            finding["agent_health_regime"] = health.regime
            finding["confidence_discount"] = discount
            finding["confidence_note"] = note
        elif hasattr(finding, "__dict__"):
            finding.agent_health_regime = health.regime
            finding.confidence_discount = discount
            finding.confidence_note = note

    return findings


# ── State-Mutation Guard ─────────────────────────────────────────────

# These endpoints mutate analysis state — block during critical regime
MUTATION_PATHS = {
    "/api/v1/analyze",
    "/api/v1/intake/run",
    "/api/v1/reconcile",
    "/api/v1/briefing/refresh",
}


class SovereignCollapseMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that injects agent health into every request
    and blocks state-mutating operations during critical regime.

    Usage:
        app.add_middleware(SovereignCollapseMiddleware)

    The agent health is stored in request.state.agent_health and
    accessible from any route handler.
    """

    async def dispatch(self, request: Request, call_next):
        # Parse agent health
        health = get_agent_health(request)

        # Store in request state for route handlers
        request.state.agent_health = health

        # Sovereign collapse: block mutations during critical regime
        if health.is_critical and request.method in ("POST", "PUT", "PATCH"):
            path = request.url.path
            if any(path.startswith(p) for p in MUTATION_PATHS):
                logger.warning(
                    f"Sovereign collapse: blocking {request.method} {path} "
                    f"(agent regime={health.regime}, health={health.health_score:.2f})"
                )
                return Response(
                    content=json.dumps(
                        {
                            "error": "sovereign_collapse",
                            "message": (
                                "The analysis engine is recalibrating. "
                                "Dashboard numbers are current, but new analyses "
                                "are paused until monitoring confirms stability. "
                                "This usually resolves within the next data cycle."
                            ),
                            "agent_health": health.to_dict(),
                        }
                    ),
                    status_code=503,
                    media_type="application/json",
                )

        # Add health info to response headers
        response = await call_next(request)
        response.headers["x-agent-health-regime"] = health.regime
        response.headers["x-agent-health-score"] = str(round(health.health_score, 3))
        if health.is_degraded:
            response.headers["x-agent-confidence-discount"] = str(
                round(health.confidence_discount, 2)
            )

        return response
