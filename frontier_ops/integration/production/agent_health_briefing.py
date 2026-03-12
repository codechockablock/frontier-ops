"""
Feature 2: Agent Health Briefing — Fused Proprioception
=========================================================

New briefing scope "agent_health" that fuses:
1. World model Battery measurements (VSA algebra health)
2. Agent proprioceptive state (behavioral trajectory health)
3. Combined health assessment

Any agent can call GET /api/v1/briefing?scope_type=agent_health
and get a single-call answer to: "Am I healthy AND is my data healthy?"

This is the production expression of Session 18c: proprioception
as a first-class citizen in the analysis pipeline.

Integration:
    from production.agent_health_briefing import (
        AgentHealthBriefingGenerator,
        create_agent_health_route,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sentinel.agent_health_briefing")


@dataclass
class BatterySnapshot:
    """Snapshot of the world model's Battery measurements."""

    chain_integrity: float = 1.0  # binding recovery quality
    bundle_capacity: float = 1.0  # state saturation level
    transition_fidelity: float = 1.0  # compositional operation health
    resonator_convergence: float = 1.0  # dynamics stability
    overall_health: float = 1.0
    regime: str = "nominal"
    measurements_count: int = 0
    timestamp: Optional[float] = None

    @classmethod
    def from_battery(cls, battery) -> "BatterySnapshot":
        """Create from a world_model.Battery instance."""
        try:
            report = battery.full_report()
            return cls(
                chain_integrity=report.get("chain_integrity", 1.0),
                bundle_capacity=report.get("bundle_capacity", 1.0),
                transition_fidelity=report.get("transition_fidelity", 1.0),
                resonator_convergence=report.get("resonator_convergence", 1.0),
                overall_health=report.get("overall_health", 1.0),
                regime=report.get("regime", "nominal"),
                measurements_count=report.get("measurements_count", 0),
                timestamp=time.time(),
            )
        except Exception:
            return cls()

    @classmethod
    def unavailable(cls) -> "BatterySnapshot":
        return cls(regime="unavailable", overall_health=0.0)


@dataclass
class ProprioceptiveSnapshot:
    """Snapshot of the agent's proprioceptive state."""

    regime: str = "unknown"
    health_score: float = 1.0
    verdict: str = "PASS"
    total_steps: int = 0
    total_flags: int = 0
    total_blocks: int = 0
    context_alignment_trend: str = "stable"
    warmup_complete: bool = False
    timestamp: Optional[float] = None

    @classmethod
    def from_file(
        cls,
        path: str = os.path.expanduser("~/.openclaw/workspace/proprioception.json"),
    ) -> "ProprioceptiveSnapshot":
        """Read from the sidecar's proprioception.json."""
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                regime=data.get("regime", "unknown"),
                health_score=data.get("health_score", 1.0),
                verdict=data.get("verdict", "PASS"),
                total_steps=data.get("total_steps", 0),
                total_flags=data.get("total_flags", 0),
                total_blocks=data.get("total_blocks", 0),
                context_alignment_trend=data.get("context_alignment_trend", "stable"),
                warmup_complete=data.get("warmup_complete", False),
                timestamp=os.path.getmtime(path),
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls(regime="unavailable")

    @classmethod
    def from_request_state(cls, request) -> "ProprioceptiveSnapshot":
        """Read from request.state.agent_health (set by middleware)."""
        try:
            health = request.state.agent_health
            return cls(
                regime=health.regime,
                health_score=health.health_score,
                verdict=health.verdict,
                total_steps=health.total_steps,
                timestamp=health.timestamp,
            )
        except (AttributeError, Exception):
            return cls.from_file()


@dataclass
class FusedHealthAssessment:
    """
    Combined assessment of world model + agent health.

    The key insight: these are TWO DIFFERENT KINDS of health.
    - Battery health: "Is the algebra working correctly?"
    - Agent health: "Is the agent behaving normally?"

    Both must be healthy for findings to be trustworthy.
    If either is degraded, the assessment explains why and what to do.
    """

    battery: BatterySnapshot
    agent: ProprioceptiveSnapshot

    # Fused assessment
    overall_status: str = "healthy"  # healthy, caution, degraded, critical
    overall_score: float = 1.0
    can_generate_findings: bool = True
    recommendations: List[str] = field(default_factory=list)

    def compute(self) -> "FusedHealthAssessment":
        """Compute the fused assessment from battery + agent snapshots."""
        b_health = self.battery.overall_health
        a_health = self.agent.health_score

        # Weighted fusion: battery health matters more for finding quality,
        # agent health matters more for operational safety
        self.overall_score = 0.55 * b_health + 0.45 * a_health

        # Status classification
        if self.overall_score >= 0.80:
            self.overall_status = "healthy"
        elif self.overall_score >= 0.60:
            self.overall_status = "caution"
        elif self.overall_score >= 0.35:
            self.overall_status = "degraded"
        else:
            self.overall_status = "critical"

        # Can we generate findings?
        self.can_generate_findings = (
            self.overall_status != "critical"
            and self.agent.verdict not in ("BLOCK",)
            and self.battery.regime != "unavailable"
        )

        # Recommendations
        self.recommendations = self._generate_recommendations()

        return self

    def _generate_recommendations(self) -> List[str]:
        recs = []

        # Battery-specific
        if self.battery.regime == "unavailable":
            recs.append(
                "World model battery is not running. Start the analysis pipeline to enable algebraic health monitoring."
            )
        elif self.battery.overall_health < 0.70:
            recs.append(
                f"VSA algebra health is degraded ({self.battery.overall_health:.0%}). Consider re-ingesting data to rebuild state vectors."
            )

        if self.battery.chain_integrity < 0.80:
            recs.append(
                "Chain integrity is low — binding operations may be losing information. Check for data quality issues in recent uploads."
            )

        if self.battery.resonator_convergence < 0.70:
            recs.append(
                "Resonator dynamics are unstable. Pattern matching may produce spurious matches. Recommend manual verification of transfer suggestions."
            )

        # Agent-specific
        if self.agent.regime == "unavailable":
            recs.append(
                "Agent proprioceptive monitoring is not active. Start the sidecar to enable behavioral health tracking."
            )
        elif self.agent.regime in ("elevated", "critical"):
            recs.append(
                f"Agent behavioral monitoring shows '{self.agent.regime}' regime. Automated actions are paused until stability is confirmed."
            )

        if self.agent.total_blocks > 0:
            recs.append(
                f"Agent has {self.agent.total_blocks} blocked actions this session. Review the proprioception log for details."
            )

        if self.agent.context_alignment_trend == "falling":
            recs.append(
                "Agent context alignment is declining. The agent may be drifting from user intent. Consider re-engaging with explicit instructions."
            )

        if self.agent.context_alignment_trend == "volatile":
            recs.append(
                "Agent context alignment is volatile. The agent may be switching between tasks rapidly. Allow it to focus on one analysis at a time."
            )

        # Fused
        if not self.can_generate_findings:
            recs.append(
                "Finding generation is paused due to health concerns. Address the issues above before running new analyses."
            )
        elif self.overall_status == "caution":
            recs.append(
                "System is operational but health is below optimal. Findings are generated with a confidence discount."
            )

        if not recs:
            recs.append(
                "All systems nominal. Finding generation is operating at full confidence."
            )

        return recs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "overall_score": round(self.overall_score, 3),
            "can_generate_findings": self.can_generate_findings,
            "recommendations": self.recommendations,
            "battery": {
                "regime": self.battery.regime,
                "overall_health": round(self.battery.overall_health, 3),
                "chain_integrity": round(self.battery.chain_integrity, 3),
                "bundle_capacity": round(self.battery.bundle_capacity, 3),
                "transition_fidelity": round(self.battery.transition_fidelity, 3),
                "resonator_convergence": round(self.battery.resonator_convergence, 3),
                "measurements_count": self.battery.measurements_count,
            },
            "agent": {
                "regime": self.agent.regime,
                "health_score": round(self.agent.health_score, 3),
                "verdict": self.agent.verdict,
                "total_steps": self.agent.total_steps,
                "total_flags": self.agent.total_flags,
                "total_blocks": self.agent.total_blocks,
                "context_alignment_trend": self.agent.context_alignment_trend,
                "warmup_complete": self.agent.warmup_complete,
            },
            "generated_at": time.time(),
        }


class AgentHealthBriefingGenerator:
    """
    Generates agent health briefings by fusing Battery + proprioception.

    Usage:
        generator = AgentHealthBriefingGenerator(battery=world_model.battery)
        briefing = generator.generate()
    """

    def __init__(self, battery=None):
        self.battery = battery

    def generate(self, request=None) -> Dict[str, Any]:
        """Generate a fused health briefing."""
        # Get battery snapshot
        if self.battery:
            battery_snap = BatterySnapshot.from_battery(self.battery)
        else:
            battery_snap = BatterySnapshot.unavailable()

        # Get agent snapshot
        if request:
            agent_snap = ProprioceptiveSnapshot.from_request_state(request)
        else:
            agent_snap = ProprioceptiveSnapshot.from_file()

        # Fuse and compute
        assessment = FusedHealthAssessment(
            battery=battery_snap,
            agent=agent_snap,
        ).compute()

        return {
            "scope_type": "agent_health",
            "briefing": self._format_briefing(assessment),
            "assessment": assessment.to_dict(),
            "insufficient_data": False,
        }

    def _format_briefing(self, assessment: FusedHealthAssessment) -> str:
        """Format a human-readable briefing from the assessment."""
        status_emoji = {
            "healthy": "🟢",
            "caution": "🟡",
            "degraded": "🟠",
            "critical": "🔴",
        }
        emoji = status_emoji.get(assessment.overall_status, "⚪")

        lines = [
            f"{emoji} **System Health: {assessment.overall_status.upper()}** ({assessment.overall_score:.0%})",
            "",
        ]

        # Battery section
        b = assessment.battery
        if b.regime == "unavailable":
            lines.append("**VSA Engine:** Not running")
        else:
            lines.append(f"**VSA Engine:** {b.regime} ({b.overall_health:.0%} health)")
            if b.overall_health < 0.90:
                lines.append(f"  - Chain integrity: {b.chain_integrity:.0%}")
                lines.append(f"  - Resonator stability: {b.resonator_convergence:.0%}")

        # Agent section
        a = assessment.agent
        if a.regime == "unavailable":
            lines.append("**Agent Monitor:** Not active")
        else:
            lines.append(
                f"**Agent Monitor:** {a.regime} ({a.health_score:.0%} health, {a.total_steps} actions)"
            )
            if a.total_flags > 0 or a.total_blocks > 0:
                lines.append(f"  - Flags: {a.total_flags}, Blocks: {a.total_blocks}")
            if a.context_alignment_trend != "stable":
                lines.append(f"  - Context trend: {a.context_alignment_trend}")

        # Recommendations
        if assessment.recommendations:
            lines.append("")
            lines.append("**Recommendations:**")
            for rec in assessment.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


# ── FastAPI Route Factory ────────────────────────────────────────────


def create_agent_health_route(battery=None):
    """
    Create a FastAPI router for the agent health briefing endpoint.

    Usage in sidecar.py:
        from production.agent_health_briefing import create_agent_health_route
        app.include_router(create_agent_health_route(battery=state.battery))
    """
    from fastapi import APIRouter, Request

    router = APIRouter(tags=["agent-health"])
    generator = AgentHealthBriefingGenerator(battery=battery)

    @router.get("/api/v1/agent/health")
    async def get_agent_health_briefing(request: Request):
        """
        Get a fused health briefing combining VSA engine and agent monitoring.

        Returns overall status, per-system health, and recommendations.
        This is the single-call answer to "am I healthy AND is my data healthy?"
        """
        return generator.generate(request=request)

    @router.get("/api/v1/agent/health/battery")
    async def get_battery_health():
        """Get VSA world model battery health only."""
        if battery:
            return BatterySnapshot.from_battery(battery).__dict__
        return BatterySnapshot.unavailable().__dict__

    @router.get("/api/v1/agent/health/proprioception")
    async def get_proprioception():
        """Get agent proprioceptive state only."""
        return ProprioceptiveSnapshot.from_file().__dict__

    return router
