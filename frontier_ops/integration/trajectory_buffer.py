"""
Phasor Trajectory Buffer — Signature 4 (Coherence Fracture)
=============================================================

Tracks consecutive-step cosine similarity in the 6-slot phasor VSA
space. Detects prompt injection (sharp angular snap) vs creative
compliance (smooth curve) vs nominal (high coherence).

Works with complex-valued phasor vectors (dim=512, complex128).
Cosine similarity: Re(vdot(a, b)) / (norm(a) * norm(b))
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np


class PhasorTrajectoryBuffer:
    """
    Sliding window of bound 6-slot hypervectors for trajectory analysis.

    Each entry is a normalized phasor HV representing the full action
    state at one timestep. Consecutive similarities reveal coherence
    fractures that signal prompt injection or trajectory hijacking.
    """

    def __init__(self, window: int = 12, dim: int = 512):
        self.window = window
        self.dim = dim
        self.hvs: deque = deque(maxlen=window)

    def push(self, hv: np.ndarray):
        """
        Accept a bound 6-slot hypervector and normalize it.

        The input should be the element-wise product (binding) of all
        6 slot fillers — the composite state vector for this action.
        """
        if hv.shape != (self.dim,):
            return  # wrong shape, skip silently

        norm = np.linalg.norm(hv)
        if norm < 1e-9:
            return  # degenerate vector, skip

        # Normalize and store
        self.hvs.append(hv / norm)

    def clear(self) -> None:
        """Clear the trajectory buffer (call on session reset)."""
        self.hvs.clear()

    @staticmethod
    def phasor_cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity for phasor (complex) vectors.

        Re(a† · b) where a, b are already unit-normalized.
        np.vdot conjugates its first argument, giving a† · b.
        """
        return float(np.real(np.vdot(a, b)))

    def consecutive_similarities(self) -> List[float]:
        """Compute cosine similarity between each consecutive pair."""
        hvs = list(self.hvs)
        return [self.phasor_cosine(hvs[i], hvs[i + 1]) for i in range(len(hvs) - 1)]

    def session_holonomy(
        self,
        budget_autonomous: float = 2.0,
        budget_focused: float = 0.5,
    ) -> Dict:
        """Compute holonomy of the trajectory window in phasor space.

        Holonomy measures how much a reasoning trajectory curved — the
        discrepancy between incremental transport (sum of step-wise phase
        rotations) and direct transport (endpoint rotation from start to
        finish). In flat space they agree; in curved space they diverge.

        Construction (per Claude.ai collaboration, 2026-02-27):
            - Incremental phase: arg(a_i† · a_{i+1}) = np.angle(vdot(a_i, a_{i+1}))
            - Cumulative transport: sum of all incremental phases across window
            - Direct rotation: arg(a_0† · a_{-1})
            - Holonomy: |cumulative_transport - direct_rotation|

        Returns:
            holonomy: scalar curvature accumulation (radians equivalent)
            cumulative_transport: total incremental phase sum
            direct_rotation: endpoint phase angle
            n_steps: number of transitions used
            budget_autonomous: reference budget for autonomous sessions (2.0)
            budget_focused: reference budget for focused single-task sessions (0.5)
            exceeds_autonomous: True if holonomy > budget_autonomous
        """
        hvs = list(self.hvs)
        if len(hvs) < 2:
            return {
                "holonomy": 0.0,
                "cumulative_transport": 0.0,
                "direct_rotation": 0.0,
                "n_steps": 0,
                "budget_autonomous": budget_autonomous,
                "budget_focused": budget_focused,
                "exceeds_autonomous": False,
            }

        # Incremental phase rotations (step-wise tangent angles)
        incremental = [
            float(np.angle(np.vdot(hvs[i], hvs[i + 1]))) for i in range(len(hvs) - 1)
        ]
        cumulative_transport = sum(incremental)

        # Direct rotation from first to last HV
        direct_rotation = float(np.angle(np.vdot(hvs[0], hvs[-1])))

        # Holonomy = discrepancy between cumulative and direct transport
        holonomy = abs(cumulative_transport - direct_rotation)

        return {
            "holonomy": round(holonomy, 4),
            "cumulative_transport": round(cumulative_transport, 4),
            "direct_rotation": round(direct_rotation, 4),
            "n_steps": len(incremental),
            "budget_autonomous": budget_autonomous,
            "budget_focused": budget_focused,
            "exceeds_autonomous": holonomy > budget_autonomous,
        }

    def fracture_signal(self) -> Dict:
        """
        Compute the trajectory coherence fracture signal.

        Returns a dict with:
        - signal: fracture score (spike-to-smoothness ratio)
        - pattern: one of nominal, coherent, injection_fracture,
                   creative_drift, chaotic
        - max_snap: largest angular velocity between consecutive steps
        - mean_coherence: average cosine similarity
        - sim_variance: variance of similarities
        - sims: list of consecutive similarity values
        """
        sims = self.consecutive_similarities()

        if len(sims) < 6:
            return {
                "signal": 0.0,
                "pattern": "insufficient_data",
                "max_snap": None,
                "mean_coherence": None,
                "sim_variance": None,
                "sims": sims,
            }

        # Angular velocity: rate of change in cosine space
        snaps = [abs(sims[i + 1] - sims[i]) for i in range(len(sims) - 1)]
        max_snap = max(snaps)
        mean_sim = float(np.mean(sims))
        sim_variance = float(np.var(sims))

        # Pattern classification — calibrated from real OpenClaw traces (2026-02-28).
        # In phasor VSA with dim=512, different tool types produce near-orthogonal
        # vectors, so consecutive-step similarity for varied tool use is typically
        # 0.0-0.3. This is NORMAL coding behavior, not chaos.
        # The old chaotic threshold (variance > 0.15) fired on every mixed-tool
        # session. Raised to 0.25 and added a mean_sim guard.
        # Injection fracture: one sharp spike that is an OUTLIER relative to
        # the session's snap distribution. In mixed-tool sessions, every tool
        # type switch produces max_snap ~1.0 — that's normal, not injection.
        # Injection = sharp snap when the session was previously stable.
        #
        # Detection: max_snap must exceed mean + 2*std of all snaps, AND
        # the baseline must be stable (mean_snap < 0.40). This ensures we
        # only flag sudden departures from an established low-snap norm.
        mean_snap = float(np.mean(snaps))
        std_snap = float(np.std(snaps)) if len(snaps) >= 3 else 0.5
        snap_is_outlier = max_snap > mean_snap + 2 * std_snap
        # Injection = sharp snap that's abnormal for this session.
        # Two routes: (1) stable baseline (mean_snap < 0.40) disrupted by
        # one big snap, OR (2) statistical outlier regardless of baseline.
        # Both require max_snap > 0.80 as a minimum bar.
        if max_snap > 0.80 and (mean_snap < 0.40 or snap_is_outlier):
            pattern = "injection_fracture"  # sharp angular snap amid stability
        elif max_snap < 0.15 and mean_sim > 0.75:
            pattern = "coherent"  # everything is smooth and aligned
        elif mean_sim < 0.60 and sim_variance < 0.12:
            pattern = "creative_drift"  # smooth curve or varied tools, acceptable
        elif sim_variance > 0.25 and mean_sim > 0.50:
            pattern = "chaotic"  # high variance in what should be similar actions
        else:
            pattern = "nominal"

        # Fracture score: normalized spike-to-smoothness ratio in [0, 1].
        # Raw ratio = max_snap / (sim_variance + floor).
        # Then sigmoid-map to [0, 1] so it composes cleanly with other
        # signals in the polytope's weighted average.
        #
        # Calibration (2026-02-28 from real traces):
        #   Normal tool switch: max_snap=0.2, var=0.02 → raw=1.7 → sig≈0.15
        #   Creative drift:     max_snap=0.4, var=0.08 → raw=2.2 → sig≈0.25
        #   Injection fracture: max_snap=0.9, var=0.03 → raw=6.9 → sig≈0.90
        raw_ratio = max_snap / (sim_variance + 0.10)
        # Sigmoid: 1 / (1 + exp(-k*(x - midpoint)))
        # midpoint=4.0 means raw_ratio=4 maps to 0.5; slope k=1.2
        import math
        fracture_score = 1.0 / (1.0 + math.exp(-1.2 * (raw_ratio - 4.0)))

        return {
            "signal": round(fracture_score, 4),
            "pattern": pattern,
            "max_snap": round(max_snap, 4),
            "mean_coherence": round(mean_sim, 4),
            "sim_variance": round(sim_variance, 4),
            "sims": [round(s, 4) for s in sims],
        }

    def escalation_check(self, base_verdict: str) -> tuple:
        """
        Signature 4 can only escalate, never de-escalate.

        Returns (possibly_escalated_verdict, fracture_info).
        """
        fracture = self.fracture_signal()

        if fracture["pattern"] == "injection_fracture":
            # Escalate PASS → MONITOR, MONITOR → FLAG
            escalation_map = {
                "PASS": "MONITOR",
                "MONITOR": "FLAG",
                "FLAG": "FLAG",
                "BLOCK": "BLOCK",
            }
            new_verdict = escalation_map.get(base_verdict, base_verdict)
            if new_verdict != base_verdict:
                fracture["escalated_from"] = base_verdict
            return new_verdict, fracture

        return base_verdict, fracture


def bind_slot_vectors(slots: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Bind slot filler vectors into a single composite hypervector.

    Element-wise complex multiplication for phasor VSA.
    Supports 6-slot (legacy) and 7-slot (with semantic) encodings.
    Order: action_type ⊗ scope ⊗ source ⊗ target_sensitivity ⊗ magnitude ⊗ context_alignment [⊗ semantic]
    """
    slot_order = [
        "action_type",
        "scope",
        "source",
        "target_sensitivity",
        "magnitude",
        "context_alignment",
        "semantic",
    ]

    result = None
    for key in slot_order:
        if key in slots:
            if result is None:
                result = slots[key].copy()
            else:
                result = result * slots[key]  # element-wise complex multiply

    if result is None:
        raise ValueError("No slot vectors provided")

    # Normalize to unit phasor
    norm = np.linalg.norm(result)
    if norm > 1e-9:
        result = result / norm

    return result
