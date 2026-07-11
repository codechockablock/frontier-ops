"""
Task Coherence Scorer (Signal B) — ADVISORY ONLY (weight=0.00 in verdict)
==========================================================================

Measures global trajectory coherence via Gram matrix analysis of phasor HVs.
Complements fracture_signal() (local discontinuities) with global structure
detection: drift, interleaving, wandering.

Currently zero-weighted in TieredVerdictEngine pending learned encoder.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List

import numpy as np


class TaskCoherenceScorer:
    """Measures task-level coherence of a phasor HV trajectory via 4 sub-signals:
    centroid drift, recurrence asymmetry, spectral concentration, alternation index."""

    # Similarity threshold for counting two composites as "same action pattern".
    # In phasor dim=512, cos > 0.5 implies all 6 slots are very close.
    # This is deliberately high — we only count near-identical full action states.
    RECURRENCE_THRESHOLD = 0.50

    # Minimum steps before producing a meaningful coherence score.
    # Below this, we output coherence=1.0 (assume coherent during warmup).
    MIN_STEPS = 7

    def __init__(self, window: int = 20, dim: int = 512):
        self.window = window
        self.dim = dim
        self.hvs: deque = deque(maxlen=window)

    def push(self, hv: np.ndarray) -> None:
        """Accept a normalized composite phasor HV."""
        if hv.shape != (self.dim,):
            return
        norm = np.linalg.norm(hv)
        if norm < 1e-9:
            return
        self.hvs.append(hv / norm)

    def clear(self) -> None:
        """Reset for a new session."""
        self.hvs.clear()

    # ── Gram Matrix ─────────────────────────────────────────────────────

    @staticmethod
    def _phasor_cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Re(a† · b) for unit-normalized phasor vectors."""
        return float(np.real(np.vdot(a, b)))

    def _gram_matrix(self, hvs: List[np.ndarray]) -> np.ndarray:
        """
        Compute the W×W phasor cosine Gram matrix.

        G[i,j] = Re(hᵢ† · hⱼ)

        For W ≤ 20 this is 400 vdot operations — trivially fast.
        """
        n = len(hvs)
        G = np.zeros((n, n))
        for i in range(n):
            G[i, i] = 1.0
            for j in range(i + 1, n):
                s = self._phasor_cosine(hvs[i], hvs[j])
                G[i, j] = s
                G[j, i] = s
        return G

    # ── Sub-signal 1: Centroid Drift ────────────────────────────────────

    def _centroid_drift(self, hvs: List[np.ndarray]) -> float:
        """Cosine distance between early-third and late-third centroids. 0=stable, 1=drifted."""
        n = len(hvs)
        k = max(2, n // 3)

        c_early = np.sum(hvs[:k], axis=0)
        c_late = np.sum(hvs[-k:], axis=0)

        norm_e = np.linalg.norm(c_early)
        norm_l = np.linalg.norm(c_late)

        if norm_e < 1e-9 or norm_l < 1e-9:
            return 1.0  # degenerate — treat as maximum drift

        cos_sim = float(np.real(np.vdot(c_early / norm_e, c_late / norm_l)))

        # Map from [-1, 1] to drift score.
        # cos_sim > 0 → similar centroids → low drift
        # cos_sim ≈ 0 → orthogonal centroids → high drift
        # cos_sim < 0 → anti-correlated → maximum drift
        drift = 1.0 - max(0.0, cos_sim)
        return float(np.clip(drift, 0.0, 1.0))

    # ── Sub-signal 2: Recurrence Asymmetry ──────────────────────────────

    def _recurrence_asymmetry(self, G: np.ndarray) -> float:
        """Compare recurrence rates between first/second half. 0=stable, 1=collapsed."""
        n = G.shape[0]
        mid = n // 2
        τ = self.RECURRENCE_THRESHOLD

        def _recurrence_rate(indices):
            count = 0
            total = 0
            for a, i in enumerate(indices):
                for b in range(a + 1, len(indices)):
                    j = indices[b]
                    total += 1
                    if G[i, j] > τ:
                        count += 1
            return count / total if total > 0 else 0.0

        first_half = list(range(0, mid))
        second_half = list(range(mid, n))

        r_first = _recurrence_rate(first_half)
        r_second = _recurrence_rate(second_half)

        if r_first < 0.01:
            return 0.0  # no baseline recurrence to compare against

        asymmetry = max(0.0, r_first - r_second) / max(r_first, 0.01)
        return float(np.clip(asymmetry, 0.0, 1.0))

    # ── Sub-signal 3: Spectral Concentration ────────────────────────────

    def _spectral_concentration(self, G: np.ndarray) -> float:
        """1 - normalized entropy of Gram matrix eigenvalues. 1=focused, 0=scattered."""
        n = G.shape[0]
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.maximum(eigvals, 0.0)  # clip numerical noise

        total = eigvals.sum()
        if total < 1e-10:
            return 0.0

        p = eigvals / total
        p = p[p > 1e-10]  # remove zero eigenvalues

        entropy = -float(np.sum(p * np.log(p)))
        max_entropy = math.log(n) if n > 1 else 1.0

        concentration = 1.0 - entropy / max_entropy
        return float(np.clip(concentration, 0.0, 1.0))

    # ── Sub-signal 4: Alternation Index ─────────────────────────────────

    def _alternation_index(self, G: np.ndarray) -> float:
        """Detect periodic interleaving (period 2 or 3). 0=none, 1=strong."""
        n = G.shape[0]
        if n < 6:
            return 0.0

        best_alternation = 0.0

        # Check period-2 (A,B,A,B,...) and period-3 (A,B,C,A,B,C,...)
        for period in (2, 3):
            groups: list = [[] for _ in range(period)]
            for i in range(n):
                groups[i % period].append(i)

            # Skip if any group has fewer than 2 members
            if any(len(g) < 2 for g in groups):
                continue

            # Within-group similarity
            within_sims = []
            for g in groups:
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        within_sims.append(G[g[a], g[b]])

            # Between-group similarity
            between_sims = []
            for g1_idx in range(period):
                for g2_idx in range(g1_idx + 1, period):
                    for i in groups[g1_idx]:
                        for j in groups[g2_idx]:
                            between_sims.append(G[i, j])

            s_within = float(np.mean(within_sims)) if within_sims else 0.0
            s_between = float(np.mean(between_sims)) if between_sims else 0.0

            # Interleaving requires within-group cohesion AND cross-group separation
            if s_within <= s_between or s_within < 0.15:
                continue

            score = (s_within - s_between) / (s_within + 0.01)
            best_alternation = max(best_alternation, score)

        return float(np.clip(best_alternation, 0.0, 1.0))

    # ── Recognition Slope (auxiliary) ───────────────────────────────────

    def _recognition_slope(self, hvs: List[np.ndarray]) -> float:
        """Slope of max-similarity-to-prior series. Negative=diverging, positive=converging."""
        n = len(hvs)
        if n < 4:
            return 0.0

        recognition = []
        for i in range(1, n):
            max_sim = max(self._phasor_cosine(hvs[i], hvs[j]) for j in range(i))
            recognition.append(max_sim)

        if len(recognition) < 3:
            return 0.0

        x = np.arange(len(recognition), dtype=float)
        x = x / max(x[-1], 1.0)
        slope = float(np.polyfit(x, recognition, 1)[0])
        return slope

    # ── Composite Score ─────────────────────────────────────────────────

    def score(self) -> Dict:
        """Compute coherence assessment. Returns dict with coherence, sub-scores, phase."""
        hvs = list(self.hvs)
        n = len(hvs)

        if n < self.MIN_STEPS:
            return {
                "coherence": 1.0,
                "drift_score": 0.0,
                "interleaving_score": 0.0,
                "wandering_score": 0.0,
                "centroid_drift": 0.0,
                "recurrence_asymmetry": 0.0,
                "spectral_concentration": 1.0,
                "recognition_slope": 0.0,
                "alternation_index": 0.0,
                "phase": "warmup",
                "n_steps": n,
            }

        # Compute Gram matrix (O(n²) vdots, n ≤ 20 → trivial)
        G = self._gram_matrix(hvs)

        # Raw sub-signals
        c_drift = self._centroid_drift(hvs)
        rec_asym = self._recurrence_asymmetry(G)
        spec_conc = self._spectral_concentration(G)
        alt_idx = self._alternation_index(G)
        rec_slope = self._recognition_slope(hvs)

        # ── Threat scores ───────────────────────────────────────────

        # Drift threat: centroid_drift > 0.7 = late actions in different phasor region
        if c_drift > 0.70:
            # Strong drift: directly treat as threat, scaled by magnitude
            base_drift = (c_drift - 0.50) / 0.50  # maps 0.5→0, 1.0→1.0
            # Boost with any available confirmation
            confirmation = 0.0
            if rec_slope < -0.03:
                confirmation += min(0.3, -rec_slope)
            if rec_asym > 0.05:
                confirmation += min(0.3, rec_asym)
            drift_score = base_drift * (0.6 + 0.4 * min(1.0, confirmation / 0.3))
        elif c_drift > 0.40 and (rec_slope < -0.10 or rec_asym > 0.20):
            # Moderate drift with confirmation
            drift_score = c_drift * 0.5
        else:
            # Low drift or drift without confirmation = not a threat
            drift_score = 0.0

        # Wandering threat: spectral concentration < 0.15 = suspiciously scattered
        wandering_score = max(0.0, 0.15 - spec_conc) / 0.15 if spec_conc < 0.15 else 0.0
        # Boost if recognition slope is also negative (unfamiliar actions accumulating)
        if rec_slope < -0.1 and spec_conc < 0.25:
            wandering_score = max(wandering_score, (0.25 - spec_conc) / 0.25 * 0.7)

        # Interleaving threat: alternation + low concentration or confirming signals
        # (benign tool cycling has moderate concentration ~0.63, so threshold is 0.45)
        if alt_idx > 0.50 and spec_conc < 0.45:
            # Low concentration + alternation = genuinely different tasks interleaving
            interleaving_score = 1.0 / (1.0 + math.exp(-8.0 * (alt_idx - 0.55)))
        elif alt_idx > 0.50 and (c_drift > 0.30 or rec_slope < -0.05):
            # Alternation with drift/recognition decline = suspicious
            interleaving_score = alt_idx * 0.5
        elif alt_idx > 0.80 and spec_conc >= 0.45:
            interleaving_score = alt_idx * 0.35
        else:
            interleaving_score = 0.0

        # Composite: max(threats) — correlated threats shouldn't double-penalize
        max_threat = max(drift_score, wandering_score, interleaving_score)
        coherence = 1.0 - float(np.clip(max_threat, 0.0, 1.0))

        return {
            "coherence": round(coherence, 4),
            "drift_score": round(float(np.clip(drift_score, 0.0, 1.0)), 4),
            "interleaving_score": round(float(np.clip(interleaving_score, 0.0, 1.0)), 4),
            "wandering_score": round(float(np.clip(wandering_score, 0.0, 1.0)), 4),
            "centroid_drift": round(float(c_drift), 4),
            "recurrence_asymmetry": round(float(rec_asym), 4),
            "spectral_concentration": round(float(spec_conc), 4),
            "recognition_slope": round(float(rec_slope), 4),
            "alternation_index": round(float(alt_idx), 4),
            "phase": "active",
            "n_steps": n,
        }

    def verdict_contribution(self) -> str:
        """Map coherence to PASS/MONITOR/FLAG. <0.25=FLAG, <0.50=MONITOR, else PASS."""
        result = self.score()
        coherence = result["coherence"]
        if result["phase"] == "warmup":
            return "PASS"
        if coherence < 0.25:
            return "FLAG"
        if coherence < 0.50:
            return "MONITOR"
        return "PASS"
