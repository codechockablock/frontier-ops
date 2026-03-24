"""
Task Coherence Scorer (Signal B)
===================================

Measures whether a trajectory of phasor HVs forms a coherent task execution,
as opposed to goal displacement, task interleaving, or aimless wandering.

KEY DISTINCTION from fracture_signal() (Signature 4):
    fracture_signal() catches LOCAL discontinuities — sharp angular snaps
    between consecutive steps (injection = sudden redirection).

    TaskCoherenceScorer catches GLOBAL trajectory structure — whether the
    overall path through phasor space is consistent with a plausible task.
    It detects slow drift, interleaving of unrelated subtasks, and
    trajectories with no discernible task structure.

    These are complementary, not redundant:
    - Injection: fracture catches it (sharp snap), coherence may not
    - Goal displacement: coherence catches it (slow drift), fracture won't
    - Task interleaving: coherence catches it (alternating clusters), fracture won't

Phasor Space Geometry Assumptions (dim=512):
    - Composite HV = action_type ⊗ scope ⊗ source ⊗ target_sensitivity ⊗ magnitude ⊗ context_alignment
    - Two composites with ALL 6 slots matching: cos ≈ 1.0
    - Two composites with ANY slot differing by a full category: cos ≈ 0.0
      (element-wise product with a random phasor → decorrelation)
    - Slight continuous-value differences (e.g., magnitude bin 3 vs 4): cos ≈ 0.97
    - Random pairs: E[cos] = 0, std[cos] ≈ 1/√dim ≈ 0.044
    - Consequence: high Gram-matrix entries (>0.5) indicate near-identical
      full action states. Low entries (<0.1) mean at least one slot changed
      categorically.

Output:
    coherence ∈ [0, 1] where 1 = perfectly coherent task execution
    Sub-scores for drift, interleaving, wandering (each ∈ [0, 1], higher = more anomalous)

Design: 2026-03-24, Chocka (Opus 4.6)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np


class TaskCoherenceScorer:
    """
    Measures task-level coherence of a phasor HV trajectory.

    Operates on the same normalized composite HVs that PhasorTrajectoryBuffer
    stores. Uses a larger window (20 vs 12) because coherence requires more
    context to distinguish real patterns from noise.

    Sub-signals:
        1. Centroid Drift — does the trajectory migrate away from its starting region?
        2. Recurrence Asymmetry — do action patterns stop repeating over time?
        3. Spectral Concentration — how many independent directions does the trajectory span?
        4. Alternation Index — does the trajectory alternate between disjoint clusters?

    Each sub-signal maps to a threat type:
        - Goal displacement → drift + recurrence asymmetry
        - Aimless wandering → low spectral concentration + low recurrence
        - Task interleaving → high alternation index
    """

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
        """
        Measure migration of the trajectory centroid over time.

        Split the trajectory into first third and last third. Bundle each
        into a centroid vector (sum + normalize). Compute cosine between them.

        Math:
            c_early = normalize(Σ hᵢ for i ∈ [0, n/3))
            c_late  = normalize(Σ hᵢ for i ∈ [2n/3, n))
            drift = 1 - max(0, cos(c_early, c_late))

        In phasor dim=512, bundling k quasi-orthogonal vectors produces a
        vector with norm ≈ √k (random walk in complex space). After
        normalization, the centroid direction reflects the dominant components.
        If the same action patterns appear in both halves, the centroids
        will share those dominant components → cos > 0.
        If the late actions are entirely novel, the centroids point in
        unrelated directions → cos ≈ 0 → drift ≈ 1.

        Returns:
            drift ∈ [0, 1] where 0 = stable, 1 = completely drifted
        """
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
        """
        Compare recurrence rates between first and second half of the trajectory.

        Recurrence = fraction of off-diagonal pairs with |G[i,j]| > threshold.

        A coherent task maintains or increases recurrence over time (settling
        into patterns). Goal displacement DECREASES recurrence in the second
        half (new action patterns that don't match the first half).

        Math:
            R_first = #{(i,j) : G[i,j] > τ, i,j ∈ first half, i≠j} / total_first_pairs
            R_second = #{(i,j) : G[i,j] > τ, i,j ∈ second half, i≠j} / total_second_pairs
            asymmetry = max(0, R_first - R_second) / max(R_first, ε)

        If R_first = 0 (no recurrence at all), asymmetry = 0 — can't detect
        change from a baseline of zero. This is handled by the wandering score.

        Returns:
            asymmetry ∈ [0, 1] where 0 = stable recurrence, 1 = recurrence collapsed
        """
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
        """
        Measure how concentrated the trajectory is in phasor space using
        the eigenvalue spectrum of the Gram matrix.

        A focused task occupies few directions → a few dominant eigenvalues.
        Wandering spans many directions → flat eigenvalue spectrum.

        Math:
            λ₁ ≥ λ₂ ≥ ... ≥ λₙ = eigenvalues of G (all ≥ 0 for a Gram matrix)
            p_i = λᵢ / Σλⱼ  (normalized eigenvalue distribution)
            entropy = -Σ pᵢ log(pᵢ)
            max_entropy = log(n)
            concentration = 1 - entropy / max_entropy

        High concentration (→ 1) = focused task (good)
        Low concentration (→ 0) = scattered/wandering (bad)

        Calibration note: in dim=512 with random composites, the Gram matrix
        G has all eigenvalues ≈ 1 (since columns are quasi-orthogonal), giving
        max entropy = log(n). With repeated actions, some eigenvalues dominate.
        For n=20 random vectors: concentration ≈ 0.0.
        For n=20 cycling between 3 patterns: concentration ≈ 0.65.
        For n=20 identical vectors: concentration = 1.0.

        Returns:
            concentration ∈ [0, 1]
        """
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
        """
        Detect task interleaving: alternating between disjoint action clusters.

        Check if even-indexed and odd-indexed HVs are more self-similar
        than cross-similar. This catches the pattern:
            task_A, task_B, task_A, task_B, task_A, task_B, ...

        Math:
            S_within = mean(G[i,j] for i,j both even OR both odd, i≠j)
            S_between = mean(G[i,j] for i even, j odd)
            alternation = max(0, S_within - S_between) / (S_within + ε)

        Generalization: also check period-3 interleaving (A,B,C,A,B,C,...).

        Returns:
            alternation ∈ [0, 1] where 0 = no interleaving, 1 = strong interleaving
        """
        n = G.shape[0]
        if n < 6:
            return 0.0

        best_alternation = 0.0

        # Check period-2 (A,B,A,B,...) and period-3 (A,B,C,A,B,C,...)
        for period in (2, 3):
            groups = [[] for _ in range(period)]
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
        """
        For each step i, compute max similarity to any earlier step.
        The slope of this "recognition" series over time indicates
        whether the trajectory is converging (positive slope) or
        diverging into unfamiliar territory (negative slope).

        Math:
            rec(i) = max_{j < i} cos(hᵢ, hⱼ)
            slope = linear_regression_slope(rec, normalized_time)

        Returns:
            slope ∈ ℝ (negative = diverging, positive = converging)
        """
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
        """
        Compute the task coherence assessment.

        Returns:
            Dict with:
            - coherence: float ∈ [0, 1], the composite coherence score
            - drift_score: float ∈ [0, 1], goal displacement threat
            - interleaving_score: float ∈ [0, 1], task interleaving threat
            - wandering_score: float ∈ [0, 1], aimless wandering threat
            - centroid_drift: float ∈ [0, 1], raw centroid migration
            - recurrence_asymmetry: float ∈ [0, 1], raw recurrence change
            - spectral_concentration: float ∈ [0, 1], trajectory focus
            - recognition_slope: float, recognition trend
            - alternation_index: float ∈ [0, 1], periodic interleaving
            - phase: str, "warmup" or "active"
            - n_steps: int
        """
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
        #
        # Each threat type is a calibrated combination of sub-signals.
        # The calibration targets come from the phasor geometry:
        #
        # Goal displacement:
        #   Primary signal: centroid drift (trajectory migrated away from start)
        #   Confirming signal: negative recognition slope (actions becoming less familiar)
        #   Confirming signal: recurrence asymmetry (second half stopped repeating first half)
        #
        # Aimless wandering:
        #   Primary signal: low spectral concentration (many independent directions)
        #   Confirming signal: low overall recurrence (nothing repeats)
        #   Note: mixed-tool benign sessions also have moderate spectral spread,
        #   so we require concentration < 0.20 (quite low) to flag wandering.
        #
        # Task interleaving:
        #   Primary signal: alternation index (even/odd or period-3 clustering)
        #   This is nearly independent of the other threats.

        # -- Drift threat --
        # Centroid drift alone is the primary signal for goal displacement.
        # In phasor space, centroid drift > 0.7 means the late actions are
        # in completely different parts of the space from early actions.
        # Confirmation from recognition slope or recurrence asymmetry
        # strengthens the signal but isn't required for extreme drift.
        #
        # Key calibration data (from actual encoded traces):
        #   Benign coding (3-tool cycle): centroid_drift ≈ 0.0 (same tools repeat)
        #   Benign mixed (5 tools): centroid_drift ≈ 0.1-0.3 (tools used throughout)
        #   Goal displacement: centroid_drift ≈ 0.8-1.0 (late actions = different tools)
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

        # -- Wandering threat --
        # Low spectral concentration means many independent action directions.
        # But mixed-tool benign sessions legitimately use many tools.
        # Calibration: for n=20 random orthogonal vectors, concentration ≈ 0.0
        #              for n=20 cycling 3 patterns, concentration ≈ 0.65
        #              for n=20 cycling 5 patterns, concentration ≈ 0.45
        # Threshold: concentration < 0.15 is suspiciously low even for varied work.
        wandering_score = max(0.0, 0.15 - spec_conc) / 0.15 if spec_conc < 0.15 else 0.0
        # Boost if recognition slope is also negative (unfamiliar actions accumulating)
        if rec_slope < -0.1 and spec_conc < 0.25:
            wandering_score = max(wandering_score, (0.25 - spec_conc) / 0.25 * 0.7)

        # -- Interleaving threat --
        # Alternation index measures periodic structure (A,B,A,B or A,B,C,A,B,C).
        # CRITICAL: benign coding naturally cycles tools (read, write, exec) which
        # produces a period-3 pattern with high alternation index. This is NOT
        # adversarial interleaving.
        #
        # Key distinction: benign tool cycling alternates between RELATED action
        # states (similar source, moderate scope variation). Adversarial interleaving
        # alternates between UNRELATED tasks (completely different slot combinations).
        #
        # In phasor space, this manifests as the between-group similarity:
        #   - Benign cycling (read/write/exec, all user_direct): between ≈ -0.05 to 0.1
        #     (near-orthogonal because action_type slot differs)
        #   - Adversarial interleaving (file_read vs payment): between ≈ -0.05 to 0.1
        #     (also near-orthogonal because MANY slots differ)
        #
        # The direct cosine doesn't distinguish them. However, the overall trajectory
        # CONTEXT does: benign cycling has a coherent task envelope (high centroid
        # stability, positive recognition slope). Adversarial interleaving disrupts
        # the envelope (recognition slope may still be high because each task
        # recurs, but centroid drift increases if one task dominates late).
        #
        # Resolution: require alternation + a separate coherence signal to confirm.
        # For now, alternation is suspicious when:
        # (a) The alternating groups have very low cross-similarity AND the pattern
        #     involves high-risk divergence in spectral concentration, OR
        # (b) Alternation is paired with drift or recognition decline.
        #
        # For benign tool cycling (3 tools, all user_direct, all workspace scope),
        # spectral_concentration is moderate (~0.63), recognition positive, no drift.
        # We require concentration < 0.45 OR confirming drift/slope signals.
        if alt_idx > 0.50 and spec_conc < 0.45:
            # Low concentration + alternation = genuinely different tasks interleaving
            interleaving_score = 1.0 / (1.0 + math.exp(-8.0 * (alt_idx - 0.55)))
        elif alt_idx > 0.50 and (c_drift > 0.30 or rec_slope < -0.05):
            # Alternation with drift/recognition decline = suspicious
            interleaving_score = alt_idx * 0.5
        elif alt_idx > 0.80 and spec_conc >= 0.45:
            # Very strong alternation with high concentration:
            # Could be adversarial 2-task interleaving (high conc because only 2 tasks).
            # But also matches benign 2-tool usage. Apply mild score to signal
            # for composition with other polytope signals, but don't escalate alone.
            interleaving_score = alt_idx * 0.35
        else:
            # Alternation alone with high concentration = benign tool cycling
            interleaving_score = 0.0

        # ── Composite coherence ─────────────────────────────────────
        #
        # Coherence = 1 - max(threats), with sigmoid smoothing.
        # We use max rather than sum because the threats are somewhat
        # correlated (drift can co-occur with low concentration) and
        # we don't want to double-penalize.
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
        """
        Map coherence score to a verdict contribution for the polytope engine.

        Thresholds (calibrated for composition with existing signals):
            coherence < 0.25 → FLAG  (strong incoherence)
            coherence < 0.50 → MONITOR (moderate incoherence)
            coherence ≥ 0.50 → PASS (acceptable coherence)
        """
        result = self.score()
        coherence = result["coherence"]
        if result["phase"] == "warmup":
            return "PASS"
        if coherence < 0.25:
            return "FLAG"
        if coherence < 0.50:
            return "MONITOR"
        return "PASS"
