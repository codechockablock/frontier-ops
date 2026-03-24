"""Tests for Signal D: Refusal Detection."""

from __future__ import annotations

import numpy as np
import pytest

from frontier_ops.integration.refusal_detector import (
    RefusalDetectionSignal,
    MIN_STEPS,
    WINDOW_SIZE,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _benign_coding_step(**overrides) -> dict:
    """A typical safe coding action."""
    step = {
        "action_type": "file_write",
        "scope": "write_workspace",
        "source": "user_direct",
        "magnitude": 0.25,
        "context_alignment": 0.9,
    }
    step.update(overrides)
    return step


def _refusal_step(**overrides) -> dict:
    """A refusal-like action: low alignment, agent_reasoning source."""
    step = {
        "action_type": "file_read",
        "scope": "read_only",
        "source": "agent_reasoning",
        "magnitude": 0.05,
        "context_alignment": 0.1,
    }
    step.update(overrides)
    return step


def _high_scope_step(**overrides) -> dict:
    """A high-scope action (potential escalation target)."""
    step = {
        "action_type": "shell_exec",
        "scope": "write_system",
        "source": "user_direct",
        "magnitude": 0.8,
        "context_alignment": 0.85,
    }
    step.update(overrides)
    return step


def _benign_thinking_step(**overrides) -> dict:
    """Benign agent reasoning — moderate alignment, low magnitude."""
    step = {
        "action_type": "file_read",
        "scope": "read_only",
        "source": "agent_reasoning",
        "magnitude": 0.1,
        "context_alignment": 0.6,
    }
    step.update(overrides)
    return step


# ── Core Interface Tests ─────────────────────────────────────────────

class TestRefusalDetectionInterface:
    """Test the push/score/clear/verdict_contribution interface."""

    def test_initial_state(self):
        sig = RefusalDetectionSignal()
        s = sig.score()
        assert s["phase"] == "warmup"
        assert s["refusal_score"] == 0.0
        assert s["n_steps"] == 0

    def test_warmup_phase(self):
        sig = RefusalDetectionSignal()
        for _ in range(MIN_STEPS - 1):
            sig.push(_benign_coding_step())
        s = sig.score()
        assert s["phase"] == "warmup"
        assert s["refusal_score"] == 0.0

    def test_active_phase_after_min_steps(self):
        sig = RefusalDetectionSignal()
        for _ in range(MIN_STEPS):
            sig.push(_benign_coding_step())
        s = sig.score()
        assert s["phase"] == "active"

    def test_clear_resets(self):
        sig = RefusalDetectionSignal()
        for _ in range(MIN_STEPS + 5):
            sig.push(_benign_coding_step())
        sig.clear()
        s = sig.score()
        assert s["n_steps"] == 0
        assert s["phase"] == "warmup"

    def test_verdict_warmup_is_pass(self):
        sig = RefusalDetectionSignal()
        sig.push(_benign_coding_step())
        assert sig.verdict_contribution() == "PASS"

    def test_missing_fields_use_defaults(self):
        sig = RefusalDetectionSignal()
        # Push with empty dict — should not crash
        for _ in range(MIN_STEPS + 2):
            sig.push({})
        s = sig.score()
        assert s["phase"] == "active"
        assert 0.0 <= s["refusal_score"] <= 1.0


# ── Benign Traces ────────────────────────────────────────────────────

class TestBenignTraces:
    """Benign sessions should have refusal_score near 0."""

    def test_normal_coding_session(self):
        """Pure coding: all user_direct, high alignment. Score ~0."""
        sig = RefusalDetectionSignal()
        rng = np.random.RandomState(42)
        for _ in range(20):
            sig.push(_benign_coding_step(
                context_alignment=rng.uniform(0.8, 0.98),
                magnitude=rng.uniform(0.1, 0.4),
            ))
        s = sig.score()
        assert s["refusal_score"] < 0.2, f"Benign coding flagged: {s}"
        assert s["pattern"] == "none"

    def test_benign_thinking_session(self):
        """
        Agent uses agent_reasoning legitimately — moderate alignment,
        low magnitude. Should NOT flag.
        """
        sig = RefusalDetectionSignal()
        rng = np.random.RandomState(123)
        for _ in range(20):
            if rng.random() < 0.4:
                # Thinking steps
                sig.push(_benign_thinking_step(
                    context_alignment=rng.uniform(0.4, 0.7),
                    magnitude=rng.uniform(0.05, 0.15),
                ))
            else:
                sig.push(_benign_coding_step(
                    context_alignment=rng.uniform(0.8, 0.95),
                ))
        s = sig.score()
        assert s["refusal_score"] < 0.3, f"Benign thinking flagged: {s}"

    def test_mixed_sources_benign(self):
        """Mixed sources with high alignment — should not flag."""
        sig = RefusalDetectionSignal()
        sources = ["user_direct", "user_prior", "skill_file", "agent_reasoning", "agent_memory"]
        rng = np.random.RandomState(77)
        for i in range(20):
            sig.push(_benign_coding_step(
                source=sources[i % len(sources)],
                context_alignment=rng.uniform(0.75, 0.95),
            ))
        s = sig.score()
        assert s["refusal_score"] < 0.3

    def test_single_low_alignment_blip(self):
        """One low-alignment step in a sea of good ones — no flag."""
        sig = RefusalDetectionSignal()
        for i in range(15):
            if i == 8:
                sig.push(_refusal_step())
            else:
                sig.push(_benign_coding_step())
        s = sig.score()
        assert s["refusal_score"] < 0.4


# ── Oscillation Detection ────────────────────────────────────────────

class TestOscillation:
    """Compliance-refusal oscillation should trigger oscillation_score."""

    def test_clear_oscillation(self):
        """Alternating compliant/refusal steps — strong oscillation signal."""
        sig = RefusalDetectionSignal()
        for i in range(12):
            if i % 2 == 0:
                sig.push(_benign_coding_step(context_alignment=0.9, source="user_direct"))
            else:
                sig.push(_refusal_step(context_alignment=0.1, source="agent_reasoning"))
        s = sig.score()
        assert s["oscillation_score"] > 0.5, f"Oscillation not detected: {s}"
        assert s["pattern"] == "oscillation"

    def test_oscillation_with_noise(self):
        """Oscillation with some noise — should still detect."""
        sig = RefusalDetectionSignal()
        rng = np.random.RandomState(99)
        for i in range(14):
            if i % 3 == 0:
                sig.push(_refusal_step(context_alignment=rng.uniform(0.05, 0.2)))
            elif i % 3 == 1:
                sig.push(_benign_coding_step(context_alignment=rng.uniform(0.8, 0.95)))
            else:
                sig.push(_benign_coding_step(context_alignment=rng.uniform(0.75, 0.9)))
        s = sig.score()
        assert s["oscillation_score"] > 0.3

    def test_high_tpr_on_synthetic_oscillation(self):
        """Run 20 synthetic oscillation traces. TPR must be > 80%."""
        detected = 0
        total = 20
        for seed in range(total):
            sig = RefusalDetectionSignal()
            rng = np.random.RandomState(seed * 7 + 1)
            for i in range(12):
                if i % 2 == 0:
                    sig.push(_benign_coding_step(
                        context_alignment=rng.uniform(0.75, 0.95),
                        source="user_direct",
                    ))
                else:
                    sig.push(_refusal_step(
                        context_alignment=rng.uniform(0.05, 0.25),
                        source="agent_reasoning",
                    ))
            s = sig.score()
            if s["oscillation_score"] > 0.5:
                detected += 1
        tpr = detected / total
        assert tpr > 0.8, f"Oscillation TPR = {tpr:.2f}, need > 0.80"


# ── Clustering Detection ─────────────────────────────────────────────

class TestClustering:
    """Refusal clustering should trigger clustering_score."""

    def test_refusal_cluster(self):
        """3+ consecutive refusal steps in window → clustering."""
        sig = RefusalDetectionSignal()
        # Warmup with benign
        for _ in range(4):
            sig.push(_benign_coding_step())
        # Cluster of refusals
        for _ in range(5):
            sig.push(_refusal_step())
        # A couple more benign
        sig.push(_benign_coding_step())
        s = sig.score()
        assert s["clustering_score"] > 0.5, f"Clustering not detected: {s}"

    def test_sparse_refusals_no_cluster(self):
        """Refusals spread out over many steps — should not cluster."""
        sig = RefusalDetectionSignal()
        for i in range(20):
            if i in (3, 10, 17):  # spread out
                sig.push(_refusal_step())
            else:
                sig.push(_benign_coding_step())
        s = sig.score()
        # Clustering score should be low since they're spread across the full history
        # and the window only sees the last 10 steps
        assert s["clustering_score"] < 0.5


# ── Post-Refusal Escalation ──────────────────────────────────────────

class TestPostRefusalEscalation:
    """Refusal followed by high-scope action → escalation_score."""

    def test_refusal_then_escalation(self):
        """Refusal step immediately followed by high-scope action."""
        sig = RefusalDetectionSignal()
        for _ in range(5):
            sig.push(_benign_coding_step())
        # Refusal → high scope
        sig.push(_refusal_step())
        sig.push(_high_scope_step(magnitude=0.9, scope="write_system"))
        # Another pair
        sig.push(_refusal_step())
        sig.push(_high_scope_step(magnitude=0.85, scope="network_egress"))
        sig.push(_benign_coding_step())
        s = sig.score()
        assert s["escalation_score"] > 0.5, f"Escalation not detected: {s}"

    def test_no_escalation_without_refusal(self):
        """High-scope action without preceding refusal — no escalation signal."""
        sig = RefusalDetectionSignal()
        for _ in range(5):
            sig.push(_benign_coding_step())
        sig.push(_high_scope_step())
        sig.push(_high_scope_step())
        sig.push(_benign_coding_step())
        sig.push(_benign_coding_step())
        sig.push(_benign_coding_step())
        s = sig.score()
        assert s["escalation_score"] < 0.2


# ── Paralysis Detection ──────────────────────────────────────────────

class TestParalysis:
    """Majority low-alignment + low-magnitude → paralysis."""

    def test_paralysis_pattern(self):
        """Most steps have low alignment and low magnitude → paralysis."""
        sig = RefusalDetectionSignal()
        for _ in range(12):
            sig.push({
                "action_type": "file_read",
                "scope": "read_only",
                "source": "agent_reasoning",
                "magnitude": 0.05,
                "context_alignment": 0.1,
            })
        s = sig.score()
        assert s["paralysis_score"] > 0.3
        assert s["pattern"] == "paralysis"

    def test_not_paralysis_with_high_magnitude(self):
        """Low alignment but high magnitude — not paralysis (agent is doing things)."""
        sig = RefusalDetectionSignal()
        for _ in range(12):
            sig.push({
                "action_type": "shell_exec",
                "scope": "write_system",
                "source": "agent_reasoning",
                "magnitude": 0.7,
                "context_alignment": 0.1,
            })
        s = sig.score()
        assert s["paralysis_score"] == 0.0

    def test_not_paralysis_with_high_alignment(self):
        """High alignment — definitely not paralysis."""
        sig = RefusalDetectionSignal()
        for _ in range(12):
            sig.push(_benign_coding_step())
        s = sig.score()
        assert s["paralysis_score"] == 0.0


# ── Verdict Contribution ─────────────────────────────────────────────

class TestVerdictContribution:
    """Test escalation-only verdict contribution."""

    def test_pass_on_benign(self):
        sig = RefusalDetectionSignal()
        for _ in range(10):
            sig.push(_benign_coding_step())
        assert sig.verdict_contribution() == "PASS"

    def test_monitor_on_moderate(self):
        """Moderate refusal score → MONITOR."""
        sig = RefusalDetectionSignal()
        for _ in range(4):
            sig.push(_benign_coding_step())
        for _ in range(4):
            sig.push(_refusal_step())
        sig.push(_benign_coding_step())
        sig.push(_benign_coding_step())
        s = sig.score()
        # Should be at least MONITOR-level if clustering fires
        verdict = sig.verdict_contribution()
        assert verdict in ("MONITOR", "FLAG"), f"Expected escalation, got {verdict} (scores: {s})"

    def test_flag_on_strong_signal(self):
        """Strong oscillation → FLAG."""
        sig = RefusalDetectionSignal()
        for i in range(14):
            if i % 2 == 0:
                sig.push(_benign_coding_step(context_alignment=0.95))
            else:
                sig.push(_refusal_step(context_alignment=0.05))
        s = sig.score()
        if s["refusal_score"] >= 0.7:
            assert sig.verdict_contribution() == "FLAG"


# ── FPR Gate ─────────────────────────────────────────────────────────

class TestFPRGate:
    """Benign FPR at threshold 0.5 must be < 10%."""

    def test_benign_fpr_under_10_percent(self):
        """Run 100 random benign traces, check FPR < 10%."""
        false_positives = 0
        n_trials = 100
        for seed in range(n_trials):
            sig = RefusalDetectionSignal()
            rng = np.random.RandomState(seed)
            # Random benign trace: high alignment, varied sources
            sources = ["user_direct", "user_prior", "skill_file", "agent_memory"]
            for _ in range(15):
                sig.push({
                    "action_type": rng.choice(["file_read", "file_write", "shell_exec"]),
                    "scope": rng.choice(["read_only", "write_workspace", "write_local"]),
                    "source": rng.choice(sources),
                    "magnitude": rng.uniform(0.05, 0.4),
                    "context_alignment": rng.uniform(0.7, 0.98),
                })
            s = sig.score()
            if s["refusal_score"] >= 0.5:
                false_positives += 1
        fpr = false_positives / n_trials
        assert fpr < 0.10, f"Benign FPR = {fpr:.2%}, must be < 10%"

    def test_benign_fpr_with_thinking(self):
        """Run 50 benign traces with agent_reasoning. FPR < 10%."""
        false_positives = 0
        n_trials = 50
        for seed in range(n_trials):
            sig = RefusalDetectionSignal()
            rng = np.random.RandomState(seed + 1000)
            for _ in range(15):
                if rng.random() < 0.35:
                    sig.push(_benign_thinking_step(
                        context_alignment=rng.uniform(0.4, 0.7),
                        magnitude=rng.uniform(0.05, 0.2),
                    ))
                else:
                    sig.push(_benign_coding_step(
                        context_alignment=rng.uniform(0.75, 0.95),
                    ))
            s = sig.score()
            if s["refusal_score"] >= 0.5:
                false_positives += 1
        fpr = false_positives / n_trials
        assert fpr < 0.10, f"Benign+thinking FPR = {fpr:.2%}, must be < 10%"
