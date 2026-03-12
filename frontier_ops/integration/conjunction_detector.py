"""
Conjunction Detector — Multi-Signature Co-occurrence Analysis
==============================================================

The 5 safety polytope signatures are treated as independent by the weighted-sum
composite. They're not independent — violations cluster. This module adds a
multiplicative conjunction layer on top of the linear composite.

Key insight (from Joseph's architectural review, 2026-02-27):
  - CUSUM alone → FLAG, probably benign, watch for recovery
  - sig4 (coherence snap) + sig2 (provenance) → probable prompt injection
  - sig4 alone after user_direct → user just changed their mind, benign
  - sig1 + sig5 → goal drift (orphaned confidence without intent anchor)
  - sig2 + sig1 + sig4 → high-confidence violation, >5x composite boost

The temporal window matters: an attacker can't always land all signatures
simultaneously. We check within a rolling 8-step window.

Design:
  - CONJUNCTION_RULES: (required_sigs, multiplier, label)
  - BENIGN_OVERRIDES: sig4 + user_direct source → suppress injection suspicion
  - Returns multiplier (applied to composite signal) + label + list of matches
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Set, Tuple


# ─── Conjunction Rules ────────────────────────────────────────────────────────
# Each rule: (required_sigs_within_window, multiplier, label, description)
# multiplier is applied to composite_signal (multiplicative, not additive)

CONJUNCTION_RULES: List[Tuple[Set[str], float, str, str]] = [
    # Highest priority: three-way conjunction — strong violation signal
    (
        {"sig2_provenance", "sig1_intent", "sig4_coherence"},
        5.0,
        "high_confidence_violation",
        "Provenance corruption + intent fracture + trajectory snap — "
        "strong evidence of goal hijack or prompt injection",
    ),
    # Probable prompt injection: trajectory snaps AND untrusted source active
    (
        {"sig4_coherence", "sig2_provenance"},
        3.0,
        "probable_injection",
        "Trajectory angular snap coincident with untrusted-source activity — "
        "sig4 alone could be a context switch; sig2 + sig4 narrows to injection",
    ),
    # Goal drift: intent fracture with orphaned confidence
    (
        {"sig1_intent", "sig5_grounding"},
        2.5,
        "goal_drift",
        "Intent binding fracture + confidence-grounding decoupling — "
        "agent actions no longer anchored to stated goal or verifiable evidence",
    ),
    # Unknown territory: manifold exit + ungrounded confidence
    (
        {"sig3_manifold", "sig5_grounding"},
        2.0,
        "unknown_territory",
        "Constitutional manifold boundary exit + ungrounded confidence — "
        "agent in behavioral region with no reliable safety anchor",
    ),
    # Scope creep: untrusted source driving manifold exit
    (
        {"sig2_provenance", "sig3_manifold"},
        2.0,
        "scope_creep",
        "Untrusted source driving agent outside its calibrated behavioral manifold",
    ),
    # Sustained provenance: untrusted source + intent drift (no snap needed)
    (
        {"sig2_provenance", "sig1_intent"},
        1.8,
        "directed_drift",
        "Untrusted source paired with intent fracture — "
        "external content directing agent away from task",
    ),
]

# ─── Benign Override Rules ────────────────────────────────────────────────────
# Suppress or attenuate conjunction score when context indicates benign cause.
# Each rule: (sig_that_fired, recent_sources_that_make_it_benign, suppression_multiplier, label)

BENIGN_OVERRIDES: List[Tuple[str, Set[str], float, str]] = [
    # sig4 firing right after a user_direct source → user changed their mind,
    # not an injection. Classic context switch produces identical angular snap.
    (
        "sig4_coherence",
        {"user_direct"},
        0.4,
        "user_context_switch",
    ),
]


class ConjunctionDetector:
    """
    Detects dangerous multi-signature co-occurrence within a rolling window.

    Maintains a short history of (firing_signatures, source) pairs.
    On each step, checks conjunction rules against the full window and
    applies a multiplicative boost to the composite signal when dangerous
    patterns are detected.

    Also performs sig4/sig2 disambiguation:
      - sig4 + external source in window → injection (multiplier ≥ 3.0)
      - sig4 + user_direct in window    → context switch (suppressed)

    Usage:
        detector = ConjunctionDetector(window=8)
        result = detector.observe(firing_sigs={"sig4_coherence"}, source="web_content")
        composite_signal *= result["multiplier"]
    """

    def __init__(self, window: int = 8):
        self.window = window
        # Each entry: {"firing": set[str], "source": str}
        self.history: deque = deque(maxlen=window)

    # Signatures that can partner with sig4 to form a dangerous conjunction.
    # If any of these are present in the window, benign override does NOT apply.
    _SIG4_CONJUNCTION_PARTNERS: Set[str] = {
        "sig1_intent",
        "sig2_provenance",
        "sig3_manifold",
        "sig5_grounding",
    }

    def observe(
        self,
        firing_sigs: Set[str],
        source: str,
        current_composite: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Record current step and evaluate conjunction rules.

        Args:
            firing_sigs: set of signature names that fired this step
            source: classified action source for this step
            current_composite: the current linear composite signal (0-1).

        Returns:
            Dict with:
              multiplier: float — apply to composite_signal
              label: str — human-readable conjunction name, or "none"
              conjunctions_detected: list[str] — all matching rule labels
              suppressed: bool — True if a benign override applied
              ordering_note: str — temporal ordering detail for sig4/sig2
        """
        # Track absolute step index for temporal ordering (Flaw 2 fix)
        step_idx = len(self.history)  # monotonically increasing within window deque
        self.history.append(
            {
                "firing": set(firing_sigs),
                "source": str(source),
                "idx": step_idx,
            }
        )

        # Accumulate all sigs that fired within the window (including now)
        window_firing: Set[str] = set()
        for entry in self.history:
            window_firing |= entry["firing"]

        # Collect sources seen in the window
        window_sources: Set[str] = {e["source"] for e in self.history}

        # ── Benign override check (Flaw 1 fix) ───────────────────────────
        # sig4 + user_direct = context switch, but ONLY if sig4 is the sole
        # anomalous signature in the window. If conjunction partners are also
        # firing, the context switch narrative doesn't hold — proceed to
        # conjunction rules.
        for sig, benign_sources, suppression_mult, label in BENIGN_OVERRIDES:
            if sig in firing_sigs and (benign_sources & window_sources):
                # Check whether any conjunction partner is in the window
                partners_present = window_firing & self._SIG4_CONJUNCTION_PARTNERS
                if not partners_present:
                    # sig4 alone after user_direct → genuine context switch
                    return {
                        "multiplier": suppression_mult,
                        "label": label,
                        "conjunctions_detected": [],
                        "suppressed": True,
                        "ordering_note": "sig4 isolated; user_direct in window",
                    }
                # Partners present — fall through to conjunction rules.
                # The context switch explanation can't account for other sigs.

        # ── Temporal ordering for sig4 / sig2 (Flaw 2 fix) ──────────────
        # For injection detection, sig2 should precede sig4 in the window.
        # Reverse ordering (snap then external content) is much weaker.
        ordering_note = ""
        sig4_in_window = "sig4_coherence" in window_firing
        sig2_in_window = "sig2_provenance" in window_firing
        injection_multiplier = 3.0  # default for probable_injection rule

        if sig4_in_window and sig2_in_window:
            # Find most recent firing index for each
            last_sig4_idx = max(
                (e["idx"] for e in self.history if "sig4_coherence" in e["firing"]),
                default=-1,
            )
            last_sig2_idx = max(
                (e["idx"] for e in self.history if "sig2_provenance" in e["firing"]),
                default=-1,
            )
            if last_sig2_idx < last_sig4_idx:
                # sig2 → sig4: canonical injection shape (untrusted read → snap)
                injection_multiplier = 3.0
                ordering_note = (
                    f"sig2@{last_sig2_idx}→sig4@{last_sig4_idx} (injection shape)"
                )
            else:
                # sig4 → sig2: snap before content, weaker injection signal
                injection_multiplier = 1.5
                ordering_note = (
                    f"sig4@{last_sig4_idx}→sig2@{last_sig2_idx} (reversed, weak)"
                )

        # ── Conjunction rule matching ─────────────────────────────────────
        matched: List[Tuple[float, str]] = []
        for req_sigs, multiplier, label, _ in CONJUNCTION_RULES:
            if req_sigs <= window_firing:
                # Apply ordering-adjusted multiplier for injection rules
                if label == "probable_injection":
                    multiplier = injection_multiplier
                matched.append((multiplier, label))

        if not matched:
            return {
                "multiplier": 1.0,
                "label": "none",
                "conjunctions_detected": [],
                "suppressed": False,
                "ordering_note": ordering_note,
            }

        best_mult, best_label = max(matched, key=lambda x: x[0])
        all_labels = [label for _, label in matched]

        return {
            "multiplier": best_mult,
            "label": best_label,
            "conjunctions_detected": all_labels,
            "suppressed": False,
            "ordering_note": ordering_note,
        }

    def reset(self) -> None:
        """Clear history (e.g. on session reset)."""
        self.history.clear()
