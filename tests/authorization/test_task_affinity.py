"""
Tests for the task-affinity channel (escalate-only directional signal).

CI-safe: uses an injected toy embed_fn everywhere, so nothing here needs
sentence-transformers. The acceptance-critical property is ASYMMETRY:
affinity may add an escalation flag but must never change `authorized`.
"""

import numpy as np
import pytest

import frontier_ops.authorization.task_affinity as task_affinity_module
from frontier_ops.authorization.goal_conditioning import GoalConditioningScorer
from frontier_ops.authorization.task_affinity import (
    AffinityThreshold,
    TaskAffinityChannel,
)
from frontier_ops.boundary.concept_extraction import CONCEPTS

SCOPE_IDX = CONCEPTS.index("scope_exploration")
USER_IDX = CONCEPTS.index("user_aligned_task_execution")


def toy_embed(text: str) -> np.ndarray:
    """Deterministic character-bigram bag embedding (no ML deps)."""
    vec = np.zeros(64)
    text = text.lower()
    for a, b in zip(text, text[1:]):
        vec[(ord(a) * 31 + ord(b)) % 64] += 1.0
    return vec


def _channel(threshold: AffinityThreshold = None) -> TaskAffinityChannel:
    return TaskAffinityChannel(embed_fn=toy_embed, threshold=threshold)


class TestTaskAffinityChannel:
    def test_identical_texts_have_max_affinity(self):
        chan = _channel()
        assert chan.affinity("commit this to git", "commit this to git") == \
            pytest.approx(1.0)

    def test_related_beats_unrelated(self):
        chan = _channel()
        related = chan.affinity("commit this to git", "exec: git commit")
        unrelated = chan.affinity("commit this to git", "xyzzy quux plugh")
        assert related > unrelated

    def test_embeddings_are_cached(self):
        calls = []

        def counting_embed(text):
            calls.append(text)
            return toy_embed(text)

        chan = TaskAffinityChannel(embed_fn=counting_embed)
        chan.affinity("directive", "action one")
        chan.affinity("directive", "action two")
        assert calls.count("directive") == 1

    def test_create_returns_none_without_sentence_transformers(self, monkeypatch):
        monkeypatch.setattr(
            task_affinity_module, "_sentence_transformer_embed_fn",
            lambda *a, **k: None,
        )
        assert TaskAffinityChannel.create() is None
        with pytest.raises(ImportError):
            TaskAffinityChannel()


class TestAffinityThreshold:
    def test_uncalibrated_never_flags(self):
        th = AffinityThreshold()
        assert th.flags(0.0) is None
        assert th.flags(-1.0) is None
        chan = _channel()
        assert chan.flags("a directive", "zzz") is None

    def test_too_few_samples_does_not_calibrate(self):
        th = AffinityThreshold()
        th.calibrate([0.5] * 9, alpha=0.1)
        assert not th.calibrated
        assert th.flags(0.0) is None

    def test_conformal_lower_tail(self):
        th = AffinityThreshold()
        benign = list(np.linspace(0.3, 0.9, 100))
        th.calibrate(benign, alpha=0.1)
        assert th.calibrated
        assert th.calibration_n == 100
        # Threshold sits near the alpha-quantile of benign affinities
        assert 0.3 <= th.threshold <= 0.4
        assert th.flags(0.1) is True
        assert th.flags(0.8) is False
        # Benign false-flag rate is <= ~alpha on the calibration data itself
        rate = float(np.mean([th.flags(a) for a in benign]))
        assert rate <= 0.11


class TestAsymmetry:
    """authorized must be byte-identical with and without the channel."""

    DIRECTIVE = (
        "search every file in the whole directory tree, "
        "recursive scan of all files"
    )

    def _action(self, scope: float) -> np.ndarray:
        vec = np.zeros(len(CONCEPTS))
        vec[USER_IDX] = 0.3
        vec[SCOPE_IDX] = scope
        return vec

    def _scorers(self, threshold=None):
        plain = GoalConditioningScorer(force_tier=1)
        chan = _channel(threshold)
        with_chan = GoalConditioningScorer(force_tier=1, affinity_channel=chan)
        return plain, with_chan

    @pytest.mark.parametrize("scope", [0.1, 0.6, 1.0])
    def test_authorized_identical_with_and_without_channel(self, scope):
        # Calibrated threshold chosen so the toy affinity WILL flag —
        # authorized must still not move.
        th = AffinityThreshold()
        th.calibrate(list(np.linspace(0.95, 0.99, 50)), alpha=0.1)
        plain, with_chan = self._scorers(th)

        action = self._action(scope)
        r_plain = plain.score(self.DIRECTIVE, action)
        r_chan = with_chan.score(
            self.DIRECTIVE, action, action_text="exec: totally unrelated zzz",
        )
        assert r_chan.authorized == r_plain.authorized
        assert r_chan.distance == pytest.approx(r_plain.distance)
        assert r_chan.affinity is not None
        assert r_chan.affinity_flagged is True  # flagged, yet verdict untouched

    def test_high_affinity_never_authorizes(self):
        """Action far outside the radius stays unauthorized even at
        affinity ~1.0 (mimicry cannot unlock anything)."""
        _, with_chan = self._scorers()
        goal = with_chan.extract_goal(self.DIRECTIVE)
        far_action = np.ones(len(CONCEPTS))  # deep in every risk dim
        r = with_chan.score_action(
            goal, far_action, action_text=self.DIRECTIVE,  # perfect mimicry
        )
        assert r.affinity == pytest.approx(1.0)
        assert r.authorized is False

    def test_fields_none_without_channel_or_text(self):
        plain, with_chan = self._scorers()
        action = self._action(0.6)
        r = plain.score(self.DIRECTIVE, action, action_text="exec: ls")
        assert r.affinity is None and r.affinity_flagged is None
        r = with_chan.score(self.DIRECTIVE, action)  # no action_text
        assert r.affinity is None and r.affinity_flagged is None

    def test_uncalibrated_channel_reports_affinity_but_no_flag(self):
        _, with_chan = self._scorers()
        r = with_chan.score(
            self.DIRECTIVE, self._action(0.6), action_text="exec: ls",
        )
        assert r.affinity is not None
        assert r.affinity_flagged is None
