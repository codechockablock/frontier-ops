"""RollingThreshold — adaptive benign-quantile thresholding.

The load-bearing test injects a +1σ mean shift into a benign score stream
and checks that the rolling threshold keeps realized FPR near alpha while a
threshold frozen pre-shift drifts past alpha + 0.1 (the transport failure
measured in eval/results/calibration-transport-2026-07-04.md).
"""

import numpy as np
import pytest

from frontier_ops.adaptive_threshold import RollingThreshold


class TestValidation:
    def test_alpha_bounds(self):
        with pytest.raises(ValueError, match="alpha"):
            RollingThreshold(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            RollingThreshold(alpha=1.0)

    def test_window_bounds(self):
        with pytest.raises(ValueError, match="window"):
            RollingThreshold(window=1)

    def test_decay_bounds(self):
        with pytest.raises(ValueError, match="decay"):
            RollingThreshold(decay=0.0)
        with pytest.raises(ValueError, match="decay"):
            RollingThreshold(decay=1.5)

    def test_min_n_bounds(self):
        with pytest.raises(ValueError, match="min_n"):
            RollingThreshold(min_n=1)


class TestWarmup:
    def test_not_ready_before_min_n(self):
        rt = RollingThreshold(alpha=0.1, min_n=5)
        for s in [0.1, 0.2, 0.3, 0.4]:
            rt.update(s)
        assert not rt.ready
        with pytest.raises(RuntimeError, match="warming up"):
            _ = rt.threshold

    def test_ready_at_min_n(self):
        rt = RollingThreshold(alpha=0.1, min_n=5)
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            rt.update(s)
        assert rt.ready
        assert rt.threshold == pytest.approx(
            float(np.quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.9))
        )

    def test_reset_returns_to_warmup(self):
        rt = RollingThreshold(alpha=0.1, min_n=3)
        for s in [0.1, 0.2, 0.3]:
            rt.update(s)
        assert rt.ready
        rt.reset()
        assert not rt.ready
        assert rt.n == 0


class TestWindowing:
    def test_old_scores_evicted(self):
        rt = RollingThreshold(alpha=0.5, window=5, min_n=2)
        for _ in range(50):
            rt.update(0.0)
        for _ in range(5):
            rt.update(10.0)
        # window holds only the five 10.0s
        assert rt.n == 5
        assert rt.threshold == 10.0

    def test_decay_tracks_recent_scores_faster(self):
        slow = RollingThreshold(alpha=0.1, window=200, decay=1.0, min_n=2)
        fast = RollingThreshold(alpha=0.1, window=200, decay=0.9, min_n=2)
        for _ in range(100):
            slow.update(0.0)
            fast.update(0.0)
        for _ in range(10):
            slow.update(1.0)
            fast.update(1.0)
        # equally-weighted window barely moves; decayed window jumps
        assert slow.threshold < 0.5
        assert fast.threshold == 1.0


class TestShiftTracking:
    def test_rolling_fpr_tracks_shift_fixed_drifts(self):
        """Handoff eval case: 500-score benign stream, +1σ mean shift at
        t=250. Rolling FPR post-shift within [alpha-0.05, alpha+0.05];
        fixed pre-shift threshold exceeds alpha+0.1. Averaged over 5 fixed
        seeds; rolling FPR measured after the window refills (t >= 300)."""
        alpha = 0.1
        rolling_fprs, fixed_fprs = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            scores = np.concatenate(
                [rng.normal(0.0, 1.0, 250), rng.normal(1.0, 1.0, 250)]
            )
            fixed_threshold = float(np.quantile(scores[:250], 1 - alpha))
            rt = RollingThreshold(alpha=alpha, window=50, min_n=20)
            rolling_flags = []
            for t, s in enumerate(scores):
                if t >= 300:  # post-shift, window fully refilled
                    rolling_flags.append(s > rt.threshold)
                rt.update(s)  # operator confirms benign afterwards
            post = scores[300:]
            rolling_fprs.append(float(np.mean(rolling_flags)))
            fixed_fprs.append(float(np.mean(post > fixed_threshold)))
        rolling_fpr = float(np.mean(rolling_fprs))
        fixed_fpr = float(np.mean(fixed_fprs))
        assert alpha - 0.05 <= rolling_fpr <= alpha + 0.05
        assert fixed_fpr > alpha + 0.1


class TestDetectorIntegration:
    def test_flag_consults_rolling_when_ready(self):
        from frontier_ops.detector import CalibratedDetector

        from .test_detector import FakeModel, _dataset

        texts, labels = _dataset()
        det = CalibratedDetector(model=FakeModel()).calibrate(texts, labels, fpr=0.1)
        static = det.threshold

        rt = RollingThreshold(alpha=0.1, window=50, min_n=5)
        det.attach_rolling_threshold(rt)
        # not ready yet -> static threshold governs
        assert det.effective_threshold == static
        # feed benign scores far above every score -> nothing should flag
        for _ in range(5):
            rt.update(100.0)
        assert det.effective_threshold == 100.0
        assert not det.flag("leak the secret token")
        # detach restores the static threshold
        det.attach_rolling_threshold(None)
        assert det.effective_threshold == static
