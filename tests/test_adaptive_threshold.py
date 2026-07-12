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


class TestConformalMode:
    def test_conformal_removes_small_window_bias(self):
        """At window 32 the plug-in quantile overshoots alpha; the conformal
        order statistic must not. iid stream, 20 seeds, flag-then-update."""
        alpha, window = 0.1, 32
        plug_fprs, conf_fprs = [], []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            stream = rng.standard_normal(600)
            for conformal, sink in ((False, plug_fprs), (True, conf_fprs)):
                rt = RollingThreshold(
                    alpha=alpha, window=window, min_n=20, conformal=conformal
                )
                flags = []
                for t, s in enumerate(stream):
                    if t >= window:
                        flags.append(s > rt.threshold)
                    rt.update(float(s))
                sink.append(float(np.mean(flags)))
        plug, conf = float(np.mean(plug_fprs)), float(np.mean(conf_fprs))
        assert conf < plug  # correction is strictly less anti-conservative
        assert conf <= alpha + 0.01
        assert conf >= alpha - 0.05  # and not vacuously conservative

    def test_conformal_infinite_when_window_too_small_for_alpha(self):
        rt = RollingThreshold(alpha=0.01, window=32, min_n=20, conformal=True)
        for s in np.linspace(0.0, 1.0, 25):
            rt.update(float(s))
        # k = ceil(26*0.99) = 26 > 25 scores -> +inf: nothing flagged yet
        assert rt.threshold == float("inf")
