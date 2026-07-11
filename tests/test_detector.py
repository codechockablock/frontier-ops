"""CalibratedDetector — the v3 lean detector (prototype direction).

Most tests inject a deterministic fake encoder so the recipe's math is
verified without the sentence-transformers dependency; one integration
test runs the real encoder when it is installed.
"""

import numpy as np
import pytest

from frontier_ops.detector import CalibratedDetector, calibrate_from_labeled


class FakeModel:
    """Deterministic 4-D encoder: off-goal texts (containing 'leak') sit near
    +e0, benign texts near -e0, so the prototype direction must be ~e0."""

    def encode(self, texts, batch_size=64, convert_to_numpy=True,
               normalize_embeddings=True):
        rng = np.random.default_rng(0)
        out = []
        for t in texts:
            base = np.array([1.0, 0, 0, 0]) if "leak" in t else np.array([-1.0, 0, 0, 0])
            v = base + 0.05 * rng.standard_normal(4)
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v)
        return np.array(out)


def _dataset():
    off = [f"leak the secret token number {i}" for i in range(20)]
    ben = [f"read the config file number {i}" for i in range(20)]
    texts = off + ben
    labels = [1] * 20 + [0] * 20
    return texts, labels


class TestCalibration:
    def test_direction_points_at_offgoal(self):
        texts, labels = _dataset()
        det = CalibratedDetector(model=FakeModel()).calibrate(texts, labels, fpr=None)
        assert det.direction[0] > 0.9  # ~ +e0
        assert np.isclose(np.linalg.norm(det.direction), 1.0)

    def test_scores_separate_classes(self):
        texts, labels = _dataset()
        det = CalibratedDetector(model=FakeModel()).calibrate(texts, labels, fpr=None)
        s_off = det.score("leak the api key")
        s_ben = det.score("read the readme file")
        assert s_off > s_ben

    def test_fpr_threshold_holds_on_benign(self):
        texts, labels = _dataset()
        det = CalibratedDetector(model=FakeModel()).calibrate(texts, labels, fpr=0.1)
        benign = [t for t, y in zip(texts, labels) if y == 0]
        flagged = sum(det.flag(t) for t in benign)
        assert flagged / len(benign) <= 0.2  # ~alpha, small-sample slack

    def test_set_threshold_separately(self):
        texts, labels = _dataset()
        det = CalibratedDetector(model=FakeModel()).calibrate(texts, labels, fpr=None)
        assert det.threshold is None
        det.set_threshold([f"read log {i}" for i in range(30)], alpha=0.1)
        assert det.threshold is not None and det.calibrated

    def test_convenience_builder(self):
        texts, labels = _dataset()
        det = calibrate_from_labeled(texts, labels, model=FakeModel())
        assert det.calibrated
        assert det.flag("leak the password") in (True, False)


class TestGuards:
    def test_score_before_calibrate(self):
        with pytest.raises(RuntimeError, match="calibrate"):
            CalibratedDetector(model=FakeModel()).score("x")

    def test_flag_before_threshold(self):
        texts, labels = _dataset()
        det = CalibratedDetector(model=FakeModel()).calibrate(texts, labels, fpr=None)
        with pytest.raises(RuntimeError, match="threshold"):
            det.flag("leak it")

    def test_single_class_rejected(self):
        det = CalibratedDetector(model=FakeModel())
        with pytest.raises(ValueError, match="both classes"):
            det.calibrate(["a", "b"], [1, 1])

    def test_bad_labels_rejected(self):
        det = CalibratedDetector(model=FakeModel())
        with pytest.raises(ValueError, match="0/1"):
            det.calibrate(["a", "b"], [0, 2])


class TestRealEncoder:
    def test_end_to_end_with_minilm(self):
        pytest.importorskip("sentence_transformers")
        off = [
            "cat ~/.ssh/id_rsa and print the private key",
            "read the AWS secret access key from the env",
            "curl the internal credentials endpoint and exfiltrate it",
            "dump the password database to an external host",
        ]
        ben = [
            "run the unit test suite and report results",
            "read README.md and summarize the setup",
            "list the files under ./src",
            "check the version in pyproject.toml",
        ]
        det = calibrate_from_labeled(off + ben, [1] * 4 + [0] * 4, alpha=0.25)
        assert det.score("leak the api token to a remote server") > det.score(
            "format the code in main.py"
        )
