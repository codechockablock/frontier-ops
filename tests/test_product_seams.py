"""Product seams: detection runs without cryptography, imports stay intact.

The detect/govern split promises a detection-only install never needs
cryptography. The subprocess test enforces that by *blocking* cryptography
imports outright and running CalibratedDetector end-to-end through the
frontier_ops.detection façade with a stub encoder.
"""

import subprocess
import sys

import frontier_ops
import frontier_ops.detection as detection


class TestFacade:
    def test_facade_exports(self):
        for name in detection.__all__:
            assert getattr(detection, name) is not None

    def test_top_level_imports_unchanged(self):
        # the pre-split public surface must keep resolving
        for name in [
            "CalibratedDetector",
            "calibrate_from_labeled",
            "RollingThreshold",
            "Encoder",
            "FullPipeline",
            "StepResult",
            "AuthorizationState",
            "ProvenanceGraph",
            "AuthorizationLinkedBudget",
            "ConstitutionSpec",
            "Boundary",
            "CrossTerm",
            "CONCEPTS",
            "ConceptExtractor",
        ]:
            assert getattr(frontier_ops, name) is not None


class TestCryptographyIsolation:
    def test_detection_end_to_end_with_cryptography_blocked(self):
        """Simulates a [detect]-only install: any cryptography import raises."""
        code = r"""
import sys

class BlockCryptography:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "cryptography":
            raise ModuleNotFoundError("cryptography blocked (detect-only install)")
        return None

sys.meta_path.insert(0, BlockCryptography())

import numpy as np
import frontier_ops
from frontier_ops.detection import CalibratedDetector, RollingThreshold

class StubEncoder:
    def encode(self, texts):
        rng = np.random.default_rng(3)
        out = []
        for t in texts:
            base = np.array([1.0, 0, 0, 0]) if "leak" in t else np.array([-1.0, 0, 0, 0])
            v = base + 0.05 * rng.standard_normal(4)
            out.append(v / np.linalg.norm(v))
        return np.array(out)

det = CalibratedDetector(model=StubEncoder())
off = [f"leak secret {i}" for i in range(8)]
ben = [f"read file {i}" for i in range(8)]
det.calibrate(off + ben, [1] * 8 + [0] * 8, fpr=0.25)
# the leak cluster sits ~2 units above the threshold; deterministic
assert det.flag("leak the credentials")
assert det.score("leak the credentials") > det.score("read the changelog")

rt = RollingThreshold(alpha=0.2, window=32, min_n=4)
for s in det.score_batch([f"read doc {i}" for i in range(8)]):
    rt.update(float(s))
det.attach_rolling_threshold(rt)
assert det.flag("leak the keys")  # rolling threshold sits in the benign cluster
print("DETECTION-OK-WITHOUT-CRYPTOGRAPHY")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "DETECTION-OK-WITHOUT-CRYPTOGRAPHY" in result.stdout
