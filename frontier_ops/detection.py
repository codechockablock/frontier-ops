"""
frontier_ops.detection — the detection product's façade.

The library contains two products with distinct dependency footprints
(docs/LEGACY.md):

- **Detection** (this module): the calibrated prototype detector and its
  supporting pieces. Needs numpy plus an encoder
  (``pip install frontier-ops[detect]`` pulls sentence-transformers; any
  :class:`Encoder` can be injected instead). No cryptography.
- **Governance/authorization**: the audit chain, scope, and provenance
  layer — ``pip install frontier-ops[govern]`` pulls cryptography; import
  via ``frontier_ops.governance`` / ``frontier_ops.authorization`` /
  ``frontier_ops.FullPipeline``. No encoder.

Everything here is a re-export; ``from frontier_ops import
CalibratedDetector`` keeps working exactly as before.

Usage::

    from frontier_ops.detection import CalibratedDetector, RollingThreshold

    det = CalibratedDetector()
    det.calibrate(texts, labels)
    det.calibrate_conformal(benign_texts, alpha=0.1)
"""

from __future__ import annotations

from frontier_ops.adaptive_threshold import RollingThreshold as RollingThreshold
from frontier_ops.boundary.static_metric import CalibratedMetric as CalibratedMetric
from frontier_ops.boundary.step_mean import StepMeanScorer as StepMeanScorer
from frontier_ops.conformal import (
    split_conformal_threshold as split_conformal_threshold,
)
from frontier_ops.detector import (
    CalibratedDetector as CalibratedDetector,
    calibrate_from_labeled as calibrate_from_labeled,
)
from frontier_ops.encoder import (
    Encoder as Encoder,
    encode_batch as encode_batch,
)
from frontier_ops.sensing.newma import DualEWMA as DualEWMA

__all__ = [
    "CalibratedDetector",
    "calibrate_from_labeled",
    "RollingThreshold",
    "split_conformal_threshold",
    "Encoder",
    "encode_batch",
    "StepMeanScorer",
    "CalibratedMetric",
    "DualEWMA",
]
