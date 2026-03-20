"""
Bayes Factor Evidence Combiner
================================

Replaces ad-hoc alert_level weighting with rigorous evidence combination.

Converts per-detector p-values to Bayes factors via Vovk-Sellke bound,
then multiplies across independent detectors. The combined Bayes factor
is interpretable as "the evidence for violation is X times stronger
than the evidence for normal behavior."

References:
  Sellke, Bayarri & Berger (2001), Am Stat, 55(1), 62-71
  Benjamin et al. (2018), Nature Human Behaviour
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

E = math.e


@dataclass
class EvidenceResult:
    """Combined evidence from multiple detectors."""
    combined_bf: float
    interpretation: str
    per_detector_bf: Dict[str, float]
    per_detector_p: Dict[str, float]
    n_detectors: int
    # Calibration
    log_bf: float  # log10(combined_bf) for numerical stability

    @property
    def strong_evidence(self) -> bool:
        return self.combined_bf > 10

    @property
    def moderate_evidence(self) -> bool:
        return self.combined_bf > 3


def vovk_sellke_bf(p: float) -> float:
    """
    Maximum Bayes factor from a p-value via Vovk-Sellke bound.

    BF_max = 1 / (-e · p · ln(p)) for p < 1/e

    This is the strongest possible evidence against the null,
    under ANY alternative hypothesis.
    """
    if p <= 0 or p >= 1:
        return 1.0
    if p >= 1 / E:
        return 1.0
    log_p = math.log(p)
    bf = 1.0 / (-E * p * log_p)
    return max(1.0, bf)


def interpret_bf(bf: float) -> str:
    """Interpret a Bayes factor on the Jeffreys scale."""
    if bf < 1:
        return "evidence_for_null"
    elif bf < 3:
        return "anecdotal"
    elif bf < 10:
        return "moderate"
    elif bf < 30:
        return "strong"
    elif bf < 100:
        return "very_strong"
    else:
        return "decisive"


class BayesFactorCombiner:
    """
    Combines evidence from multiple independent detectors.

    Each detector produces a p-value (via permutation testing on
    normal traces). The combiner converts to Bayes factors and
    multiplies them.

    Usage::
        combiner = BayesFactorCombiner(detector_names=["angular_disp", "ewma", "newma", "trend"])
        result = combiner.combine({
            "angular_disp": 0.03,
            "ewma": 0.12,
            "newma": 0.008,
            "trend": 0.25,
        })
        print(f"Combined BF: {result.combined_bf:.1f} ({result.interpretation})")
    """

    def __init__(
        self,
        detector_names: Optional[List[str]] = None,
        min_p: float = 1e-10,  # Floor to avoid numerical issues
    ):
        self.detector_names = detector_names or []
        self.min_p = min_p

    def combine(self, p_values: Dict[str, float]) -> EvidenceResult:
        """
        Combine p-values from multiple detectors into a single evidence measure.

        Args:
            p_values: detector_name → p-value mapping.

        Returns:
            EvidenceResult with combined Bayes factor and interpretation.
        """
        per_bf = {}
        log_bf_total = 0.0

        for name, p in p_values.items():
            p = max(p, self.min_p)
            bf = vovk_sellke_bf(p)
            per_bf[name] = bf
            log_bf_total += math.log10(bf)

        combined_bf = 10 ** log_bf_total

        return EvidenceResult(
            combined_bf=combined_bf,
            interpretation=interpret_bf(combined_bf),
            per_detector_bf=per_bf,
            per_detector_p=dict(p_values),
            n_detectors=len(p_values),
            log_bf=log_bf_total,
        )

    def combine_from_statistics(
        self,
        statistics: Dict[str, float],
        null_distributions: Dict[str, np.ndarray],
    ) -> EvidenceResult:
        """
        Convert raw detector statistics to p-values via permutation
        testing, then combine.

        Args:
            statistics: detector_name → test statistic value.
            null_distributions: detector_name → array of statistic values
                under the null hypothesis (from normal traces).
        """
        p_values = {}
        for name, stat in statistics.items():
            if name in null_distributions:
                null = null_distributions[name]
                p = float(np.mean(null >= stat))
                p = max(p, 1.0 / (len(null) + 1))  # Correction for finite permutations
            else:
                p = 0.5  # No null distribution → uninformative
            p_values[name] = p

        return self.combine(p_values)
