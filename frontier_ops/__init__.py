"""
frontier-ops: Middleware for AI agent governance.
Define your boundaries. Monitor your agents. Prove they stayed within bounds.
"""

__version__ = "0.3.0"

from frontier_ops.boundary.constitution import ConstitutionSpec as ConstitutionSpec, Boundary as Boundary, CrossTerm as CrossTerm
from frontier_ops.boundary.concept_extraction import CONCEPTS as CONCEPTS, ConceptExtractor as ConceptExtractor
from frontier_ops.detector import CalibratedDetector as CalibratedDetector, calibrate_from_labeled as calibrate_from_labeled
from frontier_ops.pipeline import FullPipeline as FullPipeline, StepResult as StepResult
from frontier_ops.authorization import AuthorizationState as AuthorizationState, ProvenanceGraph as ProvenanceGraph, AuthorizationLinkedBudget as AuthorizationLinkedBudget
