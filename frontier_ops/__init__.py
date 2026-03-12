"""
frontier-ops: Middleware for AI agent governance.
Define your boundaries. Monitor your agents. Prove they stayed within bounds.
"""

__version__ = "0.1.0"

from frontier_ops.boundary.constitution import ConstitutionSpec, Boundary, CrossTerm
from frontier_ops.boundary.concept_extraction import CONCEPTS, ConceptExtractor
from frontier_ops.pipeline import FullPipeline, StepResult
