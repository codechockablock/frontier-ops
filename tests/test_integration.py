"""
Integration tests -- pipeline + governance working together.

These tests exercise realistic scenarios end-to-end, not individual
components. (The memory integration tests moved to attic/pre-v0.6 with
the memory package.)
"""

from frontier_ops.boundary.concept_extraction import KeywordConceptExtractor, CONCEPTS


class TestGovernanceIntegration:
    """Test that governance chain records and validates steps."""

    def test_governance_chain_integrity(self):
        from frontier_ops.governance.chain import GovernanceChain, GovernanceAuditor
        chain = GovernanceChain()

        for i in range(5):
            chain.observe({"step": i, "angular_disp": 0.1 * i})

        assert len(chain) == 5
        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(chain.export_chain())
        assert result.valid

    def test_tampered_chain_fails_verification(self):
        from frontier_ops.governance.chain import GovernanceChain, GovernanceAuditor
        chain = GovernanceChain()

        chain.observe({"step": 0, "value": 1.0})

        # Export, tamper, verify
        exported = chain.export_chain()
        exported[0]["payload"]["step"] = 999

        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)
        assert not result.valid


class TestConceptExtractionIntegration:
    """Test concept extraction with trajectory tracking."""

    def test_concept_vectors_are_6dim(self):
        ext = KeywordConceptExtractor()
        scores = ext.extract("Solve the equation")
        assert len(scores) == len(CONCEPTS)
