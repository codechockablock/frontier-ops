"""End-to-end governance chain lifecycle.

A multi-step agent session is committed to a GovernanceChain, exported, and
verified by an independent GovernanceAuditor holding only the public key —
the full producer → export → external-verifier path that the unit tests
exercise only piecewise. The negative cases tamper with committed entries
and assert verification fails at the right sequence number.
"""

import copy

from frontier_ops.governance.chain import GovernanceAuditor, GovernanceChain


def _run_session(n_steps: int = 8):
    """Simulate a session: one chain entry per agent step."""
    chain = GovernanceChain()
    for step in range(n_steps):
        chain.observe(
            {
                "step": step,
                "tool": "exec" if step % 2 else "read",
                "verdict": "pass" if step != 5 else "flag",
                "alert_level": round(0.05 * step, 3),
                "geodesic_distance": round(0.1 + 0.02 * step, 3),
            }
        )
    return chain


class TestGovernanceLifecycle:
    def test_untampered_chain_verifies_end_to_end(self):
        chain = _run_session(8)
        exported = chain.export_chain()

        # independent auditor: only the public key crosses the trust boundary
        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)

        assert result.valid
        assert result.entries_verified == 8
        assert result.chain_length == 8
        assert result.first_failure is None
        assert result.failure_reason is None

    def test_tampered_payload_fails_at_the_mutated_entry(self):
        chain = _run_session(8)
        exported = copy.deepcopy(chain.export_chain())

        # rewrite history: entry 5's FLAG verdict becomes a PASS
        exported[5]["payload"]["verdict"] = "pass"

        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)

        assert not result.valid
        assert result.first_failure == 5
        assert result.failure_reason is not None

    def test_dropped_entry_breaks_the_link(self):
        chain = _run_session(8)
        exported = copy.deepcopy(chain.export_chain())

        del exported[3]  # silently omit one committed step

        auditor = GovernanceAuditor(chain.public_key_hex())
        result = auditor.verify_chain(exported)

        assert not result.valid

    def test_wrong_key_rejects_everything(self):
        chain = _run_session(3)
        other = GovernanceChain()  # different keypair
        auditor = GovernanceAuditor(other.public_key_hex())

        result = auditor.verify_chain(chain.export_chain())

        assert not result.valid
        assert result.first_failure == 0
