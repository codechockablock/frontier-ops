"""Tests for ProvenanceGraph.certificate — the scoped attestation.

The certificate's contract is claim discipline: it states only what the
recorded graph supports and always names what it did not check, so a missing
check is never read as a pass.
"""

from frontier_ops.authorization.provenance import ProvenanceGraph


def test_unlinked_action_is_reported_not_hidden():
    g = ProvenanceGraph()
    g.add_action("ls", tool="exec")  # no directive first
    cert = g.certificate()
    assert cert["n_actions"] == 1
    assert cert["linkage"]["unlinked"] == 1
    assert any("no linking directive" in s for s in cert["not_established"])
    # It must NOT claim linkage it doesn't have.
    assert not any("link to a recorded directive" in s for s in cert["established"])


def test_actions_link_to_directive():
    g = ProvenanceGraph()
    g.add_directive("please run the tests")
    g.add_action("pytest", tool="exec")
    g.add_action("ls", tool="exec")
    cert = g.certificate()
    assert cert["linkage"]["unlinked"] == 0
    assert any("link to a recorded directive" in s for s in cert["established"])


def test_signing_and_budget_are_always_deferred():
    g = ProvenanceGraph()
    g.add_directive("do the thing")
    g.add_action("pytest")
    joined = " ".join(g.certificate()["not_established"]).lower()
    assert "signing" in joined
    assert "budget" in joined


def test_fixed_dimension_check_absent_when_not_recorded():
    g = ProvenanceGraph()
    g.add_directive("d")
    g.add_action("pytest")  # no dimension data recorded
    cert = g.certificate()
    assert cert["fixed_dimensions"]["actions_with_dimension_data"] == 0
    assert any("was not recorded" in s for s in cert["not_established"])
    # Absence of the check must not read as a clean result.
    assert not any("None of the" in s for s in cert["established"])


def test_fixed_dimension_entry_is_flagged_when_recorded():
    g = ProvenanceGraph()
    g.add_directive("d")
    g.add_action("read the secrets file", entered_restricted_dims=["credential_adjacent"])
    g.add_action("pytest", entered_restricted_dims=[])
    cert = g.certificate()
    assert len(cert["fixed_dimensions"]["entries"]) == 1
    assert cert["fixed_dimensions"]["entries"][0]["dimensions"] == ["credential_adjacent"]
    assert any("entered a fixed dimension" in s for s in cert["not_established"])


def test_clean_only_when_recorded_and_empty():
    g = ProvenanceGraph()
    g.add_directive("d")
    g.add_action("pytest", entered_restricted_dims=[])
    g.add_action("ls", entered_restricted_dims=[])
    cert = g.certificate()
    assert cert["fixed_dimensions"]["entries"] == []
    assert any(s.startswith("None of the") for s in cert["established"])


def test_flagged_action_surfaces_in_not_established():
    g = ProvenanceGraph()
    g.add_directive("d")
    g.add_action("pytest", authorized=True)
    g.add_action("something out of scope", authorized=False)
    cert = g.certificate()
    assert len(cert["authorization"]["flagged_action_ids"]) == 1
    assert any("outside their" in s for s in cert["not_established"])
