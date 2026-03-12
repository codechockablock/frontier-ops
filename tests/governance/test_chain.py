"""
Tests for Observer Governance (frontier_ops/governance/chain.py)

Verifies:
- Ed25519 key generation
- Chain entry structure
- Signature verification
- Chain hash linkage
- Tamper detection (payload modification)
- Auditor verification
"""

import json
import pytest
import time

from frontier_ops.governance.chain import (
    GovernanceChain,
    GovernanceAuditor,
    ChainEntry,
    observe_agent_step,
    _canonical_json,
    _sha256_hex,
    _GENESIS_HASH,
)


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


class TestCanonicalJson:
    def test_sorted_keys(self):
        obj = {"z": 3, "a": 1, "m": 2}
        result = _canonical_json(obj)
        # Should be compact with sorted keys
        assert result == b'{"a":1,"m":2,"z":3}'

    def test_bytes_output(self):
        result = _canonical_json({"key": "value"})
        assert isinstance(result, bytes)

    def test_deterministic(self):
        obj = {"b": 2, "a": 1}
        assert _canonical_json(obj) == _canonical_json(obj)


# ---------------------------------------------------------------------------
# GovernanceChain
# ---------------------------------------------------------------------------


class TestGovernanceChainInit:
    def test_creates_key_pair(self):
        chain = GovernanceChain()
        pub = chain.public_key_hex()
        assert isinstance(pub, str)
        assert len(pub) == 64  # 32 bytes hex = 64 chars

    def test_empty_chain(self):
        chain = GovernanceChain()
        assert len(chain) == 0
        assert chain.head() is None


class TestGovernanceChainObserve:
    def test_observe_returns_entry(self):
        chain = GovernanceChain()
        entry = chain.observe({"step": 0, "angular_disp": 0.12})
        assert isinstance(entry, ChainEntry)

    def test_observe_increments_length(self):
        chain = GovernanceChain()
        chain.observe({"step": 0})
        assert len(chain) == 1
        chain.observe({"step": 1})
        assert len(chain) == 2

    def test_entry_seq_is_correct(self):
        chain = GovernanceChain()
        e0 = chain.observe({"step": 0})
        e1 = chain.observe({"step": 1})
        assert e0.seq == 0
        assert e1.seq == 1

    def test_first_entry_has_genesis_prev_hash(self):
        chain = GovernanceChain()
        entry = chain.observe({"data": "first"})
        assert entry.prev_hash == _GENESIS_HASH

    def test_second_entry_links_to_first(self):
        chain = GovernanceChain()
        e0 = chain.observe({"step": 0})
        e1 = chain.observe({"step": 1})
        assert e1.prev_hash == e0.chain_hash

    def test_payload_hash_correctness(self):
        chain = GovernanceChain()
        payload = {"step": 5, "angular_disp": 0.42, "verdict": "PASS"}
        entry = chain.observe(payload)
        expected = _sha256_hex(_canonical_json(payload))
        assert entry.payload_hash == expected

    def test_chain_hash_correctness(self):
        chain = GovernanceChain()
        entry = chain.observe({"x": 1})
        expected_chain_input = (_GENESIS_HASH + entry.payload_hash).encode("utf-8")
        expected_chain_hash = _sha256_hex(expected_chain_input)
        assert entry.chain_hash == expected_chain_hash

    def test_signature_is_hex_string(self):
        chain = GovernanceChain()
        entry = chain.observe({"step": 0})
        assert isinstance(entry.signature, str)
        # Ed25519 signature is 64 bytes = 128 hex chars
        assert len(entry.signature) == 128

    def test_head_returns_latest(self):
        chain = GovernanceChain()
        chain.observe({"step": 0})
        chain.observe({"step": 1})
        e2 = chain.observe({"step": 2})
        assert chain.head().seq == e2.seq
        assert chain.head().chain_hash == e2.chain_hash

    def test_entry_to_dict(self):
        chain = GovernanceChain()
        entry = chain.observe({"step": 0, "value": 1.5})
        d = entry.to_dict()
        assert "seq" in d
        assert "payload" in d
        assert "payload_hash" in d
        assert "prev_hash" in d
        assert "chain_hash" in d
        assert "signature" in d
        assert d["payload"] == {"step": 0, "value": 1.5}


class TestGovernanceChainExport:
    def test_export_is_list_of_dicts(self):
        chain = GovernanceChain()
        chain.observe({"a": 1})
        chain.observe({"b": 2})
        exported = chain.export_chain()
        assert isinstance(exported, list)
        assert len(exported) == 2
        assert all(isinstance(e, dict) for e in exported)

    def test_export_is_json_serializable(self):
        chain = GovernanceChain()
        chain.observe({"step": 0, "angular_disp": 0.3, "verdict": "PASS"})
        exported = chain.export_chain()
        # Should not raise
        serialized = json.dumps(exported)
        assert len(serialized) > 0

    def test_round_trip_from_dict(self):
        chain = GovernanceChain()
        entry = chain.observe({"step": 0})
        d = entry.to_dict()
        reconstructed = ChainEntry.from_dict(d)
        assert reconstructed.seq == entry.seq
        assert reconstructed.chain_hash == entry.chain_hash
        assert reconstructed.signature == entry.signature


# ---------------------------------------------------------------------------
# GovernanceAuditor
# ---------------------------------------------------------------------------


class TestGovernanceAuditor:
    def setup_method(self):
        self.chain = GovernanceChain()
        self.pub_key = self.chain.public_key_hex()
        self.auditor = GovernanceAuditor(self.pub_key)

    def test_empty_chain_is_valid(self):
        result = self.auditor.verify_chain([])
        assert result.valid is True
        assert result.entries_verified == 0

    def test_single_entry_is_valid(self):
        self.chain.observe({"step": 0, "angular_disp": 0.1})
        result = self.auditor.verify_chain(self.chain.export_chain())
        assert result.valid is True
        assert result.entries_verified == 1

    def test_multiple_entries_are_valid(self):
        for i in range(5):
            self.chain.observe({"step": i, "value": i * 0.1})
        result = self.auditor.verify_chain(self.chain.export_chain())
        assert result.valid is True
        assert result.entries_verified == 5
        assert result.chain_length == 5

    def test_tampered_payload_detected(self):
        """Modifying payload after signing should fail verification."""
        self.chain.observe({"step": 0, "angular_disp": 0.1})
        exported = self.chain.export_chain()
        # Tamper with the payload
        exported[0]["payload"]["angular_disp"] = 999.9
        result = self.auditor.verify_chain(exported)
        assert result.valid is False
        assert result.first_failure == 0
        assert "payload_hash" in result.failure_reason

    def test_tampered_chain_hash_detected(self):
        """Modifying chain_hash should fail verification."""
        self.chain.observe({"step": 0})
        self.chain.observe({"step": 1})
        exported = self.chain.export_chain()
        # Tamper with chain hash of first entry
        exported[0]["chain_hash"] = "a" * 64
        result = self.auditor.verify_chain(exported)
        assert result.valid is False

    def test_tampered_signature_detected(self):
        """Modifying signature should fail verification."""
        self.chain.observe({"step": 0})
        exported = self.chain.export_chain()
        # Flip one byte in signature
        sig = exported[0]["signature"]
        flipped = sig[:-2] + ("ff" if sig[-2:] != "ff" else "00")
        exported[0]["signature"] = flipped
        result = self.auditor.verify_chain(exported)
        assert result.valid is False
        assert "signature" in result.failure_reason

    def test_wrong_public_key_fails(self):
        """Verifying with a different public key should fail."""
        self.chain.observe({"step": 0})
        exported = self.chain.export_chain()

        # Create auditor with a different (new) key
        other_chain = GovernanceChain()
        wrong_auditor = GovernanceAuditor(other_chain.public_key_hex())

        result = wrong_auditor.verify_chain(exported)
        assert result.valid is False

    def test_broken_chain_linkage_detected(self):
        """Swapping entries should break the chain linkage."""
        for i in range(3):
            self.chain.observe({"step": i})
        exported = self.chain.export_chain()
        # Swap entries 0 and 1 -- breaks prev_hash chain
        exported[0], exported[1] = exported[1], exported[0]
        result = self.auditor.verify_chain(exported)
        assert result.valid is False

    def test_verification_result_structure(self):
        self.chain.observe({"step": 0})
        result = self.auditor.verify_chain(self.chain.export_chain())
        assert hasattr(result, "valid")
        assert hasattr(result, "entries_verified")
        assert hasattr(result, "first_failure")
        assert hasattr(result, "failure_reason")
        assert hasattr(result, "chain_length")
