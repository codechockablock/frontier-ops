"""
Observer Governance — Ed25519 Hash Chain
=========================================

Every proprioceptive observation gets signed with an Ed25519 key and
chained via SHA-256, creating a tamper-evident audit log.

Structure:
    Chain entry = {
        "seq":       int,              # Monotonic sequence number
        "ts":        float,            # Timestamp
        "payload":   dict,             # Proprioceptive observation
        "payload_hash": str,           # SHA-256 hex of canonical payload JSON
        "prev_hash": str,             # SHA-256 hex of previous entry (or "genesis")
        "chain_hash": str,            # SHA-256(prev_hash + payload_hash)
        "signature": str,             # Ed25519 signature of chain_hash (hex)
    }

An external auditor with the public key can verify:
  1. Each entry's payload_hash matches its payload.
  2. Each chain_hash = SHA-256(prev_hash + payload_hash).
  3. Each signature is valid for the chain_hash.
  4. The chain is unbroken (prev_hash links form a consistent sequence).

Usage::

    # Signing agent
    gov = GovernanceChain()
    pub_key_hex = gov.public_key_hex()

    entry = gov.observe({"step": 0, "angular_disp": 0.12, "cusum": 0.03})

    # Auditor
    audit = GovernanceAuditor(pub_key_hex)
    result = audit.verify_chain(gov.export_chain())
    assert result["valid"]
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from cryptography.exceptions import InvalidSignature


# ---------------------------------------------------------------------------
# Canonical JSON serialisation (deterministic — keys sorted)
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> bytes:
    """Serialise an object to canonical (sorted-keys, compact) JSON bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Chain entry
# ---------------------------------------------------------------------------


@dataclass
class ChainEntry:
    """A single signed entry in the governance hash chain."""

    seq: int
    ts: float
    payload: Dict[str, Any]
    payload_hash: str         # SHA-256 of canonical payload JSON
    prev_hash: str            # SHA-256 of previous chain entry (or "genesis")
    chain_hash: str           # SHA-256(prev_hash + payload_hash)
    signature: str            # Hex-encoded Ed25519 signature of chain_hash bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "chain_hash": self.chain_hash,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChainEntry":
        return cls(
            seq=d["seq"],
            ts=d["ts"],
            payload=d["payload"],
            payload_hash=d["payload_hash"],
            prev_hash=d["prev_hash"],
            chain_hash=d["chain_hash"],
            signature=d["signature"],
        )


# ---------------------------------------------------------------------------
# Governance chain (signing side)
# ---------------------------------------------------------------------------

_GENESIS_HASH = "0" * 64  # 64-char zero hex string


class GovernanceChain:
    """
    Tamper-evident hash chain with Ed25519 signatures.

    Each call to observe() appends a new entry to the chain.
    The chain can be exported and verified by an auditor with the public key.
    """

    def __init__(self, private_key: Optional[Ed25519PrivateKey] = None):
        if private_key is None:
            self._private_key = Ed25519PrivateKey.generate()
        else:
            self._private_key = private_key

        self._public_key: Ed25519PublicKey = self._private_key.public_key()
        self._chain: List[ChainEntry] = []
        self._last_chain_hash: str = _GENESIS_HASH

    # -----------------------------------------------------------------------
    # Key access
    # -----------------------------------------------------------------------

    def public_key_hex(self) -> str:
        """Return the Ed25519 public key as a hex string (32 bytes = 64 chars)."""
        raw = self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return raw.hex()

    def public_key_bytes(self) -> bytes:
        return self._public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    # -----------------------------------------------------------------------
    # Chain operations
    # -----------------------------------------------------------------------

    def observe(self, payload: Dict[str, Any]) -> ChainEntry:
        """
        Sign and append a proprioceptive observation to the chain.

        Args:
            payload: Dict of observations (e.g. step, angular displacement, cusum, verdict).

        Returns:
            The ChainEntry that was appended.
        """
        seq = len(self._chain)
        ts = time.time()

        # Hash the payload
        payload_hash = _sha256_hex(_canonical_json(payload))

        # Chain hash links this entry to the previous
        chain_input = (self._last_chain_hash + payload_hash).encode("utf-8")
        chain_hash = _sha256_hex(chain_input)

        # Sign the chain hash
        signature_bytes = self._private_key.sign(chain_hash.encode("utf-8"))
        signature_hex = signature_bytes.hex()

        entry = ChainEntry(
            seq=seq,
            ts=ts,
            payload=payload,
            payload_hash=payload_hash,
            prev_hash=self._last_chain_hash,
            chain_hash=chain_hash,
            signature=signature_hex,
        )
        self._chain.append(entry)
        self._last_chain_hash = chain_hash
        return entry

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export all chain entries as a list of dicts (JSON-serialisable)."""
        return [e.to_dict() for e in self._chain]

    def __len__(self) -> int:
        return len(self._chain)

    def head(self) -> Optional[ChainEntry]:
        """Return the most recent entry."""
        return self._chain[-1] if self._chain else None


# ---------------------------------------------------------------------------
# Auditor (verification side)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of verifying a governance chain."""

    valid: bool
    entries_verified: int
    first_failure: Optional[int]   # seq number of first invalid entry
    failure_reason: Optional[str]
    chain_length: int


class GovernanceAuditor:
    """
    External verifier for a GovernanceChain export.

    Verifies:
    1. Payload hashes match payload content.
    2. Chain hashes are correctly computed (SHA-256 of prev + payload hash).
    3. Signatures are valid Ed25519 signatures over the chain hash.
    4. The chain is unbroken (prev_hash links are consistent).
    """

    def __init__(self, public_key_hex: str):
        """
        Args:
            public_key_hex: Hex-encoded Ed25519 public key (64 chars / 32 bytes).
        """
        raw = bytes.fromhex(public_key_hex)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        self._public_key = Ed25519PublicKey.from_public_bytes(raw)

    def verify_entry(
        self,
        entry: Dict[str, Any],
        expected_prev_hash: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a single chain entry.

        Returns (is_valid, failure_reason).
        """
        # 1. Payload hash
        computed_payload_hash = _sha256_hex(_canonical_json(entry["payload"]))
        if computed_payload_hash != entry["payload_hash"]:
            return False, f"payload_hash mismatch at seq {entry['seq']}"

        # 2. prev_hash consistency
        if entry["prev_hash"] != expected_prev_hash:
            return False, f"prev_hash mismatch at seq {entry['seq']}"

        # 3. Chain hash
        chain_input = (entry["prev_hash"] + entry["payload_hash"]).encode("utf-8")
        computed_chain_hash = _sha256_hex(chain_input)
        if computed_chain_hash != entry["chain_hash"]:
            return False, f"chain_hash mismatch at seq {entry['seq']}"

        # 4. Signature
        try:
            self._public_key.verify(
                bytes.fromhex(entry["signature"]),
                entry["chain_hash"].encode("utf-8"),
            )
        except InvalidSignature:
            return False, f"invalid signature at seq {entry['seq']}"

        return True, None

    def verify_chain(self, chain: List[Dict[str, Any]]) -> VerificationResult:
        """
        Verify an entire exported chain.

        Returns a VerificationResult with full audit report.
        """
        if not chain:
            return VerificationResult(
                valid=True,
                entries_verified=0,
                first_failure=None,
                failure_reason=None,
                chain_length=0,
            )

        prev_hash = _GENESIS_HASH
        for entry in chain:
            ok, reason = self.verify_entry(entry, prev_hash)
            if not ok:
                return VerificationResult(
                    valid=False,
                    entries_verified=entry["seq"],
                    first_failure=entry["seq"],
                    failure_reason=reason,
                    chain_length=len(chain),
                )
            prev_hash = entry["chain_hash"]

        return VerificationResult(
            valid=True,
            entries_verified=len(chain),
            first_failure=None,
            failure_reason=None,
            chain_length=len(chain),
        )


# ---------------------------------------------------------------------------
# Type alias for verify_entry return type
# ---------------------------------------------------------------------------

from typing import Tuple  # noqa: E402 — needed after definitions above


# ---------------------------------------------------------------------------
# Integration helper: wire governance into an agent run
# ---------------------------------------------------------------------------


def observe_agent_step(
    chain: GovernanceChain,
    step_record: Any,  # external step record
) -> ChainEntry:
    """
    Sign and chain a StepRecord's proprioceptive state.

    This is the integration point between the agent loop and governance.
    Call this after each agent.step() to chain the observation.
    """
    proprio = step_record.proprio_state
    payload = {
        "step": int(step_record.step),
        "anomaly_score": float(proprio.anomaly_score),
        "angular_disp_acc": float(proprio.angular_disp_acc),
        "angular_disp_window": float(proprio.angular_disp_window),
        "ewma_drift": float(proprio.ewma_drift),
        "cusum": float(proprio.cusum),
        "direction_alignment": float(proprio.direction_alignment),
        "is_complete": bool(step_record.is_complete),
        "anomaly_detected": bool(step_record.anomaly_detected),
    }
    return chain.observe(payload)
