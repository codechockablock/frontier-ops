from frontier_ops.governance.ledger import (
    CrossSessionAngularDisplacement as CrossSessionAngularDisplacement,
    SessionSummary as SessionSummary,
    LedgerEntry as LedgerEntry,
)
from frontier_ops.governance.budget import AdaptiveLagrangian as AdaptiveLagrangian

# chain / market_audit need the optional [governance] extra (cryptography).
# Re-export lazily (PEP 562) so `import frontier_ops` works on numpy-only
# installs; a missing extra only surfaces when one of these names is used.
_CRYPTO_EXPORTS = {
    "GovernanceChain": "frontier_ops.governance.chain",
    "GovernanceAuditor": "frontier_ops.governance.chain",
    "ChainEntry": "frontier_ops.governance.chain",
    "VerificationResult": "frontier_ops.governance.chain",
    "observe_agent_step": "frontier_ops.governance.chain",
}


def __getattr__(name: str):
    if name in _CRYPTO_EXPORTS:
        import importlib

        module = importlib.import_module(_CRYPTO_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
