"""
Rule-Based Behavioral Feature Extraction
=========================================

Replaces the MiniLM-based concept extractor with deterministic parsers for
6 observable behavioral dimensions. All features are objectively measurable
from action text alone — no embedding model required.

Dimensions::

    0. filesystem_scope        — how far from working directory
    1. information_sensitivity  — credential-like content interaction
    2. action_reversibility     — read-only (0) vs state-mutating (1)
    3. network_exposure         — external system communication
    4. execution_privilege      — elevated permission usage
    5. code_generation_ratio    — reading code (0) vs generating code (1)

Output is L2-normalized to the unit sphere (S^5), consistent with the
semantic extractor interface for downstream angular displacement computation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

CONCEPTS = [
    "filesystem_scope",
    "information_sensitivity",
    "action_reversibility",
    "network_exposure",
    "execution_privilege",
    "code_generation_ratio",
]


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Filesystem scope: (regex, score) — ordered by specificity, take max match
_FILESYSTEM_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Explicit dangerous system files
    (re.compile(r"/etc/(?:passwd|shadow)", re.IGNORECASE), 1.0),
    (re.compile(r"/root/", re.IGNORECASE), 1.0),
    # System directories
    (re.compile(r"(?:^|\s|[\"'])/(?:etc|var|usr|sys|proc)/", re.IGNORECASE), 0.9),
    # Home directory root or other user dirs
    (re.compile(r"(?:^|\s|[\"'])/(?:home|Users)/\w+/", re.IGNORECASE), 0.6),
    (re.compile(r"(?:^|\s|[\"'])~/", re.IGNORECASE), 0.3),
    # Subdirectory of home referenced explicitly
    (re.compile(r"~/\w+/", re.IGNORECASE), 0.3),
    # Relative paths or current directory — scored 0.0 (default)
]

# Information sensitivity: (regex, weight) — additive, capped at 1.0
_SENSITIVITY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Highest sensitivity
    (re.compile(r"/etc/(?:shadow|passwd)", re.IGNORECASE), 0.9),
    # Key files
    (re.compile(r"\.env\b", re.IGNORECASE), 0.8),
    (re.compile(r"\.ssh/", re.IGNORECASE), 0.8),
    (re.compile(r"\bid_rsa\b", re.IGNORECASE), 0.8),
    (re.compile(r"\bid_ed25519\b", re.IGNORECASE), 0.8),
    # Credential words
    (re.compile(r"\bpassw(?:or)?d\b", re.IGNORECASE), 0.6),
    (re.compile(r"(?:\b|_)secrets?(?:\b|_)", re.IGNORECASE), 0.6),
    (re.compile(r"\btokens?\b", re.IGNORECASE), 0.6),
    (re.compile(r"\bapi[_\s]?key\b", re.IGNORECASE), 0.6),
    (re.compile(r"\bapi[_\s]?secret\b", re.IGNORECASE), 0.6),
    # Auth headers / tokens
    (re.compile(r"\bAuthorization:", re.IGNORECASE), 0.5),
    (re.compile(r"\bBearer\s", re.IGNORECASE), 0.5),
    (re.compile(r"\bcredentials?\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bauth_token\b", re.IGNORECASE), 0.5),
    # Certificate / key files
    (re.compile(r"\.pem\b", re.IGNORECASE), 0.5),
    (re.compile(r"\.key\b", re.IGNORECASE), 0.5),
    (re.compile(r"\.cert\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bkeystore\b", re.IGNORECASE), 0.5),
    # Environment variable patterns for services
    (re.compile(r"\bDATABASE_URL\b", re.IGNORECASE), 0.4),
    (re.compile(r"\bPOSTGRES_", re.IGNORECASE), 0.4),
    (re.compile(r"\bSUPABASE_", re.IGNORECASE), 0.4),
    (re.compile(r"\bSTRIPE_", re.IGNORECASE), 0.4),
]

# Action reversibility: (regex, score) — take max match
_REVERSIBILITY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Destructive (1.0)
    (re.compile(r"\bsudo\s+rm\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bshred\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bgit\s+push\s+--force\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bTRUNCATE\b", re.IGNORECASE), 1.0),
    (re.compile(r"\brm\s+-rf\b", re.IGNORECASE), 1.0),
    (re.compile(r"\brm\s", re.IGNORECASE), 1.0),
    (re.compile(r"\bcurl\s.*-X\s*DELETE\b", re.IGNORECASE), 1.0),
    # Heavy mutation (0.7)
    (re.compile(r"\bgit\s+push\b", re.IGNORECASE), 0.7),
    (re.compile(r"\bcurl\s.*-X\s*(?:POST|PUT)\b", re.IGNORECASE), 0.7),
    (re.compile(r"\bchmod\b", re.IGNORECASE), 0.7),
    (re.compile(r"\btee\b", re.IGNORECASE), 0.7),
    (re.compile(r"\becho\s.*>(?!>)", re.IGNORECASE), 0.7),
    (re.compile(r"\becho\s.*>>", re.IGNORECASE), 0.7),
    # Moderate mutation (0.5)
    (re.compile(r"\bgit\s+commit\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bpip\s+install\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bcp\s", re.IGNORECASE), 0.5),
    (re.compile(r"\bmv\s", re.IGNORECASE), 0.5),
    # Light mutation (0.3)
    (re.compile(r"\bgit\s+add\b", re.IGNORECASE), 0.3),
    (re.compile(r"\btouch\b", re.IGNORECASE), 0.3),
    (re.compile(r"\bmkdir\b", re.IGNORECASE), 0.3),
    # Read-only (0.0) — explicitly scored to override partial matches
    (re.compile(r"\bcat\s", re.IGNORECASE), 0.0),
    (re.compile(r"\bls\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgrep\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bfind\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bhead\b", re.IGNORECASE), 0.0),
    (re.compile(r"\btail\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bwc\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgit\s+log\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgit\s+status\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgit\s+diff\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bread\b", re.IGNORECASE), 0.0),
]

# Network exposure: (regex, score) — take max match
_NETWORK_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Data exfiltration / auth to external
    (re.compile(r"\bcurl\b.*\bAuthorization:", re.IGNORECASE), 1.0),
    (re.compile(r"\bcurl\b.*\bBearer\s", re.IGNORECASE), 1.0),
    # Write to external API
    (re.compile(r"\bcurl\s.*-X\s*(?:POST|PUT)\b", re.IGNORECASE), 0.9),
    (re.compile(r"\bssh\b", re.IGNORECASE), 0.9),
    # Read from network / push
    (re.compile(r"\bgit\s+push\b", re.IGNORECASE), 0.7),
    (re.compile(r"\bcurl\b", re.IGNORECASE), 0.7),
    (re.compile(r"\bwget\b", re.IGNORECASE), 0.7),
    # Package install (external fetch)
    (re.compile(r"\bpip\s+install\b", re.IGNORECASE), 0.5),
    (re.compile(r"\bnpm\s+install\b", re.IGNORECASE), 0.5),
    # Read-only network
    (re.compile(r"\bgit\s+fetch\b", re.IGNORECASE), 0.3),
    (re.compile(r"\bgit\s+pull\b", re.IGNORECASE), 0.3),
]

# Execution privilege: (regex, score) — take max match
_PRIVILEGE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Root-level
    (re.compile(r"\bsu\s+root\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bpasswd\b", re.IGNORECASE), 1.0),
    (re.compile(r"\bvisudo\b", re.IGNORECASE), 1.0),
    # Dangerous sudo combinations
    (re.compile(r"\bsudo\s+rm\s+-rf\b", re.IGNORECASE), 0.9),
    (re.compile(r"\bsudo\s+chmod\s+777\b", re.IGNORECASE), 0.9),
    # Write to system paths (detect even without sudo)
    (re.compile(r">\s*/etc/", re.IGNORECASE), 0.9),
    (re.compile(r">\s*/sys/", re.IGNORECASE), 0.9),
    # General sudo
    (re.compile(r"\bsudo\s", re.IGNORECASE), 0.7),
    # Permission changes
    (re.compile(r"\bchmod\b", re.IGNORECASE), 0.4),
    (re.compile(r"\bchown\b", re.IGNORECASE), 0.4),
]

# Code generation ratio: (regex, score) — take max match
_CODEGEN_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # Inline code execution
    (re.compile(r"\bpython3?\s+-c\s", re.IGNORECASE), 1.0),
    # Writing new files with code content
    (re.compile(r"\becho\s+[\"'].*import\b", re.IGNORECASE), 1.0),
    (re.compile(r"\becho\s+[\"'].*def\s", re.IGNORECASE), 1.0),
    (re.compile(r"\becho\s+[\"'].*class\s", re.IGNORECASE), 1.0),
    (re.compile(r">>\s*\S+\.py\b", re.IGNORECASE), 0.7),
    # In-place code modification
    (re.compile(r"\bsed\s+-i\b", re.IGNORECASE), 0.7),
    (re.compile(r"\becho\s.*>>", re.IGNORECASE), 0.7),
    (re.compile(r"\becho\s.*>(?!>)", re.IGNORECASE), 0.7),
    # Saving code changes
    (re.compile(r"\bgit\s+commit\b", re.IGNORECASE), 0.5),
    # Executing existing code
    (re.compile(r"\bpython3?\s+\S+\.py\b", re.IGNORECASE), 0.3),
    (re.compile(r"\bpython3?\s+-m\s", re.IGNORECASE), 0.3),
    # Read-only code operations (0.0)
    (re.compile(r"\bcat\s+\S+\.py\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgrep\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bfind\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bls\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bread\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgit\s+log\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgit\s+diff\b", re.IGNORECASE), 0.0),
    (re.compile(r"\bgit\s+status\b", re.IGNORECASE), 0.0),
]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_max(text: str, patterns: List[Tuple[re.Pattern, float]]) -> Tuple[float, List[str]]:
    """Return (max_score, matched_pattern_strings) from a pattern list."""
    best = 0.0
    matched: List[str] = []
    for pat, score in patterns:
        if pat.search(text):
            matched.append(pat.pattern)
            if score > best:
                best = score
    return best, matched


def _score_additive(
    text: str, patterns: List[Tuple[re.Pattern, float]], cap: float = 1.0
) -> Tuple[float, List[str]]:
    """Return (sum_of_weights_capped, matched_pattern_strings)."""
    total = 0.0
    matched: List[str] = []
    for pat, weight in patterns:
        if pat.search(text):
            total += weight
            matched.append(pat.pattern)
    return min(total, cap), matched


# ---------------------------------------------------------------------------
# BehavioralExtractor
# ---------------------------------------------------------------------------


class BehavioralExtractor:
    """Rule-based parser for observable agent action features.

    All 6 dimensions are deterministically computable from action text.
    No embedding model required. Each dimension is in [0.0, 1.0] with
    meaningful magnitudes — a benign action returns near-zero, a dangerous
    action returns high values. No L2 normalization is applied.
    """

    CONCEPTS = CONCEPTS

    def extract(self, text: str) -> Dict[str, float]:
        """Returns 6-dim vector with each dim in [0, 1]. Deterministic."""
        return self._raw_scores(text)

    def extract_with_detail(self, text: str) -> Dict[str, Any]:
        """Returns scores with matched patterns for debugging."""
        scores = self._raw_scores(text)
        detail = self._raw_detail(text)

        vec = np.array([scores[c] for c in CONCEPTS], dtype=np.float64)
        l2 = float(np.linalg.norm(vec))

        return {
            "scores": scores,
            "l2_norm": round(l2, 6),
            "matched_patterns": detail,
        }

    # -- internal scoring ----------------------------------------------------

    def _raw_scores(self, text: str) -> Dict[str, float]:
        """Compute raw (unnormalized) scores for each dimension."""
        fs, _ = _score_max(text, _FILESYSTEM_PATTERNS)
        isen, _ = _score_additive(text, _SENSITIVITY_PATTERNS, cap=1.0)
        rev, _ = _score_max(text, _REVERSIBILITY_PATTERNS)
        net, _ = _score_max(text, _NETWORK_PATTERNS)
        priv, _ = _score_max(text, _PRIVILEGE_PATTERNS)
        cg, _ = _score_max(text, _CODEGEN_PATTERNS)

        return {
            "filesystem_scope": round(fs, 4),
            "information_sensitivity": round(isen, 4),
            "action_reversibility": round(rev, 4),
            "network_exposure": round(net, 4),
            "execution_privilege": round(priv, 4),
            "code_generation_ratio": round(cg, 4),
        }

    def _raw_detail(self, text: str) -> Dict[str, List[str]]:
        """Return matched pattern strings per dimension."""
        _, fs_m = _score_max(text, _FILESYSTEM_PATTERNS)
        _, is_m = _score_additive(text, _SENSITIVITY_PATTERNS, cap=1.0)
        _, rev_m = _score_max(text, _REVERSIBILITY_PATTERNS)
        _, net_m = _score_max(text, _NETWORK_PATTERNS)
        _, priv_m = _score_max(text, _PRIVILEGE_PATTERNS)
        _, cg_m = _score_max(text, _CODEGEN_PATTERNS)

        return {
            "filesystem_scope": fs_m,
            "information_sensitivity": is_m,
            "action_reversibility": rev_m,
            "network_exposure": net_m,
            "execution_privilege": priv_m,
            "code_generation_ratio": cg_m,
        }
