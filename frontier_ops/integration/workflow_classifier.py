"""
Enterprise Workflow Classifier
==============================

Categorizes agent actions into standard enterprise workflow types
based on tool type, content keywords, and contextual signals.

Non-invasive: classifies only, never blocks or modifies actions.

Usage:
    from frontier_ops.integration.workflow_classifier import WorkflowClassifier

    clf = WorkflowClassifier()
    result = clf.classify("exec", {"command": "pytest tests/ -v"})
    # result.workflow == WorkflowType.TESTING
    # result.risk_level == RiskLevel.LOW

    # Track a session
    clf.observe("write", {"file_path": "app.py"})
    clf.observe("exec", {"command": "pytest"})
    profile = clf.session_profile()
    # {"CODE_DEVELOPMENT": 50.0, "TESTING": 50.0}
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Workflow Taxonomy ────────────────────────────────────────────────────


class WorkflowType(str, Enum):
    """Standard enterprise workflow categories."""

    CODE_DEVELOPMENT = "CODE_DEVELOPMENT"
    CODE_REVIEW = "CODE_REVIEW"
    TESTING = "TESTING"
    DEPLOYMENT = "DEPLOYMENT"
    DOCUMENTATION = "DOCUMENTATION"
    SECURITY_AUDIT = "SECURITY_AUDIT"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    SYSTEM_ADMINISTRATION = "SYSTEM_ADMINISTRATION"
    COMMUNICATION = "COMMUNICATION"
    RESEARCH = "RESEARCH"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    """Risk levels for workflow types."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Risk mapping per workflow type ───────────────────────────────────────

WORKFLOW_RISK: Dict[WorkflowType, RiskLevel] = {
    WorkflowType.CODE_DEVELOPMENT: RiskLevel.MEDIUM,
    WorkflowType.CODE_REVIEW: RiskLevel.LOW,
    WorkflowType.TESTING: RiskLevel.LOW,
    WorkflowType.DEPLOYMENT: RiskLevel.HIGH,
    WorkflowType.DOCUMENTATION: RiskLevel.LOW,
    WorkflowType.SECURITY_AUDIT: RiskLevel.MEDIUM,
    WorkflowType.DATA_ANALYSIS: RiskLevel.LOW,
    WorkflowType.SYSTEM_ADMINISTRATION: RiskLevel.HIGH,
    WorkflowType.COMMUNICATION: RiskLevel.MEDIUM,
    WorkflowType.RESEARCH: RiskLevel.LOW,
    WorkflowType.UNKNOWN: RiskLevel.MEDIUM,
}

# Notable workflow transitions that deserve attention
NOTABLE_TRANSITIONS: Dict[Tuple[WorkflowType, WorkflowType], str] = {
    (WorkflowType.CODE_DEVELOPMENT, WorkflowType.SYSTEM_ADMINISTRATION): (
        "Switched from coding to system administration — possible scope escalation"
    ),
    (WorkflowType.CODE_REVIEW, WorkflowType.DEPLOYMENT): (
        "Jumped from review directly to deployment — skipped testing?"
    ),
    (WorkflowType.RESEARCH, WorkflowType.DEPLOYMENT): (
        "Went from research to deployment — no development phase"
    ),
    (WorkflowType.CODE_DEVELOPMENT, WorkflowType.DEPLOYMENT): (
        "Deploying directly after development — consider testing first"
    ),
    (WorkflowType.TESTING, WorkflowType.SECURITY_AUDIT): (
        "Shifted from testing to security audit — may indicate discovered vulnerability"
    ),
    (WorkflowType.COMMUNICATION, WorkflowType.SYSTEM_ADMINISTRATION): (
        "Moved from communication to sysadmin — social engineering risk"
    ),
    (WorkflowType.DATA_ANALYSIS, WorkflowType.COMMUNICATION): (
        "Data analysis followed by communication — check for data exfiltration"
    ),
}


# ── Classification result ────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Result of classifying an action into a workflow type."""

    workflow: WorkflowType
    risk_level: RiskLevel
    confidence: float  # 0.0–1.0
    reason: str
    tool: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowTransition:
    """Records a notable transition between workflow types."""

    from_workflow: WorkflowType
    to_workflow: WorkflowType
    description: str
    timestamp: float = field(default_factory=time.time)


# ── Keyword patterns per workflow ────────────────────────────────────────

# (compiled_regex, workflow_type, confidence_boost)
_EXEC_PATTERNS: List[Tuple[re.Pattern, WorkflowType, float]] = [
    # TESTING
    (re.compile(r"\bpytest\b|\bunittest\b|\bnose2?\b|\btox\b|\bcoverage\b"), WorkflowType.TESTING, 0.9),
    (re.compile(r"\bnpm\s+test\b|\bjest\b|\bmocha\b|\bkarma\b|\bvitest\b"), WorkflowType.TESTING, 0.9),
    (re.compile(r"\bgo\s+test\b|\bcargo\s+test\b"), WorkflowType.TESTING, 0.9),
    (re.compile(r"\bmake\s+test\b|\bmake\s+check\b"), WorkflowType.TESTING, 0.85),
    # DEPLOYMENT
    (re.compile(r"\bdocker\b|\bpodman\b|\bkubectl\b|\bhelm\b"), WorkflowType.DEPLOYMENT, 0.9),
    (re.compile(r"\bpip\s+install\b|\bnpm\s+install\b|\byarn\s+add\b"), WorkflowType.DEPLOYMENT, 0.8),
    (re.compile(r"\bterraform\b|\bansible\b|\bpulumi\b|\bcdk\b"), WorkflowType.DEPLOYMENT, 0.9),
    (re.compile(r"\bvercel\b|\bheroku\b|\baws\b|\bgcloud\b|\baz\b"), WorkflowType.DEPLOYMENT, 0.85),
    (re.compile(r"\bsystemctl\s+(start|stop|restart|enable)\b"), WorkflowType.DEPLOYMENT, 0.85),
    # SYSTEM_ADMINISTRATION
    (re.compile(r"\bssh\b.*\b(firewall|iptables|ufw)\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.95),
    (re.compile(r"\b(iptables|ufw|firewalld)\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.9),
    (re.compile(r"\bssh\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.7),
    (re.compile(r"\b(apt|yum|dnf|brew|pacman)\s+(install|update|upgrade|remove)\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.8),
    (re.compile(r"\bchmod\b|\bchown\b|\buseradd\b|\bpasswd\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.85),
    (re.compile(r"\bnetstat\b|\bss\b|\blsof\b.*-i\b|\bnmap\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.8),
    (re.compile(r"\bcrontab\b|\bsystemctl\s+(status|list)\b|\bjournalctl\b"), WorkflowType.SYSTEM_ADMINISTRATION, 0.8),
    # SECURITY_AUDIT
    (re.compile(r"\bbandit\b|\bsemgrep\b|\bsnyk\b|\btrivy\b|\bgrype\b"), WorkflowType.SECURITY_AUDIT, 0.9),
    (re.compile(r"\bnmap\b.*(-sV|-sS|-A)\b|\bnikto\b|\bzap\b"), WorkflowType.SECURITY_AUDIT, 0.9),
    (re.compile(r"\baudit\b|\bvuln(erab)?\b|\bcve\b", re.IGNORECASE), WorkflowType.SECURITY_AUDIT, 0.7),
    (re.compile(r"\bsafety\s+check\b|\bpip-audit\b|\bnpm\s+audit\b"), WorkflowType.SECURITY_AUDIT, 0.9),
    # DATA_ANALYSIS
    (re.compile(r"\bpandas\b|\bnumpy\b|\bmatplotlib\b|\bjupyter\b"), WorkflowType.DATA_ANALYSIS, 0.85),
    (re.compile(r"\bawk\b|\bsed\b.*\bgrep\b|\bwc\b|\bsort\b|\buniq\b"), WorkflowType.DATA_ANALYSIS, 0.6),
    (re.compile(r"\bsql\b|\bsqlite3?\b|\bpsql\b|\bmysql\b", re.IGNORECASE), WorkflowType.DATA_ANALYSIS, 0.8),
    # CODE_DEVELOPMENT
    (re.compile(r"\bgit\s+(commit|push|merge|rebase)\b"), WorkflowType.CODE_DEVELOPMENT, 0.8),
    (re.compile(r"\bpython\b.*\.py\b|\bnode\b.*\.js\b|\bgo\s+build\b"), WorkflowType.CODE_DEVELOPMENT, 0.7),
    # CODE_REVIEW
    (re.compile(r"\bgit\s+(diff|log|blame|show)\b"), WorkflowType.CODE_REVIEW, 0.8),
    (re.compile(r"\bgh\s+pr\s+(review|view|diff)\b"), WorkflowType.CODE_REVIEW, 0.9),
    # DOCUMENTATION
    (re.compile(r"\bmkdocs\b|\bsphinx\b|\bjsdoc\b|\btypedoc\b"), WorkflowType.DOCUMENTATION, 0.9),
    # RESEARCH
    (re.compile(r"\bcurl\b.*arxiv\b|\bwget\b.*paper\b"), WorkflowType.RESEARCH, 0.8),
]

# File extension → workflow type for read/write tools
_SOURCE_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".zig", ".hs", ".ml", ".ex", ".exs",
}
_DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".tex"}
_TEST_PATH_PATTERNS = re.compile(r"(test_|_test\.|\.test\.|spec\.|tests/|__tests__/)")
_DATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".xlsx", ".tsv", ".sql"}


# ── Classifier ───────────────────────────────────────────────────────────


class WorkflowClassifier:
    """
    Classifies agent tool calls into enterprise workflow types.

    Maintains session history for transition tracking and profile computation.
    """

    def __init__(self) -> None:
        self._history: List[ClassificationResult] = []
        self._transitions: List[WorkflowTransition] = []

    # ── Core classification ──────────────────────────────────────────

    def classify(
        self,
        tool: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        Classify a tool call into a workflow type.

        Args:
            tool: Tool name (exec, read, write, edit, message, etc.)
            params: Tool parameters dict.

        Returns:
            ClassificationResult with workflow type, risk, and confidence.
        """
        params = params or {}
        tool_lower = tool.lower()

        workflow, confidence, reason = self._classify_impl(tool_lower, params)
        risk = WORKFLOW_RISK[workflow]

        return ClassificationResult(
            workflow=workflow,
            risk_level=risk,
            confidence=confidence,
            reason=reason,
            tool=tool,
        )

    def observe(
        self,
        tool: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        Classify and record an action in session history.

        Also detects notable transitions.
        """
        result = self.classify(tool, params)
        prev = self._history[-1] if self._history else None
        self._history.append(result)

        # Check for notable transition
        if prev and prev.workflow != result.workflow:
            key = (prev.workflow, result.workflow)
            if key in NOTABLE_TRANSITIONS:
                transition = WorkflowTransition(
                    from_workflow=prev.workflow,
                    to_workflow=result.workflow,
                    description=NOTABLE_TRANSITIONS[key],
                )
                self._transitions.append(transition)

        return result

    # ── Session analytics ────────────────────────────────────────────

    def session_profile(self) -> Dict[str, float]:
        """
        Compute percentage of session time spent in each workflow type.

        Returns dict mapping workflow name → percentage (0–100).
        """
        if not self._history:
            return {}

        counts: Dict[str, int] = {}
        for entry in self._history:
            name = entry.workflow.value
            counts[name] = counts.get(name, 0) + 1

        total = len(self._history)
        return {k: round((v / total) * 100, 1) for k, v in counts.items()}

    @property
    def history(self) -> List[ClassificationResult]:
        """All classified actions in this session."""
        return list(self._history)

    @property
    def transitions(self) -> List[WorkflowTransition]:
        """Notable workflow transitions detected."""
        return list(self._transitions)

    @property
    def current_workflow(self) -> Optional[WorkflowType]:
        """Most recent workflow type, or None if no history."""
        return self._history[-1].workflow if self._history else None

    def reset(self) -> None:
        """Clear session history."""
        self._history.clear()
        self._transitions.clear()

    # ── Internal classification logic ────────────────────────────────

    def _classify_impl(
        self, tool: str, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Returns (workflow, confidence, reason)."""

        # Direct tool-type mappings
        if tool == "message":
            return (WorkflowType.COMMUNICATION, 0.95, "message tool → communication")

        if tool == "tts":
            return (WorkflowType.COMMUNICATION, 0.8, "tts tool → communication")

        if tool in ("exec", "process"):
            return self._classify_exec(params)

        if tool in ("read",):
            return self._classify_read(params)

        if tool in ("write", "edit"):
            return self._classify_write(params)

        if tool == "web_search":
            return (WorkflowType.RESEARCH, 0.85, "web_search → research")

        if tool == "web_fetch":
            return self._classify_web_fetch(params)

        if tool in ("browser", "canvas"):
            return self._classify_browser(params)

        if tool in ("image", "pdf"):
            return self._classify_media(params)

        return (WorkflowType.UNKNOWN, 0.3, f"unrecognized tool: {tool}")

    def _classify_exec(
        self, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Classify exec/process commands."""
        command = params.get("command", "")

        # Match against keyword patterns (first match wins; patterns ordered by specificity)
        for pattern, wf, conf in _EXEC_PATTERNS:
            if pattern.search(command):
                return (wf, conf, f"exec command matches {wf.value} pattern")

        # Fallback: generic exec is unknown
        return (WorkflowType.UNKNOWN, 0.4, "exec command with no matching pattern")

    def _classify_read(
        self, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Classify file reads."""
        path = params.get("file_path", "") or params.get("path", "")

        if _TEST_PATH_PATTERNS.search(path):
            return (WorkflowType.TESTING, 0.7, "reading test file")

        ext = _get_extension(path)

        if ext in _SOURCE_CODE_EXTENSIONS:
            return (WorkflowType.CODE_REVIEW, 0.8, f"reading source code ({ext})")

        if ext in _DOC_EXTENSIONS:
            return (WorkflowType.DOCUMENTATION, 0.6, f"reading documentation ({ext})")

        if ext in _DATA_EXTENSIONS:
            return (WorkflowType.DATA_ANALYSIS, 0.7, f"reading data file ({ext})")

        return (WorkflowType.CODE_REVIEW, 0.4, "reading file (default to review)")

    def _classify_write(
        self, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Classify file writes/edits."""
        path = params.get("file_path", "") or params.get("path", "")

        if _TEST_PATH_PATTERNS.search(path):
            return (WorkflowType.TESTING, 0.8, "writing test file")

        ext = _get_extension(path)

        if ext in _SOURCE_CODE_EXTENSIONS:
            return (WorkflowType.CODE_DEVELOPMENT, 0.85, f"writing source code ({ext})")

        if ext in _DOC_EXTENSIONS:
            return (WorkflowType.DOCUMENTATION, 0.8, f"writing documentation ({ext})")

        if ext in _DATA_EXTENSIONS:
            return (WorkflowType.DATA_ANALYSIS, 0.7, f"writing data file ({ext})")

        # Config files
        if ext in (".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"):
            return (WorkflowType.SYSTEM_ADMINISTRATION, 0.6, f"writing config ({ext})")

        # Dockerfiles, CI configs
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        if basename.lower() in ("dockerfile", ".dockerignore", "docker-compose.yml", "docker-compose.yaml"):
            return (WorkflowType.DEPLOYMENT, 0.9, "writing Docker config")
        if basename.lower() in (".github", "jenkinsfile", ".gitlab-ci.yml", ".circleci"):
            return (WorkflowType.DEPLOYMENT, 0.85, "writing CI/CD config")

        return (WorkflowType.CODE_DEVELOPMENT, 0.4, "writing file (default to development)")

    def _classify_web_fetch(
        self, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Classify web fetches."""
        url = params.get("url", "")

        if "arxiv" in url or "papers" in url or "scholar.google" in url:
            return (WorkflowType.RESEARCH, 0.9, "fetching academic content")

        if "github.com" in url and "/pull/" in url:
            return (WorkflowType.CODE_REVIEW, 0.8, "fetching PR page")

        if "docs." in url or "documentation" in url or "readthedocs" in url:
            return (WorkflowType.DOCUMENTATION, 0.7, "fetching documentation")

        if "cve" in url.lower() or "nvd.nist" in url or "security" in url.lower():
            return (WorkflowType.SECURITY_AUDIT, 0.8, "fetching security advisory")

        return (WorkflowType.RESEARCH, 0.6, "web fetch (default to research)")

    def _classify_browser(
        self, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Classify browser/canvas actions."""
        url = params.get("url", "") or params.get("targetUrl", "")

        if "github.com" in url and "/pull/" in url:
            return (WorkflowType.CODE_REVIEW, 0.7, "browsing PR")

        if "arxiv" in url or "scholar" in url:
            return (WorkflowType.RESEARCH, 0.8, "browsing academic site")

        return (WorkflowType.RESEARCH, 0.5, "browser action (default to research)")

    def _classify_media(
        self, params: Dict[str, Any]
    ) -> Tuple[WorkflowType, float, str]:
        """Classify image/pdf analysis."""
        path = params.get("image", "") or params.get("pdf", "")

        if "report" in path.lower() or "chart" in path.lower():
            return (WorkflowType.DATA_ANALYSIS, 0.7, "analyzing report/chart")

        return (WorkflowType.RESEARCH, 0.5, "media analysis (default to research)")


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_extension(path: str) -> str:
    """Extract lowercase file extension from a path."""
    dot_idx = path.rfind(".")
    if dot_idx == -1:
        return ""
    return path[dot_idx:].lower()
