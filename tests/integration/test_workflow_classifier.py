"""
Tests for the enterprise workflow classifier.
"""

import pytest

from frontier_ops.integration.workflow_classifier import (
    RiskLevel,
    WorkflowClassifier,
    WorkflowType,
    WORKFLOW_RISK,
)


@pytest.fixture
def clf():
    return WorkflowClassifier()


# ── Taxonomy & enums ─────────────────────────────────────────────────────


class TestWorkflowType:
    def test_all_types_defined(self):
        expected = {
            "CODE_DEVELOPMENT", "CODE_REVIEW", "TESTING", "DEPLOYMENT",
            "DOCUMENTATION", "SECURITY_AUDIT", "DATA_ANALYSIS",
            "SYSTEM_ADMINISTRATION", "COMMUNICATION", "RESEARCH", "UNKNOWN",
        }
        assert {wt.value for wt in WorkflowType} == expected

    def test_string_enum(self):
        assert WorkflowType.TESTING == "TESTING"
        assert str(WorkflowType.TESTING) == "WorkflowType.TESTING"

    def test_risk_levels(self):
        assert set(RiskLevel) == {
            RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL
        }

    def test_every_workflow_has_risk(self):
        for wt in WorkflowType:
            assert wt in WORKFLOW_RISK, f"Missing risk for {wt}"


# ── Exec classification ─────────────────────────────────────────────────


class TestExecClassification:
    def test_pytest(self, clf):
        r = clf.classify("exec", {"command": "pytest tests/ -v"})
        assert r.workflow == WorkflowType.TESTING
        assert r.confidence >= 0.8

    def test_pip_install(self, clf):
        r = clf.classify("exec", {"command": "pip install requests"})
        assert r.workflow == WorkflowType.DEPLOYMENT

    def test_docker(self, clf):
        r = clf.classify("exec", {"command": "docker build -t myapp ."})
        assert r.workflow == WorkflowType.DEPLOYMENT

    def test_kubectl(self, clf):
        r = clf.classify("exec", {"command": "kubectl apply -f deploy.yaml"})
        assert r.workflow == WorkflowType.DEPLOYMENT

    def test_terraform(self, clf):
        r = clf.classify("exec", {"command": "terraform plan"})
        assert r.workflow == WorkflowType.DEPLOYMENT

    def test_ssh_firewall(self, clf):
        r = clf.classify("exec", {"command": "ssh server 'ufw allow 80'"})
        assert r.workflow == WorkflowType.SYSTEM_ADMINISTRATION
        assert r.confidence >= 0.9

    def test_ssh_alone(self, clf):
        r = clf.classify("exec", {"command": "ssh user@server"})
        assert r.workflow == WorkflowType.SYSTEM_ADMINISTRATION

    def test_apt_install(self, clf):
        r = clf.classify("exec", {"command": "apt install nginx"})
        assert r.workflow == WorkflowType.SYSTEM_ADMINISTRATION

    def test_git_commit(self, clf):
        r = clf.classify("exec", {"command": "git commit -m 'fix bug'"})
        assert r.workflow == WorkflowType.CODE_DEVELOPMENT

    def test_git_diff(self, clf):
        r = clf.classify("exec", {"command": "git diff HEAD~3"})
        assert r.workflow == WorkflowType.CODE_REVIEW

    def test_gh_pr_review(self, clf):
        r = clf.classify("exec", {"command": "gh pr review 42 --approve"})
        assert r.workflow == WorkflowType.CODE_REVIEW

    def test_bandit(self, clf):
        r = clf.classify("exec", {"command": "bandit -r src/"})
        assert r.workflow == WorkflowType.SECURITY_AUDIT

    def test_npm_audit(self, clf):
        r = clf.classify("exec", {"command": "npm audit"})
        assert r.workflow == WorkflowType.SECURITY_AUDIT

    def test_go_test(self, clf):
        r = clf.classify("exec", {"command": "go test ./..."})
        assert r.workflow == WorkflowType.TESTING

    def test_jest(self, clf):
        r = clf.classify("exec", {"command": "npx jest --coverage"})
        assert r.workflow == WorkflowType.TESTING

    def test_pandas_script(self, clf):
        r = clf.classify("exec", {"command": "python -c 'import pandas; pandas.read_csv(\"x.csv\")'"})
        assert r.workflow == WorkflowType.DATA_ANALYSIS

    def test_vercel_deploy(self, clf):
        r = clf.classify("exec", {"command": "vercel --prod"})
        assert r.workflow == WorkflowType.DEPLOYMENT

    def test_mkdocs(self, clf):
        r = clf.classify("exec", {"command": "mkdocs build"})
        assert r.workflow == WorkflowType.DOCUMENTATION

    def test_unknown_exec(self, clf):
        r = clf.classify("exec", {"command": "echo hello"})
        assert r.workflow == WorkflowType.UNKNOWN

    def test_empty_command(self, clf):
        r = clf.classify("exec", {"command": ""})
        assert r.workflow == WorkflowType.UNKNOWN


# ── Read classification ──────────────────────────────────────────────────


class TestReadClassification:
    def test_python_source(self, clf):
        r = clf.classify("read", {"file_path": "src/main.py"})
        assert r.workflow == WorkflowType.CODE_REVIEW

    def test_javascript(self, clf):
        r = clf.classify("read", {"path": "app/index.tsx"})
        assert r.workflow == WorkflowType.CODE_REVIEW

    def test_markdown(self, clf):
        r = clf.classify("read", {"file_path": "README.md"})
        assert r.workflow == WorkflowType.DOCUMENTATION

    def test_csv(self, clf):
        r = clf.classify("read", {"file_path": "data/sales.csv"})
        assert r.workflow == WorkflowType.DATA_ANALYSIS

    def test_test_file(self, clf):
        r = clf.classify("read", {"file_path": "tests/test_main.py"})
        assert r.workflow == WorkflowType.TESTING

    def test_spec_file(self, clf):
        r = clf.classify("read", {"file_path": "src/app.spec.ts"})
        assert r.workflow == WorkflowType.TESTING

    def test_unknown_file(self, clf):
        r = clf.classify("read", {"file_path": "some_file"})
        assert r.workflow == WorkflowType.CODE_REVIEW  # default fallback


# ── Write classification ─────────────────────────────────────────────────


class TestWriteClassification:
    def test_python_file(self, clf):
        r = clf.classify("write", {"file_path": "src/utils.py"})
        assert r.workflow == WorkflowType.CODE_DEVELOPMENT

    def test_edit_go_file(self, clf):
        r = clf.classify("edit", {"file_path": "main.go"})
        assert r.workflow == WorkflowType.CODE_DEVELOPMENT

    def test_test_file(self, clf):
        r = clf.classify("write", {"file_path": "tests/test_auth.py"})
        assert r.workflow == WorkflowType.TESTING

    def test_markdown(self, clf):
        r = clf.classify("write", {"file_path": "docs/API.md"})
        assert r.workflow == WorkflowType.DOCUMENTATION

    def test_dockerfile(self, clf):
        r = clf.classify("write", {"file_path": "Dockerfile"})
        assert r.workflow == WorkflowType.DEPLOYMENT

    def test_yaml_config(self, clf):
        r = clf.classify("write", {"file_path": "config/app.yaml"})
        assert r.workflow == WorkflowType.SYSTEM_ADMINISTRATION

    def test_data_file(self, clf):
        r = clf.classify("write", {"file_path": "output/results.json"})
        assert r.workflow == WorkflowType.DATA_ANALYSIS


# ── Direct tool mappings ─────────────────────────────────────────────────


class TestDirectToolMappings:
    def test_message(self, clf):
        r = clf.classify("message", {"action": "send", "message": "hello"})
        assert r.workflow == WorkflowType.COMMUNICATION
        assert r.confidence >= 0.9

    def test_tts(self, clf):
        r = clf.classify("tts", {"text": "hello world"})
        assert r.workflow == WorkflowType.COMMUNICATION

    def test_web_search(self, clf):
        r = clf.classify("web_search", {"query": "python decorators"})
        assert r.workflow == WorkflowType.RESEARCH

    def test_web_fetch_arxiv(self, clf):
        r = clf.classify("web_fetch", {"url": "https://arxiv.org/abs/2301.01234"})
        assert r.workflow == WorkflowType.RESEARCH
        assert r.confidence >= 0.9

    def test_web_fetch_pr(self, clf):
        r = clf.classify("web_fetch", {"url": "https://github.com/org/repo/pull/42"})
        assert r.workflow == WorkflowType.CODE_REVIEW

    def test_web_fetch_docs(self, clf):
        r = clf.classify("web_fetch", {"url": "https://docs.python.org/3/library/re.html"})
        assert r.workflow == WorkflowType.DOCUMENTATION

    def test_web_fetch_cve(self, clf):
        r = clf.classify("web_fetch", {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-1234"})
        assert r.workflow == WorkflowType.SECURITY_AUDIT

    def test_web_fetch_generic(self, clf):
        r = clf.classify("web_fetch", {"url": "https://example.com"})
        assert r.workflow == WorkflowType.RESEARCH

    def test_browser(self, clf):
        r = clf.classify("browser", {"url": "https://arxiv.org"})
        assert r.workflow == WorkflowType.RESEARCH

    def test_image(self, clf):
        r = clf.classify("image", {"image": "report_chart.png"})
        assert r.workflow == WorkflowType.DATA_ANALYSIS

    def test_unknown_tool(self, clf):
        r = clf.classify("some_new_tool", {})
        assert r.workflow == WorkflowType.UNKNOWN


# ── Risk levels ──────────────────────────────────────────────────────────


class TestRiskLevels:
    def test_deployment_is_high(self, clf):
        r = clf.classify("exec", {"command": "docker push myapp"})
        assert r.risk_level == RiskLevel.HIGH

    def test_testing_is_low(self, clf):
        r = clf.classify("exec", {"command": "pytest"})
        assert r.risk_level == RiskLevel.LOW

    def test_code_review_is_low(self, clf):
        r = clf.classify("read", {"file_path": "main.py"})
        assert r.risk_level == RiskLevel.LOW

    def test_sysadmin_is_high(self, clf):
        r = clf.classify("exec", {"command": "ssh server 'iptables -A INPUT -p tcp --dport 22 -j ACCEPT'"})
        assert r.risk_level == RiskLevel.HIGH

    def test_communication_is_medium(self, clf):
        r = clf.classify("message", {"message": "hi"})
        assert r.risk_level == RiskLevel.MEDIUM


# ── Workflow transitions ─────────────────────────────────────────────────


class TestTransitions:
    def test_notable_transition_detected(self, clf):
        clf.observe("write", {"file_path": "app.py"})
        assert clf.current_workflow == WorkflowType.CODE_DEVELOPMENT
        assert len(clf.transitions) == 0

        clf.observe("exec", {"command": "ssh server 'ufw status'"})
        assert clf.current_workflow == WorkflowType.SYSTEM_ADMINISTRATION
        assert len(clf.transitions) == 1
        assert clf.transitions[0].from_workflow == WorkflowType.CODE_DEVELOPMENT
        assert clf.transitions[0].to_workflow == WorkflowType.SYSTEM_ADMINISTRATION
        assert "scope escalation" in clf.transitions[0].description

    def test_same_workflow_no_transition(self, clf):
        clf.observe("write", {"file_path": "a.py"})
        clf.observe("write", {"file_path": "b.py"})
        assert len(clf.transitions) == 0

    def test_non_notable_transition_no_record(self, clf):
        clf.observe("exec", {"command": "pytest"})
        clf.observe("write", {"file_path": "app.py"})
        # TESTING → CODE_DEVELOPMENT is not in NOTABLE_TRANSITIONS
        assert len(clf.transitions) == 0

    def test_review_to_deploy(self, clf):
        clf.observe("read", {"file_path": "main.py"})
        clf.observe("exec", {"command": "docker push myapp"})
        assert len(clf.transitions) == 1
        assert "skipped testing" in clf.transitions[0].description


# ── Session profile ──────────────────────────────────────────────────────


class TestSessionProfile:
    def test_empty_profile(self, clf):
        assert clf.session_profile() == {}

    def test_single_action(self, clf):
        clf.observe("exec", {"command": "pytest"})
        profile = clf.session_profile()
        assert profile == {"TESTING": 100.0}

    def test_mixed_session(self, clf):
        clf.observe("write", {"file_path": "app.py"})       # CODE_DEVELOPMENT
        clf.observe("write", {"file_path": "utils.py"})      # CODE_DEVELOPMENT
        clf.observe("exec", {"command": "pytest"})            # TESTING
        clf.observe("message", {"message": "done"})           # COMMUNICATION

        profile = clf.session_profile()
        assert profile["CODE_DEVELOPMENT"] == 50.0
        assert profile["TESTING"] == 25.0
        assert profile["COMMUNICATION"] == 25.0
        assert len(profile) == 3

    def test_reset(self, clf):
        clf.observe("exec", {"command": "pytest"})
        assert len(clf.history) == 1
        clf.reset()
        assert len(clf.history) == 0
        assert len(clf.transitions) == 0
        assert clf.session_profile() == {}

    def test_current_workflow_none_initially(self, clf):
        assert clf.current_workflow is None

    def test_history_is_copy(self, clf):
        clf.observe("exec", {"command": "pytest"})
        h = clf.history
        h.clear()
        assert len(clf.history) == 1  # original untouched


# ── Classification result dataclass ──────────────────────────────────────


class TestClassificationResult:
    def test_has_timestamp(self, clf):
        r = clf.classify("exec", {"command": "pytest"})
        assert r.timestamp > 0

    def test_fields(self, clf):
        r = clf.classify("message", {"message": "hi"})
        assert r.tool == "message"
        assert r.workflow == WorkflowType.COMMUNICATION
        assert r.risk_level == RiskLevel.MEDIUM
        assert isinstance(r.reason, str)
        assert isinstance(r.confidence, float)

    def test_no_params(self, clf):
        """Passing None params shouldn't crash."""
        r = clf.classify("exec", None)
        assert r.workflow == WorkflowType.UNKNOWN
