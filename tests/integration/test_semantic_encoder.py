"""Tests for semantic_encoder: MiniLM → phasor projection."""

import numpy as np

from frontier_ops.integration.semantic_encoder import encode_semantic


class TestEncodeSemanticBasic:
    """Basic shape, dtype, and crash-resistance tests."""

    def test_credential_command_shape(self):
        vec = encode_semantic("env | grep KEY")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (512,)
        assert vec.dtype == complex

    def test_test_command_shape(self):
        vec = encode_semantic("python -m pytest tests/")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (512,)
        assert vec.dtype == complex

    def test_empty_string_no_crash(self):
        vec = encode_semantic("")
        assert vec.shape == (512,)

    def test_very_long_string_no_crash(self):
        vec = encode_semantic("x" * 5000)
        assert vec.shape == (512,)

    def test_unit_magnitude(self):
        vec = encode_semantic("ls -la /tmp")
        norm = np.linalg.norm(vec)
        # Should be ~1.0 (or 0.0 if st unavailable)
        assert norm < 1e-9 or abs(norm - 1.0) < 1e-6


class TestCaching:
    """LRU cache behavior."""

    def test_same_input_returns_cached_object(self):
        a = encode_semantic("echo hello")
        b = encode_semantic("echo hello")
        # lru_cache returns the exact same object
        assert a is b

    def test_different_input_returns_different_object(self):
        a = encode_semantic("echo hello")
        b = encode_semantic("echo world")
        assert a is not b


class TestSemanticSeparation:
    """Verify that semantically similar commands cluster together
    and semantically different commands are separated."""

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.real(np.vdot(a, b)) / (na * nb))

    def test_similar_test_commands_more_similar_than_cross_category(self):
        """pytest ↔ 'python -m pytest' should be more similar
        than 'env | grep KEY' ↔ 'python -m pytest tests/'."""
        v_pytest = encode_semantic("pytest")
        v_pytest_full = encode_semantic("python -m pytest")
        v_env = encode_semantic("env | grep KEY")
        v_test = encode_semantic("python -m pytest tests/")

        sim_within = self.cosine(v_pytest, v_pytest_full)
        sim_cross = self.cosine(v_env, v_test)
        assert sim_within > sim_cross, (
            f"Within-category sim ({sim_within:.4f}) should exceed "
            f"cross-category sim ({sim_cross:.4f})"
        )

    def test_credential_vs_testing_no_near_duplicates(self):
        """No credential↔testing cross-pair should exceed 0.9 similarity."""
        credential_cmds = ["env | grep KEY", "cat ~/.ssh/id_rsa", "cat .env"]
        test_cmds = ["python -m pytest tests/", "pytest -v", "cargo test"]

        cred_vecs = [encode_semantic(c) for c in credential_cmds]
        test_vecs = [encode_semantic(c) for c in test_cmds]

        for i, cv in enumerate(cred_vecs):
            for j, tv in enumerate(test_vecs):
                sim = self.cosine(cv, tv)
                assert sim < 0.9, (
                    f"Cross-pair ({credential_cmds[i]!r}, {test_cmds[j]!r}) "
                    f"similarity {sim:.4f} >= 0.9"
                )

    def test_credential_commands_distinct_from_testing(self):
        """Mean within-credential similarity should exceed mean cross-category."""
        credential_cmds = ["env | grep KEY", "cat ~/.ssh/id_rsa", "cat .env"]
        test_cmds = ["python -m pytest tests/", "pytest -v", "cargo test"]

        cred_vecs = [encode_semantic(c) for c in credential_cmds]
        test_vecs = [encode_semantic(c) for c in test_cmds]

        # Within-credential similarities
        within_sims = []
        for i in range(len(cred_vecs)):
            for j in range(i + 1, len(cred_vecs)):
                within_sims.append(self.cosine(cred_vecs[i], cred_vecs[j]))

        # Cross-category similarities
        cross_sims = []
        for cv in cred_vecs:
            for tv in test_vecs:
                cross_sims.append(self.cosine(cv, tv))

        mean_within = np.mean(within_sims)
        mean_cross = np.mean(cross_sims)

        # Credential commands should cluster more tightly among themselves
        # than with testing commands
        assert mean_within > mean_cross, (
            f"Within-credential mean sim ({mean_within:.4f}) should exceed "
            f"cross-category mean sim ({mean_cross:.4f})"
        )
