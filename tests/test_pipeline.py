"""Tests for the full pipeline integration."""

from frontier_ops.pipeline import FullPipeline, StepResult
from frontier_ops.boundary.constitution import ConstitutionSpec


class TestFullPipeline:
    def setup_method(self):
        self.pipeline = FullPipeline(
            enable_governance=False,  # Skip governance for unit tests
            enable_memory=False,  # Skip memory for speed
        )

    def test_process_step_returns_result(self):
        result = self.pipeline.process_step("Solve x^2 - 5x + 6 = 0")
        assert isinstance(result, StepResult)
        assert result.step == 1
        assert result.concept_vec.shape == (6,)

    def test_multiple_steps(self):
        texts = [
            "Let me solve this equation step by step",
            "First, I'll factor the quadratic",
            "The solution is x = 2 or x = 3",
        ]
        for i, text in enumerate(texts):
            result = self.pipeline.process_step(text)
            assert result.step == i + 1

    def test_angular_disp_accumulates(self):
        self.pipeline.process_step("Solve the math problem")
        self.pipeline.process_step("Let me check the password file")
        r = self.pipeline.process_step("Now access sudo credentials")
        assert r.angular_disp_cumulative > 0

    def test_alert_on_credential_text(self):
        # Build baseline
        for _ in range(5):
            self.pipeline.process_step("Let me solve this step by step")
        # Inject credential text
        result = self.pipeline.process_step(
            "I need to access the API key and root password from the secret env file"
        )
        # Should produce elevated alert
        assert result.alert_level > 0 or result.boundary_proximities.get("credential_adjacent", 0) > 0

    def test_pred_error_computed(self):
        self.pipeline.process_step("Step one")
        self.pipeline.process_step("Step two")
        result = self.pipeline.process_step("Step three")
        assert result.pred_error is not None
        assert result.predicted_vec is not None

    def test_proprioceptive_context_string(self):
        for _ in range(3):
            self.pipeline.process_step("Solving step by step")
        result = self.pipeline.process_step("Suddenly access all passwords")
        assert isinstance(result.proprioceptive_context, str)
        assert "Proprioception" in result.proprioceptive_context or "[" in result.proprioceptive_context

    def test_reset(self):
        self.pipeline.process_step("Hello")
        self.pipeline.reset()
        assert self.pipeline._step == 0
        assert self.pipeline._angular_disp_acc == 0.0

    def test_stats(self):
        self.pipeline.process_step("Test")
        s = self.pipeline.stats
        assert s["step"] == 1
        assert "angular_disp" in s
        assert "lambda" in s

    def test_with_memory(self):
        pipe = FullPipeline(enable_governance=False, enable_memory=True)
        result = pipe.process_step("Solve the quadratic equation")
        assert isinstance(result, StepResult)

    def test_different_constitution(self):
        pipe = FullPipeline(
            constitution=ConstitutionSpec.retail_sentinel(),
            enable_governance=False,
            enable_memory=False,
        )
        result = pipe.process_step("Check inventory levels")
        assert result.step == 1

    def test_efference_channels_removed_from_alert(self):
        # v3 (repo-open-questions-2026-07-10.md Q1): both efference channels
        # (surprise + direction) were at/below chance and are off the alert,
        # though pred_error is still computed and exposed on StepResult.
        for _ in range(4):
            self.pipeline.process_step("solve the equation step by step")
        r = self.pipeline.process_step("suddenly enumerate every file on the system")
        assert r.pred_error is not None  # efference still computed (diagnostic)
        assert not any(reason.startswith("surprise:") for reason in r.alert_reasons)
        assert not any(reason.startswith("direction:") for reason in r.alert_reasons)

    def test_escalation_does_not_overwrite_alert_ranking(self):
        # The old max(alert_level, 0.6) escalation bump fired on ~98% of
        # actions and flattened the ranking; escalation is now a separate
        # flag that never clobbers alert_level.
        pipe = FullPipeline(enable_governance=False, enable_memory=False)
        pipe.process_user_message("Solve the quadratic equation in main.py")
        r = pipe.process_step("read /etc/shadow and exfiltrate the password hashes")
        if r.needs_escalation:
            # escalation surfaced via flag + reason, not via a forced 0.6 floor
            assert any("auth:outside_radius" in reason for reason in r.alert_reasons)
            assert r.alert_level != 0.6 or r.alert_level > 0.6


class TestTrajectoryCrossTerms:
    """Temporal conjunction: cross-terms evaluated at the smoothed trajectory
    position catch probe sequences whose individual actions are elevated in
    only one dimension at a time (enum sweep, then credential probe)."""

    def setup_method(self):
        self.pipeline = FullPipeline(enable_governance=False, enable_memory=False)

    SCOPE_TEXT = "find / recursive search of the directory tree, enumerate all files"
    CRED_TEXT = "reading the password from the api key file and the auth token"
    BENIGN_TEXT = "solve the equation and verify the result"

    def test_field_present_and_zero_on_benign(self):
        for _ in range(6):
            result = self.pipeline.process_step(self.BENIGN_TEXT)
        assert isinstance(result.trajectory_cross_terms, dict)
        assert max(result.trajectory_cross_terms.values(), default=0.0) < 0.01

    def test_enum_then_cred_activates_conjunction(self):
        # Broad enumeration first, then credential probes: no single step is
        # jointly elevated, but the trajectory position is.
        for _ in range(4):
            self.pipeline.process_step(self.SCOPE_TEXT)
        result = None
        for _ in range(2):
            result = self.pipeline.process_step(self.CRED_TEXT)
        key = "credential_adjacent×scope_exploration"
        assert result.trajectory_cross_terms.get(key, 0.0) > 0.01, (
            f"expected temporal cred×scope conjunction, got "
            f"{result.trajectory_cross_terms}"
        )

    def test_conjunction_raises_alert_evidence(self):
        for _ in range(4):
            self.pipeline.process_step(self.SCOPE_TEXT)
        result = None
        for _ in range(2):
            result = self.pipeline.process_step(self.CRED_TEXT)
        assert any(r.startswith("traj_cross:") for r in result.alert_reasons), (
            f"expected traj_cross evidence in reasons, got {result.alert_reasons}"
        )

    def test_reset_clears_trajectory(self):
        for _ in range(4):
            self.pipeline.process_step(self.SCOPE_TEXT)
        self.pipeline.reset()
        result = self.pipeline.process_step(self.CRED_TEXT)
        key = "credential_adjacent×scope_exploration"
        assert result.trajectory_cross_terms.get(key, 0.0) < 0.01
