"""Tests for the full pipeline integration."""

import numpy as np
import pytest
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
