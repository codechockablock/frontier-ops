"""Tests for constitutional manifold metric tensor."""

import numpy as np
import pytest
from frontier_ops.boundary.constitution import (
    ConstitutionSpec, ConstitutionalMetric, Boundary, softplus, softplus_derivative,
)
from frontier_ops.boundary.concept_extraction import CONCEPTS


class TestSoftplus:
    def test_positive_far(self):
        assert softplus(10.0, 5.0) == pytest.approx(10.0, abs=0.01)

    def test_negative_far(self):
        assert softplus(-10.0, 5.0) == pytest.approx(0.0, abs=0.01)

    def test_at_zero(self):
        assert softplus(0.0, 5.0) == pytest.approx(np.log(2) / 5.0, abs=0.01)

    def test_monotonic(self):
        vals = [softplus(x, 5.0) for x in np.linspace(-5, 5, 100)]
        assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))


class TestSoftplusDerivative:
    def test_sigmoid_shape(self):
        assert softplus_derivative(-10, 5.0) < 0.01
        assert softplus_derivative(0, 5.0) == pytest.approx(0.5, abs=0.01)
        assert softplus_derivative(10, 5.0) > 0.99


class TestConstitutionalMetric:
    def setup_method(self):
        self.spec = ConstitutionSpec.agent_safety_default()
        self.metric = ConstitutionalMetric(self.spec, dim_names=CONCEPTS)

    def test_tensor_shape(self):
        x = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        G = self.metric.tensor_at(x)
        assert G.shape == (6, 6)

    def test_tensor_symmetric(self):
        x = np.array([0.5, 0.3, 0.4, 0.2, 0.3, 0.2])
        G = self.metric.tensor_at(x)
        assert np.allclose(G, G.T)

    def test_tensor_positive_definite(self):
        x = np.array([0.5, 0.3, 0.4, 0.2, 0.3, 0.2])
        G = self.metric.tensor_at(x)
        eigenvalues = np.linalg.eigvalsh(G)
        assert all(v > 0 for v in eigenvalues)

    def test_far_from_boundaries_near_identity(self):
        x = np.zeros(6)  # Far below all thresholds
        G = self.metric.tensor_at(x)
        assert np.allclose(G, np.eye(6), atol=0.5)

    def test_near_boundary_amplified(self):
        far = np.zeros(6)
        near = np.zeros(6)
        near[2] = 0.8  # credential_adjacent above threshold
        G_far = self.metric.tensor_at(far)
        G_near = self.metric.tensor_at(near)
        # Credential dimension should be amplified near boundary
        assert G_near[2, 2] > G_far[2, 2]

    def test_cross_term_activates(self):
        # Both credential and scope elevated
        x = np.array([0.1, 0.6, 0.6, 0.1, 0.1, 0.1])
        G = self.metric.tensor_at(x)
        # Cross-term should add off-diagonal entries
        # credential=index 2, scope=index 1
        assert G[1, 2] > 0 or G[2, 1] > 0

    def test_cross_term_inactive_when_below_threshold(self):
        x = np.zeros(6)  # All below activation_threshold
        G = self.metric.tensor_at(x)
        # Off-diagonal should be zero (only baseline diagonal)
        for i in range(6):
            for j in range(6):
                if i != j:
                    assert G[i, j] == pytest.approx(0.0, abs=0.01)

    def test_metric_weighted_distance(self):
        # Deprecated in v2 (handoff §1: curvature-in-distance lost its
        # pre-registered kill test); behavior kept, so keep asserting it.
        x1 = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
        x2 = np.array([0.3, 0.1, 0.5, 0.1, 0.1, 0.3])  # Credential spike
        with pytest.warns(DeprecationWarning):
            d = self.metric.metric_weighted_distance(x1, x2)
        # Compare to unweighted
        d_raw = np.linalg.norm(x2 - x1)
        # Metric should amplify credential direction
        assert d > d_raw

    def test_path_length(self):
        # Deprecated in v2 (handoff §1: path-energy features carried no
        # signal); behavior kept, so keep asserting it.
        traj = [np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]) for _ in range(5)]
        with pytest.warns(DeprecationWarning):
            length = self.metric.metric_weighted_path_length(traj)
        assert length == 0.0  # Stationary trajectory

        traj2 = [np.array([0.5 - i*0.1, 0.1 + i*0.1, 0.1, 0.1, 0.1, 0.1]) for i in range(5)]
        with pytest.warns(DeprecationWarning):
            length2 = self.metric.metric_weighted_path_length(traj2)
        assert length2 > 0

    def test_boundary_proximity(self):
        low = np.zeros(6)
        high = np.ones(6) * 0.8
        prox_low = self.metric.boundary_proximity(low)
        prox_high = self.metric.boundary_proximity(high)
        for concept in prox_low:
            assert prox_high[concept] > prox_low[concept]

    def test_summary(self):
        x = np.array([0.3, 0.4, 0.5, 0.2, 0.3, 0.4])
        s = self.metric.summary_at(x)
        assert "metric_trace" in s
        assert "boundary_proximity" in s
        assert "eigenvalues" in s
        assert len(s["eigenvalues"]) == 6

    def test_retail_constitution(self):
        spec = ConstitutionSpec.retail_sentinel()
        metric = ConstitutionalMetric(spec, dim_names=CONCEPTS)
        x = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
        G = metric.tensor_at(x)
        assert G.shape == (6, 6)
        eigenvalues = np.linalg.eigvalsh(G)
        assert all(v > 0 for v in eigenvalues)

    def test_invalid_concept_raises(self):
        spec = ConstitutionSpec(boundaries=[Boundary("nonexistent", 0.5)])
        with pytest.raises(ValueError, match="not in dim_names"):
            ConstitutionalMetric(spec, dim_names=CONCEPTS)
