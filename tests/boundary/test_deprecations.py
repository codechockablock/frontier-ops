"""v2 deprecations (handoff Phase 4): dead paths warn, code stays.

Acceptance: under `python -W error::DeprecationWarning` each deprecated
path raises; the modules stay importable and the deprecated code still
computes when the warning is not an error.
"""

import warnings

import numpy as np
import pytest

from frontier_ops.boundary.constitution import (
    Boundary,
    ConstitutionSpec,
    ConstitutionalMetric,
)
from frontier_ops.boundary import static_metric


CONST = ConstitutionSpec(
    name="t",
    boundaries=[Boundary("credential_adjacent", threshold=0.5)],
)


class TestCurvatureAsMetricDeprecated:
    def test_metric_weighted_distance_warns_but_computes(self):
        cm = ConstitutionalMetric(CONST)
        with pytest.warns(DeprecationWarning, match="kill test"):
            d = cm.metric_weighted_distance(np.zeros(6), np.ones(6) * 0.5)
        assert d > 0

    def test_path_length_warns_once_but_computes(self):
        cm = ConstitutionalMetric(CONST)
        traj = [np.zeros(6), np.ones(6) * 0.3, np.ones(6) * 0.5]
        with pytest.warns(DeprecationWarning, match="path-energy"):
            total = cm.metric_weighted_path_length(traj)
        assert total > 0

    def test_tensor_at_not_deprecated(self):
        # The position-dependent tensor itself stays warning-free: the live
        # pipeline (streaming regime) evaluates it every step.
        cm = ConstitutionalMetric(CONST)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            G = cm.tensor_at(np.ones(6) * 0.4)
        assert G.shape == (6, 6)


class TestAssertedGDeprecated:
    def test_bare_construction_warns(self):
        with pytest.warns(DeprecationWarning, match="asserted"):
            static_metric.CalibratedMetric()

    def test_estimated_path_does_not_warn(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(1, 1, (30, 4))])
        y = np.array([0] * 30 + [1] * 30)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            static_metric.CalibratedMetric.from_labeled(X, y)


class TestQuarantinedDeadCode:
    def test_modes_import_warns(self):
        # v3: orphaned ICA detector, quarantined not deleted (importable,
        # warns). Import fresh so the module-level warning re-fires.
        import importlib
        import sys

        sys.modules.pop("frontier_ops.sensing.modes", None)
        with pytest.warns(DeprecationWarning, match="dead code"):
            importlib.import_module("frontier_ops.sensing.modes")


class TestNameCollisionResolved:
    def test_alias_kept(self):
        assert static_metric.ConstitutionalMetric is static_metric.CalibratedMetric

    def test_boundary_exports(self):
        from frontier_ops.boundary import CalibratedMetric, StaticMetric

        assert StaticMetric is CalibratedMetric

    def test_two_metrics_are_distinct_classes(self):
        assert static_metric.CalibratedMetric is not ConstitutionalMetric
