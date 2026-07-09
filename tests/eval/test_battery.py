"""Slow battery checks as pytest (v2 handoff Phase 2).

Never makes network calls: skips unless the Apollo data is already present
at the pinned commit (fetch it once with `python -m eval.battery --fetch-only`).
Deselect with `-m "not slow"`.
"""

import pytest

pytest.importorskip("sentence_transformers")
pytestmark = pytest.mark.slow

from eval.battery import load_expected  # noqa: E402
from eval.battery.checks import ALL_CHECKS  # noqa: E402
from eval.battery.data import DataUnavailable, fetch_dataset  # noqa: E402


@pytest.fixture(scope="module")
def expected():
    try:
        fetch_dataset(allow_network=False)
    except DataUnavailable as e:
        pytest.skip(str(e))
    return load_expected()


@pytest.mark.parametrize("name", list(ALL_CHECKS))
def test_battery_check(name, expected):
    result = ALL_CHECKS[name](
        expected["checks"][name], expected["tolerance"], log=lambda *a: None
    )
    assert not result.invariant_failures, result.invariant_failures
    assert not result.tol_failures, result.tol_failures
