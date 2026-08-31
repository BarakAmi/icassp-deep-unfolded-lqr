import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbl.core.utils.guards import (
    GUARDS,
    GuardTier,
    content_cached,
    guard,
    guard_tier,
    guards_disabled,
    is_tier_active,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _restore_guard_state():
    """Snapshot/restore the process-global guard config around every test, so
    a test that mutates it cannot leak into the others."""
    enabled, max_tier = GUARDS.enabled, GUARDS.max_tier
    yield
    GUARDS.enabled, GUARDS.max_tier = enabled, max_tier


def test_all_tiers_active_by_default() -> None:
    assert is_tier_active(GuardTier.SHAPE)
    assert is_tier_active(GuardTier.DOMAIN)
    assert is_tier_active(GuardTier.EXPENSIVE)


def test_guard_runs_validator_when_tier_active() -> None:
    @guard(GuardTier.DOMAIN)
    def must_be_positive(x: int) -> None:
        if x <= 0:
            raise ValueError("must be positive")

    must_be_positive(1)  # no raise
    with pytest.raises(ValueError, match="positive"):
        must_be_positive(-1)


def test_guards_disabled_context_skips_validator() -> None:
    @guard(GuardTier.DOMAIN)
    def always_fail(_x: object) -> None:
        raise ValueError("should be skipped")

    with guards_disabled():
        always_fail(object())  # skipped -> no raise
    with pytest.raises(ValueError):
        always_fail(object())  # restored


def test_guard_tier_threshold_disables_higher_tiers_only() -> None:
    @guard(GuardTier.EXPENSIVE)
    def expensive_fail(_x: object) -> None:
        raise ValueError("expensive")

    @guard(GuardTier.SHAPE)
    def shape_fail(_x: object) -> None:
        raise ValueError("shape")

    with guard_tier(GuardTier.SHAPE):
        expensive_fail(object())  # EXPENSIVE > SHAPE -> skipped
        with pytest.raises(ValueError, match="shape"):
            shape_fail(object())  # SHAPE still active


def test_content_cached_computes_once_per_distinct_content() -> None:
    calls = {"n": 0}

    @content_cached()
    def min_eig(matrix: np.ndarray) -> float:
        calls["n"] += 1
        return float(np.linalg.eigvalsh(matrix)[0])

    a = np.eye(3)
    b = np.eye(3)  # identical content, different object
    min_eig(a)
    min_eig(b)
    assert calls["n"] == 1  # cached on content, not identity

    min_eig(2 * np.eye(3))
    assert calls["n"] == 2  # different content recomputes


def test_content_cached_handles_multiple_array_args() -> None:
    calls = {"n": 0}

    @content_cached()
    def combine(A: np.ndarray, B: np.ndarray) -> int:
        calls["n"] += 1
        return A.shape[0] + B.shape[1]

    A, B = np.eye(2), np.ones((2, 3))
    assert combine(A, B) == 5
    assert combine(np.eye(2), np.ones((2, 3))) == 5
    assert calls["n"] == 1


def test_guards_compile_out_under_O_flag() -> None:
    """Under ``python -O`` (__debug__ False) a @guard validator is stripped at
    decoration time, so bad input does not raise -- the zero-overhead bypass."""
    script = (
        "from mbl.core.utils.guards import guard, GuardTier\n"
        "@guard(GuardTier.DOMAIN)\n"
        "def must_be_positive(x):\n"
        "    raise ValueError('should be stripped under -O')\n"
        "must_be_positive(-1)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
