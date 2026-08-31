"""Stage S1 acceptance tests for the validator surfaces (T1.5.a/T1.5.b),
including the Import-Time Identity Law (plan §7.4c)."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float, TypeCheckError
from pydantic import ValidationError

from mbl.core.runtime import (
    ValidationMode,
    mode_gated,
    set_validation_mode,
    shape_checked,
    validate_boundary,
)
from mbl.core.runtime import modes as modes_module
from mbl.core.utils.guards import GUARDS

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _restore_mode_and_guards():
    previous_mode = modes_module._STATE.mode
    previous_enabled, previous_tier = GUARDS.enabled, GUARDS.max_tier
    yield
    modes_module._STATE.mode = previous_mode
    GUARDS.enabled, GUARDS.max_tier = previous_enabled, previous_tier


def _rollout_like(x: Float[np.ndarray, "batch n"]) -> Float[np.ndarray, "batch n"]:
    return x


class TestImportTimeIdentityLaw:
    def test_fast_mode_yields_the_original_function_object(self):
        """The law itself: not a pass-through wrapper -- the same object."""
        set_validation_mode(ValidationMode.FAST)
        decorated = shape_checked(_rollout_like)
        assert decorated is _rollout_like

    def test_mode_gated_lifts_any_decorator_under_the_law(self):
        def noisy_validator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        set_validation_mode(ValidationMode.FAST)
        assert mode_gated(noisy_validator)(_rollout_like) is _rollout_like
        set_validation_mode(ValidationMode.STRICT)
        assert mode_gated(noisy_validator)(_rollout_like) is not _rollout_like

    def test_identity_holds_on_fresh_import_under_env_fast(self):
        """§7.4c as specified: imported fresh under FAST, the decorated name
        and the raw function are the same object."""
        code = (
            "from mbl.core.runtime import shape_checked\n"
            "def f(x):\n"
            "    return x\n"
            "assert shape_checked(f) is f\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "SC_VALIDATION_MODE": "fast"},
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr

    def test_decision_is_burned_at_decoration_time(self):
        """Accepted consequence, stated honestly: a surface decorated under
        STRICT keeps checking even after a later switch to FAST."""
        set_validation_mode(ValidationMode.STRICT)
        decorated = shape_checked(_rollout_like)
        set_validation_mode(ValidationMode.FAST)
        with pytest.raises(TypeCheckError):
            decorated(np.zeros((2, 3, 4)))


class TestStrictModeTensorContracts:
    def test_conforming_arrays_pass(self):
        set_validation_mode(ValidationMode.STRICT)
        decorated = shape_checked(_rollout_like)
        assert decorated is not _rollout_like
        batch = np.zeros((8, 3))
        np.testing.assert_array_equal(decorated(batch), batch)

    def test_rank_violation_raises(self):
        set_validation_mode(ValidationMode.STRICT)
        decorated = shape_checked(_rollout_like)
        with pytest.raises(TypeCheckError):
            decorated(np.zeros((8, 3, 1)))

    def test_dtype_violation_raises(self):
        set_validation_mode(ValidationMode.STRICT)
        decorated = shape_checked(_rollout_like)
        with pytest.raises(TypeCheckError):
            decorated(np.zeros((8, 3), dtype=np.int64))


@dataclass(frozen=True)
class _BoundaryConfig:
    horizon: int
    learning_rate: float


class TestAlwaysOnBoundary:
    def test_valid_payload_coerces_and_returns_typed_instance(self):
        config = validate_boundary(
            _BoundaryConfig, {"horizon": 20, "learning_rate": 1e-3}
        )
        assert config == _BoundaryConfig(horizon=20, learning_rate=1e-3)

    def test_malformed_payload_rejected_in_strict_mode(self):
        set_validation_mode(ValidationMode.STRICT)
        with pytest.raises(ValidationError):
            validate_boundary(
                _BoundaryConfig, {"horizon": "not-an-int", "learning_rate": 1e-3}
            )

    def test_malformed_payload_rejected_in_fast_mode_too(self):
        """§7.4d: boundary ingress is mode-independent -- silent config
        corruption is never acceptable."""
        set_validation_mode(ValidationMode.FAST)
        with pytest.raises(ValidationError):
            validate_boundary(
                _BoundaryConfig, {"horizon": "not-an-int", "learning_rate": 1e-3}
            )
