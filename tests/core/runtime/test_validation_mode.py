"""Stage S1 acceptance tests for the dual-mode policy (T1.5.b/T1.5.c).

Import-order-sensitive behaviors (env ingestion, ``python -O`` forcing,
guard-tier seeding) are exercised in subprocesses so each observes a fresh
interpreter, exactly as the plan's resolution-precedence chain specifies.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mbl.core.runtime import (
    ValidationMode,
    get_validation_mode,
    is_strict,
    set_validation_mode,
    validation_mode,
)
from mbl.core.runtime import modes as modes_module
from mbl.core.utils.guards import GUARDS, GuardTier

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _restore_mode_and_guards():
    """Snapshot/restore the process-global mode and guard state around every
    test in this module."""
    previous_mode = modes_module._STATE.mode
    previous_enabled, previous_tier = GUARDS.enabled, GUARDS.max_tier
    yield
    modes_module._STATE.mode = previous_mode
    GUARDS.enabled, GUARDS.max_tier = previous_enabled, previous_tier


def _run_fresh_interpreter(
    code: str, *flags: str, **env: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *flags, "-c", code],
        env={**os.environ, **env},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestInProcessPolicy:
    def test_default_mode_is_strict(self):
        assert get_validation_mode() is ValidationMode.STRICT
        assert is_strict()

    def test_set_validation_mode_switches_and_syncs_guard_tier(self):
        set_validation_mode(ValidationMode.FAST)
        assert get_validation_mode() is ValidationMode.FAST
        assert GUARDS.max_tier is GuardTier.SHAPE
        set_validation_mode("strict")
        assert get_validation_mode() is ValidationMode.STRICT
        restored_tier: GuardTier = GUARDS.max_tier
        assert restored_tier is GuardTier.EXPENSIVE

    def test_scoped_override_restores_mode_and_tier(self):
        assert get_validation_mode() is ValidationMode.STRICT
        with validation_mode(ValidationMode.FAST):
            assert get_validation_mode() is ValidationMode.FAST
            assert GUARDS.max_tier is GuardTier.SHAPE
        assert get_validation_mode() is ValidationMode.STRICT
        tier_after_scope: GuardTier = GUARDS.max_tier
        assert tier_after_scope is GuardTier.EXPENSIVE

    def test_scoped_override_restores_on_exception(self):
        with pytest.raises(RuntimeError, match="boom"):
            with validation_mode("fast"):
                raise RuntimeError("boom")
        assert get_validation_mode() is ValidationMode.STRICT

    def test_unknown_mode_rejected_loudly(self):
        with pytest.raises(ValueError):
            set_validation_mode("paranoid")


class TestResolutionPrecedence:
    def test_env_fast_mode_seeds_mode_and_guard_tier(self):
        result = _run_fresh_interpreter(
            "from mbl.core.runtime import get_validation_mode, ValidationMode\n"
            "from mbl.core.utils.guards import GUARDS, GuardTier\n"
            "assert get_validation_mode() is ValidationMode.FAST\n"
            "assert GUARDS.max_tier is GuardTier.SHAPE\n",
            SC_VALIDATION_MODE="fast",
        )
        assert result.returncode == 0, result.stderr

    def test_explicit_guard_level_wins_over_env_mode_seeding(self):
        result = _run_fresh_interpreter(
            "from mbl.core.runtime import get_validation_mode, ValidationMode\n"
            "from mbl.core.utils.guards import GUARDS, GuardTier\n"
            "assert get_validation_mode() is ValidationMode.FAST\n"
            "assert GUARDS.max_tier is GuardTier.EXPENSIVE\n",
            SC_VALIDATION_MODE="fast",
            SC_GUARD_LEVEL="EXPENSIVE",
        )
        assert result.returncode == 0, result.stderr

    def test_invalid_env_mode_fails_loudly_at_import(self):
        result = _run_fresh_interpreter(
            "import mbl.core.runtime\n",
            SC_VALIDATION_MODE="strick",
        )
        assert result.returncode != 0
        assert "SC_VALIDATION_MODE" in result.stderr

    def test_python_dash_o_forces_fast(self):
        result = _run_fresh_interpreter(
            "from mbl.core.runtime import get_validation_mode, ValidationMode\n"
            "assert get_validation_mode() is ValidationMode.FAST\n",
            "-O",
        )
        assert result.returncode == 0, result.stderr


class TestProvenanceExemption:
    def test_mode_never_enters_a_signature(self):
        """T1.5.c: a STRICT run and a FAST run of the same experiment are the
        same experiment -- the context signs identically across modes."""
        from mbl.core.runtime import ComputeContext

        set_validation_mode(ValidationMode.STRICT)
        strict_signature = ComputeContext().get_signature()
        set_validation_mode(ValidationMode.FAST)
        fast_signature = ComputeContext().get_signature()
        assert strict_signature == fast_signature
        assert not any("mode" in key for key in strict_signature)
