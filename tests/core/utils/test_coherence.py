import numpy as np
import pytest

from mbl.core.utils.coherence import (
    ensure_cost_stack_coherence,
    ensure_system_cost_dims_match,
)


def test_cost_stack_coherence_accepts_q_one_longer_than_r() -> None:
    ensure_cost_stack_coherence(np.zeros((6, 2, 2)), np.zeros((5, 1, 1)))


def test_cost_stack_coherence_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="one more slice"):
        ensure_cost_stack_coherence(np.zeros((4, 2, 2)), np.zeros((5, 1, 1)))


def test_cost_stack_coherence_skips_time_invariant_matrices() -> None:
    ensure_cost_stack_coherence(np.eye(2), np.eye(1))  # 2D broadcast -> always ok


class _Dims:
    state_dim = 2
    control_dim = 1


class _System:
    dimensions = _Dims()


class _Cost:
    state_dim = 2
    control_dim = 1


def test_system_cost_dims_match_accepts_agreeing_dims() -> None:
    ensure_system_cost_dims_match(_System(), _Cost())


def test_system_cost_dims_match_rejects_state_mismatch() -> None:
    class _BadCost:
        state_dim = 3
        control_dim = 1

    with pytest.raises(ValueError, match="state dimension"):
        ensure_system_cost_dims_match(_System(), _BadCost())


def test_system_cost_dims_match_skips_when_dimensions_absent() -> None:
    class _AbstractSystem:
        pass

    class _BadCost:
        state_dim = 99
        control_dim = 99

    # No `dimensions` attribute -> duck-typed skip, must not raise.
    ensure_system_cost_dims_match(_AbstractSystem(), _BadCost())
