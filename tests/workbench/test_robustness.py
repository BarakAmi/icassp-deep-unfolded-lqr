"""Acceptance test for `workbench.robustness`: the notebook-facing
dimension-robustness spot check must pass on a well-formed configuration and
raise `AssertionError` the moment a probed control dimension breaks a
contender's forward shape contract -- run against the real unfolded backend,
never mocked.
"""

import torch

from mbl.applications.recipes import UnfoldedKind
from mbl.workbench import DimensionRobustnessSpec, assert_unfolded_dimension_robustness


def _spec(**overrides: object) -> DimensionRobustnessSpec:
    defaults: dict[str, object] = dict(
        state_dim=3,
        horizon=6,
        num_iterations=3,
        step_size_init=0.02,
        step_size_max=0.3,
        seed=0,
        dtype=torch.float64,
        process_noise_std=0.1,
        control_dims=(1, 2, 3),
        batch_size=4,
    )
    defaults.update(overrides)
    return DimensionRobustnessSpec(**defaults)


def test_passes_for_every_unfolded_kind_across_control_dims() -> None:
    message = assert_unfolded_dimension_robustness(_spec())
    assert "passed" in message
    assert "[1, 2, 3]" in message


def test_restricting_to_a_single_kind_still_passes() -> None:
    message = assert_unfolded_dimension_robustness(
        _spec(kinds=(UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,))
    )
    assert "passed" in message


def test_dimension_adaptivity_holds_even_when_m_exceeds_n() -> None:
    # Rectangular (m > n) is the case a hidden square-only assumption would
    # break first; dimension-adaptivity must not depend on n >= m.
    spec = _spec(state_dim=2, control_dims=(3, 4))
    message = assert_unfolded_dimension_robustness(spec)
    assert "passed" in message
