"""The tensor-broadcasting layout proof for `grids.GridProjector`: only the
spec'd ``(timestep, component)`` entries of the batch backdrop vary; every
other entry equals the reference exactly; chunked evaluation matches
unchunked; 2D/3D grid shapes come out right; and, on a hand-defined
quadratic oracle with a known closed-form minimizer, the recovered argmin
grid node matches it -- the generalization of `visualizations.loss_landscape
.compute_loss_grid`'s scalar-per-point loop to one vectorized (chunked)
oracle call the plan's Part 3 commits to.
"""

import numpy as np
import pytest
import torch

from mbl.viz.landscape.grids import GridProjector
from mbl.viz.landscape.types import SliceSpec

HORIZON, CONTROL_DIM = 6, 2


def _reference_U() -> np.ndarray:
    """A non-trivial (nonzero, distinct-per-entry) backdrop, so "only the
    spec'd entries vary" is a meaningful assertion rather than a vacuous
    all-zeros check."""
    return (
        np.arange(HORIZON * CONTROL_DIM, dtype=float).reshape(HORIZON, CONTROL_DIM)
        * 0.1
    )


def _quadratic_oracle_factory(center: torch.Tensor):
    """A hand-defined CostOracle with a KNOWN closed-form minimizer: the sum
    of squared distances from `center` over whichever (timestep, component)
    entries the grid actually varies -- independent of RolloutModel/
    torch.no_grad plumbing, isolating GridProjector's own broadcasting logic."""

    def oracle(U: torch.Tensor) -> torch.Tensor:
        return ((U - center) ** 2).sum(dim=(1, 2))

    return oracle


def test_2d_grid_shape_and_only_spec_entries_vary() -> None:
    reference = _reference_U()
    spec = SliceSpec(
        timestep=2,
        components=(0, 1),
        ranges=(np.linspace(-1, 1, 5), np.linspace(-2, 2, 7)),
    )
    center = torch.as_tensor(reference, dtype=torch.float64)
    oracle = _quadratic_oracle_factory(center)

    field = GridProjector(chunk_size=8).project(
        reference, oracle, spec, dtype=torch.float64
    )

    assert field.cost_grid.shape == (
        7,
        5,
    )  # meshgrid 'xy': (len(range_b), len(range_a))
    assert field.grids[0].shape == (7, 5)
    assert field.grids[1].shape == (7, 5)
    assert np.array_equal(field.reference_U, reference)


def test_3d_grid_shape_with_three_components() -> None:
    control_dim = 3
    reference = (
        np.arange(HORIZON * control_dim, dtype=float).reshape(HORIZON, control_dim)
        * 0.1
    )
    spec = SliceSpec(
        timestep=1,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 5), np.linspace(-1, 1, 3)),
    )
    center = torch.as_tensor(reference, dtype=torch.float64)
    oracle = _quadratic_oracle_factory(center)

    field = GridProjector(chunk_size=13).project(
        reference, oracle, spec, dtype=torch.float64
    )

    assert field.cost_grid.shape == (
        4,
        5,
        3,
    )  # meshgrid 'ij': matches ranges' own order
    assert all(g.shape == (4, 5, 3) for g in field.grids)


def test_only_the_varied_timestep_component_entries_differ_from_reference() -> None:
    """The core broadcasting invariant (Part 3.1): every (t, j) entry of
    every grid row equals reference_U, except the spec'd (timestep,
    components) -- verified by reconstructing the grid batch the same way
    GridProjector does internally and diffing against a capturing oracle."""
    reference = _reference_U()
    spec = SliceSpec(
        timestep=3,
        components=(0, 1),
        ranges=(np.linspace(-5, 5, 6), np.linspace(-5, 5, 6)),
    )
    captured = {}

    def capturing_oracle(U: torch.Tensor) -> torch.Tensor:
        captured["U"] = U.clone()
        return U.sum(dim=(1, 2))

    GridProjector(chunk_size=100).project(
        reference, capturing_oracle, spec, dtype=torch.float64
    )

    U_batch = captured["U"].numpy()
    reference_broadcast = np.broadcast_to(reference, U_batch.shape)
    # Zero out the two varied entries in both, then everything else must match exactly.
    diff = U_batch - reference_broadcast
    mask = np.ones_like(diff, dtype=bool)
    mask[:, spec.timestep, spec.components[0]] = False
    mask[:, spec.timestep, spec.components[1]] = False
    assert np.allclose(diff[mask], 0.0)
    # And the varied entries actually do vary (not all equal to reference).
    assert not np.allclose(diff[:, spec.timestep, spec.components[0]], 0.0)


def test_chunked_evaluation_matches_unchunked() -> None:
    reference = _reference_U()
    spec = SliceSpec(
        timestep=4,
        components=(0, 1),
        ranges=(np.linspace(-1, 1, 11), np.linspace(-1, 1, 13)),
    )
    center = torch.as_tensor(reference, dtype=torch.float64)
    oracle = _quadratic_oracle_factory(center)

    field_chunked = GridProjector(chunk_size=7).project(
        reference, oracle, spec, dtype=torch.float64
    )
    field_unchunked = GridProjector(chunk_size=10_000).project(
        reference, oracle, spec, dtype=torch.float64
    )

    assert np.allclose(field_chunked.cost_grid, field_unchunked.cost_grid)


def test_recovers_the_analytic_minimizer_of_a_known_quadratic() -> None:
    """The paraboloid-recovery acceptance test (Phase 2B plan Part 8): for
    an oracle with a known closed-form minimizer, GridProjector's grid
    argmin lands on the grid node nearest that minimizer."""
    reference = _reference_U()
    true_center = reference.copy()
    true_center[2, 0] = 0.37
    true_center[2, 1] = -0.82
    center = torch.as_tensor(true_center, dtype=torch.float64)
    oracle = _quadratic_oracle_factory(center)

    spec = SliceSpec(
        timestep=2,
        components=(0, 1),
        ranges=(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101)),
    )
    field = GridProjector(chunk_size=512).project(
        reference, oracle, spec, dtype=torch.float64
    )

    argmin = np.unravel_index(np.argmin(field.cost_grid), field.cost_grid.shape)
    u_a = field.grids[0][argmin]
    u_b = field.grids[1][argmin]

    grid_spacing = 2.0 / 100  # range span / (num points - 1)
    assert abs(u_a - 0.37) <= grid_spacing
    assert abs(u_b - (-0.82)) <= grid_spacing
    assert field.cost_grid[argmin] == pytest.approx(0.0, abs=1e-2)


def test_project_rejects_batched_reference_control() -> None:
    reference_batched = np.zeros((2, HORIZON, CONTROL_DIM))
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    with pytest.raises(ValueError, match="use types.normalize_control"):
        GridProjector().project(reference_batched, lambda U: U.sum(dim=(1, 2)), spec)


def test_project_rejects_out_of_range_timestep() -> None:
    reference = _reference_U()
    spec = SliceSpec(
        timestep=HORIZON, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2
    )

    with pytest.raises(ValueError, match="timestep"):
        GridProjector().project(reference, lambda U: U.sum(dim=(1, 2)), spec)


def test_project_rejects_out_of_range_component() -> None:
    reference = _reference_U()
    spec = SliceSpec(
        timestep=0, components=(0, CONTROL_DIM), ranges=(np.linspace(-1, 1, 3),) * 2
    )

    with pytest.raises(ValueError, match="components"):
        GridProjector().project(reference, lambda U: U.sum(dim=(1, 2)), spec)


def test_grid_projector_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        GridProjector(chunk_size=0)
