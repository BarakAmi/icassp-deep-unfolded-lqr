"""The landscape grid-framing policies (`viz.landscape.framing`), relocated
from the notebook adapter layer in Stage S5 (T4.d)."""

import numpy as np
import pytest

from mbl.viz.landscape.framing import compute_bowl_ranges, span_control_range


def test_span_control_range_covers_values_with_padding() -> None:
    values = np.array([1.0, 3.0])  # span = 2.0
    result = span_control_range(values, 5, pad=0.25)

    assert result.shape == (5,)
    assert result[0] == pytest.approx(1.0 - 0.25 * 2.0)
    assert result[-1] == pytest.approx(3.0 + 0.25 * 2.0)


def test_span_control_range_falls_back_to_unit_span_when_values_are_constant() -> None:
    values = np.array([2.0, 2.0, 2.0])
    result = span_control_range(values, 3, pad=0.5)

    assert result[0] == pytest.approx(2.0 - 0.5)
    assert result[-1] == pytest.approx(2.0 + 0.5)


def test_compute_bowl_ranges_centers_every_axis_exactly_on_u_star() -> None:
    """Micro-Prompt 4d: the analytical optimum must sit at the exact
    geometric center of every returned range, not merely inside it."""
    path_u = np.array([[5.0, -3.0], [3.0, -1.0], [1.0, 0.5]])
    u_star = np.array([1.0, 0.5])  # the path's own final (converged) row

    ranges = compute_bowl_ranges(path_u, u_star, 11, radius_scale=1.4)

    assert len(ranges) == 2
    for j, axis_range in enumerate(ranges):
        assert axis_range.shape == (11,)
        midpoint = (axis_range[0] + axis_range[-1]) / 2.0
        assert midpoint == pytest.approx(u_star[j])


def test_compute_bowl_ranges_uses_a_single_shared_radius_across_components() -> None:
    """The mandate's "square domain" requirement: every axis spans the SAME
    radius (the max per-component deviation, scaled), even when components
    have very different natural scales -- otherwise the surface's box_aspect
    would stretch one axis relative to the other."""
    path_u = np.array([[10.0, 0.6]])  # component 0 deviates far more than 1
    u_star = np.array([0.0, 0.5])

    ranges = compute_bowl_ranges(path_u, u_star, 5, radius_scale=1.0)

    radius_0 = (ranges[0][-1] - ranges[0][0]) / 2.0
    radius_1 = (ranges[1][-1] - ranges[1][0]) / 2.0
    assert radius_0 == pytest.approx(radius_1)
    assert radius_0 == pytest.approx(10.0)  # max(|10-0|, |0.6-0.5|) * 1.0


def test_compute_bowl_ranges_scales_radius_by_radius_scale() -> None:
    path_u = np.array([[2.0, 2.0]])
    u_star = np.array([0.0, 0.0])

    ranges = compute_bowl_ranges(path_u, u_star, 5, radius_scale=1.4)

    radius = (ranges[0][-1] - ranges[0][0]) / 2.0
    assert radius == pytest.approx(2.0 * 1.4)


def test_compute_bowl_ranges_falls_back_to_unit_radius_when_path_equals_optimum() -> (
    None
):
    path_u = np.array([[1.0, -2.0], [1.0, -2.0]])
    u_star = np.array([1.0, -2.0])  # zero deviation everywhere

    ranges = compute_bowl_ranges(path_u, u_star, 5, radius_scale=1.4)

    radius = (ranges[0][-1] - ranges[0][0]) / 2.0
    assert radius == pytest.approx(1.4)  # unit deviation fallback * radius_scale
