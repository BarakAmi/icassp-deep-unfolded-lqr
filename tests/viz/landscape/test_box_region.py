"""Unit tests for `viz.landscape.box_region`: the feasible-box overlay
helpers used by the four landscape renderers (NB04 plan Sec 3.4), restyled
to a dashed, unfilled outline by the NB04 COCP/viz refinement plan Sec 2.1.
Exercised directly against plain matplotlib Axes, independent of the full
`GridProjector`/`CostField` rendering pipeline (`test_landscapes.py` covers
the wired-in, per-renderer integration).
"""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mbl.viz.landscape.box_region import (
    BOX_REGION_LINESTYLE,
    draw_box_region_1d,
    draw_box_region_2d,
    draw_box_region_3d_shadow,
    draw_box_region_3d_wireframe,
    format_box_region_label,
    resolve_component_bounds,
)


class TestResolveComponentBounds:
    def test_none_resolves_to_none(self) -> None:
        assert resolve_component_bounds(None, 2) is None

    def test_single_pair_broadcasts_to_every_component(self) -> None:
        resolved = resolve_component_bounds((-0.5, 0.5), 3)
        assert resolved == ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))

    def test_single_pair_for_one_component(self) -> None:
        resolved = resolve_component_bounds((-1.0, 1.0), 1)
        assert resolved == ((-1.0, 1.0),)

    def test_per_component_sequence_is_used_as_given(self) -> None:
        resolved = resolve_component_bounds([(-1.0, 1.0), (-2.0, 2.0)], 2)
        assert resolved == ((-1.0, 1.0), (-2.0, 2.0))

    def test_per_component_sequence_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="per-component"):
            resolve_component_bounds([(-1.0, 1.0)], 2)


class TestFormatBoxRegionLabel:
    def test_single_symmetric_pair_uses_the_infinity_norm_form(self) -> None:
        label = format_box_region_label((-0.2, 0.2))
        assert label == r"Feasible box ($|u|_\infty \leq 0.20$)"

    def test_identical_symmetric_pairs_across_components_still_use_it(self) -> None:
        label = format_box_region_label((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))
        assert label == r"Feasible box ($|u|_\infty \leq 0.20$)"

    def test_differing_pairs_fall_back_to_a_per_component_listing(self) -> None:
        label = format_box_region_label((-0.5, 0.5), (-1.0, 1.0))
        assert label == "Feasible box ([-0.50, 0.50], [-1.00, 1.00])"

    def test_identical_but_asymmetric_pair_also_falls_back(self) -> None:
        """A repeated (-0.1, 0.3) pair is the same across components, but
        cannot be described by a single infinity-norm bound |u| <= u_max
        (which is inherently symmetric about zero) -- must still fall back."""
        label = format_box_region_label((-0.1, 0.3), (-0.1, 0.3))
        assert label == "Feasible box ([-0.10, 0.30], [-0.10, 0.30])"


class TestDrawBoxRegion1D:
    def test_draws_two_dashed_lines_no_fill(self) -> None:
        fig = Figure()
        ax = fig.add_subplot()
        draw_box_region_1d(ax, (-0.3, 0.3))
        assert len(ax.patches) == 0  # no filled band
        assert len(ax.lines) == 2
        xdata = sorted(float(np.asarray(line.get_xdata())[0]) for line in ax.lines)
        assert xdata == pytest.approx([-0.3, 0.3])
        assert all(line.get_linestyle() == BOX_REGION_LINESTYLE for line in ax.lines)

    def test_only_one_line_carries_the_legend_label(self) -> None:
        fig = Figure()
        ax = fig.add_subplot()
        draw_box_region_1d(ax, (-0.3, 0.3))
        labels = [line.get_label() for line in ax.lines]
        assert labels.count(format_box_region_label((-0.3, 0.3))) == 1


class TestDrawBoxRegion2D:
    def test_adds_an_unfilled_dashed_rectangle_at_the_correct_extent(self) -> None:
        fig = Figure()
        ax = fig.add_subplot()
        draw_box_region_2d(ax, (-1.0, 1.0), (-2.0, 2.0))
        assert len(ax.patches) == 1
        patch = ax.patches[0]
        assert isinstance(patch, Rectangle)
        assert patch.get_x() == pytest.approx(-1.0)
        assert patch.get_y() == pytest.approx(-2.0)
        assert patch.get_width() == pytest.approx(2.0)
        assert patch.get_height() == pytest.approx(4.0)
        assert patch.get_facecolor() == to_rgba("none")
        assert patch.get_linestyle() == BOX_REGION_LINESTYLE
        assert patch.get_label() == format_box_region_label((-1.0, 1.0), (-2.0, 2.0))


class TestDrawBoxRegion3DShadow:
    def test_draws_a_closed_dashed_outline_no_fill(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(projection="3d")
        draw_box_region_3d_shadow(ax, (-1.0, 1.0), (-1.0, 1.0), z=-5.0)
        assert not [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(ax.lines) == 1
        line = ax.lines[0]
        assert line.get_linestyle() == BOX_REGION_LINESTYLE
        assert line.get_label() == format_box_region_label((-1.0, 1.0), (-1.0, 1.0))
        xs, ys, zs = line.get_data_3d()
        assert (xs[0], ys[0]) == (xs[-1], ys[-1])  # closed loop
        assert np.all(np.asarray(zs) == -5.0)


class TestDrawBoxRegion3DWireframe:
    def test_draws_twelve_edges(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(projection="3d")
        draw_box_region_3d_wireframe(ax, (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        assert len(ax.lines) == 12
        assert all(line.get_linestyle() == BOX_REGION_LINESTYLE for line in ax.lines)

    def test_only_the_first_edge_carries_the_legend_label(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(projection="3d")
        draw_box_region_3d_wireframe(ax, (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        labels = [line.get_label() for line in ax.lines]
        expected = format_box_region_label((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        assert labels.count(expected) == 1

    def test_edges_span_exactly_the_configured_bounds(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(projection="3d")
        draw_box_region_3d_wireframe(ax, (-1.0, 2.0), (-3.0, 3.0), (0.0, 5.0))
        all_x = np.concatenate([line.get_data_3d()[0] for line in ax.lines])
        all_y = np.concatenate([line.get_data_3d()[1] for line in ax.lines])
        all_z = np.concatenate([line.get_data_3d()[2] for line in ax.lines])
        assert (all_x.min(), all_x.max()) == pytest.approx((-1.0, 2.0))
        assert (all_y.min(), all_y.max()) == pytest.approx((-3.0, 3.0))
        assert (all_z.min(), all_z.max()) == pytest.approx((0.0, 5.0))
