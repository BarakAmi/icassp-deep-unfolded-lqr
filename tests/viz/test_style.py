import numpy as np
import matplotlib.pyplot as plt
import pytest
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection

import mbl.applications.styles  # noqa: F401  registers the model-family series styles
from mbl.applications.styles import MODEL_FAMILY_COLORS
from mbl.viz.style.io import save_vector_figure
from mbl.viz.style.semantic import (
    SEMANTIC_COLORS,
    ReferenceLineStyle,
    draw_reference_lines,
    resolve_series_family,
    series_color,
    series_linestyle,
)
from mbl.viz.style.theme import (
    LegendStyle,
    BASELINE_LINESTYLES,
    CATEGORICAL_PALETTE,
    DEFAULT_3D_AZIM,
    DEFAULT_3D_ELEV,
    autoscaled_ylim_from_data,
    place_legend_outside,
    place_measured_legend,
    style_3d_axes,
)


def test_categorical_palette_has_no_duplicate_colors() -> None:
    assert len(CATEGORICAL_PALETTE) == len(set(CATEGORICAL_PALETTE))


def test_semantic_colors_reference_the_palette() -> None:
    """T4.e/N6: the semantic roles are defined by palette reference, so a
    palette change propagates to every renderer through one definition."""
    assert SEMANTIC_COLORS.baseline in CATEGORICAL_PALETTE
    assert SEMANTIC_COLORS.optimum == SEMANTIC_COLORS.baseline
    assert SEMANTIC_COLORS.candidate in CATEGORICAL_PALETTE
    assert SEMANTIC_COLORS.aggregate in CATEGORICAL_PALETTE


def test_model_family_colors_only_use_palette_entries() -> None:
    assert set(MODEL_FAMILY_COLORS.values()) <= set(CATEGORICAL_PALETTE)


def test_series_color_is_fixed_for_registered_names() -> None:
    assert series_color("cocp") == MODEL_FAMILY_COLORS["cocp"]
    assert (
        series_color("unfolded_learned_step_size")
        == MODEL_FAMILY_COLORS["unfolded_learned_step_size"]
    )


def test_series_color_falls_back_for_unknown_names_instead_of_raising() -> None:
    color = series_color("some_future_model")
    assert color in CATEGORICAL_PALETTE


def test_series_color_fallback_cycles_by_index() -> None:
    first = series_color("brand_new_model_a", fallback_index=0)
    second = series_color("brand_new_model_b", fallback_index=1)
    assert first != second


def test_resolve_series_family_matches_a_known_name_inside_a_composite_label() -> None:
    assert (
        resolve_series_family("run_20260101_000000_analytic :: trajectory_states")
        == "analytic"
    )


def test_resolve_series_family_returns_none_for_unrecognized_labels() -> None:
    assert resolve_series_family("states") is None
    assert resolve_series_family("Learned Controller") is None


def test_series_linestyle_is_solid_for_closed_form_families() -> None:
    assert series_linestyle("analytic") == "-"
    assert series_linestyle("truncated_riccati") == "-"
    assert series_linestyle("cocp") == "-"
    assert series_linestyle("cocp_lower_bound") == "-"


def test_series_linestyle_is_dashed_for_iterative_families() -> None:
    assert series_linestyle("neural") == "--"
    assert series_linestyle("unfolded_learned_step_size") == "--"
    assert series_linestyle("run_..._unfolded_fixed :: trajectory_controls") == "--"


def test_series_linestyle_defaults_to_solid_for_unrecognized_labels() -> None:
    assert series_linestyle("some_future_model") == "-"


def test_draw_reference_lines_folds_the_value_into_the_label() -> None:
    fig, ax = plt.subplots()
    try:
        draw_reference_lines(ax, {"Theoretical Optimal": 1.2345})
        line = ax.get_lines()[0]
        assert line.get_label() == "Theoretical Optimal: 1.234"
    finally:
        plt.close(fig)


def _rendered_dash_patterns(ax: plt.Axes) -> list[tuple]:
    """The actual on/off dash pattern each line renders with, keyed the same
    way regardless of whether its `linestyle` was a named string (``"--"``)
    or a custom ``(offset, (on, off, ...))`` tuple -- `Line2D.get_linestyle()`
    collapses every custom tuple back to the generic string ``"--"``, so it
    cannot distinguish two different custom dash patterns from each other or
    from the plain ``"--"`` style; `_unscaled_dash_pattern` is matplotlib's
    own internal (unstable-API, but faithful) round trip of exactly what was
    passed to `linestyle=`."""
    return [line._unscaled_dash_pattern for line in ax.get_lines()]  # noqa: SLF001


def test_draw_reference_lines_cycles_distinct_linestyles() -> None:
    fig, ax = plt.subplots()
    try:
        labels = [chr(ord("a") + i) for i in range(len(BASELINE_LINESTYLES))]
        draw_reference_lines(ax, dict.fromkeys(labels, 1.0))
        patterns = _rendered_dash_patterns(ax)
        assert len(set(patterns)) == len(BASELINE_LINESTYLES), (
            f"expected {len(BASELINE_LINESTYLES)} mutually distinct dash "
            f"patterns, got {patterns}"
        )
    finally:
        plt.close(fig)


def test_draw_reference_lines_wraps_the_cycle_past_the_style_count() -> None:
    fig, ax = plt.subplots()
    try:
        labels = [chr(ord("a") + i) for i in range(len(BASELINE_LINESTYLES) + 1)]
        draw_reference_lines(ax, dict.fromkeys(labels, 1.0))
        patterns = _rendered_dash_patterns(ax)
        assert patterns[-1] == patterns[0]
    finally:
        plt.close(fig)


def test_draw_reference_lines_explicit_style_overrides_color_and_linestyle() -> None:
    fig, ax = plt.subplots()
    try:
        draw_reference_lines(
            ax,
            {"J_SDP": 2.8},
            styles={"J_SDP": ReferenceLineStyle(color="#123456", linestyle=":")},
        )
        (line,) = ax.get_lines()
        assert line.get_color() == "#123456"
        assert line.get_linestyle() == ":"
    finally:
        plt.close(fig)


def test_draw_reference_lines_none_styles_matches_default_behavior() -> None:
    """`styles=None` (the default) must reproduce today's index-cycled
    color/linestyle assignment exactly -- proving the new parameter is
    purely additive (NB04 reference-bounds plan Sec 2.2)."""
    reference_lines = {"a": 1.0, "b": 2.0, "c": 3.0}
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    try:
        draw_reference_lines(ax1, reference_lines)
        draw_reference_lines(ax2, reference_lines, styles=None)
        colors1 = [line.get_color() for line in ax1.get_lines()]
        colors2 = [line.get_color() for line in ax2.get_lines()]
        styles1 = _rendered_dash_patterns(ax1)
        styles2 = _rendered_dash_patterns(ax2)
        assert colors1 == colors2
        assert styles1 == styles2
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_draw_reference_lines_removing_one_entry_does_not_restyle_the_others() -> None:
    """The direct regression guard for the index-coupling defect (NB04
    reference-bounds plan Sec 2.2): fixing EVERY remaining line's style via
    `styles` must make it invariant to another entry being removed from
    `reference_lines`, even though removing an entry shifts every
    subsequent line's enumeration index."""
    fixed_styles = {
        "J_LQR": ReferenceLineStyle(color="#111111", linestyle="-."),
        "J_SDP": ReferenceLineStyle(color="#222222", linestyle=":"),
        "COCP": ReferenceLineStyle(color="#333333", linestyle="--"),
    }
    full_lines = {"J_LQR": 1.0, "J_SDP": 2.0, "COCP": 3.0}
    reduced_lines = {"J_SDP": 2.0, "COCP": 3.0}  # J_LQR toggled off

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    try:
        draw_reference_lines(ax1, full_lines, styles=fixed_styles)
        draw_reference_lines(ax2, reduced_lines, styles=fixed_styles)

        by_label1 = {line.get_label().split(":")[0]: line for line in ax1.get_lines()}
        by_label2 = {line.get_label().split(":")[0]: line for line in ax2.get_lines()}
        for label in ("J_SDP", "COCP"):
            assert by_label1[label].get_color() == by_label2[label].get_color()
            assert by_label1[label].get_linestyle() == by_label2[label].get_linestyle()
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_autoscaled_ylim_from_data_pads_the_raw_data_span() -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1, 2], [1.0, 3.0, 2.0])
        low, high = autoscaled_ylim_from_data(ax, margin=0.1)
        assert low == pytest.approx(1.0 - 0.1 * 2.0)
        assert high == pytest.approx(3.0 + 0.1 * 2.0)
    finally:
        plt.close(fig)


def test_autoscaled_ylim_from_data_ignores_lines_added_after_the_call() -> None:
    """The whole point (NB04 reference-bounds plan Sec 2.4): a caller
    snapshots the curves' own range BEFORE reference lines are drawn, so a
    reference line far outside the curve data must not widen it."""
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [1.0, 2.0])
        curve_ylim = autoscaled_ylim_from_data(ax, margin=0.0)
        ax.axhline(100.0)  # a far-outside reference line, drawn AFTER
        assert curve_ylim == pytest.approx((1.0, 2.0))
    finally:
        plt.close(fig)


def test_place_legend_outside_reserves_right_margin_for_the_legend() -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="a")
        place_legend_outside(fig, ax)
        assert ax.get_legend() is not None
        assert fig.subplotpars.right < 0.85
    finally:
        plt.close(fig)


def test_place_legend_outside_respects_a_custom_rect() -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="a")
        place_legend_outside(fig, ax, rect=(0.0, 0.0, 0.5, 1.0))
        assert fig.subplotpars.right < 0.6
    finally:
        plt.close(fig)


def test_place_legend_outside_inside_toggle_draws_legend_without_reserving_margin() -> (
    None
):
    """The accessible inside/outside toggle (NB03 overhaul Phase 4): `inside`
    draws the legend within the axes -- no right-margin reservation."""
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="a")
        place_legend_outside(fig, ax, inside=True)
        assert ax.get_legend() is not None
        assert fig.subplotpars.right > 0.85  # no external margin reserved
    finally:
        plt.close(fig)


def test_place_legend_outside_ncol_one_reads_top_to_bottom() -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="a")
        ax.plot([0, 1], [1, 0], label="b")
        place_legend_outside(fig, ax, style=LegendStyle(ncol=1))
        assert ax.get_legend()._ncols == 1
    finally:
        plt.close(fig)


# --- place_measured_legend (NB04 COCP/viz refinement plan Sec 2.3) ----------
#
# Every assertion below is against MEASURED rendered extents (get_window_
# extent after a forced draw), never anchor/margin literals -- that is the
# whole point of this helper, and what a class of bug like the fixed
# `ax.text(0.5, ...)` placement (axes-fraction 0.5 is mid-plot, not past the
# axes' own right edge) would otherwise slip past.


def _measure(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reference_px = max(a.get_window_extent(renderer).x1 for a in fig.axes)
    fig_width_px = fig.get_window_extent(renderer).width
    return renderer, reference_px, fig_width_px


def test_place_measured_legend_keeps_legend_right_of_axes_and_on_canvas() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.plot([0, 1], [0, 1], label="a")
        legend = place_measured_legend(fig, ax)
        renderer, reference_px, fig_width_px = _measure(fig, ax)
        lb = legend.get_window_extent(renderer)
        assert lb.x0 >= reference_px - 1.0
        assert lb.x1 <= fig_width_px
    finally:
        plt.close(fig)


def test_place_measured_legend_returns_the_legend() -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="a")
        legend = place_measured_legend(fig, ax)
        assert legend is ax.get_legend()
    finally:
        plt.close(fig)


def test_place_measured_legend_respects_legend_kwargs() -> None:
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="a")
        place_measured_legend(fig, ax, legend_kwargs={"fontsize": 21})
        assert ax.get_legend().get_texts()[0].get_fontsize() == 21
    finally:
        plt.close(fig)


def test_place_measured_legend_five_entry_legend_still_fits() -> None:
    """A1/A2 together add a 5th legend entry (COCP) on top of the previous
    4 (path, per-iteration markers, optimum, box) -- the plan's own
    called-out risk case for an unmeasured layout."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        for i in range(5):
            ax.plot([0, 1], [i, i + 1], label=f"Entry {i}\n(u=[0.123, 0.456], J=1.234)")
        legend = place_measured_legend(fig, ax)
        renderer, reference_px, fig_width_px = _measure(fig, ax)
        lb = legend.get_window_extent(renderer)
        assert lb.x0 >= reference_px - 1.0
        assert lb.x1 <= fig_width_px
    finally:
        plt.close(fig)


def test_place_measured_legend_positions_status_text_below_the_legend_outside_right() -> (
    None
):
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.plot([0, 1], [0, 1], label="a")
        ax.plot([0, 1], [1, 0], label="b")
        text = ax.text(
            0,
            0,
            "",
            transform=ax.transAxes,
            clip_on=False,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        )
        text.set_text("Iteration 3 | J=1.234 | u=[0.111, -0.222]")
        legend = place_measured_legend(fig, ax, status_text=text)
        renderer, reference_px, fig_width_px = _measure(fig, ax)
        lb = legend.get_window_extent(renderer)
        tb = text.get_window_extent(renderer)
        assert tb.x0 >= reference_px - 1.0  # right of the axes, like the legend
        assert tb.x1 <= fig_width_px  # never runs off the canvas
        assert tb.y1 <= lb.y0 + 1.0  # entirely BELOW the legend, no overlap
        # Structurally cannot regress to the `ax.text(0.5, ...)` mid-plot
        # bug: the text's position is derived from the legend's measured
        # location, never a hand-picked literal.
        assert text.get_transform() is ax.transAxes  # stays an axes child (blit-safe)
    finally:
        plt.close(fig)


def test_place_measured_legend_clears_a_colorbar_even_with_a_bad_starting_anchor() -> (
    None
):
    """A figure-fraction starting anchor tuned for one figure size/colorbar
    combination can land the legend ON TOP of a colorbar at another --
    the measured reservation must correct for it, not just trust the
    caller's requested anchor."""
    fig = plt.figure(figsize=(15.0, 7.0))
    ax = fig.add_subplot(projection="3d")
    try:
        scatter = ax.scatter([0, 1, 2], [0, 1, 2], [0, 1, 2], c=[0, 1, 2], label="path")
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.05, label="Cost")
        # Deliberately too-far-left for this figure/colorbar combination.
        legend = place_measured_legend(
            fig,
            ax,
            legend_kwargs={
                "bbox_to_anchor": (0.55, 0.5),
                "bbox_transform": fig.transFigure,
            },
        )
        renderer, reference_px, fig_width_px = _measure(fig, ax)
        lb = legend.get_window_extent(renderer)
        assert lb.x0 >= reference_px - 1.0  # cleared the colorbar, not just the axes
        assert lb.x1 <= fig_width_px
    finally:
        plt.close(fig)


def test_save_vector_figure_writes_a_pdf_and_creates_parent_directories(
    tmp_path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    destination = tmp_path / "nested" / "figures" / "cost_curve.pdf"

    written = save_vector_figure(fig, destination)

    assert written == destination.resolve()
    assert written.is_file()
    assert written.read_bytes().startswith(b"%PDF-")


def test_save_vector_figure_writes_svg(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    written = save_vector_figure(fig, tmp_path / "cost_curve.svg")

    assert written.is_file()
    assert b"<svg" in written.read_bytes()


def test_save_vector_figure_rejects_raster_suffixes(tmp_path) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="vector"):
        save_vector_figure(fig, tmp_path / "cost_curve.png")


def test_style_3d_axes_applies_labelpad_to_all_three_axes() -> None:
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        style_3d_axes(ax, labelpad=22.0)
        assert ax.xaxis.labelpad == 22.0
        assert ax.yaxis.labelpad == 22.0
        assert ax.zaxis.labelpad == 22.0
    finally:
        plt.close(fig)


def test_style_3d_axes_applies_box_aspect() -> None:
    """`Axes3D.get_box_aspect` normalizes the vector's overall scale
    (Matplotlib's own internal zoom convention), so only the RATIO between
    entries is meaningfully checkable, not their raw magnitude."""
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        style_3d_axes(ax, box_aspect=(1.0, 1.0, 0.5))
        x, y, z = np.asarray(ax.get_box_aspect())
        assert x == pytest.approx(y)
        assert z / x == pytest.approx(0.5)
    finally:
        plt.close(fig)


def test_style_3d_axes_applies_view_angle() -> None:
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        style_3d_axes(ax, elev=25.0, azim=-45.0)
        assert ax.elev == pytest.approx(25.0)
        assert ax.azim == pytest.approx(-45.0)
    finally:
        plt.close(fig)


def test_style_3d_axes_uses_documented_defaults() -> None:
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        style_3d_axes(ax)
        assert ax.xaxis.labelpad == 15.0
        x, y, z = np.asarray(ax.get_box_aspect())
        assert x == pytest.approx(y)
        assert z / x == pytest.approx(1.2)
        assert ax.elev == pytest.approx(DEFAULT_3D_ELEV)
        assert ax.azim == pytest.approx(DEFAULT_3D_AZIM)
    finally:
        plt.close(fig)
