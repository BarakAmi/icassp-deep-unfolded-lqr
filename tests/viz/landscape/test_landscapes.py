"""Assets 3 (contour), 4 (surface), and 5 (scatter) static landscape
renderers: each returns the expected projection (2D vs. '3d' Axes), draws
the trajectory overlay, and rejects a wrong-arity SliceSpec."""

import numpy as np
import pytest
import torch
from matplotlib.collections import QuadMesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mbl.viz.landscape.box_region import format_box_region_label
from mbl.viz.landscape.grids import GridProjector
from mbl.viz.landscape.landscapes import (
    ContourLandscapeRenderer,
    LineLandscapeRenderer,
    ScatterLandscapeRenderer,
    SurfaceLandscapeRenderer,
)
from mbl.viz.landscape.overlay import TrajectoryOverlay
from mbl.viz.landscape.types import SliceSpec

HORIZON = 4


def _quadratic_oracle(center: torch.Tensor):
    def oracle(U: torch.Tensor) -> torch.Tensor:
        return ((U - center) ** 2).sum(dim=(1, 2))

    return oracle


def _field_2d(control_dim: int = 2):
    reference = np.zeros((HORIZON, control_dim))
    spec = SliceSpec(
        timestep=0,
        components=(0, 1),
        ranges=(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    return GridProjector(chunk_size=32).project(
        reference, oracle, spec, dtype=torch.float64
    )


def _field_1d(control_dim: int = 1):
    reference = np.zeros((HORIZON, control_dim))
    spec = SliceSpec(timestep=0, components=(0,), ranges=(np.linspace(-1, 1, 9),))
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    return GridProjector(chunk_size=32).project(
        reference, oracle, spec, dtype=torch.float64
    )


def _field_3d():
    control_dim = 3
    reference = np.zeros((HORIZON, control_dim))
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    return GridProjector(chunk_size=32).project(
        reference, oracle, spec, dtype=torch.float64
    )


def _overlay_for(field, *, with_costs=False, control_dim=None):
    control_dim = control_dim or (len(field.spec.components))
    num_iterations = 8
    history = np.linspace(-0.8, 0.0, num_iterations * HORIZON * control_dim).reshape(
        num_iterations, HORIZON, control_dim
    )
    costs = np.linspace(5.0, 0.1, num_iterations) if with_costs else None
    return TrajectoryOverlay(
        history, iteration_costs=costs, baseline_U=np.zeros((HORIZON, control_dim))
    )


def test_contour_renderer_returns_2d_figure_with_overlay() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    fig = ContourLandscapeRenderer().render(field, overlay)

    ax = fig.axes[0]
    assert ax.name != "3d"
    assert len(ax.collections) > 0  # contourf + scatter markers
    assert len(ax.lines) > 0  # candidate path


def test_contour_renderer_rejects_3_component_spec() -> None:
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    with pytest.raises(ValueError, match="2-component"):
        ContourLandscapeRenderer().render(field, overlay)


def test_surface_renderer_returns_3d_figure() -> None:
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    fig = SurfaceLandscapeRenderer().render(field, overlay)

    ax = fig.axes[0]
    assert ax.name == "3d"


def test_surface_renderer_without_iteration_costs_still_renders_bare_surface() -> None:
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=False)
    fig = SurfaceLandscapeRenderer().render(field, overlay)  # must not raise
    assert fig.axes[0].name == "3d"


def test_surface_renderer_rejects_3_component_spec() -> None:
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    with pytest.raises(ValueError, match="2-component"):
        SurfaceLandscapeRenderer().render(field, overlay)


def test_surface_renderer_gives_each_sparse_marker_its_own_legend_entry() -> None:
    """NB03 overhaul Phase 4b: each sparse 3D marker is its own labeled
    artist (`marker_legend_label`), so it reads as its own dedicated,
    top-to-bottom legend entry -- no on-plot text annotation."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    fig = SurfaceLandscapeRenderer().render(field, overlay)

    ax = fig.axes[0]
    indices, _ = overlay.selected_path(field.spec)
    expected = {overlay.marker_legend_label(i) for i in indices}
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert expected <= legend_labels
    assert len(ax.texts) == 0  # no on-plot annotation text


def test_surface_renderer_shades_sparse_markers_distinctly() -> None:
    """NB03 overhaul: each sparse 3D marker gets its own `point_shades`
    color (previously every marker shared the same flat `CANDIDATE_COLOR`),
    and the marker line itself must carry no `marker="x"` -- the plain
    connecting line drew a black X under each colored circle before this
    fix, confirmed in a rendered PNG."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    fig = SurfaceLandscapeRenderer().render(field, overlay)

    ax = fig.axes[0]
    marker_paths = [c for c in ax.collections if c.get_offsets().shape[0] >= 1]
    # One collection per sparse marker (plus the surface's own Poly3DCollection).
    indices, _ = overlay.selected_path(field.spec)
    colors = [tuple(c.get_facecolor()[0]) for c in marker_paths[-len(indices) :]]
    assert len(set(colors)) == len(indices)

    trajectory_lines = [ln for ln in ax.lines if ln.get_label() != "_nolegend_"]
    assert all(ln.get_marker() in ("None", "", None) for ln in trajectory_lines)


def test_surface_renderer_has_no_marker_legend_entries_without_iteration_costs() -> (
    None
):
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=False)
    fig = SurfaceLandscapeRenderer().render(field, overlay)

    assert len(fig.axes[0].texts) == 0


def test_contour_renderer_uses_custom_legend_labels() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    fig = ContourLandscapeRenderer().render(
        field,
        overlay,
        path_label="Gradient descent Path",
        optimum_label="Optimum by Riccati",
    )
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "Gradient descent Path" in labels
    assert "Optimum by Riccati" in labels


def test_surface_renderer_draws_optimum_marker_when_optimum_value_given() -> None:
    """The alignment-fix requirement: a 3D surface renderer can only place
    the optimum marker's z-height from an explicitly supplied
    `overlay.optimum_value` (the point generally doesn't fall on a
    discretized grid node) -- without it, no optimum legend entry appears,
    even though the trajectory itself is still drawn (`with_costs=True`).
    When it IS given, Micro-Prompt 4c enriches the legend entry with the
    exact tracked coordinates/cost rather than the bare label."""
    field = _field_2d()
    overlay_without = _overlay_for(field, with_costs=True)
    fig_without = SurfaceLandscapeRenderer().render(
        field, overlay_without, optimum_label="Optimum by Riccati"
    )
    labels_without = [
        text.get_text() for text in fig_without.axes[0].get_legend().get_texts()
    ]
    assert not any(label.startswith("Optimum by Riccati") for label in labels_without)

    overlay_with = _overlay_for(field, with_costs=True)
    overlay_with.optimum_value = 0.0
    fig_with = SurfaceLandscapeRenderer().render(
        field, overlay_with, optimum_label="Optimum by Riccati"
    )
    labels_with = [
        text.get_text() for text in fig_with.axes[0].get_legend().get_texts()
    ]
    assert any(
        label == "Optimum by Riccati\n(u*=[0.000, 0.000], J*=0.000)"
        for label in labels_with
    )


def test_surface_renderer_draws_cocp_marker_when_set() -> None:
    """NB04 COCP/viz refinement plan Sec 2.2: the violet diamond, drawn
    manually (not via the shared `plot_loss_landscape_3d`'s own overlay --
    see the renderer's docstring), with its own enriched legend entry."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    overlay.cocp_point = np.zeros(2)
    overlay.cocp_value = 0.0
    fig = SurfaceLandscapeRenderer().render(field, overlay)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert any(label == "COCP control\n(u=[0.000, 0.000], J=0.000)" for label in labels)


def test_surface_renderer_omits_cocp_marker_without_cocp_value() -> None:
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    overlay.cocp_point = np.zeros(2)  # no cocp_value
    fig = SurfaceLandscapeRenderer().render(field, overlay)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert not any(label.startswith("COCP control") for label in labels)


def test_surface_renderer_draws_a_legend_for_a_marker_alone_without_a_trajectory() -> (
    None
):
    """The widened legend condition (NB04 COCP/viz refinement plan Sec
    2.2): previously this renderer only redrew a legend `if trajectory is
    not None`, so a marker-only overlay (no persisted iteration history --
    e.g. an analytic/frozen contender) would draw a labeled artist with NO
    legend to ever show its label."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=False)  # no trajectory
    overlay.baseline_U = np.zeros((HORIZON, 2))
    overlay.optimum_value = 0.0
    fig = SurfaceLandscapeRenderer().render(field, overlay)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert any(label.startswith("Optimum") for label in labels)


def test_surface_renderer_draws_no_legend_when_nothing_is_labeled() -> None:
    field = _field_2d()
    # _overlay_for always sets baseline_U, but the optimum marker's OWN gate
    # additionally requires optimum_value (never set here) -- so, with no
    # trajectory (with_costs=False) and no cocp_point either, nothing on
    # this axes ends up labeled.
    overlay = _overlay_for(field, with_costs=False)
    fig = SurfaceLandscapeRenderer().render(field, overlay)
    assert fig.axes[0].get_legend() is None


def test_surface_renderer_uses_custom_trajectory_label() -> None:
    """Micro-Prompt 4c: the legend entry is enriched with the candidate
    path's final tracked coordinates/cost, not left as the bare label."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    fig = SurfaceLandscapeRenderer().render(
        field, overlay, trajectory_label="Gradient descent Path"
    )
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    full_path = overlay.path_at(field.spec)
    expected_point = full_path[-1]
    expected_cost = float(overlay.iteration_costs[-1])
    expected = (
        f"Gradient descent Path\n(u_final=[{expected_point[0]:.3f}, "
        f"{expected_point[1]:.3f}], J_final={expected_cost:.3f})"
    )
    assert expected in labels


def test_contour_renderer_enriches_legend_with_tracking_data() -> None:
    """Micro-Prompt 4c: the 2D contour legend must carry the same
    coordinate/cost enrichment as the 3D surface's, in the exact mandated
    format."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    overlay.optimum_value = 0.25
    fig = ContourLandscapeRenderer().render(
        field,
        overlay,
        path_label="Gradient descent Path",
        optimum_label="Optimum by Riccati",
    )
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert any(
        label.startswith("Gradient descent Path\n(u_final=[") and "J_final=" in label
        for label in labels
    )
    assert any(
        label == "Optimum by Riccati\n(u*=[0.000, 0.000], J*=0.250)" for label in labels
    )


def test_contour_renderer_enriches_cocp_legend_entry() -> None:
    """NB04 COCP/viz refinement plan Sec 2.2: ContourLandscapeRenderer has
    no `cocp_label` parameter (PLR0913=6 is already at the cap for this
    method) -- the enrichment happens internally, using a fixed base
    string, exactly like LineLandscapeRenderer/ScatterLandscapeRenderer."""
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    overlay.cocp_point = np.array([0.1, -0.2])
    overlay.cocp_value = 0.75
    fig = ContourLandscapeRenderer().render(field, overlay)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert any(
        label == "COCP control\n(u=[0.100, -0.200], J=0.750)" for label in labels
    )


def test_scatter_renderer_returns_3d_figure_with_overlay() -> None:
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    fig = ScatterLandscapeRenderer().render(field, overlay)

    ax = fig.axes[0]
    assert ax.name == "3d"
    assert len(ax.collections) > 0


def test_scatter_renderer_disables_computed_zorder() -> None:
    """NB03 overhaul, THE OCCLUSION FIX: mplot3d otherwise depth-sorts each
    ARTIST by its own mean camera-space depth, ignoring the `zorder=` values
    `overlay.draw` already sets on the candidate path/markers/optimum star
    -- confirmed directly in a rendered PNG (the path was fully hidden
    behind the dense grid cloud). `ax.computed_zorder = False` is the
    documented matplotlib mechanism to make explicit `zorder` values
    actually govern draw order in a 3D scene."""
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    fig = ScatterLandscapeRenderer().render(field, overlay)

    assert fig.axes[0].computed_zorder is False


def test_scatter_renderer_colorbar_matches_scatter_alpha() -> None:
    """NB03 overhaul, THE COLORBAR-INTENSITY FIX (third review pass): a
    single colorbar swatch can only show ONE alpha, and the only choice
    that stays truthful regardless of screen density/zoom/figure size is to
    match it to the exact alpha each plotted point actually uses -- an
    "undimmed"/fully-opaque colorbar read as visibly BRIGHTER than the
    still-translucent cloud once the figure was widened (a second-review-
    pass fix that broke again), confirmed directly in a rendered PNG."""
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    alpha = 0.18  # ScatterLandscapeRenderer.render's own default
    fig = ScatterLandscapeRenderer().render(field, overlay, alpha=alpha)

    colorbar_axes = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(colorbar_axes) == 1
    scatter = fig.axes[0].collections[0]
    (cbar_solids,) = (
        c for c in colorbar_axes[0].collections if isinstance(c, QuadMesh)
    )
    assert scatter.get_alpha() == pytest.approx(alpha)
    assert cbar_solids.get_alpha() == pytest.approx(alpha)


def test_scatter_renderer_rejects_2_component_spec() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    with pytest.raises(ValueError, match="3-component"):
        ScatterLandscapeRenderer().render(field, overlay)


# --- LineLandscapeRenderer (Asset 3a, m=1) --------------------------------


def test_line_renderer_returns_figure_with_overlay() -> None:
    field = _field_1d()
    overlay = _overlay_for(field, with_costs=True, control_dim=1)
    fig = LineLandscapeRenderer().render(field, overlay)

    ax = fig.axes[0]
    assert ax.name != "3d"
    assert len(ax.lines) > 0  # cost landscape curve + candidate path
    assert len(ax.collections) > 0  # sparse markers


def test_line_renderer_rejects_2_component_spec() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    with pytest.raises(ValueError, match="1-component"):
        LineLandscapeRenderer().render(field, overlay)


def test_line_renderer_requires_iteration_costs() -> None:
    """Distinct failure mode from the arity guard: a valid 1-component
    field/spec, but no iteration_costs to plot the overlay's y-coordinate
    from (see `TrajectoryOverlay.draw`'s 1-component branch)."""
    field = _field_1d()
    overlay = _overlay_for(field, with_costs=False, control_dim=1)
    with pytest.raises(ValueError, match="iteration_costs"):
        LineLandscapeRenderer().render(field, overlay)


def test_line_renderer_uses_custom_legend_labels() -> None:
    field = _field_1d()
    overlay = _overlay_for(field, with_costs=True, control_dim=1)
    fig = LineLandscapeRenderer().render(
        field,
        overlay,
        path_label="Unfolded inner-loop path",
        optimum_label="Riccati optimum",
    )
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert any(label.startswith("Unfolded inner-loop path") for label in labels)


def test_slicespec_accepts_1_component() -> None:
    spec = SliceSpec(timestep=0, components=(0,), ranges=(np.linspace(-1, 1, 5),))
    assert spec.components == (0,)


def test_slicespec_rejects_0_and_4_components() -> None:
    with pytest.raises(ValueError, match="1 \\(line\\), 2"):
        SliceSpec(timestep=0, components=(), ranges=())
    with pytest.raises(ValueError, match="1 \\(line\\), 2"):
        SliceSpec(
            timestep=0,
            components=(0, 1, 2, 3),
            ranges=(np.linspace(-1, 1, 3),) * 4,
        )


# --- overlay.box_bounds (NB04 plan Sec 3.4) ----------------------------------
#
# Every renderer's default (`overlay.box_bounds = None`, the dataclass
# default) is already exercised by every test above -- this section proves
# ONLY the opt-in behavior, via the same fixtures.


def _labeled(artists) -> list:
    """Artists (of any kind: patches, collections, lines) carrying the
    feasible-box overlay's own legend label -- the robust way to detect "did
    THIS feature draw something", independent of how many OTHER artists (the
    surface itself, the candidate polyline, ...) an axes already legitimately
    contains. A prefix check (not an exact match against one fixed constant):
    `format_box_region_label` renders different text depending on the bounds
    each test below configures (isotropic vs. anisotropic)."""
    return [a for a in artists if str(a.get_label()).startswith("Feasible box")]


def test_line_renderer_box_bounds_none_is_unaffected() -> None:
    field = _field_1d()
    overlay = _overlay_for(field, with_costs=True, control_dim=1)
    fig = LineLandscapeRenderer().render(field, overlay)
    assert _labeled(fig.axes[0].lines) == []


def test_line_renderer_draws_feasible_band_when_box_bounds_set() -> None:
    field = _field_1d()
    overlay = _overlay_for(field, with_costs=True, control_dim=1)
    overlay.box_bounds = (-0.5, 0.5)
    fig = LineLandscapeRenderer().render(field, overlay)
    ax = fig.axes[0]
    assert len(_labeled(ax.lines)) == 1
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert format_box_region_label((-0.5, 0.5)) in labels


def test_contour_renderer_box_bounds_none_is_unaffected() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    fig = ContourLandscapeRenderer().render(field, overlay)
    assert _labeled(fig.axes[0].patches) == []


def test_contour_renderer_draws_feasible_rectangle_when_box_bounds_set() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    overlay.box_bounds = (-0.5, 0.5)  # isotropic: broadcasts to both components
    fig = ContourLandscapeRenderer().render(field, overlay)
    ax = fig.axes[0]
    assert len(_labeled(ax.patches)) == 1
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert format_box_region_label((-0.5, 0.5), (-0.5, 0.5)) in labels


def test_contour_renderer_supports_anisotropic_per_component_bounds() -> None:
    field = _field_2d()
    overlay = _overlay_for(field)
    overlay.box_bounds = [(-0.5, 0.5), (-1.0, 1.0)]
    fig = ContourLandscapeRenderer().render(field, overlay)
    (patch,) = _labeled(fig.axes[0].patches)
    assert patch.get_x() == pytest.approx(-0.5)
    assert patch.get_width() == pytest.approx(1.0)
    assert patch.get_y() == pytest.approx(-1.0)
    assert patch.get_height() == pytest.approx(2.0)


def test_surface_renderer_box_bounds_none_is_unaffected() -> None:
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    fig = SurfaceLandscapeRenderer().render(field, overlay)
    assert _labeled(fig.axes[0].lines) == []


def test_surface_renderer_draws_shadow_when_box_bounds_set() -> None:
    field = _field_2d()
    overlay = _overlay_for(field, with_costs=True)
    overlay.box_bounds = (-0.5, 0.5)
    fig = SurfaceLandscapeRenderer().render(field, overlay)
    shadow = _labeled(fig.axes[0].lines)
    assert len(shadow) == 1
    assert not any(
        isinstance(c, Poly3DCollection) and str(c.get_label()).startswith("Feasible")
        for c in fig.axes[0].collections
    )  # no more filled patch


def test_scatter_renderer_box_bounds_none_is_unaffected() -> None:
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    fig = ScatterLandscapeRenderer().render(field, overlay)
    assert _labeled(fig.axes[0].lines) == []


def test_scatter_renderer_draws_wireframe_when_box_bounds_set() -> None:
    """12 wireframe edges (box_region.draw_box_region_3d_wireframe), only
    the first carrying the shared legend label -- checked as a line-count
    DELTA against the unconstrained render, since the other 11 are
    unlabeled (`_nolegend_`) and so can't be found via `_labeled` alone."""
    field = _field_3d()
    baseline_overlay = _overlay_for(field, control_dim=3)
    baseline_count = len(
        ScatterLandscapeRenderer().render(field, baseline_overlay).axes[0].lines
    )

    boxed_overlay = _overlay_for(field, control_dim=3)
    boxed_overlay.box_bounds = (-0.5, 0.5)
    boxed_fig = ScatterLandscapeRenderer().render(field, boxed_overlay)

    assert len(boxed_fig.axes[0].lines) == baseline_count + 12
    assert len(_labeled(boxed_fig.axes[0].lines)) == 1


def test_scatter_renderer_rejects_wrong_length_per_component_bounds() -> None:
    """`resolve_component_bounds`'s own validation surfaces through the
    renderer, not just in isolation."""
    field = _field_3d()
    overlay = _overlay_for(field, control_dim=3)
    overlay.box_bounds = [(-0.5, 0.5), (-1.0, 1.0)]  # only 2, need 3
    with pytest.raises(ValueError, match="per-component"):
        ScatterLandscapeRenderer().render(field, overlay)
