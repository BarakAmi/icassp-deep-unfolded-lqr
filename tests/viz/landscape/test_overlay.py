"""Asset 6's sparse subsampling/annotation engine: select_iterations'
log/linear spacing invariants, and TrajectoryOverlay's data-shaping
(path_at/selected_path/cost_at/optimum_point) and draw() smoke tests."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

from mbl.viz.landscape.overlay import TrajectoryOverlay, select_iterations
from mbl.viz.landscape.types import IterationSelection, SliceSpec
from mbl.viz.style import COCP_COLOR, COCP_MARKER_ALPHA


def test_select_iterations_log_includes_first_and_last() -> None:
    selection = select_iterations(50, k=8, mode="log")
    assert selection.indices[0] == 0
    assert selection.indices[-1] == 49


def test_select_iterations_linear_includes_first_and_last() -> None:
    selection = select_iterations(50, k=8, mode="linear")
    assert selection.indices[0] == 0
    assert selection.indices[-1] == 49


def test_select_iterations_is_monotone_and_bounded_by_k() -> None:
    selection = select_iterations(200, k=10, mode="log")
    assert np.all(np.diff(selection.indices) > 0)
    assert len(selection.indices) <= 10


def test_select_iterations_log_is_denser_early_than_late() -> None:
    """Dense early (GD moves fastest), sparse late -- the first half of the
    iteration range should contain at least as many selected indices as the
    second half for a log-spaced selection."""
    selection = select_iterations(1000, k=12, mode="log")
    midpoint = 500
    early = np.sum(selection.indices < midpoint)
    late = np.sum(selection.indices >= midpoint)
    assert early >= late


def test_select_iterations_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        select_iterations(10, k=3, mode="explicit")


def test_select_iterations_handles_k_larger_than_num_iterations() -> None:
    selection = select_iterations(5, k=100, mode="log")
    assert len(selection.indices) <= 5
    assert selection.indices[-1] == 4


def test_point_shades_returns_n_distinct_valid_hex_colors() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)

    shades = overlay.point_shades(5)

    assert len(shades) == 5
    assert len(set(shades)) == 5  # every point gets a visually distinct shade
    assert all(color.startswith("#") for color in shades)


def test_point_shades_orders_light_to_dark() -> None:
    """Earliest iteration (index 0) should be the palest shade, latest the
    fullest -- so a reader can read "how far along" a point is directly off
    its color, darkest = closest to convergence."""
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)

    shades = overlay.point_shades(4)

    def luminance(hex_color: str) -> float:
        rgb = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
        return sum(rgb)

    luminances = [luminance(c) for c in shades]
    assert luminances == sorted(luminances, reverse=True)  # pale -> dark


def test_point_shades_degenerate_n() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)

    assert overlay.point_shades(0) == []
    assert len(overlay.point_shades(1)) == 1


def _make_history(num_iterations: int, horizon: int, control_dim: int) -> np.ndarray:
    """A deterministic, distinct-per-cell (I, T, m) history."""
    return np.arange(num_iterations * horizon * control_dim, dtype=float).reshape(
        num_iterations, horizon, control_dim
    )


def test_overlay_defaults_to_a_sparse_selection() -> None:
    history = _make_history(num_iterations=50, horizon=6, control_dim=2)
    overlay = TrajectoryOverlay(history)
    assert overlay.selection is not None
    assert len(overlay.selection.indices) < 50


def test_overlay_path_at_selects_timestep_and_components() -> None:
    history = _make_history(num_iterations=10, horizon=6, control_dim=3)
    overlay = TrajectoryOverlay(
        history, selection=IterationSelection(np.arange(10), "explicit")
    )
    spec = SliceSpec(timestep=2, components=(0, 2), ranges=(np.linspace(0, 1, 3),) * 2)

    path = overlay.path_at(spec)

    assert path.shape == (10, 2)
    assert np.array_equal(path, history[:, 2, [0, 2]])


def test_overlay_selected_path_matches_selection_indices() -> None:
    history = _make_history(num_iterations=20, horizon=5, control_dim=2)
    selection = select_iterations(20, k=5, mode="log")
    overlay = TrajectoryOverlay(history, selection=selection)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    indices, points = overlay.selected_path(spec)

    assert np.array_equal(indices, selection.indices)
    assert points.shape == (len(selection.indices), 2)
    assert np.array_equal(points, history[selection.indices][:, 1, :])


def test_overlay_cost_at_returns_none_without_iteration_costs() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)
    assert overlay.cost_at(0) is None


def test_overlay_cost_at_reads_supplied_costs() -> None:
    history = _make_history(5, 4, 2)
    costs = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    overlay = TrajectoryOverlay(history, iteration_costs=costs)
    assert overlay.cost_at(2) == 3.0


def test_overlay_rejects_mismatched_iteration_costs_length() -> None:
    history = _make_history(5, 4, 2)
    with pytest.raises(ValueError, match="iteration_costs"):
        TrajectoryOverlay(history, iteration_costs=np.zeros(3))


def test_overlay_optimum_point_none_without_baseline() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)
    assert overlay.optimum_point(spec) is None


def test_overlay_optimum_point_reads_baseline_at_spec() -> None:
    history = _make_history(5, 4, 2)
    baseline = np.arange(4 * 2, dtype=float).reshape(4, 2)
    overlay = TrajectoryOverlay(history, baseline_U=baseline)
    spec = SliceSpec(timestep=2, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    optimum = overlay.optimum_point(spec)

    assert np.array_equal(optimum, baseline[2, [0, 1]])


def test_overlay_cocp_point_at_none_without_cocp_point() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)
    assert overlay.cocp_point_at(spec) is None


def test_overlay_cocp_point_at_projects_onto_spec_components_no_timestep() -> None:
    """Unlike `optimum_point`, `cocp_point` carries no time dimension -- it
    is already the single frozen instant's full control vector (NB04
    COCP/viz refinement plan Sec 2.2) -- so only `spec.components` (in the
    given order) selects from it, never `spec.timestep`."""
    history = _make_history(5, 4, 3)
    cocp_point = np.array([10.0, 20.0, 30.0])
    overlay = TrajectoryOverlay(history, cocp_point=cocp_point)
    spec = SliceSpec(timestep=2, components=(2, 0), ranges=(np.linspace(0, 1, 3),) * 2)

    projected = overlay.cocp_point_at(spec)

    assert np.array_equal(projected, np.array([30.0, 10.0]))


def test_overlay_cocp_legend_label_reports_coordinates_and_cost() -> None:
    history = _make_history(5, 4, 2)
    cocp_point = np.array([0.1, 0.2])
    overlay = TrajectoryOverlay(history, cocp_point=cocp_point, cocp_value=1.5)
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    label = overlay.cocp_legend_label(spec, "COCP control")

    assert label == "COCP control\n(u=[0.100, 0.200], J=1.500)"


def test_overlay_cocp_legend_label_falls_back_without_cocp_value() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history, cocp_point=np.array([0.1, 0.2]))
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    assert overlay.cocp_legend_label(spec, "COCP control") == "COCP control"


def test_overlay_cocp_legend_label_falls_back_without_cocp_point() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history, cocp_value=1.5)
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    assert overlay.cocp_legend_label(spec, "COCP control") == "COCP control"


def test_overlay_draw_2d_draws_cocp_marker_when_set() -> None:
    history = _make_history(20, 6, 2)
    overlay = TrajectoryOverlay(
        history, cocp_point=np.array([0.3, 0.4]), cocp_value=2.0
    )
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    # draw() uses whatever label string it's given directly (the CALLER
    # enriches it via cocp_legend_label, exactly like optimum_label) -- so
    # the bare default "COCP control" is what's actually drawn here.
    overlay.draw(ax, spec)
    _, labels = ax.get_legend_handles_labels()
    assert "COCP control" in labels
    diamond = next(c for c in ax.collections if c.get_label() == "COCP control")
    face = np.asarray(diamond.get_facecolor())
    assert tuple(face[0][:3]) == pytest.approx(to_rgb(COCP_COLOR))
    # Face alpha < 1 (reference-bounds plan Sec 5.1) so the underlying
    # unfolding path reads through the marker; the edge stays fully opaque
    # so the diamond keeps a crisp outline against a viridis/plasma backdrop.
    assert face[0][3] == pytest.approx(COCP_MARKER_ALPHA)
    edge = np.asarray(diamond.get_edgecolor())
    assert edge[0][3] == pytest.approx(1.0)
    plt.close(fig)


def test_overlay_draw_2d_omits_cocp_marker_when_absent() -> None:
    history = _make_history(20, 6, 2)
    overlay = TrajectoryOverlay(history)  # no cocp_point
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)
    _, labels = ax.get_legend_handles_labels()
    assert not any(label.startswith("COCP control") for label in labels)
    plt.close(fig)


def test_overlay_draw_1d_requires_both_cocp_point_and_cocp_value() -> None:
    """The 1-component slice's y-axis IS cost (mirrors the optimum star's
    own dual gate there) -- cocp_point alone cannot place the marker."""
    history = _make_history(10, 4, 1)
    costs = np.linspace(5.0, 0.1, 10)
    overlay = TrajectoryOverlay(
        history, iteration_costs=costs, cocp_point=np.array([0.5])
    )  # no cocp_value
    spec = SliceSpec(timestep=0, components=(0,), ranges=(np.linspace(-1, 1, 5),))

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)
    _, labels = ax.get_legend_handles_labels()
    assert not any(label.startswith("COCP control") for label in labels)
    plt.close(fig)


def test_overlay_draw_optimum_star_is_fully_opaque() -> None:
    """NB04 COCP/viz refinement plan Sec 2.2: the Riccati star drops its
    previous alpha=0.7 so its black edge reads crisply against the new
    second (COCP) marker."""
    history = _make_history(20, 6, 2)
    baseline = np.zeros((6, 2))
    overlay = TrajectoryOverlay(history, baseline_U=baseline, optimum_value=0.0)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)
    star = next(c for c in ax.collections if c.get_label() == "Optimum")
    assert np.asarray(star.get_facecolor())[0][3] == pytest.approx(1.0)  # alpha
    plt.close(fig)


def test_overlay_draw_2d_smoke() -> None:
    history = _make_history(20, 6, 2)
    costs = np.linspace(10, 1, 20)
    baseline = np.zeros((6, 2))
    overlay = TrajectoryOverlay(history, iteration_costs=costs, baseline_U=baseline)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)  # must not raise
    assert len(ax.lines) > 0
    plt.close(fig)


def test_overlay_draw_2d_gives_each_sparse_marker_its_own_legend_entry() -> None:
    """NB03 overhaul Phase 4b: every selected iteration's marker is its OWN
    labeled artist (`marker_legend_label`), so it becomes its own dedicated
    legend entry instead of an on-plot text annotation -- no `ax.texts`
    entries, one legend-eligible label per selected marker."""
    history = _make_history(20, 6, 2)
    costs = np.linspace(10, 1, 20)
    baseline = np.zeros((6, 2))
    overlay = TrajectoryOverlay(history, iteration_costs=costs, baseline_U=baseline)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)
    assert len(ax.texts) == 0  # no on-plot text annotation

    handles, labels = ax.get_legend_handles_labels()
    marker_labels = {overlay.marker_legend_label(i) for i in overlay.selection.indices}
    assert marker_labels <= set(labels)
    assert len(labels) == len(set(labels))  # every marker gets a distinct label
    plt.close(fig)


def test_overlay_draw_2d_shades_sparse_markers_distinctly() -> None:
    """NB03 overhaul: each sparse marker gets its own `point_shades` color
    (previously every marker shared the same flat `CANDIDATE_COLOR`), so a
    reader can match a legend row to its point by color alone."""
    history = _make_history(20, 6, 2)
    costs = np.linspace(10, 1, 20)
    overlay = TrajectoryOverlay(history, iteration_costs=costs)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)
    marker_colors = [
        tuple(c.get_facecolor()[0])
        for c in ax.collections
        if c.get_offsets().shape[0] == 1
    ]
    assert len(marker_colors) == len(overlay.selection.indices)
    assert len(set(marker_colors)) == len(marker_colors)
    plt.close(fig)


def test_overlay_draw_2d_annotate_false_omits_marker_legend_entries() -> None:
    history = _make_history(20, 6, 2)
    costs = np.linspace(10, 1, 20)
    baseline = np.zeros((6, 2))
    overlay = TrajectoryOverlay(history, iteration_costs=costs, baseline_U=baseline)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(-1, 1, 3),) * 2)

    fig, ax = plt.subplots()
    overlay.draw(ax, spec, annotate=False)
    _, labels = ax.get_legend_handles_labels()
    marker_labels = {overlay.marker_legend_label(i) for i in overlay.selection.indices}
    assert not (marker_labels & set(labels))
    plt.close(fig)


def test_overlay_marker_legend_label_reports_iteration_and_3_decimal_cost() -> None:
    history = _make_history(5, 4, 2)
    costs = np.array([5.0, 4.0, 3.123456, 2.0, 1.0])
    overlay = TrajectoryOverlay(history, iteration_costs=costs)

    assert overlay.marker_legend_label(2) == "i=2\nJ=3.123"


def test_overlay_marker_legend_label_omits_cost_without_iteration_costs() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)

    assert overlay.marker_legend_label(2) == "i=2"


def test_overlay_animated_status_text_reports_one_row_with_cost() -> None:
    """The second "Optimal | J* = ... | u* = ..." row was dropped
    (reference-bounds plan Sec 5.3): `optimum_legend_label` already puts the
    identical figures on the legend's own optimum entry, so the row only
    ever duplicated information already on screen. `animated_status_text`
    no longer takes an optimum-point argument at all."""
    history = _make_history(5, 4, 2)
    costs = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    overlay = TrajectoryOverlay(history, iteration_costs=costs, optimum_value=0.125)
    point = np.array([1.234, -0.5])

    text = overlay.animated_status_text(2, point)

    assert text == "Iteration 2 | J = 3.0000 | u = [1.234, -0.500]"


def test_overlay_animated_status_text_degrades_without_costs() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)  # no iteration_costs
    point = np.array([1.0, 2.0])

    text = overlay.animated_status_text(0, point)

    assert text == "Iteration 0 | u = [1.000, 2.000]"


def test_overlay_optimum_legend_label_reports_coordinates_and_cost() -> None:
    history = _make_history(5, 4, 2)
    baseline = np.arange(4 * 2, dtype=float).reshape(4, 2)
    overlay = TrajectoryOverlay(history, baseline_U=baseline, optimum_value=0.125)
    spec = SliceSpec(timestep=2, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    label = overlay.optimum_legend_label(spec, "Optimum by Riccati")

    point = baseline[2, [0, 1]]
    assert label == (
        f"Optimum by Riccati\n(u*=[{point[0]:.3f}, {point[1]:.3f}], J*=0.125)"
    )


def test_overlay_optimum_legend_label_falls_back_without_optimum_value() -> None:
    history = _make_history(5, 4, 2)
    baseline = np.zeros((4, 2))
    overlay = TrajectoryOverlay(history, baseline_U=baseline)  # no optimum_value
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    assert overlay.optimum_legend_label(spec, "Optimum") == "Optimum"


def test_overlay_optimum_legend_label_falls_back_without_baseline() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    assert overlay.optimum_legend_label(spec, "Optimum") == "Optimum"


def test_overlay_path_legend_label_reports_final_coordinates_and_cost() -> None:
    history = _make_history(5, 4, 2)
    costs = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    overlay = TrajectoryOverlay(history, iteration_costs=costs)
    spec = SliceSpec(timestep=1, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    label = overlay.path_legend_label(spec, "Gradient descent Path")

    final_point = history[-1, 1, [0, 1]]
    assert label == (
        "Gradient descent Path\n(u_final="
        f"[{final_point[0]:.3f}, {final_point[1]:.3f}], J_final=1.000)"
    )


def test_overlay_path_legend_label_falls_back_without_iteration_costs() -> None:
    history = _make_history(5, 4, 2)
    overlay = TrajectoryOverlay(history)
    spec = SliceSpec(timestep=0, components=(0, 1), ranges=(np.linspace(0, 1, 3),) * 2)

    assert overlay.path_legend_label(spec, "Gradient descent Path") == (
        "Gradient descent Path"
    )


def test_overlay_draw_3d_smoke() -> None:
    history = _make_history(15, 4, 3)
    overlay = TrajectoryOverlay(history)
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)),
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    overlay.draw(ax, spec)  # must not raise
    plt.close(fig)


def test_overlay_draw_1d_requires_iteration_costs() -> None:
    """A 1-component slice's y-axis IS cost -- there is no second control
    component to plot the overlay against, so draw() must fail fast rather
    than silently omit the candidate path."""
    history = _make_history(10, 4, 1)
    overlay = TrajectoryOverlay(history)  # no iteration_costs
    spec = SliceSpec(timestep=0, components=(0,), ranges=(np.linspace(-1, 1, 5),))

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="iteration_costs"):
        overlay.draw(ax, spec)
    plt.close(fig)


def test_overlay_draw_1d_smoke_plots_path_against_cost() -> None:
    history = _make_history(10, 4, 1)
    costs = np.linspace(5.0, 0.1, 10)
    baseline = np.zeros((4, 1))
    overlay = TrajectoryOverlay(
        history, iteration_costs=costs, baseline_U=baseline, optimum_value=0.0
    )
    spec = SliceSpec(timestep=0, components=(0,), ranges=(np.linspace(-1, 1, 5),))

    fig, ax = plt.subplots()
    overlay.draw(ax, spec)  # must not raise
    assert len(ax.lines) > 0  # the candidate path
    assert len(ax.collections) > 0  # sparse markers + optimum star
    plt.close(fig)
