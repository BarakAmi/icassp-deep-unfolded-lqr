"""Animated counterparts (Assets 1/2/3-5): smoke + structure tests per the
Phase 2B plan's own test-manifest scoping -- num_frames, update()'s
returned artists, the derived blit policy, and that save() actually writes
a non-empty file (a vector suffix must raise instead)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from mbl.viz.landscape.animators import (
    FrameAnimator,
    LandscapeAnimationOptions,
    LandscapeAnimator,
    LineAnimator,
    RelativeErrorAnimator,
    SignalAnimator,
)
from mbl.viz.landscape.box_region import format_box_region_label
from mbl.viz.landscape.types import SliceSpec
from mbl.viz.style import COCP_COLOR, COCP_MARKER_ALPHA
from mbl.viz.style.theme import DEFAULT_3D_AZIM, DEFAULT_3D_ELEV

HORIZON, CONTROL_DIM, NUM_ITERATIONS = 4, 2, 6


def _baseline() -> np.ndarray:
    return np.stack(
        [np.linspace(1.0, 2.0, HORIZON), np.linspace(-1.0, 0.5, HORIZON)], axis=1
    )


def _history() -> np.ndarray:
    baseline = _baseline()
    return np.stack(
        [baseline + (baseline + 2.0) * (0.5**i) for i in range(NUM_ITERATIONS)], axis=0
    )


def _build_started(animator: FrameAnimator) -> FuncAnimation:
    """Build the animation and render one canvas draw, so the returned
    `FuncAnimation` is *started* (matplotlib warns at garbage collection when
    an animation is deleted without ever rendering; a plain `build()` whose
    result is discarded would pollute the suite with exactly that warning)."""
    animation = animator.build()
    animator._fig.canvas.draw()
    return animation


def _quadratic_oracle(center: torch.Tensor):
    def oracle(U: torch.Tensor) -> torch.Tensor:
        return ((U - center) ** 2).sum(dim=(1, 2))

    return oracle


# --- SignalAnimator ----------------------------------------------------


def test_signal_animator_num_frames_equals_iteration_count() -> None:
    animator = SignalAnimator(_baseline(), _history())
    assert animator.num_frames == NUM_ITERATIONS


def test_signal_animator_is_blit_true() -> None:
    animator = SignalAnimator(_baseline(), _history())
    assert animator.blit is True


def test_signal_animator_update_returns_lines_with_correct_data() -> None:
    animator = SignalAnimator(_baseline(), _history())
    _anim = _build_started(animator)
    artists = animator.update(2)
    assert all(isinstance(a, Line2D) for a in artists)
    assert np.allclose(artists[0].get_ydata(), _history()[2, :, 0])


def test_signal_animator_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="candidate_history"):
        SignalAnimator(_baseline(), _history()[:, :, :1])


def test_signal_animator_save_gif_writes_nonempty_file(tmp_path) -> None:
    animator = SignalAnimator(_baseline(), _history())
    path = animator.save(tmp_path / "signals.gif", fps=5)
    assert path.exists()
    assert path.stat().st_size > 0


def test_signal_animator_save_rejects_vector_suffix(tmp_path) -> None:
    animator = SignalAnimator(_baseline(), _history())
    with pytest.raises(ValueError, match="vector"):
        animator.save(tmp_path / "signals.pdf")


def test_save_closes_its_own_figure_no_double_frame(tmp_path) -> None:
    """Regression (the "double frame" bug, NB03 overhaul Phase 4): `save()`
    must close the figure `build()`/`setup()` created, or `%matplotlib
    inline` auto-displays that leftover open figure as a spurious extra
    static frame alongside the saved GIF."""
    open_before = set(plt.get_fignums())

    animator = SignalAnimator(_baseline(), _history())
    animator.save(tmp_path / "signals.gif", fps=5)

    assert set(plt.get_fignums()) == open_before  # no net-new open figure
    assert animator._fig is not None
    assert not plt.fignum_exists(animator._fig.number)


# --- RelativeErrorAnimator ----------------------------------------------


def test_relative_error_animator_num_frames_equals_iteration_count() -> None:
    animator = RelativeErrorAnimator(_baseline(), _history())
    assert animator.num_frames == NUM_ITERATIONS


def test_relative_error_animator_update_returns_lines() -> None:
    animator = RelativeErrorAnimator(_baseline(), _history())
    _anim = _build_started(animator)
    artists = animator.update(0)
    assert all(isinstance(a, Line2D) for a in artists)


def test_relative_error_animator_save_gif_writes_nonempty_file(tmp_path) -> None:
    animator = RelativeErrorAnimator(_baseline(), _history())
    path = animator.save(tmp_path / "error.gif", fps=5)
    assert path.exists()
    assert path.stat().st_size > 0


# --- LandscapeAnimator ---------------------------------------------------


def _spec_2d() -> SliceSpec:
    return SliceSpec(
        timestep=0,
        components=(0, 1),
        ranges=(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)),
    )


def test_landscape_animator_iteration_contour_num_frames_and_blit() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour"),
    )
    assert animator.num_frames == NUM_ITERATIONS
    assert animator.blit is True


def test_landscape_animator_iteration_contour_update_returns_three_artists() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour"),
    )
    _anim = _build_started(animator)
    artists = animator.update(1)
    assert len(artists) == 3


def test_landscape_animator_contour_update_text_reports_dynamic_cost() -> None:
    """NB03 overhaul Phase 4b: the 2D contour animation's per-frame caption
    must report the "Iteration i | J = ... | u = ..." status, sourced from
    `TrajectoryOverlay.animated_status_text`, and must live OUTSIDE the
    axes' own data area (axes-fraction x > 1) while still being an AXES
    child -- required for `Animation.save`'s blit machinery to actually
    render it into the saved frames (a bare figure-level `fig.text` is
    silently never blitted; see the implementation's "THE BLIT-SAVE FIX"
    note). The status text no longer carries a second "Optimal | J* = ..."
    row (reference-bounds plan Sec 5.3): `optimum_legend_label` already
    reports those same figures on the legend's own optimum entry."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="contour", iteration_costs=costs, optimum_value=0.05
        ),
    )
    _anim = _build_started(animator)
    animator.update(2)
    _, _, text = animator._dynamic_artists
    assert text.get_text() == f"Iteration 2 | J = {costs[2]:.4f} | u = [1.750, -0.750]"
    ax = animator._fig.axes[0]
    assert text in ax.texts  # an axes child (blit-compatible)
    assert text.get_transform() == ax.transAxes  # positioned in axes-fraction
    assert text.get_clip_on() is False  # x > 1.0 must not be clipped


def test_landscape_animator_surface_update_text_reports_dynamic_cost() -> None:
    """Same dynamic one-row caption requirement, for the animated 3D
    surface."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="surface", iteration_costs=costs, optimum_value=0.05
        ),
    )
    _anim = _build_started(animator)
    animator.update(1)
    _, _, text = animator._dynamic_artists
    assert text.get_text() == f"Iteration 1 | J = {costs[1]:.4f} | u = [2.500, -0.500]"


def test_landscape_animator_timestep_mode_num_frames_and_blit() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour", frame_axis="timestep"),
    )
    assert animator.num_frames == HORIZON
    assert animator.blit is False


def test_landscape_animator_timestep_mode_update_redraws_backdrop() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour", frame_axis="timestep"),
    )
    _anim = _build_started(animator)
    artists = animator.update(2)
    assert (
        len(artists) == 2
    )  # point marker + text (backdrop is a collection, not returned)


def test_landscape_animator_surface_requires_iteration_costs() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="iteration_costs"):
        LandscapeAnimator(
            _baseline(),
            _history(),
            oracle,
            _spec_2d(),
            options=LandscapeAnimationOptions(kind="surface"),
        )


def test_landscape_animator_surface_with_costs_builds_and_is_not_blit() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="surface", iteration_costs=costs),
    )
    assert animator.blit is False
    _anim = _build_started(animator)
    artists = animator.update(1)
    # path line + moving point marker + caption text
    assert len(artists) == 3


def test_landscape_animator_surface_has_colorbar_and_deep_bowl_styling() -> None:
    """Asset 4b's polish must apply to the animated surface backdrop too,
    not just the static PDF."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="surface", iteration_costs=costs),
    )
    fig, ax = animator.setup()

    colorbar_axes = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(colorbar_axes) == 1
    assert ax.xaxis.labelpad == 15.0
    x, y, z = np.asarray(ax.get_box_aspect())
    assert x == pytest.approx(y)
    assert z / x == pytest.approx(1.2)
    assert ax.elev == pytest.approx(DEFAULT_3D_ELEV)
    assert ax.azim == pytest.approx(DEFAULT_3D_AZIM)


def test_landscape_animator_surface_draws_optimum_marker_when_given() -> None:
    """The alignment-fix requirement extended to the animated counterpart:
    a static optimum marker (not part of `_dynamic_artists`, since it never
    moves) is added to the 3D axes only when `optimum_value` is supplied."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)

    animator_without = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="surface", iteration_costs=costs),
    )
    fig_without, ax_without = animator_without.setup()
    collections_without = len(ax_without.collections)

    animator_with = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="surface", iteration_costs=costs, optimum_value=0.0
        ),
    )
    fig_with, ax_with = animator_with.setup()
    collections_with = len(ax_with.collections)

    # The surface itself is a Poly3DCollection; the star scatter adds one
    # more collection only when optimum_value is given.
    assert collections_with == collections_without + 1


def test_landscape_animator_contour_draws_box_bounds_when_set() -> None:
    """NB04 COCP/viz refinement plan Sec 2.1: box_bounds now reaches the
    animated backdrop too -- previously only the static renderers received
    it (`UnfoldingLandscapeSpec.box_bounds`' own docstring documented this
    gap explicitly)."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour", box_bounds=(-0.5, 0.5)),
    )
    _fig, ax = animator.setup()
    expected_label = format_box_region_label((-0.5, 0.5), (-0.5, 0.5))
    assert any(p.get_label() == expected_label for p in ax.patches)
    assert to_rgb(ax.patches[0].get_facecolor()) == to_rgb("none")


def test_landscape_animator_surface_draws_box_bounds_when_set() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="surface", iteration_costs=costs, box_bounds=(-0.5, 0.5)
        ),
    )
    _fig, ax = animator.setup()
    expected_label = format_box_region_label((-0.5, 0.5), (-0.5, 0.5))
    assert any(line.get_label() == expected_label for line in ax.lines)


def test_landscape_animator_scatter_draws_box_bounds_when_set() -> None:
    control_dim = 3
    baseline = np.zeros((HORIZON, control_dim))
    history = np.linspace(-0.5, 0.0, NUM_ITERATIONS * HORIZON * control_dim).reshape(
        NUM_ITERATIONS, HORIZON, control_dim
    )
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        baseline,
        history,
        oracle,
        spec,
        options=LandscapeAnimationOptions(kind="scatter", box_bounds=(-0.5, 0.5)),
    )
    _fig, ax = animator.setup()
    expected_label = format_box_region_label((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))
    assert sum(1 for line in ax.lines if line.get_label() == expected_label) == 1


def test_landscape_animator_contour_draws_cocp_marker_when_set() -> None:
    """NB04 COCP/viz refinement plan Sec 2.2: the violet-diamond COCP marker,
    fixed for the whole animation (never part of `_dynamic_artists`), with
    its own coordinate/cost-enriched legend entry."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="contour", cocp_point=np.array([0.1, 0.2]), cocp_value=0.5
        ),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert "COCP control\n(u=[0.100, 0.200], J=0.500)" in labels
    diamond = next(
        c
        for c in ax.collections
        if c.get_label() == "COCP control\n(u=[0.100, 0.200], J=0.500)"
    )
    face = np.asarray(diamond.get_facecolor())
    assert tuple(face[0][:3]) == pytest.approx(to_rgb(COCP_COLOR))
    # Translucent face, opaque edge (reference-bounds plan Sec 5.1).
    assert face[0][3] == pytest.approx(COCP_MARKER_ALPHA)
    edge = np.asarray(diamond.get_edgecolor())
    assert edge[0][3] == pytest.approx(1.0)


def test_landscape_animator_surface_draws_cocp_marker_and_a_legend() -> None:
    """Closes the gap the plan calls out explicitly: this 3D branch never
    had a legend at all before this pass, even with markers labeled."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="surface",
            iteration_costs=costs,
            cocp_point=np.array([0.1, 0.2]),
            cocp_value=0.5,
        ),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert "COCP control\n(u=[0.100, 0.200], J=0.500)" in labels


def test_landscape_animator_surface_status_text_never_overflows_after_update() -> None:
    """Regression -- see the contour-kind version of this test (further
    below) for the full empty-placeholder-measurement bug this guards
    against; the 3D branch uses `ax.text2D`, a separate code path with the
    identical bug."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="surface", iteration_costs=costs, optimum_value=0.05
        ),
    )
    fig, ax = animator.setup()
    animator.update(animator.num_frames - 1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = animator._dynamic_artists[-1]
    assert text.get_text() != ""
    fig_width_px = fig.get_window_extent(renderer).width
    text_bbox = text.get_window_extent(renderer)
    assert text_bbox.x1 <= fig_width_px


def test_landscape_animator_scatter_draws_cocp_marker_and_a_legend() -> None:
    control_dim = 3
    baseline = np.zeros((HORIZON, control_dim))
    history = np.linspace(-0.5, 0.0, NUM_ITERATIONS * HORIZON * control_dim).reshape(
        NUM_ITERATIONS, HORIZON, control_dim
    )
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        baseline,
        history,
        oracle,
        spec,
        options=LandscapeAnimationOptions(
            kind="scatter", cocp_point=np.array([0.1, 0.2, 0.3])
        ),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert any(label.startswith("COCP control") for label in labels)


def test_landscape_animator_scatter_builds_without_iteration_costs() -> None:
    control_dim = 3
    baseline = np.zeros((HORIZON, control_dim))
    history = np.linspace(-0.5, 0.0, NUM_ITERATIONS * HORIZON * control_dim).reshape(
        NUM_ITERATIONS, HORIZON, control_dim
    )
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        baseline,
        history,
        oracle,
        spec,
        options=LandscapeAnimationOptions(kind="scatter"),
    )
    _anim = _build_started(animator)
    artists = animator.update(2)
    # path line + moving point marker + caption text
    assert len(artists) == 3


def test_landscape_animator_scatter_disables_computed_zorder() -> None:
    """NB03 overhaul, THE OCCLUSION FIX: mplot3d otherwise depth-sorts each
    ARTIST by its own mean camera-space depth, ignoring the `zorder=` values
    already set on the candidate path/markers/optimum star -- the optimum
    star was confirmed invisible in a rendered GIF frame before this fix.
    `ax.computed_zorder = False` is the documented matplotlib mechanism to
    make explicit `zorder` actually govern draw order in a 3D scene."""
    control_dim = 3
    baseline = np.zeros((HORIZON, control_dim))
    history = np.linspace(-0.5, 0.0, NUM_ITERATIONS * HORIZON * control_dim).reshape(
        NUM_ITERATIONS, HORIZON, control_dim
    )
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        baseline,
        history,
        oracle,
        spec,
        options=LandscapeAnimationOptions(kind="scatter"),
    )
    _fig, ax = animator.setup()
    assert ax.computed_zorder is False


def test_landscape_animator_scatter_colorbar_matches_scatter_alpha() -> None:
    """NB03 overhaul, THE COLORBAR-INTENSITY FIX (third review pass): a
    single colorbar swatch can only show ONE alpha, and the only choice
    that stays truthful regardless of screen density/zoom/figure size is to
    match it to the exact alpha each plotted point actually uses -- an
    "undimmed"/fully-opaque colorbar read as visibly BRIGHTER than the
    still-translucent cube once the figure was widened (a second-review-
    pass fix that broke again), confirmed directly in a rendered GIF."""
    control_dim = 3
    baseline = np.zeros((HORIZON, control_dim))
    history = np.linspace(-0.5, 0.0, NUM_ITERATIONS * HORIZON * control_dim).reshape(
        NUM_ITERATIONS, HORIZON, control_dim
    )
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        baseline,
        history,
        oracle,
        spec,
        options=LandscapeAnimationOptions(kind="scatter"),
    )
    fig, ax = animator.setup()
    colorbar_axes = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(colorbar_axes) == 1
    scatter = ax.collections[0]
    (cbar_solids,) = (
        c for c in colorbar_axes[0].collections if isinstance(c, QuadMesh)
    )
    assert scatter.get_alpha() == pytest.approx(cbar_solids.get_alpha())


def test_landscape_animator_contour_draws_a_compact_external_legend() -> None:
    """NB03 overhaul: the contour animation previously labeled its path/
    optimum artists but never actually called `ax.legend()` -- no legend
    rendered at all. It must now exist, anchored outside the axes (axes-
    fraction x > 1) so it never overlaps the plotted data. NB04 COCP/viz
    refinement plan Sec 2.2, item 1(b): the optimum entry is now coordinate/
    cost-enriched here too, not just in the static renderers -- `_baseline()`
    at `timestep=0` is `[1.0, -1.0]`."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour", optimum_value=0.05),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert {"Candidate path", "Optimum\n(u*=[1.000, -1.000], J*=0.050)"} <= labels


def test_landscape_animator_honors_a_custom_optimum_label() -> None:
    """Reference-bounds plan Sec 5.2: `optimum_label` used to be hardcoded
    to "Optimum" inside the animator itself, independent of whatever label
    the static renderer for the SAME figure was given (e.g. NB04's own
    "Riccati optimum") -- now threaded through `LandscapeAnimationOptions`,
    so the two can no longer disagree."""
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="contour", optimum_value=0.05, optimum_label="Riccati optimum"
        ),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert "Riccati optimum\n(u*=[1.000, -1.000], J*=0.050)" in labels
    assert not any(label.startswith("Optimum") for label in labels)


def test_landscape_animator_status_text_never_overflows_the_canvas_after_update() -> (
    None
):
    """Regression: `place_measured_legend` reserves margin by measuring
    `status_text` at `setup()` time, when it is still an EMPTY placeholder
    -- an empty string measures far narrower than the real, multi-row
    status text `_update_iteration_mode` sets on every actual frame, so the
    reserved margin came out too small and the populated text clipped at
    the saved GIF's own fixed canvas edge (confirmed directly in a rendered
    frame). The fix seeds `status_text` with its REAL frame-0 content
    before the margin is ever measured; this checks the text still fits
    after `update()` sets the (potentially longer, more-decimal-places)
    real content for the LAST frame too."""
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(
            kind="contour", iteration_costs=costs, optimum_value=0.05
        ),
    )
    fig, ax = animator.setup()
    animator.update(animator.num_frames - 1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = animator._dynamic_artists[-1]
    assert text.get_text() != ""  # never left as the empty placeholder
    fig_width_px = fig.get_window_extent(renderer).width
    text_bbox = text.get_window_extent(renderer)
    assert text_bbox.x1 <= fig_width_px


def test_landscape_animator_rejects_unknown_frame_axis() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="frame_axis"):
        LandscapeAnimator(
            _baseline(),
            _history(),
            oracle,
            _spec_2d(),
            options=LandscapeAnimationOptions(frame_axis="bogus"),
        )


def test_landscape_animator_rejects_unknown_kind() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="kind"):
        LandscapeAnimator(
            _baseline(),
            _history(),
            oracle,
            _spec_2d(),
            options=LandscapeAnimationOptions(kind="bogus"),
        )


def test_landscape_animator_rejects_timestep_mode_with_3d_kind() -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="frame_axis='timestep'"):
        LandscapeAnimator(
            _baseline(),
            _history(),
            oracle,
            _spec_2d(),
            options=LandscapeAnimationOptions(
                kind="surface",
                frame_axis="timestep",
                iteration_costs=np.linspace(5.0, 0.1, NUM_ITERATIONS),
            ),
        )


def test_landscape_animator_save_gif_writes_nonempty_file(tmp_path) -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour"),
    )
    path = animator.save(tmp_path / "landscape.gif", fps=5)
    assert path.exists()
    assert path.stat().st_size > 0


def test_landscape_animator_save_rejects_vector_suffix(tmp_path) -> None:
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        _baseline(),
        _history(),
        oracle,
        _spec_2d(),
        options=LandscapeAnimationOptions(kind="contour"),
    )
    with pytest.raises(ValueError, match="vector"):
        animator.save(tmp_path / "landscape.svg")


def test_landscape_animator_scatter_with_optimum_value_does_not_crash() -> None:
    """Regression test: `optimum_value` combined with `kind="scatter"` used
    to crash `_setup_iteration_mode` (it appended `optimum_value` as a
    spurious 4th positional arg to a 3-component `ax.scatter(*optimum,
    ...)` call, misread as the `s` marker-size kwarg). Unlike the
    2-component surface case (where `optimum_value` supplies the only
    z-coordinate), a 3-component `optimum` already carries every coordinate
    the scatter marker needs from `baseline_U` alone, so the marker is drawn
    the same way (and the collection count is unaffected) whether or not
    `optimum_value` is additionally given."""
    control_dim = 3
    baseline = np.full((HORIZON, control_dim), 0.3)
    history = np.linspace(-0.5, 0.0, NUM_ITERATIONS * HORIZON * control_dim).reshape(
        NUM_ITERATIONS, HORIZON, control_dim
    )
    spec = SliceSpec(
        timestep=0,
        components=(0, 1, 2),
        ranges=(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)),
    )
    oracle = _quadratic_oracle(
        torch.zeros(1, HORIZON, control_dim, dtype=torch.float64)
    )
    animator = LandscapeAnimator(
        baseline,
        history,
        oracle,
        spec,
        options=LandscapeAnimationOptions(kind="scatter", optimum_value=0.0),
    )
    _, ax = animator.setup()  # must not raise

    # The main grid scatter plus the optimum star -- both collections; the
    # star's own offsets must be the raw (x, y, z) = optimum coordinates,
    # never contaminated by optimum_value substituting for one of them.
    assert len(ax.collections) == 2
    star_offsets = ax.collections[-1]._offsets3d
    assert np.allclose([float(v[0]) for v in star_offsets], [0.3, 0.3, 0.3])


# --- LineAnimator (Asset 3a, m=1) -----------------------------------------


def _baseline_1d() -> np.ndarray:
    return np.linspace(1.0, 2.0, HORIZON).reshape(HORIZON, 1)


def _history_1d() -> np.ndarray:
    baseline = _baseline_1d()
    return np.stack(
        [baseline + (baseline + 2.0) * (0.5**i) for i in range(NUM_ITERATIONS)], axis=0
    )


def _spec_1d() -> SliceSpec:
    return SliceSpec(timestep=0, components=(0,), ranges=(np.linspace(-1, 1, 9),))


def _line_oracle():
    center = torch.zeros(1, HORIZON, 1, dtype=torch.float64)
    return _quadratic_oracle(center)


def test_line_animator_num_frames_and_blit() -> None:
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(iteration_costs=costs),
    )
    assert animator.num_frames == NUM_ITERATIONS
    assert animator.blit is True


def test_line_animator_update_returns_three_artists_with_correct_data() -> None:
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    history = _history_1d()
    animator = LineAnimator(
        _baseline_1d(),
        history,
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(iteration_costs=costs),
    )
    _anim = _build_started(animator)
    artists = animator.update(2)
    assert len(artists) == 3
    path_line, point_marker, text = artists
    assert np.allclose(path_line.get_xdata(), history[: 2 + 1, 0, 0])
    assert np.allclose(path_line.get_ydata(), costs[: 2 + 1])
    assert text.get_text() == f"Iteration 2 | J = {costs[2]:.4f} | u = [1.750]"


def test_line_animator_draws_a_compact_external_legend() -> None:
    """NB03 overhaul: the m=1 animation previously labeled its "Cost
    landscape"/"Optimum" artists but never actually called `ax.legend()` --
    no legend rendered at all. NB04 COCP/viz refinement plan Sec 2.2, item
    1(b): the optimum entry is now coordinate/cost-enriched here too --
    `_baseline_1d()` at `timestep=0` is ``1.0``."""
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(iteration_costs=costs, optimum_value=0.05),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert {"Cost landscape", "Optimum\n(u*=[1.000], J*=0.050)"} <= labels


def test_line_animator_draws_box_bounds_when_set() -> None:
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(
            iteration_costs=costs, box_bounds=(-0.5, 0.5)
        ),
    )
    _fig, ax = animator.setup()
    expected_label = format_box_region_label((-0.5, 0.5))
    labeled_lines = [line for line in ax.lines if line.get_label() == expected_label]
    assert len(labeled_lines) == 1
    # Both bound lines are drawn, but only the first carries the label
    # (matching `draw_box_region_1d`'s own "one label per pair" contract).
    # path_line/point_marker (the dynamic artists) start out empty, so only
    # non-empty lines' x-data is meaningful here.
    xdata = sorted(
        {
            float(np.asarray(line.get_xdata())[0])
            for line in ax.lines
            if len(line.get_xdata())
        }
        & {-0.5, 0.5}
    )
    assert xdata == [-0.5, 0.5]


def test_line_animator_draws_cocp_marker_when_set() -> None:
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(
            iteration_costs=costs, cocp_point=np.array([1.5]), cocp_value=0.3
        ),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert "COCP control\n(u=[1.500], J=0.300)" in labels


def test_line_animator_honors_a_custom_optimum_label() -> None:
    """`LineAnimator`'s own copy of the same gap (reference-bounds plan Sec
    5.2) -- see `test_landscape_animator_honors_a_custom_optimum_label`."""
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(
            iteration_costs=costs, optimum_value=0.05, optimum_label="Riccati optimum"
        ),
    )
    _fig, ax = animator.setup()
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert any(label.startswith("Riccati optimum") for label in labels)
    assert not any(label.startswith("Optimum") for label in labels)


def test_line_animator_status_text_never_overflows_the_canvas_after_update() -> None:
    """Regression -- see `LandscapeAnimator`'s identical test for the full
    empty-placeholder-measurement bug this guards against."""
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(iteration_costs=costs, optimum_value=0.05),
    )
    fig, ax = animator.setup()
    animator.update(animator.num_frames - 1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = animator._dynamic_artists[-1]
    assert text.get_text() != ""
    fig_width_px = fig.get_window_extent(renderer).width
    text_bbox = text.get_window_extent(renderer)
    assert text_bbox.x1 <= fig_width_px


def test_line_animator_requires_iteration_costs() -> None:
    with pytest.raises(ValueError, match="iteration_costs"):
        LineAnimator(
            _baseline_1d(), _history_1d(), _line_oracle(), _spec_1d()
        )  # default options has iteration_costs=None


def test_line_animator_rejects_non_1_component_spec() -> None:
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    with pytest.raises(ValueError, match="1-component"):
        LineAnimator(
            _baseline(),
            _history(),
            _quadratic_oracle(
                torch.zeros(1, HORIZON, CONTROL_DIM, dtype=torch.float64)
            ),
            _spec_2d(),
            options=LandscapeAnimationOptions(iteration_costs=costs),
        )


def test_line_animator_save_gif_writes_nonempty_file(tmp_path) -> None:
    costs = np.linspace(5.0, 0.1, NUM_ITERATIONS)
    animator = LineAnimator(
        _baseline_1d(),
        _history_1d(),
        _line_oracle(),
        _spec_1d(),
        options=LandscapeAnimationOptions(iteration_costs=costs),
    )
    path = animator.save(tmp_path / "line.gif", fps=5)
    assert path.exists()
    assert path.stat().st_size > 0
