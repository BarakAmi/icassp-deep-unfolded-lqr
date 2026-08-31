"""Depth-parameterized trajectory diagnostics (NB03 Phase 3, requirement 4):
`plot_trajectory_vs_depth_with_error`'s ``m + 1`` grid and
`DepthTrajectoryAnimator`'s frame-per-depth counterpart -- both fully
shape-derived over the control dimension `m` (parametrized below over
m=1,2,3,4 to prove no code path special-cases a particular dimension)."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mbl.viz.plots.depth_trajectories import (
    DepthTrajectoryAnimator,
    DepthTrajectoryStyle,
    plot_trajectory_vs_depth_with_error,
)

HORIZON = 10
DEPTHS = (2, 4, 6)


def _baseline(m: int) -> np.ndarray:
    return np.sin(np.linspace(0, 3, HORIZON * m)).reshape(HORIZON, m)


def _depth_trajectories(m: int) -> dict[int, np.ndarray]:
    base = _baseline(m)
    return {
        depth: base
        + (0.5 / depth) * np.random.default_rng(depth).normal(size=(HORIZON, m))
        for depth in DEPTHS
    }


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_plot_trajectory_vs_depth_with_error_axes_count_is_m_plus_1(m: int) -> None:
    fig = plot_trajectory_vs_depth_with_error(_baseline(m), _depth_trajectories(m))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == m + 1


def test_plot_trajectory_vs_depth_with_error_one_line_per_depth_plus_baseline() -> None:
    m = 2
    fig = plot_trajectory_vs_depth_with_error(_baseline(m), _depth_trajectories(m))
    signal_ax = fig.axes[0]
    assert len(signal_ax.get_lines()) == len(DEPTHS) + 1  # + baseline


def test_plot_trajectory_vs_depth_with_error_last_axis_is_relative_error() -> None:
    m = 2
    fig = plot_trajectory_vs_depth_with_error(_baseline(m), _depth_trajectories(m))
    error_ax = fig.axes[-1]
    assert error_ax.get_yscale() == "log"
    assert len(error_ax.get_lines()) == len(DEPTHS)


def test_plot_trajectory_vs_depth_with_error_rejects_empty_depth_trajectories() -> None:
    with pytest.raises(ValueError, match="depth_trajectories"):
        plot_trajectory_vs_depth_with_error(_baseline(2), {})


def test_plot_trajectory_vs_depth_with_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"\(T, m\)"):
        plot_trajectory_vs_depth_with_error(_baseline(2), {2: _baseline(3)})


def test_plot_trajectory_vs_depth_with_error_rejects_wrong_length_dim_labels() -> None:
    with pytest.raises(ValueError, match="dim_labels"):
        plot_trajectory_vs_depth_with_error(
            _baseline(2),
            _depth_trajectories(2),
            style=DepthTrajectoryStyle(dim_labels=["only_one"]),
        )


def test_plot_trajectory_vs_depth_with_error_uses_custom_dim_labels() -> None:
    m = 2
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(dim_labels=["position", "velocity"]),
    )
    ylabels = [ax.get_ylabel() for ax in fig.axes[:m]]
    assert ylabels == ["position", "velocity"]


def test_plot_trajectory_vs_depth_with_error_accepts_batched_inputs() -> None:
    """`normalize_control`'s (B, T, m) reduction, reused from
    `viz.landscape.types`."""
    m = 2
    batched_baseline = np.stack([_baseline(m)] * 3)
    batched_depths = {k: np.stack([v] * 3) for k, v in _depth_trajectories(m).items()}
    fig = plot_trajectory_vs_depth_with_error(batched_baseline, batched_depths)
    assert len(fig.axes) == m + 1


def test_plot_trajectory_vs_depth_with_error_legend_reports_costs() -> None:
    m = 2
    trajectories = _depth_trajectories(m)
    costs = {depth: 1.0 / depth for depth in trajectories}
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(m), trajectories, costs=costs, baseline_cost=0.5
    )
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels[0] == "Riccati (J=0.5000)"
    for depth, label in zip(sorted(trajectories), labels[1:]):
        assert label == f"K={depth} (J={costs[depth]:.4f})"


def test_plot_trajectory_vs_depth_with_error_default_title_names_contender() -> None:
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(2),
        _depth_trajectories(2),
        style=DepthTrajectoryStyle(contender_name="Unfolded-alpha"),
    )
    assert "Unfolded-alpha" in fig._suptitle.get_text()


def test_plot_trajectory_vs_depth_with_error_default_title_names_the_actual_baseline() -> (
    None
):
    """Reference-bounds plan Sec 3 (found by actually re-running NB04): the
    default title used to hardcode "vs. Riccati" regardless of
    `style.baseline_label`, so swapping the baseline to COCP left the
    legend correctly saying "COCP" while the title still claimed "Riccati" --
    a rendered factual mismatch. The title must now name whatever
    `baseline_label` actually is."""
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(2),
        _depth_trajectories(2),
        style=DepthTrajectoryStyle(
            contender_name="Unfolded-alpha_p", baseline_label="COCP"
        ),
    )
    title = fig._suptitle.get_text()
    assert "vs. COCP" in title
    assert "Riccati" not in title


# --- DepthTrajectoryAnimator ------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3])
def test_depth_trajectory_animator_num_frames_equals_depth_count(m: int) -> None:
    animator = DepthTrajectoryAnimator(_baseline(m), _depth_trajectories(m))
    assert animator.num_frames == len(DEPTHS)


def test_depth_trajectory_animator_is_blit_true() -> None:
    animator = DepthTrajectoryAnimator(_baseline(2), _depth_trajectories(2))
    assert animator.blit is True


def test_depth_trajectory_animator_update_returns_expected_artist_count_and_data() -> (
    None
):
    m = 2
    trajectories = _depth_trajectories(m)
    animator = DepthTrajectoryAnimator(_baseline(m), trajectories)
    animator.build()
    animator._fig.canvas.draw()

    artists = animator.update(1)
    # m signal lines + 1 error line + 1 title text + 1 legend text
    assert len(artists) == m + 3
    depth = sorted(trajectories)[1]
    signal_line = artists[0]
    assert isinstance(signal_line, Line2D)
    assert np.allclose(signal_line.get_ydata(), trajectories[depth][:, 0])


def test_depth_trajectory_animator_title_names_the_current_depth() -> None:
    m = 2
    trajectories = _depth_trajectories(m)
    animator = DepthTrajectoryAnimator(_baseline(m), trajectories)
    animator.build()
    animator._fig.canvas.draw()
    animator.update(2)
    depth = sorted(trajectories)[2]
    assert animator._title is not None
    assert f"K = {depth}" in animator._title.get_text()


def test_depth_trajectory_animator_title_names_the_actual_baseline() -> None:
    """The animator's own copy of the same title/legend mismatch (reference-
    bounds plan Sec 3) -- see the static renderer's identical test."""
    m = 2
    trajectories = _depth_trajectories(m)
    animator = DepthTrajectoryAnimator(
        _baseline(m), trajectories, style=DepthTrajectoryStyle(baseline_label="COCP")
    )
    animator.build()
    animator._fig.canvas.draw()
    animator.update(0)
    assert animator._title is not None
    title = animator._title.get_text()
    assert "vs. COCP" in title
    assert "Riccati" not in title


def test_depth_trajectory_animator_save_gif_writes_nonempty_file(tmp_path) -> None:
    animator = DepthTrajectoryAnimator(_baseline(2), _depth_trajectories(2))
    path = animator.save(tmp_path / "depth.gif", fps=2)
    assert path.exists()
    assert path.stat().st_size > 0


# --- box_bounds (NB04 plan Sec 3.5) -----------------------------------------


def test_plot_trajectory_vs_depth_with_error_box_bounds_none_is_unaffected() -> None:
    """The default (unconstrained, NB03) call site must be byte-for-byte
    unaffected: no extra lines, no explicit ylim call this feature didn't
    already make."""
    m = 2
    fig_without = plot_trajectory_vs_depth_with_error(
        _baseline(m), _depth_trajectories(m)
    )
    fig_with_none = plot_trajectory_vs_depth_with_error(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(box_bounds=None),
    )
    for ax_a, ax_b in zip(fig_without.axes, fig_with_none.axes):
        assert len(ax_a.get_lines()) == len(ax_b.get_lines())
        assert ax_a.get_ylim() == ax_b.get_ylim()


def test_plot_trajectory_vs_depth_with_error_draws_two_bound_lines_per_signal_row() -> (
    None
):
    m = 2
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(box_bounds=(-1.0, 1.0)),
    )
    for j in range(m):
        signal_ax = fig.axes[j]
        # baseline + len(DEPTHS) contenders + 2 bound lines
        assert len(signal_ax.get_lines()) == len(DEPTHS) + 1 + 2
    # The error row (last axis) must NEVER get the bound lines.
    error_ax = fig.axes[-1]
    assert len(error_ax.get_lines()) == len(DEPTHS)


def test_plot_trajectory_vs_depth_with_error_bound_lines_sit_at_the_configured_values() -> (
    None
):
    m = 1
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(box_bounds=(-0.7, 0.7)),
    )
    signal_ax = fig.axes[0]
    bound_lines = signal_ax.get_lines()[-2:]
    y_values = sorted(float(np.asarray(line.get_ydata())[0]) for line in bound_lines)
    assert y_values == pytest.approx([-0.7, 0.7])


def test_plot_trajectory_vs_depth_with_error_supports_per_component_bounds() -> None:
    m = 2
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(box_bounds=[(-1.0, 1.0), (-2.0, 2.0)]),
    )
    for j, expected in enumerate([(-1.0, 1.0), (-2.0, 2.0)]):
        bound_lines = fig.axes[j].get_lines()[-2:]
        y_values = sorted(
            float(np.asarray(line.get_ydata())[0]) for line in bound_lines
        )
        assert y_values == pytest.approx(list(expected))


def test_plot_trajectory_vs_depth_with_error_ylim_widens_for_a_tight_bound_outside_data() -> (
    None
):
    """A bound far outside the plotted data must still be visible (NB04
    plan Sec 3.5's "never clipped off-axis" requirement)."""
    m = 1
    fig = plot_trajectory_vs_depth_with_error(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(box_bounds=(-10.0, 10.0)),
    )
    ylim = fig.axes[0].get_ylim()
    assert ylim[0] < -10.0
    assert ylim[1] > 10.0


def test_depth_trajectory_animator_draws_bound_lines_in_every_signal_axis() -> None:
    m = 2
    animator = DepthTrajectoryAnimator(
        _baseline(m),
        _depth_trajectories(m),
        style=DepthTrajectoryStyle(box_bounds=(-1.0, 1.0)),
    )
    animator.build()
    animator._fig.canvas.draw()

    for j in range(m):
        ax = animator._fig.axes[j]
        # baseline + the (empty, then updated) candidate line + 2 bound lines
        assert len(ax.get_lines()) == 1 + 1 + 2

    ylim = animator._fig.axes[0].get_ylim()
    assert ylim[0] <= -1.0
    assert ylim[1] >= 1.0


def test_depth_trajectory_animator_legend_reports_depth_and_cost_per_frame() -> None:
    m = 2
    trajectories = _depth_trajectories(m)
    costs = {depth: 1.0 / depth for depth in trajectories}
    animator = DepthTrajectoryAnimator(
        _baseline(m),
        trajectories,
        style=DepthTrajectoryStyle(contender_name="Unfolded-alpha"),
        costs=costs,
        baseline_cost=0.25,
    )
    animator.build()
    animator._fig.canvas.draw()

    assert animator._candidate_legend_text is not None
    baseline_legend_text = animator._fig.axes[0].get_legend().get_texts()[0]
    assert baseline_legend_text.get_text() == "Riccati (J=0.2500)"

    for frame_index, depth in enumerate(sorted(trajectories)):
        animator.update(frame_index)
        expected = f"Unfolded-alpha (K={depth} (J={costs[depth]:.4f}))"
        assert animator._candidate_legend_text.get_text() == expected
