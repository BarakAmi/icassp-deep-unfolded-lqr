import numpy as np
import pytest
from matplotlib.figure import Figure

from mbl.viz.plots.loss_landscape import (
    LossLandscapeLabels,
    LossLandscapeOverlay,
    plot_loss_landscape_3d,
)
from mbl.workbench.analysis import compute_loss_grid
from mbl.viz.style.theme import DEFAULT_3D_AZIM, DEFAULT_3D_ELEV


def _bowl_loss(params: np.ndarray) -> float:
    """Simple convex loss with a known minimum at the origin, used to check
    `compute_loss_grid` evaluates correctly rather than just "not crashing"."""
    return float(params[0] ** 2 + 2.0 * params[1] ** 2)


def test_compute_loss_grid_matches_direct_evaluation() -> None:
    p1_range = np.linspace(-2.0, 2.0, 25)
    p2_range = np.linspace(-3.0, 3.0, 30)

    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    assert p1_grid.shape == (30, 25)
    assert loss_grid.shape == (30, 25)
    for i in (0, 10, 29):
        for j in (0, 12, 24):
            expected = _bowl_loss(np.array([p1_grid[i, j], p2_grid[i, j]]))
            assert loss_grid[i, j] == pytest.approx(expected)


def test_plot_loss_landscape_3d_returns_figure_with_3d_axes() -> None:
    p1_range = np.linspace(-2.0, 2.0, 40)
    p2_range = np.linspace(-2.0, 2.0, 40)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
    )

    assert isinstance(fig, Figure)
    # The main 3D axes, plus its colorbar's own (2D) axes.
    assert len(fig.axes) == 2
    assert fig.axes[0].name == "3d"


def test_plot_loss_landscape_3d_overlays_trajectory_and_optimum() -> None:
    p1_range = np.linspace(-2.0, 2.0, 30)
    p2_range = np.linspace(-2.0, 2.0, 30)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    n_iters = 20
    trajectory = np.column_stack(
        [np.linspace(1.8, 0.0, n_iters), np.linspace(-1.8, 0.0, n_iters)]
    )
    trajectory_losses = np.array([_bowl_loss(p) for p in trajectory])

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
        overlay=LossLandscapeOverlay(
            trajectory=trajectory,
            trajectory_losses=trajectory_losses,
            optimal_point=np.array([0.0, 0.0]),
            optimal_value=0.0,
        ),
    )

    ax = fig.axes[0]
    assert ax.get_legend() is not None


def test_plot_loss_landscape_3d_uses_custom_legend_labels() -> None:
    p1_range = np.linspace(-2.0, 2.0, 20)
    p2_range = np.linspace(-2.0, 2.0, 20)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    n_iters = 5
    trajectory = np.column_stack(
        [np.linspace(1.8, 0.0, n_iters), np.linspace(-1.8, 0.0, n_iters)]
    )
    trajectory_losses = np.array([_bowl_loss(p) for p in trajectory])

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
        overlay=LossLandscapeOverlay(
            trajectory=trajectory,
            trajectory_losses=trajectory_losses,
            trajectory_label="Gradient descent Path",
            optimal_point=np.array([0.0, 0.0]),
            optimal_value=0.0,
            optimum_label="Optimum by Riccati",
        ),
    )

    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Gradient descent Path", "Optimum by Riccati"]


def test_plot_loss_landscape_3d_default_labels_unchanged_for_existing_callers() -> None:
    """`dashboard/app.py` and any other pre-existing caller must keep seeing
    the original generic labels when it doesn't opt into the new params."""
    p1_range = np.linspace(-2.0, 2.0, 20)
    p2_range = np.linspace(-2.0, 2.0, 20)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
        overlay=LossLandscapeOverlay(
            trajectory=np.zeros((3, 2)),
            trajectory_losses=np.zeros(3),
            optimal_point=np.array([0.0, 0.0]),
            optimal_value=0.0,
        ),
    )

    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Optimization Path", "Optimum"]


def test_plot_loss_landscape_3d_has_a_colorbar() -> None:
    p1_range = np.linspace(-2.0, 2.0, 20)
    p2_range = np.linspace(-2.0, 2.0, 20)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
        labels=LossLandscapeLabels(loss_label="Cost"),
    )

    colorbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(colorbar_axes) == 1
    assert colorbar_axes[0].get_ylabel() == "Cost"


def test_plot_loss_landscape_3d_applies_the_deep_bowl_3d_styling() -> None:
    """Asset 4b's perspective fix (label padding, box aspect, camera angle)
    must be applied to every rendered surface, not left to caller opt-in."""
    p1_range = np.linspace(-2.0, 2.0, 20)
    p2_range = np.linspace(-2.0, 2.0, 20)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
    )

    ax = fig.axes[0]
    assert ax.xaxis.labelpad == 15.0
    assert ax.yaxis.labelpad == 15.0
    assert ax.zaxis.labelpad == 15.0
    x, y, z = np.asarray(ax.get_box_aspect())
    assert x == pytest.approx(y)
    assert z / x == pytest.approx(1.2)
    assert ax.elev == pytest.approx(DEFAULT_3D_ELEV)
    assert ax.azim == pytest.approx(DEFAULT_3D_AZIM)


def test_plot_loss_landscape_3d_places_annotations_as_screen_space_text() -> None:
    """Micro-Prompt 4c: trajectory labels are placed via `ax.text2D` (a
    fixed screen-space/axes-fraction position from the greedy collision
    search), not `ax.text` (a 3D data coordinate re-projected on every
    redraw) -- otherwise a search-picked non-colliding position wouldn't
    survive verbatim. Two well-separated real points must land at clearly
    distinct, well-separated screen positions."""
    p1_range = np.linspace(-2.0, 2.0, 20)
    p2_range = np.linspace(-2.0, 2.0, 20)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    trajectory = np.array([[1.5, -1.5], [0.0, 0.0]])
    trajectory_losses = np.array([_bowl_loss(p) for p in trajectory])
    annotations = ["i=0, J=6.750", "i=10, J=0.000"]

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
        overlay=LossLandscapeOverlay(
            trajectory=trajectory,
            trajectory_losses=trajectory_losses,
            trajectory_annotations=annotations,
        ),
    )

    ax = fig.axes[0]
    texts = {t.get_text(): t for t in ax.texts if t.get_text() in annotations}
    assert set(texts) == set(annotations)
    positions = []
    for label in annotations:
        assert not hasattr(texts[label], "get_position_3d")  # screen-space, not 3D data
        x, y = texts[label].get_position()
        assert np.isfinite(x) and np.isfinite(y)
        positions.append((x, y))
    separation = np.linalg.norm(np.array(positions[0]) - np.array(positions[1]))
    assert separation > 0.05


def test_plot_loss_landscape_3d_separates_annotations_for_clustered_points() -> None:
    """Micro-Prompt 4c: near-coincident trajectory points (as exponential GD
    contraction near the optimum produces) must still get text labels
    genuinely separated on screen, via the greedy collision-avoiding search
    -- not merely non-identical, which a coincidental tie could still
    satisfy while visually overlapping."""
    p1_range = np.linspace(-2.0, 2.0, 20)
    p2_range = np.linspace(-2.0, 2.0, 20)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)

    # Four nearly-coincident late-iteration points, as clustering near the
    # optimum produces in practice.
    trajectory = np.array([[1.5, -1.5], [0.01, -0.01], [0.005, -0.005], [0.0, 0.0]])
    trajectory_losses = np.array([_bowl_loss(p) for p in trajectory])
    annotations = ["i=0, J=6.750", "i=13, J=0.000", "i=26, J=0.000", "i=100, J=0.000"]

    fig = plot_loss_landscape_3d(
        p1_grid,
        p2_grid,
        loss_grid,
        overlay=LossLandscapeOverlay(
            trajectory=trajectory,
            trajectory_losses=trajectory_losses,
            trajectory_annotations=annotations,
        ),
    )

    ax = fig.axes[0]
    texts = {t.get_text(): t for t in ax.texts if t.get_text() in annotations}
    clustered_positions = [
        np.array(texts[label].get_position()) for label in annotations[1:]
    ]
    for i in range(len(clustered_positions)):
        for j in range(i + 1, len(clustered_positions)):
            distance = np.linalg.norm(clustered_positions[i] - clustered_positions[j])
            assert distance >= 0.08  # pairwise-isolated, not merely non-identical


def test_plot_loss_landscape_3d_rejects_annotations_without_trajectory() -> None:
    p1_grid = np.zeros((5, 5))
    p2_grid = np.zeros((5, 5))
    loss_grid = np.zeros((5, 5))

    with pytest.raises(ValueError, match="trajectory_annotations"):
        plot_loss_landscape_3d(
            p1_grid,
            p2_grid,
            loss_grid,
            overlay=LossLandscapeOverlay(trajectory_annotations=["i=0"]),
        )


def test_plot_loss_landscape_3d_rejects_mismatched_annotation_length() -> None:
    p1_range = np.linspace(-2.0, 2.0, 10)
    p2_range = np.linspace(-2.0, 2.0, 10)
    p1_grid, p2_grid, loss_grid = compute_loss_grid(_bowl_loss, p1_range, p2_range)
    trajectory = np.array([[1.5, -1.5], [0.0, 0.0]])
    trajectory_losses = np.array([_bowl_loss(p) for p in trajectory])

    with pytest.raises(ValueError, match="trajectory_annotations"):
        plot_loss_landscape_3d(
            p1_grid,
            p2_grid,
            loss_grid,
            overlay=LossLandscapeOverlay(
                trajectory=trajectory,
                trajectory_losses=trajectory_losses,
                trajectory_annotations=["only one"],
            ),
        )


def test_plot_loss_landscape_3d_rejects_mismatched_grid_shapes() -> None:
    p1_grid = np.zeros((10, 10))
    p2_grid = np.zeros((10, 5))
    loss_grid = np.zeros((10, 10))

    with pytest.raises(ValueError, match="param_2_grid"):
        plot_loss_landscape_3d(
            p1_grid,
            p2_grid,
            loss_grid,
        )


def test_plot_loss_landscape_3d_requires_trajectory_and_losses_together() -> None:
    p1_grid = np.zeros((5, 5))
    p2_grid = np.zeros((5, 5))
    loss_grid = np.zeros((5, 5))

    with pytest.raises(ValueError, match="trajectory_losses"):
        plot_loss_landscape_3d(
            p1_grid,
            p2_grid,
            loss_grid,
            overlay=LossLandscapeOverlay(trajectory=np.zeros((3, 2))),
        )


def test_plot_loss_landscape_3d_requires_optimal_point_and_value_together() -> None:
    p1_grid = np.zeros((5, 5))
    p2_grid = np.zeros((5, 5))
    loss_grid = np.zeros((5, 5))

    with pytest.raises(ValueError, match="optimal_value"):
        plot_loss_landscape_3d(
            p1_grid,
            p2_grid,
            loss_grid,
            overlay=LossLandscapeOverlay(optimal_point=np.array([0.0, 0.0])),
        )
