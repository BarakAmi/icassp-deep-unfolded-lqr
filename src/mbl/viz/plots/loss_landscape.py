"""3D loss/cost landscape plots over a 2D parameter grid -- generalizes the
legacy 2D contour plot (which hardcoded the LQR quadratic stage-cost formula)
to an arbitrary scalar loss function of two parameters.

Pure renderer (REFACTOR_PLAN v3, T4.b): grids in, Figure out. The grid
*computation* lives with the workbench
(`src.workbench.analysis.compute_loss_grid`); the collision-avoiding 3D
annotation placement is shared with Asset 5's scatter renderer via
`viz.plots.annotations3d`.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from ...core.utils import ensure_array_shape, ensure_last_dim, ensure_ndim, to_numpy
from .annotations3d import (
    ANNOTATION_Z_OFFSET_FRACTION,
    place_trajectory_annotations,
)
from ..style import style_3d_axes, styled_figure


@dataclass(frozen=True)
class LossLandscapeLabels:
    """Axis/legendless labeling for `plot_loss_landscape_3d`.

    Attributes:
        param_1_label/param_2_label: The two parameter-axis labels.
        loss_label: The z-axis (and colorbar) label.
        title: Figure title.
    """

    param_1_label: str = r"$\theta_1$"
    param_2_label: str = r"$\theta_2$"
    loss_label: str = "Loss"
    title: str = "Loss Landscape"


@dataclass(frozen=True)
class LossLandscapeOverlay:
    """The optional optimizer-path and optimum-marker overlay of
    `plot_loss_landscape_3d`, as one value.

    Attributes:
        trajectory: Optional ``(n_iters, 2)`` sequence of
            ``(param_1, param_2)`` iterates.
        trajectory_losses: The loss at each `trajectory` row (must accompany
            it).
        trajectory_label: Legend label for the path (generic by default; a
            caller working with a specific algorithm may pass something more
            descriptive, e.g. "Gradient descent Path").
        trajectory_annotations: Optional per-point text label (e.g.
            ``"i=0, J=5.547"``), one per `trajectory` row -- placed near its
            marker by the greedy screen-space collision search
            (Micro-Prompt 4c).
        optimal_point: Optional ``(2,)`` point in parameter space to mark.
        optimal_value: The loss at `optimal_point` (must accompany it).
        optimum_label: Legend label for `optimal_point`.
    """

    trajectory: np.ndarray | None = None
    trajectory_losses: np.ndarray | None = None
    trajectory_label: str = "Optimization Path"
    trajectory_annotations: Sequence[str] | None = None
    optimal_point: np.ndarray | None = None
    optimal_value: float | None = None
    optimum_label: str = "Optimum"


def plot_loss_landscape_3d(
    param_1_grid: np.ndarray,
    param_2_grid: np.ndarray,
    loss_grid: np.ndarray,
    *,
    labels: LossLandscapeLabels = LossLandscapeLabels(),
    overlay: LossLandscapeOverlay = LossLandscapeOverlay(),
) -> Figure:
    """Render a loss surface over a 2-parameter grid, with an optional
    optimizer trajectory and optimum marker overlaid.

    Args:
        param_1_grid, param_2_grid: meshgrid arrays (as from `np.meshgrid` or
            `compute_loss_grid`), each of shape (rows, cols).
        loss_grid: loss value at each grid point, same shape.
        labels: axis labels and title (see `LossLandscapeLabels`).
        overlay: the optional optimizer path / optimum marker (see
            `LossLandscapeOverlay`).

    Returns:
        The Figure, containing a single 3D-projected Axes.

    Raises:
        ValueError: If ``overlay.trajectory``/``trajectory_losses`` or
            ``optimal_point``/``optimal_value`` are given one without the
            other, or ``trajectory_annotations`` is given without
            ``trajectory`` or with a mismatched length.
    """
    trajectory = overlay.trajectory
    trajectory_losses = overlay.trajectory_losses
    trajectory_annotations = overlay.trajectory_annotations
    optimal_point, optimal_value = overlay.optimal_point, overlay.optimal_value
    param_1_grid = to_numpy(param_1_grid)
    param_2_grid = to_numpy(param_2_grid)
    loss_grid = to_numpy(loss_grid)
    ensure_array_shape(param_2_grid, param_1_grid.shape, "param_2_grid")
    ensure_array_shape(loss_grid, param_1_grid.shape, "loss_grid")

    if (trajectory is None) != (trajectory_losses is None):
        raise ValueError("trajectory and trajectory_losses must be provided together.")
    if trajectory_annotations is not None:
        if trajectory is None:
            raise ValueError(
                "trajectory_annotations requires trajectory to be given too."
            )
        if len(trajectory_annotations) != len(trajectory):
            raise ValueError(
                f"trajectory_annotations must have length {len(trajectory)} "
                f"(one per trajectory row), got {len(trajectory_annotations)}."
            )
    if (optimal_point is None) != (optimal_value is None):
        raise ValueError("optimal_point and optimal_value must be provided together.")

    with styled_figure():
        # Wider than the (10, 6) themed default (NB03 overhaul, second
        # review pass): this figure carries BOTH a colorbar AND (when an
        # overlay is given) a side-anchored legend -- measured directly
        # against `fig.bbox`, the 3D axes collapsed to ~21% of the default
        # canvas width trying to share it with both. THIRD review pass:
        # 15in (then 9.4in) over-corrected -- the legend/colorbar's content
        # doesn't scale with figure width, so any excess becomes a dead
        # margin past the legend's own right edge (measured ~540px of
        # unused canvas at 15in); 7.5in is sized empirically (by directly
        # measuring `get_window_extent()`/`get_tightbbox()` of the axes,
        # colorbar, and legend together) to what the content actually
        # needs, snug gaps included.
        fig = plt.figure(figsize=(15.0, 7.0))
        ax = fig.add_subplot(projection="3d")
        surface = ax.plot_surface(
            param_1_grid,
            param_2_grid,
            loss_grid,
            cmap="viridis",
            alpha=0.7,
            linewidth=0,
            antialiased=True,
        )
        # pad=0.05 (third review pass, down from 0.12): the larger pad left
        # a big dead gap between the axes and the colorbar, confirmed
        # directly in a rendered PNG; the z-tick/label clipping the larger
        # pad originally guarded against (a 3D axes' z-label/ticks project
        # further right than a 2D axes' would) did not reproduce at 0.05,
        # re-checked directly against a rendered PNG after this change.
        fig.colorbar(
            surface, ax=ax, shrink=0.5, aspect=10, pad=0.05, label=labels.loss_label
        )

        show_legend = False
        if trajectory is not None:
            assert trajectory_losses is not None  # validated together above
            trajectory = to_numpy(trajectory)
            trajectory_losses = to_numpy(trajectory_losses)
            ensure_ndim("trajectory", trajectory, 2)
            ensure_last_dim("trajectory", trajectory, 2)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory_losses,
                color="black",
                # No marker (NB03 overhaul): this function's one caller,
                # `landscapes.SurfaceLandscapeRenderer`, already draws its
                # own circular per-point markers on top of this connecting
                # line -- the previous `marker="x"` doubled up as a visible
                # black X under each colored circle.
                label=overlay.trajectory_label,
            )
            show_legend = True

        if optimal_point is not None:
            optimal_point = to_numpy(optimal_point)
            ax.scatter(
                *optimal_point,
                optimal_value,
                color="red",
                edgecolor="black",
                s=80,
                marker="*",
                label=overlay.optimum_label,
            )
            show_legend = True

        ax.set_xlabel(labels.param_1_label)
        ax.set_ylabel(labels.param_2_label)
        ax.set_zlabel(labels.loss_label)
        ax.set_title(labels.title)
        if show_legend:
            # Shrunk font + anchored to the SIDE, not below (NB03 overhaul: a
            # bottom anchor forces `fig.tight_layout(rect=...)` below to
            # reserve a large bottom margin, squashing a 3D axes into a small
            # fraction of the figure -- confirmed directly in a rendered
            # PNG) -- `ax.legend()`'s bare default for a 3D axes lands near
            # the bottom of the cube and collides with its own tick labels,
            # so this stays an explicit external placement either way.
            # FIGURE-fraction anchor (`bbox_transform=fig.transFigure`), not
            # axes-fraction (third review pass): an axes-fraction anchor
            # COUPLES the legend's position to the axes' own (variable)
            # width, so retuning it to sit snugly against the colorbar at
            # one figure size drifted back into either an overlap or a
            # large gap at another -- confirmed directly by sweeping
            # several figsize/anchor combinations and measuring
            # `get_tightbbox()` (not `get_window_extent()`, which omits a
            # colorbar's own tick labels and undercounts its true width) on
            # the axes, colorbar, and legend together. A figure-fraction
            # anchor stays put regardless of how wide the axes render.
            ax.legend(
                loc="center left",
                bbox_to_anchor=(0.555, 0.5),
                bbox_transform=fig.transFigure,
                ncol=1,
                borderaxespad=0.0,
                prop={"size": 8.5},
            )
        # Fixes the camera (box aspect, elevation, azimuth) BEFORE any
        # annotation is placed below: `_place_trajectory_annotations`'s
        # collision search projects against this exact final view, so
        # placing it any earlier would search against a stale projection.
        style_3d_axes(ax)

        if trajectory is not None and trajectory_annotations is not None:
            assert trajectory_losses is not None  # validated together above
            z_span = float(loss_grid.max() - loss_grid.min()) or 1.0
            place_trajectory_annotations(
                ax,
                fig,
                trajectory,
                trajectory_losses,
                trajectory_annotations,
                ANNOTATION_Z_OFFSET_FRACTION * z_span,
            )

        # Reserve the right-hand margin the side-anchored legend above needs
        # only when one was actually drawn; otherwise let the axes use the
        # full figure. 0.52 (third review pass, empirically fit to the
        # narrower figure above) reserves just the axes + colorbar strip.
        fig.tight_layout(
            rect=(0.02, 0.0, 0.52, 1.0) if show_legend else (0.0, 0.0, 1.0, 1.0)
        )

    return fig
