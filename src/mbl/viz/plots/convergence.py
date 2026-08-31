"""Convergence visualizations for the "unfolded" gradient-descent
controllers, compared against COCP's one-shot convex solve:

- `plot_control_convergence_contour_2d`: for a 2D control space, the classic
  "cost landscape + optimization path" view -- an unfolded model's per-
  iteration control iterates walking toward its optimum, overlaid on both its
  own and COCP's implicit stage-cost landscape (COCP jumps there in one QP
  solve; the unfolded model walks there over K gradient steps).
- `plot_high_dimensional_convergence`: the same convergence story for a
  control space with more than 2 dimensions, where a single 2D contour can no
  longer show every axis at once.
- `plot_convergence_curves`: the Global Convergence Diagnostics figure --
  one log-scale expected-cost-vs-iteration curve per solved
  `OptimizationResult`-shaped record, against the theoretical optimum.

Pure renderers (REFACTOR_PLAN v3, T4.b): frozen result data in, Figure out.
The grid/path *computation* these figures need lives with the workbench
(`src.workbench.analysis.compute_stage_cost_grid` /
`compute_gradient_descent_path`), never in this module.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ...core.utils import ensure_last_dim, ensure_ndim, to_numpy
from ..style import OPTIMUM_COLOR, series_color, styled_figure

#: Reserved exclusively for "the optimum" marker (closed-form Riccati/COCP
#: solution) so it always reads unambiguously as the target -- the semantic
#: role from `style.semantic.SEMANTIC_COLORS`, defined once.
_OPTIMUM_COLOR = OPTIMUM_COLOR
#: Kept distinct from _OPTIMUM_COLOR (both being "red" would make the box
#: boundary and the optimum marker easy to confuse at a glance).
_BOX_CONSTRAINT_COLOR = "black"


@runtime_checkable
class ConvergenceRecord(Protocol):
    """The slice of an iterative solve's result this module's renderers
    consume -- structurally satisfied by `models.iterative.OptimizationResult`
    without this Tier-4 layer ever importing it (the T4.b decoupling law),
    and by the sidecar-reconstructed `ConvergenceCurve` alike."""

    @property
    def J_history(self) -> np.ndarray: ...

    @property
    def J_final(self) -> float: ...


def _draw_box_constraint(ax: Any, u_max: float) -> None:
    """Draw the |u_i| <= u_max box constraint as a dashed square directly on
    the control-space contour, so the feasible region is visible alongside
    the cost landscape it's carved out of."""
    ax.add_patch(
        Rectangle(
            (-u_max, -u_max),
            width=2 * u_max,
            height=2 * u_max,
            fill=False,
            edgecolor=_BOX_CONSTRAINT_COLOR,
            linestyle="--",
            linewidth=2.0,
            zorder=7,
            label=f"Box constraint (|u| ≤ {u_max:g})",
        )
    )


def plot_convergence_curves(
    records: Mapping[str, ConvergenceRecord],
    J_opt: float,
    *,
    title: str,
) -> Figure:
    """The log-scale expected-cost-vs-iteration figure shared by the Global
    Convergence Diagnostics experiments: one curve per record, each legend
    entry reporting that strategy's final empirical cost (e.g. `"Gauss-Seidel
    (Final J: 0.471)"`), plus a dashed reference line at the theoretical
    optimum `J_opt`. Pure: renders and returns the Figure -- persistence and
    display belong to the adapter layer's `FigureSink`.

    Args:
        records: label -> solved record exposing `J_history`/`J_final`
            (e.g. `models.iterative.OptimizationResult`).
        J_opt: the theoretical (Riccati) optimal expected cost, drawn as a
            dashed reference line.
        title: figure title.

    Returns:
        The Figure.
    """
    if not records:
        raise ValueError("records must contain at least one named record.")

    markers_iter = iter(plt.Line2D.filled_markers)
    linestyles_iter = iter(plt.Line2D.lineStyles)
    linewidths_iter = iter(np.linspace(1.0, 3.0, num=len(records)))
    with styled_figure():
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, record in records.items():
            ax.plot(
                to_numpy(record.J_history),
                linewidth=next(linewidths_iter),
                markersize=5,
                marker=next(markers_iter),
                linestyle=next(linestyles_iter),
                label=f"{name} (Final J: {record.J_final:.3f})",
            )
        ax.axhline(
            J_opt,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=rf"$J_{{\mathrm{{opt}}}} = {J_opt:.3f}$ (optimal feedback law)",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration $i$")
        ax.set_ylabel(r"Expected cost $\widehat{\mathbb{E}}[J(U^{(i)})]$")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
    return fig


@dataclass(frozen=True)
class CocpOverlay:
    """The optional COCP-comparison overlay of
    `plot_control_convergence_contour_2d`: COCP's own implicit landscape,
    its one-shot solution point, and the box bound.

    Attributes:
        cost_grid: Optional stage cost at each grid point under COCP's
            learned cost-to-go, same shape as the unfolded grid.
        point: Optional ``(2,)`` COCP directly-solved control.
        u_max: Optional box-constraint bound (``|u_i| <= u_max``); when
            given, drawn as a dashed square so the feasible region is
            visible directly on the contour.
    """

    cost_grid: np.ndarray | None = None
    point: np.ndarray | None = None
    u_max: float | None = None


@dataclass(frozen=True)
class ContourLabels:
    """Axis labels and title for `plot_control_convergence_contour_2d`."""

    u1_label: str = r"$u_1$"
    u2_label: str = r"$u_2$"
    title: str = "Control Convergence: Unfolded vs. COCP"


def plot_control_convergence_contour_2d(
    u1_grid: np.ndarray,
    u2_grid: np.ndarray,
    unfolded_cost_grid: np.ndarray,
    unfolded_trajectory: np.ndarray,
    *,
    cocp: CocpOverlay = CocpOverlay(),
    labels: ContourLabels = ContourLabels(),
) -> Figure:
    """A 2D stage-cost contour (control_dim == 2) with an unfolded
    controller's per-iteration convergence path overlaid, optionally compared
    against COCP's own implicit cost landscape and its single one-shot
    solution point.

    Args:
        u1_grid, u2_grid: meshgrid arrays (as from `compute_stage_cost_grid`).
        unfolded_cost_grid: stage cost at each grid point under the
            unfolded model's cost-to-go, same shape as u1_grid.
        unfolded_trajectory: (K+1, 2) control iterates, initial guess to
            converged value.
        cocp: the optional COCP-comparison overlay (see `CocpOverlay`).
        labels: axis labels and title (see `ContourLabels`).

    Returns:
        The Figure.
    """
    cocp_cost_grid, cocp_point, u_max = cocp.cost_grid, cocp.point, cocp.u_max
    u1_grid, u2_grid = to_numpy(u1_grid), to_numpy(u2_grid)
    unfolded_cost_grid = to_numpy(unfolded_cost_grid)
    unfolded_trajectory = to_numpy(unfolded_trajectory)
    if u1_grid.shape != u2_grid.shape or u1_grid.shape != unfolded_cost_grid.shape:
        raise ValueError(
            "u1_grid, u2_grid, and unfolded_cost_grid must all share the same "
            f"shape; got {u1_grid.shape}, {u2_grid.shape}, "
            f"{unfolded_cost_grid.shape}."
        )
    ensure_ndim("unfolded_trajectory", unfolded_trajectory, 2)
    ensure_last_dim("unfolded_trajectory", unfolded_trajectory, 2)
    if cocp_cost_grid is not None and to_numpy(cocp_cost_grid).shape != u1_grid.shape:
        raise ValueError("cocp_cost_grid must have the same shape as u1_grid.")

    with styled_figure():
        fig, ax = plt.subplots()
        contour_fill = ax.contourf(
            u1_grid, u2_grid, unfolded_cost_grid, levels=30, cmap="viridis", alpha=0.9
        )
        fig.colorbar(contour_fill, ax=ax, label="Stage cost (unfolded landscape)")

        if cocp_cost_grid is not None:
            ax.contour(
                u1_grid,
                u2_grid,
                to_numpy(cocp_cost_grid),
                levels=30,
                colors=series_color("cocp"),
                linewidths=1.0,
                linestyles="--",
            )

        if u_max is not None:
            _draw_box_constraint(ax, u_max)

        unfolded_color = series_color("unfolded_learned_step_size")
        ax.plot(
            unfolded_trajectory[:, 0],
            unfolded_trajectory[:, 1],
            color=unfolded_color,
            marker="o",
            markersize=5,
            linewidth=2.0,
            alpha=0.95,
            label="Unfolded path",
            zorder=8,
        )
        ax.scatter(
            unfolded_trajectory[-1, 0],
            unfolded_trajectory[-1, 1],
            s=90,
            color=unfolded_color,
            edgecolor="black",
            zorder=8,
            label="Unfolded final",
        )

        if cocp_point is not None:
            cocp_point = to_numpy(cocp_point)
            ax.scatter(
                cocp_point[0],
                cocp_point[1],
                marker="*",
                s=400,
                color=_OPTIMUM_COLOR,
                edgecolor="black",
                linewidth=0.9,
                zorder=9,
                alpha=0.95,
                label="COCP solution (optimum)",
            )

        ax.set_xlabel(labels.u1_label)
        ax.set_ylabel(labels.u2_label)
        ax.set_aspect("equal")
        ax.set_title(labels.title)
        # Legend placed below the axes (not to the side): this figure already
        # has a colorbar occupying the right margin, so an external legend
        # there would collide with it.
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, borderaxespad=0.0
        )
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    return fig


def plot_high_dimensional_convergence(
    trajectory: np.ndarray,
    *,
    dim_labels: Sequence[str] | None = None,
    title: str = "High-Dimensional Control Convergence",
) -> Figure:
    """For control_dim > 2, where a single 2D contour can't show every axis:
    a per-dimension heatmap of the control iterates over unfolding iteration
    k, paired with the aggregate distance-to-converged-value curve -- the
    detailed and summary views of the same convergence process.

    Args:
        trajectory: (K+1, m) control iterates, initial guess to converged
            value, for any m >= 1 (most useful for m > 2).
        dim_labels: optional per-dimension labels; defaults to "u_0", "u_1", ...

    Returns:
        The Figure, with a heatmap panel and a convergence-norm panel.
    """
    trajectory = to_numpy(trajectory)
    ensure_ndim("trajectory", trajectory, 2)
    num_iterations, dim = trajectory.shape
    if dim_labels is not None and len(dim_labels) != dim:
        raise ValueError(
            f"dim_labels must have length {dim} (the control dimension), "
            f"got {len(dim_labels)}."
        )
    labels = list(dim_labels) if dim_labels else [f"$u_{{{i}}}$" for i in range(dim)]

    with styled_figure():
        fig, (ax_heatmap, ax_norm) = plt.subplots(
            nrows=2, ncols=1, figsize=(10, 8), height_ratios=(2, 1)
        )

        image = ax_heatmap.imshow(
            trajectory.T, aspect="auto", cmap="Blues", origin="lower"
        )
        ax_heatmap.set_yticks(range(dim))
        ax_heatmap.set_yticklabels(labels)
        ax_heatmap.set_xlabel("Unfolding iteration k")
        ax_heatmap.set_title(title)
        fig.colorbar(image, ax=ax_heatmap, label="Control value")

        converged_value = trajectory[-1]
        distance_to_converged = np.linalg.norm(trajectory - converged_value, axis=1)
        iterations = np.arange(num_iterations)
        ax_norm.plot(
            iterations,
            distance_to_converged,
            marker="o",
            color=series_color("unfolded_learned_step_size"),
        )
        ax_norm.set_xlabel("Unfolding iteration k")
        ax_norm.set_ylabel(r"$\|u_k - u_K\|_2$")
        ax_norm.set_title("Distance to converged control")

        fig.tight_layout()

    return fig
