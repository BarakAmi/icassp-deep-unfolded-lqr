"""Cost-over-time and cost-over-unfolding-depth plots -- the "flagship"
comparison for this project's applications: how a single rollout's cost
accumulates over the horizon, and how a trained unfolded model's converged
cost depends on its unfolding depth K, against every non-iterative baseline
(analytic/COCP/neural/SDP lower bound) as a flat reference line.

Pure renderers (REFACTOR_PLAN v3, T4.b): plain arrays / named-series
mappings in, Figure out -- the cumulative-cost *computation* lives with the
workbench (`src.workbench.analysis.compute_cumulative_average_cost`), never
in this module.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from ...core.utils import to_numpy
from ..style import (
    AGGREGATE_COLOR,
    CATEGORICAL_PALETTE,
    OPTIMUM_COLOR,
    ReferenceLineStyle,
    autoscaled_ylim_from_data,
    draw_reference_lines,
    place_legend_outside,
    series_color,
    styled_figure,
)

#: Triangles/squares/diamonds first (per the publication style guide), circle
#: and plus as later fallbacks for a 7th+ curve.
_MARKERS = ("^", "s", "D", "v", "o", "P")
#: Cycled per K-dependent curve so multiple iterative sweeps stay
#: distinguishable by line style too, not just marker/color.
_LINESTYLES = ("-", "-.")


def plot_cumulative_cost_over_time(
    costs: Mapping[str, np.ndarray],
    *,
    reference_lines: Mapping[str, float] | None = None,
    xlabel: str = "Time step",
    ylabel: str = "Cumulative average cost",
    title: str = "Cumulative Cost Over Time",
    log_scale: bool = False,
) -> Figure:
    """Plot one or more named cumulative-cost-over-the-horizon curves (e.g.
    the running average of x_t^T Q x_t + u_t^T R u_t from t=0), with optional
    flat reference lines for non-time-varying baselines.

    Args:
        costs: label -> 1D sequence of per-time-step cumulative cost values.
        reference_lines: label -> constant value, drawn as a dashed horizontal
            line (e.g. {"Theoretical Optimal": 1.23}).

    Returns:
        The Figure.
    """
    if not costs:
        raise ValueError("costs must contain at least one named curve.")

    with styled_figure():
        fig, ax = plt.subplots()
        for i, (label, values) in enumerate(costs.items()):
            values = to_numpy(values)
            if values.ndim != 1:
                raise ValueError(
                    f"Curve '{label}' must be 1D (one value per time step), "
                    f"got shape {values.shape}."
                )
            ax.plot(
                np.arange(1, len(values) + 1),
                values,
                color=series_color(label, fallback_index=i),
                label=label,
            )

        draw_reference_lines(ax, reference_lines)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        place_legend_outside(fig, ax)

    return fig


@dataclass(frozen=True)
class CostVsDepthStyle:
    """Presentation knobs for `plot_cost_vs_unfolding_depth` beyond its data
    inputs (PLR0913: keeps the renderer under the 6-argument cap).

    Attributes:
        xlabel: Shared x-axis label.
        ylabel: Shared y-axis label.
        title: Figure title.
        legend_inside: Draw the legend inside the axes instead of the
            default external placement (see `viz.style.theme
            .place_legend_outside`).
        ylim: explicit ``(low, high)`` y-axis override. ``None`` (default)
            keeps matplotlib's own autoscale over every artist, curves and
            reference lines alike.
        autoscale_to_curves: when ``True``, set the y-limits from the
            CURVE data only, so a reference line far outside the curves'
            own range (e.g. a loose theoretical floor) no longer dictates
            the axis -- it stays fully reported via its legend entry's
            folded-in value (NB04 reference-bounds plan Sec 2.4). Ignored
            when `ylim` is also set (`ylim` wins).
    """

    xlabel: str = "Unfolding depth K"
    ylabel: str = "Estimated average cost"
    title: str = "Cost vs. Unfolding Depth"
    legend_inside: bool = False
    ylim: tuple[float, float] | None = None
    autoscale_to_curves: bool = False


def plot_cost_vs_unfolding_depth(
    k_values: Sequence[int],
    curves: Mapping[str, Sequence[float]],
    *,
    reference_lines: Mapping[str, float] | None = None,
    reference_line_styles: Mapping[str, ReferenceLineStyle] | None = None,
    style: CostVsDepthStyle = CostVsDepthStyle(),
) -> Figure:
    """The flagship plot: final evaluation cost against an unfolded model's
    unfolding depth K, with every non-iterative baseline (analytic Riccati,
    COCP, neural, the SDP lower bound, ...) overlaid as a flat horizontal
    line -- since those models' cost doesn't depend on K.

    Args:
        k_values: the unfolding depths evaluated, shared x-axis for every
            entry in `curves`.
        curves: label -> cost per K (same length/order as `k_values`) for
            each iterative (K-dependent) model.
        reference_lines: label -> constant cost for each non-iterative model.
        reference_line_styles: optional label -> `ReferenceLineStyle`
            overrides, so removing an entry from `reference_lines` never
            silently reassigns the remaining lines' colors/linestyles (see
            `draw_reference_lines`).
        style: labeling/legend-placement/axis knobs (see `CostVsDepthStyle`).

    Returns:
        The Figure.
    """
    if not curves:
        raise ValueError("curves must contain at least one named curve.")

    k_arr = np.asarray(k_values)
    with styled_figure():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for i, (label, values) in enumerate(curves.items()):
            value_arr = to_numpy(np.asarray(values))
            if value_arr.shape != k_arr.shape:
                raise ValueError(
                    f"Curve '{label}' has shape {value_arr.shape}, expected "
                    f"{k_arr.shape} (one cost per entry in k_values)."
                )
            ax.plot(
                k_arr,
                value_arr,
                marker=_MARKERS[i % len(_MARKERS)],
                linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                markersize=7,
                color=series_color(label, fallback_index=i),
                label=label,
            )

        # Captured BEFORE the reference lines are drawn -- see
        # `training_curves.plot_training_curves`'s identical pattern.
        curve_ylim = (
            autoscaled_ylim_from_data(ax) if style.autoscale_to_curves else None
        )

        draw_reference_lines(ax, reference_lines, styles=reference_line_styles)

        if style.ylim is not None:
            ax.set_ylim(style.ylim)
        elif curve_ylim is not None:
            ax.set_ylim(curve_ylim)

        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        place_legend_outside(fig, ax, inside=style.legend_inside)

    return fig


@dataclass(frozen=True)
class CostComparisonStyle:
    """Labeling/scale knobs for `plot_empirical_vs_theoretical_cost`, as one
    value (its sidecar `config` maps onto these fields verbatim, so a
    re-render override like ``title="..."`` reconstructs the style).

    Attributes:
        empirical_label: Legend label for the Monte-Carlo curve.
        theoretical_label: Legend label for the closed-form curve.
        xlabel: Shared x-axis label.
        ylabel: Top (cost) panel's y-axis label.
        title: Figure title.
        log_scale: Plot the top (cost) panel's y-axis on a log scale.
    """

    empirical_label: str = "Empirical (Monte Carlo)"
    theoretical_label: str = "Theoretical (Riccati)"
    xlabel: str = "Time step"
    ylabel: str = "Cumulative average cost"
    title: str = "Empirical vs. Theoretical Expected Cost"
    log_scale: bool = False


def plot_empirical_vs_theoretical_cost(
    empirical_cost: np.ndarray,
    theoretical_cost: np.ndarray,
    *,
    style: CostComparisonStyle = CostComparisonStyle(),
) -> Figure:
    """The fundamental LQR sanity-check plot: a closed-form controller's (e.g.
    `RiccatiController`) Monte-Carlo rollout cost (`QuadraticCost.__call__` on
    real simulated trajectories) against its theoretical cost computed in
    closed form (`models.analytic.riccati.compute_theoretical_expected_cost`).
    The two curves should converge as the rollout batch size grows, since the
    theoretical curve is the batch_size -> infinity limit of the empirical
    one -- exactly what this plot is meant to make visually obvious.

    Because the two curves typically agree to within ~1% (and so are nearly
    indistinguishable when simply overlaid), a second, stacked panel plots
    their relative error (%) on a log scale, making the O(M^{-1/2}) Monte
    Carlo convergence rate visible even where the top panel shows what looks
    like a single line.

    Args:
        empirical_cost: 1D Monte-Carlo cumulative average cost, shape (horizon,).
        theoretical_cost: 1D theoretical cumulative average cost, same shape.
        style: labeling/scale knobs (see `CostComparisonStyle`).

    Returns:
        The Figure, with `fig.axes == [cost_ax, relative_error_ax]`.
    """
    empirical_cost = to_numpy(empirical_cost)
    theoretical_cost = to_numpy(theoretical_cost)
    if empirical_cost.shape != theoretical_cost.shape:
        raise ValueError(
            "empirical_cost and theoretical_cost must share the same shape; "
            f"got {empirical_cost.shape} and {theoretical_cost.shape}."
        )

    with styled_figure():
        fig, (cost_ax, error_ax) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(10, 7.5),
            gridspec_kw={"height_ratios": (3, 1.2)},
        )
        t = np.arange(1, len(empirical_cost) + 1)
        cost_ax.plot(
            t,
            empirical_cost,
            color=CATEGORICAL_PALETTE[0],
            label=style.empirical_label,
        )
        cost_ax.plot(
            t,
            theoretical_cost,
            # The semantic "closed-form ground truth" role, defined once in
            # `style.semantic.SEMANTIC_COLORS` and shared by every renderer.
            color=OPTIMUM_COLOR,
            linestyle="--",
            label=style.theoretical_label,
        )
        if style.log_scale:
            cost_ax.set_yscale("log")
        cost_ax.set_ylabel(style.ylabel)
        cost_ax.set_title(style.title)
        cost_ax.tick_params(axis="x", labelbottom=False)

        relative_error_pct = (
            100.0 * np.abs(empirical_cost - theoretical_cost) / np.abs(theoretical_cost)
        )
        error_ax.plot(t, relative_error_pct, color=AGGREGATE_COLOR)
        error_ax.set_yscale("log")
        error_ax.set_xlabel(style.xlabel)
        error_ax.set_ylabel("Relative error (%)")

        place_legend_outside(fig, cost_ax)

    return fig
