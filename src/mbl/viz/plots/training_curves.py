"""Training/optimization loss curves across epochs -- accepts plain named
sequences (e.g. accumulated from successive `RunContext.metrics` snapshots or
loaded back from a persisted artifact), never the engine's `RunContext`
itself."""

from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ...core.utils import to_numpy
from ..style import (
    ReferenceLineStyle,
    autoscaled_ylim_from_data,
    draw_reference_lines,
    series_color,
    place_legend_outside,
    styled_figure,
)

#: Triangles/squares/diamonds first, matching the flagship cost_analysis
#: plots' marker convention so the same visual language reads consistently
#: across every "publication-ready" figure in the dashboard.
_MARKERS = ("^", "s", "D", "v", "o", "P")


@dataclass(frozen=True)
class TrainingCurveStyle:
    """Presentation knobs for `plot_training_curves` beyond its data inputs
    (PLR0913: keeps the renderer under the 6-argument cap, the same pattern
    `CostComparisonStyle`/`SignalStyle`/`DepthTrajectoryStyle` already use).

    Attributes:
        xlabel: Shared x-axis label.
        ylabel: Shared y-axis label.
        title: Figure title.
        log_scale: Plot the y-axis on a log scale (common for loss curves).
        show_markers: Draw per-curve markers; ``False`` for a plain line --
            accessible toggle for a figure overlaying many curves (a marker
            per line reads as clutter once curve count grows) on a long
            epoch axis.
        linestyle: Shared line style for every curve (matplotlib format,
            e.g. ``"-"``, ``"--"``, ``":"``).
        linewidth: Shared line width; ``None`` keeps the themed default
            (`viz.style.theme.PLOT_STYLE`'s ``lines.linewidth``).
        legend_inside: Draw the legend inside the axes instead of the
            default external placement (see `viz.style.theme
            .place_legend_outside`).
        ylim: explicit ``(low, high)`` y-axis override. ``None`` (default)
            keeps matplotlib's own autoscale over EVERY artist, curves and
            reference lines alike.
        autoscale_to_curves: when ``True``, set the y-limits from the
            CURVE data only (`ax.dataLim` before any reference line is
            drawn) -- a reference line far outside the curves' own range
            (e.g. a loose theoretical floor) no longer dictates the axis; it
            stays fully reported via its legend entry's folded-in value
            (`draw_reference_lines`), just not necessarily visible as a
            line (NB04 reference-bounds plan Sec 2.4). Ignored when `ylim`
            is also set (`ylim` wins).
    """

    xlabel: str = "Batch"
    ylabel: str = "Loss"
    title: str = "Training Curves"
    log_scale: bool = False
    show_markers: bool = True
    linestyle: str = "-"
    linewidth: float | None = None
    legend_inside: bool = False
    ylim: tuple[float, float] | None = None
    autoscale_to_curves: bool = False


def plot_training_curves(
    curves: Mapping[str, np.ndarray],
    *,
    reference_lines: Mapping[str, float] | None = None,
    reference_line_styles: Mapping[str, ReferenceLineStyle] | None = None,
    style: TrainingCurveStyle = TrainingCurveStyle(),
) -> Figure:
    """Plot one or more named loss/cost curves against epoch index, with
    optional horizontal reference lines (e.g. a theoretical or simulated
    optimum to compare convergence against).

    Each curve's legend entry reports its OWN final (last-epoch) value --
    e.g. ``"$J=4$ (final=0.0231)"`` -- so a reader sees each curve's
    converged cost at a glance without hunting for the corresponding line's
    right-hand endpoint (NB03 overhaul Phase 4a: valuable specifically when
    several depth-indexed curves share one axis, mirroring how the flat
    Riccati baselines already report their value via `draw_reference_lines`).

    Args:
        curves: label -> 1D sequence of per-epoch values.
        reference_lines: label -> constant value, drawn as a dashed horizontal
            line (e.g. {"Theoretical Optimal": 1.23}).
        reference_line_styles: optional label -> `ReferenceLineStyle`
            overrides, so removing an entry from `reference_lines` never
            silently reassigns the remaining lines' colors/linestyles (see
            `draw_reference_lines`).
        style: labeling/scale/marker/line/axis knobs (see
            `TrainingCurveStyle`).

    Returns:
        The Figure.
    """
    if not curves:
        raise ValueError("curves must contain at least one named curve.")

    with styled_figure():
        fig, ax = plt.subplots(figsize=(12, 6.5))
        for i, (label, values) in enumerate(curves.items()):
            values = to_numpy(values)
            if values.ndim != 1:
                raise ValueError(
                    f"Curve '{label}' must be 1D (one value per epoch), "
                    f"got shape {values.shape}."
                )
            n_epochs = len(values)
            final_value = float(values[-1])
            ax.plot(
                np.arange(1, n_epochs + 1),
                values,
                marker=_MARKERS[i % len(_MARKERS)] if style.show_markers else None,
                # Sparse markers (~20 across the whole curve) keep long runs
                # (hundreds of epochs) readable instead of a solid marker
                # blob, while still tracking the convergence shape clearly.
                markevery=max(1, n_epochs // 20),
                markersize=6,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                color=series_color(label, fallback_index=i),
                label=f"{label} (final={final_value:.4g})",
            )

        # Captured BEFORE the reference lines are drawn: `ax.dataLim` only
        # reflects artists added so far, so this is exactly "the curves'
        # own range" (style.autoscale_to_curves' contract) regardless of
        # how many reference lines follow.
        curve_ylim = (
            autoscaled_ylim_from_data(ax) if style.autoscale_to_curves else None
        )

        draw_reference_lines(ax, reference_lines, styles=reference_line_styles)

        if style.log_scale:
            ax.set_yscale("log")

        if style.ylim is not None:
            ax.set_ylim(style.ylim)
        elif curve_ylim is not None:
            ax.set_ylim(curve_ylim)

        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        place_legend_outside(
            fig, ax, rect=(0.0, 0.0, 0.78, 1.0), inside=style.legend_inside
        )

    return fig
