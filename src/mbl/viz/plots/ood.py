"""Renderers for NB06's zero-shot-OOD figures
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 5): frozen
`workbench.ood_sweep.CurveWithBand`-shaped/array data in, `matplotlib` Figures
out.

`plot_ood_cost_and_constraint` is the workhorse (Sec 5.1/5.2): two vertically
stacked panels sharing one x-axis -- a suboptimality-gap-or-cost median+IQR
band on top, the saturation-rate band directly below it. Per the plan's
corrected constraint mandate, **no cost figure ships without its constraint
panel** -- this module has no single-panel cost-only renderer at all, so
that law cannot be forgotten at a call site.

`CurveWithBandLike` is deliberately a STRUCTURAL stand-in for
`workbench.ood_sweep.CurveWithBand`, never an import of it: `viz.plots`'
own package `__init__` is reached from deep inside `applications`/
`experiments`' own import chain (`applications.styles` ->
`viz.style.semantic` -> `viz/__init__.py` -> `viz.plots`), and
`workbench/__init__.py` eagerly imports submodules that themselves import
`applications.studies` -> `experiments` again. A concrete import of
`workbench.ood_sweep` here would therefore load the ENTIRE `workbench`
package as a side effect of importing this pure-rendering module, closing a
real circular-import cycle whenever `experiments` is the first module a
fresh process touches -- exactly what every notebook's bootstrap cell does
(verified by executing `06_zero_shot_ood_generalization.ipynb`; a plain
`pytest` run never surfaces it, since some earlier-collected test module
happens to import `workbench`/`viz` first). The same `ProblemFactory`/
`BatchSpec` structural-Protocol pattern `applications.factories` already
uses to avoid analogous tight coupling applies here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..style import place_legend_outside, series_color, series_linestyle, styled_figure


@runtime_checkable
class CurveWithBandLike(Protocol):
    """Structural counterpart of `workbench.ood_sweep.CurveWithBand` (see
    module docstring for why this is a Protocol, never a direct import)."""

    @property
    def x(self) -> tuple: ...
    @property
    def median(self) -> tuple: ...
    @property
    def q25(self) -> tuple: ...
    @property
    def q75(self) -> tuple: ...


#: Shared band transparency: low enough that overlapping bands (several
#: contenders on one axis) stay individually legible.
_BAND_ALPHA = 0.2

#: Saturation rate lives in [0, 1] by construction; a fixed y-limit with a
#: small margin makes every saturation panel visually comparable across
#: figures instead of auto-scaling to whatever sliver of [0, 1] is occupied.
_SATURATION_YLIM = (-0.05, 1.05)


def _plot_bands(ax: plt.Axes, bands: Mapping[str, CurveWithBandLike]) -> None:
    """Shared band-plus-line drawing loop -- both panels of
    `plot_ood_cost_and_constraint` and every panel of `plot_ood_depth_ablation`
    use it. Color AND linestyle are both resolved by label
    (`applications.styles.NB06_CONTENDER_COLORS`/`NB06_DASHED_CONTENDERS`,
    NB06 plan Sec 5.0): every registered NB06 contender therefore renders in
    its one fixed color/dash pattern regardless of this mapping's iteration
    order, closing the index-coupling defect an earlier, color-only fix
    left half-open (dashed=iterative was registered but never actually
    drawn dashed)."""
    for i, (label, band) in enumerate(bands.items()):
        if not band.x:
            continue
        x = np.asarray(band.x, dtype=float)
        color = series_color(label, fallback_index=i)
        linestyle = series_linestyle(label)
        ax.plot(
            x,
            band.median,
            marker="o",
            markersize=4,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        ax.fill_between(
            x, band.q25, band.q75, color=color, alpha=_BAND_ALPHA, linewidth=0
        )


@dataclass(frozen=True)
class OODCostConstraintStyle:
    """Presentation knobs for `plot_ood_cost_and_constraint` (PLR0913
    budget, the same pattern `BandCurveStyle`/`TrainingCurveStyle` use).

    Attributes:
        title: Figure title.
        xlabel: Shared x-axis label (the perturbation level).
        cost_ylabel: Top panel's y-axis label.
        log_x: Log-scale the shared x-axis.
        log_y_cost: Log-scale the cost/gap panel's y-axis.
        legend_inside: See `viz.style.theme.place_legend_outside`.
        band_note: Appended to the figure via a caption-style note,
            distinguishing pure Monte-Carlo variability (Axes H/S/X) from
            variability that also mixes perturbation-draw randomness (Axis
            D) -- the two must never be conflated in one unlabeled band
            (NB06 plan Sec 3.4).
    """

    title: str = ""
    xlabel: str = "Perturbation level"
    cost_ylabel: str = "Suboptimality gap"
    log_x: bool = False
    log_y_cost: bool = False
    legend_inside: bool = False
    band_note: str = "(median, IQR across seeds)"


def plot_ood_cost_and_constraint(
    cost_bands: Mapping[str, CurveWithBandLike],
    saturation_bands: Mapping[str, CurveWithBandLike],
    *,
    reference_lines: Mapping[str, float] | None = None,
    style: OODCostConstraintStyle = OODCostConstraintStyle(),
) -> Figure:
    """The paired OOD figure (NB06 plan Sec 5.1/5.2): median-cost-or-gap band
    on top, saturation-rate band directly below, both against the SAME
    perturbation-level x-axis.

    Args:
        cost_bands: label -> `CurveWithBand` (e.g.
            `OODSweepResult.cost_band`/`.suboptimality_gap_band`).
        saturation_bands: label -> `CurveWithBand`, same x-axis as
            `cost_bands` (`OODSweepResult.saturation_band`).
        reference_lines: Optional label -> constant value drawn as a dashed
            horizontal line on the cost panel (e.g. a nominal reference).
        style: Labeling/scale knobs.

    Returns:
        The Figure, two axes (`fig.axes[0]` cost/gap, `fig.axes[1]`
        saturation).

    Raises:
        ValueError: If `cost_bands` is empty.
    """
    if not cost_bands:
        raise ValueError("cost_bands must contain at least one named curve.")

    with styled_figure():
        fig, (cost_ax, constraint_ax) = plt.subplots(
            2,
            1,
            figsize=(11, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1]},
        )
        _plot_bands(cost_ax, cost_bands)
        _plot_bands(constraint_ax, saturation_bands)

        if reference_lines:
            for i, (label, value) in enumerate(reference_lines.items()):
                cost_ax.axhline(
                    value,
                    linestyle=":",
                    linewidth=1.5,
                    color=series_color(label, fallback_index=len(cost_bands) + i),
                    label=f"{label} (reference)",
                )

        if style.log_x:
            cost_ax.set_xscale("log")
            constraint_ax.set_xscale("log")
        if style.log_y_cost:
            cost_ax.set_yscale("log")
        cost_ax.set_ylabel(f"{style.cost_ylabel} {style.band_note}")
        cost_ax.set_title(style.title)
        place_legend_outside(
            fig, cost_ax, rect=(0.0, 0.0, 0.78, 1.0), inside=style.legend_inside
        )

        constraint_ax.set_ylim(*_SATURATION_YLIM)
        constraint_ax.set_xlabel(style.xlabel)
        constraint_ax.set_ylabel("Saturation rate")

    return fig


@dataclass(frozen=True)
class DepthAblationStyle:
    """Presentation knobs for `plot_ood_depth_ablation`.

    Attributes:
        title: Figure title.
        xlabel: Shared x-axis label.
        ylabel: Shared y-axis label.
        log_x: Log-scale the shared x-axis.
        log_y: Log-scale the shared y-axis.
    """

    title: str = "Over-Thinking: OOD Cost vs. Unfolding Depth"
    xlabel: str = "Perturbation level"
    ylabel: str = "Suboptimality gap"
    log_x: bool = False
    log_y: bool = False


def plot_ood_depth_ablation(
    bands_by_contender: Mapping[str, Mapping[int, CurveWithBandLike]],
    *,
    style: DepthAblationStyle = DepthAblationStyle(),
) -> Figure:
    """The "over-thinking" ablation figure (NB06 plan Sec 5.4): one subplot
    per unfolded contender, each overlaying one band per swept unfolding
    depth ``K`` -- a curve crossing below another at high severity is the
    "shallow generalizes better under heavy noise" signature; a null result
    (no crossing) is exactly as readable.

    Args:
        bands_by_contender: contender label -> {K -> `CurveWithBand`} (e.g.
            built from `run_ood_depth_ablation`'s ``dict[int,
            OODSweepResult]`` by calling ``.cost_band(label)``/
            ``.suboptimality_gap_band(label)`` per depth).
        style: Labeling/scale knobs.

    Returns:
        The Figure, one axes per entry in `bands_by_contender`.

    Raises:
        ValueError: If `bands_by_contender` is empty.
    """
    if not bands_by_contender:
        raise ValueError("bands_by_contender must contain at least one contender.")

    with styled_figure():
        fig, axes = plt.subplots(
            1,
            len(bands_by_contender),
            figsize=(6.5 * len(bands_by_contender), 5.5),
            sharey=True,
            squeeze=False,
        )
        for panel_index, (label, depth_bands) in enumerate(bands_by_contender.items()):
            ax = axes[0, panel_index]
            depth_curves = {
                f"K={depth}": band for depth, band in sorted(depth_bands.items())
            }
            _plot_bands(ax, depth_curves)
            if style.log_x:
                ax.set_xscale("log")
            if style.log_y:
                ax.set_yscale("log")
            ax.set_title(label)
            ax.set_xlabel(style.xlabel)
            ax.legend(loc="best", fontsize=9)

        axes[0, 0].set_ylabel(style.ylabel)
        fig.suptitle(style.title)

    return fig


@dataclass(frozen=True)
class PhasePortraitStyle:
    """Presentation knobs for `plot_ood_phase_portrait`.

    Attributes:
        title: Figure title.
        component_labels: Labels for the two projected axes.
    """

    title: str = "OOD Trajectory Snapshot"
    component_labels: tuple[str, str] = (
        "Principal component 1",
        "Principal component 2",
    )


def plot_ood_phase_portrait(
    trajectories: Mapping[str, np.ndarray],
    *,
    style: PhasePortraitStyle = PhasePortraitStyle(),
) -> Figure:
    """The single static OOD trajectory snapshot (NB06 plan Sec 5.5): every
    named trajectory's states projected onto the SAME two leading principal
    directions (fit jointly across all of them, so the projection is one
    shared basis, not a per-contender re-projection that would make the
    panels incomparable), plus a state-norm-vs-time inset where divergence
    reads unambiguously on a log scale.

    Args:
        trajectories: label -> states, shape ``(T+1, n)`` (a single
            trajectory per label -- one batch element of a captured OOD
            rollout, e.g. `experiments.zero_shot.evaluate_under_shift`'s
            ``capture_trajectories=True`` output, sliced to one sample).
        style: Labeling knobs.

    Returns:
        The Figure: `fig.axes[0]` the 2-D phase portrait, `fig.axes[1]` the
        log-scale norm-vs-time inset.

    Raises:
        ValueError: If `trajectories` is empty.
    """
    if not trajectories:
        raise ValueError("trajectories must contain at least one named trajectory.")

    stacked = np.concatenate(
        [np.asarray(states) for states in trajectories.values()], axis=0
    )
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    # SVD of the centered, pooled states: the top two right singular vectors
    # are the two leading principal directions, shared across every
    # trajectory (fit ONCE, jointly) -- the same basis every panel projects
    # onto, so the portraits are directly comparable.
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:2].T  # (n, 2)

    with styled_figure():
        fig, (portrait_ax, norm_ax) = plt.subplots(1, 2, figsize=(13, 5.5))
        for i, (label, states) in enumerate(trajectories.items()):
            states = np.asarray(states)
            color = series_color(label, fallback_index=i)
            projected = (states - stacked.mean(axis=0, keepdims=True)) @ basis
            portrait_ax.plot(
                projected[:, 0],
                projected[:, 1],
                color=color,
                label=label,
                linewidth=1.8,
            )
            portrait_ax.scatter(
                projected[0, 0],
                projected[0, 1],
                color=color,
                marker="o",
                s=40,
                zorder=3,
            )
            norms = np.linalg.norm(states, axis=-1)
            norm_ax.plot(np.arange(len(norms)), norms, color=color, label=label)

        portrait_ax.set_xlabel(style.component_labels[0])
        portrait_ax.set_ylabel(style.component_labels[1])
        portrait_ax.legend(loc="best", fontsize=9)

        norm_ax.set_yscale("log")
        norm_ax.set_xlabel("Time step")
        norm_ax.set_ylabel(r"$\|x_t\|$ (log scale)")

        fig.suptitle(style.title)

    return fig


@dataclass(frozen=True)
class LabeledScatterStyle:
    """Shared presentation knobs for the three NB06 Sec 13 scatter/heatmap
    renderers below (PLR0913 budget, the `BandCurveStyle`-style pattern).

    Attributes:
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        log_x: Log-scale the x-axis.
        log_y: Log-scale the y-axis.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    log_x: bool = False
    log_y: bool = False


def _labeled_scatter(
    ax: plt.Axes, x_by_label: Mapping[str, float], y_by_label: Mapping[str, float]
) -> None:
    """Shared one-point-per-contender scatter drawing loop, sharing colors
    with every band-based NB06 figure."""
    for i, label in enumerate(x_by_label):
        color = series_color(label, fallback_index=i)
        ax.scatter(x_by_label[label], y_by_label[label], color=color, s=64, label=label)


def plot_ood_cost_vs_compute(
    latency_us_by_label: Mapping[str, float],
    degradation_by_label: Mapping[str, float],
    *,
    style: LabeledScatterStyle = LabeledScatterStyle(
        xlabel="Online inference latency (microseconds, log scale)",
        ylabel="Relative cost degradation at the severest swept level",
        log_x=True,
    ),
) -> Figure:
    """The cost-vs-compute scatter (NB06 plan Sec 13.5): does a contender's
    OOD robustness cost extra online compute, or is it free? One point per
    contender; the Pareto front (lower-left) is the honest summary of the
    whole notebook's architecture comparison.

    Args:
        latency_us_by_label: label -> per-step online inference latency, in
            microseconds (e.g.
            `workbench.benchmarks.OnlineInferenceRecord.online_latency_us`).
        degradation_by_label: label -> `workbench.ood_sweep
            .build_ood_degradation_table`'s ``relative_degradation`` at
            whichever axis/level the caller considers "severest" -- same
            keys as `latency_us_by_label`.
        style: Labeling/scale knobs.

    Returns:
        The Figure.

    Raises:
        ValueError: If either mapping is empty.
    """
    if not latency_us_by_label or not degradation_by_label:
        raise ValueError(
            "latency_us_by_label and degradation_by_label must both be non-empty."
        )
    with styled_figure():
        fig, ax = plt.subplots(figsize=(7, 6))
        _labeled_scatter(ax, latency_us_by_label, degradation_by_label)
        if style.log_x:
            ax.set_xscale("log")
        if style.log_y:
            ax.set_yscale("log")
        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        place_legend_outside(fig, ax, rect=(0.0, 0.0, 0.78, 1.0))
    return fig


def plot_ood_mandate_scatter(
    saturation_by_label: Mapping[str, float],
    degradation_by_label: Mapping[str, float],
    *,
    style: LabeledScatterStyle = LabeledScatterStyle(
        xlabel="Nominal (in-distribution) saturation rate",
        ylabel="Relative cost degradation at the severest swept level",
    ),
) -> Figure:
    """The mandate scatter (NB06 plan Sec 13.9): the blueprint's own
    question, in one panel -- do contenders that sit deep in the interior
    at the nominal point (low nominal saturation) degrade LESS under OOD
    shift than ones already leaning on the constraint? A positive
    correlation (upper-right populated, lower-left populated) confirms the
    mandate's hypothesis; a scatter with no visible trend is an equally
    legible null.

    Args:
        saturation_by_label: label -> nominal (in-distribution) saturation
            rate, in ``[0, 1]``.
        degradation_by_label: label -> relative cost degradation at the
            severest swept level (same keys as `saturation_by_label`).
        style: Labeling/scale knobs.

    Returns:
        The Figure.

    Raises:
        ValueError: If either mapping is empty.
    """
    if not saturation_by_label or not degradation_by_label:
        raise ValueError(
            "saturation_by_label and degradation_by_label must both be non-empty."
        )
    with styled_figure():
        fig, ax = plt.subplots(figsize=(7, 6))
        _labeled_scatter(ax, saturation_by_label, degradation_by_label)
        if style.log_x:
            ax.set_xscale("log")
        if style.log_y:
            ax.set_yscale("log")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        place_legend_outside(fig, ax, rect=(0.0, 0.0, 0.78, 1.0))
    return fig


@dataclass(frozen=True)
class InteractionHeatmapStyle:
    """Presentation knobs for `plot_ood_interaction_heatmap`.

    Attributes:
        title: Figure title.
        xlabel: X-axis (bound-multiplier) label.
        ylabel: Y-axis (scale-multiplier) label.
        colorbar_label: Colorbar label.
        cmap: Matplotlib colormap name.
    """

    title: str = "Scale x Bound Interaction"
    xlabel: str = "Control-bound multiplier"
    ylabel: str = "Noise/initial-state scale multiplier"
    colorbar_label: str = "Median cost"
    cmap: str = "viridis"


def plot_ood_interaction_heatmap(
    scale_levels: tuple[float, ...],
    bound_levels: tuple[float, ...],
    grid: np.ndarray,
    *,
    style: InteractionHeatmapStyle = InteractionHeatmapStyle(),
) -> Figure:
    """The scale x bound interaction heatmap (NB06 plan Sec 13.7): does a
    noisier environment and a de-rated actuator degrade cost
    super-additively (a curved, non-separable pattern) or independently (a
    heatmap that factors into a row effect times a column effect)? Cell
    values are read from `workbench.ood_sweep.OODSweepResult
    .interaction_grid`, never recomputed here (pure rendering).

    Args:
        scale_levels: Row coordinates (ascending), one per grid row.
        bound_levels: Column coordinates (ascending), one per grid column.
        grid: Shape ``(len(scale_levels), len(bound_levels))``; ``NaN``
            cells render blank (masked).
        style: Labeling/colormap knobs.

    Returns:
        The Figure.

    Raises:
        ValueError: If `grid`'s shape does not match
            ``(len(scale_levels), len(bound_levels))``.
    """
    expected_shape = (len(scale_levels), len(bound_levels))
    if grid.shape != expected_shape:
        raise ValueError(
            f"grid.shape {grid.shape} does not match "
            f"(len(scale_levels), len(bound_levels)) = {expected_shape}."
        )
    masked = np.ma.masked_invalid(grid)
    with styled_figure():
        fig, ax = plt.subplots(figsize=(7, 6))
        mesh = ax.pcolormesh(
            np.arange(len(bound_levels) + 1),
            np.arange(len(scale_levels) + 1),
            masked,
            cmap=style.cmap,
            shading="flat",
        )
        ax.set_xticks(np.arange(len(bound_levels)) + 0.5)
        ax.set_xticklabels([f"{b:g}" for b in bound_levels])
        ax.set_yticks(np.arange(len(scale_levels)) + 0.5)
        ax.set_yticklabels([f"{s:g}" for s in scale_levels])
        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        fig.colorbar(mesh, ax=ax, label=style.colorbar_label)
    return fig
