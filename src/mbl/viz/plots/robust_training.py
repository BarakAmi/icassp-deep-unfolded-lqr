"""Renderers for NB07's robust-training figures
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 6): frozen `CurveWithBand`-
shaped data in (`workbench.robust_training_sweep`), `matplotlib` Figures out.
The first renderers in this tree to use `ax.fill_between` -- every prior
plotting module reports a single point estimate per condition; NB07's
statistical-rigor mandate (Sec 5.4/8) requires a variance BAND everywhere.

`CurveWithBandLike` is deliberately a STRUCTURAL stand-in for
`workbench.robust_training_sweep.CurveWithBand`, never an import of it:
`viz.plots`' own package `__init__` is reached from deep inside
`applications`/`experiments`' own import chain (`applications.styles` ->
`viz.style.semantic` -> `viz/__init__.py` -> `viz.plots`), and
`workbench/__init__.py` eagerly imports submodules that themselves import
`applications.studies` -> `experiments` again. A concrete import of
`workbench.robust_training_sweep` here would therefore load the ENTIRE
`workbench` package as a side effect of importing this pure-rendering
module, closing a real circular-import cycle whenever `experiments` is the
first module a fresh process touches -- exactly what every notebook's
bootstrap cell does (verified by executing NB06's
`06_zero_shot_ood_generalization.ipynb`, whose `viz.plots.ood` sibling
module had the identical defect; a plain `pytest` run never surfaces it,
since some earlier-collected test module happens to import `workbench`/
`viz` first). The same `ProblemFactory`/`BatchSpec` structural-Protocol
pattern `applications.factories` already uses to avoid analogous tight
coupling applies here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..style import place_legend_outside, series_color, styled_figure


@runtime_checkable
class CurveWithBandLike(Protocol):
    """Structural counterpart of `workbench.robust_training_sweep
    .CurveWithBand` (see module docstring for why this is a Protocol, never
    a direct import)."""

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


@dataclass(frozen=True)
class BandCurveStyle:
    """Presentation knobs for `plot_band_curves` (PLR0913 budget, the same
    pattern `TrainingCurveStyle`/`CostComparisonStyle` already use).

    Attributes:
        xlabel: Shared x-axis label.
        ylabel: Shared y-axis label.
        title: Figure title.
        log_x: Log-scale x-axis (the sample-complexity figure's convention).
        log_y: Log-scale y-axis.
        legend_inside: See `viz.style.theme.place_legend_outside`.
        band_label_suffix: Appended to the legend's band-explanation note
            (NB07 plan Sec 3.4/5.4: a plant axis mixes perturbation-draw and
            Monte-Carlo variability; a batch-spec axis's band is pure
            Monte-Carlo -- the two must be labeled differently, never
            conflated in one unlabeled shaded region).
    """

    xlabel: str = "Level"
    ylabel: str = "Cost"
    title: str = ""
    log_x: bool = False
    log_y: bool = False
    legend_inside: bool = False
    band_label_suffix: str = "(median, IQR)"


def plot_band_curves(
    bands: Mapping[str, CurveWithBandLike],
    *,
    reference_lines: Mapping[str, float] | None = None,
    style: BandCurveStyle = BandCurveStyle(),
) -> Figure:
    """One median line + IQR band per contender, against a shared x-axis
    (level, epoch, or trajectory count) -- the generic workhorse behind
    every NB07 band figure (learning dynamics, price of robustness, sample
    complexity, spectral isotropy).

    Args:
        bands: label -> `CurveWithBand` (e.g. `RobustTrainingResult
            .cost_band`/`.learning_curve_band`/`.isotropy_band`).
        reference_lines: Optional label -> constant value, drawn as a
            dashed horizontal line (e.g. the nominally-trained baseline).
        style: Labeling/scale knobs.

    Returns:
        The Figure.

    Raises:
        ValueError: If `bands` is empty.
    """
    if not bands:
        raise ValueError("bands must contain at least one named curve.")

    # The X (noise-family) axis's levels are `NoiseFamily` members, not
    # numbers -- plotted against integer positions with the values
    # themselves as tick labels, exactly like every other (numeric) axis
    # otherwise, rather than crashing on `float(NoiseFamily.GAUSSIAN)`.
    all_x_values = [value for band in bands.values() for value in band.x]
    categorical = bool(all_x_values) and not all(
        isinstance(value, (int, float, np.integer, np.floating))
        for value in all_x_values
    )
    categories: list[Any] = []
    if categorical:
        for value in all_x_values:
            if value not in categories:
                categories.append(value)

    with styled_figure():
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for i, (label, band) in enumerate(bands.items()):
            if not band.x:
                continue
            x = (
                np.asarray([categories.index(value) for value in band.x], dtype=float)
                if categorical
                else np.asarray(band.x, dtype=float)
            )
            color = series_color(label, fallback_index=i)
            ax.plot(
                x,
                band.median,
                marker="o",
                markersize=5,
                color=color,
                label=f"{label} {style.band_label_suffix}",
            )
            ax.fill_between(
                x, band.q25, band.q75, color=color, alpha=_BAND_ALPHA, linewidth=0
            )

        if reference_lines:
            for i, (label, value) in enumerate(reference_lines.items()):
                ax.axhline(
                    value,
                    linestyle=":",
                    linewidth=1.5,
                    color=series_color(label, fallback_index=i),
                    label=f"{label} (reference)",
                )

        if categorical:
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([str(value) for value in categories])
        elif style.log_x:
            ax.set_xscale("log")
        if style.log_y:
            ax.set_yscale("log")
        ax.set_xlabel(style.xlabel)
        ax.set_ylabel(style.ylabel)
        ax.set_title(style.title)
        place_legend_outside(
            fig, ax, rect=(0.0, 0.0, 0.78, 1.0), inside=style.legend_inside
        )

    return fig


@dataclass(frozen=True)
class LearningDynamicsStyle:
    """Presentation knobs for `plot_learning_dynamics_with_constraint`.

    Attributes:
        title: Figure title.
        cost_ylabel: Top panel's y-axis label.
        constraint_ylabel: Bottom panel's y-axis label.
        log_y: Log-scale the cost panel's y-axis.
        legend_inside: See `viz.style.theme.place_legend_outside`.
    """

    title: str = "Learning Dynamics Under Uncertainty"
    cost_ylabel: str = "Training loss"
    constraint_ylabel: str = "Saturation rate"
    log_y: bool = False
    legend_inside: bool = False


def plot_learning_dynamics_with_constraint(
    cost_bands: Mapping[str, CurveWithBandLike],
    constraint_bands: Mapping[str, CurveWithBandLike],
    *,
    style: LearningDynamicsStyle = LearningDynamicsStyle(),
) -> Figure:
    """Two vertically stacked panels on a shared (epoch) x-axis: per-epoch
    training-loss median+IQR band on top, per-epoch saturation-rate band
    below (NB07 plan Sec 4.6/6.2: "no cost figure ships without its
    constraint panel").

    Args:
        cost_bands: label -> `CurveWithBand` (e.g.
            `RobustTrainingResult.learning_curve_band` per contender at one
            level).
        constraint_bands: label -> `CurveWithBand`, same x-axis as
            `cost_bands` (e.g. a per-epoch saturation-rate band).
        style: Labeling/scale knobs.

    Returns:
        The Figure.

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
        for i, (label, band) in enumerate(cost_bands.items()):
            if not band.x:
                continue
            x = np.asarray(band.x, dtype=float)
            color = series_color(label, fallback_index=i)
            cost_ax.plot(x, band.median, color=color, label=label)
            cost_ax.fill_between(
                x, band.q25, band.q75, color=color, alpha=_BAND_ALPHA, linewidth=0
            )

        for i, (label, band) in enumerate(constraint_bands.items()):
            if not band.x:
                continue
            x = np.asarray(band.x, dtype=float)
            color = series_color(label, fallback_index=i)
            constraint_ax.plot(x, band.median, color=color, label=label)
            constraint_ax.fill_between(
                x, band.q25, band.q75, color=color, alpha=_BAND_ALPHA, linewidth=0
            )

        if style.log_y:
            cost_ax.set_yscale("log")
        cost_ax.set_ylabel(style.cost_ylabel)
        cost_ax.set_title(style.title)
        place_legend_outside(
            fig, cost_ax, rect=(0.0, 0.0, 0.78, 1.0), inside=style.legend_inside
        )

        constraint_ax.set_ylim(-0.05, 1.05)
        constraint_ax.set_xlabel("Epoch")
        constraint_ax.set_ylabel(style.constraint_ylabel)

    return fig
