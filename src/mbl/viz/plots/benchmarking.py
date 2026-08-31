"""Asset 7: the empirical counterpart to a notebook's asymptotic-complexity
markdown, comparing this project's closed-form and iterative LQR solvers
across the **offline synthesis vs. online inference** split control theory
draws between pre-computing a policy and executing it -- a 2x2 grid of
barplots (offline/online columns, time/memory rows) sharing the package's
own IEEE-style `PLOT_STYLE` (`style.py`) rather than a bespoke look.

Memory is a peak watermark, not an additive interval, so unlike wall-clock
time it is never stacked *across phases*: offline and online memory each get
their own barplot, phase-split exactly like the time panels above them.
Within a single phase's panel, though, each bar IS a two-segment stack (Micro
-Prompt 5d): a closed-form NumPy/CPU solver (e.g. Riccati) and a PyTorch
CPU/GPU solver (e.g. gradient descent) hold their bytes in physically
different memory subsystems, so a single flat bar-height would either miss
one solver's footprint entirely or silently conflate the two. Each bar is
instead the NumPy/CPU layer (`MemoryBreakdown.numpy_mb`) stacked under the
PyTorch layer (`MemoryBreakdown.torch_mb`), so both are visible even when one
of them is near zero (e.g. Riccati's ~0 PyTorch layer, or GD's ~0 NumPy layer
online).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..style import CATEGORICAL_PALETTE, styled_figure

#: Cycled per bar (by position, not identity) so the method comparison this
#: asset targets stays visually distinguishable even in grayscale print --
#: the same "color is not the only channel" rationale as `style
#: .model_linestyle`. Shared across all four panels so a given method (e.g.
#: "Riccati") always reads as the same color/hatch, regardless of which
#: phase/metric the panel shows.
_BAR_HATCHES = ("//", "xx", "..", "oo", "\\\\")

#: Fractional opacity of a bar's PyTorch/CUDA (upper) segment, vs. the fully
#: opaque (``1.0``) NumPy/CPU (lower) segment -- the "lighter shade" half of
#: the memory panels' two-channel (shade + hatch density) encoding of layer
#: identity, independent of the per-method color/hatch encoding above. Its
#: hatch counterpart -- each `_BAR_HATCHES` entry's first character alone,
#: e.g. ``"//"`` -> ``"/"``, one density step sparser -- is derived inline in
#: `_plot_stratified_memory_panel` rather than kept as a second module-level
#: tuple, since it must always stay a pure function of whichever `hatches`
#: that panel was actually called with.
_MEMORY_LAYER_ALPHA = 0.55

#: Fractional headroom (of the tallest finite bar) reserved above every bar
#: for its value annotation, so the text never collides with the axes' own
#: top spine.
_ANNOTATION_HEADROOM_FRACTION = 0.03

#: Fractional padding (of the tallest bar) added to each panel's own y-limit
#: -- well beyond `_ANNOTATION_HEADROOM_FRACTION` -- so the annotation text
#: itself (which `ax.bar`'s autoscale never accounts for, since it sizes the
#: axes off the bars' data alone) always has clear air below `ax.set_title`,
#: rather than colliding with it whenever a panel's bars nearly fill their
#: own axes (e.g. offline-memory panels, where every bar is a few KB).
_YLIM_TOP_PADDING_FRACTION = 0.22


@dataclass(frozen=True)
class MemoryBreakdown:
    """One phase's memory footprint, stratified by the heterogeneous
    device/library layer that actually holds the bytes -- this module's own
    MB-scaled counterpart of `core.profiling.StratifiedMemoryReport` (which
    works in bytes; a `MemoryBreakdown` is what the memory panels below
    actually render, one two-segment stacked bar per entry).

    Attributes:
        numpy_mb: NumPy/CPU-resident footprint, in MB, or `None`/`nan` if
            unmeasured (renders as an explicit "N/A" bar; see
            `_annotate_bars`).
        torch_mb: PyTorch-native (CPU + CUDA) allocation footprint, in MB, or
            `None`/`nan` if unmeasured.
    """

    numpy_mb: float | None
    torch_mb: float | None

    @property
    def total_mb(self) -> float | None:
        """`numpy_mb + torch_mb`, or `None` if either layer is missing or
        non-finite -- an entirely unmeasured phase must never silently
        render as a `0`-height bar segment indistinguishable from a phase
        that genuinely used no memory in that layer.
        """
        if (
            self.numpy_mb is None
            or self.torch_mb is None
            or not np.isfinite(self.numpy_mb)
            or not np.isfinite(self.torch_mb)
        ):
            return None
        return self.numpy_mb + self.torch_mb


@dataclass(frozen=True)
class BenchmarkRecord:
    """One benchmarked method's complete phase-split measurement -- the
    frozen renderer input (REFACTOR_PLAN v3, T4.b) that replaces the five
    parallel sequences `plot_computational_benchmarks` previously took (N10),
    so a label can never silently drift out of alignment with its own
    measurements.

    Attributes:
        label: the method's display name (e.g. ``"Riccati"``).
        offline_s: one-time pre-computation seconds (e.g. the backward
            Riccati recursion).
        online_s: real-time control-generation seconds (e.g. the ``u=-Kx``
            rollout, or the GD macro-iteration loop).
        offline_memory: the offline synthesis phase's stratified footprint.
        online_memory: the online inference phase's stratified footprint.
    """

    label: str
    offline_s: float
    online_s: float
    offline_memory: MemoryBreakdown
    online_memory: MemoryBreakdown


def _annotate_bars(
    ax: Axes, bars: BarContainer, values: Sequence[float | None], fmt: str
) -> None:
    """Print each `fmt`-formatted entry of `values` centered just above its
    bar in `bars` -- ``"N/A"`` for a non-finite (``None``/``nan``) value
    (e.g. `core.profiling.current_peak_ram_bytes` returns ``None`` on a
    platform without the ``resource`` module) rather than a misleading
    ``0``-height bar with no explanation.
    """
    finite = [v for v in values if v is not None and np.isfinite(v)]
    headroom = _ANNOTATION_HEADROOM_FRACTION * (max(finite) if finite else 1.0)
    for bar, value in zip(bars, values):
        is_finite = value is not None and np.isfinite(value)
        label = fmt.format(value) if is_finite else "N/A"
        height = value if is_finite and value is not None else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + headroom,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )


@dataclass(frozen=True)
class _PanelStyle:
    """The shared per-method identity styling plus one panel's own labeling:
    `colors`/`hatches` are common to all four panels of the 2x2 grid (a
    method keeps one color family everywhere), while `ylabel`/`title`/`fmt`
    name the individual panel."""

    colors: Sequence
    hatches: Sequence[str]
    ylabel: str
    title: str
    fmt: str = ""


def _plot_metric_panel(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float | None],
    style: _PanelStyle,
) -> None:
    """Render one phase/metric panel of the 2x2 grid: a single bar per
    `labels` entry, colored/hatched by method identity (`colors`/`hatches`,
    shared across all four panels), annotated with its exact value.
    """
    x = np.arange(len(labels))
    plotted = [0.0 if v is None or not np.isfinite(v) else float(v) for v in values]
    bars = ax.bar(
        x,
        plotted,
        color=style.colors,
        hatch=style.hatches,
        edgecolor="black",
        linewidth=0.9,
    )
    max_height = max(plotted, default=0.0)
    if max_height > 0:
        ax.set_ylim(top=max_height * (1.0 + _YLIM_TOP_PADDING_FRACTION))
    _annotate_bars(ax, bars, values, style.fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(style.ylabel)
    ax.set_title(style.title)
    ax.grid(axis="y")


def _plot_stratified_memory_panel(
    ax: Axes,
    labels: Sequence[str],
    breakdowns: Sequence[MemoryBreakdown],
    style: _PanelStyle,
) -> None:
    """Render one memory panel of the 2x2 grid (Micro-Prompt 5d) as a
    two-segment STACKED bar per `labels` entry: an opaque, densely-hatched
    NumPy/CPU segment (`MemoryBreakdown.numpy_mb`) topped by a
    `_MEMORY_LAYER_ALPHA`-lightened, sparsely-hatched PyTorch CPU/GPU segment
    (`MemoryBreakdown.torch_mb`) -- both colored/hatched by the SAME
    per-method `colors`/`hatches` `_plot_metric_panel` uses, so a method's
    identity still reads as one color family across all four panels, while
    shade + hatch density alone (never color) separates its two memory
    layers. Needed because a single flat bar cannot represent two
    heterogeneous device/library layers that can each independently be near-
    zero (e.g. Riccati's ~0 PyTorch layer, GD's ~0 NumPy layer online)
    without misleadingly implying the OTHER layer is also zero.
    """
    colors, hatches = style.colors, style.hatches
    x = np.arange(len(labels))
    numpy_values = [
        0.0 if b.numpy_mb is None or not np.isfinite(b.numpy_mb) else float(b.numpy_mb)
        for b in breakdowns
    ]
    torch_values = [
        0.0 if b.torch_mb is None or not np.isfinite(b.torch_mb) else float(b.torch_mb)
        for b in breakdowns
    ]
    sparse_hatches = [h[0] for h in hatches]

    ax.bar(
        x,
        numpy_values,
        color=colors,
        hatch=hatches,
        edgecolor="black",
        linewidth=0.9,
        label="NumPy / CPU",
    )
    torch_bars = ax.bar(
        x,
        torch_values,
        bottom=numpy_values,
        color=colors,
        hatch=sparse_hatches,
        alpha=_MEMORY_LAYER_ALPHA,
        edgecolor="black",
        linewidth=0.9,
        label="PyTorch / CPU+CUDA",
    )

    total_values = [b.total_mb for b in breakdowns]
    stack_tops = [n + t for n, t in zip(numpy_values, torch_values)]
    max_height = max(stack_tops, default=0.0)
    if max_height > 0:
        ax.set_ylim(top=max_height * (1.0 + _YLIM_TOP_PADDING_FRACTION))
    _annotate_bars(ax, torch_bars, total_values, "{:.3g} MB")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(style.ylabel)
    ax.set_title(style.title)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(axis="y")


def plot_computational_benchmarks(
    records: Sequence[BenchmarkRecord],
    *,
    title: str = "Computational Benchmarking",
) -> Figure:
    """Asset 7: a 2x2 Figure isolating the two operational phases -- offline
    synthesis (left column) vs. online inference (right column) -- across the
    two measured metrics -- wall-clock time (top row) vs. stratified memory
    footprint (bottom row) -- comparing `records`' methods.

    Args:
        records: one complete phase-split measurement per method (times in
            seconds; memory as `MemoryBreakdown`s from
            `core.profiling.track_stratified_memory`, converted to MB --
            rendered as stacked NumPy/CPU-then-PyTorch-CPU/GPU bars, see
            `_plot_stratified_memory_panel`, so a NumPy-only closed-form
            solver and a PyTorch-native iterative solver each show their own
            footprint rather than one silently reading as zero).
        title: figure suptitle.

    Returns:
        The Figure, with ``fig.axes == [offline_time_ax, online_time_ax,
        offline_memory_ax, online_memory_ax]``.

    Raises:
        ValueError: If `records` is empty.
    """
    if not records:
        raise ValueError("records must contain at least one BenchmarkRecord.")

    labels = [record.label for record in records]
    offline_s = [record.offline_s for record in records]
    online_s = [record.online_s for record in records]
    offline_memory = [record.offline_memory for record in records]
    online_memory = [record.online_memory for record in records]

    x = np.arange(len(labels))
    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(x))]
    hatches = [_BAR_HATCHES[i % len(_BAR_HATCHES)] for i in range(len(x))]

    with styled_figure():
        (
            fig,
            (
                (offline_time_ax, online_time_ax),
                (offline_memory_ax, online_memory_ax),
            ),
        ) = plt.subplots(2, 2, figsize=(13, 10))

        _plot_metric_panel(
            offline_time_ax,
            labels,
            offline_s,
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Wall-clock time (s)",
                title="Offline Synthesis: Execution Time",
                fmt="{:.3g} s",
            ),
        )
        _plot_metric_panel(
            online_time_ax,
            labels,
            online_s,
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Wall-clock time (s)",
                title="Online Inference: Execution Time",
                fmt="{:.3g} s",
            ),
        )
        _plot_stratified_memory_panel(
            offline_memory_ax,
            labels,
            offline_memory,
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Stratified memory (MB)",
                title="Offline Synthesis: Memory Footprint",
            ),
        )
        _plot_stratified_memory_panel(
            online_memory_ax,
            labels,
            online_memory,
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Stratified memory (MB)",
                title="Online Inference: Memory Footprint",
            ),
        )

        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    return fig


def plot_cartesian_performance_tradeoff(
    records: Sequence[BenchmarkRecord],
    *,
    title: str = "The Deep-Unfolding Trade-off: Offline vs. Online x Time vs. Space",
) -> Figure:
    """NB03 Phase B (directive 4): the 2x2 Cartesian performance barplot --
    **Offline vs. Online** (columns) crossed with **Time vs. Space/Memory**
    (rows) -- purpose-built to make the deep-unfolding trade-off legible: the
    learned contenders pay a high ONE-TIME OFFLINE cost (top-left training
    time, bottom-left training memory) to buy an ULTRA-LOW, horizon-independent
    ONLINE cost (top-right per-control-step latency in MICROSECONDS,
    bottom-right resident memory).

    Reuses `plot_computational_benchmarks`' own panel helpers
    (`_plot_metric_panel`/`_plot_stratified_memory_panel`), differing only in
    that the online-time panel is expressed in **microseconds per control
    step** (the deployed real-time unit) rather than seconds, and the panel
    titles name the phase in trade-off terms.

    Args:
        records: one phase-split `BenchmarkRecord` per contender. ``online_s``
            is the mean per-control-step latency in seconds (rendered as
            microseconds here); memories are `MemoryBreakdown`s in MB.
        title: figure suptitle.

    Returns:
        The Figure, with ``fig.axes == [offline_time_ax, online_latency_ax,
        offline_memory_ax, online_memory_ax]``.

    Raises:
        ValueError: If `records` is empty.
    """
    if not records:
        raise ValueError("records must contain at least one BenchmarkRecord.")

    labels = [record.label for record in records]
    x = np.arange(len(labels))
    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i in range(len(x))]
    hatches = [_BAR_HATCHES[i % len(_BAR_HATCHES)] for i in range(len(x))]

    with styled_figure():
        (
            fig,
            (
                (offline_time_ax, online_latency_ax),
                (offline_memory_ax, online_memory_ax),
            ),
        ) = plt.subplots(2, 2, figsize=(13, 10))

        _plot_metric_panel(
            offline_time_ax,
            labels,
            [record.offline_s for record in records],
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Wall-clock time (s)",
                title="Offline (Synthesis / Training): Time  [pay once]",
                fmt="{:.3g} s",
            ),
        )
        _plot_metric_panel(
            online_latency_ax,
            labels,
            [record.online_s * 1e6 for record in records],
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Latency (microseconds / control step)",
                title="Online (Inference): Time  [pay every step, O(1) in N]",
                fmt="{:.1f} us",
            ),
        )
        _plot_stratified_memory_panel(
            offline_memory_ax,
            labels,
            [record.offline_memory for record in records],
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Memory (MB)",
                title="Offline (Synthesis / Training): Space",
            ),
        )
        _plot_stratified_memory_panel(
            online_memory_ax,
            labels,
            [record.online_memory for record in records],
            _PanelStyle(
                colors=colors,
                hatches=hatches,
                ylabel="Memory (MB)",
                title="Online (Inference): Space  [resident footprint]",
            ),
        )

        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    return fig
