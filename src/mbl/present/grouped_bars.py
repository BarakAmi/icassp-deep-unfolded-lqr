"""`grouped_bars` — an aggregate metric over a categorical axis (§B.6).

Figure 2's renderer, and Figure 4's: *"both layouts are the same grouped-bar
renderer Phase D already builds"*. It renders `cost_by_category`'s tidy table
and computes nothing.

**The contenders are the x axis and the categories are the fills, and that is
a measurement rather than a preference.** A bar has no line and no marker, so
its only shape channel is the texture, and §B.4.1 clause 2 then decides every
pair the texture does not separate on luminance alone. This repository's
palette cannot carry seven of those: 17 of its 28 unordered pairs sit below
the 0.15 floor and the largest mutually separable subset is **three**. Putting
the contenders on the axis costs nothing and gains the strongest identity
channel there is — a tick label, which colour removal cannot touch at all.

**Three rules the shape then has to keep**, each of which was a defect first:

1. **Every drawn container is labelled** (§B.4.2 rule 3). `ax.bar(label=…)`
   puts the label on the container and leaves every patch at `_nolegend_`, so
   an unlabelled bar chart gives the greyscale gate 0 encodings, 0 conflicts
   and a pass. The gate refuses that outright now; this renderer never
   produces it.
2. **The value axis includes zero unless a baseline is DECLARED AND MARKED**
   (§B.5.2, amended 2026-08-10). A bar says how much by how long, so a
   truncated axis multiplies the apparent effect by whatever the author chose
   — which is why `ylim` is still absent and always will be. `ybase` is the
   one narrow permission: the value is declared in the document (never
   inferred from the data), the renderer draws a break mark at it, and the
   caption may not compare bar lengths as a ratio. Where those cannot be met
   the figure changes its *quantity*, not its axis.
3. **A bound is a level, not a bar** (§B.5.2). It has no category to stand at,
   so a bar drawn for it would claim it belongs to one — and §B.5 labels it at
   the right margin, in its colour, with its value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from ..spec.contender import Role
from ..spec.errors import SpecificationError
from ..viz.style.theme import LegendStyle, place_legend_outside
from .axis_scaling import (
    AXIS_KIND_COLUMN,
    CAPSIZE,
    CATEGORICAL_AXIS,
    DISPERSION_CHANNELS,
    DISPERSION_LEGEND_TITLE,
    LEGEND_ALPHA,
    SPREAD_COLUMN,
    Y_MARGIN,
)
from .encodings import (
    FillStyle,
    SeriesStyle,
    encode_fills,
    encode_series,
    legend_label,
)
from .margin_labels import label_levels
from .profiles import profile_context
from .registry import FigureContext, register_figure
from .selection import select_series

#: Columns `cost_by_category` emits that this renderer reads. Named so a schema
#: change is refused here rather than surfacing as an empty figure.
REQUIRED_COLUMNS = (
    "contender",
    "role",
    "axis_value",
    "axis_label",
    "aggregate",
)

#: Every key this renderer reads. Anything else is **refused** — the defect
#: class this project has shipped six times, where a key parses, reaches
#: nothing and renders a byte-identical figure.
#:
#: **`ylim` is absent on purpose** and its absence is §B.5.2 enforced: see the
#: module docstring. `ybase` is NOT its return under another name -- it moves
#: the floor only, the renderer marks the interruption, and a base above a bar
#: is refused; `ylim` would let an author move the TOP as well and clip the
#: very marks the figure exists to show.
CONFIG_KEYS = frozenset(
    {
        "series",
        "xlabel",
        "ylabel",
        "title",
        "dispersion",
        "yscale",
        "ybase",
        "omit",
    }
)

#: Half-length of the baseline's break mark, in axes-fraction units, and its
#: slope — §B.5.1's device, applied to the one truncation §B.5.2 admits.
BASE_MARK_SIZE = 0.012
BASE_MARK_SLOPE = 2.0

#: What `yscale` may say. §B.5.2: linear includes zero; log is admissible only
#: where the quantity spans decades and has no meaningful zero — §A.5's compute
#: and memory, where the Riccati stack is ~12.8 kB against a GRU's ~108 kB.
Y_SCALES = ("linear", "log")

#: Total share of one contender's x slot that its bars occupy, leaving the rest
#: as the gutter between neighbouring contenders. Below about 0.9 the groups
#: read as separate; at 1.0 they touch and the grouping stops being visible.
GROUP_WIDTH = 0.8

#: How much of its own slot a bar fills, leaving a hairline between the bars
#: WITHIN a group. Without it two adjacent fills meet edge to edge and read as
#: one wider bar once colour is removed, which is the whole comparison the
#: grouping exists to make.
BAR_INSET = 0.92

#: §B.5's "data under study": the roles that SET the axis, as opposed to the
#: ones drawn clipped against it. The same set `axis_scaling` uses, for the
#: same reason -- a depth-invariant CONTENDER is still data.
_DATA_UNDER_STUDY = frozenset({Role.CONTENDER.value, Role.BASELINE.value})


@dataclass(frozen=True)
class _Bars:
    """One category's bars across every contender, ready to draw."""

    category: str
    fill: FillStyle
    positions: np.ndarray
    heights: np.ndarray
    spreads: np.ndarray | None
    #: The bar's own width. Carried rather than re-derived at draw time: the
    #: obvious derivation is the spacing between this category's positions,
    #: which is the distance between two CONTENDERS (1.0) and not the width of
    #: one bar -- measured, that drew every bar 0.92 wide inside a 0.8 group
    #: and the neighbouring contenders' bars overlapped by more than half.
    width: float


@register_figure("grouped_bars")
def grouped_bars(context: FigureContext) -> Figure:
    """An aggregate metric per contender, grouped by category.

    Args:
        context: The tidy table, the declaration and the style profile.

    Returns:
        The rendered figure. The caller owns it — the greyscale gate must be
        able to inspect it before anything is written.

    Raises:
        SpecificationError: If the table is not `cost_by_category`'s, if the
            configuration names a key this renderer does not read, or if it
            selects a contender the table does not carry.
        EncodingCapacityError: Past the nine fills §B.4.2 can separate.
    """
    table = _validated(context.table)
    _require_known_config(context.config)
    # Idempotent: the RUNNER has already selected before freezing the frame
    # (§B.1.2 clause 2). Kept here so a renderer invoked directly honours the
    # same declaration.
    selected = select_series(table, context.config)
    channel = _dispersion_channel(context.config)

    bars = selected[selected["axis_value"].notna()]
    levels = selected[selected["axis_value"].isna()]
    contenders = _contender_order(bars, context.series_order)
    # The CATEGORIES are read before the omissions are applied, so a cell a
    # family cannot have does not silently take its category off the figure
    # for everyone (§B.5.2, 2026-08-10).
    categories = _category_order(bars)
    fills = encode_fills(categories)
    bars = _omit_cells(bars, context.config, contenders, categories)

    with profile_context(context.profile):
        figure, axes = plt.subplots()
        drew_a_bar = False
        for group in _grouped(bars, contenders, categories, fills, channel):
            drew_a_bar |= _draw_bars(axes, group)
        # The levels are drawn AFTER the bars so a reference sits over them —
        # a bound behind a filled bar is a bound a reader cannot see.
        styles = encode_series(context.series_order or contenders, context.roles)
        artist_labels: dict[str, str] = {}
        for label in _level_order(levels, context.series_order):
            artist_labels[label] = _draw_level(
                axes, levels, label, styles, context.roles, context.display_names
            )

        _label_contenders(axes, contenders, context.display_names)
        _apply_scale(axes, context.config)
        _apply_axis_rule(axes, selected, context.config)
        _apply_baseline(axes, bars, context.config)
        axes.set_xlabel(str(context.config.get("xlabel", "")))
        axes.set_ylabel(str(context.config.get("ylabel", "aggregate cost")))
        if context.config.get("title") is not None:
            axes.set_title(str(context.config["title"]))
        # §B.5: the legend names the CATEGORIES, and the levels are named at the
        # right margin instead — which is the whole reason the key stays short
        # enough to sit inside the axes.
        #
        # A CLIPPED level is the exception and it keeps its legend row (§B.3.1).
        # The margin can only label a series at the height it is drawn, and a
        # clipped one is not drawn at any: measured, a bound at 40 under an axis
        # ending at 9.9 had its margin label placed at axes fraction **4.04** —
        # four panel heights above the figure, off the canvas, where
        # `bbox_inches="tight"` would then expand the page to hold it. Its line
        # was invisible too, so the bound was reported nowhere at all.
        contained, clipped = _split_by_containment(axes, _levels(levels))
        place_legend_outside(
            figure,
            axes,
            inside=True,
            only=list(categories)
            + [artist_labels[label] for label, _ in clipped if label in artist_labels],
            style=LegendStyle(loc="upper center", framealpha=LEGEND_ALPHA),
        )
        if drew_a_bar:
            # §A.3.2 rule 1, and only when there is ink to name.
            legend = axes.get_legend()
            if legend is not None:
                legend.set_title(DISPERSION_LEGEND_TITLE)
        _clear_the_legend(figure, axes)
        if context.config.get("ybase") is not None:
            # AFTER the headroom step, which moves the top limit only: the
            # mark is anchored in axes fractions, so it rides any later
            # change, but drawing it before `_clear_the_legend` would put a
            # line artist in `_drawn_peak`'s way.
            _mark_baseline(axes)
        _slant_crowded_ticks(figure, axes)
        label_levels(figure, [(axes, contained)], styles, context.display_names)
    return figure


#: Angles to try, in order, when the category labels overprint (§B.5.2). A
#: LADDER rather than a constant: the first angle that clears is used, so the
#: figure is slanted as little as its own names require. Whether names fit
#: depends on how many there are, how long they are and how wide the venue is,
#: and a single value chosen against one of those is the free parameter this
#: project has a standing rule against.
TICK_SLANTS = (30.0, 45.0, 60.0, 90.0)


def _slant_crowded_ticks(figure: Figure, axes: Axes) -> None:
    """Slant the category labels, and only as far as they actually need.

    §B.4.2 puts the contenders on the x axis because a tick label is an
    identity channel colour removal cannot touch — which is worth nothing if
    the labels overprint. Measured on ICASSP Figure 2 at `ieee-2col`: nine
    display names, five of them overlapping a neighbour.

    **Measured, not declared, in both directions.** A two-category figure reads
    better flat and is left flat; a nine-category one is slanted until it
    clears. Asked after `_clear_the_legend`, so the layout it measures is the
    one the reader gets.

    A label that still overlaps at the last angle is left there: the figure has
    more categories than its width can name, and §B.5.2 calls that a
    composition to change rather than a label to shrink — shrinking it would
    trade against §B.2's type floors, which are not negotiable.
    """
    figure.canvas.draw()
    if not _ticks_overlap(axes):
        return
    for angle in TICK_SLANTS:
        for text in axes.get_xticklabels():
            text.set_rotation(angle)
            # Anchored at the RIGHT end. Slanted text anchored at its centre
            # drifts away from the tick it names, which on a bar chart is the
            # one thing the label is for.
            text.set_horizontalalignment("right")
            text.set_rotation_mode("anchor")
        figure.canvas.draw()
        if not _ticks_overlap(axes):
            return


def _ticks_overlap(axes: Axes) -> bool:
    """Whether any two category labels share pixels, as rendered.

    The half-pixel slack keeps two boxes that merely touch from counting: at
    `bbox_inches="tight"` an exact abutment prints as a gap, and refusing it
    would slant every figure whose names happen to fill their slot.
    """
    boxes = [
        text.get_window_extent() for text in axes.get_xticklabels() if text.get_text()
    ]
    boxes.sort(key=lambda box: box.x0)
    return any(
        first.x1 > second.x0 + 0.5
        for first, second in zip(boxes, boxes[1:], strict=False)
    )


def _clear_the_legend(figure: Figure, axes: Axes) -> None:
    """Raise the top of the axis until the legend covers no bar top.

    **§B.5's inside-legend argument was made for curves and does not transfer
    to bars.** It permits the overlap because "what it can cover is the middle
    of the swept axis and never the ends — and on a cost-versus-depth figure
    the ends are the claim". A bar has no ends: it runs from zero to its value,
    so the *top* is the only part that carries the number, and the top centre
    is exactly where §B.5 puts the key. Measured on the end-to-end render: a
    two-category figure had the legend across the tops of **both** bars.

    **Headroom rather than relegation**, for two reasons. §B.5's own
    measurement stands — moving the legend below costs about a quarter of the
    plot — and on a zero-based axis extra headroom distorts nothing: every bar
    still starts at zero, so the ratio a reader reads between two lengths is
    unchanged (§B.5.2). Only the ink-to-space ratio moves.

    **The question is asked at a moment when the answer is true**, which is the
    ordering defect this project has shipped twice: the legend's extent is
    matplotlib's own only after a draw, and the axis limit it is measured
    against must already be set. Both hold here — this runs after
    `_apply_axis_rule` and after the legend is placed.

    The arithmetic closes in one step rather than iterating: the legend is
    anchored in *axes fractions*, so its bottom edge stays at the same fraction
    `f` when the limits change, and `low + f · (high − low) ≥ peak` solves to
    `high = low + (peak − low) / f`.
    """
    legend = axes.get_legend()
    if legend is None or axes.get_yscale() != "linear":
        return
    figure.canvas.draw()
    box = legend.get_window_extent().transformed(axes.transAxes.inverted())
    fraction = float(box.y0)
    low, high = axes.get_ylim()
    peak = _drawn_peak(axes)
    if peak is None or fraction <= 0.0 or fraction >= 1.0:
        return
    if low + fraction * (high - low) >= peak:
        return
    axes.set_ylim(low, low + (peak - low) / fraction)


def _drawn_peak(axes: Axes) -> float | None:
    """The highest mark on the canvas: a bar top, or the cap above it.

    Read from the ARTISTS rather than from the table, because the error bar's
    cap is drawn above the aggregate and a peak taken from the aggregates alone
    would leave the caps under the key.
    """
    heights: list[float] = []
    for container in axes.containers:
        if isinstance(container, BarContainer):
            # BOTH edges of each patch. A bar of negative height has its top at
            # `get_y()` and its foot at `get_y() + get_height()`, so reading the
            # sum alone finds the bottom of a downward bar and leaves the key
            # over the very ends a percentage figure is read from.
            for patch in container.patches:
                foot = float(patch.get_y())
                heights.extend((foot, foot + float(patch.get_height())))
        else:
            for line in getattr(container, "lines", ()) or ():
                for part in line if isinstance(line, tuple) else (line,):
                    data = np.asarray(getattr(part, "get_ydata", list)(), dtype=float)
                    heights.extend(float(value) for value in data if np.isfinite(value))
    return max(heights) if heights else None


def _split_by_containment(
    axes: Axes, levels: Sequence[tuple[str, float]]
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """`(labelled at the margin, kept in the legend)`.

    Asked **after** `_apply_axis_rule`, because "is this level on the canvas"
    is a question about the final limits and the answer is a different one
    before they are set — this project has shipped an ordering defect of
    exactly that shape twice, where a decision read a geometry matplotlib had
    not yet assigned.
    """
    low, high = axes.get_ylim()
    contained = [(label, value) for label, value in levels if low <= value <= high]
    clipped = [(label, value) for label, value in levels if not low <= value <= high]
    return contained, clipped


# -- what the table says ----------------------------------------------------


def _validated(table: pd.DataFrame) -> pd.DataFrame:
    """Refuse a table this renderer cannot draw, before any artist exists."""
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise SpecificationError(
            f"grouped_bars needs the columns {', '.join(missing)}, which this "
            "table does not carry; it renders `cost_by_category`'s output and "
            "computes nothing of its own"
        )
    if table.empty:
        raise SpecificationError(
            "grouped_bars was given an empty table; a figure with no series is "
            "not a figure, and an analysis that produced none should have said so"
        )
    _require_a_categorical_axis(table)
    return table


def _require_a_categorical_axis(table: pd.DataFrame) -> None:
    """Refuse a numeric table (§A.6.1's last paragraph, the other direction).

    A numeric axis reduced by `cost_vs_axis` puts the axis *value* in
    `axis_value` — 2, 4, 8 — and this renderer would place bars at positions
    0, 1, 2 and label them from `axis_label`, silently discarding the spacing
    that made the axis numeric in the first place. The pairing is therefore
    checked in both directions rather than trusted in either.

    **An absent column means numeric** (§A.6.1), so this refuses that too, and
    the message says how to get a categorical table rather than only that this
    one is wrong.

    Raises:
        SpecificationError: Naming the analysis kind that emits what is needed.
    """
    declared = (
        set(table[AXIS_KIND_COLUMN]) if AXIS_KIND_COLUMN in table.columns else set()
    )
    if declared == {CATEGORICAL_AXIS}:
        return
    raise SpecificationError(
        "grouped_bars renders a categorical axis and this table declares "
        f"{sorted(declared) or 'none'} (Annex 03 §A.6.1: an absent "
        f"{AXIS_KIND_COLUMN!r} column means numeric). Its x positions are "
        "categories in the study's declared order, so a numeric axis rendered "
        "here would lose the spacing that made it numeric. Produce the table "
        "with kind = 'cost_by_category', which takes a numeric axis too and "
        "orders its values as the document declares them"
    )


def _contender_order(bars: pd.DataFrame, order: Sequence[str]) -> tuple[str, ...]:
    """The x ticks: contenders that carry a category, in declaration order.

    §B.3.1's "dropping a contender must not repaint the survivors" is trivial
    here — no contender owns a colour — but the ORDER is still the study's,
    never the table's, so two figures of one study put the same contender in
    the same place.
    """
    present = list(dict.fromkeys(bars["contender"]))
    declared = [label for label in order if label in present]
    return tuple(declared + [label for label in present if label not in declared])


def _category_order(bars: pd.DataFrame) -> tuple[str, ...]:
    """The categories, ordered by the position the analysis assigned them.

    §A.6.1: that position is the study's declaration order and is never the
    sorted-by-value order, so this renderer sorts by the integer and never by
    the name — a figure that alphabetised its own fills would disagree with
    the table beside it.
    """
    pairs = {
        str(row["axis_label"]): float(row["axis_value"]) for _, row in bars.iterrows()
    }
    return tuple(sorted(pairs, key=lambda category: pairs[category]))


# -- drawing ----------------------------------------------------------------


def _grouped(
    bars: pd.DataFrame,
    contenders: Sequence[str],
    categories: Sequence[str],
    fills: Mapping[str, FillStyle],
    channel: str,
) -> list[_Bars]:
    """One `_Bars` per category, aligned on the contender positions.

    A contender with no row for some category simply has no bar there, which
    is drawn as a gap rather than as a zero: a missing measurement and a
    measured zero are different facts, and a zero-height bar claims the second.
    """
    width = GROUP_WIDTH / max(len(categories), 1)
    offsets = [
        (index - 0.5 * (len(categories) - 1)) * width
        for index in range(len(categories))
    ]
    groups: list[_Bars] = []
    for category, offset in zip(categories, offsets, strict=True):
        rows = bars[bars["axis_label"] == category]
        by_contender = {str(row["contender"]): row for _, row in rows.iterrows()}
        positions: list[float] = []
        heights: list[float] = []
        spreads: list[float] = []
        for slot, contender in enumerate(contenders):
            row = by_contender.get(contender)
            if row is None:
                continue
            positions.append(slot + offset)
            heights.append(float(row["aggregate"]))
            spreads.append(_spread_of(row))
        groups.append(
            _Bars(
                category=category,
                fill=fills[category],
                positions=np.asarray(positions, dtype=np.float64),
                heights=np.asarray(heights, dtype=np.float64),
                spreads=_drawable(np.asarray(spreads, dtype=np.float64), channel),
                width=width,
            )
        )
    return groups


def _apply_baseline(axes: Axes, bars: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """§B.5.2's declared baseline: the bars are read from `ybase`, not zero.

    Admitted 2026-08-10 and narrowly: the value is DECLARED (an axis inferred
    from data moves when a seed does, and two renders then cannot be
    compared), the renderer draws the break mark that makes the truncation
    legible, and the caption may not compare bar lengths as a ratio. The
    concern the blanket ban expressed is recorded in the annex rather than
    resolved.

    Raises:
        SpecificationError: If the baseline is not below every drawn bar —
            a base above a bar turns it downward, which reads as a negative
            quantity — or if it is declared on a log axis, where §B.5.2's
            floor is the scale's own and a second one would fight it.
    """
    declared = config.get("ybase")
    if declared is None:
        return
    if axes.get_yscale() != "linear":
        raise SpecificationError(
            "`ybase` is declared on a logarithmic axis; a log scale has no "
            "zero to include and states its floor already (§B.5.2), so a "
            "second declared baseline would be two floors on one axis"
        )
    base = float(declared)
    heights = np.asarray(bars["aggregate"], dtype=np.float64)
    finite = heights[np.isfinite(heights)]
    if finite.size and base >= float(finite.min()):
        raise SpecificationError(
            f"`ybase` is {base:g} and the smallest drawn bar is "
            f"{float(finite.min()):g}; a baseline at or above a bar draws it "
            "downward, which a reader meets as a negative quantity"
        )
    # The limit alone, and the patches are left anchored at zero: they extend
    # below the visible range and matplotlib clips them there, which is what
    # a truncated bar IS. Moving each rectangle's foot would draw the same
    # ink and give the renderer a second place to be wrong about geometry.
    _, high = axes.get_ylim()
    axes.set_ylim(base, high)


def _omit_cells(
    bars: pd.DataFrame,
    config: Mapping[str, Any],
    contenders: Sequence[str],
    categories: Sequence[str],
) -> pd.DataFrame:
    """Drop the (series, category) cells the figure declares it cannot have.

    §B.5.2 as amended 2026-08-10. Not every cell of a contenders × categories
    grid is a measurement: a controller that never trains has no train–test
    relation to be mismatched, so a bar there would answer a question nobody
    asked of it. The omission is a FIGURE declaration — the analysis keeps
    every measured row — and it names both halves so a reader of the document
    can see which cell is missing and why the caption says so.

    Raises:
        SpecificationError: On an unknown series or category, on an omission
            matching no measured cell (inert, this project's most-repeated
            defect), or on one that would empty a series — dropping a series
            is what `series` is for, and doing it through a cell omission
            would leave a tick label naming nothing.
    """
    declared = config.get("omit")
    if declared is None:
        return bars
    known_series, known_categories = set(contenders), set(categories)
    kept = bars
    for series, omitted in dict(declared).items():
        if series not in known_series:
            raise SpecificationError(
                f"`omit` names series {series!r}, which this figure does not "
                f"draw; available: {', '.join(sorted(known_series))}"
            )
        names = [omitted] if isinstance(omitted, str) else list(omitted)
        for name in names:
            if name not in known_categories:
                raise SpecificationError(
                    f"`omit` names category {name!r} for {series!r}, which "
                    f"this figure does not draw; available: "
                    f"{', '.join(sorted(known_categories))}"
                )
            cell = (kept["contender"] == series) & (kept["axis_label"] == name)
            if not bool(cell.any()):
                raise SpecificationError(
                    f"`omit` drops {series!r} x {name!r}, which the frame does "
                    "not carry; an omission of a cell that is not there takes "
                    "no effect and reads as a decision that was made"
                )
            kept = kept[~cell]
        if not bool((kept["contender"] == series).any()):
            raise SpecificationError(
                f"`omit` leaves series {series!r} with no bar at all; drop it "
                "with `series` instead, or its tick label names nothing"
            )
    return kept


def _mark_baseline(axes: Axes) -> None:
    """§B.5.2's break mark at a declared baseline, on both ends of the axis.

    Not optional and not decoration: the permission to truncate a length
    encoding is granted *because* the interruption is drawn, so a reader
    meets it at the same moment they meet the bars. The same device §B.5.1
    already uses between two panels, applied to the one truncation §B.5.2
    admits — drawn in axes fractions so it stays put under any later limit
    change.
    """
    for x in (0.0, 1.0):
        axes.plot(
            [x - BASE_MARK_SIZE, x + BASE_MARK_SIZE],
            [-BASE_MARK_SIZE * BASE_MARK_SLOPE, BASE_MARK_SIZE * BASE_MARK_SLOPE],
            transform=axes.transAxes,
            color="black",
            linewidth=1.0,
            clip_on=False,
            # `_break`, the convention `axis_scaling._mark_break` already
            # uses: out of the legend, and read by the greyscale gate as
            # chrome rather than as an unencoded series.
            label="_break",
            zorder=5,
        )


def _draw_bars(axes: Axes, group: _Bars) -> bool:
    """One category's bars. Returns whether an error bar was drawn.

    **`label=` is not optional and is the point** (§B.4.2 rule 3): it is what
    the greyscale gate reads, and without it the gate inspects nothing and
    reports green.
    """
    axes.bar(
        group.positions,
        group.heights,
        width=group.width * BAR_INSET,
        label=group.category,
        color=group.fill.color,
        hatch=group.fill.hatch,
        edgecolor="white",
        linewidth=0.6,
    )
    if group.spreads is None:
        return False
    # UNLABELLED, and that is load-bearing: a labelled companion would make the
    # gate meet this series twice and refuse the figure for conflicting with
    # itself. The caps inherit no fill, so they read as the bar's own.
    axes.errorbar(
        group.positions,
        group.heights,
        yerr=group.spreads,
        fmt="none",
        ecolor="black",
        capsize=CAPSIZE,
        elinewidth=0.8,
    )
    return True


def _draw_level(
    axes: Axes,
    levels: pd.DataFrame,
    label: str,
    styles: Mapping[str, SeriesStyle],
    roles: Mapping[str, Role],
    display_names: Mapping[str, str],
) -> str:
    """A contender the axis does not apply to: one horizontal reference.

    §B.5.2's last clause. Returns the label the ARTIST carries, which the
    legend filter must match on: `legend_label` appends a bound's value
    ("COCP-LB (8.273)"), so filtering by the contender key would silently drop
    exactly the clipped bound §B.3.1 requires to keep its row.

    The artist is labelled whether or not it takes a legend row, because the
    greyscale gate reads **artist labels** and a series whose label were
    dropped to suppress its row would leave the gate inspecting nothing.
    """
    rows = levels[levels["contender"] == label]
    if rows.empty:
        return label
    value = float(rows.iloc[0]["aggregate"])
    style = styles.get(label, SeriesStyle("black", "-", ""))
    text = legend_label(label, value, roles.get(label, Role.CONTENDER), display_names)
    axes.axhline(
        value,
        label=text,
        color=style.color,
        linestyle=style.linestyle,
        marker="",
        linewidth=1.1,
    )
    return text


def _label_contenders(
    axes: Axes, contenders: Sequence[str], display_names: Mapping[str, str]
) -> None:
    """The x ticks — where a bar figure's identity actually lives.

    Annex 04 §1.3: what a reader meets is the display name, and the join key
    stays the join key everywhere upstream of this line.
    """
    axes.set_xticks(list(range(len(contenders))))
    axes.set_xticklabels(
        [display_names.get(label, label) for label in contenders],
        rotation=0.0,
    )
    axes.set_xlim(-0.5, len(contenders) - 0.5)


def _level_order(levels: pd.DataFrame, order: Sequence[str]) -> list[str]:
    present = list(dict.fromkeys(levels["contender"]))
    declared = [label for label in order if label in present]
    return declared + [label for label in present if label not in declared]


def _levels(levels: pd.DataFrame) -> list[tuple[str, float]]:
    """`(label, value)` for every level, name-sorted — one reading, as
    `_flat_levels` is for `axis_scaling`."""
    return sorted(
        (str(row["contender"]), float(row["aggregate"])) for _, row in levels.iterrows()
    )


# -- the axes ---------------------------------------------------------------


def _apply_scale(axes: Axes, config: Mapping[str, Any]) -> None:
    """§B.5.2's one exception, declared rather than inferred.

    Raises:
        SpecificationError: On a scale this renderer does not implement.
    """
    declared = str(config.get("yscale", Y_SCALES[0]))
    if declared not in Y_SCALES:
        raise SpecificationError(
            f"yscale {declared!r} is not one of {', '.join(Y_SCALES)}; Annex 03 "
            "§B.5.2 keeps a bar's value axis linear and including zero, and "
            "admits a log axis only where the quantity spans decades and has "
            "no meaningful zero"
        )
    if declared == "log":
        axes.set_yscale("log")


def _apply_axis_rule(
    axes: Axes, table: pd.DataFrame, config: Mapping[str, Any]
) -> None:
    """§B.5 scaled to the data under study, §B.5.2 anchored at zero.

    **The two rules meet here and §B.5.2 wins the end of the axis that zero is
    on.** §B.5 would scale to the contenders and leave a bound clipped; for a
    *length* encoding that is the truncation §B.5.2 forbids, so zero is always
    an endpoint and only the other end is chosen. A reference between zero and
    the data is therefore always shown, which costs nothing.

    **Zero is a floor OR a ceiling, and getting that wrong made a figure of
    solid ink.** The first version took `low = min(0, ...)` and padded the top,
    which is right for positive data and silently wrong for negative: measured
    on a real degradation table whose values were −68.06 % and −66.02 %, the
    limits came out `(−68.06, −65.75)` — zero **outside the axis**, every bar
    drawn from an origin the reader cannot see, and the panel a single filled
    rectangle. It passed the greyscale gate, because a bar that fills the panel
    is still separable. A percentage is exactly the quantity that goes
    negative, so this is the case Figure 2 is made of rather than an edge one.

    Both ends are therefore taken against zero, and only an end that has data
    beyond zero is padded — so an all-positive figure keeps its floor at
    exactly 0.0 rather than gaining a margin below it, which would put the
    baseline off the axis in the other direction.

    On a log axis neither applies — there is no zero to include and nothing is
    read as a length from an origin — so the limits are matplotlib's own.
    """
    if axes.get_yscale() == "log":
        return
    under_study = table[table["role"].isin(_DATA_UNDER_STUDY)]
    # ONE frame for both, or the two arrays are different lengths -- which is
    # not a broadcast to be fixed but a sign that the spread of row `i` was
    # about to be added to the aggregate of a different row.
    scaling = under_study if not under_study.empty else table
    values = np.asarray(scaling["aggregate"], dtype=np.float64)
    spreads = (
        np.asarray(scaling[SPREAD_COLUMN], dtype=np.float64)
        if SPREAD_COLUMN in scaling.columns
        else np.zeros_like(values)
    )
    drawable = np.where(np.isfinite(spreads), spreads, 0.0)
    # THE DATA UNDER STUDY ONLY, and a level is deliberately not in it. Adding
    # every level here was tried and is wrong for the same reason §B.5 gives
    # for a line figure: a bound at 40 against bars at 8-9 stretches the axis
    # 4.4x and compresses the comparison the figure exists to make. A bar
    # figure has less room to give away than a curve, not more, because its
    # axis already spends the whole span from zero. A clipped level keeps its
    # legend row with its value (§B.3.1), which is what `_split_by_containment`
    # is for.
    everything = np.concatenate([values, values + drawable, values - drawable])
    finite = everything[np.isfinite(everything)]
    if finite.size == 0:
        return
    low = min(0.0, float(finite.min()))
    high = max(0.0, float(finite.max()))
    pad = Y_MARGIN * (high - low or 1.0)
    axes.set_ylim(
        low - (pad if low < 0.0 else 0.0),
        high + (pad if high > 0.0 else 0.0),
    )


# -- the declaration --------------------------------------------------------


def _require_known_config(config: Mapping[str, Any]) -> None:
    """Refuse a figure declaring a key this renderer does not read.

    Raises:
        SpecificationError: Naming the unknown keys and what is available. A
            `ylim` lands here rather than in a missing branch, which is how
            §B.5.2's "no truncation key" is enforced rather than trusted.
    """
    unknown = sorted(set(config) - CONFIG_KEYS)
    if not unknown:
        return
    raise SpecificationError(
        f"figure declares {', '.join(unknown)}, which `grouped_bars` does not "
        f"read; available: {', '.join(sorted(CONFIG_KEYS))}. Annex 03 §B.5.2 "
        "gives a bar figure no truncation key at all: a bar encodes magnitude "
        "by length, so its axis includes zero, and where that makes the "
        "comparison unreadable the figure changes its quantity rather than "
        "its axis"
    )


def _dispersion_channel(config: Mapping[str, Any]) -> str:
    """The figure's declared dispersion channel (§A.3.2 rule 4).

    Raises:
        SpecificationError: On a channel this renderer does not implement.
    """
    declared = str(config.get("dispersion", DISPERSION_CHANNELS[0]))
    if declared not in DISPERSION_CHANNELS:
        raise SpecificationError(
            f"dispersion {declared!r} is not one of "
            f"{', '.join(DISPERSION_CHANNELS)}; §A.3.2 rule 4 lets a figure "
            "draw the across-seed spread as an error bar or draw none at all, "
            "and 'none' requires the spread in the figure's table"
        )
    return declared


def _spread_of(row: Any) -> float:
    value = float(row[SPREAD_COLUMN]) if SPREAD_COLUMN in row.index else float("nan")
    return value if np.isfinite(value) else 0.0


def _drawable(spreads: np.ndarray, channel: str) -> np.ndarray | None:
    """§A.3.2 rule 2: an absent spread and a zero spread both draw nothing.

    One seed has no across-seed quantity at all; an analytic contender under
    §A.2's common-random-numbers law has a spread of exactly zero and that
    zero is the answer. A zero-height bar still draws two caps, which a reader
    meets as a narrow measured interval — so both cases draw nothing and the
    table carries the distinction as `—` against `0.000`.
    """
    if channel == "none" or spreads.size == 0:
        return None
    return spreads if bool((spreads > 0.0).any()) else None


__all__ = ["CONFIG_KEYS", "GROUP_WIDTH", "REQUIRED_COLUMNS", "Y_SCALES", "grouped_bars"]
