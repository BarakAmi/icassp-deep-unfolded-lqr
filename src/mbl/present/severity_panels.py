"""§B.6 renderer — one quantity against one axis, in several conditions.

Three tables that share an x axis, drawn as panels side by side: the campaign's
mismatch figure is cost against rotation angle under *uninformed*, *informed*
and *model mismatch*. The panels share both scales, so a vertical distance
means the same thing in each and the comparison a reader makes across them is
the one the experiment supports.

**Why this is a registered kind rather than a drawing in a tool.** It began as
one, and that was a defect the author found by trying to reopen the figure: a
stored figure is re-rendered from its `spec.json` and `data.parquet` through
the registry, so a `kind` no renderer answers to produces four artifacts that
cannot be rebuilt — while every other figure in the store can. The asymmetry is
invisible until someone opens the file.

The table is the three analyses concatenated with a `panel` column naming which
each row came from; the panel order and titles are declared in the config, so
the figure is reproducible from its artifacts alone.

`highlight` names one contender to point at in every panel, and
`highlight_at` the positions to point at -- one label per panel with an arrow
to each, in that series' own colour and linestyle, with a filled head. It exists because seven curves at this scale resolve into a band, and a
reader asked to find the proposed one has to count legend entries against line
styles. It is declared in the config rather than drawn by the tool for the
reason the whole renderer is: a figure is rebuilt from its `spec.json`, so an
annotation a tool adds afterwards is one a rebuild silently drops.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ..spec.errors import SpecificationError
from .encodings import encode_series
from .profiles import profile_context
from .registry import FigureContext, register_figure

#: The column naming which panel a row belongs to.
PANEL_COLUMN = "panel"

#: Columns this renderer reads. Named so a schema change is refused here rather
#: than surfacing as an empty panel.
REQUIRED_COLUMNS = ("contender", "axis_label", "aggregate", PANEL_COLUMN)


def _series(
    table: pd.DataFrame, contender: str
) -> tuple[list[float], list[float], list[float]]:
    """One contender's (angle, value, spread) within a panel, sorted by angle.

    The x position is READ from the axis label the study declared, never
    recomputed from anything: the label is what the study says the position is,
    and a second derivation would be a second answer.
    """
    rows = table[table["contender"] == contender]
    spread = "across_seed_spread"
    points = sorted(
        (
            float(row["axis_label"]),
            float(row["aggregate"]),
            float(row[spread]) if spread in rows.columns else 0.0,
        )
        for _, row in rows.iterrows()
    )
    return ([p[0] for p in points], [p[1] for p in points], [p[2] for p in points])


def _label_spot(
    axis: Any,
    panel: pd.DataFrame,
    order: list[str],
    anchors: list[tuple[float, float]],
    avoid_legend: bool,
) -> tuple[float, float]:
    """Where the label goes, in AXES FRACTION: near its arrows, and clear.

    Axes fraction rather than an offset in points from a data anchor, because
    the requirement is that the label and every arrow stay inside the panel.
    Where a curve sits says nothing about how much room is left beyond it, and
    an offset put the third panel's label outside the frame.

    **Near the anchors, not merely in the emptiest place.** A first version
    picked the panel's emptiest spot outright and got three arrows spanning the
    whole width, crossing every other curve on the way -- the opposite of
    pointing something out. Candidates now ring the anchors' centroid, so the
    arrows stay short, and the ring position is chosen by clearance from every
    drawn point, so they stay out of the traffic.
    """
    left, right = axis.get_xlim()
    low, high = axis.get_ylim()
    span_x = (right - left) or 1.0
    span_y = (high - low) or 1.0
    drawn = [
        ((x - left) / span_x, (value - low) / span_y)
        for name in order
        for x, value in zip(*_series(panel, name)[:2], strict=True)
    ]
    centre = (
        sum((x - left) / span_x for x, _ in anchors) / len(anchors),
        sum((y - low) / span_y for _, y in anchors) / len(anchors),
    )
    radius = 0.26
    ring = [
        (0.0, 1.0),
        (0.0, -1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-0.7, 0.7),
        (0.7, 0.7),
        (-0.7, -0.7),
        (0.7, -0.7),
    ]
    candidates = [
        (
            min(max(centre[0] + dx * radius, 0.10), 0.90),
            min(max(centre[1] + dy * radius, 0.08), 0.92),
        )
        for dx, dy in ring
    ]
    if avoid_legend:
        # The legend occupies the upper left of the first panel, and a label
        # beneath it reads as another legend entry.
        clear_of_legend = [
            spot for spot in candidates if not (spot[0] < 0.52 and spot[1] > 0.62)
        ]
        candidates = clear_of_legend or candidates

    def clearance(spot: tuple[float, float]) -> float:
        if not drawn:  # pragma: no cover -- a panel with no rows is refused earlier
            return 1.0
        return float(
            min(((spot[0] - px) ** 2 + (spot[1] - py) ** 2) ** 0.5 for px, py in drawn)
        )

    return max(candidates, key=clearance)


@dataclasses.dataclass(frozen=True)
class _Highlight:
    """What one panel is asked to point out: the series, how it is drawn, and
    where to point. Bundled rather than passed as six parameters, because the
    repository's argument-count gate is a real constraint and a record of what
    an annotation IS reads better than a longer signature."""

    contender: str
    style: Any
    label: str
    targets: tuple[float, ...]


def _clearance(axis: Any, text: Any) -> float:
    """Points to shrink every arrow by at the label end, so none crosses it.

    Half the label's bounding-box **diagonal**, plus a pad -- one value for all
    of a label's arrows, independent of where each one points.

    A direction-aware clearance is the obvious idea and it does not work here.
    `shrinkA` is fixed when the annotation is created, but the data-to-display
    transform is not final until the figure is laid out, so the direction the
    clearance was computed for is not the direction that gets drawn. Measured:
    with a per-arrow value the informed panel's 45 deg shaft still began inside
    its own label after `draw()`, because the axes had been resized underneath
    it. The diagonal clears the box from the centre along *any* ray, so it
    cannot be invalidated by a later layout.

    The cost is a slightly generous gap on a vertical arrow. It is small: this
    label is far wider than it is tall, so its half-diagonal and its half-width
    are within a point of each other, and the vertical case is the one the
    previous constant already over-served.
    """
    pad = 2.0
    try:
        renderer = axis.figure.canvas.get_renderer()
        box = text.get_window_extent(renderer)
    except Exception:  # pragma: no cover - backend without a live renderer
        return 9.0
    half_diagonal = (box.width**2 + box.height**2) ** 0.5 / 2.0
    return float(half_diagonal * 72.0 / axis.figure.dpi + pad)


def _point_at(
    axis: Any,
    panel: pd.DataFrame,
    order: list[str],
    highlight: _Highlight,
    *,
    avoid_legend: bool,
) -> None:
    """One label for `contender`, with a leader arrow to each target position.

    `targets` names the x positions to point at; absent, the single position
    where the highlighted curve is FURTHEST from its nearest neighbour is used.
    That fallback exists because a first version anchored at the series' middle
    position on the reasoning that endpoints are where curves converge, and on
    this figure the opposite holds -- the seven curves sit within 0.3 of each
    other at the middle angles and fan out by three units at 45 deg, so the
    arrow landed inside the band it existed to resolve.

    **The head is filled and the shaft carries the series' linestyle.** A dashed
    `->` head is dashed all the way to its tip and stops reading as an
    arrowhead; `-|>` draws a filled triangle, so the shaft still says which
    curve is meant while the head stays a head.

    Raises:
        SpecificationError: If a requested target is not a position the panel
            measured -- an arrow to an angle nobody ran points at nothing.
    """
    positions, values, _ = _series(panel, highlight.contender)
    at = dict(zip(positions, values, strict=True))
    if highlight.targets:
        missing = [target for target in highlight.targets if target not in at]
        if missing:
            raise SpecificationError(
                f"highlight_at names {missing}, which this panel did not "
                f"measure; it holds {positions}"
            )
        anchors = [(target, at[target]) for target in highlight.targets]
    else:
        others = [
            dict(zip(*_series(panel, name)[:2], strict=True))
            for name in order
            if name != highlight.contender
        ]
        separations = [
            min((abs(v - other[x]) for other in others if x in other), default=0.0)
            for x, v in zip(positions, values, strict=True)
        ]
        index = max(range(len(positions)), key=separations.__getitem__)
        anchors = [(positions[index], values[index])]

    spot = _label_spot(axis, panel, order, anchors, avoid_legend)
    text = axis.text(
        spot[0],
        spot[1],
        highlight.label,
        transform=axis.transAxes,
        color=highlight.style.color,
        fontsize="small",
        ha="center",
        va="center",
        zorder=5,
    )
    shrink = _clearance(axis, text)
    for x, y in anchors:
        axis.annotate(
            "",
            xy=(x, y),
            xycoords="data",
            xytext=spot,
            textcoords=axis.transAxes,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-|>",
                # `color` alone: it sets edge AND face, and passing
                # `facecolor` beside it raises "Setting the 'color' property
                # will override the edgecolor or facecolor properties" --
                # which this project's zero-warning pytest gate turns into a
                # failure. The head is filled either way, which is the point.
                "color": highlight.style.color,
                "linestyle": highlight.style.linestyle,
                "linewidth": 0.9,
                # CLEAR THE LABEL'S OWN BOX, not a fixed radius from its
                # centre. `shrinkA` is measured in points from the text anchor,
                # and the anchor is the centre of a centred label -- so a
                # constant 9.0 cleared an arrow leaving vertically and left one
                # leaving sideways starting *inside* its own text. Seen on the
                # informed panel, where the 45 deg arrow crossed the label.
                "shrinkA": shrink,
                "shrinkB": 4.0,
            },
        )


@register_figure("severity_panels")
def severity_panels(context: FigureContext) -> Figure:
    """One quantity against one axis, in several conditions sharing both scales.

    Args:
        context: The concatenated tables, the declaration and the profile.
            `config` declares `panels` (a list of `[panel key, title]` pairs, in
            the order they are drawn), `xlabel`, `ylabel`, the optional
            `draw_spread`, the optional `highlight` and the optional
            `highlight_at`.

    Returns:
        The rendered figure.

    Raises:
        SpecificationError: If a required column or a declared panel is
            missing, if a panel lacks a contender its neighbours draw — six
            curves beside seven, unexplained, is worse than a refusal — or if
            `highlight` names a contender the figure does not draw.
    """
    table = context.table
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise SpecificationError(
            f"figure {context.figure_id!r} of kind 'severity_panels' needs "
            f"columns {missing} that its table does not carry"
        )
    config: dict[str, Any] = dict(context.config)
    panels = [tuple(pair) for pair in config.get("panels", ())]
    if not panels:
        raise SpecificationError(
            f"figure {context.figure_id!r} declares no panels; a panelled "
            "figure with nothing to panel is a figure with no subject"
        )
    order = list(context.series_order)
    styles = encode_series(order, dict(context.roles))
    draw_spread = bool(config.get("draw_spread", False))
    highlight = config.get("highlight") or None
    highlight_at = [float(value) for value in config.get("highlight_at", ()) or ()]
    if highlight is not None and highlight not in order:
        raise SpecificationError(
            f"figure {context.figure_id!r} declares highlight {highlight!r}, "
            f"which is not one of the series it draws ({order}); an arrow "
            "pointing at a curve that is not there is worse than none"
        )

    width, height = context.profile.figsize
    # The profile is opened around the WHOLE draw, not just the save.
    # A text object takes its family when it is created, so a figure drawn
    # outside these settings is in the wrong face however it is written
    # afterwards. `axis_scaling` and `grouped_bars` were already written
    # this way; this renderer was not, and Figure 2 -- the one figure it
    # draws -- reached the manuscript in matplotlib's ambient sans-serif.
    with profile_context(context.profile):
        figure, axes = plt.subplots(
            1, len(panels), figsize=(width, height * 0.85), sharex=True, sharey=True
        )
        axes = list(axes) if len(panels) > 1 else [axes]
        for axis, (key, title) in zip(axes, panels, strict=True):
            panel = table[table[PANEL_COLUMN] == key]
            if panel.empty:
                raise SpecificationError(
                    f"figure {context.figure_id!r} declares panel {key!r}, and the "
                    "table has no rows for it; an empty panel reads as 'measured, "
                    "and flat'"
                )
            for contender in order:
                positions, values, spreads = _series(panel, contender)
                if not positions:
                    raise SpecificationError(
                        f"panel {key!r} of {context.figure_id!r} has no rows for "
                        f"{contender!r}; it would drop a series its neighbours draw"
                    )
                style = styles[contender]
                axis.errorbar(
                    positions,
                    values,
                    yerr=spreads if draw_spread else None,
                    label=context.display_names.get(contender, contender),
                    color=style.color,
                    linestyle=style.linestyle,
                    marker=style.marker,
                    markersize=3.0,
                    linewidth=1.0,
                    capsize=1.5 if draw_spread else 0.0,
                    elinewidth=0.6,
                )
            axis.set_title(str(title))
            axis.grid(True, linewidth=0.3, alpha=0.4)
            # Ticks AT the measured positions, not at matplotlib's round numbers:
            # the sampling is deliberately uneven and evenly spaced ticks would
            # tell the reader it was uniform.
            axis.set_xticks(sorted({float(value) for value in panel["axis_label"]}))
            if highlight is not None:
                _point_at(
                    axis,
                    panel,
                    order,
                    _Highlight(
                        contender=highlight,
                        style=styles[highlight],
                        label=context.display_names.get(highlight, highlight),
                        targets=tuple(highlight_at),
                    ),
                    avoid_legend=axis is axes[0],
                )
        axes[0].set_ylabel(str(config.get("ylabel", "")))
        axes[0].legend(loc="upper left", frameon=False, handlelength=2.4)
        # ONE x label for panels that share the axis, as there is one y label for
        # panels that share that: they differ in condition, not in what x means.
        figure.supxlabel(str(config.get("xlabel", "")))
        figure.tight_layout()
        return figure
