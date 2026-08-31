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
"""

from __future__ import annotations

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


@register_figure("severity_panels")
def severity_panels(context: FigureContext) -> Figure:
    """One quantity against one axis, in several conditions sharing both scales.

    Args:
        context: The concatenated tables, the declaration and the profile.
            `config` declares `panels` (a list of `[panel key, title]` pairs, in
            the order they are drawn), `xlabel`, `ylabel` and the optional
            `draw_spread`.

    Returns:
        The rendered figure.

    Raises:
        SpecificationError: If a required column or a declared panel is
            missing, or if a panel lacks a contender its neighbours draw — six
            curves beside seven, unexplained, is worse than a refusal.
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
        axes[0].set_ylabel(str(config.get("ylabel", "")))
        axes[0].legend(loc="upper left", frameon=False, handlelength=2.4)
        # ONE x label for panels that share the axis, as there is one y label for
        # panels that share that: they differ in condition, not in what x means.
        figure.supxlabel(str(config.get("xlabel", "")))
        figure.tight_layout()
        return figure
