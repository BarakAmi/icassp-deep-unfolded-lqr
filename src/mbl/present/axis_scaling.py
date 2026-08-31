"""`axis_scaling` — an aggregate metric against a swept axis (Annex 03 §B.6).

The slice's figure: cost versus unrolling depth, with the study's bound and
baselines bracketing the contenders. It renders `cost_vs_axis`'s tidy table and
computes nothing.

**Why this draws rather than calling `viz.plots.plot_cost_vs_unfolding_depth`.**
That function is a good renderer for the shape it was written against and the
wrong one here, for three reasons recorded in the slice plan:

1. Its signature separates `curves` from `reference_lines`, which is exactly
   the hand-maintained dictionary the tidy table's `role` column replaces —
   passing one to the other would rebuild the split the new stack removed.
2. It assigns marker, linestyle and colour by **enumeration index**, so
   dropping a series repaints the survivors (§B.3.1 forbids it).
3. It hardcodes `figsize=(11, 6.5)`, so it cannot be profile-independent
   (§B.2).

What §B.6 says is retained is `src/mbl/viz/{style,plots,landscape}` as "the
rendering backend these specs drive", and `viz.style` is driven here in full:
the palette, the registered series colours, the reference-line dash cycle and
the legend placement are all its own. It is one function in `viz.plots` that
does not fit, not the layer.

**§B.5's axis rule is the figure's one real obligation.** The axis is scaled to
the data under study, not to the reference lines: a bound far outside the
contenders' range is drawn clipped with its value in the legend, rather than
compressing every curve into a band. NB04's own numbers are why — a 0.71 %
contender band inside a 1.23–3.87 axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, NullLocator

from ..spec.contender import Role
from ..spec.errors import SpecificationError
from ..viz.style.theme import LegendStyle, place_legend_outside
from .encodings import SeriesStyle, encode_series, legend_label
from .margin_labels import label_levels
from .profiles import profile_context
from .registry import FigureContext, register_figure
from .selection import select_series

#: Columns `cost_vs_axis` emits that this renderer reads. Named so a schema
#: change is refused here rather than surfacing as an empty figure.
REQUIRED_COLUMNS = (
    "contender",
    "role",
    "axis_value",
    "aggregate",
    "interval_low",
    "interval_high",
)

#: A level-only panel's span, as a fraction of the widest panel's. Small,
#: because a flat level has no extent and clause 3 makes height proportional to
#: span — so any span invented for it is height taken from the panels that
#: carry the claim. Measured need: ICASSP Figure 1's top panel took ~0.5 of
#: span for one line at 9.127 and dwarfed the cluster at 8.26-8.36.
LEVEL_PANEL_SPAN = 0.06

#: A level-only panel's height, as a fraction of the **combined height of all
#: the panels** — the floor §B.5.1 clause 3 gives it in place of the
#: proportional rule the rest of them obey.
#:
#: Of the panels' own region rather than of the figure, because that is the
#: quantity being divided up: the title, the axis labels and the legend have
#: already taken what they need by the time this is applied, and a floor
#: measured against the whole figure would shrink or grow with them.
#:
#: **It needs a floor because its span is invented.** `LEVEL_PANEL_SPAN` above
#: makes the span small so the panel does not dwarf the one the break was made
#: for, and proportional height then makes the panel small in the same measure:
#: measured on ICASSP Figure 1, **6.2 pixels tall with zero ticks**, its line
#: 3.1 px from a frame in either direction. A reader can neither find the line
#: nor read its value, and — the author's own complaint — cannot see that the
#: axis is broken at all, because a break is legible as a jump between the
#: numbers on either side of it and a panel with no numbers contributes none.
#:
#: Exempting these panels costs nothing clause 3 protects: it exists so one
#: slope is never drawn at two angles, and a panel of horizontal lines has no
#: slope. 0.09 of the panel region leaves room for a tick label plus the line
#: clear of both spines at every profile.
LEVEL_PANEL_HEIGHT = 0.09

#: Fractional padding above and below the plotted range.
#:
#: **Raised from matplotlib's own 0.05 default on 2026-08-07** (§B.5.1 clause
#: 7). Matching the default made an axis read here look like an autoscaled one,
#: which was the whole argument for it -- but a broken figure is not an
#: ordinary axis: the break mark sits on the very edge the padding is measured
#: from, so a series near the end of its panel is drawn onto the mark
#: announcing the gap. Measured on ICASSP Figure 1 at 0.05: the nearest curve
#: 5.0 px from a frame, and the level panels' lines 3.1 px. At print size, and
#: after colour removal, a line that close to a spine is read as part of it.
Y_MARGIN = 0.10

#: Where a legend that fits in no panel is anchored, and the margin
#: `tight_layout` reserves for it. §B.5, 2026-08-07: a legend never covers a
#: mark, and on a broken axis there may be no panel it fits in -- measured on
#: ICASSP Figure 1, 108 px of legend against 30 px of free space in the
#: roomiest of four panels.
#: The anchor is the TOP edge of the legend (`loc="upper center"`), so it must
#: sit at the bottom of the band `tight_layout` reserved -- not near zero.
#: Anchored at 0.02 the legend hung off the bottom of the canvas and
#: `bbox_inches="tight"` expanded the page to hold it, leaving a band of white
#: between the axis label and the key about as tall as the figure.
#: How opaque the legend's own box is. §B.5, 2026-08-07: the key sits inside
#: the axes and a mark behind it must be DIMMED rather than deleted -- a reader
#: has to be able to see that a curve passes behind it. Low enough to see
#: through, high enough that the legend text stays the most legible thing in
#: its own box.
LEGEND_ALPHA = 0.78

#: Inside-legend anchors, tried in order (§B.5, 2026-08-13). `upper center`
#: first because it can only ever cover the middle of the swept axis, where a
#: cost-versus-depth figure makes no claim; the corners next, because giving up
#: on an inside placement costs a quarter of the figure and a corner that is
#: measured clear costs nothing. Lower corners are absent deliberately: a
#: converging figure's curves land at the bottom, which is exactly the claim.
LEGEND_INSIDE_ANCHORS = ("upper center", "upper right", "upper left")

LEGEND_BELOW_ANCHOR = 0.13
LEGEND_BELOW_RECT = (0.0, 0.15, 1.0, 1.0)

#: Opacity of an across-seed interval band. Low enough that two overlapping
#: bands stay readable, which is the case a five-seed study produces.
BAND_ALPHA = 0.18

#: The column §A.3.2 makes the figure's dispersion channel. Optional on the
#: frame: an artifact rendered before §A.3.2 existed does not carry it, and
#: §B.1.1 requires `mbl figure rebuild` to reproduce such a figure rather than
#: refuse it or silently restyle it.
SPREAD_COLUMN = "across_seed_spread"

#: Half-width of an error bar's caps, in points. §A.3.2 says "with caps" and
#: means it: a bare segment reads as part of the line it hangs off.
CAPSIZE = 2.5

#: §A.3.2 rule 1 — the bar is NAMED where it is drawn. "A reader who assumes a
#: 95 % interval where one standard deviation was drawn misreads the result by
#: a factor nothing in the figure lets them recover." It goes in the legend's
#: title, which is the one piece of figure text that is neither a series name
#: nor an axis name, and it appears only when a bar was actually drawn.
DISPERSION_LEGEND_TITLE = "bars: across-seed spread"

#: Every key this renderer reads. Anything else in a figure's `config` is
#: **refused**, because before this existed `xscale`, `x_scale`, `log_x`,
#: `logx`, `scale`, `axis_value` and an outright `typo_key` were all accepted
#: in silence and all did nothing — a figure declaring any of them rendered a
#: byte-identical PNG. A key that parses, reaches nothing and says nothing is
#: the defect class this project has shipped six times.
#:
#: Kept beside the code that reads them and checked against it by a test that
#: greps this module, because a hand-written list and its call sites drift.
#: §B.5's "data under study": the roles that SET the axis, as opposed to the
#: ones drawn clipped against it. Stated as roles rather than inferred from a
#: null axis value, because a depth-invariant CONTENDER is still data — see
#: `_apply_axis_rule`.
_DATA_UNDER_STUDY = frozenset({Role.CONTENDER.value, Role.BASELINE.value})

CONFIG_KEYS = frozenset(
    {
        "series",
        "xlabel",
        "ylabel",
        "title",
        "ylim",
        "ybreak",
        "ypanel_weights",
        "xscale",
        "dispersion",
    }
)

#: What §A.3.2 rule 4 lets a figure declare as its dispersion channel.
#:
#: `"none"` is permitted only where the figure's table carries the spread, and
#: that pairing is enforced on `FigureSpec` -- which sees both halves, where
#: this renderer is handed the config without the table.
DISPERSION_CHANNELS = ("bar", "none")

#: What `xscale` may say. §B.5.1: the *value* axis stays linear and takes a
#: break; a swept axis that genuinely spans decades takes a log scale instead —
#: the ICASSP campaign has two, the ablation's widths (8…256) and Figure 3's
#: state dimensions (4…50), and base 2 is the ablation's own spacing.
X_SCALES: dict[str, dict[str, Any]] = {
    "linear": {"value": "linear"},
    "log": {"value": "log"},
    "log2": {"value": "log", "base": 2},
}


@register_figure("axis_scaling")
def axis_scaling(context: FigureContext) -> Figure:
    """Aggregate metric against a swept axis, with intervals and bounds.

    Args:
        context: The tidy table, the declaration and the style profile.

    Returns:
        The rendered figure. The caller owns it — the greyscale gate must be
        able to inspect it before anything is written.

    Raises:
        SpecificationError: If the table is not `cost_vs_axis`'s, or if the
            configuration selects a contender the table does not carry.
    """
    table = _validated(context.table)
    _require_known_config(context.config)
    # Idempotent: the RUNNER has already selected before freezing the frame
    # (§B.1.2 clause 2). Kept here so a renderer invoked directly — a test,
    # a notebook — honours the same declaration, and it is one function
    # rather than two readings of one config key.
    selected = select_series(table, context.config)
    styles = encode_series(
        context.series_order or sorted(set(table["contender"])), context.roles
    )

    # Decided ONCE for the whole figure, not per series. Found by rendering the
    # tracked study rather than a fixture: `standard_pgd` is analytic and its
    # across-seed spread is exactly 0.0 at every depth, so a per-series rule
    # gave it a shaded BCa band while three other series carried standard-
    # deviation bars — two dispersion channels on one figure, with the legend
    # naming only one of them. That is the misreading §A.3.2 rule 1 exists to
    # prevent, arrived at by accident.
    channel = _dispersion_channel(context.config)
    # NOT `channel == "bar" and ...`: mutation testing showed that conjunct is
    # redundant, because `_draw_curve` and `_draw_flat` each return before
    # drawing anything when the channel is "none". Two places deciding one
    # thing is how they come to disagree.
    draws_bars = _figure_draws_bars(selected)

    # DECLARED, never derived (§B.5.1 clause 4): "a tool may report the widest
    # empty range to a human choosing; it may not choose". Validated against
    # the data before a single artist exists, because a break drawn through a
    # mark is refused rather than rendered.
    omitted = _declared_breaks(selected, context.config)

    with profile_context(context.profile):
        panels = _panels(omitted, selected, context.config)
        drew_a_bar = False
        # contender -> the label the ARTIST carries. The legend filter must
        # match on this and not on the contender key: `legend_label` appends a
        # reference's value ("riccati_unconstrained (0.05)"), so filtering by
        # key silently dropped every series whose label is decorated -- which
        # is exactly the clipped bound §B.3.1 requires to keep its row.
        artist_labels: dict[str, str] = {}
        for label in _draw_order(selected, context.series_order):
            series = _Series(
                label=label,
                rows=selected[selected["contender"] == label],
                style=styles.get(label, SeriesStyle("black", "-", "o")),
                role=context.roles.get(label, Role.CONTENDER),
                display_names=context.display_names,
            )
            # Every series is drawn on EVERY panel. A break removes a range,
            # not a series: drawing the curves only above and the bound only
            # below would be two axes sharing an x, which is exactly what
            # §B.5.1 exists to distinguish itself from — and it would take the
            # lower panel out of the greyscale gate's reach entirely.
            artist_labels[label] = series.legend_text(
                None
                if not bool(series.rows["axis_value"].isna().all())
                else float(np.asarray(series.rows["aggregate"], dtype=np.float64)[0])
            )
            for axes in panels.axes:
                drawn = (
                    _draw_flat(axes, series, channel=channel)
                    if bool(series.rows["axis_value"].isna().all())
                    else _draw_curve(axes, series, channel=channel, bars=draws_bars)
                )
                drew_a_bar |= drawn

        panels.apply_x_scale(selected, context.config)
        panels.apply_axis_rule(selected, context.config)
        panels.label(
            figure_x=str(context.config.get("xlabel", "axis value")),
            figure_y=str(context.config.get("ylabel", "aggregate cost")),
            title=context.config.get("title"),
        )
        figure = panels.figure
        # §B.5: a FLAT series is labelled at the right margin, not in the
        # legend. The legend is built from the curves alone, so a figure whose
        # series are mostly levels no longer spends half its canvas on a key.
        # A CLIPPED flat series keeps its legend row. It has no height on the
        # canvas to be labelled at, so the margin cannot serve it -- and
        # §B.3.1 requires a bound outside the axis to state its value in the
        # legend. Without this a clipped reference took neither, and vanished
        # from the figure's key entirely; three existing tests caught it.
        legend_rows = _curve_labels(selected, context.series_order) + tuple(
            panels.clipped_flat_series(selected)
        )
        # INSIDE, unconditionally -- §B.5's own words: the legend leaves the
        # figure only when it cannot fit inside at all, "and then it goes
        # BELOW rather than beside". This used to be gated on the right margin
        # being spoken for by flat-series labels, which sent an all-curves
        # figure's legend beside the axes; measured on ICASSP Figure 3 -- the
        # first tracked figure with no flat series -- the axes kept 45 % of
        # the canvas. `settle_legend` still relegates a legend larger than
        # every panel, which is the one case the annex sends below.
        panels.place_legend(
            only=[artist_labels[key] for key in legend_rows if key in artist_labels],
        )
        panels.finalise()
        panels.settle_legend(
            only=[artist_labels[key] for key in legend_rows if key in artist_labels]
        )
        panels.label_flat_series(selected, styles, context.display_names)
        if drew_a_bar:
            # §A.3.2 rule 1, and only when there is ink to name: a caption
            # naming a quantity the figure does not show is a claim about
            # marks that are not on the page.
            legend = panels.legend_axes.get_legend()
            if legend is not None:
                legend.set_title(DISPERSION_LEGEND_TITLE)
    return figure


#: Half-length of a break mark, in axes-fraction units, and its slope. Drawn on
#: both panels' facing edges so the discontinuity is "impossible to miss"
#: (§B.5.1 clause 3) rather than merely implied by a hidden spine.
BREAK_MARK_SIZE = 0.012
BREAK_MARK_SLOPE = 2.0


@dataclass
class _Panels:
    """The figure's value axis: one `Axes`, or N + 1 with N ranges removed.

    One type for both cases, so the drawing loop has no idea which it is in.
    The alternative — a broken branch beside an unbroken one — is how a second
    panel comes to be drawn by code no test of the first panel covers.

    Attributes:
        figure: The owning figure.
        axes: Every panel, TOP FIRST. Length is `len(omitted) + 1`.
        omitted: The removed ranges, ascending; empty for an unbroken axis.
    """

    figure: Figure
    axes: tuple[Axes, ...]
    omitted: tuple[tuple[float, float], ...]
    #: §B.5.1 clause 3 as amended 2026-08-10: the declared per-panel weight on
    #: the proportional share, top first, or `()` for the unweighted rule.
    #: Validated at parse (`_declared_panel_weights`), applied in
    #: `_make_proportional`.
    weights: tuple[float, ...] = ()
    #: `id()` of every panel holding nothing but flat levels. These are exempt
    #: from clause 3's proportional height and take `LEVEL_PANEL_HEIGHT`
    #: instead. Recorded when the panel is thinned rather than recomputed,
    #: because the condition is "this panel's values had no extent" and that is
    #: known exactly once.
    _level_panels: set[int] = field(default_factory=set)
    #: Whether the legend was placed INSIDE the axes. `settle_legend` may only
    #: relegate one that was: an OUTSIDE legend is anchored beyond the panel's
    #: right edge and so is never "contained" in it, so a containment test
    #: applied to it relegates every figure that has no flat series at all.
    _legend_inside: bool = False

    @property
    def legend_axes(self) -> Axes:
        """Where the one legend goes.

        The TALLEST panel, which is the one holding the curves.

        It used to be `axes[0]`, on the reasoning that the upper panel holds
        the curves and a legend beside the lower one would read as belonging to
        the references. With several breaks that stopped being true: Figure 1's
        top panel is a thin level, and a legend anchored at its centre-left
        rode up past the figure's top edge and printed over the title. Height
        is the property the reasoning was really about.

        Measured by DATA SPAN and not by `get_position().height`, which is the
        obvious spelling and is wrong: heights are assigned in `finalise`,
        which runs AFTER the legend is placed, so at this point every panel
        still carries matplotlib's default and `max` simply returned the first
        -- the thin one. The legend then drew across the panel above it and
        came out cut in half. Span is proportional to height by clause 3 and is
        already final here, so it says the same thing at a moment when it is
        true.
        """
        return max(self.axes, key=lambda axes: axes.get_ylim()[1] - axes.get_ylim()[0])

    def place_legend(self, *, only: Sequence[str]) -> None:
        """The one legend, inside the axes (§B.5, 2026-08-07; corrected
        2026-08-09).

        Top centre of the panel with the most free vertical space, drawn
        translucent; it leaves the axes only if it does not FIT there, and
        then it goes BELOW (`settle_legend`) — never beside, which is the
        annex's own wording. The first implementation gated inside-placement
        on the right margin being spoken for by §B.5's margin labels and sent
        an all-curves figure's legend beside the axes; measured on ICASSP
        Figure 3, the axes kept 45 % of the canvas.

        **Top centre and translucent are what make an inside legend acceptable,
        and both are load-bearing.** Placed at the top centre the most it can
        cover is the middle of the swept axis, never the ends — and on a
        cost-versus-depth figure the ends are the claim. Drawn at
        `LEGEND_ALPHA` a curve behind it is dimmed rather than deleted.

        **The alternative was measured and is worse.** An earlier version moved
        the legend below the figure whenever it covered any vertex at all. That
        is a quarter of the plot: the axes take 59.5 % of the height with the
        legend below and 74.5 % with it inside, and a figure whose subject is a
        convergence of 0.0686 % cannot spend that on a key. Below survives only
        for the case transparency cannot help — a legend larger than every
        panel.

        **Anchors are tried in order, and the order is the preference above
        made falsifiable** (2026-08-13, on the author's report of Figure 3).
        `upper center` remains first for the reason stated: it can only ever
        cover the middle of the swept axis. But a panel can be tall enough for
        a legend and still refuse one there — Figure 3's comparison panel is
        52.6 % of the canvas and the key went below it anyway — and the cost of
        giving up is a quarter of the figure, which for a page-limited paper is
        the most expensive thing this renderer does. So the corner anchors are
        tried before the figure's height is spent, and each is kept only if the
        box lands inside its panel AND covers no plotted vertex. A corner that
        covers a mark is not accepted merely because it fits.
        """
        self._legend_inside = True
        place_legend_outside(
            self.figure,
            self.roomiest,
            only=list(only),
            inside=True,
            style=LegendStyle(loc=LEGEND_INSIDE_ANCHORS[0], framealpha=LEGEND_ALPHA),
        )

    def settle_legend(self, *, only: Sequence[str]) -> None:
        """Move the legend below if it does not fit — AFTER `finalise`.

        **The ordering is the whole point, and getting it wrong is a defect
        this renderer has now shipped twice.** Panel heights are assigned by
        `_make_proportional` inside `finalise`, which runs after
        `place_legend`, because the proportional step needs the positions
        `tight_layout` computes and `tight_layout` runs when the legend is
        placed. So at placement time every panel still carries matplotlib's
        equal split: measured on ICASSP Figure 1 at `ieee-2col`, four panels of
        90 px each against a legend of 94 px, which reported "does not fit" and
        sent the legend below a panel that is 133 px once the heights are real.

        The same mistake in its first form is recorded on `legend_axes`, which
        chose the tallest panel by `get_position().height` before any height
        had been assigned. Asking the question twice — once for placement, once
        here for fit — is what keeps each answer at a moment when it is true.
        """
        if not self._legend_inside:
            return
        axes = next((panel for panel in self.axes if panel.get_legend()), None)
        if axes is None or not self._legend_is_misplaced(axes):
            return
        # The centre does not fit. Before spending a quarter of the canvas,
        # try the corners -- HERE and not in `place_legend`, because this is
        # the first moment the geometry is true. Placed earlier, the choice is
        # made against matplotlib's equal split and a corner that looked clear
        # covers a vertex once the panels are proportioned; measured as four
        # failing tests the first time this was written one method too soon.
        for anchor in LEGEND_INSIDE_ANCHORS[1:]:
            place_legend_outside(
                self.figure,
                axes,
                only=list(only),
                inside=True,
                style=LegendStyle(loc=anchor, framealpha=LEGEND_ALPHA),
            )
            self.figure.canvas.draw()
            if not self._legend_is_misplaced(axes) and self._legend_clears_the_ends(
                axes
            ):
                return
        self.relegate_legend(only=only)

    def relegate_legend(self, *, only: Sequence[str]) -> None:
        """Move the legend out of the axes and below the figure.

        The last resort of §B.5's placement rule, and the one case transparency
        cannot help: a legend larger than every panel is not made to fit by
        being seen through. It costs the axes about a quarter of their height,
        which is why it is reached by measurement rather than by preference.
        """
        axes = next((panel for panel in self.axes if panel.get_legend()), None)
        if axes is None:
            return
        legend = axes.get_legend()
        if legend is not None:
            legend.remove()
        # Below, spanning the figure, in as few rows as the entries allow. It
        # is anchored to the FIGURE and not to an axes: on a broken figure an
        # axes-anchored legend below the bottom panel would be positioned
        # against whichever panel happened to be last.
        handles, labels = axes.get_legend_handles_labels()
        chosen = [
            (h, s) for h, s in zip(handles, labels, strict=True) if s in set(only)
        ]
        if chosen:
            self.figure.legend(
                handles=[h for h, _ in chosen],
                labels=[s for _, s in chosen],
                loc="upper center",
                bbox_to_anchor=(0.5, LEGEND_BELOW_ANCHOR),
                ncol=min(2, len(chosen)),
                frameon=True,
            )
        self.figure.tight_layout(rect=LEGEND_BELOW_RECT)
        # `tight_layout` recomputes every position from scratch, so the
        # proportional heights `finalise` had just assigned are gone. Clause 3
        # is not optional on the way out of a placement decision.
        if self.omitted:
            self._make_proportional()

    @property
    def roomiest(self) -> Axes:
        """The panel with the most vertical space no series is drawn in.

        **A level-only panel is never a candidate**, however empty it is — and
        it always reports as entirely empty, which is exactly the trap. Its
        height is a floor chosen to hold one horizontal line clear of its own
        frame (§B.5.1 clause 3); a key placed in it would make the panel about
        the key. Measured before this exclusion: on a four-panel figure the
        legend went to the 30 px top panel, because "free height" of a panel
        with nothing in it is all of it.
        """
        candidates = [axes for axes in self.axes if id(axes) not in self._level_panels]
        return max(candidates or list(self.axes), key=self._free_height)

    def _free_height(self, axes: Axes) -> float:
        """Free height a panel WILL have, in the units clause 3 shares by.

        **Not `get_position().height`, which is the trap this measure fell into
        twice.** Heights are assigned in `finalise`, which runs after the legend
        is placed, so at placement time every panel still carries matplotlib's
        equal split — the same number for all of them. Ranking by
        `position.height × free fraction` therefore ranks by the FRACTION
        alone, and a thin, nearly-empty panel wins on being empty.

        Measured on ICASSP Figure 3 (2026-08-13): the top panel holds two
        points and reads 0.37 free against the comparison panel's 0.16, so the
        legend was placed there, fitted a 98 px box into the 132 px every panel
        then had — and was relegated below the figure once proportioning made
        that panel **47 px**. A quarter of the canvas spent on a key because
        the roomiest panel was measured before rooms were assigned.

        Clause 3 shares the ranged height by `span × weight`, and both are
        known here, so that is what the free fraction is scaled by. The
        docstring on `roomiest` already records this trap in its level-panel
        form; this is the same defect one step along, and the fix is to stop
        asking the question in units that are not yet true.
        """
        low, high = axes.get_ylim()
        span = high - low
        if span <= 0:
            return 0.0
        index = next(
            (position for position, panel in enumerate(self.axes) if panel is axes),
            0,
        )
        share = span * (self.weights[index] if self.weights else 1.0)
        drawn: list[float] = []
        for line in axes.get_lines():
            if str(line.get_label()).startswith("_"):
                continue
            ydata = np.asarray(line.get_ydata(), dtype=np.float64)
            drawn.extend(
                float(value) for value in ydata[(ydata >= low) & (ydata <= high)]
            )
        if not drawn:
            return share
        occupied = (max(drawn) - min(drawn)) / span
        return share * (1.0 - occupied)

    def _legend_clears_the_ends(self, axes: Axes) -> bool:
        """Whether the placed legend covers no series' FIRST or LAST vertex.

        §B.5's argument for the top centre is precisely that it "can only ever
        cover the middle of the swept axis, never the ends — and on a
        cost-versus-depth figure the ends are the claim". A corner has no such
        guarantee, so it must earn one by measurement: covering the middle is
        the accepted cost of an inside legend, covering an end is not.

        **Only the ends, deliberately.** A test that refused any overlap would
        send every crowded figure below, which is the placement this whole rule
        exists to avoid, and the legend is translucent so a curve behind it is
        dimmed rather than deleted.
        """
        legend = axes.get_legend()
        if legend is None:
            return True
        box = legend.get_window_extent()
        for line in axes.get_lines():
            if str(line.get_label()).startswith("_"):
                continue
            x_data = np.asarray(line.get_xdata(), dtype=np.float64)
            y_data = np.asarray(line.get_ydata(), dtype=np.float64)
            if x_data.size == 0:
                continue
            ends = axes.transData.transform(
                np.column_stack([x_data[[0, -1]], y_data[[0, -1]]])
            )
            if np.any(
                (ends[:, 0] >= box.x0)
                & (ends[:, 0] <= box.x1)
                & (ends[:, 1] >= box.y0)
                & (ends[:, 1] <= box.y1)
            ):
                return False
        return True

    def _legend_is_misplaced(self, axes: Axes) -> bool:
        """Whether the placed legend does not FIT inside its panel.

        Overlap is no longer the question (§B.5, corrected 2026-08-07): the
        legend sits at the top centre and is translucent, so covering the
        middle of the swept axis is the accepted cost of not spending a quarter
        of the figure's height on a key. What remains disqualifying is a legend
        that does not fit, because there is no transparency that fixes a box
        drawn outside the panel it belongs to.

        Found by looking, and the reason containment is asked separately from
        overlap: `roomiest` reports a level panel as entirely free, which is
        true and useless — it is free because nothing is drawn in it and it is
        30 px tall. The legend went there, covered no plotted vertex, and
        spilled across the figure's title.
        """
        legend = axes.get_legend()
        if legend is None:
            return False
        self.figure.canvas.draw()
        box = legend.get_window_extent()
        frame = axes.get_window_extent()
        return not (
            frame.x0 <= box.x0
            and box.x1 <= frame.x1
            and frame.y0 <= box.y0
            and box.y1 <= frame.y1
        )

    def apply_x_scale(self, table: pd.DataFrame, config: Mapping[str, Any]) -> None:
        """The declared x-scale, on **every** panel.

        Applied to all of them rather than to the first: a broken figure shares
        one x axis (§B.5.1 clause 1), and a scale set on the upper panel alone
        would leave two different x axes under one shared label.

        Raises:
            SpecificationError: On an unknown scale, or on a log scale over an
                axis value that is not positive. matplotlib does not refuse the
                second — it silently drops the point and draws the rest, which
                is a figure missing a mark its own table still carries
                (§B.1.2 clause 2).
        """
        declared = str(config.get("xscale", "linear"))
        if declared not in X_SCALES:
            raise SpecificationError(
                f"xscale {declared!r} is not one of {', '.join(sorted(X_SCALES))}; "
                "§B.5.1 keeps the VALUE axis linear and gives it a break, and a "
                "swept axis that genuinely spans decades takes a log scale"
            )
        if declared == "linear":
            return
        values = np.asarray(table["axis_value"], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.min()) <= 0.0:
            offenders = sorted({float(value) for value in finite if value <= 0.0})
            raise SpecificationError(
                f"xscale {declared!r} cannot show axis value(s) "
                f"{', '.join(f'{value:g}' for value in offenders)}; a log axis "
                "has no place for zero or a negative, and matplotlib drops such "
                "a point silently rather than refusing it — leaving a figure "
                "missing a mark its own table still carries"
            )
        settings = dict(X_SCALES[declared])
        name = str(settings.pop("value"))
        # Ticks AT the plotted values, not at matplotlib's decades. Its
        # `LogLocator` thins a six-value base-2 axis to 2^4, 2^6, 2^8 — so
        # three of six marks sit at no tick at all, and a reader cannot say
        # what width they are looking at. That is the ruler problem §B.1.2
        # names, reintroduced by the axis rather than by the table.
        #
        # If a figure ever has too many values for a tick each, the answer is
        # fewer marks rather than fewer ticks: a mark a reader cannot put a
        # number to is a mark the figure did not really make.
        #
        # Applied to every panel, and mutation testing shows that is currently
        # REDUNDANT rather than wrong: with `sharex=True` a scale and a tick
        # set written to the upper panel propagate to the lower one (measured;
        # without `sharex` they do not). The loop is kept because it states
        # this method's contract — §B.5.1 clause 1's one x axis — locally,
        # rather than making it depend on a construction detail two functions
        # away. The linkage that makes it redundant is itself asserted, by
        # `test_the_panels_are_linked_and_not_merely_equal`.
        ticks = sorted({float(value) for value in finite})
        for axes in self.axes:
            axes.set_xscale(name, **settings)
            axes.set_xticks(ticks)
            axes.set_xticklabels([f"{value:g}" for value in ticks])
            axes.xaxis.set_minor_locator(NullLocator())

    def apply_axis_rule(self, table: pd.DataFrame, config: Mapping[str, Any]) -> None:
        """§B.5's axis rule, or §B.5.1's inversion of it."""
        if not self.omitted:
            _apply_axis_rule(self.axes[0], table, config)
            return
        values = np.asarray(table["aggregate"], dtype=np.float64)
        for index, axes in enumerate(self.axes):
            floor, ceiling = self._band(index)
            inside = table[(values >= floor) & (values <= ceiling)]
            # Clause 5, GENERALISED. It used to read "the lower panel", which
            # was the two-panel spelling of the real rule: a panel holding no
            # data under study has nothing to scale to, so `_apply_axis_rule`
            # returns early and matplotlib's autoscale silently decides the
            # axis instead — measured once at a span of 0.214 where the
            # references occupied 5e-05. The condition is therefore the
            # ABSENCE OF DATA UNDER STUDY, not the position of the panel: with
            # several breaks a middle panel can hold nothing but a bound, and
            # the bottom one can hold a baseline that must scale normally.
            holds_data = bool(inside["role"].isin(_DATA_UNDER_STUDY).any())
            _apply_axis_rule(axes, inside, config, include_flat=not holds_data)
            low, high = axes.get_ylim()
            axes.set_ylim(
                bottom=low if floor == -np.inf else max(floor, low),
                top=high if ceiling == np.inf else min(ceiling, high),
            )
        if config.get("ylim") is None:
            self._thin_the_level_panels(table)

    def _thin_the_level_panels(self, table: pd.DataFrame) -> None:
        """A panel holding one flat level gets a THIN band, not a padded one.

        Clause 3's proportional heights turn against the figure otherwise, and
        this was measured on ICASSP Figure 1 rather than reasoned about. Its
        top panel holds a single baseline at 9.127; with nothing to take a
        range from, matplotlib padded it to a span of ~0.5 — and because height
        is proportional to span, **the panel with one flat line took more of
        the figure than the panel the break was made to show**, which came out
        a sliver with its tick labels overlapping.

        A level has no extent, so any span drawn for it is a rendering choice
        rather than a property of the data. The choice made here is the
        smallest one that still separates the level from its own frame: a fixed
        fraction of the widest panel's span, so it stays proportional to the
        figure it is in and does not depend on the level's magnitude.

        Clause 3 is not weakened. Height stays proportional to span in every
        panel, so one unit of y is still the same number of pixels everywhere —
        which is exactly why a panel with no extent must not claim one.
        """
        values = np.asarray(table["aggregate"], dtype=np.float64)
        spans: list[float] = []
        for index in range(len(self.axes)):
            floor, ceiling = self._band(index)
            inside = values[(values >= floor) & (values <= ceiling)]
            spans.append(float(inside.max() - inside.min()) if inside.size else 0.0)
        widest = max(spans, default=0.0)
        if widest <= 0.0:
            return
        for axes, span, index in zip(self.axes, spans, range(len(spans)), strict=True):
            if span > 0.0:
                continue
            floor, ceiling = self._band(index)
            inside = values[(values >= floor) & (values <= ceiling)]
            if not inside.size:
                continue
            centre = float(inside[0])
            half = 0.5 * LEVEL_PANEL_SPAN * widest
            axes.set_ylim(centre - half, centre + half)
            # ONE tick, at the level itself. Removing them entirely was the
            # over-correction: three ticks crowded into a band this thin did
            # print on top of each other, but with none at all the panel
            # contributes no number, and a break is legible precisely as a
            # jump between the numbers on either side of it (§B.5.1 clause 3,
            # amended 2026-08-07 on the author's review).
            axes.set_yticks([centre])
            self._level_panels.add(id(axes))

    def _band(self, index: int) -> tuple[float, float]:
        """The value range panel `index` holds, top panel first.

        `omitted` is ascending, so the topmost panel sits above the LAST gap
        and the bottom panel below the first.
        """
        gaps = list(reversed(self.omitted))
        floor = gaps[index][1] if index < len(gaps) else -np.inf
        ceiling = gaps[index - 1][0] if index > 0 else np.inf
        return (floor, ceiling)

    def _make_proportional(self) -> None:
        """Clause 3: one unit of y is the same number of pixels in both.

        Without this the same slope is drawn at two angles, and a reader
        comparing the gradient of a curve against the flatness of a level is
        comparing two different scalings of one quantity — which is the twin
        axis §B.5 forbids, arrived at through the back door.

        **Called last, and that ordering is load-bearing.**
        `place_legend_outside` runs `tight_layout`, which recomputes every
        `Axes` position from scratch — so heights set before it are silently
        discarded. Measured: the two panels came out at 0.94 and 1.77 units of
        height per unit of y, a factor of 1.9 on a figure whose entire subject
        is a comparison of levels.
        """
        spans = [axes.get_ylim()[1] - axes.get_ylim()[0] for axes in self.axes]
        if sum(spans) <= 0:
            return
        boxes = [axes.get_position() for axes in self.axes]
        box = boxes[0].union(boxes)
        gaps = 0.04 * box.height
        gap = gaps / max(1, len(self.axes) - 1)
        usable = box.height - gaps
        # §B.5.1 clause 3 as amended 2026-08-07: a panel holding nothing but
        # flat levels is EXEMPT and takes a floor. Its span is invented by
        # `_thin_the_level_panels`, so a height proportional to that span is a
        # height proportional to a rendering choice -- measured at 6.2 px with
        # no room for a tick. The panels that carry extent stay proportional
        # among themselves, which is all clause 3 protects: it exists so one
        # slope is never drawn at two angles, and a level has no slope.
        floors = {
            index: LEVEL_PANEL_HEIGHT * box.height
            for index, axes in enumerate(self.axes)
            if id(axes) in self._level_panels
        }
        share = usable - sum(floors.values())
        # §B.5.1 clause 3 as amended 2026-08-10: the share may be re-weighted
        # BY DECLARATION, because proportionality answers "which panel is this
        # figure about?" with "the widest cluster" -- measured wrong on ICASSP
        # Figure 1 by a factor of 3.5. A level panel's entry is 1.0 and takes
        # no effect (its height is a floor, not a share), which is why the
        # parse-time check refuses anything else there rather than letting an
        # inert number look like a knob.
        weighted = [
            span * (self.weights[index] if self.weights else 1.0)
            for index, span in enumerate(spans)
        ]
        extent = sum(span for index, span in enumerate(weighted) if index not in floors)
        if extent <= 0 or share <= 0:
            # Every panel is a level, or the floors leave nothing to share.
            # Equal heights: there is no extent anywhere to be proportional to.
            heights = [usable / len(self.axes)] * len(self.axes)
        else:
            heights = [
                floors.get(index, share * weighted[index] / extent)
                for index in range(len(spans))
            ]
        # Laid out bottom-up so each panel's y0 is the sum of what is beneath
        # it: with two panels the arithmetic was writable by hand, with N it
        # is not, and an off-by-one here would silently overlap two panels.
        offset = box.y0
        for axes, height in zip(reversed(self.axes), reversed(heights), strict=True):
            axes.set_position((box.x0, offset, box.width, height))
            offset += height + gap

    def label(self, *, figure_x: str, figure_y: str, title: Any) -> None:
        """One x label, one y label, one title — for the pair.

        A y label on each panel would name one quantity twice and read as two
        measures, so a broken figure carries a single figure-level label
        (clause 1: "one quantity, one unit, both panels").
        """
        self.axes[-1].set_xlabel(figure_x)
        if title:
            self.axes[0].set_title(str(title))
        if not self.omitted:
            self.axes[0].set_ylabel(figure_y)
            return
        for axes in self.axes[:-1]:
            axes.set_xlabel("")
        self.figure.supylabel(figure_y)

    def require_weightable_panels(self) -> None:
        """§B.5.1 clause 3: a level-only panel's declared weight must be 1.0.

        The half of the weights' validation that cannot be decided at parse:
        which panels hold nothing but flat levels is known only once the data
        is placed and `_thin_the_level_panels` has run. Such a panel takes a
        FLOOR rather than a share, so any other number there is a declaration
        that takes no effect — this project's most-repeated defect class.

        Raises:
            SpecificationError: Naming the panel's index and its weight.
        """
        if not self.weights:
            return
        for index, axes in enumerate(self.axes):
            if id(axes) in self._level_panels and self.weights[index] != 1.0:
                raise SpecificationError(
                    f"`ypanel_weights`[{index}] is {self.weights[index]:g} on a "
                    "panel that holds nothing but flat levels; its height is a "
                    "floor rather than a share (§B.5.1 clause 3), so the weight "
                    "would take no effect. Declare 1.0 there"
                )

    def finalise(self) -> None:
        """Everything that must happen after the layout is settled.

        Both steps read or write `Axes` positions, which `tight_layout` inside
        `place_legend_outside` recomputes — so doing either earlier is doing it
        to a layout that no longer exists.
        """
        if not self.omitted:
            return
        self.require_weightable_panels()
        self._make_proportional()
        self._share_tick_format()
        self._mark_break()

    def label_flat_series(
        self,
        table: pd.DataFrame,
        styles: Mapping[str, SeriesStyle],
        display_names: Mapping[str, str],
    ) -> None:
        """§B.5: a flat series is named at the right margin, in its colour.

        **What this must run after is `apply_axis_rule`, not `finalise`.** Every
        height here is an axes FRACTION, so it depends on the panel's final
        `ylim` and on nothing else — which `apply_axis_rule` and
        `_thin_the_level_panels` settle. It does not depend on the panel's
        position, which is what `tight_layout` and `_make_proportional` move.

        This docstring said the opposite until a mutant asked: swapping the
        call with `finalise` was measured to leave the figure **unchanged**, so
        "called after `finalise` and the ordering is load-bearing" was a claim
        borrowed from `_make_proportional`, where it is true because that
        method reads positions. Recorded rather than quietly deleted, because
        an ordering comment that is wrong is worse than none — the next reader
        preserves it.

        The value is carried here rather than in a legend row, which is where
        §B.3.1 used to put it for a bound. Same information, at the height the
        reader is already looking at.
        """
        # Grouped by PANEL before anything is placed, because §B.5's
        # displacement rule is a statement about a panel's whole set of labels
        # -- making room for one at the bottom moves the ones above it -- and
        # a loop that annotated as it went could only ever look upwards.
        #
        # WHICH panel holds a level is this figure's own question, and it is
        # the only part of §B.5's rule that is: `grouped_bars` has one axes and
        # no such question. The placement itself lives in `margin_labels`,
        # shared, because two readings of one rule is how the three callers of
        # `_flat_levels` came to agree "only by accident".
        grouped: dict[int, tuple[Axes, list[tuple[str, float]]]] = {}
        for label, value in _flat_levels(table):
            axes = self._panel_holding(value)
            if axes is None:
                continue
            grouped.setdefault(id(axes), (axes, []))[1].append((label, value))
        # §B.5 as amended 2026-08-10: where a near pair's lower label does not
        # fit below its line, the PANEL makes room -- which changes that
        # panel's span, and clause 3's heights are proportional to spans. The
        # hook re-assigns them before the labels are placed against the final
        # geometry; without it a reservation would silently leave one panel
        # drawn at a different number of pixels per unit than the rest.
        label_levels(
            self.figure,
            list(grouped.values()),
            styles,
            display_names,
            relayout=self._make_proportional if self.omitted else None,
        )

    def flat_series_labelled(self, table: pd.DataFrame) -> tuple[str, ...]:
        """Flat series that WILL take a margin label, i.e. are on the canvas.

        The right strip is theirs, so the legend must not also be placed there.
        """
        flat = {label for label, _ in _flat_levels(table)}
        return tuple(sorted(flat - set(self.clipped_flat_series(table))))

    def clipped_flat_series(self, table: pd.DataFrame) -> tuple[str, ...]:
        """Flat series that no panel's limits contain.

        These keep their legend row (§B.3.1): the margin can only label a
        series at the height it is drawn, and a clipped one is not drawn at any.
        """
        return tuple(
            label
            for label, value in _flat_levels(table)
            if self._panel_holding(value) is None
        )

    def _panel_holding(self, value: float) -> Axes | None:
        """The panel whose limits contain `value`, or None if it is clipped.

        A clipped reference keeps the legend row it already had (§B.3.1), so
        returning None here is not a silent loss -- it is the case the margin
        label cannot serve, because there is no height to put it at.
        """
        for axes in self.axes:
            low, high = axes.get_ylim()
            if low <= value <= high:
                return axes
        return None

    def _share_tick_format(self) -> None:
        """One quantity, one number format, both panels (clause 1).

        matplotlib picks a formatter per `Axes` from that panel's own range,
        so the two halves of ONE axis come out at different precision — seen
        on the real render: `2.30` above and `2.150` below, which reads as two
        quantities measured to different accuracy. The finer of the two is
        applied to both, because rounding the narrow panel to suit the wide
        one would discard the resolution the break was made to show.
        """
        # A LEVEL PANEL DOES NOT GET A VOTE, and this is the regression that
        # taught it: its single tick is placed AT the level, so matplotlib
        # labels it with the value's full precision -- 9.150350 and 1.859845 on
        # ICASSP Figure 1 -- and the `max` below then printed SIX decimals on
        # every panel, including 8.840000 where two would do. The tick is
        # placed at the value on purpose (it is what makes the break legible),
        # so the fix is to read the precision from the panels that carry a
        # RANGE. They are the ones whose ticks matplotlib chose; a level
        # panel's was chosen here.
        ranged = [axes for axes in self.axes if id(axes) not in self._level_panels]
        decimals = max(_tick_decimals(axes) for axes in ranged or self.axes)
        for axes in self.axes:
            axes.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))

    def _mark_break(self) -> None:
        """The diagonal marks, on both panels (clause 3).

        Labelled `_break…` so matplotlib keeps them out of the legend — and so
        the greyscale gate reads them as chrome rather than as an unencoded
        series, which is the same convention gridlines and hatching use.
        """
        pairs = list(zip(self.axes, self.axes[1:], strict=False))
        for upper, lower in pairs:
            upper.spines["bottom"].set_visible(False)
            lower.spines["top"].set_visible(False)
            upper.tick_params(bottom=False, labelbottom=False)
        edges = [(upper, 0.0) for upper, _ in pairs] + [
            (lower, 1.0) for _, lower in pairs
        ]
        for axes, edge in edges:
            for x in (0.0, 1.0):
                axes.plot(
                    [x - BREAK_MARK_SIZE, x + BREAK_MARK_SIZE],
                    [
                        edge - BREAK_MARK_SIZE * BREAK_MARK_SLOPE,
                        edge + BREAK_MARK_SIZE * BREAK_MARK_SLOPE,
                    ],
                    transform=axes.transAxes,
                    color="black",
                    clip_on=False,
                    lw=0.8,
                    label="_break",
                )


def _flat_levels(table: pd.DataFrame) -> list[tuple[str, float]]:
    """`(label, value)` for every series with no swept axis value, name-sorted.

    One reading of "flat", for the three callers that need it: which series
    take a margin label, which of those the axis clips, and where each one is
    drawn. They disagreed only by accident before, but a fourth caller written
    against a fourth spelling is how a series comes to be labelled twice.
    """
    levels: list[tuple[str, float]] = []
    for label in sorted(set(table["contender"])):
        rows = table[table["contender"] == label]
        if not bool(rows["axis_value"].isna().all()):
            continue
        value = float(np.asarray(rows["aggregate"], dtype=np.float64)[0])
        levels.append((str(label), value))
    return levels


def _curve_labels(table: pd.DataFrame, order: Sequence[str]) -> tuple[str, ...]:
    """The series that VARY across the swept axis, in drawing order.

    §B.5's legend rule after the 2026-08-07 amendment: only these take legend
    rows. A flat series has one height and is labelled there instead.
    """
    varies = {
        str(label)
        for label in table["contender"].unique()
        if not bool(table[table["contender"] == label]["axis_value"].isna().all())
    }
    return tuple(label for label in _draw_order(table, order) if label in varies)


def _tick_decimals(axes: Axes) -> int:
    """Decimals the ticks matplotlib chose for `axes` actually carry."""
    labels = [
        text
        for text in (str(label.get_text()) for label in axes.get_yticklabels())
        if "." in text
    ]
    return max((len(text.split(".")[-1]) for text in labels), default=0)


def _panels(
    omitted: tuple[tuple[float, float], ...],
    table: pd.DataFrame,
    config: Mapping[str, Any],
) -> _Panels:
    """One panel, or `len(omitted) + 1` stacked panels sharing an x axis."""
    del table
    weights = _declared_panel_weights(omitted, config)
    if not omitted:
        figure, axes = plt.subplots()
        return _Panels(figure=figure, axes=(axes,), omitted=())
    figure, stacked = plt.subplots(len(omitted) + 1, 1, sharex=True)
    return _Panels(figure=figure, axes=tuple(stacked), omitted=omitted, weights=weights)


def _declared_panel_weights(
    omitted: tuple[tuple[float, float], ...], config: Mapping[str, Any]
) -> tuple[float, ...]:
    """§B.5.1 clause 3's declared re-weighting, validated.

    Refusals, each by name, and each closing a way for the declaration to be
    decorative: a weighting on an unbroken axis (one panel has the whole
    height and the numbers reach nothing), the wrong count (a reader cannot
    tell which panel an entry names), and a non-positive entry (a panel of
    zero or negative height is not a compression, it is a disappearance).
    The level-panel-must-be-1.0 half cannot be decided here — which panels
    hold only levels is known after the data is placed — and lives in
    `_Panels.require_weightable_panels`.

    Returns:
        One weight per panel, top first, or `()` when nothing is declared.
    """
    declared = config.get("ypanel_weights")
    if declared is None:
        return ()
    if not omitted:
        raise SpecificationError(
            "`ypanel_weights` is declared on a figure with no `ybreak`; there "
            "is one panel and it already has the whole height, so the weights "
            "reach nothing (§B.5.1 clause 3)"
        )
    weights = tuple(float(value) for value in declared)
    if len(weights) != len(omitted) + 1:
        raise SpecificationError(
            f"`ypanel_weights` has {len(weights)} entries for "
            f"{len(omitted) + 1} panels; one entry per panel, top first, so "
            "that a reader of the document can tell which panel each names"
        )
    for index, weight in enumerate(weights):
        if weight <= 0.0:
            raise SpecificationError(
                f"`ypanel_weights`[{index}] is {weight:g}; a weight must be "
                "positive — zero or less is a panel that disappears rather "
                "than one that is compressed"
            )
    return weights


def widest_empty_range(table: pd.DataFrame) -> tuple[float, float] | None:
    """The widest gap in the plotted values, for an author choosing a break.

    §B.5.1 clause 4 draws the line this function sits on: "**A tool may report
    the widest empty range to a human choosing; it may not choose.**" So this
    exists, is exported, and is called by nothing in the render path — a break
    inferred from the data would move whenever a seed or a contender changed,
    and a figure whose axis moved with no document edit is a figure whose two
    renders cannot be compared.

    Args:
        table: The tidy frame.

    Returns:
        `(low, high)` of the widest gap between consecutive plotted values, or
        `None` when there are fewer than two distinct values to have a gap
        between.

        **It reports and does not judge.** A dense column still has a widest
        gap and this still names it; whether that gap is worth breaking an
        axis over is the author's call, and a threshold here would be this
        function quietly choosing after all.
    """
    values = np.asarray(table["aggregate"], dtype=np.float64)
    if SPREAD_COLUMN in table.columns:
        spreads = np.asarray(table[SPREAD_COLUMN], dtype=np.float64)
        spreads = np.where(np.isfinite(spreads), spreads, 0.0)
        values = np.concatenate([values - spreads, values + spreads])
    finite = np.sort(values[np.isfinite(values)])
    if finite.size < 2:
        return None
    gaps = np.diff(finite)
    widest = int(np.argmax(gaps))
    if gaps[widest] <= 0.0:
        return None
    return (float(finite[widest]), float(finite[widest + 1]))


def _declared_breaks(
    table: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[tuple[float, float], ...]:
    """The omitted range, validated against the data, or `None`.

    Raises:
        SpecificationError: If the declaration is malformed, inverted, or would
            remove a range that is not empty. §B.5.1 clause 2: "a break drawn
            through data is **refused** rather than rendered — a gap that hides
            marks is the compression §B.5 forbids, performed deliberately."
    """
    declared = config.get("ybreak")
    if declared is None:
        return ()
    ranges = _break_ranges(declared)
    # THE DECLARATION'S OWN SHAPE FIRST, then the data. Both orders refuse the
    # same documents, but only this one gives the author the specific reason:
    # a pair of overlapping ranges that also happens to cover a mark was
    # reported as "would remove a range that is not empty", which sends them
    # looking at their data when the fault is in the two lines they typed.
    for low, high in ranges:
        if not high > low:
            raise SpecificationError(
                f"`ybreak` upper bound {high:g} is not above lower bound "
                f"{low:g}; the declaration names the range the axis removes, "
                "low first"
            )
    ordered = tuple(sorted(ranges))
    for (_, first_high), (second_low, _) in zip(ordered, ordered[1:], strict=False):
        if second_low < first_high:
            raise SpecificationError(
                f"`ybreak` ranges overlap: one begins at {second_low:g} before "
                f"another ends at {first_high:g}. Each break removes a distinct "
                "empty range, and overlapping ones do not describe a panel "
                "layout at all"
            )
    for low, high in ordered:
        hidden = _values_inside(table, low, high)
        if hidden:
            shown = ", ".join(f"{label} at {value:g}" for label, value in hidden[:3])
            raise SpecificationError(
                f"`ybreak` [{low:g}, {high:g}] would remove a range that is not "
                f"empty: {shown}. §B.5.1 permits a break only over an empty "
                "range — a gap that hides marks is the axis compression §B.5 "
                "forbids, performed deliberately. Use `widest_empty_range` to "
                "see where the data actually leaves room"
            )
    return ordered


def _break_ranges(declared: Any) -> tuple[tuple[float, float], ...]:
    """`[low, high]` or `[[low, high], …]`, as ranges.

    The flat two-number spelling is what every existing document writes and it
    keeps working: §B.5.1 permitted exactly one break until 2026-08-07, so
    refusing it would invalidate every figure already declared.
    """
    values = list(declared)
    if values and all(isinstance(item, (int, float)) for item in values):
        values = [values]
    ranges: list[tuple[float, float]] = []
    for item in values:
        pair = list(item)
        if len(pair) != 2:
            raise SpecificationError(
                f"`ybreak` names a range and takes exactly two numbers, got "
                f"{item!r}; §B.5.1 allows several breaks, each of them a pair"
            )
        ranges.append((float(pair[0]), float(pair[1])))
    if not ranges:
        raise SpecificationError(
            "`ybreak` was declared with no range in it; omit the key entirely "
            "for an unbroken axis"
        )
    return tuple(ranges)


def _values_inside(
    table: pd.DataFrame, low: float, high: float
) -> list[tuple[str, float]]:
    """Every plotted value strictly inside the omitted range.

    Strictly, so a bound sitting exactly on an edge is shown rather than
    refused — otherwise the natural declaration, "break from the bound up to
    the curves", could not be written at all.
    """
    spreads = (
        np.where(
            np.isfinite(np.asarray(table[SPREAD_COLUMN], dtype=np.float64)),
            np.asarray(table[SPREAD_COLUMN], dtype=np.float64),
            0.0,
        )
        if SPREAD_COLUMN in table.columns
        else np.zeros(len(table), dtype=np.float64)
    )
    aggregates = np.asarray(table["aggregate"], dtype=np.float64)
    labels = list(table["contender"])
    found: list[tuple[str, float]] = []
    for label, aggregate, spread in zip(labels, aggregates, spreads):
        # The bar's ends are marks too: a break through the top of an error bar
        # hides a measurement exactly as a break through a point would.
        for value in (aggregate, aggregate - spread, aggregate + spread):
            if np.isfinite(value) and low < value < high:
                found.append((str(label), float(value)))
                break
    return found


def _require_known_config(config: Mapping[str, Any]) -> None:
    """Refuse a figure declaring a key this renderer does not read.

    Measured before this existed: `xscale`, `x_scale`, `log_x`, `logx`,
    `scale`, `axis_value` and an outright `typo_key` were **all** accepted in
    silence and **all** did nothing — every one rendered a byte-identical PNG.
    So an author asking for a log axis got a linear one and no indication, and
    the next key added for a campaign would have joined the six
    declared-everywhere-executed-nowhere fields this project has already
    shipped.

    Raises:
        SpecificationError: Naming the unknown keys and what is available,
            because a refusal an author cannot act on is barely better than
            the silence it replaces.
    """
    unknown = sorted(set(config) - CONFIG_KEYS)
    if not unknown:
        return
    raise SpecificationError(
        f"figure declares {', '.join(unknown)}, which `axis_scaling` does not "
        f"read; available: {', '.join(sorted(CONFIG_KEYS))}. A key that parses "
        "and reaches nothing renders a figure that silently ignores it"
    )


def _dispersion_channel(config: Mapping[str, Any]) -> str:
    """The figure's declared dispersion channel (§A.3.2 rule 4).

    Raises:
        SpecificationError: On a channel this renderer does not implement.
            Named rather than silently defaulted, because a figure asking for a
            band and getting bars would be wrong in the one direction rule 1
            exists to prevent — a mark named as one quantity and drawn as
            another.
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


def _figure_draws_bars(table: pd.DataFrame) -> bool:
    """Whether ANY row of the whole figure carries a drawable spread.

    A figure has exactly one dispersion channel. Deciding per series lets an
    analytic contender — whose across-seed spread is a legitimate 0.0 — fall
    back to the BCa band while its neighbours draw standard-deviation bars,
    and nothing on the page would tell a reader that the shading and the bars
    are two different quantities.
    """
    if SPREAD_COLUMN not in table.columns:
        return False
    values = np.asarray(table[SPREAD_COLUMN], dtype=np.float64)
    return bool((np.isfinite(values) & (values > 0.0)).any())


def _spreads(rows: pd.DataFrame) -> np.ndarray | None:
    """The drawable across-seed spreads of `rows`, or `None` if there are none.

    §A.3.2 rule 2: an absent spread and a zero spread are different facts and
    **neither draws a mark**. One seed has no across-seed quantity at all; an
    analytic contender under §A.2's common-random-numbers law has a spread of
    exactly zero and that zero is the answer, not a measurement too small to
    see. A zero-height bar still draws two caps, which a reader meets as a
    narrow measured interval — so both cases draw nothing and the table
    carries the distinction as `—` against `0.000`.

    Returns `None` when the column is absent entirely, which is an artifact
    written before §A.3.2 existed rather than a defect (§B.1.1).
    """
    if SPREAD_COLUMN not in rows.columns:
        return None
    values = np.asarray(rows[SPREAD_COLUMN], dtype=np.float64)
    drawable = np.isfinite(values) & (values > 0.0)
    if not drawable.any():
        return None
    # A row whose own spread is absent or zero contributes no mark while its
    # neighbours keep theirs: suppressing the whole series because one point
    # is analytic would delete the measurement from the rest.
    return np.where(drawable, values, 0.0)


#: The column §A.6.1 adds so the analysis/figure pairing is checked rather
#: than trusted, and the value that means this renderer must decline. Named as
#: literals rather than imported from `analysis`: Tier 7 may import Tier 6, but
#: this is one string in each direction and a test asserts the spellings agree —
#: the same treatment `spec.figure` already gives `SPREAD_COLUMN`.
AXIS_KIND_COLUMN = "axis_kind"
CATEGORICAL_AXIS = "categorical"


def _validated(table: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise SpecificationError(
            f"axis_scaling needs the columns {', '.join(missing)}, which this "
            "table does not carry; it renders `cost_vs_axis`'s output and "
            "computes nothing of its own"
        )
    _refuse_a_categorical_axis(table)
    if table.empty:
        raise SpecificationError(
            "axis_scaling was given an empty table; a figure with no series is "
            "not a figure, and an analysis that produced none should have said so"
        )
    return table


def _refuse_a_categorical_axis(table: pd.DataFrame) -> None:
    """Refuse `cost_by_category`'s table (§A.6.1's last paragraph).

    This renderer joins its points with a line, and a line between two
    categories asserts a rate of change that does not exist. The failure is
    silent in every other respect: the positions are 0, 1, 2, the curve is
    smooth, and the figure reads as a trend — plausible, and wrong, which is
    the output this architecture exists to prevent.

    **An absent column means numeric**, so an artifact written before §A.6.1
    existed still rebuilds (§B.1.1). Only an explicit `"categorical"` is
    refused, which is why the column is emitted rather than inferred.

    Raises:
        SpecificationError: Naming the renderer that takes such a table.
    """
    if AXIS_KIND_COLUMN not in table.columns:
        return
    if not (table[AXIS_KIND_COLUMN] == CATEGORICAL_AXIS).any():
        return
    raise SpecificationError(
        "this table declares a categorical axis (Annex 03 §A.6.1) and "
        "`axis_scaling` draws a line between neighbouring x positions, which "
        "for categories asserts a rate of change that does not exist. Render "
        "it with kind = 'grouped_bars', whose x axis carries the contenders "
        "and whose fills carry the categories"
    )


def _draw_order(table: pd.DataFrame, order: Sequence[str]) -> list[str]:
    """Labels in the study's declaration order, then any the study omitted."""
    present = list(dict.fromkeys(table["contender"]))
    declared = [label for label in order if label in present]
    return declared + [label for label in present if label not in declared]


@dataclass(frozen=True)
class _Series:
    """One contender's rows plus everything needed to draw them.

    Bundled rather than passed as six parameters: adding `draws_bars` took
    `_draw_curve` to seven against the permanent `PLR0913=6`, and the standing
    rule is to bundle rather than raise a gate to fit a feature.
    """

    label: str
    rows: pd.DataFrame
    style: SeriesStyle
    role: Role
    display_names: Mapping[str, str]

    def legend_text(self, value: float | None) -> str:
        return legend_label(self.label, value, self.role, self.display_names)


def _draw_curve(axes: Axes, series: _Series, *, channel: str, bars: bool) -> bool:
    """One contender's curve. Returns whether an error bar was drawn.

    Args:
        axes: Where to draw.
        series: The contender's rows and encoding.
        channel: The FIGURE's declared dispersion channel (§A.3.2 rule 4).
            ``"none"`` draws no dispersion at all — neither the bar nor the
            band it would otherwise fall back to, because falling back is how
            a figure that declared no dispersion acquires one.
        bars: Whether the figure's channel resolves to the error bar. A series
            with no drawable spread of its own draws neither a bar nor a band
            when this is set — its dispersion is genuinely zero, and a band in
            its place would be a second quantity.
    """
    if channel == "none":
        style = series.style
        ordered = series.rows.sort_values("axis_value")
        axes.plot(
            np.asarray(ordered["axis_value"], dtype=np.float64),
            np.asarray(ordered["aggregate"], dtype=np.float64),
            label=series.legend_text(None),
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
        )
        return False
    style = series.style
    ordered = series.rows.sort_values("axis_value")
    x = np.asarray(ordered["axis_value"], dtype=np.float64)
    y = np.asarray(ordered["aggregate"], dtype=np.float64)
    axes.plot(
        x,
        y,
        label=series.legend_text(None),
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker,
    )
    spreads = _spreads(ordered)
    if spreads is not None:
        # UNLABELLED, and that is load-bearing. `errorbar` puts its label on
        # the container, so a labelled companion would make the greyscale gate
        # meet this series twice and refuse the figure for conflicting with
        # itself. The bar inherits the curve's colour, which is what makes it
        # read as the same series without a second legend entry.
        axes.errorbar(
            x, y, yerr=spreads, fmt="none", ecolor=style.color, capsize=CAPSIZE
        )
        # The bar IS the figure's dispersion channel (§A.3.2), so the band is
        # not also drawn: two shadings of two different quantities, with
        # nothing on the page saying which is which, is the misreading rule 1
        # exists to prevent. The interval remains reported in the table.
        return True
    if bars:
        return False
    low = np.asarray(ordered["interval_low"], dtype=np.float64)
    high = np.asarray(ordered["interval_high"], dtype=np.float64)
    if not np.isnan(low).all():
        # One seed produces no across-seed interval at all (§A.3), and a band
        # drawn from nothing would be a claim the analysis refused to make.
        axes.fill_between(x, low, high, color=style.color, alpha=BAND_ALPHA, lw=0)
    return False


def _draw_flat(axes: Axes, series: _Series, *, channel: str = "bar") -> bool:
    """A contender the axis does not apply to: one horizontal reference.

    Its value goes in the legend (§B.3.1), so a bound the axis rule clips is
    still fully reported rather than silently absent.

    Returns whether an error bar was drawn. A flat contender is still measured
    over every seed, so it still has an across-seed spread — drawing bars only
    on curves would silently exempt exactly the rows a reader compares the
    curves against.
    """
    style = series.style
    value = float(series.rows["aggregate"].iloc[0])
    axes.axhline(
        value,
        label=series.legend_text(value),
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker or "",
    )
    spreads = _spreads(series.rows)
    if spreads is None or channel == "none":
        return False
    # At the middle of the x range: a horizontal line has no natural x, and
    # anchoring the mark at an edge would put it under the axis spine.
    left, right = axes.get_xlim()
    axes.errorbar(
        [0.5 * (left + right)],
        [value],
        yerr=[float(spreads[0])],
        fmt="none",
        ecolor=style.color,
        capsize=CAPSIZE,
    )
    return True


def _apply_axis_rule(
    axes: Axes,
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    include_flat: bool = False,
) -> None:
    """§B.5: the axis is scaled to the data under study.

    **"Data under study" is a fact about ROLE, not about whether a row happens
    to be flat.** A `CONTENDER` or a `BASELINE` sets the limits whether or not
    it carries a swept axis value; only a `REFERENCE` or a `BOUND` is context,
    drawn clipped with its value in the legend, which is §B.3.1's own encoding
    for them. A bound an order of magnitude below every contender would
    otherwise compress the whole result into a sliver — a measured complaint
    about this project's own NB04 figure, not a hypothetical.

    This read ``table["axis_value"].notna()`` until ICASSP Figure 1, using "has
    a swept axis value" as a proxy for "is not a reference". The proxy held
    only while every flat series *was* a reference or a bound. Figure 1 is the
    first study in which it is not: `truncated_riccati` is a `BASELINE` — an
    attained non-learned policy, and the number a learned box-aware controller
    has to beat — and it carries no unrolling depth. Measured on the rendered
    figure: the depth-swept contenders spanned 8.2742–8.8240, the panel was
    scaled to exactly that, and Truncated-Riccati at **9.1272** was drawn off
    the canvas — present in the legend and nowhere on the axes. Two flat
    CONTENDERS (the GRU, COCP) survived only because they happened to land
    inside the swept range, which is the clearest statement of the defect:
    whether a contender was shown depended on luck.

    `include_flat` is §B.5.1 clause 5's inversion, and it is a parameter rather
    than a second function because it is one word of the rule that changes:
    inside a broken figure the LOWER panel "exists precisely to hold the
    references, so it is scaled *to* them". Without it that panel took no
    limits at all whenever every row below the break was flat — the early
    return below fires on an empty `curved` — and matplotlib's autoscale
    silently decided the axis instead. Measured on a lower panel holding one
    bound: a span of 0.214 where the references occupy 5e-05.

    **Computed from the table, not from `ax.dataLim`.** `viz.style.theme
    .autoscaled_ylim_from_data` reads `dataLim`, which grows as artists are
    added — so using it would make the axis depend on whether the flat
    references happened to be drawn before or after the curves. The range is
    a property of the data, and reading it from the data is the only form
    that a change of draw order cannot move. The interval bands are included,
    because a band that fell outside the axis would be a reported claim the
    figure did not show.
    """
    if config.get("ylim") is not None:
        low, high = config["ylim"]
        axes.set_ylim(float(low), float(high))
        return
    curved = table if include_flat else table[table["role"].isin(_DATA_UNDER_STUDY)]
    if curved.empty:
        return
    aggregates = np.asarray(curved["aggregate"], dtype=np.float64)
    # The error bars are part of the data under study, not chrome. §A.3.2's
    # spread is normally WIDER than §A.3's interval — measured on the tracked
    # table, 9.4e-5 of interval against an axis range of 0.1825 — so an axis
    # padded from the interval alone would clip the very marks this figure
    # exists to show.
    # An explicit finite mask, NOT `np.nan_to_num`: that helper replaces `+inf`
    # with 1.797e308 rather than with the fill value, so an infinite spread
    # survived as a finite-but-enormous limit and `high - low` then overflowed
    # to `inf`. matplotlib refuses an infinite limit outright, so the whole
    # figure failed to draw. Found by a test written for `_spreads`, one
    # function away.
    declared = (
        np.asarray(curved[SPREAD_COLUMN], dtype=np.float64)
        if SPREAD_COLUMN in curved.columns
        else np.zeros_like(aggregates)
    )
    spreads = np.where(np.isfinite(declared), declared, 0.0)
    # THE AXIS IS SCALED TO WHAT IS DRAWN. Under `dispersion = "none"` neither
    # the bar nor the band reaches the canvas, so padding the range to hold
    # them reserves height for marks that do not exist -- and it reserves the
    # most for the widest spread, which on ICASSP Figure 1 is the GRU's 0.0173
    # against a 0.0344 cluster. Suppressing the bar and keeping its footprint
    # would answer the author's complaint about the ink while leaving the
    # geometry that made it dominate.
    values = (
        aggregates
        if _dispersion_channel(config) == "none"
        else np.concatenate(
            [
                aggregates,
                aggregates - spreads,
                aggregates + spreads,
                np.asarray(curved["interval_low"], dtype=np.float64),
                np.asarray(curved["interval_high"], dtype=np.float64),
            ]
        )
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    low, high = float(finite.min()), float(finite.max())
    pad = Y_MARGIN * (high - low) or abs(high) * Y_MARGIN or Y_MARGIN
    axes.set_ylim(low - pad, high + pad)
