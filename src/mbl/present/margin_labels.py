"""§B.5's flat-series labels: a flat series is named where it is drawn.

"A FLAT series is labelled at its own line INSIDE the axes, not in the legend
— in its own colour, with its value, written immediately above the line and
right-aligned flush against the right spine." *Corrected 2026-08-09, on the
advisors' instruction*: as first shipped the label sat in the OUTER right
margin, which spent a horizontal strip of the canvas as wide as the longest
label on white space — the same economy §B.5's legend-inside rule refuses
vertically. The near-pair rule joins the correction: when two flat lines sit
within one label height of each other in display space, the LOWER line's
label is written below its line instead of above it, so each label still hugs
its own line and the order of labels remains the order of values by
construction.

**One implementation, because it is one rule.** `axis_scaling` needs it per
panel of a broken axis and `grouped_bars` needs it on a single axes, and this
project has already been bitten by a second reading of one rule: `_flat_levels`
records three callers that agreed "only by accident". A second copy of the
displacement arithmetic would be a second answer to how far apart two labels
must be.

**The displacement rule is the delicate half.** Two levels too close to print
apart are displaced — never re-ordered, and never outside the panel they
belong to. Both halves are load-bearing: a label names the height it is drawn
at, so the order of the labels down the figure must be the order of the
values; and the panel's own extent bounds the displacement, because the strip
below a panel's floor belongs to the *next* panel. Under the inside placement
the near-pair above/below split is the displacement's first resort; the
floor-bounded nudge (`separated`) remains the rule when even the
below-position collides.

*Measured on four levels within 6e-4 of each other:* a fixed downward nudge
with no floor placed them at 0.103, 0.058, 0.013 and **−0.032** of the panel
height — the fourth label 20 pixels below its own axes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .encodings import SeriesStyle

#: Minimum vertical separation between two margin labels, in axes fractions.
#: Two levels 7e-4 apart otherwise print on top of each other. A CEILING on the
#: separation actually used, not a constant: `separated` narrows it when a
#: panel holds more levels than this spacing fits, so that no label is ever
#: displaced out of its own panel.
MARGIN_LABEL_GAP = 0.045

#: Leading between two margin labels, as a multiple of the taller one's
#: rendered height. A separation of exactly one text height leaves the two
#: glyph boxes touching, which prints as a single smudged line; this is
#: ordinary typographic leading on top of it. It also decides the near-pair
#: rule: two lines closer than this many text heights cannot both carry an
#: above-label without the lower text striking the upper line.
MARGIN_LABEL_LEADING = 1.25

#: Horizontal inset of an inside label from its spine, in points. The
#: outside placement offset the text 4 pt PAST the right spine; the inside
#: placement offsets it the same 4 pt in from the LEFT one (§B.5 as amended
#: 2026-08-10), so the label's ink and the spine's never touch (the mutant
#: that deleted the offset printed them joined).
LABEL_INSET_POINTS = 4.0

#: Vertical breathing between a line and its label's nearest ink edge, in
#: points. Zero would rest the glyph boxes' baseline exactly on the line it
#: names, which prints as a struck-through label.
LABEL_PAD_POINTS = 2.0


def separated(
    naturals: Sequence[float],
    *,
    minimum: float = 0.0,
    floors: Sequence[float] | None = None,
) -> list[float]:
    """Label heights that neither overprint, re-order, nor leave the panel.

    §B.5's displacement rule, amended 2026-08-07. `naturals` are the heights
    the labels *want*, as axes fractions in **descending** order; the return
    is the heights they get, in the same order.

    **Two passes, and the second is the one the first version was missing.**
    Going down, each label is pushed below the one above it — that alone was
    the whole rule, and it has no floor: four levels within 6e-4 of each other
    at the bottom of a panel came out at 0.103, 0.058, 0.013 and **-0.032**,
    the last label twenty pixels below its own axes and, on a broken figure,
    inside the panel beneath. Coming back up from a floor of zero restores
    containment without touching the order, because both passes only ever
    enforce the same one-sided inequality.

    `minimum` is the separation the LABELS THEMSELVES need — their rendered
    height as a fraction of this panel — and `MARGIN_LABEL_GAP` is a floor
    under it for looks. Taking the larger is what makes the rule about
    legibility rather than about a number: measured, two labels 0.045 of a
    panel apart printed clear of each other on a tall panel and OVERLAPPED on
    a short one, because a fraction of a panel is not a constant number of
    points and the text is.

    The result is therefore ordered, separated by at least `gap`, and inside
    `[0, 1]` — the last of those provably, since a label raised off the floor
    can reach at most `(n - 1) · gap`, which is what `gap` is capped at.

    **The gap narrows rather than the panel overflowing.** A panel with more
    levels than `1 / MARGIN_LABEL_GAP` cannot separate them all at that
    spacing, and the alternatives are to crowd some pair by an amount nobody
    chose or to push a label off the figure. Sharing the shortfall across the
    whole set is the one option that keeps every label inside its panel and in
    order, and it needs no answer to "how many levels will a figure have" —
    which is a free parameter, and this project has a standing rule about
    designing around one.

    The cap is `1 / n` and not `1 / (n - 1)`, which is the spacing that would
    pack `n` labels exactly edge to edge. Measured: at 40 levels the exact fit
    came out at **1.0000000000000004**, because the second pass reaches the top
    by adding `gap` thirty-nine times and float64 addition is not associative
    with the division that produced it. Leaving one label's worth of slack
    costs nothing a reader can see and makes the containment an arithmetic
    fact rather than one that holds to within an epsilon.

    **`floors` (added 2026-08-10, measured on the retrained Figure 1)** are
    per-label lower bounds the upward pass honours — the inside placement's
    line-adjacency constraint: an above-the-line label may be displaced UP but
    never back down through the line it names. Found on the real figure, not
    by the suite: the COCP pair sits 0.08 of a panel above its floor, the
    below-position did not fit, and the down-then-floor bias of the two
    passes parked a label's box across its own line. Each floor is capped at
    ``1 − index·gap`` so the stack a raised floor pushes upward still fits —
    the same arithmetic that caps the gap itself; when adjacency and
    containment conflict, containment wins, exactly as it always has.
    """
    if not naturals:
        return []
    gap = min(max(minimum, MARGIN_LABEL_GAP), 1.0 / max(1, len(naturals)))
    bounded = (
        [0.0] * len(naturals)
        if floors is None
        else [min(floor, 1.0 - index * gap) for index, floor in enumerate(floors)]
    )
    placed = list(naturals)
    for index in range(1, len(placed)):
        placed[index] = min(placed[index], placed[index - 1] - gap)
    placed[-1] = max(placed[-1], bounded[-1], 0.0)
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = max(placed[index], bounded[index], placed[index + 1] + gap)
    return placed


#: Candidate anchors for a label the left placement would put on top of a
#: mark, as fractions of the panel's width. Left-first, because §B.5 prefers
#: the left spine; the last resort is the right spine, which is where the
#: rule put every label before 2026-08-10.
LABEL_SLIDE_STOPS = (0.0, 0.08, 0.16, 0.24, 0.32, 0.40)


def _strikes_a_mark(axes: Axes, text: Any) -> bool:
    """Whether a placed label's ink covers any plotted line or marker.

    §B.5 as amended 2026-08-10 says a label that strikes a mark is a defect
    "wherever the rule prefers", so the preference is checked rather than
    trusted. Segments are sampled rather than intersected analytically: a
    dozen samples per segment is exact enough for a text box tens of pixels
    wide, and the alternative is a clipping routine nobody would read.
    """
    import numpy as np

    box = text.get_window_extent()
    for line in axes.get_lines():
        # A leader is the label's OWN chrome, not a plotted mark. It runs from
        # the label's near edge, so it lies inside the text's bounding box by
        # construction and every label would report striking itself -- measured
        # the moment leaders shipped, as a green suite turning red on the
        # slide-clear test. §B.5's rule is about *marks*; excluding chrome is
        # the rule read correctly, not an exemption from it.
        if line.get_label() == LEADER_ARTIST_LABEL:
            continue
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        if x_data.size < 2:
            continue
        points = axes.transData.transform(np.column_stack([x_data, y_data]))
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=True):
            steps = np.linspace(0.0, 1.0, 32)
            xs, ys = x0 + (x1 - x0) * steps, y0 + (y1 - y0) * steps
            if bool(
                np.any(
                    (xs >= box.x0) & (xs <= box.x1) & (ys >= box.y0) & (ys <= box.y1)
                )
            ):
                return True
    return False


def _slide_clear_of_marks(figure: Figure, axes: Axes, annotations: list[Any]) -> None:
    """Slide a struck label along its own line until its ink is clear.

    The label's HEIGHT is what names its value, so the remedy for a collision
    moves it horizontally and never vertically — it still hugs the same line,
    at the same height, and says the same thing. Left-first through
    `LABEL_SLIDE_STOPS`; a label clear nowhere ends at the right spine, the
    placement this rule had before the left one, because a label somewhere is
    worth more than a label on top of a curve.

    Measured on ICASSP Figure 1 (2026-08-10): the recurrent baseline's level
    sits 0.005 under the proposed controller's first point, so at the left
    spine its label lay across both proposed curves — found by measuring the
    render, which is what the amendment demands.
    """
    for text in annotations:
        if not _strikes_a_mark(axes, text):
            continue
        for stop in LABEL_SLIDE_STOPS[1:]:
            text.xy = (stop, text.xy[1])
            figure.canvas.draw()
            if not _strikes_a_mark(axes, text):
                break
        else:
            text.xy = (1.0, text.xy[1])
            text.set_ha("right")
            text.xyann = (-LABEL_INSET_POINTS, 0.0)
            figure.canvas.draw()


#: A leader is drawn once a label sits further from its NOMINAL placement than
#: this many of its own text heights.
#:
#: **Measured against the nominal position and not against the line**, which the
#: first implementation got wrong and the fixture caught: every label is placed
#: `LABEL_PAD_POINTS + half its height` clear of its line so it does not print
#: struck through, so a threshold measured from the line fires on *every* label
#: in *every* figure -- 5 of 5 in the suite's own cast, where the rule wants 2.
#: The quantity the rule is about is DISPLACEMENT: how far the near-pair split,
#: `separated` and the floor moved the label from where an undisplaced one sits.
LEADER_TRIGGER_HEIGHTS = 0.5

#: The artist label a leader carries. Matplotlib hides `_`-prefixed labels from
#: the legend, and `_strikes_a_mark` uses it to tell the label's own chrome from
#: a plotted mark.
LEADER_ARTIST_LABEL = "_leader"

#: Hairline, per §B.5's recessive-chrome rule. The leader is chrome that says
#: which line a label belongs to; it is not a series mark and must never read as
#: one.
LEADER_LINEWIDTH = 0.6


def panel_height(axes: Axes) -> float:
    """The panel's drawn height in pixels, for turning an axes fraction back
    into the pixels the leader rule measures in."""
    return float(axes.get_window_extent().height)


def _draw_leaders(
    figure: Figure,
    axes: Axes,
    placements: Sequence[tuple[Any, str, float, bool]],
    styles: Mapping[str, SeriesStyle],
    pad_pixels: float,
) -> None:
    """§B.5 (2026-08-12): connect a displaced label to the line it names.

    Every placement rule above may move a label off its own height — the
    near-pair rule by design, `separated` by necessity, the floor-bounded nudge
    as a last resort — and none of them said what then tells the reader WHICH
    line the text belongs to. Measured on ICASSP Figure 3, where the author
    reported it: three labels at 8.3, 8.5 and 14.6 px above their lines, two of
    them also slid sideways to clear a curve, reading as text floating between
    the lines rather than naming one.

    **Called last, and that ordering is load-bearing** — unlike the ordering
    claim this module's `label_flat_series` docstring once made and had to
    withdraw. This one is testable: the leader's endpoints are the label's
    settled bbox and the line's height, so running it before `separated` or
    `_slide_clear_of_marks` would connect a position the label no longer
    occupies. The horizontal slide is what makes it non-optional: a label 0.24
    of the panel to the right of the spine has no other tie to its line.

    The remedy is a connector and never a move: displacement decides where the
    text goes, and the leader only says what it belongs to.
    """
    if not placements:
        return
    transform = axes.transData
    inverse = axes.transData.inverted()
    # One sequence rather than three parallel ones: they were always zipped at
    # the top of this loop, and three sequences that must stay aligned is an
    # invariant the caller has to keep rather than one the type states.
    for text, label, value, goes_below in placements:
        # No renderer argument: `FigureCanvasBase` does not declare
        # `get_renderer`, so passing one is a strict-mypy error on any backend
        # the annotations cover. Every other measurement in this module takes
        # the extent the same way, and the caller has already drawn the canvas.
        box = text.get_window_extent()
        line_y = float(transform.transform((0.0, value))[1])
        centre_y = (box.y0 + box.y1) / 2.0
        height = box.y1 - box.y0
        # Where an UNDISPLACED label of this height sits: its own breathing
        # room and no more. Anything beyond it is displacement, which is the
        # only thing a leader exists to explain.
        nominal_y = line_y + (-1.0 if goes_below else 1.0) * (pad_pixels + height / 2.0)
        raised = abs(centre_y - nominal_y) > LEADER_TRIGGER_HEIGHTS * height
        # A label the slide rule pushed off a spine floats in open plot area
        # with nothing tying it to its line, however small its vertical gap.
        # Measured on Figure 3: both flat labels sat 0.01 heights off nominal --
        # not displaced at all -- yet one was anchored at 0.24 of the panel
        # width, which is the "scattered" reading the author reported. A rule
        # keyed on vertical displacement alone fires on neither.
        anchored_x = float(text.xy[0])
        slid = 0.0 < anchored_x < 1.0
        if not (raised or slid):
            continue
        # From the label's NEAR edge, so the leader spans the gap and no more.
        edge_y = box.y0 if centre_y > line_y else box.y1
        # Anchored under the label's leading edge: the eye reads a label
        # left-to-right, so the tie belongs where the reading starts.
        anchor_x = box.x0 + min(height, box.x1 - box.x0) / 2.0
        (x0, y0), (_, y1) = (
            inverse.transform((anchor_x, edge_y)),
            inverse.transform((anchor_x, line_y)),
        )
        axes.plot(
            [x0, x0],
            [y0, y1],
            color=styles.get(label, SeriesStyle("black", "-", "o")).color,
            linewidth=LEADER_LINEWIDTH,
            linestyle="-",
            marker="",
            zorder=text.get_zorder() - 0.1,
            clip_on=False,
            label=LEADER_ARTIST_LABEL,
        )


def _naturals(axes: Axes, ordered: Sequence[tuple[str, float]]) -> list[float]:
    """Each level's own height, as a fraction of the panel it is drawn in."""
    low, high = axes.get_ylim()
    return [(value - low) / (high - low) if high > low else 0.5 for _, value in ordered]


def _below_flags(naturals: Sequence[float], heights: Sequence[float]) -> list[bool]:
    """§B.5's near-pair rule: the LOWER member of a pair closer than one
    leading of text height carries its label below its own line.

    Decided in display space against the line one step up the value order —
    within that distance a label cannot sit above its line without striking
    the line above it.
    """
    flags = [False]
    flags.extend(
        (naturals[index - 1] - naturals[index])
        < MARGIN_LABEL_LEADING * max(heights[index - 1], heights[index])
        for index in range(1, len(naturals))
    )
    return flags


def _reserve_room_below(
    axes: Axes,
    ordered: Sequence[tuple[str, float]],
    naturals: Sequence[float],
    heights: Sequence[float],
    below: Sequence[bool],
    *,
    pad: float,
) -> bool:
    """Extend the panel's lower limit until every below-label fits inside it.

    §B.5 as amended 2026-08-10. A below-position that does not fit used to
    revert to above — the rule declining to apply on exactly the figure it
    was written for. The panel's lower limit is a **rendering** choice and
    not a datum, so the answer to "no room below" is room: the reservation
    only ever ADDS range, hides nothing and moves no mark.

    Returns:
        Whether the panel's limits changed — the caller re-assigns clause 3's
        proportional heights and re-places against the final geometry.
    """
    low, high = axes.get_ylim()
    if high <= low:
        return False
    needed = max(
        (
            pad + heights[index] - naturals[index]
            for index, flag in enumerate(below)
            if flag and naturals[index] - pad - heights[index] < 0.0
        ),
        default=0.0,
    )
    if needed <= 0.0:
        return False
    # Solve for the new floor from the lowest below-label's own fraction:
    # (value - floor) / (high - floor) = pad + height, at that label.
    index = min((i for i, flag in enumerate(below) if flag), key=lambda i: naturals[i])
    target = min(pad + heights[index], 0.9)  # never invert the panel
    value = ordered[index][1]
    floor = (value - target * high) / (1.0 - target)
    if floor >= low:
        return False
    axes.set_ylim(floor, high)
    return True


def label_levels(
    figure: Figure,
    groups: Sequence[tuple[Axes, Sequence[tuple[str, float]]]],
    styles: Mapping[str, SeriesStyle],
    display_names: Mapping[str, str],
    *,
    relayout: Callable[[], None] | None = None,
) -> None:
    """Annotate each group's levels inside the axes at their lines, then
    separate them.

    §B.5 as corrected 2026-08-09 and amended 2026-08-10: each label sits
    **inside** the axes, left-aligned a fixed inset past the **left** spine,
    immediately above its own line — except that the lower member of a near
    pair (two lines within one leading of text height in display space) goes
    below its line, so neither text strikes the other's line and the label
    order remains the value order by construction. Where the below-position
    does not fit, **the panel makes room** rather than the rule giving way.

    **Grouped before anything is placed**, because §B.5's displacement rule is
    a statement about a panel's whole set of labels — making room for one at
    the bottom moves the ones above it — and a loop that annotated as it went
    could only ever look upwards.

    **Separated after measuring**, because how far apart two labels must be is
    a fact about the TEXT and not about the panel. `MARGIN_LABEL_GAP` is an
    axes fraction, so the same 0.045 is a comfortable gap on a tall panel and
    an overlap on a short one. The above/below split is the first resort; the
    measured desired centres then pass through `separated`, which is what
    keeps a *cluster* — where even the below-position collides — ordered,
    legible and inside its panel (the floor-bounded nudge, unchanged).

    Args:
        figure: The owning figure; its canvas is drawn to measure, once per
            geometry.
        groups: `(axes, [(label, value), ...])`. The caller decides which axes
            holds which level — on a broken axis that is the panel containing
            it, and on a single-axes figure there is only one answer.
        styles: The encoding each label is drawn in; the text takes its colour.
        display_names: What a reader is shown for each label.
        relayout: Called after a reservation changed a panel's limits, so the
            caller can re-assign clause 3's proportional heights before the
            labels are placed against the final geometry. `None` for a caller
            with no such rule (a single-axes figure).
    """
    placed: list[tuple[Axes, list[Any], list[tuple[str, float]]]] = []
    for axes, entries in groups:
        if not entries:
            continue
        ordered = sorted(entries, key=lambda item: -item[1])
        annotations = [
            axes.annotate(
                f"{display_names.get(label, label)} {value:.4g}",
                xy=(0.0, fraction),
                xycoords="axes fraction",
                xytext=(LABEL_INSET_POINTS, 0.0),
                textcoords="offset points",
                color=styles.get(label, SeriesStyle("black", "-", "o")).color,
                va="center",
                ha="left",
                annotation_clip=False,
                fontsize="small",
            )
            for (label, value), fraction in zip(
                ordered, _naturals(axes, ordered), strict=True
            )
        ]
        placed.append((axes, annotations, ordered))
    if not placed:
        return

    figure.canvas.draw()
    pad_pixels = LABEL_PAD_POINTS * float(figure.dpi) / 72.0

    def measured(
        axes: Axes, annotations: list[Any], ordered: list[tuple[str, float]]
    ) -> tuple[list[float], list[float], list[bool], float] | None:
        panel = float(axes.get_window_extent().height)
        if panel <= 0.0:
            return None
        naturals = _naturals(axes, ordered)
        heights = [
            float(text.get_window_extent().height) / panel for text in annotations
        ]
        return naturals, heights, _below_flags(naturals, heights), pad_pixels / panel

    # One reservation pass over every panel, then ONE relayout: clause 3's
    # heights depend on the spans this may have changed, and the labels are
    # placed only against the final geometry.
    reserved = False
    for axes, annotations, ordered in placed:
        state = measured(axes, annotations, ordered)
        if state is None:
            continue
        naturals, heights, below, pad = state
        reserved |= _reserve_room_below(
            axes, ordered, naturals, heights, below, pad=pad
        )
    if reserved:
        if relayout is not None:
            relayout()
        figure.canvas.draw()

    for axes, annotations, ordered in placed:
        state = measured(axes, annotations, ordered)
        if state is None:
            continue
        naturals, heights, below, pad = state
        desired = [
            fraction - pad - heights[index] / 2.0
            if below[index]
            else fraction + pad + heights[index] / 2.0
            for index, fraction in enumerate(naturals)
        ]
        # Containment pre-clamp; monotone, so the descending order survives
        # and `separated` restores any marginal inversion by push-down only.
        desired = [
            min(max(centre, heights[index] / 2.0), 1.0 - heights[index] / 2.0)
            for index, centre in enumerate(desired)
        ]
        # An above-the-line label's own line is its floor: displacement may
        # raise it, never park it back across the line it names.
        floors = [
            0.0 if below[index] else centre for index, centre in enumerate(desired)
        ]
        for text, fraction in zip(
            annotations,
            separated(
                desired,
                minimum=MARGIN_LABEL_LEADING * max(heights, default=0.0),
                floors=floors,
            ),
            strict=True,
        ):
            text.xy = (0.0, fraction)
        # The height is settled; only now can a collision with a MARK be
        # decided, and its remedy is horizontal (see `_slide_clear_of_marks`).
        figure.canvas.draw()
        _slide_clear_of_marks(figure, axes, annotations)
        # Last of all, because a leader must connect where the label ENDED UP:
        # every rule above may have moved it off the height it names.
        figure.canvas.draw()
        _draw_leaders(
            figure,
            axes,
            [
                (text, label, value, goes_below)
                for text, (label, value), goes_below in zip(
                    annotations, ordered, below, strict=True
                )
            ],
            styles,
            pad * panel_height(axes),
        )
