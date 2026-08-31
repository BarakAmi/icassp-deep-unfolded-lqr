"""The greyscale law, as a gate — Annex 03 §B.4.

**Every series must be identifiable with all colour removed.** Decision D7,
and §B.4 says it is enforced rather than encouraged. What §B.4 did not say,
until §B.4.1 was written for this phase, is what the gate *computes* — and a
gate each implementer defines differently is not a gate.

The rule, per unordered pair of series drawn on one axes:

1. **Differing linestyle or marker passes.** Triple encoding is the standard;
   two series that already differ in a channel colour removal cannot touch are
   separable in print by construction, whatever their lightness.
2. **Otherwise colour was the only channel, so lightness must carry it.**
   Relative luminance — the sRGB linearisation of WCAG 2.x, the same quantity
   §B.3.3's contrast floors are built on — must differ by at least
   `LUMINANCE_FLOOR`.

**The gate reads the rendered figure, not the declared encodings.** A renderer
that ignored what it was told to draw would satisfy any check of its inputs;
`encodings_of` walks the axes' own artists, so what is asserted is what a
reader will see.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

#: Minimum relative-luminance separation for two series distinguished by
#: colour ALONE (§B.4.1). Deliberately coarser than §B.3.2's ordinal
#: `ΔL ≥ 0.06`: an ordinal ramp is read as a ramp and its neighbours are meant
#: to be similar, whereas two categorical series a reader must tell apart at a
#: glance are not.
LUMINANCE_FLOOR = 0.15

#: sRGB channel weights of the WCAG relative-luminance definition.
_WEIGHTS = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class SeriesEncoding:
    """What one drawn series looks like.

    Attributes:
        label: The legend entry. Series with no label are excluded before
            this type is built — an unlabelled artist is chrome, not a series.
        color: Its colour, in any form matplotlib accepts.
        linestyle: What matplotlib *names* the style — for messages and for
            callers asking which of the four named styles a series got.
        dash: What is actually **drawn**, and what separability is decided on.
            Distinct from `linestyle` because that name is lossy: see
            `dash_pattern_of`.
        marker: Its marker, as matplotlib reports it.
    """

    label: str
    color: Any
    linestyle: Any
    marker: Any
    dash: Any = None

    @property
    def shape(self) -> tuple[Any, Any]:
        """The channels that survive colour removal.

        `dash` rather than `linestyle`, so eight distinct dash patterns are
        eight encodings rather than one.
        """
        return (self.linestyle if self.dash is None else self.dash, self.marker)

    @property
    def luminance(self) -> float:
        """Relative luminance on 0–1, WCAG 2.x."""
        return relative_luminance(self.color)


@dataclass(frozen=True)
class GreyscaleConflict:
    """Two series a reader could not tell apart in print.

    Attributes:
        first: One label.
        second: The other.
        separation: Their relative-luminance difference.
    """

    first: str
    second: str
    separation: float

    def describe(self) -> str:
        """One line, naming both series and the number that failed."""
        return (
            f"{self.first!r} and {self.second!r} share a linestyle and a marker "
            f"and differ by only {self.separation:.3f} in relative luminance "
            f"(floor {LUMINANCE_FLOOR})"
        )


def dash_pattern_of(line: Any) -> Any:
    """The dash pattern a reader actually sees, which the name does not report.

    **matplotlib normalises every custom `(offset, on_off_seq)` linestyle to
    the NAME `'--'`.** Measured on 3.10.8: `(3,1,1,1)`, `(5,2)`, `(1,1)`,
    `(7,2,1,2)` and four more all return `'--'` from `get_linestyle()`. A gate
    reading the name therefore cannot tell §B.4.2's slots 5-12 apart — it would
    see eight encodings as one, and refuse a figure whose series are in fact
    perfectly separable. The channel would be decorative in the one place it is
    supposed to be decisive.

    **There is no public accessor.** `Line2D` has `set_dashes` and no getter
    (checked against `dir(Line2D)` on 3.10.8: `get_dash_capstyle`,
    `get_dash_joinstyle`, `get_linestyle`, and nothing else). So the pattern is
    read from the attribute matplotlib stores it in, falling back to the name.
    `test_the_gate_reads_dash_patterns_and_not_their_names` pins the
    dependency, so a matplotlib that renames the attribute fails loudly instead
    of quietly collapsing the channel back to one value.

    Args:
        line: A `Line2D`.

    Returns:
        Its unscaled dash pattern, or its linestyle name if the attribute is
        gone. Named styles have distinct patterns too -- `'-'` is `(0, None)`
        and `':'` is `(0.0, (1.0, 1.65))` -- so one key serves both.
    """
    pattern = getattr(line, "_unscaled_dash_pattern", None)
    return line.get_linestyle() if pattern is None else pattern


def relative_luminance(color: Any) -> float:
    """`color`'s relative luminance on 0–1 (WCAG 2.x sRGB linearisation).

    Args:
        color: Anything `matplotlib.colors.to_rgb` accepts.

    Returns:
        The luminance a greyscale conversion would produce.
    """
    channels = []
    for value in to_rgb(color):
        channels.append(
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        )
    return sum(weight * channel for weight, channel in zip(_WEIGHTS, channels))


def encodings_by_axes(figure: Figure) -> tuple[tuple[SeriesEncoding, ...], ...]:
    """Every labelled series on `figure`, grouped by the `Axes` it was drawn on.

    §B.4.1 says the pair rule holds "per unordered pair of series drawn on one
    axes", and it means it. The implementation pooled every `Axes` of a figure,
    which was indistinguishable while every figure was single-panel and becomes
    wrong the moment one is not: measured on a two-panel broken axis, the pooled
    form pairs each series with its own copy on the other panel and refuses the
    figure with `'alpha' and 'alpha' … differ by only 0.000`.

    Grouping here rather than in the caller is what makes a second panel
    non-bypassable. The alternative — suppressing the lower panel's labels so a
    pooled check passes — removes that panel from the gate entirely, which is
    how a check stops checking without anyone noticing.
    """
    return tuple(tuple(_encodings_of_axes(axes)) for axes in figure.axes)


def encodings_of(figure: Figure) -> tuple[SeriesEncoding, ...]:
    """Every labelled series drawn on `figure`, as it was actually drawn.

    Walks the axes' own artists rather than trusting a declaration: §B.4's gate
    is about what a reader sees, and a renderer that ignored its own style
    arguments would pass any check of those arguments.

    Labels beginning with an underscore are matplotlib's convention for
    "exclude from the legend" and are chrome — gridlines, spines, the
    constraint hatching — rather than series.

    Order and signature are unchanged; this is now the flattening of
    `encodings_by_axes`, and remains the inspection surface the suite reads.
    """
    return tuple(encoding for group in encodings_by_axes(figure) for encoding in group)


def _encodings_of_axes(axes: Axes) -> Iterable[SeriesEncoding]:
    """One `Axes`'s labelled series, from its lines and then its containers.

    Containers are walked because `ax.errorbar(label=…)` and `ax.bar(label=…)`
    put the label on the **container** and leave every `Line2D` they create at
    `_nolegend_`. Measured: `encodings_of` returned the empty tuple for a
    three-series errorbar figure, so `require_greyscale_separable` passed a
    figure it had checked nothing on — the print law silently off. That is the
    same defect class as a declared-but-inert field, and it is why a bar chart
    could clear the black-and-white gate without being looked at.

    Underscore labels are skipped here too, so the unlabelled
    `errorbar(fmt="none")` companion a curve draws its caps with — measured to
    register as `_container0` — adds nothing and cannot double-count the series
    its `plot` call already contributed.
    """
    for line in axes.get_lines():
        label = str(line.get_label())
        if not label or label.startswith("_"):
            continue
        yield SeriesEncoding(
            label=label,
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            dash=dash_pattern_of(line),
        )
    for container in axes.containers:
        label = str(container.get_label())
        if not label or label.startswith("_"):
            _require_labelled(container)
            continue
        yield _encoding_of_container(label, container)


def _require_labelled(container: Any) -> None:
    """Refuse an unlabelled bar container (§B.4.2, rule 3).

    **The half of the container defect the container walk cannot see.**
    Walking `axes.containers` closed the case where a label exists and sits on
    the container rather than on its children. This is the case where there is
    nothing to walk: measured, one `ax.bar` with six contender names on the x
    axis and no `label=` gives `encodings_of` the **empty tuple**, so
    `require_greyscale_separable` returns green having inspected nothing and
    the print law is silently off for the whole figure.

    Skipping it is what an unlabelled artist *should* get — gridlines, spines,
    the constraint hatching and an `errorbar(fmt="none")` companion are all
    chrome, and the last of these is registered as `_container0` beside every
    error bar this project draws. A `BarContainer` is not chrome: a bar is a
    series by construction, so an unlabelled one is a series that escaped the
    gate.

    Narrowed to `BarContainer` by type rather than by "any container", because
    the alternative refuses every capped error bar in the repository — the
    negative control `test_an_unlabelled_errorbar_companion_is_still_chrome`
    exists to keep that narrowing honest.

    Raises:
        UnreadableSeriesError: Naming what to add and where.
    """
    if not isinstance(container, BarContainer):
        return
    raise UnreadableSeriesError(
        "a figure draws unlabelled bars (Annex 03 §B.4.2): a bar container "
        "carries no legend label, so the greyscale gate has nothing to inspect "
        "and would report green on a figure it never looked at. `ax.bar` puts "
        "the label on the CONTAINER and leaves every patch at `_nolegend_`, so "
        "the label has to be passed to `bar(..., label=...)`. Measured: six "
        "unlabelled bars give 0 encodings, 0 conflicts and a pass"
    )


def _encoding_of_container(label: str, container: Any) -> SeriesEncoding:
    """The encoding of a labelled container — an errorbar or a bar group.

    Raises:
        UnreadableSeriesError: If the container carries a label but exposes no
            artist this can read. Refused rather than skipped: a labelled thing
            the gate cannot inspect is a series that escaped the gate, and
            silently dropping it is exactly how the container case was missed
            in the first place.
    """
    patches = list(getattr(container, "patches", ()) or ())
    if patches:
        face = patches[0]
        return SeriesEncoding(
            label=label,
            # Composited over the white page: a bar at alpha 0.4 prints as its
            # blend with the paper, not as its declared colour, and the gate
            # must compare what is printed.
            color=printed_color(face.get_facecolor(), face.get_alpha()),
            # A bar has no dash pattern and no marker, so its hatch is the only
            # shape channel it owns. `None` and `""` must not collide, hence
            # the explicit empty string for "no hatch".
            linestyle=f"hatch:{face.get_hatch() or ''}",
            marker="bar",
        )
    lines = list(getattr(container, "lines", ()) or ())
    for line in lines:
        if line is None:
            continue
        if isinstance(line, tuple):
            continue
        return SeriesEncoding(
            label=label,
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            dash=dash_pattern_of(line),
        )
    raise UnreadableSeriesError(
        f"series {label!r} is drawn by a {type(container).__name__} the "
        "greyscale gate cannot read. It is refused rather than skipped: a "
        "labelled artist the gate steps over is a series that escaped Annex 03 "
        "§B.4 while the gate reported green"
    )


def printed_color(color: Any, alpha: float | None) -> tuple[float, float, float]:
    """`color` composited over the white page — what it actually prints as.

    A translucent fill is not its own colour on paper. Two bars at alpha 0.3
    whose declared colours differ sharply can print within a hair of each
    other, and the gate compares luminance, so the compositing has to happen
    before the comparison rather than in the reader's eye.
    """
    red, green, blue = to_rgb(color)
    if alpha is None or alpha >= 1.0:
        return (red, green, blue)
    return tuple(alpha * channel + (1.0 - alpha) for channel in (red, green, blue))  # type: ignore[return-value]


@dataclass(frozen=True)
class EncodingDrift:
    """One label drawn two different ways on one figure (§B.4.3).

    Attributes:
        label: The series drawn inconsistently.
        first: Its `(color, linestyle, marker)` where it was first seen.
        second: The triple that disagreed.
    """

    label: str
    first: tuple[Any, Any, Any]
    second: tuple[Any, Any, Any]

    def describe(self) -> str:
        """One line, naming the label and both triples."""
        return (
            f"{self.label!r} is drawn as {self.first} on one axes and "
            f"{self.second} on another"
        )


def encoding_drifts(figure: Figure) -> tuple[EncodingDrift, ...]:
    """Every label drawn with two different encodings on `figure` (§B.4.3).

    §B.4.1 compares *pairs*; this compares a series with itself, and it is what
    makes a second panel non-bypassable. A broken axis whose lower panel painted
    every series in one colour satisfies a per-axes pair check on the upper
    panel and says nothing at all about the lower one.
    """
    seen: dict[str, tuple[Any, Any, Any]] = {}
    drifts: list[EncodingDrift] = []
    for encoding in encodings_of(figure):
        triple = (encoding.color, *encoding.shape)
        first = seen.setdefault(encoding.label, triple)
        if first != triple and not any(
            drift.label == encoding.label for drift in drifts
        ):
            drifts.append(EncodingDrift(encoding.label, first, triple))
    return tuple(drifts)


def figure_conflicts(figure: Figure) -> tuple[GreyscaleConflict, ...]:
    """§B.4.1's pairs, formed within each `Axes`, once per unordered pair."""
    conflicts: list[GreyscaleConflict] = []
    seen: set[tuple[str, str]] = set()
    for group in encodings_by_axes(figure):
        for conflict in greyscale_conflicts(group):
            key = (conflict.first, conflict.second)
            if key in seen:
                continue
            seen.add(key)
            conflicts.append(conflict)
    return tuple(conflicts)


def greyscale_conflicts(
    encodings: Sequence[SeriesEncoding],
) -> tuple[GreyscaleConflict, ...]:
    """Every pair §B.4.1 says a reader could not separate in print.

    Args:
        encodings: The drawn series.

    Returns:
        One conflict per offending unordered pair, in a stable order.
    """
    conflicts: list[GreyscaleConflict] = []
    for first, second in combinations(encodings, 2):
        if first.shape != second.shape:
            continue
        separation = abs(first.luminance - second.luminance)
        if separation < LUMINANCE_FLOOR:
            conflicts.append(GreyscaleConflict(first.label, second.label, separation))
    return tuple(conflicts)


def require_greyscale_separable(figure: Figure, where: str) -> None:
    """Refuse a figure whose series merge when colour is removed.

    Args:
        figure: The rendered figure.
        where: What to name in the message — the figure's id.

    Raises:
        SeriesDriftError: If one label is drawn two ways (§B.4.3). Checked
            first, because a drifted series makes the pair report confusing —
            the same label appears with two encodings and the pairs it forms
            are not the ones the author declared.
        GreyscaleError: Naming every offending pair and its separation, so an
            author is told which two series to re-encode rather than that
            "the figure failed".
    """
    drifts = encoding_drifts(figure)
    if drifts:
        detail = "; ".join(drift.describe() for drift in drifts)
        raise SeriesDriftError(
            f"figure {where!r} draws one series two ways (Annex 03 §B.4.3): "
            f"{detail}. A label is an identity, and a reader who meets it twice "
            "in two encodings has been shown two series"
        )
    conflicts = figure_conflicts(figure)
    if not conflicts:
        return
    detail = "; ".join(conflict.describe() for conflict in conflicts)
    raise GreyscaleError(
        f"figure {where!r} is not separable in greyscale (Annex 03 §B.4): "
        f"{detail}. Every categorical series carries colour AND linestyle AND "
        "marker; colour is the accelerator, never the identity channel"
    )


class GreyscaleError(Exception):
    """A figure that would lose a series in print."""


class SeriesDriftError(GreyscaleError):
    """One label drawn two different ways on one figure (§B.4.3)."""


class UnreadableSeriesError(GreyscaleError):
    """A labelled artist the gate cannot inspect. Refused, never skipped."""
