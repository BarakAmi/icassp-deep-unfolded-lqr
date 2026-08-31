"""How a series is drawn — Annex 03 §B.3.1 and §B.4.

**Colour is assigned by the job it does, and never cycled per plot.** §B.3.1's
sentence that decides this module's whole shape is "a figure that drops a
contender must not repaint the survivors", and the way to satisfy it is to make
every encoding a function of the *study's* declaration rather than of the
figure's series list. A figure that omits one contender therefore changes
nothing about the others, by construction.

`viz.style.semantic`'s registry already holds the project-wide label → colour
map, and its own docstring records the defect §B.3.1 warns about: NB06's
flagship once landed on the untrained baseline's colour purely because of where
it sat in an iteration order. That registry is populated by importing
`applications.styles`, which this module does explicitly — a registry nothing
imported would be empty, and every label would fall through to the positional
fallback that caused the defect in the first place.

**The label decides the colour; the SLOT decides the linestyle and the marker;
role decides only whether a series is in the slot cycle at all.**

That middle clause changed on 2026-08-07. §B.3.1's role table was read as one
linestyle per role — contenders solid, baselines dash-dot — and §B.4.2 said the
opposite in a sentence nobody had implemented: "slot *i* takes palette entry
*i*, linestyle *i* and marker *i*, each cycle at least as long as the series
count". The role reading suits the two-series figure it was written against and
fails at nine. Measured on ICASSP Figure 1: **six of nine** series drawn solid,
and among the **four that vary** — the ones a reader is actually comparing —
**two** linestyles between them, so colour removal left the marker alone to
separate three curves. §B.3.1 is now corrected to say that its linestyle column
describes the RESERVED roles only.

What role still decides is membership: a `REFERENCE` and a `BOUND` are not among
the things being compared, so they take no slot, keep their reserved encodings,
and cannot push a contender's marker along by being declared earlier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typing import Any

from ..spec.contender import Role
from ..viz.style.semantic import registered_series_styles
from ..viz.style.theme import BASELINE_LINESTYLES, CATEGORICAL_PALETTE

# Populating the project-wide registry is the point of this import; the module
# has no other surface. Without it `series_color` falls through to positional
# cycling for every label, which is the defect §B.3.1 exists to prevent.
from ..applications import styles as _styles  # noqa: F401

#: Markers, in §B.4's slot order. At least as long as any contender set this
#: project declares, so the cycle never wraps within one figure.
#:
#: **Twelve, and the first eight are the original eight in their original
#: order.** §B.4.2 pairs slot *i* across all three channels, so extending the
#: cycle may only append: renumbering it would reshape every figure already
#: rendered. The four added are the remaining filled markers matplotlib draws
#: at a legible size — an octagon and a hexagon are too near the circle to
#: separate in print at 4 pt, so the two remaining triangle orientations are
#: preferred over them.
MARKERS = ("o", "s", "^", "D", "v", "p", "*", "P", "X", "h", "<", ">")

#: What §B.3.1 reserves for a computed floor or ceiling: grey, so it cannot be
#: read as a policy, and never a palette slot.
BOUND_COLOR = "0.45"

#: What §B.3.1 reserves for an exact attained optimum: black, solid, no
#: marker. Black rather than a palette slot because a reference is *not* one
#: of the things being compared, and solid rather than dashed because —
#: unlike a bound — it **is** attained. The unmarked line is what keeps it
#: apart from a contender, which shares the solid linestyle: the marker is a
#: channel colour removal cannot touch, so the pair separates in print.
REFERENCE_COLOR = "#000000"
REFERENCE_LINESTYLE = "-"

#: Linestyles, in §B.4's slot order, paired one-to-one with `MARKERS`.
#:
#: **A cycle, not one style per role** (§B.3.1 corrected against §B.4.2,
#: 2026-08-07). Read as "contender = solid, baseline = dash-dot", the role
#: table gives every contender one line: measured on ICASSP Figure 1, **six of
#: nine** series were solid and the **four that actually vary** carried **two**
#: distinct linestyles between them, so removing colour left the marker as the
#: only channel separating three curves. §B.4.2's own sentence — "slot *i*
#: takes palette entry *i*, linestyle *i* and marker *i*" — was already the
#: rule; the implementation followed the other section.
#:
#: Twelve, the length of `MARKERS`, so the capacity guard is one question
#: rather than two, and slot *i* is one encoding rather than a marker from one
#: cycle and a dash from another. Solid is first because slot 0 is the figure's
#: most prominent series. Patterns are given as explicit on/off sequences past
#: the four matplotlib names, because the named ones run out at four and a
#: reader has to tell them apart in print.
LINESTYLES: tuple[Any, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
    (0, (1, 1)),
    (0, (7, 2, 1, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (5, 1, 1, 1)),
    (0, (2, 1)),
    (0, (8, 2, 2, 2)),
)

#: Roles whose encoding §B.3.1 **reserves**, "distinct from the contender
#: slots". These claim neither a palette colour nor a marker, so adding or
#: moving one repaints nothing — and, at Figure 1's size, freeing a colour is
#: the difference between a cast that fits §B.4.2's cycle rule and one that
#: exhausts it.
RESERVED_ROLES = frozenset({Role.BOUND, Role.REFERENCE})

#: Roles whose legend entry carries the value (§B.3.1, amended 2026-08-06).
#:
#: The same set as `RESERVED_ROLES`, and that is not a coincidence worth
#: collapsing: these are the roles §B.5's axis rule may CLIP, and the
#: annotation is what keeps a clipped line reported rather than silently
#: absent. The table annotated only `BOUND` until a `REFERENCE` was actually
#: drawn — on the GRU ablation the unconstrained optimum sat at 1.86 under an
#: 8.2–35.8 axis, so a reader met a legend entry for a line that was neither
#: visible nor quantified.
#:
#: A `BASELINE` is deliberately absent. It is an attained policy under
#: comparison, the axis is scaled to include it, and a value in its legend
#: entry would privilege one contender's number over the others'.
ANNOTATED_ROLES = frozenset({Role.BOUND, Role.REFERENCE})


@dataclass(frozen=True)
class SeriesStyle:
    """The three channels §B.4 requires every categorical series to carry.

    Attributes:
        color: The accelerator channel — never the identity channel.
        linestyle: Survives colour removal; the slot's, from `LINESTYLES`.
            Either a matplotlib name or an ``(offset, on_off_seq)`` dash
            tuple, which is what the cycle needs past its fourth entry.
        marker: Survives colour removal; what keeps two series of one role
            apart. Empty for a bound, which is a line rather than a sampling
            of one.
    """

    color: str
    linestyle: str | tuple[float, tuple[float, ...]]
    marker: str


def encode_series(
    order: Sequence[str], roles: Mapping[str, Role]
) -> dict[str, SeriesStyle]:
    """One encoding per label, keyed by the study's own declaration order.

    Args:
        order: Every contender the **study** declares, in declaration order.
            Not the figure's series: an encoding derived from what a figure
            happens to draw would repaint the survivors whenever one was
            dropped, which is exactly what §B.3.1 forbids.
        roles: Each label's `Role`.

    Returns:
        `label -> SeriesStyle`, for every label in `order`.
    """
    drawn = [
        label
        for label in order
        if roles.get(label, Role.CONTENDER) not in RESERVED_ROLES
    ]
    _require_marker_headroom(drawn)
    colors = _colors(drawn)
    markers = {label: MARKERS[slot] for slot, label in enumerate(drawn)}
    # §B.4.2's pairing: slot `i` is ONE encoding, so the dash pattern comes
    # from the same index the marker does rather than from the series' role.
    linestyles = {label: LINESTYLES[slot] for slot, label in enumerate(drawn)}
    encoded: dict[str, SeriesStyle] = {}
    bound_index = 0
    for label in order:
        role = roles.get(label, Role.CONTENDER)
        if role is Role.BOUND:
            encoded[label] = SeriesStyle(
                color=BOUND_COLOR,
                linestyle=str(
                    BASELINE_LINESTYLES[bound_index % len(BASELINE_LINESTYLES)]
                ),
                marker="",
            )
            bound_index += 1
            continue
        if role is Role.REFERENCE:
            # §B.3.1's reserved encoding, and deliberately NOT cycled: there
            # is one exact optimum, and a study naming two of them is making a
            # claim it cannot support. Two references therefore encode
            # identically and the §B.4 gate refuses the figure by name, which
            # is the report the author needs — a silent second dash pattern
            # would draw the contradiction instead of surfacing it.
            encoded[label] = SeriesStyle(
                color=REFERENCE_COLOR,
                linestyle=REFERENCE_LINESTYLE,
                marker="",
            )
            continue
        encoded[label] = SeriesStyle(
            color=colors[label],
            linestyle=linestyles[label],
            marker=markers[label],
        )
    return encoded


def _require_marker_headroom(drawn: Sequence[str]) -> None:
    """Refuse a study with more drawn series than there are marker slots.

    The rule this replaces was `MARKERS[index % len(MARKERS)]`, and both of
    its parts were wrong in the same direction.

    `index` ran over the **full declaration order**, which the bounds are part
    of. A bound draws no marker, so every bound declared before a contender
    consumed a slot and drew nothing with it — and Figure 1's two bounds
    pushed two drawn series past the end of an eight-entry cycle. Measured over
    the 45 ways to place two bounds among ten declaration slots, **35 raise**.
    The remedy indexes within `drawn`, so a bound's position cannot move any
    other series' marker.

    `% len(MARKERS)` then turned running out of shapes into two series drawn
    identically. That is a silent answer to a question with no answer: the
    greyscale gate reported it as a figure defect, several frames from the
    study document whose contender list actually caused it. Exhaustion is now
    refused by name and at the point of encoding.

    Raises:
        EncodingCapacityError: If more series would be drawn than §B.4.2's
            cycle has slots.
    """
    if len(drawn) <= len(MARKERS):
        return
    raise EncodingCapacityError(
        f"{len(drawn)} series would be drawn and there are {len(MARKERS)} "
        "marker slots (Annex 03 §B.4.2 requires each cycle to be at least as "
        "long as the series count). Colour cannot take over: pairwise "
        "luminance separation of 0.15 is arithmetically impossible above "
        "seven colours, so separability rides on shape. Split the figure, or "
        "extend MARKERS and re-check §B.4.1 for the whole cast"
    )


class EncodingCapacityError(Exception):
    """More series to draw than §B.4.2's encoding cycle has slots."""


#: The three palette colours a bar fill may take, and the whole reason a bar
#: figure puts its contenders on the x axis (§B.4.2, 2026-08-07).
#:
#: A bar has no line and no marker, so the hatch is its only shape channel and
#: §B.4.1 clause 2 decides every pair the hatch does not separate. This
#: palette cannot carry that: of its 28 unordered pairs, **17 sit below the
#: 0.15 luminance floor**, and the largest subset whose members are all
#: mutually separable is **three**. These are it — luminances 0.4349, 0.0727
#: and 0.2781, minimum pairwise separation 0.1568.
#:
#: **Not the registered contender colours, and that is the point.** §B.3.1
#: binds colour to the entity for lines, where linestyle and marker carry
#: identity independently; a fill has neither, so seven contenders cannot be
#: seven fills at any assignment. The contenders go on the x axis instead,
#: where the tick label is an identity channel colour removal cannot touch at
#: all, and these three encode the *categories* being compared.
BAR_FILL_COLORS = ("#eda100", "#4a3aa7", "#eb6834")

#: §B.4.2's textures: 45° and 135° only. The empty string is the absence of a
#: texture rather than a third angle, which is what keeps this inside the two
#: the annex admits while still giving the cycle three shape values.
BAR_HATCHES = ("", "//", "\\\\")


@dataclass(frozen=True)
class FillStyle:
    """How one categorical series' bars are filled (§B.4.2).

    Attributes:
        color: One of `BAR_FILL_COLORS`.
        hatch: One of `BAR_HATCHES`; the bar's only shape channel.
    """

    color: str
    hatch: str


def encode_fills(categories: Sequence[str]) -> dict[str, FillStyle]:
    """One fill per category, in declaration order (§B.4.2).

    **Both channels vary fastest, and the pairs form a Latin square**: slot
    `i` takes colour `i mod 3` and hatch `(i + i // 3) mod 3`, so the first
    three slots differ in *both* channels — which is the case nearly every
    figure is in — while all nine pairs stay distinct. Any two slots
    therefore either differ in colour by at least the luminance floor or
    carry the same colour and differ in hatch. Separability is a property of
    the construction rather than of the particular list, and
    `test_every_pair_of_slots_separates_in_print` computes it rather than
    trusting it.

    *Corrected 2026-08-10, on the author's review of a rendered three-category
    figure.* §B.4.2 rule 2 has always said the first three slots differ in
    both channels; the construction gave them `hatch = i // 3`, which is the
    SAME (empty) texture for all three — three solid bars separated by colour
    alone. Legal under §B.4.1, because those three colours clear the
    luminance floor, and still not what the annex describes: on a greyscale
    print the reader is left with three greys and no texture at all. The
    annex was right and the code was wrong, which is why this is a fix rather
    than an amendment.

    Args:
        categories: The categories, in the order the study declares them.

    Returns:
        `category -> FillStyle`.

    Raises:
        EncodingCapacityError: Past the cycle's nine slots. A modulo wrap
            answers "which fill now?" silently when the honest answer is that
            there is none — the same defect the marker cycle carried until it
            was measured, where two series came out drawn identically and the
            greyscale gate reported it several frames from the document that
            caused it.
    """
    capacity = len(BAR_FILL_COLORS) * len(BAR_HATCHES)
    if len(categories) > capacity:
        raise EncodingCapacityError(
            f"{len(categories)} categories would be drawn as bar fills and "
            f"there are {capacity} fill slots ({len(BAR_FILL_COLORS)} mutually "
            f"separable colours x {len(BAR_HATCHES)} textures). Annex 03 §B.4.2: "
            "a bar's only shape channel is its texture, and pairwise luminance "
            "separation of 0.15 is unavailable above three of this palette's "
            "colours — so this cannot be widened by adding a colour. Split the "
            "figure, or move a distinction onto the x axis"
        )
    return {
        category: FillStyle(
            color=BAR_FILL_COLORS[slot % len(BAR_FILL_COLORS)],
            hatch=BAR_HATCHES[(slot + slot // len(BAR_FILL_COLORS)) % len(BAR_HATCHES)],
        )
        for slot, category in enumerate(categories)
    }


def _colors(labels: Sequence[str]) -> dict[str, str]:
    """One palette slot per label, registered colours first.

    Two passes rather than one, because of a collision measured on the real
    NB04 study: `standard_pgd` is not in the project-wide registry, and the
    positional fallback handed it `CATEGORICAL_PALETTE[1]` -- which is
    `unfolded_learned_step_size`'s REGISTERED colour, so two contenders of one
    figure were the same blue. The triple encoding kept them separable and the
    §B.4 gate passed, which is exactly why it needed catching here: §B.3.1
    wants each contender to read as itself, not merely to be distinguishable.

    So every registered colour is claimed first, and an unregistered label
    takes the first palette slot nobody claimed. The result depends only on
    the study's declaration order, so dropping a series from a FIGURE still
    repaints nothing.
    """
    registered = registered_series_styles()
    assigned = {label: registered[label] for label in labels if label in registered}
    taken = set(assigned.values())
    spare = (slot for slot in CATEGORICAL_PALETTE if slot not in taken)
    for label in labels:
        if label in assigned:
            continue
        assigned[label] = next(spare, CATEGORICAL_PALETTE[0])
    return assigned


def legend_label(
    label: str,
    value: float | None,
    role: Role,
    display_names: Mapping[str, str] | None = None,
) -> str:
    """What the legend says for one series.

    §B.3.1 requires a bound to be "annotated with its value in the legend",
    and §B.5 requires a bound outside the plotted range to be reported there
    rather than allowed to compress every curve — so the value travels with
    the name, and a bound that the axis clips is still fully reported.

    **The one place a label becomes text a reader sees.** Annex 04 §1.3 binds on
    what a reader meets, and this is where the join key (`resolved_label`) is
    exchanged for the display name — at the last possible moment, so that
    everything upstream of it still keys on the one string that indexes the
    table, the colour registry and the encodings.

    Args:
        label: The contender's `resolved_label` — the join key, always.
        value: The bound's attained value, or `None`.
        role: The contender's role.
        display_names: `label -> what a reader is shown`. A missing key falls
            back to the label; the check that a *rendered* figure never shows a
            raw identifier lives with the reader-facing suite, not here.
    """
    shown = (display_names or {}).get(label, label)
    if role in ANNOTATED_ROLES and value is not None:
        return f"{shown} ({value:.4g})"
    return shown
