"""`Role.REFERENCE` draws as a reference (Annex 03 §B.3.1).

§B.3.1's table reserves an encoding per role:

| Role | Encoding |
|---|---|
| `REFERENCE` (exact optimum) | black, solid, no marker |
| `BOUND` | grey, dashed, annotated with its value in the legend |
| `BASELINE` (attained, non-learned) | slot 8, dash-dot |

`Role.REFERENCE` was defined in `spec/contender.py` and `present/encodings.py`
branched on `BOUND` and `BASELINE` only, so a reference fell through to the
contender arm and was drawn **solid, in a palette colour, with a marker** --
indistinguishable from a learned policy under test. The ICASSP campaign's
Figure 1 declares the unconstrained-Riccati optimum, so the floor the whole
figure is measured against would have read as one more result.

The second half of §B.3.1's sentence is load-bearing too: the reserved role
colours are "**distinct from the contender slots**". A reference therefore
consumes neither a palette slot nor a marker slot. That is not a tidiness
point at these sizes -- Figure 1 draws eight non-bound series against an
eight-colour palette, and freeing one is the difference between a cast that
fits §B.4.2's cycle rule and one that exhausts it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import axis_scaling
from mbl.present.encodings import (
    BOUND_COLOR,
    MARKERS,
    REFERENCE_COLOR,
    REFERENCE_LINESTYLE,
    encode_series,
)
from mbl.present.greyscale import (
    GreyscaleError,
    encodings_of,
    require_greyscale_separable,
)
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role

ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "standard_pgd": Role.BASELINE,
    "truncated_riccati": Role.BASELINE,
    "riccati_unconstrained": Role.REFERENCE,
    "cocp_lower_bound": Role.BOUND,
}
ORDER = tuple(ROLES)


def _table() -> pd.DataFrame:
    """Two swept contenders, one swept baseline, a flat baseline, a flat
    reference and a flat bound."""
    rows: list[dict[str, Any]] = []
    for label, base in (
        ("unfolded_alpha", 3.0),
        ("unfolded_alpha_p", 3.6),
        ("standard_pgd", 3.2),
    ):
        for depth in (1, 2, 4, 8):
            rows.append(
                {
                    "contender": label,
                    "role": ROLES[label].value,
                    "axis_value": float(depth),
                    "aggregate": base - 0.1 * depth,
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                }
            )
    for label, value in (
        ("truncated_riccati", 2.9),
        ("riccati_unconstrained", 1.85),
        ("cocp_lower_bound", 1.2),
    ):
        rows.append(
            {
                "contender": label,
                "role": ROLES[label].value,
                "axis_value": np.nan,
                "aggregate": value,
                "interval_low": np.nan,
                "interval_high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _drawn(figure: Any) -> dict[str, tuple[Any, Any, Any]]:
    return {
        encoding.label.split(" (")[0]: (
            encoding.color,
            encoding.linestyle,
            encoding.marker,
        )
        for encoding in encodings_of(figure)
    }


class TestTheReservedEncoding:
    def test_a_reference_is_black_solid_and_unmarked(self) -> None:
        style = encode_series(ORDER, ROLES)["riccati_unconstrained"]
        assert (style.color, style.linestyle, style.marker) == (
            REFERENCE_COLOR,
            REFERENCE_LINESTYLE,
            "",
        )

    def test_a_reference_takes_no_palette_colour(self) -> None:
        """The failure this item exists to prevent, stated as itself.

        A reference in a palette colour IS a contender to a reader -- the
        colour is the identity channel, and every other series in the figure
        uses it to mean "one of the things being compared".
        """
        styles = encode_series(ORDER, ROLES)
        contenders = {
            styles[label].color
            for label, role in ROLES.items()
            if role in {Role.CONTENDER, Role.BASELINE}
        }
        assert styles["riccati_unconstrained"].color not in contenders

    def test_a_reference_is_not_a_bound(self) -> None:
        """§B.3.1 separates them deliberately: a bound is not an attained
        policy and a reference is. Collapsing them would tell a reader the
        unconstrained optimum is unreachable, which is the *opposite* of what
        it is."""
        styles = encode_series(ORDER, ROLES)
        assert styles["riccati_unconstrained"].color != BOUND_COLOR
        assert (
            styles["riccati_unconstrained"].linestyle
            != styles["cocp_lower_bound"].linestyle
        )


class TestItConsumesNoContenderSlot:
    def test_declaring_a_reference_first_repaints_no_contender(self) -> None:
        """A reference is not a claimant on the palette, so moving it must not
        move anybody. Under the pre-change encoder it took a palette slot and
        a marker slot, so declaring it first shifted both for every series
        after it."""
        reference_first = encode_series(
            (
                "riccati_unconstrained",
                *(name for name in ORDER if name != "riccati_unconstrained"),
            ),
            ROLES,
        )
        reference_late = encode_series(ORDER, ROLES)
        for label in ORDER:
            if label == "riccati_unconstrained":
                continue
            assert reference_first[label] == reference_late[label], label

    def test_removing_the_reference_entirely_repaints_no_contender(self) -> None:
        without = tuple(name for name in ORDER if name != "riccati_unconstrained")
        roles = {k: v for k, v in ROLES.items() if k != "riccati_unconstrained"}
        with_it = encode_series(ORDER, ROLES)
        assert encode_series(without, roles) == {
            label: with_it[label] for label in without
        }

    def test_the_markers_are_the_first_slots_with_no_gap(self) -> None:
        # The structural form of the same claim: a reserved-role series that
        # silently consumed slot i would leave marker i unused, which the
        # equality above can miss if it happens to shift nothing.
        styles = encode_series(ORDER, ROLES)
        markers = [
            styles[label].marker
            for label, role in ROLES.items()
            if role in {Role.CONTENDER, Role.BASELINE}
        ]
        assert markers == list(MARKERS[: len(markers)])


class TestItRendersDistinguishably:
    """Phase B's acceptance clause: a REFERENCE renders distinguishably from a
    BASELINE. Asserted on the DRAWN artists, not on the encoder's return."""

    def test_the_real_renderer_draws_it_apart_from_every_baseline(self) -> None:
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=_table(),
                config={},
                profile=resolve_profile("thesis"),
                series_order=ORDER,
                roles=ROLES,
            )
        )
        drawn = _drawn(figure)
        reference = drawn["riccati_unconstrained"]
        for baseline in ("standard_pgd", "truncated_riccati"):
            assert reference != drawn[baseline]
            # And apart in a channel that SURVIVES colour removal, which is
            # the only difference a printed figure keeps.
            assert reference[1:] != drawn[baseline][1:], baseline
        require_greyscale_separable(figure, "fig")
        plt.close(figure)

    def test_the_whole_cast_passes_the_print_law(self) -> None:
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=_table(),
                config={},
                profile=resolve_profile("thesis"),
                series_order=ORDER,
                roles=ROLES,
            )
        )
        require_greyscale_separable(figure, "fig")
        plt.close(figure)

    def test_two_references_are_refused_rather_than_drawn_alike(self) -> None:
        """The negative control. §B.3.1 reserves ONE encoding for the exact
        optimum, and a study naming two of them is making a claim it cannot
        support -- so the gate must refuse rather than silently draw them as
        one series."""
        order = (*ORDER, "riccati_unconstrained_2")
        roles = {**ROLES, "riccati_unconstrained_2": Role.REFERENCE}
        styles = encode_series(order, roles)
        figure, axes = plt.subplots()
        for label in order:
            style = styles[label]
            axes.plot(
                [1, 2],
                [1, 2],
                label=label,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
            )
        with pytest.raises(GreyscaleError, match="riccati_unconstrained"):
            require_greyscale_separable(figure, "fig")
        plt.close(figure)


class TestAClippedReferenceIsStillReported:
    """§B.3.1, amended 2026-08-06: a reference carries its value too.

    §B.5's third rule scales the axis to the data under study, so a reference
    line outside the contenders' range is CLIPPED — and the legend annotation
    is the only thing that keeps it reported rather than silently absent. The
    table said that of a `BOUND` and nothing of a `REFERENCE`, which stayed
    invisible until one was drawn: on the GRU ablation's standard-tier render
    the unconstrained-Riccati optimum sat at 1.86 under an axis running
    8.2–35.8, and a reader met a legend entry for a line that was neither
    visible nor quantified.
    """

    def test_a_reference_carries_its_value(self) -> None:
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=_table(),
                config={},
                profile=resolve_profile("thesis"),
                series_order=ORDER,
                roles=ROLES,
            )
        )
        texts = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        reference = [text for text in texts if text.startswith("riccati_unconstrained")]
        assert reference and "1.85" in reference[0], texts
        plt.close(figure)

    def test_it_is_reported_even_when_the_axis_clips_it(self) -> None:
        """The case the amendment is about, and the one that could not have
        been noticed from a passing figure: the reference is far below every
        contender, so §B.5 excludes it from the axis range."""
        table = _table()
        table.loc[table["contender"] == "riccati_unconstrained", "aggregate"] = 0.05
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=table,
                config={},
                profile=resolve_profile("thesis"),
                series_order=ORDER,
                roles=ROLES,
            )
        )
        low, _ = figure.axes[0].get_ylim()
        assert low > 0.05, "the axis included the reference, so nothing was clipped"
        texts = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        assert any("0.05" in text for text in texts), texts
        plt.close(figure)

    def test_a_flat_baseline_is_annotated_at_the_margin(self) -> None:
        """REVERSED 2026-08-07, by an Annex 03 §B.5 amendment and on the
        author's evidence.

        This used to assert that a baseline is deliberately NOT annotated: it
        is an attached policy under comparison, and a value in the LEGEND would
        privilege one contender's number over the others'. That reasoning was
        about the legend, and the margin is not the legend -- a flat series is
        now named where it is drawn, in its own colour, with its value, and
        every flat series is treated alike. So the concern the old rule had
        (one contender privileged) is answered by uniformity rather than by
        silence, and the reference figure the author supplied labels its flat
        contenders exactly this way.

        What has NOT changed: a series that VARIES takes a legend row and no
        margin label, because a curve has no single height to be labelled at.
        """
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=_table(),
                config={},
                profile=resolve_profile("thesis"),
                series_order=ORDER,
                roles=ROLES,
            )
        )
        annotations = [t.get_text() for t in figure.axes[0].texts]
        baseline = [t for t in annotations if t.startswith("truncated_riccati")]
        assert baseline, annotations
        # Its value, at the height it is drawn -- not parenthesised, which is
        # the legend's spelling for a series it cannot show.
        assert any(character.isdigit() for character in baseline[0]), baseline
        plt.close(figure)

    def test_the_bound_still_carries_its_own(self) -> None:
        # Neutrality: the role that already worked must keep working.
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=_table(),
                config={},
                profile=resolve_profile("thesis"),
                series_order=ORDER,
                roles=ROLES,
            )
        )
        texts = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
        assert any(t.startswith("cocp_lower_bound") and "1.2" in t for t in texts), (
            texts
        )
        plt.close(figure)
