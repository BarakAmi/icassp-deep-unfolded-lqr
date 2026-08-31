"""A broken axis is one axis interrupted, not two axes (Annex 03 §B.5.1).

§B.5's first rule forbids two vertical axes and its third prescribes clipping.
Both guard the same failure — a scale chosen so that a reader cannot audit what
they are shown — and between them they leave one legitimate case unnamed.

"When a bound sits far from the curves, clipping reports it in the legend and
shows nothing of it. That is the right trade when the bound is *context*. It is
the wrong one when the distance to the bound is the *claim* — a convergence
figure whose whole subject is that a family reaches a floor."

Figure 1 is exactly that figure. Its bound levels sit far below the
iteration-dependent curves, and clipping would show the reader the convergence
this figure exists to show compressed into a few pixels.

The permissions are narrow, because a break has the same power a twin axis
has, and each is a class here.

1. **One quantity, one unit, both panels.** A break removes a *range*; it
   never changes what the axis measures.
2. **Only an empty range may be removed.** A break drawn through data is
   **refused** rather than rendered — "a gap that hides marks is the
   compression §B.5 forbids, performed deliberately".
3. **At most one break per axis**, legible, and **the panels' heights are
   proportional to their ranges**, so one unit of y is the same number of
   pixels in both and the same slope is not drawn at two angles.
4. **DECLARED, never derived.** "A tool may *report* the widest empty range to
   a human choosing; it may not choose." A break inferred from the data would
   move whenever a seed or a contender changed, and a figure whose axis moved
   without any document changing is a figure whose two renders cannot be
   compared.
5. **The reference-line rule inverts.** The lower panel exists precisely to
   hold the references, so it is scaled *to* them.
6. **It exempts nothing.** The bound still carries its value in the legend,
   and the greyscale gate still runs on both panels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import (
    LEVEL_PANEL_HEIGHT,
    axis_scaling,
    widest_empty_range,
)
from mbl.present.greyscale import encodings_by_axes, require_greyscale_separable
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError

ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "cocp_lower_bound": Role.BOUND,
}
ORDER = tuple(ROLES)

#: The shape Figure 1 has: curves near 3.0, a bound at 1.2, nothing between.
BREAK = [1.3, 2.7]


def _table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, base in (("unfolded_alpha", 3.0), ("unfolded_alpha_p", 3.2)):
        for depth in (1, 2, 4, 8):
            rows.append(
                {
                    "contender": label,
                    "role": ROLES[label].value,
                    "axis_value": float(depth),
                    "aggregate": base - 0.02 * depth,
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                }
            )
    rows.append(
        {
            "contender": "cocp_lower_bound",
            "role": Role.BOUND.value,
            "axis_value": np.nan,
            "aggregate": 1.2,
            "interval_low": np.nan,
            "interval_high": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _figure(table: pd.DataFrame | None = None, **config: Any) -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig",
            table=_table() if table is None else table,
            config=config,
            profile=resolve_profile("thesis"),
            series_order=ORDER,
            roles=ROLES,
        )
    )


class TestItIsDeclaredNeverDerived:
    def test_without_a_declaration_the_figure_is_one_panel(self) -> None:
        """Clause 4's real content: the same data must NOT break itself.

        The gap here is enormous and obvious, which is exactly the case an
        inferring implementation would break on — and then a figure's axis
        would move whenever a seed or a contender changed, with no document
        edit to explain it.
        """
        figure = _figure()
        assert len(figure.axes) == 1
        plt.close(figure)

    def test_declaring_one_gives_two_panels(self) -> None:
        figure = _figure(ybreak=BREAK)
        assert len(figure.axes) == 2
        plt.close(figure)

    def test_a_tool_may_report_the_widest_empty_range(self) -> None:
        """ "A tool MAY report the widest empty range to a human choosing; it
        may not choose." So the reporter exists and nothing calls it from the
        render path — it is an author's aid, not a decision."""
        reported = widest_empty_range(_table())
        assert reported is not None
        low, high = reported
        assert low == pytest.approx(1.2)
        assert high == pytest.approx(2.84)

    def test_the_reporter_reports_and_does_not_judge(self) -> None:
        """A dense column still has a widest gap, and the tool still names it.

        Suppressing it below some threshold would be this function choosing
        after all — clause 4 draws the line at *choosing*, not at reporting,
        and the author is the one who can see whether the gap is worth a
        break. Measured here: the widest gap collapses from 1.64 to 0.10 when
        the bound is moved up among the curves, which is the signal an author
        reads.
        """
        table = _table()
        table.loc[table["contender"] == "cocp_lower_bound", "aggregate"] = 2.9
        reported = widest_empty_range(table)
        assert reported is not None
        assert reported[1] - reported[0] < 0.2

    def test_the_reporter_needs_two_distinct_values(self) -> None:
        table = _table()
        table["aggregate"] = 1.0
        assert widest_empty_range(table) is None


class TestOnlyAnEmptyRangeMayBeRemoved:
    def test_a_break_through_data_is_refused(self) -> None:
        """ "A gap that hides marks is the compression §B.5 forbids, performed
        deliberately." Refused rather than rendered, and named so the author
        knows which value is in the way."""
        with pytest.raises(SpecificationError, match="2.84|unfolded_alpha"):
            _figure(ybreak=[2.5, 3.5])

    def test_a_break_touching_a_value_exactly_is_allowed(self) -> None:
        # The boundary from the passing side: the omitted interval is open, so
        # a bound sitting exactly on the lower edge is shown, not hidden. An
        # implementation refusing this would make the natural declaration —
        # "break from the bound to the curves" — impossible to write.
        figure = _figure(ybreak=[1.2, 2.8])
        assert len(figure.axes) == 2
        plt.close(figure)

    def test_an_inverted_range_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="not above"):
            _figure(ybreak=[2.7, 1.3])

    def test_a_malformed_declaration_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="two"):
            _figure(ybreak=[1.0])


class TestOneQuantityOneUnitBothPanels:
    def test_both_panels_share_the_x_axis(self) -> None:
        figure = _figure(ybreak=BREAK)
        upper, lower = figure.axes
        assert lower.get_xlim() == upper.get_xlim()
        plt.close(figure)

    def test_only_the_lower_panel_carries_the_x_label(self) -> None:
        figure = _figure(ybreak=BREAK, xlabel="unrolling depth $J$")
        upper, lower = figure.axes
        assert lower.get_xlabel() == "unrolling depth $J$"
        assert upper.get_xlabel() == ""
        plt.close(figure)

    def test_the_y_label_is_stated_once_for_the_pair(self) -> None:
        """One quantity, so one name. A label on each panel would read as two
        measures, which is the twin axis §B.5 forbids wearing a disguise."""
        figure = _figure(ybreak=BREAK, ylabel="cost")
        labels = [axes.get_ylabel() for axes in figure.axes]
        assert labels.count("cost") <= 1
        assert "cost" in figure.texts[0].get_text() or "cost" in labels
        plt.close(figure)

    def test_the_facing_spines_are_hidden(self) -> None:
        figure = _figure(ybreak=BREAK)
        upper, lower = figure.axes
        assert not upper.spines["bottom"].get_visible()
        assert not lower.spines["top"].get_visible()
        plt.close(figure)

    def test_the_break_is_marked_on_both_panels(self) -> None:
        """Ticks and hidden spines alone leave the discontinuity invisible;
        §B.5.1 requires marks "on both panels"."""
        figure = _figure(ybreak=BREAK)
        upper, lower = figure.axes
        for axes in (upper, lower):
            marks = [
                line
                for line in axes.get_lines()
                if str(line.get_label()).startswith("_break")
            ]
            assert marks, axes


def _has_extent(axes: Any, table: pd.DataFrame) -> bool:
    """Whether the DATA inside `axes`'s limits has a vertical range.

    Read from the table and not from the artists, because every series is
    drawn on every panel by design (a break removes a range, not a series) —
    so the curve's `Line2D` exists on the level panels too, clipped out of
    view, and an artist-based predicate calls every panel ranged.
    """
    low, high = axes.get_ylim()
    values = np.asarray(table["aggregate"], dtype=np.float64)
    inside = values[(values >= low) & (values <= high)]
    return bool(inside.size) and float(inside.max() - inside.min()) > 0.0


def _panel_region(figure: Any) -> float:
    """The combined height the panels share, which the floor is a fraction of.

    Not the figure's height: the title, the axis labels and the legend have
    already taken theirs by the time the panels are laid out.
    """
    boxes = [axes.get_position() for axes in figure.axes]
    return float(boxes[0].union(boxes).height)


class TestThePanelsAreProportionalToTheirRanges:
    def test_one_unit_of_y_is_the_same_height_in_every_panel_that_has_a_range(
        self,
    ) -> None:
        """§B.5.1 clause 3, amended 2026-08-07 — and the reason it matters is
        unchanged: otherwise "the same slope is not drawn at two angles", and a
        reader comparing the gradient of a curve against the flatness of a
        level would be comparing two different scalings of one quantity.

        NARROWED to the panels that carry a range. A panel holding nothing but
        flat levels has no range — its span is invented by the renderer — so
        making its height proportional to that span makes it proportional to a
        rendering choice, and measured, it came out at 6.2 px with no room for
        a tick. Such a panel takes a floor instead, and it has no slope for the
        rule to protect.
        """
        table = _table()
        figure = _figure(table, ybreak=BREAK)
        density = [
            axes.get_position().height / (axes.get_ylim()[1] - axes.get_ylim()[0])
            for axes in figure.axes
            if _has_extent(axes, table)
        ]
        assert len(density) >= 1, "the fixture has no ranged panel to compare"
        assert max(density) == pytest.approx(min(density), rel=0.02), density
        plt.close(figure)

    def test_a_level_only_panel_takes_the_floor_instead(self) -> None:
        """The other half of the amendment: exempt from the ratio, and given a
        height a reader can find the line in."""
        table = _table()
        figure = _figure(table, ybreak=BREAK)
        levels = [axes for axes in figure.axes if not _has_extent(axes, table)]
        for axes in levels:
            assert axes.get_position().height == pytest.approx(
                LEVEL_PANEL_HEIGHT * _panel_region(figure), rel=0.05
            ), axes.get_ylim()
        plt.close(figure)

    def test_the_upper_panel_holds_the_curves(self) -> None:
        figure = _figure(ybreak=BREAK)
        upper, _ = figure.axes
        low, high = upper.get_ylim()
        assert low >= BREAK[1] - 0.2
        assert high >= 3.18
        plt.close(figure)

    def test_the_lower_panel_is_scaled_to_the_references(self) -> None:
        """Clause 5: inside a broken figure §B.5's reference-line rule
        INVERTS. "The lower panel exists precisely to hold the references, so
        it is scaled *to* them."""
        figure = _figure(ybreak=BREAK)
        _, lower = figure.axes
        low, high = lower.get_ylim()
        assert low <= 1.2 <= high
        assert high <= BREAK[0] + 0.2
        plt.close(figure)


#: Two RANGED clusters, spans 0.4 and 0.1, with nothing between: the shape
#: clause 3's re-weighting exists for. Unweighted their heights come out 4:1,
#: which is exactly the complaint the author made about ICASSP Figure 1 —
#: proportionality answers "which panel is this figure about?" with "the
#: widest cluster".
def _two_ranged_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, base, step in (
        ("unfolded_alpha", 3.0, 0.1),  # 3.0 .. 3.4
        ("unfolded_alpha_p", 1.0, 0.025),  # 1.0 .. 1.1
    ):
        for index, depth in enumerate((1, 2, 4, 8)):
            rows.append(
                {
                    "contender": label,
                    "role": ROLES[label].value,
                    "axis_value": float(depth),
                    "aggregate": base + step * index,
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _heights(figure: Any) -> list[float]:
    return [float(axes.get_position().height) for axes in figure.axes]


class TestTheShareMayBeReWeightedByDeclaration:
    """§B.5.1 clause 3 as amended 2026-08-10, on the author's review of the
    rendered ICASSP Figure 1: the panel the figure exists for was drawn at
    less than a third of its neighbour because its cluster is narrower."""

    def test_unweighted_the_wider_cluster_takes_the_taller_panel(self) -> None:
        """The premise, measured — without it the amendment fixes nothing."""
        figure = _figure(_two_ranged_table(), ybreak=BREAK)
        upper, lower = _heights(figure)
        assert upper / lower == pytest.approx(4.0, rel=0.05), (upper, lower)
        plt.close(figure)

    def test_the_declared_weights_set_the_ratio(self) -> None:
        """`weight × span` replaces `span`: 0.4 × 1 against 0.1 × 4 is an
        even split of the same figure."""
        figure = _figure(_two_ranged_table(), ybreak=BREAK, ypanel_weights=[1.0, 4.0])
        upper, lower = _heights(figure)
        assert upper / lower == pytest.approx(1.0, rel=0.05), (upper, lower)
        plt.close(figure)

    def test_the_panels_still_fill_the_same_region(self) -> None:
        """A re-weighting redistributes the figure; it never grows it."""
        plain = _figure(_two_ranged_table(), ybreak=BREAK)
        weighted = _figure(_two_ranged_table(), ybreak=BREAK, ypanel_weights=[1.0, 4.0])
        assert _panel_region(weighted) == pytest.approx(_panel_region(plain), rel=0.02)
        assert sum(_heights(weighted)) == pytest.approx(sum(_heights(plain)), rel=0.02)
        plt.close(plain)
        plt.close(weighted)

    def test_a_weighting_without_a_break_is_refused(self) -> None:
        """One panel already has the whole height, so the numbers reach
        nothing — the inert-declaration rule."""
        with pytest.raises(SpecificationError, match="ypanel_weights"):
            _figure(_two_ranged_table(), ypanel_weights=[1.0])

    def test_the_wrong_count_is_refused_naming_both_numbers(self) -> None:
        with pytest.raises(SpecificationError, match="3 entries for 2 panels"):
            _figure(_two_ranged_table(), ybreak=BREAK, ypanel_weights=[1.0, 2.0, 3.0])

    def test_a_non_positive_weight_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match=r"ypanel_weights.\[1\]"):
            _figure(_two_ranged_table(), ybreak=BREAK, ypanel_weights=[1.0, 0.0])

    def test_a_weight_on_a_level_only_panel_is_refused(self) -> None:
        """Its height is a FLOOR, not a share, so any other number there is a
        declaration that takes no effect — decided after the data is placed,
        which is the only moment it can be."""
        with pytest.raises(SpecificationError, match="floor rather than a share"):
            _figure(_table(), ybreak=BREAK, ypanel_weights=[1.0, 2.0])

    def test_the_level_panel_may_declare_its_own_1_point_0(self) -> None:
        """The declaration is total — one entry per panel — so the exempt
        panel's entry must be spellable rather than omitted."""
        figure = _figure(_table(), ybreak=BREAK, ypanel_weights=[2.0, 1.0])
        assert len(figure.axes) == 2
        plt.close(figure)


class TestItExemptsNothing:
    def test_the_greyscale_gate_runs_on_both_panels(self) -> None:
        figure = _figure(ybreak=BREAK)
        groups = encodings_by_axes(figure)
        assert len(groups) == 2
        assert all(group for group in groups), "a panel with no series under the gate"
        require_greyscale_separable(figure, "fig")
        plt.close(figure)

    def test_a_series_drawn_two_ways_across_the_panels_is_refused(self) -> None:
        """§B.4.3 is what makes the second panel non-bypassable, and a broken
        figure is the case it was written for."""
        from mbl.present.greyscale import SeriesDriftError

        figure = _figure(ybreak=BREAK)
        upper, lower = figure.axes
        for line in lower.get_lines():
            if not str(line.get_label()).startswith("_"):
                line.set_color("#000000")
                line.set_linestyle(":")
        with pytest.raises(SeriesDriftError):
            require_greyscale_separable(figure, "fig")
        plt.close(figure)

    def test_the_bound_still_carries_its_value(self) -> None:
        """Clause 6, unchanged in substance and moved in place.

        Annex 03 §B.5 gained a rule on 2026-08-07: a FLAT series is labelled at
        the right margin rather than in the legend, because on a nine-series
        figure the key took half the canvas. A bound on the canvas therefore
        states its value at the height it is drawn instead of in a legend row.

        Same information, and the assertion is deliberately over BOTH places --
        a bound the axis CLIPS has no height to be labelled at and keeps its
        legend row, which `test_reference_role.py` covers directly. What must
        never happen, and what this catches, is the value appearing in neither.
        """
        figure = _figure(ybreak=BREAK)
        texts = [
            text.get_text()
            for axes in figure.axes
            if axes.get_legend() is not None
            for text in axes.get_legend().get_texts()
        ] + [text.get_text() for axes in figure.axes for text in axes.texts]
        assert any("1.2" in text for text in texts), texts
        plt.close(figure)

    def test_every_series_appears_on_both_panels(self) -> None:
        """A break removes a RANGE, not a series. Drawing the curves only
        above and the bound only below would be two axes with one x, which is
        the thing §B.5.1 exists to distinguish itself from."""
        figure = _figure(ybreak=BREAK)
        drawn = [
            {encoding.label.split(" (")[0] for encoding in group}
            for group in encodings_by_axes(figure)
        ]
        assert drawn[0] == drawn[1] == set(ORDER)
        plt.close(figure)

    def test_the_error_bars_survive_the_break(self) -> None:
        table = _table()
        table["across_seed_spread"] = 0.05
        figure = _figure(table, ybreak=BREAK)
        from matplotlib.container import ErrorbarContainer

        bars = [
            container
            for axes in figure.axes
            for container in axes.containers
            if isinstance(container, ErrorbarContainer)
        ]
        assert bars
        plt.close(figure)

    def test_both_panels_print_their_ticks_the_same_way(self) -> None:
        """One quantity, one number format (clause 1).

        matplotlib chooses a formatter per `Axes` from that panel's own range,
        so the two halves of ONE axis came out at different precision — seen
        on the real render of the tracked frame: `2.30` above and `2.150`
        below. A reader meets that as two quantities measured to different
        accuracy rather than as one axis with a gap in it.
        """
        figure = _figure(ybreak=BREAK)
        figure.canvas.draw()
        decimals = {
            len(text.split(".")[-1])
            for axes in figure.axes
            for text in (str(t.get_text()) for t in axes.get_yticklabels())
            if "." in text
        }
        assert len(decimals) == 1, decimals
        plt.close(figure)


class TestTheSurvivorsMutationTestingFound:
    """Three assertions that held with their mechanism deleted.

    Each was the single-specimen trap in my own fixture: a property asserted on
    data that exhibits it whether or not the code produces it.
    """

    def test_the_panels_are_linked_and_not_merely_equal(self) -> None:
        """`sharex=True` survived deletion, because both panels draw the same
        series and therefore land on the same limits by themselves.

        Equality is the symptom; LINKAGE is the mechanism, and it is what
        §B.5.1's "two stacked panels sharing one x axis" asks for. Perturbing
        one panel is the only way to tell the two apart.
        """
        figure = _figure(ybreak=BREAK)
        upper, lower = figure.axes
        lower.set_xlim(-5.0, 5.0)
        assert upper.get_xlim() == pytest.approx((-5.0, 5.0))
        plt.close(figure)

    def test_panels_of_very_different_span_print_alike(self) -> None:
        """`_share_tick_format` survived, because the default fixture's two
        panels happen to want the same number of decimals anyway.

        The real case does not: on the tracked frame the upper span is 0.114
        and the lower 0.071, and matplotlib chose `2.30` above against `2.150`
        below. This fixture makes the disagreement unmissable — two bounds
        5e-05 apart under curves spread over 0.5.
        """
        roles = {
            "unfolded_alpha": Role.CONTENDER,
            "unfolded_alpha_p": Role.CONTENDER,
            "cocp_lower_bound": Role.BOUND,
            "sdp_floor": Role.BOUND,
        }
        rows: list[dict[str, Any]] = []
        for label, base in (("unfolded_alpha", 3.0), ("unfolded_alpha_p", 3.5)):
            for depth in (1, 2, 4, 8):
                rows.append(
                    {
                        "contender": label,
                        "role": Role.CONTENDER.value,
                        "axis_value": float(depth),
                        "aggregate": base - 0.02 * depth,
                        "interval_low": np.nan,
                        "interval_high": np.nan,
                    }
                )
        for label, value in (("cocp_lower_bound", 1.2), ("sdp_floor", 1.200_05)):
            rows.append(
                {
                    "contender": label,
                    "role": Role.BOUND.value,
                    "axis_value": np.nan,
                    "aggregate": value,
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                }
            )
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=pd.DataFrame(rows),
                config={"ybreak": [1.3, 2.8]},
                profile=resolve_profile("thesis"),
                series_order=tuple(roles),
                roles=roles,
            )
        )
        figure.canvas.draw()
        spans = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in figure.axes]
        assert max(spans) / min(spans) > 100, spans
        decimals = {
            len(text.split(".")[-1])
            for axes in figure.axes
            for text in (str(t.get_text()) for t in axes.get_yticklabels())
            if "." in text
        }
        assert len(decimals) == 1, decimals
        plt.close(figure)

    def test_a_break_through_only_a_bar_end_is_refused(self) -> None:
        """`_values_inside` checking the aggregate alone survived, because the
        default fixture's bars are nowhere near the omitted range.

        A bar's ends are marks too. Here every AGGREGATE sits outside
        [1.3, 2.7] and the lowest bar reaches down to 2.65, inside it — so a
        check that looked only at the aggregates would draw the gap straight
        through a measurement and report nothing.
        """
        table = _table()
        curves = table["axis_value"].notna()
        table.loc[curves, "aggregate"] = 2.75
        table["across_seed_spread"] = 0.0
        table.loc[curves, "across_seed_spread"] = 0.1

        assert (table.loc[curves, "aggregate"] > BREAK[1]).all(), "aggregate is inside"
        with pytest.raises(SpecificationError, match="not empty"):
            _figure(table, ybreak=BREAK)
