"""§B.5's margin label: a flat series is named where it is drawn.

The acceptance suite this behaviour shipped without. It was covered only by two
tests reversed in the commit that introduced it, which assert that a value
appears *somewhere* — not where, not in what colour, not that the series it
names still exists in the figure's key exactly once.

**The rule, in the four claims this file is organised around.** A series that is
FLAT across the swept axis is labelled at the right margin, in its own colour,
with its value, at the height it is drawn; only series that VARY take legend
rows; a flat series the axis CLIPS has no height to be labelled at and keeps its
legend row instead; and every series keeps its artist label either way, because
that is what the greyscale gate reads and a figure that dropped it to suppress a
legend row would be inspected on nothing.

Two of these exist because a render was looked at and the figure was wrong, so
each one is asserted on a property of the rendered figure rather than on a call
this module could have made itself.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import _Panels, axis_scaling
from mbl.present.encodings import encode_series
from mbl.present.margin_labels import _strikes_a_mark
from mbl.present.margin_labels import separated as _separated
from mbl.present.greyscale import encodings_of, require_greyscale_separable
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role
from mbl.viz.style.theme import place_legend_outside

#: ICASSP Figure 1's own cast, reduced to what the rule turns on: two series
#: that vary, and five that do not — one of each role that can be flat.
ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "neural": Role.CONTENDER,
    "cocp": Role.CONTENDER,
    "truncated_riccati": Role.BASELINE,
    "cocp_lower_bound": Role.BOUND,
    "riccati_unconstrained": Role.REFERENCE,
}
DISPLAY = {
    "unfolded_alpha": r"UF-$\alpha$",
    "unfolded_alpha_p": r"UF-$\alpha$+P",
    "neural": "GRU",
    "cocp": "COCP",
    "truncated_riccati": "Truncated-Riccati",
    "cocp_lower_bound": "COCP-LB",
    "riccati_unconstrained": "Riccati (unconstrained)",
}

#: The five flat levels, at Figure 1's measured publication values. COCP and
#: its bound are **7e-4 apart**, which is the collision the displacement rule
#: exists for; the two panels either side of them are level-only.
LEVELS = {
    "neural": 8.3010,
    "cocp": 8.2571,
    "cocp_lower_bound": 8.2578,
    "truncated_riccati": 9.1272,
    "riccati_unconstrained": 1.8599,
}

#: Three breaks -> four panels: 1.86 | 8.25-8.31 | 8.83-8.84 | 9.127
BREAK = [[2.0, 8.2], [8.4, 8.7], [8.9, 9.0]]

#: The least separation, in axes fractions, at which this suite accepts two
#: labels as readable. **Deliberately not `MARGIN_LABEL_GAP`.** Importing the
#: renderer's own constant makes an assertion that moves with the code it
#: checks: a mutant setting it to zero let two labels print on top of each
#: other with the suite still green. A floor stated here is a second place the
#: decision has to be made.
MINIMUM_LEGIBLE_GAP = 0.04


def _row(label: str, value: float, axis_value: float | None = None) -> dict[str, Any]:
    return {
        "contender": label,
        "role": ROLES[label].value,
        "axis_value": np.nan if axis_value is None else float(axis_value),
        "aggregate": float(value),
        "across_seed_spread": 0.0,
        "interval_low": float(value),
        "interval_high": float(value),
        "n_seeds": 5,
        "n_trajectories": 32768,
    }


def _table() -> pd.DataFrame:
    rows = []
    for depth in (1, 2, 3, 5, 8, 10):
        rows.append(_row("unfolded_alpha", 8.8294 + 0.001 * depth, axis_value=depth))
        rows.append(_row("unfolded_alpha_p", 8.2959 - 0.0016 * depth, axis_value=depth))
    rows.extend(_row(label, value) for label, value in LEVELS.items())
    return pd.DataFrame(rows)


def _render(
    config: dict[str, Any] | None = None, table: pd.DataFrame | None = None
) -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig1",
            table=_table() if table is None else table,
            config={"ybreak": BREAK} if config is None else config,
            profile=resolve_profile("ieee-2col"),
            series_order=list(ROLES),
            roles=dict(ROLES),
            display_names=dict(DISPLAY),
        )
    )


@pytest.fixture(autouse=True)
def _close_every_figure():
    """matplotlib warns past twenty open figures and this repository runs
    pytest at zero warnings, so a leak here fails whichever file runs next."""
    yield
    plt.close("all")


def _margin_labels(figure: Any) -> dict[str, tuple[Any, float]]:
    """display name -> (the annotation, the axes fraction it is anchored at)."""
    return {
        text.get_text().rsplit(" ", 1)[0]: (text, float(text.xy[1]))
        for axes in figure.axes
        for text in axes.texts
    }


def _legends(figure: Any) -> list[Any]:
    """Every legend on the figure, wherever it was placed."""
    return [axes.get_legend() for axes in figure.axes if axes.get_legend()] + list(
        figure.legends
    )


def _legend_rows(figure: Any) -> list[str]:
    """Every legend row, wherever the legend ended up."""
    return [
        text.get_text() for legend in _legends(figure) for text in legend.get_texts()
    ]


def _which(text: str) -> str | None:
    """Which contender a piece of figure text names, margin or legend.

    Both forms are prefixed by the display name — the margin appends a value
    and the legend parenthesises one — so the longest display name the text
    begins with is the series it belongs to. Longest, because `COCP` is a
    prefix of `COCP-LB` and a shorter match would attribute the bound's label
    to the contender.
    """
    matches = [
        label
        for name, label in ((DISPLAY[key], key) for key in DISPLAY)
        if text == name or text.startswith(name + " ")
    ]
    return max(matches, key=lambda label: len(DISPLAY[label])) if matches else None


def _panel_of(figure: Any, value: float) -> Any:
    for axes in figure.axes:
        low, high = axes.get_ylim()
        if low <= value <= high:
            return axes
    return None


def _natural(axes: Any, value: float) -> float:
    low, high = axes.get_ylim()
    return (value - low) / (high - low)


def _line_pixel(axes: Any, value: float) -> float:
    """The pixel row a level's line is drawn at."""
    return float(axes.transData.transform((0.0, value))[1])


def _assert_hugs_from_above(figure: Any, key: str) -> None:
    """§B.5 as corrected: the label's ink sits immediately ABOVE its own line
    — strictly clear of it (a zero pad prints a struck-through label), and by
    no more than its own text height (an unconditionally-displaced label does
    not hug anything, which is the mutant the outside-placement suite caught
    at 0.91 of a panel whose line sat at 0.06)."""
    figure.canvas.draw()
    value = LEVELS[key]
    axes = _panel_of(figure, value)
    assert axes is not None, key
    box = _margin_labels(figure)[DISPLAY[key]][0].get_window_extent()
    line = _line_pixel(axes, value)
    assert box.y0 >= line + 1.0, (key, box.y0, line)
    assert box.y0 - line <= box.height, (key, box.y0 - line, box.height)


class TestAFlatSeriesIsNamedAtTheMargin:
    def test_every_flat_series_on_the_canvas_carries_one(self) -> None:
        figure = _render()
        assert set(_margin_labels(figure)) == {DISPLAY[key] for key in LEVELS}

    def test_no_series_that_varies_carries_one(self) -> None:
        """A curve has no single height to be labelled at, which is the whole
        reason the rule is about flat series rather than about all of them."""
        figure = _render()
        named = set(_margin_labels(figure))
        assert DISPLAY["unfolded_alpha"] not in named
        assert DISPLAY["unfolded_alpha_p"] not in named

    def test_the_label_carries_the_value(self) -> None:
        """At four significant figures, which is a resolution the data needs:
        COCP and its bound differ in the fourth (8.2571 against 8.2578), so
        two decimals would print one number twice."""
        figure = _render()
        texts = {text.get_text() for axes in figure.axes for text in axes.texts}
        assert "COCP 8.257" in texts, texts
        assert "COCP-LB 8.258" in texts, texts

    def test_the_label_uses_the_display_name_and_not_the_join_key(self) -> None:
        """Annex 04 §1.3: a reader never meets an identifier."""
        figure = _render()
        texts = " ".join(text.get_text() for a in figure.axes for text in a.texts)
        for key in LEVELS:
            assert key not in texts, texts

    def test_each_label_wears_its_own_series_colour(self) -> None:
        """§B.5 says "in its own colour" — the label is the series' name, and a
        uniform ink colour would make five of them one voice."""
        figure = _render()
        styles = encode_series(list(ROLES), ROLES)
        labels = _margin_labels(figure)
        for key in LEVELS:
            annotation, _ = labels[DISPLAY[key]]
            assert annotation.get_color() == styles[key].color, key
        # And they are not all the same, so a hardcoded colour cannot pass.
        assert len({labels[DISPLAY[key]][0].get_color() for key in LEVELS}) >= 3

    def test_it_is_anchored_at_the_left_edge(self) -> None:
        """§B.5 as amended 2026-08-10 (the author's review of the rendered
        figure): left-aligned INSIDE the axes, flush against the LEFT spine —
        `ha = "left"` on an anchor at that spine."""
        figure = _render()
        for axes in figure.axes:
            for text in axes.texts:
                assert text.xycoords == "axes fraction"
                assert float(text.xy[0]) == pytest.approx(0.0)
                assert text.get_ha() == "left"

    def test_it_is_drawn_inside_the_axes_it_belongs_to(self) -> None:
        """INSIDE, measured on the rendered canvas rather than inferred from
        the anchor: an anchor on the right spine with a zero offset would
        satisfy every assertion above and print over the spine.

        **Strictly inside, by at least a pixel.** The outside-placement twin
        of this test caught a mutant deleting the offset — the label then
        touches the spine and the two inks run together; the inside placement
        keeps the same strictness on the other side of the spine.
        """
        figure = _render()
        figure.canvas.draw()
        for axes in figure.axes:
            box = axes.get_window_extent()
            for text in axes.texts:
                extent = text.get_window_extent()
                clearance = box.x1 - extent.x1
                assert clearance >= 1.0, (text.get_text(), clearance)
                assert extent.x0 >= box.x0, (text.get_text(), extent.x0, box.x0)

    def test_the_saved_artifact_contains_the_whole_label(self) -> None:
        """Under the outside placement this pinned the save-time crop, which
        was all that kept five labels overflowing the canvas by 66–190 px in
        the stored PNG. Inside placement makes containment a property of the
        FIGURE, and this asserts it: every label's ink lies within the
        canvas, so no crop arithmetic is load-bearing any more.
        """
        figure = _render()
        figure.canvas.draw()
        canvas = figure.get_window_extent()
        for axes in figure.axes:
            for text in axes.texts:
                extent = text.get_window_extent()
                assert extent.x0 >= canvas.x0 and extent.x1 <= canvas.x1, (
                    text.get_text()
                )
                assert extent.y0 >= canvas.y0 and extent.y1 <= canvas.y1, (
                    text.get_text()
                )

    def test_it_is_placed_on_the_panel_whose_range_holds_its_value(self) -> None:
        figure = _render()
        for axes in figure.axes:
            low, high = axes.get_ylim()
            for text in axes.texts:
                key = _which(text.get_text())
                assert key is not None, text.get_text()
                assert low <= LEVELS[key] <= high, (key, low, high)

    def test_it_is_drawn_at_its_own_line(self) -> None:
        """The claim §B.5 actually makes, as corrected 2026-08-09: the label
        hugs its line from above. Asserted on the three levels that have no
        neighbour close enough to trigger the near-pair rule — the crowded
        pair is its own class's subject."""
        figure = _render()
        for key in ("neural", "truncated_riccati", "riccati_unconstrained"):
            _assert_hugs_from_above(figure, key)

    def test_every_label_below_the_topmost_one_also_hugs_its_line(self) -> None:
        """The case the test above cannot see, and a mutant found it: every
        level it checks is the TOPMOST on its panel, and displacement only ever
        moves a label that has one above it. A rule that displaced
        unconditionally therefore passed — measured, it put COCP-LB at 0.91 of
        the panel while its line was drawn at 0.06. Adjacency-to-the-line
        keeps that power: an unconditionally-displaced label hugs nothing.

        Three levels far enough apart that none may move, on one panel.
        """
        figure = _render(config={}, table=_spread())
        figure.canvas.draw()
        axes = figure.axes[0]
        labels = _margin_labels(figure)
        for key, value in _SPREAD.items():
            box = labels[DISPLAY[key]][0].get_window_extent()
            line = _line_pixel(axes, value)
            assert box.y0 >= line + 1.0, key
            assert box.y0 - line <= box.height, key

    def test_a_partially_swept_series_is_not_a_level(self) -> None:
        """ "Flat" is EVERY axis value being absent, not any of them. A series
        whose run failed at one depth still has nine points and a shape; read
        the other way it would be drawn as one horizontal line through the
        first of them, with the other eight silently gone."""
        table = _table()
        gap = (table["contender"] == "unfolded_alpha") & (table["axis_value"] == 3.0)
        table.loc[gap, "axis_value"] = np.nan
        figure = _render(config={}, table=table)
        assert DISPLAY["unfolded_alpha"] not in _margin_labels(figure)
        assert DISPLAY["unfolded_alpha"] in _legend_rows(figure)


class TestOnlyASeriesThatVariesTakesALegendRow:
    def test_the_legend_names_exactly_the_curves(self) -> None:
        figure = _render()
        assert set(_legend_rows(figure)) == {
            DISPLAY["unfolded_alpha"],
            DISPLAY["unfolded_alpha_p"],
        }

    def test_the_figure_gets_its_canvas_back(self) -> None:
        """The author's report, and the measurement that made this a rule: the
        figures are "very narrow". Rendered both ways — the pre-amendment call
        put all seven series in an outside legend — the axes go from 0.380 of
        the figure's width to 0.878, a factor of 2.31.

        Asserted as a comparison rather than against a threshold, so it does
        not encode this matplotlib version's idea of how wide a legend is.
        """

        def pre_amendment(panels: Any, *, only: Any) -> None:
            """What the renderer did before §B.5 gained the margin label: one
            legend, every series in it, anchored outside the first panel."""
            place_legend_outside(panels.figure, panels.axes[0])

        with mock.patch.object(_Panels, "place_legend", pre_amendment):
            before = _axes_width_fraction(_render())
        after = _axes_width_fraction(_render())
        assert after > 1.5 * before, (before, after)


class TestEverySeriesIsNamedExactlyOnce:
    def test_in_the_legend_or_at_the_margin_and_never_neither(self) -> None:
        """The invariant that catches the defect this feature shipped with and
        was fixed before commit: a clipped flat series took no legend row
        *and* no margin label, and vanished from the figure's key entirely."""
        figure = _render()
        named = [_which(text) for text in _legend_rows(figure)]
        named += [
            _which(text.get_text()) for axes in figure.axes for text in axes.texts
        ]
        assert sorted(name for name in named if name) == sorted(ROLES)

    def test_and_never_both(self) -> None:
        figure = _render()
        legend = {_which(text) for text in _legend_rows(figure)}
        margin = {
            _which(text.get_text()) for axes in figure.axes for text in axes.texts
        }
        assert not legend & margin, legend & margin


class TestAClippedFlatSeriesKeepsItsLegendRow:
    """§B.3.1: a bound the axis rule draws off the canvas states its value in
    the legend. The margin cannot serve it — there is no height to label.

    The fixture needs no arrangement. Drop the break and §B.5's axis rule
    scales the one panel to the data under study, 8.257–9.127; the
    unconstrained floor at 1.860 is a REFERENCE and falls outside it, which is
    the case the rule exists for.
    """

    def test_the_fixture_really_does_clip_it(self) -> None:
        """Stated first, because everything below is vacuous if it does not —
        and a test whose premise silently stopped holding is a failure mode
        this repository has met in a fixture built by `str.replace`."""
        figure = _render(config={})
        low, high = figure.axes[0].get_ylim()
        assert not low <= LEVELS["riccati_unconstrained"] <= high, (low, high)

    def test_it_takes_no_margin_label(self) -> None:
        figure = _render(config={})
        assert DISPLAY["riccati_unconstrained"] not in _margin_labels(figure)

    def test_it_keeps_its_legend_row(self) -> None:
        """Also the regression test for a defect found by looking: the legend
        filter matched contender KEYS while artists carry DISPLAY labels, and
        a reference's label is decorated with its value — so every decorated
        label was silently dropped, this one included."""
        figure = _render(config={})
        assert any(
            _which(text) == "riccati_unconstrained" for text in _legend_rows(figure)
        ), _legend_rows(figure)

    def test_the_legend_row_carries_its_value(self) -> None:
        figure = _render(config={})
        row = [
            text
            for text in _legend_rows(figure)
            if _which(text) == "riccati_unconstrained"
        ]
        assert row and "1.86" in row[0], row

    def test_the_series_on_the_canvas_are_still_at_the_margin(self) -> None:
        """The clipped case must not turn the rule off for everything else."""
        figure = _render(config={})
        assert DISPLAY["neural"] in _margin_labels(figure)


class TestTheGreyscaleGateStillSeesEverySeries:
    def test_every_series_keeps_its_artist_label(self) -> None:
        """The annex states this as the thing the implementation must not reach
        for: suppressing a legend row by clearing the artist's label would
        leave the gate inspecting nothing, which this project shipped once."""
        figure = _render()
        seen = {_which(encoding.label) for encoding in encodings_of(figure)}
        assert seen == set(ROLES), seen

    def test_and_on_an_UNBROKEN_figure_too(self) -> None:
        """Asserted twice on purpose, because the broken figure cannot see it.
        Every series is drawn on EVERY panel, so an implementation that cleared
        labels to suppress legend rows would clear them only on the panel it
        asked for handles from — and the gate, which walks the whole figure,
        would still find three untouched copies. Measured: that mutant survived
        the four-panel fixture and dies here, where there is one copy.
        """
        figure = _render(config={})
        seen = {_which(encoding.label) for encoding in encodings_of(figure)}
        assert seen == set(ROLES), seen

    def test_the_figure_passes_the_gate(self) -> None:
        require_greyscale_separable(_render(), "fig1")


class TestTwoLevelsTooCloseToPrintApart:
    """The displacement rule, amended into §B.5 on 2026-08-07. COCP and its
    bound are 7e-4 apart on a panel spanning 0.048 — 1.5 % of the height."""

    def test_their_ink_does_not_overlap(self) -> None:
        """Asserted on the rendered boxes, deliberately NOT on the module's own
        `MARGIN_LABEL_GAP`.

        Written that way first and a mutant setting the constant to zero
        survived: the assertion imported the very number the implementation
        uses, so the two moved together and the labels printed on top of each
        other while the test agreed. Same family as this project's
        default-passed-back-as-the-override trap. What the rule is about is two
        readable labels, and pixels are where that is true or false.
        """
        figure = _render()
        figure.canvas.draw()
        boxes = [
            text.get_window_extent() for axes in figure.axes for text in axes.texts
        ]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                assert not first.overlaps(second), (first, second)

    def test_they_are_separated_by_a_legible_fraction(self) -> None:
        """The same claim in axes fractions, against a literal chosen here.

        `MINIMUM_LEGIBLE_GAP` is not imported from the renderer on purpose —
        see the test above. It is a floor on what this suite will accept, so
        lowering the renderer's spacing is a decision that has to be made
        twice.
        """
        labels = _margin_labels(_render())
        gap = abs(labels["COCP"][1] - labels["COCP-LB"][1])
        assert gap >= MINIMUM_LEGIBLE_GAP, gap

    def test_the_displacement_never_reorders(self) -> None:
        """A label names the height it is drawn at, so the order down the
        margin is the order of the values. COCP-LB is the larger."""
        figure = _render()
        labels = _margin_labels(figure)
        assert LEVELS["cocp_lower_bound"] > LEVELS["cocp"]
        assert labels["COCP-LB"][1] > labels["COCP"][1]

    def test_an_uncrowded_label_is_not_displaced(self) -> None:
        """Separation is bought only where it is needed: the GRU is 0.04 above
        the pair and keeps hugging its own line from above."""
        _assert_hugs_from_above(_render(), "neural")

    def test_a_whole_cluster_stays_ordered_and_separated(self) -> None:
        figure = _render(config={}, table=_cluster())
        placed = _margin_labels(figure)
        heights = [placed[DISPLAY[key]][1] for key in _CLUSTER]
        assert heights == sorted(heights, reverse=True), heights
        for upper, lower in zip(heights, heights[1:], strict=False):
            assert upper - lower >= MINIMUM_LEGIBLE_GAP, heights

    def test_no_label_is_displaced_below_its_panel(self) -> None:
        """The half of the rule the implementation was missing, measured:
        four levels within 6e-4 of each other at the bottom of a panel a swept
        curve had made wide were placed at 0.103, 0.058, 0.013 and **-0.032**
        — the last one twenty pixels below its own axes. On a broken figure
        that strip is the next panel, so the label names a height that panel's
        range excludes."""
        figure = _render(config={}, table=_cluster())
        for _, fraction in _margin_labels(figure).values():
            assert fraction >= 0.0, _margin_labels(figure)

    def test_no_label_is_displaced_above_its_panel(self) -> None:
        """The other end of the same invariant, on the same fixture and not on
        a mirrored one: making room at the floor is paid for upwards, so this
        is where a label can leave the top."""
        figure = _render(config={}, table=_cluster())
        for _, fraction in _margin_labels(figure).values():
            assert fraction <= 1.0, _margin_labels(figure)

    def test_a_crowded_cluster_still_lands_inside_the_axes(self) -> None:
        """The anchor is in [0, 1]; this asserts the ink is, which is what the
        reader actually meets."""
        figure = _render(config={}, table=_cluster())
        figure.canvas.draw()
        for axes in figure.axes:
            box = axes.get_window_extent()
            for text in axes.texts:
                centre = 0.5 * (
                    text.get_window_extent().y0 + text.get_window_extent().y1
                )
                assert box.y0 <= centre <= box.y1, text.get_text()


class TestTheNearPairSplitsAboveAndBelow:
    """§B.5's 2026-08-09 near-pair rule, on Figure 1's own collision: COCP and
    its bound are 7e-4 apart, far less than a text height on their panel, so
    both labels above would print the lower text through the upper line. The
    LOWER line's label goes below its line; each still hugs its own."""

    def test_the_upper_label_sits_above_its_line(self) -> None:
        figure = _render()
        figure.canvas.draw()
        value = LEVELS["cocp_lower_bound"]
        axes = _panel_of(figure, value)
        box = _margin_labels(figure)["COCP-LB"][0].get_window_extent()
        assert box.y0 >= _line_pixel(axes, value) + 1.0, box

    def test_the_lower_label_sits_below_its_line(self) -> None:
        figure = _render()
        figure.canvas.draw()
        value = LEVELS["cocp"]
        axes = _panel_of(figure, value)
        box = _margin_labels(figure)["COCP"][0].get_window_extent()
        assert box.y1 <= _line_pixel(axes, value) - 1.0, box

    def test_neither_text_strikes_either_line(self) -> None:
        """The rule's purpose, asserted directly: no label's ink spans the
        pixel row of either of the pair's lines — a struck-through label is
        exactly what the above/below split exists to prevent."""
        figure = _render()
        figure.canvas.draw()
        axes = _panel_of(figure, LEVELS["cocp"])
        lines = [_line_pixel(axes, LEVELS[key]) for key in ("cocp", "cocp_lower_bound")]
        labels = _margin_labels(figure)
        for name in ("COCP", "COCP-LB"):
            box = labels[name][0].get_window_extent()
            for line in lines:
                assert not (box.y0 < line < box.y1), (name, line, box)

    def test_a_pair_at_the_panel_floor_IS_GIVEN_ROOM_below(self) -> None:
        """§B.5 as amended 2026-08-10: where the below-position does not fit,
        the PANEL makes room — the rule does not give way.

        The retrained Figure 1's own geometry, which the class above cannot
        see: the near pair sits against its panel's floor. The first
        inside-placement version parked a box across its own line; the second
        reverted the flip to above, which is the near-pair rule quietly
        declining on exactly the figure it was written for. A panel's lower
        limit is a rendering choice, so the reservation ADDS range and the
        lower label goes below its line as declared.
        """
        table = _table()
        # The pair sits ON the panel's lower break edge: no room below at all
        # until the reservation makes some.
        table.loc[table["contender"] == "cocp", "aggregate"] = 8.2004
        table.loc[table["contender"] == "cocp_lower_bound", "aggregate"] = 8.2011
        figure = _render(config={"ybreak": BREAK}, table=table)
        figure.canvas.draw()
        axes = _panel_of(figure, 8.2004)
        labels = _margin_labels(figure)
        lines = [_line_pixel(axes, value) for value in (8.2004, 8.2011)]

        # The lower member is BELOW its own line, inside its panel, and
        # neither text strikes either line.
        lower = labels["COCP"][0].get_window_extent()
        assert lower.y1 <= _line_pixel(axes, 8.2004), "the lower label is not below"
        upper = labels["COCP-LB"][0].get_window_extent()
        assert upper.y0 >= _line_pixel(axes, 8.2011), "the upper label is not above"
        panel = axes.get_window_extent()
        for name, box in (("COCP", lower), ("COCP-LB", upper)):
            for line in lines:
                assert not (box.y0 < line < box.y1), (name, line, box)
            assert box.y0 >= panel.y0 and box.y1 <= panel.y1, (name, "left panel")

    def test_the_reservation_only_ever_adds_range(self) -> None:
        """The one thing that must never move is the mark.

        Asserted on the rendered panel rather than by comparing two renders:
        every vertex any series draws in that panel is still inside its
        limits, the floor did drop below the lowest level (room was made),
        and the room taken is smaller than the panel it was taken in — a
        reservation that dwarfed the data would be a compression by another
        name.
        """
        table = _table()
        table.loc[table["contender"] == "cocp", "aggregate"] = 8.2004
        table.loc[table["contender"] == "cocp_lower_bound", "aggregate"] = 8.2011
        figure = _render(config={"ybreak": BREAK}, table=table)
        axes = _panel_of(figure, 8.2004)
        low, high = axes.get_ylim()

        # Every value this panel's cluster holds, named rather than filtered
        # off the canvas: the two crowded levels, the GRU level above them,
        # and both ends of the swept curve that shares the panel. A filter by
        # the panel's own limits would assert its own conclusion — every
        # series is drawn on every panel by design (a break removes a range,
        # not a series), so the other clusters' vertices are on this axes too.
        held = [
            8.2004,
            8.2011,
            LEVELS["neural"],
            8.2959 - 0.0016 * 1,
            8.2959 - 0.0016 * 10,
        ]
        assert min(held) >= low and max(held) <= high, (min(held), max(held), low, high)
        assert low < 8.2004, "no room was made below the lowest level"
        assert (8.2004 - low) < (high - 8.2004), "the reservation took the panel"


class TestALabelNeverLiesAcrossAMark:
    """§B.5 as amended 2026-08-10: the left spine is the preference, and "a
    label that would strike a mark on the left is a defect wherever the rule
    prefers". Measured on the real Figure 1 the moment the side changed — the
    recurrent baseline's level sits 0.005 under the proposed controller's
    first point, so its label lay across both proposed curves.

    The remedy is HORIZONTAL: the height is what names the value, so the
    label slides along its own line and never off it.
    """

    def _crossing(self) -> pd.DataFrame:
        """Figure 1's own geometry: a curve that starts ABOVE a level and
        descends through it, so the level's left-spine label is under the
        curve's first segment."""
        table = _table()
        rows = table["contender"] == "unfolded_alpha_p"
        # Depth 1 just above the level, then a steep drop through it — so the
        # crossing happens within the first tenth of the axis, exactly where
        # a left-spine label sits.
        table.loc[rows, "aggregate"] = [
            LEVELS["neural"] + 0.004,
            LEVELS["neural"] - 0.011,
            LEVELS["neural"] - 0.014,
            LEVELS["neural"] - 0.016,
            LEVELS["neural"] - 0.018,
            LEVELS["neural"] - 0.019,
        ]
        return table

    def test_the_struck_label_slides_right_until_it_is_clear(self) -> None:
        figure = _render(config={"ybreak": BREAK}, table=self._crossing())
        figure.canvas.draw()
        axes = _panel_of(figure, LEVELS["neural"])
        text = _margin_labels(figure)["GRU"][0]
        assert _strikes_a_mark(axes, text) is False, "the label lies on a curve"
        assert float(text.xy[0]) > 0.0, "it never moved, so nothing was fixed"

    def test_it_slides_along_its_own_line_and_not_off_it(self) -> None:
        """The height is the claim; only x may move."""
        figure = _render(config={"ybreak": BREAK}, table=self._crossing())
        figure.canvas.draw()
        axes = _panel_of(figure, LEVELS["neural"])
        box = _margin_labels(figure)["GRU"][0].get_window_extent()
        line = _line_pixel(axes, LEVELS["neural"])
        assert box.y0 >= line, "the slide moved the label off its own line"
        assert box.y0 - line <= box.height, "it stopped hugging the line"

    def test_an_uncrowded_label_stays_at_the_left_spine(self) -> None:
        """The slide is a remedy, not a default: with nothing in the way the
        preference §B.5 states is what a reader gets."""
        figure = _render()
        figure.canvas.draw()
        for name in ("COCP", "COCP-LB", "Truncated-Riccati"):
            assert float(_margin_labels(figure)[name][0].xy[0]) == 0.0, name


class TestTheLegendGoesWhereTheMarginIsNot:
    def test_it_is_inside_when_a_flat_series_is_labelled(self) -> None:
        """They compete for the same strip and collided on the real figure —
        "Truncated-Riccati 9.127" printed over the legend box."""
        figure = _render()
        figure.canvas.draw()
        axes = [a for a in figure.axes if a.get_legend() is not None][0]
        legend = axes.get_legend().get_window_extent()
        assert legend.x1 <= axes.get_window_extent().x1

    def test_it_is_inside_also_when_nothing_is_flat(self) -> None:
        """§B.5: the legend leaves the figure only when it cannot fit inside,
        "and then it goes below rather than beside". The first implementation
        sent an all-curves figure's legend BESIDE the axes — the margin-based
        gate read "inside when the margin is spoken for" where the annex says
        inside always — and ICASSP Figure 3, the first tracked all-curves
        figure, rendered with its axes on 45 % of the canvas. Corrected on
        the author's review of that render (2026-08-09)."""
        curves = _table()
        figure = _render(config={}, table=curves[curves["axis_value"].notna()])
        figure.canvas.draw()
        axes = [a for a in figure.axes if a.get_legend() is not None][0]
        legend = axes.get_legend().get_window_extent()
        assert legend.x1 <= axes.get_window_extent().x1

    def test_a_figure_of_curves_only_is_annotated_nowhere(self) -> None:
        curves = _table()
        figure = _render(config={}, table=curves[curves["axis_value"].notna()])
        assert _margin_labels(figure) == {}


class TestTheLegendSitsInsideAtTheTopCentre:
    """§B.5, corrected 2026-08-07 on the author's review of what the first
    version produced.

    That version read "a legend never covers a mark, and where none such place
    exists it goes below the figure" — and below costs a quarter of the plot:
    the axes take 59.5 % of the figure's height with the legend below and
    74.5 % with it inside. A figure whose subject is a convergence of 0.0686 %
    cannot spend that on a key.

    Two things make the overlap acceptable and both are asserted here: the
    legend is at the TOP CENTRE, so what it can cover is the middle of the
    swept axis and never the ends — which on a cost-versus-depth figure are the
    claim — and it is TRANSLUCENT, so a mark behind it is dimmed rather than
    deleted.
    """

    def test_it_is_inside_the_axes(self) -> None:
        figure = _render_crowded()
        assert not figure.legends, "the legend was relegated below"
        assert any(axes.get_legend() is not None for axes in figure.axes)

    def test_it_is_translucent(self) -> None:
        """Asserted against the value the renderer DECLARES, not against
        "somewhere between 0 and 1".

        Written the loose way first and a mutant deleting the whole pathway
        survived: matplotlib's own `legend.framealpha` default is 0.8, so a
        legend that was never told anything reads as translucent too. The
        input has to differ from the fallback in the property under test —
        this repository's recurring default-passed-back-as-the-override trap,
        in a new place. `test_the_declared_opacity_is_not_matplotlibs_default`
        below is what keeps this assertion unsatisfiable by the default.
        """
        from mbl.present.axis_scaling import LEGEND_ALPHA

        figure = _render_crowded()
        legend = next(a.get_legend() for a in figure.axes if a.get_legend())
        assert legend.get_frame().get_alpha() == pytest.approx(LEGEND_ALPHA)

    def test_the_declared_opacity_is_not_matplotlibs_default(self) -> None:
        """So that the test above cannot pass on a figure nobody told."""
        from matplotlib import rcParams

        from mbl.present.axis_scaling import LEGEND_ALPHA

        assert LEGEND_ALPHA != rcParams["legend.framealpha"]
        assert 0.0 < LEGEND_ALPHA < 1.0

    def test_it_is_at_the_top_of_its_panel(self) -> None:
        figure = _render_crowded()
        figure.canvas.draw()
        axes = next(a for a in figure.axes if a.get_legend())
        box, frame = axes.get_legend().get_window_extent(), axes.get_window_extent()
        top_gap = frame.y1 - box.y1
        # Tight, because a loose bound admits the opposite placement: at 0.25
        # of the panel a mutant moving the legend to `lower center` still
        # passed, and the whole point of the top is that the flagship curves
        # sit low in this panel.
        assert 0.0 <= top_gap <= 0.10 * frame.height, (top_gap, frame.height)
        assert (box.y0 + box.y1) / 2 > (frame.y0 + frame.y1) / 2, "not in the top half"

    def test_it_is_horizontally_centred(self) -> None:
        figure = _render_crowded()
        figure.canvas.draw()
        axes = next(a for a in figure.axes if a.get_legend())
        box, frame = axes.get_legend().get_window_extent(), axes.get_window_extent()
        offset = abs((box.x0 + box.x1) / 2 - (frame.x0 + frame.x1) / 2)
        assert offset <= 0.05 * frame.width, offset

    def test_it_covers_neither_END_of_the_swept_axis(self) -> None:
        """The property centring buys, and the one the rule is really about:
        where the curve starts and where it has converged to."""
        figure = _render_crowded()
        figure.canvas.draw()
        axes = next(a for a in figure.axes if a.get_legend())
        box = axes.get_legend().get_window_extent()
        depths = _crowded()["axis_value"].dropna()
        ends = {float(depths.min()), float(depths.max())}
        for panel in figure.axes:
            for line in panel.get_lines():
                if str(line.get_label()).startswith("_"):
                    continue
                points = line.get_transform().transform(
                    np.column_stack([line.get_xdata(), line.get_ydata()])
                )
                for (x, y), depth in zip(points, line.get_xdata(), strict=True):
                    if float(depth) not in ends:
                        continue
                    covered = box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1
                    assert not covered, (line.get_label(), depth)

    def test_it_fits_entirely_inside_the_panel_holding_it(self) -> None:
        figure = _render_crowded()
        figure.canvas.draw()
        axes = next(a for a in figure.axes if a.get_legend())
        box, frame = axes.get_legend().get_window_extent(), axes.get_window_extent()
        assert frame.x0 <= box.x0 and box.x1 <= frame.x1
        assert frame.y0 <= box.y0 and box.y1 <= frame.y1

    def test_it_is_never_placed_on_a_level_only_panel(self) -> None:
        """However empty such a panel is — and it always reports as entirely
        empty, which is the trap. Its height is a floor chosen to hold one
        horizontal line, so a key in it makes the panel about the key."""
        figure = _render_crowded()
        for key in ("truncated_riccati", "riccati_unconstrained"):
            axes = _panel_of(figure, CROWDED_LEVELS[key])
            assert axes is not None and axes.get_legend() is None, key

    def test_the_axes_get_the_height_back(self) -> None:
        """The measurement behind the correction. Rendered both ways, the axes
        go from 59.5 % of the figure's height to 74.5 %."""
        from mbl.present.axis_scaling import _Panels

        inside = _axes_height_fraction(_render_crowded())
        with mock.patch.object(_Panels, "settle_legend", _Panels.relegate_legend):
            below = _axes_height_fraction(_render_crowded())
        assert inside > below * 1.15, (below, inside)

    def test_a_legend_that_cannot_fit_still_goes_below(self) -> None:
        """The one absolute that survives: transparency does not make a legend
        smaller than the panel it is drawn in.

        Driven through `relegate_legend` rather than through a narrow profile.
        At `ieee-1col` this figure genuinely does over-subscribe — four panels
        and nine series in 3.5 in — but matplotlib warns when it does, and this
        repository runs pytest at zero warnings, so a test that reached the
        branch that way would be asserting on a configuration the suite cannot
        hold. The limit is recorded; the branch is exercised here.
        """
        from mbl.present.axis_scaling import _Panels

        moved: dict[str, Any] = {}

        def relegate(panels: Any, *, only: Any) -> None:
            moved["only"] = list(only)
            _Panels.relegate_legend(panels, only=only)

        with mock.patch.object(_Panels, "settle_legend", relegate):
            figure = _render_crowded()
        figure.canvas.draw()
        assert figure.legends, "the legend did not move"
        assert not any(axes.get_legend() for axes in figure.axes)
        lowest = min(axes.get_window_extent().y0 for axes in figure.axes)
        assert figure.legends[0].get_window_extent().y1 <= lowest

    def test_the_panels_stay_proportional_after_it_moves(self) -> None:
        """Relocating the legend re-runs `tight_layout`, which discards every
        position `_make_proportional` had assigned. Clause 3 is not optional on
        the way out of a placement decision."""
        from mbl.present.axis_scaling import _Panels

        with mock.patch.object(_Panels, "settle_legend", _Panels.relegate_legend):
            figure = _render_crowded()
        values = np.asarray(_crowded()["aggregate"], dtype=np.float64)
        density = []
        for axes in figure.axes:
            low, high = axes.get_ylim()
            inside = values[(values >= low) & (values <= high)]
            if not inside.size or float(inside.max() - inside.min()) <= 0.0:
                continue
            density.append(axes.get_position().height / (high - low))
        assert len(density) >= 2
        assert max(density) == pytest.approx(min(density), rel=0.02), density

    def test_wherever_it_goes_it_names_the_same_series(self) -> None:
        """Moving the legend is a placement decision and may not change what
        the key says."""
        from mbl.present.axis_scaling import _Panels

        inside = set(_legend_rows(_render_crowded()))
        with mock.patch.object(_Panels, "settle_legend", _Panels.relegate_legend):
            below = set(_legend_rows(_render_crowded()))
        assert inside == below and inside


def _axes_height_fraction(figure: Any) -> float:
    figure.canvas.draw()
    union = None
    for axes in figure.axes:
        box = axes.get_window_extent()
        union = box if union is None else union.union([union, box])
    assert union is not None
    return float(union.height / figure.get_window_extent().height)


class TestALevelOnlyPanelCarriesOneTick:
    """Not "a thin panel": the fixture's second panel spans 0.0099 and holds a
    CURVE, which needs its ticks. The property is holding nothing but a level.
    """

    def test_it_carries_exactly_one(self) -> None:
        """REVERSED 2026-08-07, on the author's review and by the §B.5.1
        clause 3 amendment.

        This asserted NO ticks, which was the right correction to three of them
        crowding a band 6 px tall and printing over each other — and it went
        one step too far. A break is legible precisely as a jump between the
        numbers on either side of it, so a panel with no numbers contributes
        none, and the author's report was that the breaks could not be seen at
        all. The panel now takes a height floor, and one tick fits in it: the
        level's own value, which is the number the jump is measured from.
        """
        figure = _render()
        for key in ("truncated_riccati", "riccati_unconstrained"):
            axes = _panel_of(figure, LEVELS[key])
            assert axes is not None, key
            assert len(axes.get_yticks()) == 1, (key, axes.get_yticks())
            assert axes.get_yticks()[0] == pytest.approx(LEVELS[key], abs=1e-9)

    def test_a_panel_holding_a_curve_keeps_its_ticks(self) -> None:
        """Including the thin one, which is the case a span threshold would
        have got wrong: it spans 0.0099 against the level panels' 0.0026."""
        figure = _render()
        for value in (8.8394, 8.2943):
            axes = _panel_of(figure, value)
            assert axes is not None, value
            assert len(axes.get_yticks()) > 0, (value, axes.get_ylim())


#: Four levels within 6e-4 of each other — the cluster that has to be
#: displaced as a group. Values are Figure 1's own publication numbers for the
#: four contenders that land in one band.
_CLUSTER = ("neural", "cocp_lower_bound", "cocp", "truncated_riccati")


#: Three levels far enough apart that the displacement rule must leave all
#: three alone — including the two that are NOT topmost, which is the case a
#: rule displacing unconditionally gets wrong.
_SPREAD = {"neural": 8.330, "cocp": 8.300, "cocp_lower_bound": 8.270}


def _spread() -> pd.DataFrame:
    rows = [
        _row("unfolded_alpha", 8.26 + 0.008 * depth, axis_value=depth)
        for depth in range(1, 11)
    ]
    rows += [_row(key, value) for key, value in _SPREAD.items()]
    return pd.DataFrame(rows)


def _cluster() -> pd.DataFrame:
    """The four levels at the BOTTOM of a panel a swept curve made wide.

    The axis is scaled to the data under study, so a cluster only crowds when
    something else sets the range — which is exactly Figure 1's shape once a
    swept contender spans more than the levels do. **The curve must stop at
    the cluster, not below it**: a curve reaching lower puts the levels part
    way up the panel, where a nudge with no floor has room and the defect does
    not appear. That is a fixture whose premise is one number.
    """
    base = 8.2551
    values = [base + 0.0006, base + 0.0004, base + 0.0002, base]
    rows = [
        _row("unfolded_alpha", 8.35 - 0.01 * depth, axis_value=depth)
        for depth in range(1, 11)
    ]
    rows += [_row(key, value) for key, value in zip(_CLUSTER, values, strict=True)]
    return pd.DataFrame(rows)


class TestTheSeparationItself:
    """`_separated` is a pure function from wanted heights to given ones, so
    its contract is testable without a figure — and one clause of it is only
    testable that way.

    A panel would need more than `1 / MARGIN_LABEL_GAP` levels for the spacing
    to have to narrow, which is twenty-three: more series than the encoding
    cycle has slots, so no rendered fixture reaches it. Asserting it through a
    figure would mean building a study that cannot exist; asserting it here
    costs a list of floats. The alternative — dropping the clause because
    today's figures do not reach it — is an argument from the current value of
    a free parameter, which this project has a standing rule against.
    """

    def test_it_returns_what_it_was_given_when_nothing_crowds(self) -> None:
        naturals = [0.9, 0.6, 0.3]
        assert _separated(naturals) == naturals

    def test_it_separates_what_does(self) -> None:
        placed = _separated([0.50, 0.49, 0.48])
        for upper, lower in zip(placed, placed[1:], strict=False):
            assert upper - lower >= MINIMUM_LEGIBLE_GAP, placed

    def test_it_never_reorders(self) -> None:
        placed = _separated([0.10, 0.099, 0.098, 0.097])
        assert placed == sorted(placed, reverse=True), placed

    def test_it_keeps_everything_inside_the_panel(self) -> None:
        """The floor and the ceiling, on the input that has neither: four
        levels at the very bottom, which is the measured defect."""
        placed = _separated([0.103, 0.101, 0.099, 0.097])
        assert min(placed) >= 0.0 and max(placed) <= 1.0, placed

    @pytest.mark.parametrize("count", [1, 2, 12, 23, 40])
    def test_it_holds_for_any_number_of_levels(self, count: int) -> None:
        """Including more than the fixed spacing fits, where the spacing has
        to narrow rather than the panel overflow."""
        placed = _separated([0.5] * count)
        assert len(placed) == count
        assert min(placed) >= 0.0 and max(placed) <= 1.0, placed
        assert placed == sorted(placed, reverse=True), placed
        for upper, lower in zip(placed, placed[1:], strict=False):
            assert upper > lower, placed

    def test_no_levels_is_no_labels(self) -> None:
        assert _separated([]) == []


def _axes_width_fraction(figure: Any) -> float:
    figure.canvas.draw()
    union = None
    for axes in figure.axes:
        box = axes.get_window_extent()
        union = box if union is None else union.union([union, box])
    assert union is not None
    return float(union.width / figure.get_window_extent().width)


#: ICASSP Figure 1's real shape: FOUR series that vary, not two, and the four
#: clusters its three breaks separate. The fixture above is deliberately
#: smaller, and mutation testing showed what that costs — three mutants that
#: put the legend over the data survived it, because a two-row legend fits in a
#: panel where a four-row one does not.
CROWDED_ROLES = {
    "riccati_unconstrained": Role.REFERENCE,
    "truncated_riccati": Role.BASELINE,
    "standard_pgd": Role.BASELINE,
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "unfolded_alpha_pj": Role.CONTENDER,
    "neural": Role.CONTENDER,
    "cocp": Role.CONTENDER,
    "cocp_lower_bound": Role.BOUND,
}
CROWDED_DISPLAY = {
    "riccati_unconstrained": "Riccati (unconstrained)",
    "truncated_riccati": "Truncated-Riccati",
    "standard_pgd": "Standard-PGD",
    "unfolded_alpha": r"UF-$\alpha$ (learned steps)",
    "unfolded_alpha_p": r"UF-$\alpha$P (proposed)",
    "unfolded_alpha_pj": r"UF-$\alpha$P$^{(j)}$ (per-iteration P)",
    "neural": "GRU",
    "cocp": "COCP",
    "cocp_lower_bound": "COCP-LB",
}
CROWDED_LEVELS = {
    "truncated_riccati": 9.150350,
    "neural": 8.301042,
    "cocp_lower_bound": 8.272536,
    "cocp": 8.270588,
    "riccati_unconstrained": 1.859845,
}
CROWDED_BREAK = [[2.2, 7.8], [8.40, 8.75], [8.87, 9.05]]


def _crowded() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for depth in range(1, 11):
        rows.append(_crowded_row("standard_pgd", 8.8383 + 0.0007 * depth, depth))
        rows.append(_crowded_row("unfolded_alpha", 8.8178 + 0.0013 * depth, depth))
        rows.append(_crowded_row("unfolded_alpha_p", 8.2828 - 0.0004 * depth, depth))
        rows.append(_crowded_row("unfolded_alpha_pj", 8.2828 - 0.0006 * depth, depth))
    rows.extend(_crowded_row(k, v) for k, v in CROWDED_LEVELS.items())
    return pd.DataFrame(rows)


def _crowded_row(label: str, value: float, axis_value: float | None = None) -> dict:
    return {
        "contender": label,
        "role": CROWDED_ROLES[label].value,
        "axis_value": np.nan if axis_value is None else float(axis_value),
        "aggregate": float(value),
        "across_seed_spread": 0.0,
        "interval_low": float(value),
        "interval_high": float(value),
        "n_seeds": 5,
        "n_trajectories": 32768,
    }


def _render_crowded(profile: str = "thesis") -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig1",
            table=_crowded(),
            config={"ybreak": CROWDED_BREAK, "dispersion": "none"},
            profile=resolve_profile(profile),
            series_order=list(CROWDED_ROLES),
            roles=dict(CROWDED_ROLES),
            display_names=dict(CROWDED_DISPLAY),
        )
    )


class TestTheRealFigureOneShape:
    """Everything above, on the cast and the four clusters of the publication
    figure — the case the smaller fixture cannot reproduce."""

    def test_the_legend_fits_entirely_inside_whatever_holds_it(self) -> None:
        """The second half of "misplaced", and the half an overlap-only test
        misses: a legend placed on a level panel covers no vertex — there are
        none — and spills across the figure's title."""
        figure = _render_crowded()
        figure.canvas.draw()
        for axes in figure.axes:
            legend = axes.get_legend()
            if legend is None:
                continue
            box, frame = legend.get_window_extent(), axes.get_window_extent()
            assert frame.x0 <= box.x0 and box.x1 <= frame.x1, (box, frame)
            assert frame.y0 <= box.y0 and box.y1 <= frame.y1, (box, frame)

    def test_no_two_margin_labels_touch(self) -> None:
        """COCP and its bound are 1.9e-3 apart on this figure. A separation
        fixed as a fraction of the panel is not a fixed number of points, and
        the text is — measured, the two printed into each other here while the
        wider fixture stayed clear."""
        figure = _render_crowded()
        figure.canvas.draw()
        boxes = [
            (text.get_text(), text.get_window_extent())
            for axes in figure.axes
            for text in axes.texts
        ]
        for index, (first, box) in enumerate(boxes):
            for second, other in boxes[index + 1 :]:
                assert not box.overlaps(other), (first, second)

    def test_every_series_stands_clear_of_every_frame(self) -> None:
        """§B.5.1 clause 7. A break mark sits on the very edge the padding is
        measured from, so a line drawn near it is drawn onto it."""
        figure = _render_crowded()
        figure.canvas.draw()
        table = _crowded()
        values = np.asarray(table["aggregate"], dtype=np.float64)
        for axes in figure.axes:
            low, high = axes.get_ylim()
            inside = values[(values >= low) & (values <= high)]
            if not inside.size:
                continue
            height = axes.get_window_extent().height
            clearance = min(
                min(value - low, high - value) / (high - low) for value in inside
            )
            assert clearance * height >= 6.0, (axes.get_ylim(), clearance * height)

    def test_every_panel_prints_its_ticks_at_the_same_precision(self) -> None:
        """And a level panel does not get to choose it. Its single tick is
        placed AT the level, so matplotlib labels it with the value's full
        precision — measured, 9.150350 and 1.859845, which then printed six
        decimals on every panel including 8.840000 where two would do."""
        figure = _render_crowded()
        decimals = {
            len(text.split(".")[-1])
            for axes in figure.axes
            for text in (t.get_text() for t in axes.get_yticklabels())
            if "." in text
        }
        assert len(decimals) == 1, decimals
        assert max(decimals) <= 3, decimals


class TestMisplacedMeansCoveringAMarkOrNotFitting:
    """`_legend_is_misplaced` asked directly, because its two halves are not
    both reachable from a rendered fixture.

    Overlap is: the crowded figure produces it. Containment is not, once a
    level panel is excluded as a candidate — every remaining panel is large
    enough that a legend overflowing it also lands on a vertex. The clause
    still has to hold, because "it fits" and "it covers nothing" are different
    questions and the answer to one does not imply the other: measured, a
    legend on a level panel covered no vertex — there are none to cover — and
    spilled across the figure's title.
    """

    @staticmethod
    def _panels(figure: Any) -> Any:
        from mbl.present.axis_scaling import _Panels

        return _Panels(figure=figure, axes=tuple(figure.axes), omitted=())

    def test_a_legend_that_overflows_its_panel_is_misplaced(self) -> None:
        figure, axes = plt.subplots(figsize=(3.0, 1.0))
        try:
            for index in range(8):
                axes.plot([0, 1], [index, index], label=f"a rather long name {index}")
            axes.legend(loc="center")
            panels = self._panels(figure)
            figure.canvas.draw()
            box = axes.get_legend().get_window_extent()
            frame = axes.get_window_extent()
            assert box.height > frame.height, "the fixture's legend does fit"
            assert panels._legend_is_misplaced(axes)
        finally:
            plt.close(figure)

    def test_a_legend_that_fits_and_covers_nothing_is_not(self) -> None:
        figure, axes = plt.subplots(figsize=(6.0, 4.0))
        try:
            axes.plot([0, 1], [0, 0], label="one")
            axes.set_ylim(-1.0, 1.0)
            axes.legend(loc="upper right")
            panels = self._panels(figure)
            figure.canvas.draw()
            assert not panels._legend_is_misplaced(axes)
        finally:
            plt.close(figure)

    def test_a_panel_with_no_legend_is_not_misplaced(self) -> None:
        figure, axes = plt.subplots()
        try:
            axes.plot([0, 1], [0, 1], label="one")
            assert not self._panels(figure)._legend_is_misplaced(axes)
        finally:
            plt.close(figure)


class TestADisplacedLabelCarriesALeader:
    """§B.5 (2026-08-12, on the author's report of Figure 3): a label that does
    not sit adjacent to its own line is tied back to it by a hairline leader.

    **What "adjacent" fails to mean was measured rather than assumed, and the
    first two attempts at this rule were both wrong.** Measuring the gap from
    the LINE fires on every label in every figure, because each is placed
    `LABEL_PAD_POINTS + half its height` clear so it does not print struck
    through — 5 of 5 in this cast. Measuring displacement from that nominal
    position instead fires on *nothing* in the real figures: their labels sit
    within 0.42 heights of nominal. What actually floats a label is the
    horizontal slide — Figure 3's `Truncated-Riccati` is anchored at 0.24 of
    the panel width, mid-plot, 0.01 heights off nominal, and that is the
    "scattered" reading the report was about.
    """

    @staticmethod
    def _leaders(figure: Any) -> list[Any]:
        return [
            line
            for axes in figure.axes
            for line in axes.lines
            if line.get_label() == "_leader"
        ]

    def _slid(self) -> pd.DataFrame:
        """A curve descending through a level within the first tenth of the
        axis, so the level's left-spine label is struck and must slide."""
        table = _table()
        rows = table["contender"] == "unfolded_alpha_p"
        table.loc[rows, "aggregate"] = [
            LEVELS["neural"] + 0.004,
            LEVELS["neural"] - 0.011,
            LEVELS["neural"] - 0.014,
            LEVELS["neural"] - 0.016,
            LEVELS["neural"] - 0.018,
            LEVELS["neural"] - 0.019,
        ]
        return table

    def test_a_slid_label_is_tied_back_to_its_line(self) -> None:
        figure = _render(config={"ybreak": BREAK}, table=self._slid())
        figure.canvas.draw()
        slid = [
            text
            for axes in figure.axes
            for text in axes.texts
            if 0.0 < float(text.xy[0]) < 1.0
        ]
        assert slid, "fixture no longer slides a label off the spine"
        assert len(self._leaders(figure)) >= len(slid)

    def test_labels_resting_at_their_spine_get_none(self) -> None:
        """The undisplaced, unslid case must draw no ink at all — otherwise the
        rule puts a connector under every level in every figure, which is the
        first version of it and what the fixture caught."""
        figure = _render()
        figure.canvas.draw()
        assert all(
            float(text.xy[0]) in (0.0, 1.0)
            for axes in figure.axes
            for text in axes.texts
        ), "fixture no longer exercises the at-the-spine case"
        assert self._leaders(figure) == []

    def test_every_leader_ends_exactly_on_the_level_it_names(self) -> None:
        """The leader's whole job is to name a height, so its far end is the
        datum and not an approximation of it."""
        figure = _render(config={"ybreak": BREAK}, table=self._slid())
        figure.canvas.draw()
        levels = set(LEVELS.values())
        for line in self._leaders(figure):
            far = float(line.get_ydata()[1])
            assert any(abs(far - level) < 1e-9 for level in levels), (
                f"leader ends at {far!r}, which is no drawn level"
            )

    def test_a_leader_is_vertical_so_it_cannot_misname_a_height(self) -> None:
        """A sloped connector arrives at its line at an x the label does not sit
        above, which on a broken axis is how a leader comes to point into the
        panel below."""
        figure = _render(config={"ybreak": BREAK}, table=self._slid())
        figure.canvas.draw()
        for line in self._leaders(figure):
            xs = line.get_xdata()
            assert float(xs[0]) == float(xs[1])

    def test_the_leader_wears_its_series_colour(self) -> None:
        """The flat label is drawn in the series' own colour and the leader is
        part of that label; a grey one would read as axes chrome."""
        figure = _render(config={"ybreak": BREAK}, table=self._slid())
        figure.canvas.draw()
        styles = encode_series(list(ROLES), roles=dict(ROLES))
        colours = {styles[label].color for label in LEVELS}
        for line in self._leaders(figure):
            assert line.get_color() in colours

    def test_a_label_never_strikes_its_own_leader(self) -> None:
        """The leader runs from the label's near edge, so it lies inside the
        text's own bounding box — and `_strikes_a_mark` walks every line on the
        axes. Shipped without the exclusion, every label reported striking
        itself and the slide-clear rule turned a green suite red."""
        figure = _render(config={"ybreak": BREAK}, table=self._slid())
        figure.canvas.draw()
        for axes in figure.axes:
            for text in axes.texts:
                assert _strikes_a_mark(axes, text) is False, (
                    f"{text.get_text()!r} is reported as striking a mark"
                )

    def test_a_leader_takes_no_legend_row(self) -> None:
        """It is chrome. A matplotlib artist labelled without a leading
        underscore joins the key, and a figure whose legend gained five
        nameless rows would be a worse figure than the one this rule fixes."""
        figure = _render(config={"ybreak": BREAK}, table=self._slid())
        figure.canvas.draw()
        for line in self._leaders(figure):
            assert line.get_label().startswith("_")
