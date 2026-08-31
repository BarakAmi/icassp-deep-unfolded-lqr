"""§A.3.2 rule 4: a figure declares its dispersion channel, and may draw none.

Added 2026-08-07 on the author's review of the ICASSP Figure 1 publication
render. Rule 2 already said a spread of zero draws no mark; what it did not
anticipate is spreads that differ by two orders of magnitude from *each other*.

Measured on that figure: four of nine contenders have exactly zero spread, four
more draw a bar between 0.5 and 2.9 pixels, and the recurrent baseline — whose
0.0173 is a legitimate measurement — draws one **1.085x the height of the panel
holding it** and **10.1x the depth effect the figure exists to show**. One
enormous bar beside eight invisible ones does not read as "these eight are
small"; it reads as "these eight have none", which is false for four of them.

**The condition is the point of the rule.** Suppressing the mark is a
presentation decision. Suppressing the quantity is not one this standard
permits, so `dispersion = "none"` is refused unless the figure's own table
carries the spread — and that is checked on `FigureSpec`, which sees both
halves, rather than in the renderer, which is handed the config without the
table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import (
    DISPERSION_CHANNELS,
    DISPERSION_LEGEND_TITLE,
    axis_scaling,
)
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError
from mbl.spec.figure import SPREAD_COLUMN, FigureSpec

ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "neural": Role.CONTENDER,
    "truncated_riccati": Role.BASELINE,
}

#: The GRU's real across-seed spread beside the unfolded family's real one:
#: a factor of 225, which is what makes one bar visible and the other not.
SPREADS = {"unfolded_alpha": 7.7e-5, "neural": 1.7332e-2, "truncated_riccati": 0.0}


def _table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for depth in (1, 2, 5, 10):
        rows.append(_row("unfolded_alpha", 8.83 - 0.001 * depth, axis_value=depth))
    rows.append(_row("neural", 8.3010))
    rows.append(_row("truncated_riccati", 9.1504))
    return pd.DataFrame(rows)


def _row(label: str, value: float, axis_value: float | None = None) -> dict[str, Any]:
    return {
        "contender": label,
        "role": ROLES[label].value,
        "axis_value": np.nan if axis_value is None else float(axis_value),
        "aggregate": float(value),
        SPREAD_COLUMN: SPREADS[label],
        "interval_low": float(value) - 1e-5,
        "interval_high": float(value) + 1e-5,
        "n_seeds": 5,
        "n_trajectories": 32768,
    }


def _render(**config: Any) -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig",
            table=_table(),
            config=config,
            profile=resolve_profile("ieee-2col"),
            series_order=list(ROLES),
            roles=dict(ROLES),
        )
    )


@pytest.fixture(autouse=True)
def _close_every_figure():
    yield
    plt.close("all")


def _bars(figure: Any) -> int:
    """Error-bar containers, which is what §A.3.2's channel draws."""
    return sum(len(axes.containers) for axes in figure.axes)


def _bands(figure: Any) -> int:
    """Shaded interval bands — the other thing that must not appear."""
    return sum(len(axes.collections) for axes in figure.axes)


class TestTheDefaultIsUnchanged:
    """Nothing about a figure that does not declare the key may move."""

    def test_a_figure_that_declares_nothing_still_draws_bars(self) -> None:
        assert _bars(_render()) > 0

    def test_and_declaring_bar_explicitly_is_the_same_figure(self) -> None:
        assert _bars(_render(dispersion="bar")) == _bars(_render())

    def test_the_default_is_the_first_channel(self) -> None:
        assert DISPERSION_CHANNELS[0] == "bar"


class TestNoneDrawsNoDispersionAtAll:
    def test_no_error_bar_is_drawn(self) -> None:
        assert _bars(_render(dispersion="none")) == 0

    def test_and_no_BAND_is_drawn_either(self) -> None:
        """The failure mode a naive implementation has. `_draw_curve` falls
        back to the interval band whenever the figure's channel is not the
        bar — so switching the bars off by that route would give the figure a
        *different* dispersion quantity rather than none, which is precisely
        the two-channels-on-one-figure defect §A.3.2 rule 1 exists to stop."""
        assert _bands(_render(dispersion="none")) == 0

    def test_the_curves_are_still_drawn(self) -> None:
        """Suppressing the dispersion must not suppress the data."""
        figure = _render(dispersion="none")
        drawn = {
            str(line.get_label())
            for axes in figure.axes
            for line in axes.get_lines()
            if not str(line.get_label()).startswith("_")
        }
        assert drawn == set(ROLES), drawn

    def test_the_marks_are_where_they_were(self) -> None:
        """The one property that makes this presentation and not a change of
        result: every plotted value is identical with and without the bars."""
        with_bars = _points(_render())
        without = _points(_render(dispersion="none"))
        assert with_bars == without

    def test_the_legend_stops_naming_a_quantity_it_no_longer_draws(self) -> None:
        """§A.3.2 rule 1 in the other direction: a caption naming a bar the
        figure does not show is a claim about marks that are not on the page."""
        figure = _render(dispersion="none")
        titles = [
            axes.get_legend().get_title().get_text()
            for axes in figure.axes
            if axes.get_legend() is not None
        ]
        assert DISPERSION_LEGEND_TITLE not in titles, titles

    def test_and_still_names_it_when_it_does(self) -> None:
        figure = _render()
        titles = [
            axes.get_legend().get_title().get_text()
            for axes in figure.axes
            if axes.get_legend() is not None
        ]
        assert DISPERSION_LEGEND_TITLE in titles, titles


class TestTheAxisIsNotSetByABarThatIsNotDrawn:
    def test_suppressing_the_bars_tightens_the_axis(self) -> None:
        """The measured motivation, as a property. The GRU's spread is 225x
        the unfolded family's, so an axis padded to hold a bar nobody draws
        wastes the range on an invisible mark."""
        drawn = _render()
        suppressed = _render(dispersion="none")
        assert _span(suppressed) < _span(drawn)

    def test_every_value_is_still_inside_the_axis(self) -> None:
        figure = _render(dispersion="none")
        limits = [axes.get_ylim() for axes in figure.axes]
        for value in _table()["aggregate"]:
            assert any(low <= value <= high for low, high in limits), value


class TestAnUnknownChannelIsRefused:
    def test_by_name(self) -> None:
        with pytest.raises(SpecificationError, match="dispersion 'band'"):
            _render(dispersion="band")

    def test_and_the_message_names_what_is_available(self) -> None:
        """A refusal an author cannot act on is barely better than silence."""
        with pytest.raises(SpecificationError) as caught:
            _render(dispersion="bar_with_caps")
        assert "bar" in str(caught.value) and "none" in str(caught.value)


class TestSuppressingTheMarkMayNotSuppressTheQuantity:
    """§A.3.2 rule 3, enforced on `FigureSpec` — it sees the config and the
    table together, which the renderer never does."""

    def test_a_figure_with_the_column_is_accepted(self) -> None:
        spec = FigureSpec(
            id="f",
            kind="axis_scaling",
            source="a",
            config={"dispersion": "none"},
            table={"columns": ["contender", SPREAD_COLUMN]},
        )
        assert spec.config["dispersion"] == "none"

    def test_a_figure_with_NO_TABLE_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="does not carry"):
            FigureSpec(
                id="f", kind="axis_scaling", source="a", config={"dispersion": "none"}
            )

    def test_a_figure_whose_table_omits_the_column_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match=SPREAD_COLUMN):
            FigureSpec(
                id="f",
                kind="axis_scaling",
                source="a",
                config={"dispersion": "none"},
                table={"columns": ["contender", "aggregate"]},
            )

    def test_the_two_refusals_say_different_things(self) -> None:
        """ "No table at all" and "a table without the column" are different
        mistakes and an author fixes them differently."""
        messages = []
        for table in (None, {"columns": ["contender"]}):
            with pytest.raises(SpecificationError) as caught:
                FigureSpec(
                    id="f",
                    kind="axis_scaling",
                    source="a",
                    config={"dispersion": "none"},
                    table=table,
                )
            messages.append(str(caught.value))
        assert messages[0] != messages[1]
        assert "no table at all" in messages[0]

    def test_a_figure_that_DRAWS_its_bars_needs_no_table(self) -> None:
        """The condition attaches to suppression, not to figures in general —
        a figure drawing its dispersion has already reported it."""
        assert FigureSpec(id="f", kind="axis_scaling", source="a").table is None

    def test_the_column_name_matches_the_one_the_renderer_reads(self) -> None:
        """`spec` is Tier 3 and may not import the analysis tier, so the name
        is written twice. Two spellings that drift make the guard check a
        column nothing emits."""
        from mbl.present.axis_scaling import SPREAD_COLUMN as RENDERER_COLUMN

        assert SPREAD_COLUMN == RENDERER_COLUMN


def _points(figure: Any) -> list[tuple[float, ...]]:
    return sorted(
        tuple(round(float(value), 12) for value in line.get_ydata())
        for axes in figure.axes
        for line in axes.get_lines()
        if not str(line.get_label()).startswith("_")
    )


def _span(figure: Any) -> float:
    return sum(axes.get_ylim()[1] - axes.get_ylim()[0] for axes in figure.axes)
