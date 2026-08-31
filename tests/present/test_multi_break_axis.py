"""§B.5.1 clause 3, amended 2026-08-07: a value axis may carry N breaks.

Why, measured on ICASSP Figure 1: its values form four clusters — a baseline at
9.127, two iteration-dependent curves at 8.80–8.82, five contenders at
8.26–8.36, and the unconstrained floor at 1.86. With a single break the upper
panel spans 0.88 and the convergence the figure exists to show (8.2959 → 8.2801)
occupies **1.8 %** of its height. A reader cannot see whether it converged,
which is the compression §B.5.1 forbids, one scale down.

What must NOT change, and is asserted here rather than assumed: clause 2 still
refuses a break over a non-empty range, and the proportional-height rule still
holds across **every** panel. Those two are what make an extra break carry no
claim — it removes empty space, and empty space says nothing.
"""

import numpy as np
import pandas as pd
import pytest

from typing import Any

import matplotlib.pyplot as plt
from mbl.present.axis_scaling import LEVEL_PANEL_HEIGHT, axis_scaling
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError

#: Figure 1's own shape, reduced to four clusters and two swept series.
ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "unfolded_alpha_p": Role.CONTENDER,
    "truncated_riccati": Role.BASELINE,
    "riccati_unconstrained": Role.REFERENCE,
}


def _table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for j in (1, 2, 3, 5, 8, 10):
        rows.append(_row("unfolded_alpha", 8.80 + 0.002 * j, axis_value=float(j)))
        rows.append(_row("unfolded_alpha_p", 8.30 - 0.002 * j, axis_value=float(j)))
    rows.append(_row("truncated_riccati", 9.127))
    rows.append(_row("riccati_unconstrained", 1.860))
    return pd.DataFrame(rows)


def _row(label: str, value: float, *, axis_value: float | None = None) -> dict:
    return {
        "contender": label,
        "role": ROLES[label].value,
        "axis_value": np.nan if axis_value is None else axis_value,
        "axis_label": "" if axis_value is None else f"{axis_value:g}",
        "aggregate": value,
        "across_seed_spread": 0.0,
        "interval_low": value,
        "interval_high": value,
        "n_seeds": 5,
        "n_trajectories": 32768,
    }


def _render(config: dict[str, Any]) -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig1",
            table=_table(),
            config=config,
            profile=resolve_profile("thesis"),
            series_order=list(ROLES),
            roles=dict(ROLES),
        )
    )


@pytest.fixture(autouse=True)
def _close_every_figure():
    """Each test renders one or two figures and must not leave them open.

    matplotlib warns past twenty open figures, and this repository runs pytest
    at zero tolerance for warnings -- so a leak here does not fail this file,
    it fails whichever file happens to run next. Measured: 17 failures across
    two other modules, none of which reproduced when run alone.
    """
    yield
    plt.close("all")


def _has_extent(axes: Any) -> bool:
    """Whether the DATA inside `axes`'s limits has a vertical range.

    From the table, not the artists: every series is drawn on every panel, so
    the curves' `Line2D` objects exist on the level panels too, clipped out of
    view.
    """
    low, high = axes.get_ylim()
    values = np.asarray(_table()["aggregate"], dtype=np.float64)
    inside = values[(values >= low) & (values <= high)]
    return bool(inside.size) and float(inside.max() - inside.min()) > 0.0


def _panel_region(figure: Any) -> float:
    """The combined height the panels share, which the floor is a fraction of.

    Not the figure's height: the title, the axis labels and the legend have
    already taken theirs by the time the panels are laid out.
    """
    boxes = [axes.get_position() for axes in figure.axes]
    return float(boxes[0].union(boxes).height)


#: Three breaks -> four panels: 1.86 | 8.26-8.36 | 8.80-8.82 | 9.127
THREE = {"ybreak": [[2.0, 8.2], [8.4, 8.7], [8.9, 9.0]]}


class TestNBreaks:
    def test_three_breaks_give_four_panels(self) -> None:
        figure = _render({**THREE})
        assert len(figure.axes) == 4, [a.get_ylim() for a in figure.axes]

    def test_one_break_still_gives_two_panels_declared_the_old_way(self) -> None:
        """The flat two-number spelling must keep working: it is what every
        existing document and every existing test writes."""
        figure = _render({"ybreak": [2.0, 8.2]})
        assert len(figure.axes) == 2

    def test_one_break_declared_as_a_nested_pair_is_the_same_figure(self) -> None:
        flat = _render({"ybreak": [2.0, 8.2]})
        nested = _render({"ybreak": [[2.0, 8.2]]})
        assert [a.get_ylim() for a in flat.axes] == [a.get_ylim() for a in nested.axes]

    def test_every_value_lands_inside_some_panel(self) -> None:
        """The property that makes N breaks safe: nothing is off-canvas. This
        is the defect that a fourth panel could most easily introduce."""
        figure = _render({**THREE})
        limits = [a.get_ylim() for a in figure.axes]
        for value in _table()["aggregate"]:
            assert any(low <= value <= high for low, high in limits), (value, limits)

    def test_the_panels_descend(self) -> None:
        """Top panel holds the largest values. A figure whose panels were
        ordered the other way would read as inverted without saying so."""
        figure = _render({**THREE})
        tops = [a.get_ylim()[1] for a in figure.axes]
        assert tops == sorted(tops, reverse=True), tops

    def test_one_unit_of_y_is_the_same_height_in_EVERY_RANGED_panel(self) -> None:
        """Clause 3, generalised, and the clause that keeps N breaks honest:
        without it an extra break becomes a twin axis by the back door.

        NARROWED 2026-08-07 to the panels that carry a range. Here that is the
        two holding the swept curves; the top panel holds one baseline at 9.127
        and the bottom one the floor at 1.860, and a panel of horizontal lines
        has no slope for this rule to protect. Its span is invented, so a
        height proportional to it is proportional to nothing -- measured at
        6.2 px, too thin for its own tick.
        """
        figure = _render({**THREE})
        ranged = [a for a in figure.axes if _has_extent(a)]
        assert len(ranged) == 2, [a.get_ylim() for a in figure.axes]
        density = [
            a.get_position().height / (a.get_ylim()[1] - a.get_ylim()[0])
            for a in ranged
        ]
        assert max(density) == pytest.approx(min(density), rel=0.02), density

    def test_the_two_level_panels_take_the_floor(self) -> None:
        figure = _render({**THREE})
        levels = [a for a in figure.axes if not _has_extent(a)]
        assert len(levels) == 2, [a.get_ylim() for a in figure.axes]
        for axes in levels:
            assert axes.get_position().height == pytest.approx(
                LEVEL_PANEL_HEIGHT * _panel_region(figure), rel=0.05
            )
            assert len(axes.get_yticks()) == 1, axes.get_yticks()

    def test_a_break_through_data_is_still_refused(self) -> None:
        """Clause 2, unchanged, on the SECOND break -- a check that only ran
        on the first would pass everything an extra break could hide."""
        with pytest.raises(SpecificationError, match="not.*empty"):
            _render({"ybreak": [[2.0, 8.2], [8.29, 8.31]]})

    def test_overlapping_breaks_are_refused(self) -> None:
        with pytest.raises(SpecificationError, match="overlap"):
            _render({"ybreak": [[2.0, 8.5], [8.3, 8.7]]})

    def test_an_inverted_range_is_still_refused(self) -> None:
        with pytest.raises(SpecificationError, match="not above"):
            _render({"ybreak": [[8.2, 2.0]]})

    def test_a_malformed_nested_range_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="exactly two"):
            _render({"ybreak": [[2.0, 8.2, 9.0]]})

    def test_every_gap_is_marked_on_both_of_its_sides(self) -> None:
        """Clause 3: each discontinuity is "impossible to miss". Two marks per
        side, per gap -- left and right edge -- so three gaps is twelve."""
        figure = _render({**THREE})
        marks = sum(
            1
            for axes in figure.axes
            for line in axes.get_lines()
            if str(line.get_label()).startswith("_break")
        )
        assert marks == 12, marks

    def test_the_facing_spines_are_hidden_at_every_gap(self) -> None:
        figure = _render({**THREE})
        for axes in figure.axes[:-1]:
            assert not axes.spines["bottom"].get_visible()
        for axes in figure.axes[1:]:
            assert not axes.spines["top"].get_visible()

    def test_only_the_bottom_panel_carries_the_x_label(self) -> None:
        figure = _render({**THREE, "xlabel": "unrolling depth $J$"})
        assert figure.axes[-1].get_xlabel() == "unrolling depth $J$"
        for axes in figure.axes[:-1]:
            assert axes.get_xlabel() == ""

    def test_every_panel_prints_its_ticks_the_same_way(self) -> None:
        """Clause 1: one quantity, one unit, one number format -- across four
        panels now, not two."""
        figure = _render({**THREE})
        formats = {
            getattr(axes.yaxis.get_major_formatter(), "fmt", None)
            for axes in figure.axes
        }
        assert len(formats) == 1, formats

    def test_a_panel_holding_only_a_reference_is_scaled_to_it(self) -> None:
        """Clause 5's inversion, generalised: it is not "the lower panel", it
        is "a panel with no data under study in it". The bottom panel here
        holds the unconstrained Riccati alone."""
        figure = _render({**THREE})
        bottom = figure.axes[-1]
        low, high = bottom.get_ylim()
        assert low <= 1.860 <= high
        assert high - low < 1.0, (low, high)
