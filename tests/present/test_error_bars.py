"""The spread a figure draws (Annex 03 §A.3.2).

§A.3 fixes what is *reported* — the across-seed BCa interval. It says nothing
about what is *drawn*, and the two came apart the first time a figure was
measured rather than looked at: on the tracked publication table the interval
is **9.4e-5 wide against an axis range of 0.1825**, a band a fraction of a
pixel high, present in the artifact and invisible to every reader of it.

So a figure draws a declared across-seed **dispersion** — for a mean, the
standard deviation of the per-seed means — as an error bar with caps. There
was no `errorbar`, `yerr` or `capsize` anywhere under `src/` (verified), and
the analysis emitted no column to draw.

Three rules from §A.3.2, each tested here:

1. **It is named where it is drawn.** A reader who assumes a 95 % interval
   where one standard deviation was drawn misreads the result by a factor
   nothing in the figure lets them recover.
2. **A spread that does not exist and a spread that is zero are different
   facts, and neither may produce a mark a reader could mistake for a
   measured one.** One seed has no across-seed quantity at all (`NaN`); an
   analytic contender under §A.2's common-random-numbers law has a spread of
   exactly zero and that zero IS the answer. Neither draws a bar; the table
   carries the distinction as `—` and `0.000`.
3. **The interval is still reported.** The bar is the figure's channel and
   §A.4's table continues to state the interval. Substituting one for the
   other would lower the standard of evidence to whichever quantity happened
   to look better.

**The bar and the band are mutually exclusive**, and that is this module's one
design decision beyond the annex. Two dispersion channels on one figure is
precisely the confusion rule 1 forbids: nothing on the page tells a reader
which of the two the shading is. A frame carrying a drawable
`across_seed_spread` therefore draws the bar and no band; a frame without the
column — an artifact rendered before this existed — still draws the band, so
`mbl figure rebuild` reproduces an old figure rather than silently restyling
it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present.axis_scaling import DISPERSION_LEGEND_TITLE, axis_scaling
from mbl.present.greyscale import require_greyscale_separable
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import Role

ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "truncated_riccati": Role.BASELINE,
    "cocp_lower_bound": Role.BOUND,
}
ORDER = tuple(ROLES)

DEPTHS = (1, 2, 4, 8)


def _table(
    *,
    spread: float | None = 0.04,
    flat_spread: float | None = 0.0,
    intervals: bool = True,
    with_column: bool = True,
) -> pd.DataFrame:
    """One swept contender, one flat baseline, one flat bound.

    Args:
        spread: `across_seed_spread` for the swept contender, or `None` for
            `NaN` (the one-seed case).
        flat_spread: the same for the flat baseline.
        intervals: whether a BCa interval is present.
        with_column: whether the frame carries `across_seed_spread` at all —
            `False` is an artifact written before §A.3.2 existed.
    """
    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        aggregate = 3.0 - 0.1 * depth
        rows.append(
            {
                "contender": "unfolded_alpha",
                "role": ROLES["unfolded_alpha"].value,
                "axis_value": float(depth),
                "aggregate": aggregate,
                "interval_low": aggregate - 0.005 if intervals else np.nan,
                "interval_high": aggregate + 0.005 if intervals else np.nan,
                "across_seed_spread": np.nan if spread is None else spread,
            }
        )
    for label, value, own in (
        ("truncated_riccati", 2.9, flat_spread),
        ("cocp_lower_bound", 1.2, 0.0),
    ):
        rows.append(
            {
                "contender": label,
                "role": ROLES[label].value,
                "axis_value": np.nan,
                "aggregate": value,
                "interval_low": np.nan,
                "interval_high": np.nan,
                "across_seed_spread": np.nan if own is None else own,
            }
        )
    frame = pd.DataFrame(rows)
    if not with_column:
        frame = frame.drop(columns=["across_seed_spread"])
    return frame


def _figure(table: pd.DataFrame, **config: Any) -> Any:
    return axis_scaling(
        FigureContext(
            figure_id="fig",
            table=table,
            config=config,
            profile=resolve_profile("thesis"),
            series_order=ORDER,
            roles=ROLES,
        )
    )


def _bars(figure: Any) -> list[Any]:
    """Every error-bar container drawn on the figure."""
    from matplotlib.container import ErrorbarContainer

    return [
        container
        for axes in figure.axes
        for container in axes.containers
        if isinstance(container, ErrorbarContainer)
    ]


def _bands(figure: Any) -> list[Any]:
    from matplotlib.collections import PolyCollection

    return [
        collection
        for axes in figure.axes
        for collection in axes.collections
        if isinstance(collection, PolyCollection)
    ]


class TestTheBarIsDrawn:
    def test_a_curve_carries_one_capped_bar_per_point(self) -> None:
        figure = _figure(_table(spread=0.04))
        bars = _bars(figure)
        assert len(bars) == 1
        # §A.3.2: "on each point of a curve" -- and WITH CAPS, which is the
        # word the annex uses and the difference between a readable mark and
        # a bare line segment.
        assert len(bars[0].lines[1]) == 2, "no caps drawn"
        plt.close(figure)

    def test_the_bar_height_is_the_frames_own_number(self) -> None:
        """Read off the DRAWN artist and compared to the frame, because a
        renderer that halved, doubled or ignored the column would still draw
        something bar-shaped."""
        spread = 0.04
        figure = _figure(_table(spread=spread))
        segments = _bars(figure)[0].lines[2][0].get_segments()
        assert len(segments) == len(DEPTHS)
        for segment, depth in zip(segments, DEPTHS):
            aggregate = 3.0 - 0.1 * depth
            assert segment[0][1] == pytest.approx(aggregate - spread)
            assert segment[1][1] == pytest.approx(aggregate + spread)
        plt.close(figure)

    def test_a_flat_reference_with_a_spread_carries_one_too(self) -> None:
        # A contender the axis does not apply to is still measured over five
        # seeds, so it still has an across-seed spread. Drawing it only on
        # curves would silently exempt exactly the rows a reader compares
        # against.
        figure = _figure(_table(spread=0.04, flat_spread=0.02))
        labels = {
            container.get_label()
            for container in _bars(figure)
            if not str(container.get_label()).startswith("_")
        }
        assert len(_bars(figure)) == 2, labels
        plt.close(figure)


class TestTheBarIsNamedWhereItIsDrawn:
    def test_the_legend_states_which_quantity_the_bar_is(self) -> None:
        figure = _figure(_table(spread=0.04))
        legend = figure.axes[0].get_legend()
        assert legend is not None
        assert DISPERSION_LEGEND_TITLE in legend.get_title().get_text()
        plt.close(figure)

    def test_nothing_is_named_when_no_bar_is_drawn(self) -> None:
        """A caption naming a quantity the figure does not show is worse than
        no caption: it is a claim about ink that is not on the page."""
        figure = _figure(_table(spread=None, flat_spread=None))
        legend = figure.axes[0].get_legend()
        title = "" if legend is None else legend.get_title().get_text()
        assert DISPERSION_LEGEND_TITLE not in title
        plt.close(figure)


class TestAbsentAndZeroAreBothSilent:
    def test_an_absent_spread_draws_no_bar(self) -> None:
        figure = _figure(_table(spread=None, flat_spread=None))
        assert _bars(figure) == []
        plt.close(figure)

    def test_a_zero_spread_draws_no_bar(self) -> None:
        """An analytic contender has exactly zero training variance, and that
        zero is the ANSWER. A zero-height bar still draws two caps, which a
        reader meets as a measured interval too small to see -- the mark rule
        2 forbids."""
        figure = _figure(_table(spread=0.0, flat_spread=0.0))
        assert _bars(figure) == []
        plt.close(figure)

    def test_a_zero_row_beside_a_finite_row_drops_only_its_own_mark(self) -> None:
        # The mixed case a real study produces: learned contenders spread,
        # analytic ones do not. Suppressing the whole figure's bars because
        # one row is zero would delete the measurement from the others.
        figure = _figure(_table(spread=0.04, flat_spread=0.0))
        assert len(_bars(figure)) == 1
        plt.close(figure)

    def test_an_infinite_spread_draws_no_bar(self) -> None:
        """What the `isfinite` half of the guard is actually for.

        Mutation testing dropped `np.isfinite` and nothing failed, because
        `NaN > 0.0` is already `False` in numpy — the comparison excludes the
        absent case on its own. `+inf > 0.0` is `True`, so the finiteness
        check is doing exactly one job, and it was untested. An infinite bar
        is not merely wrong: it propagates into the axis rule and makes the
        limits infinite, so the figure fails to draw at all.
        """
        table = _table(spread=0.04)
        table.loc[0, "across_seed_spread"] = np.inf
        figure = _figure(table)
        segments = _bars(figure)[0].lines[2][0].get_segments()
        assert np.isfinite(np.asarray(segments)).all()
        assert segments[0][0][1] == segments[0][1][1], "the infinite row drew a bar"
        assert np.isfinite(figure.axes[0].get_ylim()).all()
        plt.close(figure)

    def test_the_column_being_absent_entirely_is_not_an_error(self) -> None:
        """§B.1.1: a figure must rebuild from its own frozen parquet. An
        artifact written before §A.3.2 existed has no such column, and
        `mbl figure rebuild` must reproduce it rather than refuse it."""
        figure = _figure(_table(with_column=False))
        assert _bars(figure) == []
        assert len(_bands(figure)) == 1
        plt.close(figure)


class TestTheBarAndTheBandAreMutuallyExclusive:
    def test_a_drawable_spread_replaces_the_band(self) -> None:
        figure = _figure(_table(spread=0.04, intervals=True))
        assert len(_bars(figure)) == 1
        assert _bands(figure) == [], "two dispersion channels on one figure"
        plt.close(figure)

    def test_without_a_drawable_spread_the_band_still_draws(self) -> None:
        figure = _figure(_table(spread=None, intervals=True))
        assert _bars(figure) == []
        assert len(_bands(figure)) == 1
        plt.close(figure)

    def test_an_analytic_series_beside_a_learned_one_draws_no_band(self) -> None:
        """The defect the fixtures missed and the tracked study exposed.

        `standard_pgd` is analytic: its across-seed spread is exactly 0.0 at
        every depth, which is the ANSWER and not a gap. A per-series rule
        therefore gave it a shaded BCa band while three neighbours carried
        standard-deviation bars — two dispersion channels on one figure, the
        legend naming only one of them. Measured on the real render:
        3 error-bar containers and 1 surviving band.

        The channel is a property of the FIGURE, so a series with no spread of
        its own draws nothing rather than falling back to the other quantity.
        """
        rows = _table(spread=0.04, intervals=True).copy()
        analytic = rows.copy()
        analytic["contender"] = "standard_pgd"
        analytic["across_seed_spread"] = 0.0
        mixed = pd.concat([rows, analytic], ignore_index=True)
        figure = axis_scaling(
            FigureContext(
                figure_id="fig",
                table=mixed,
                config={},
                profile=resolve_profile("thesis"),
                series_order=(*ORDER, "standard_pgd"),
                roles={**ROLES, "standard_pgd": Role.BASELINE},
            )
        )
        assert len(_bars(figure)) == 1, "the learned series lost its bar"
        assert _bands(figure) == [], "the analytic series fell back to a band"
        plt.close(figure)


class TestTheAxisStillHoldsTheMarks:
    def test_the_axis_includes_the_whole_bar(self) -> None:
        """A bar drawn outside the axis is a measurement the figure reports
        and does not show. The pre-change axis rule padded from the aggregate
        and the interval only, so a spread wider than the interval -- which is
        the normal case, and 9.4e-5 against 0.04 here -- would have been
        clipped."""
        spread = 0.4
        figure = _figure(_table(spread=spread))
        low, high = figure.axes[0].get_ylim()
        aggregates = [3.0 - 0.1 * depth for depth in DEPTHS]
        assert low <= min(aggregates) - spread
        assert high >= max(aggregates) + spread
        plt.close(figure)


class TestThePrintLawStillHolds:
    def test_a_figure_with_bars_passes_the_greyscale_gate(self) -> None:
        figure = _figure(_table(spread=0.04, flat_spread=0.02))
        require_greyscale_separable(figure, "fig")
        plt.close(figure)

    def test_the_bars_do_not_double_count_their_series(self) -> None:
        """The trap the gate hardening already paid for: an unlabelled
        `errorbar(fmt="none")` companion registers as `_container0`. If it
        carried the series' label instead, every series would conflict with
        itself and no figure with error bars could ever render."""
        from mbl.present.greyscale import encodings_of

        figure = _figure(_table(spread=0.04, flat_spread=0.02))
        labels = [encoding.label.split(" (")[0] for encoding in encodings_of(figure)]
        assert sorted(labels) == sorted(ORDER)
        plt.close(figure)
