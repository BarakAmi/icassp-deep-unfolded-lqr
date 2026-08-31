"""Acceptance tests for `grouped_bars` — ICASSP Phase D, the figure half.

Written before the implementation. Two of these clauses are the plan's own
Phase-D acceptance, quoted: *"a bar figure whose artists carry no label is
refused rather than silently passed"*, and *"a labelled bar figure fails the
gate when two adjacent bars are given indistinguishable fills"*.

**The trap is already measured and is why the first clause exists.** The
greyscale gate is **label**-driven: `ax.bar(label=…)` puts the label on the
*container* and leaves every child `Patch` at `_nolegend_`, so one `ax.bar`
with six contender names on the x axis and no labels gives the gate **0
encodings, 0 conflicts, and a PASS** — the print law silently off. Phase B
taught the gate to walk containers, which closed the case where a label
exists; this is the case where there is nothing to walk.

Two defences, tested independently, because either alone leaves a hole:

* the **renderer** labels every container it draws, so this figure is
  inspected; and
* the **gate** refuses an unlabelled `BarContainer` outright, so the *next*
  renderer is inspected too. A property that holds only because one function
  is careful is not a gate.

**The chart's shape is a measurement, not a preference** (Annex 03 §B.4.2). Of
the 28 unordered pairs in `CATEGORICAL_PALETTE`, 17 sit below the 0.15
luminance floor and the largest mutually separable subset is **three** — so
seven contenders cannot be seven fills, and the contenders go on the x axis
where the tick label is an identity channel colour removal cannot touch.
`TestTheFillCycle` computes that separability rather than trusting the list.

**§B.5.2 is the other half.** A bar says how much by how long, so its axis
includes zero and there is deliberately **no truncation key** to declare.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.container import BarContainer

from mbl.analysis.cost_by_category import AXIS_KIND_COLUMN, CATEGORICAL, NUMERIC
from mbl.present.axis_scaling import axis_scaling
from mbl.present.encodings import (
    BAR_FILL_COLORS,
    BAR_HATCHES,
    EncodingCapacityError,
    encode_fills,
)
from mbl.present.greyscale import (
    LUMINANCE_FLOOR,
    GreyscaleError,
    encodings_of,
    figure_conflicts,
    relative_luminance,
    require_greyscale_separable,
)
from mbl.present.grouped_bars import CONFIG_KEYS, _apply_axis_rule, grouped_bars
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext, registered_figures, resolve_figure
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError

#: Figure 2's real shape, reduced to what the renderer can be tested on: the
#: contenders that carry a category, plus a bound the axis does not apply to.
ROLES = {
    "unfolded_alpha_p": Role.CONTENDER,
    "neural": Role.CONTENDER,
    "truncated_riccati": Role.BASELINE,
    "riccati_unconstrained": Role.REFERENCE,
}
ORDER = tuple(ROLES)

#: Two mismatch experiments — the categorical axis Figure 2 is against.
CATEGORIES = ("nominal", "rotated")

DISPLAY = {
    "unfolded_alpha_p": "UF-aP (proposed)",
    "neural": "GRU",
    "truncated_riccati": "Truncated-Riccati",
    "riccati_unconstrained": "Riccati (unconstrained)",
}


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    """Every `present/` module has one: matplotlib warns past 20 open figures
    and this repository runs pytest at zero warnings, so a leak here fails a
    different file."""
    yield
    plt.close("all")


def _table(
    *,
    spread: float = 0.04,
    categories: tuple[str, ...] = CATEGORIES,
    axis_kind: str = CATEGORICAL,
    with_flat: bool = True,
) -> pd.DataFrame:
    """A `cost_by_category` frame: two bar contenders x N categories, plus
    two rows the axis does not apply to."""
    rows: list[dict[str, Any]] = []
    for position, category in enumerate(categories):
        for index, label in enumerate(("unfolded_alpha_p", "neural")):
            aggregate = 8.3 + 0.4 * position + 0.15 * index
            rows.append(
                {
                    "contender": label,
                    "role": ROLES[label].value,
                    "axis_path": "evaluation.problem",
                    "axis_value": float(position),
                    "axis_label": category,
                    AXIS_KIND_COLUMN: axis_kind,
                    "aggregate": aggregate,
                    "interval_low": aggregate - 0.01,
                    "interval_high": aggregate + 0.01,
                    "across_seed_spread": spread,
                }
            )
    if with_flat:
        for label, value in (
            ("truncated_riccati", 9.15),
            ("riccati_unconstrained", 1.86),
        ):
            rows.append(
                {
                    "contender": label,
                    "role": ROLES[label].value,
                    "axis_path": "",
                    "axis_value": np.nan,
                    "axis_label": "",
                    AXIS_KIND_COLUMN: axis_kind,
                    "aggregate": value,
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                    "across_seed_spread": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _render(table: pd.DataFrame | None = None, **config: Any):
    return grouped_bars(
        FigureContext(
            figure_id="fig2_mismatch",
            table=_table() if table is None else table,
            config=config,
            profile=resolve_profile(None),
            series_order=ORDER,
            roles=ROLES,
            display_names=DISPLAY,
        )
    )


def _bar_containers(figure: Any) -> list[BarContainer]:
    return [
        container
        for axes in figure.axes
        for container in axes.containers
        if isinstance(container, BarContainer)
    ]


class TestTheRegistry:
    def test_the_kind_resolves_by_its_annex_name(self) -> None:
        assert resolve_figure("grouped_bars") is grouped_bars
        assert "grouped_bars" in registered_figures()


class TestEveryDrawnContainerIsLabelled:
    """The plan's first Phase-D clause, from both sides."""

    def test_the_renderer_labels_every_bar_container(self) -> None:
        figure = _render()

        containers = _bar_containers(figure)
        assert len(containers) == len(CATEGORIES)
        assert all(
            not str(container.get_label()).startswith("_") for container in containers
        ), [str(c.get_label()) for c in containers]

    def test_the_gate_therefore_inspects_one_encoding_per_category(self) -> None:
        """Measured before this existed: it inspected **nothing**."""
        figure = _render()

        bars = [
            encoding for encoding in encodings_of(figure) if encoding.marker == "bar"
        ]
        assert len(bars) == len(CATEGORIES)
        assert {encoding.label for encoding in bars} == set(CATEGORIES)

    def test_an_unlabelled_bar_figure_is_refused_by_the_gate(self) -> None:
        """The second defence, and the one that covers the next renderer.

        Hand-built rather than rendered, because the renderer cannot produce
        this — which is exactly why the gate has to.
        """
        figure, axes = plt.subplots()
        axes.bar([0, 1, 2], [1.0, 2.0, 3.0])

        with pytest.raises(GreyscaleError, match="unlabelled"):
            require_greyscale_separable(figure, "hand_built")

    def test_the_refusal_survives_a_labelled_neighbour(self) -> None:
        """One labelled series must not launder an unlabelled one: the gate
        would otherwise pass a figure whose *other* bars escaped it."""
        figure, axes = plt.subplots()
        axes.bar([0, 1], [1.0, 2.0], width=0.4, label="offline")
        axes.bar([0.4, 1.4], [2.0, 3.0], width=0.4)

        with pytest.raises(GreyscaleError, match="unlabelled"):
            require_greyscale_separable(figure, "hand_built")

    def test_an_unlabelled_errorbar_companion_is_still_chrome(self) -> None:
        """The negative control. `errorbar(fmt="none")` registers as
        `_container0` and is the bar's own caps, not a second series — so the
        refusal above must be about `BarContainer` and not about every
        underscore-labelled container, or every error bar in the repository
        becomes a refusal."""
        figure, axes = plt.subplots()
        axes.bar([0, 1], [1.0, 2.0], label="offline", color=BAR_FILL_COLORS[0])
        axes.errorbar([0, 1], [1.0, 2.0], yerr=[0.1, 0.1], fmt="none")

        require_greyscale_separable(figure, "hand_built")


class TestIndistinguishableFillsAreRefused:
    """The plan's second Phase-D clause."""

    def test_two_adjacent_bars_with_one_fill_fail_the_gate(self) -> None:
        figure, axes = plt.subplots()
        axes.bar([0, 1], [1.0, 2.0], width=0.4, label="nominal", color="#2a78d6")
        axes.bar([0.4, 1.4], [2.0, 3.0], width=0.4, label="rotated", color="#008300")

        assert figure_conflicts(figure), "the fixture must actually be inseparable"
        with pytest.raises(GreyscaleError, match="nominal"):
            require_greyscale_separable(figure, "hand_built")

    def test_the_figure_the_renderer_produces_passes(self) -> None:
        """The positive control the negative one is worthless without."""
        require_greyscale_separable(_render(), "fig2_mismatch")


class TestTheFillCycle:
    """§B.4.2's separability, computed rather than asserted from a list."""

    def test_every_pair_of_slots_separates_in_print(self) -> None:
        capacity = len(BAR_FILL_COLORS) * len(BAR_HATCHES)
        fills = encode_fills(tuple(f"c{index}" for index in range(capacity)))
        assert len(fills) == capacity

        styles = list(fills.values())
        for first in range(len(styles)):
            for second in range(first + 1, len(styles)):
                one, other = styles[first], styles[second]
                if one.hatch != other.hatch:
                    continue
                separation = abs(
                    relative_luminance(one.color) - relative_luminance(other.color)
                )
                assert separation >= LUMINANCE_FLOOR, (
                    f"slots {first} and {second} share hatch {one.hatch!r} and "
                    f"differ by only {separation:.4f}"
                )

    def test_the_first_three_slots_differ_in_both_channels(self) -> None:
        """Nearly every figure is in this case, so it is worth its own check:
        colour varies fastest."""
        fills = list(encode_fills(("a", "b", "c")).values())
        assert len({fill.color for fill in fills}) == 3

    def test_the_declared_colours_are_mutually_separable(self) -> None:
        """The measurement Annex 03 §B.4.2 records — recomputed here so a
        change to the constant is caught rather than inherited."""
        for first in range(len(BAR_FILL_COLORS)):
            for second in range(first + 1, len(BAR_FILL_COLORS)):
                separation = abs(
                    relative_luminance(BAR_FILL_COLORS[first])
                    - relative_luminance(BAR_FILL_COLORS[second])
                )
                assert separation >= LUMINANCE_FLOOR

    def test_the_textures_are_45_and_135_degrees_only(self) -> None:
        """§B.4.2 restricts textures to two angles; the third slot is the
        absence of one, which is not a texture."""
        assert set(BAR_HATCHES) <= {"", "//", "\\\\"}

    def test_exhaustion_is_refused_by_name_never_wrapped(self) -> None:
        """A modulo wrap answers "which fill now?" silently when the honest
        answer is that there is none — the defect the marker cycle already had.
        """
        too_many = tuple(
            f"c{index}" for index in range(len(BAR_FILL_COLORS) * len(BAR_HATCHES) + 1)
        )
        with pytest.raises(EncodingCapacityError, match="fill"):
            encode_fills(too_many)

    def test_a_figure_at_every_admissible_width_still_passes(self) -> None:
        """A property test over the whole cycle, not one sample of it."""
        capacity = len(BAR_FILL_COLORS) * len(BAR_HATCHES)
        for count in range(1, capacity + 1):
            categories = tuple(f"c{index}" for index in range(count))
            figure = _render(_table(categories=categories))
            require_greyscale_separable(figure, f"width_{count}")
            plt.close(figure)


class TestTheContendersAreTheXAxis:
    def test_the_ticks_are_the_contenders_in_declaration_order(self) -> None:
        figure = _render()
        axes = figure.axes[0]

        drawn = [text.get_text() for text in axes.get_xticklabels()]
        assert drawn == [DISPLAY["unfolded_alpha_p"], DISPLAY["neural"]]

    def test_a_tick_shows_the_display_name_never_the_join_key(self) -> None:
        """Annex 04 §1.3: a reader never meets a raw identifier."""
        drawn = [text.get_text() for text in _render().axes[0].get_xticklabels()]
        assert "unfolded_alpha_p" not in drawn

    def test_one_bar_per_contender_per_category(self) -> None:
        figure = _render()

        for container in _bar_containers(figure):
            assert len(container.patches) == 2, str(container.get_label())

    def test_the_fills_are_ordered_by_declaration_and_not_by_name(self) -> None:
        """§A.6.1: the position the analysis assigned, never the sorted name.

        **A mutant found this gap rather than a reading of the code.**
        `sorted(pairs)` in place of `sorted(pairs, key=position)` survived the
        whole suite, because the default fixture declares `("nominal",
        "rotated")` — which is *already* alphabetical, so the two orderings
        agree and neither test could tell them apart. The categories here are
        declared in the reverse of their alphabetical order for exactly that
        reason.
        """
        declared = ("rotated", "nominal")
        assert list(declared) != sorted(declared), (
            "the fixture must declare its categories out of alphabetical "
            "order, or a mutant that sorts them by name cannot be killed"
        )
        figure = _render(_table(categories=declared))

        leftmost = {
            str(container.get_label()): float(
                np.mean([patch.get_x() for patch in container.patches])
            )
            for container in _bar_containers(figure)
        }
        assert leftmost["rotated"] < leftmost["nominal"]

    def test_the_bars_of_one_category_do_not_overlap_the_next(self) -> None:
        """Grouped, not stacked and not overplotted: a reader reads two bars
        side by side or the grouping says nothing."""
        figure = _render()
        spans = [
            (patch.get_x(), patch.get_x() + patch.get_width())
            for container in _bar_containers(figure)
            for patch in container.patches
        ]
        spans.sort()
        for (_, first_end), (second_start, _) in zip(spans, spans[1:], strict=False):
            assert first_end <= second_start + 1e-9


class TestTheValueAxisIncludesZero:
    """Annex 03 §B.5.2."""

    def test_zero_is_in_the_limits(self) -> None:
        low, high = _render().axes[0].get_ylim()
        assert low <= 0.0 < high

    def test_there_is_no_truncation_key_to_declare(self) -> None:
        """The rule is enforced by the absence of the surface, not by a
        convention: a key that exists is a key that gets declared."""
        assert "ylim" not in CONFIG_KEYS
        with pytest.raises(SpecificationError, match="ylim"):
            _render(None, ylim=[8.0, 9.5])

    def test_a_log_axis_is_the_one_exception_and_it_is_declared(self) -> None:
        """§B.5.2: admissible where the quantity spans decades and has no
        meaningful zero — §A.5's compute and memory."""
        figure = _render(None, yscale="log")
        axes = figure.axes[0]

        assert axes.get_yscale() == "log"
        assert axes.get_ylim()[0] > 0.0

    def test_an_unknown_scale_is_refused_by_name(self) -> None:
        with pytest.raises(SpecificationError, match="yscale"):
            _render(None, yscale="symlog")


class TestTheDispersion:
    """§A.3.2, the same three rules `axis_scaling` obeys."""

    def test_a_positive_spread_draws_a_capped_bar(self) -> None:
        figure = _render()
        assert any(
            container.get_label().startswith("_")
            and not isinstance(container, BarContainer)
            for axes in figure.axes
            for container in axes.containers
        )

    def test_a_zero_spread_draws_no_mark(self) -> None:
        """Rule 2: an analytic contender's exact zero IS the answer, and a
        zero-height bar still draws two caps a reader meets as a narrow
        measured interval."""
        figure = _render(_table(spread=0.0))
        error_bars = [
            container
            for axes in figure.axes
            for container in axes.containers
            if not isinstance(container, BarContainer)
        ]
        assert error_bars == []

    def test_the_declared_channel_none_draws_no_mark(self) -> None:
        figure = _render(None, dispersion="none")
        error_bars = [
            container
            for axes in figure.axes
            for container in axes.containers
            if not isinstance(container, BarContainer)
        ]
        assert error_bars == []

    def test_an_unknown_channel_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="dispersion"):
            _render(None, dispersion="band")


class TestReferencesAreLevelsNotBars:
    """§B.5.2's last clause."""

    def test_a_flat_row_draws_no_bar(self) -> None:
        figure = _render()
        drawn = {
            patch.get_x()
            for container in _bar_containers(figure)
            for patch in container.patches
        }
        assert len(drawn) == 2 * len(CATEGORIES)

    def test_a_flat_row_draws_a_horizontal_line(self) -> None:
        figure = _render()
        levels = {
            round(float(line.get_ydata()[0]), 6)
            for line in figure.axes[0].get_lines()
            if len(set(np.asarray(line.get_ydata(), dtype=float))) == 1
        }
        assert {9.15, 1.86} <= levels

    def test_a_level_is_named_at_the_margin_with_its_value(self) -> None:
        """§B.5: a flat series is labelled at the right margin, in its colour,
        with its value — not in the legend."""
        figure = _render()
        texts = [text.get_text() for text in figure.axes[0].texts]

        assert any("Truncated-Riccati" in text and "9.15" in text for text in texts)

    def test_the_legend_names_the_categories_and_not_the_levels(self) -> None:
        figure = _render()
        legend = next(
            (axes.get_legend() for axes in figure.axes if axes.get_legend()), None
        )
        assert legend is not None
        entries = {text.get_text() for text in legend.get_texts()}
        assert entries == set(CATEGORIES)


class TestAClippedLevel:
    """Found by rendering the real cast, not by any test written before it.

    §B.5's margin label can only name a series at the height it is drawn, and
    a level the axis clips is not drawn at any. Measured before the fix, on a
    bound at 40 under an axis ending at 9.9: its margin label was placed at
    **axes fraction 4.04** — four panel heights above the figure and off the
    canvas, where `bbox_inches="tight"` would expand the page to hold it — and
    its line was invisible, so the bound was reported nowhere at all.

    §B.3.1 requires a bound outside the plotted range to keep its value in the
    legend, which is exactly the case the margin cannot serve.
    """

    def _clipped(self) -> pd.DataFrame:
        """A REFERENCE above every bar. Roles other than contender and baseline
        do not set the limits (§B.5), so this one is genuinely clipped.

        Its role is `REFERENCE` in **both** the table's `role` column and the
        study's declared roles, deliberately. The renderer reads the first for
        the axis rule and the second for the legend text, which is a known
        pre-existing split — Phase C recorded it as "the renderer ignores the
        tidy table's own `role` column" — and a fixture that disagreed with
        itself would be testing that defect rather than this one.
        """
        table = _table(with_flat=False)
        return pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            "contender": "riccati_unconstrained",
                            "role": Role.REFERENCE.value,
                            "axis_path": "",
                            "axis_value": np.nan,
                            "axis_label": "",
                            AXIS_KIND_COLUMN: CATEGORICAL,
                            "aggregate": 40.0,
                            "interval_low": np.nan,
                            "interval_high": np.nan,
                            "across_seed_spread": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def test_it_is_outside_the_limits_so_the_fixture_is_the_case(self) -> None:
        axes = _render(self._clipped()).axes[0]
        low, high = axes.get_ylim()
        assert not low <= 40.0 <= high

    def test_it_takes_no_margin_label(self) -> None:
        figure = _render(self._clipped())
        assert [text.get_text() for text in figure.axes[0].texts] == []

    def test_it_keeps_a_legend_row_carrying_its_value(self) -> None:
        figure = _render(self._clipped())
        legend = figure.axes[0].get_legend()
        assert legend is not None
        entries = {text.get_text() for text in legend.get_texts()}

        assert set(CATEGORIES) <= entries
        assert any(
            "Riccati (unconstrained)" in entry and "40" in entry for entry in entries
        ), entries

    def test_a_contained_level_still_takes_the_margin_and_not_the_legend(
        self,
    ) -> None:
        """The negative control: the fix must not have moved every level into
        the legend, which is the half of §B.5 that reclaimed a quarter of the
        canvas."""
        figure = _render()
        legend = figure.axes[0].get_legend()
        assert legend is not None

        assert {text.get_text() for text in legend.get_texts()} == set(CATEGORIES)
        assert [text.get_text() for text in figure.axes[0].texts] != []

    def test_every_margin_label_stays_inside_its_own_panel(self) -> None:
        """The general form of the defect: an axes fraction outside `[0, 1]` is
        a label naming a height the panel does not contain."""
        figure = _render()
        for text in figure.axes[0].texts:
            assert 0.0 <= float(text.xy[1]) <= 1.0, text.get_text()


class TestThePairingIsChecked:
    """§A.6.1's last paragraph: a mispaired kind draws a plausible, wrong
    figure, so neither renderer may accept the other's table."""

    def test_grouped_bars_refuses_a_numeric_table(self) -> None:
        with pytest.raises(SpecificationError, match="categorical"):
            _render(_table(axis_kind=NUMERIC))

    def test_grouped_bars_refuses_a_table_with_no_axis_kind_at_all(self) -> None:
        """Absent means numeric (§A.6.1), so this is the same refusal reached
        by the backwards-compatibility path."""
        table = _table()
        with pytest.raises(SpecificationError, match="categorical"):
            _render(table.drop(columns=[AXIS_KIND_COLUMN]))

    def test_axis_scaling_refuses_a_categorical_table(self) -> None:
        with pytest.raises(SpecificationError, match="categorical"):
            axis_scaling(
                FigureContext(
                    figure_id="mispaired",
                    table=_table(),
                    config={},
                    profile=resolve_profile(None),
                    series_order=ORDER,
                    roles=ROLES,
                    display_names=DISPLAY,
                )
            )

    def test_axis_scaling_still_accepts_a_table_without_the_column(self) -> None:
        """§B.1.1: an artifact written before §A.6.1 existed must still
        rebuild, so an absent `axis_kind` may not become a refusal."""
        table = _table(axis_kind=NUMERIC).drop(columns=[AXIS_KIND_COLUMN])
        table = table.assign(
            axis_value=[
                value if np.isnan(value) else float(value) + 1.0
                for value in table["axis_value"]
            ]
        )
        figure = axis_scaling(
            FigureContext(
                figure_id="legacy",
                table=table,
                config={},
                profile=resolve_profile(None),
                series_order=ORDER,
                roles=ROLES,
                display_names=DISPLAY,
            )
        )
        assert figure.axes


class TestTheConfigSurface:
    def test_an_unknown_key_is_refused_naming_what_is_available(self) -> None:
        """The defect class this project has shipped six times: a key that
        parses, reaches nothing and says nothing."""
        with pytest.raises(SpecificationError, match="typo_key"):
            _render(None, typo_key=1)

    def test_the_declared_keys_are_the_ones_the_renderer_reads(self) -> None:
        for key in ("series", "xlabel", "ylabel", "title", "dispersion", "yscale"):
            assert key in CONFIG_KEYS

    def test_selecting_a_series_drops_it_from_the_figure(self) -> None:
        figure = _render(None, series=["unfolded_alpha_p", "neural"])
        drawn = [text.get_text() for text in figure.axes[0].get_xticklabels()]
        assert drawn == [DISPLAY["unfolded_alpha_p"], DISPLAY["neural"]]

    def test_an_empty_table_is_refused_rather_than_drawn(self) -> None:
        with pytest.raises(SpecificationError, match="empty"):
            _render(_table().iloc[0:0])

    def test_a_table_missing_a_required_column_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="axis_label"):
            _render(_table().drop(columns=["axis_label"]))


class TestTheLegendCoversNoBarTop:
    """Found by rendering the end-to-end path, not by any test written before.

    §B.5 permits an inside legend to overlap because "what it can cover is the
    middle of the swept axis and never the ends". A bar has no ends: it runs
    from zero to its value, so its **top** is the only part carrying the
    number, and the top centre is exactly where §B.5 puts the key. Measured on
    the end-to-end render: the legend lay across the tops of both bars.

    The remedy is headroom rather than relegation — §B.5's own measurement says
    a legend below costs about a quarter of the plot, and on a zero-based axis
    extra headroom changes no ratio a reader reads (§B.5.2).
    """

    def _peak_and_legend_floor(self, figure: Any) -> tuple[float, float]:
        axes = figure.axes[0]
        figure.canvas.draw()
        legend = axes.get_legend()
        assert legend is not None
        floor = float(
            legend.get_window_extent().transformed(axes.transData.inverted()).y0
        )
        peak = max(
            float(patch.get_y() + patch.get_height())
            for container in _bar_containers(figure)
            for patch in container.patches
        )
        return peak, floor

    def test_no_bar_reaches_the_legend(self) -> None:
        peak, floor = self._peak_and_legend_floor(_render())
        assert peak <= floor, (
            f"tallest bar {peak:.4f} sits above legend floor {floor:.4f}"
        )

    def test_it_holds_at_the_widest_admissible_cast(self) -> None:
        categories = tuple(
            f"c{index}" for index in range(len(BAR_FILL_COLORS) * len(BAR_HATCHES))
        )
        peak, floor = self._peak_and_legend_floor(
            _render(_table(categories=categories))
        )
        assert peak <= floor

    def test_the_axis_still_starts_at_zero(self) -> None:
        """The headroom may not buy its clearance by lifting the floor, which
        would be the truncation §B.5.2 forbids."""
        assert _render().axes[0].get_ylim()[0] <= 0.0

    def test_the_ratio_between_two_bars_is_untouched(self) -> None:
        """Headroom is admissible precisely because it changes no ratio: every
        bar still starts at zero, so lengths stay proportional to values."""
        figure = _render()
        heights = sorted(
            float(patch.get_height())
            for container in _bar_containers(figure)
            for patch in container.patches
        )
        table = _table()
        values = sorted(
            float(v) for v in table[table["axis_value"].notna()]["aggregate"]
        )
        assert heights == pytest.approx(values)


class TestNegativeAndMixedValues:
    """Found by rendering a real degradation table, not by any test.

    §B.5.2 says a bar's axis includes zero. The first implementation took
    `low = min(0, ...)` and padded the top, which is right for positive data
    and silently wrong for negative: measured on real values of -68.06 % and
    -66.02 %, the limits came out (-68.06, -65.75) -- zero OUTSIDE the axis,
    every bar drawn from an origin the reader cannot see, and the panel a
    single filled rectangle. It passed the greyscale gate, because a bar that
    fills the panel is still separable.

    A percentage is exactly the quantity that goes negative, so this is the
    case Figure 2 is made of rather than an edge one.
    """

    def _signed(self, *values: float) -> pd.DataFrame:
        rows = [
            {
                "contender": label,
                "role": Role.CONTENDER.value,
                "axis_path": "p",
                "axis_value": 0.0,
                "axis_label": "shifted",
                AXIS_KIND_COLUMN: CATEGORICAL,
                "aggregate": value,
                "interval_low": np.nan,
                "interval_high": np.nan,
                "across_seed_spread": 0.0,
            }
            for label, value in zip(("unfolded_alpha_p", "neural"), values, strict=True)
        ]
        return pd.DataFrame(rows)

    def test_zero_is_the_ceiling_when_every_value_is_negative(self) -> None:
        axes = _render(self._signed(-68.06, -66.02)).axes[0]
        low, high = axes.get_ylim()

        assert low <= -68.06
        assert high >= 0.0
        assert low <= 0.0 <= high

    def test_no_bar_leaves_the_axis(self) -> None:
        """The symptom the reader actually met: a panel of solid ink."""
        figure = _render(self._signed(-68.06, -66.02))
        low, high = figure.axes[0].get_ylim()
        for container in _bar_containers(figure):
            for patch in container.patches:
                foot = float(patch.get_y())
                for edge in (foot, foot + float(patch.get_height())):
                    assert low <= edge <= high

    def test_mixed_signs_keep_zero_inside(self) -> None:
        axes = _render(self._signed(-12.0, 30.0)).axes[0]
        low, high = axes.get_ylim()

        assert low < 0.0 < high
        assert low <= -12.0
        assert high >= 30.0

    def test_an_all_positive_figure_keeps_its_floor_at_exactly_zero(self) -> None:
        """The negative control. Padding both ends unconditionally would put
        the baseline below the axis, which is the same defect mirrored."""
        assert _render(self._signed(8.0, 9.0)).axes[0].get_ylim()[0] == 0.0

    def test_the_legend_clears_a_downward_bar(self) -> None:
        """`get_y() + get_height()` is the FOOT of a negative bar, so a peak
        read from the sum alone leaves the key over the very ends a percentage
        figure is read from."""
        figure = _render(self._signed(-68.06, -66.02))
        axes = figure.axes[0]
        figure.canvas.draw()
        legend = axes.get_legend()
        assert legend is not None
        floor = float(
            legend.get_window_extent().transformed(axes.transData.inverted()).y0
        )
        peak = max(
            max(float(patch.get_y()), float(patch.get_y() + patch.get_height()))
            for container in _bar_containers(figure)
            for patch in container.patches
        )
        assert peak <= floor


class TestTheAxisRuleOnItsOwn:
    """Why this is a unit test, and it is the whole reason it exists.

    `test_zero_is_the_ceiling_when_every_value_is_negative` asserts the right
    property on a rendered figure and **a mutant deleting the rule survived
    it**: `_clear_the_legend` runs afterwards and raises the top of the axis
    until the key clears the tallest bar, which for downward bars means until
    it clears zero. The end-to-end assertion therefore passed for a second
    rule's reason, which is the shape of a check that has stopped checking.

    So §B.5.2's anchor is asserted where nothing else can repair it.
    """

    def _axes(self, *values: float, roles: Any = None) -> Any:
        figure, axes = plt.subplots()
        table = pd.DataFrame(
            [
                {
                    "contender": f"c{index}",
                    "role": (roles or Role.CONTENDER).value,
                    "aggregate": value,
                    "across_seed_spread": 0.0,
                }
                for index, value in enumerate(values)
            ]
        )
        _apply_axis_rule(axes, table, {})
        return axes

    def test_zero_is_the_ceiling_for_negative_data(self) -> None:
        low, high = self._axes(-68.06, -66.02).get_ylim()
        assert high >= 0.0
        assert low <= -68.06

    def test_zero_is_the_floor_for_positive_data(self) -> None:
        low, high = self._axes(8.0, 9.0).get_ylim()
        assert low == 0.0
        assert high >= 9.0

    def test_zero_is_interior_for_mixed_data(self) -> None:
        low, high = self._axes(-12.0, 30.0).get_ylim()
        assert low < 0.0 < high

    def test_a_far_level_does_not_stretch_the_axis(self) -> None:
        """§B.5: the axis is scaled to the data under study. A bar figure has
        LESS room to give away than a curve, not more, because its axis already
        spends the whole span from zero — so a bound at 40 against bars at 8-9
        must stay clipped and keep its legend row."""
        figure, axes = plt.subplots()
        table = pd.DataFrame(
            [
                {
                    "contender": "a",
                    "role": Role.CONTENDER.value,
                    "aggregate": 8.0,
                    "across_seed_spread": 0.0,
                },
                {
                    "contender": "b",
                    "role": Role.CONTENDER.value,
                    "aggregate": 9.0,
                    "across_seed_spread": 0.0,
                },
                {
                    "contender": "far",
                    "role": Role.BOUND.value,
                    "aggregate": 40.0,
                    "across_seed_spread": 0.0,
                },
            ]
        )
        _apply_axis_rule(axes, table, {})
        assert axes.get_ylim()[1] < 40.0

    def test_it_leaves_a_log_axis_alone(self) -> None:
        figure, axes = plt.subplots()
        axes.set_yscale("log")
        before = axes.get_ylim()
        _apply_axis_rule(
            axes,
            pd.DataFrame(
                [
                    {
                        "contender": "a",
                        "role": Role.CONTENDER.value,
                        "aggregate": 5.0,
                        "across_seed_spread": 0.0,
                    }
                ]
            ),
            {},
        )
        assert axes.get_ylim() == before


class TestTheCategoryAxisLabels:
    """§B.5.2's collision clause, added after the first real bar figure.

    §B.4.2 puts the contenders on the x axis BECAUSE a tick label is an
    identity channel colour removal cannot touch. Measured on ICASSP Figure 2
    at `ieee-2col`: nine display names, five overprinting a neighbour. A label
    a reader cannot read is not an identity channel.

    The rotation is MEASURED, not declared: whether names fit depends on how
    many there are, how long they are and how wide the venue is, so a constant
    chosen against one of them would slant a two-category figure that reads
    perfectly flat.
    """

    def _crowded(self, count: int) -> pd.DataFrame:
        """`count` contenders with names as long as the real cast's."""
        names = [
            f"UF-$\\alpha$P$^{{({index})}}$ (per-iteration P) {index}"
            for index in range(count)
        ]
        rows = [
            {
                "contender": name,
                "role": Role.CONTENDER.value,
                "axis_path": "p",
                "axis_value": 0.0,
                "axis_label": "rotated",
                AXIS_KIND_COLUMN: CATEGORICAL,
                "aggregate": 10.0 + index,
                "interval_low": np.nan,
                "interval_high": np.nan,
                "across_seed_spread": 0.0,
            }
            for index, name in enumerate(names)
        ]
        return pd.DataFrame(rows)

    def _render_crowded(self, count: int) -> Any:
        table = self._crowded(count)
        order = tuple(table["contender"])
        return grouped_bars(
            FigureContext(
                figure_id="crowded",
                table=table,
                config={},
                profile=resolve_profile("ieee-2col"),
                series_order=order,
                roles=dict.fromkeys(order, Role.CONTENDER),
                display_names={name: name for name in order},
            )
        )

    def _overlaps(self, figure: Any) -> int:
        figure.canvas.draw()
        boxes = [
            text.get_window_extent()
            for text in figure.axes[0].get_xticklabels()
            if text.get_text()
        ]
        boxes.sort(key=lambda box: box.x0)
        return sum(
            1
            for first, second in zip(boxes, boxes[1:], strict=False)
            if first.x1 > second.x0 + 0.5
        )

    def test_a_crowded_axis_does_not_overprint(self) -> None:
        """The defect itself: nine long names on a two-column figure."""
        assert self._overlaps(self._render_crowded(9)) == 0

    def test_the_fixture_would_overprint_unslanted(self) -> None:
        """The control that makes the test above mean something. Without it a
        fixture that happens to fit would pass on any implementation at all."""
        figure = self._render_crowded(9)
        axes = figure.axes[0]
        for text in axes.get_xticklabels():
            text.set_rotation(0.0)
            text.set_horizontalalignment("center")
        assert self._overlaps(figure) > 0

    def test_a_roomy_axis_is_left_flat(self) -> None:
        """A constant rotation would slant this, and a flat label is easier to
        read — so the rule has to be measured in both directions."""
        figure = _render()
        assert all(
            float(text.get_rotation()) == 0.0
            for text in figure.axes[0].get_xticklabels()
        )

    def test_a_slanted_label_is_anchored_at_its_right(self) -> None:
        """Slanted text anchored at its centre drifts away from the tick it
        names, which on a bar chart is the one thing the label is for."""
        figure = self._render_crowded(9)
        slanted = [
            text
            for text in figure.axes[0].get_xticklabels()
            if float(text.get_rotation()) != 0.0
        ]
        assert slanted
        assert all(text.get_horizontalalignment() == "right" for text in slanted)


class TestADeclaredBaseline:
    """§B.5.2 as amended 2026-08-10: a length encoding may be read from a
    declared baseline ONLY because the interruption is drawn. Every clause of
    that permission is a test here, because the rule it narrows exists to
    prevent a real deception."""

    def test_the_axis_starts_at_the_declared_base(self) -> None:
        figure = _render(ybase=7.5)
        assert float(figure.axes[0].get_ylim()[0]) == pytest.approx(7.5)

    def test_without_it_the_axis_still_includes_zero(self) -> None:
        """The rule is unchanged where nothing is declared — a truncation that
        arrived by default would be the ban repealed rather than narrowed."""
        assert float(_render().axes[0].get_ylim()[0]) <= 0.0

    def test_the_break_mark_is_drawn_at_the_baseline(self) -> None:
        """Not optional: the permission is granted BECAUSE the interruption is
        legible. Two marks, one at each end of the value axis, in axes
        fractions so a later limit change cannot strand them."""
        marks = [
            line
            for line in _render(ybase=7.5).axes[0].get_lines()
            if line.get_label() == "_break"
        ]
        assert len(marks) == 2, marks
        assert {round(float(line.get_xdata()[0]) + 0.012, 3) for line in marks} == {
            0.0,
            1.0,
        }

    def test_no_mark_is_drawn_without_a_baseline(self) -> None:
        assert not [
            line
            for line in _render().axes[0].get_lines()
            if line.get_label() == "_break"
        ]

    def test_a_base_at_or_above_a_bar_is_refused(self) -> None:
        """A bar drawn downward from its baseline reads as a negative
        quantity, which is a different claim entirely."""
        with pytest.raises(SpecificationError, match="draws it downward"):
            _render(ybase=8.31)

    def test_a_base_on_a_log_axis_is_refused(self) -> None:
        """§B.5.2: a log scale states its own floor, and two floors on one
        axis is a figure arguing with itself."""
        with pytest.raises(SpecificationError, match="two floors on one axis"):
            _render(ybase=7.5, yscale="log")


class TestACellAFamilyCannotHave:
    """§B.5.2's second 2026-08-10 clause: a cell that is not a measurement for
    its family is omitted by declaration, and every way of declaring it
    wrongly is refused by name."""

    def test_the_declared_cell_is_not_drawn(self) -> None:
        figure = _render(omit={"neural": ["rotated"]})
        heights = [
            float(patch.get_height())
            for container in figure.axes[0].containers
            if isinstance(container, BarContainer)
            for patch in container.patches
        ]
        # Two contenders x two categories, less the one omitted cell.
        assert len(heights) == 3

    def test_the_category_survives_for_everyone_else(self) -> None:
        """The omission is per CELL: dropping one family's bar must not take
        the category off the figure, or the legend would lose a fill every
        other contender still draws."""
        figure = _render(omit={"neural": ["rotated"]})
        rows = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert "rotated" in rows

    def test_an_unknown_series_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="no_such_contender"):
            _render(omit={"no_such_contender": ["rotated"]})

    def test_an_unknown_category_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="no_such_category"):
            _render(omit={"neural": ["no_such_category"]})

    def test_emptying_a_series_is_refused(self) -> None:
        """Its tick label would name nothing; `series` is the key for that."""
        with pytest.raises(SpecificationError, match="no bar at all"):
            _render(omit={"neural": list(CATEGORIES)})


class TestTheFirstThreeFillsDifferInBothChannels:
    """§B.4.2 rule 2's own words, which the construction did not honour until
    2026-08-10: `hatch = slot // 3` gave the first three slots the SAME empty
    texture, so a three-category figure separated by colour alone and a
    greyscale reader met three greys."""

    def test_three_categories_take_three_textures(self) -> None:
        from mbl.present.encodings import encode_fills

        fills = encode_fills(["a", "b", "c"])
        assert len({fill.hatch for fill in fills.values()}) == 3
        assert len({fill.color for fill in fills.values()}) == 3

    def test_all_nine_slots_stay_distinct(self) -> None:
        """The Latin square's other half: making the first three differ in
        both channels must not collide two slots further down the cycle."""
        from mbl.present.encodings import encode_fills

        fills = encode_fills([f"c{index}" for index in range(9)])
        pairs = {(fill.color, fill.hatch) for fill in fills.values()}
        assert len(pairs) == 9

    def test_the_rendered_bars_carry_them(self) -> None:
        """Through the renderer, not just the encoder: a fill computed and
        not applied is the defect class this project keeps finding."""
        figure = _render(_table(categories=("a", "b", "c")))
        hatches = {
            container.patches[0].get_hatch()
            for container in figure.axes[0].containers
            if isinstance(container, BarContainer)
        }
        assert len(hatches) == 3, hatches
