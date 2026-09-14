"""Acceptance for the panelled-conditions renderer (Annex 03 §B.6).

The defect this suite exists for was found by the author trying to REOPEN the
figure: it had been drawn inside a tool under a `kind` no renderer answered to,
so the store held four artifacts that could not be rebuilt while every other
figure could. The asymmetry is invisible until someone opens the file — so the
central test here is the round trip, not the drawing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ArrowStyle  # noqa: E402

from mbl.present.artifacts import write_figure_artifacts  # noqa: E402
from mbl.present.profiles import profile_context, resolve_profile  # noqa: E402
from mbl.present.registry import FigureContext, resolve_figure  # noqa: E402
from mbl.present.runner import figure_from_artifacts  # noqa: E402
from mbl.present.severity_panels import severity_panels  # noqa: E402
from mbl.spec.contender import Role  # noqa: E402
from mbl.spec.errors import SpecificationError  # noqa: E402

ORDER = ("baseline", "proposed", "recurrent")
DISPLAY = {"baseline": "Clipped", "proposed": r"UF-$\alpha$P", "recurrent": "GRU"}
ROLES = {"baseline": Role.BASELINE}
PANELS = [["blind", "uninformed"], ["told", "informed"], ["world", "model mismatch"]]
ANGLES = (0.0, 5.0, 20.0, 45.0)


def _table() -> pd.DataFrame:
    rows = []
    for panel_index, (key, _) in enumerate(PANELS):
        for index, contender in enumerate(ORDER):
            for angle in ANGLES:
                rows.append(
                    {
                        "contender": contender,
                        "axis_label": f"{angle:g}",
                        "aggregate": 8.0 + panel_index + index * 0.1 + angle * 0.01,
                        "across_seed_spread": 0.01 * (index + 1),
                        "panel": key,
                    }
                )
    return pd.DataFrame(rows)


def _context(table: pd.DataFrame, **config: object) -> FigureContext:
    return FigureContext(
        figure_id="severity",
        table=table,
        config={
            "panels": PANELS,
            "xlabel": "rotation [deg]",
            "ylabel": "expected cost",
            **config,
        },
        profile=resolve_profile("ieee-2col"),
        series_order=ORDER,
        roles=ROLES,
        display_names=DISPLAY,
    )


class TestTheFaceSurvivesTheRoundTrip:
    """The defect that reached the manuscript.

    This renderer is the only registered one that does not open
    `profile_context` itself, so it took its face from whatever rcParams
    happened to be active. `tools/render_severity_figure.py` opens one, so the
    tool's output is right; `mbl figure rebuild` did not, so the obvious
    command produced Figure 2 in matplotlib's ambient sans-serif at 10 pt
    instead of the venue's serif at 8 pt — with exit code 0 and no warning.

    The repair is in two places on purpose: `_draw` opens the profile so no
    future renderer can repeat the omission, and this renderer opens its own so
    that a direct caller — which is how Figure 2 is actually produced — is
    right without depending on who called it.
    """

    def test_a_reloaded_panel_figure_carries_the_profile_s_face(
        self, tmp_path: Path
    ) -> None:
        profile = resolve_profile("ieee-2col")
        with profile_context(profile):
            figure = severity_panels(_context(_table()))
        try:
            write_figure_artifacts(
                figure,
                directory=tmp_path,
                figure_id="severity",
                table=_table(),
                spec={
                    "kind": "severity_panels",
                    "config": {
                        "panels": PANELS,
                        "xlabel": "rotation [deg]",
                        "ylabel": "expected cost",
                    },
                    "series_order": list(ORDER),
                    "roles": {k: v.value for k, v in ROLES.items()},
                    "display_names": dict(DISPLAY),
                    "rendered_at_profile": profile.name,
                },
                profile=profile,
            )
        finally:
            plt.close(figure)

        reloaded = figure_from_artifacts(tmp_path / "severity.spec.json")
        try:
            families = {
                text.get_fontfamily()[0]
                for axes in reloaded.axes
                for text in (*axes.get_xticklabels(), *axes.get_yticklabels())
                if text.get_text()
            }
            assert families, "no tick labels to inspect"
            assert families == {profile.font_family}, (
                f"the reloaded Figure 2 is in {families}; the paper's profile "
                f"is {profile.font_family}."
            )
        finally:
            plt.close(reloaded)

    def test_the_renderer_does_not_depend_on_its_caller_for_the_face(self) -> None:
        """Called with no context open at all — which is what `_draw` did."""
        figure = severity_panels(_context(_table()))
        try:
            families = {
                text.get_fontfamily()[0]
                for axes in figure.axes
                for text in axes.get_xticklabels()
                if text.get_text()
            }
            assert families == {resolve_profile("ieee-2col").font_family}
        finally:
            plt.close(figure)


class TestTheRoundTrip:
    def test_a_stored_figure_rebuilds_from_its_artifacts(self, tmp_path: Path) -> None:
        """The author's report, 2026-08-14: opening the stored spec raised
        "no figure is registered under kind 'severity_panels'". A figure the
        store cannot rebuild is a figure that only exists as a picture."""
        profile = resolve_profile("ieee-2col")
        table = _table()
        with profile_context(profile):
            figure = severity_panels(_context(table))
            write_figure_artifacts(
                figure,
                directory=tmp_path,
                figure_id="severity",
                table=table,
                spec={
                    "kind": "severity_panels",
                    "config": {
                        "panels": PANELS,
                        "xlabel": "rotation [deg]",
                        "ylabel": "expected cost",
                    },
                    "series_order": list(ORDER),
                    "roles": {"baseline": Role.BASELINE.value},
                    "display_names": DISPLAY,
                },
                profile=profile,
            )
        plt.close(figure)

        rebuilt = figure_from_artifacts(tmp_path / "severity.spec.json")
        try:
            axes = rebuilt.get_axes()
            assert [axis.get_title() for axis in axes] == [t for _, t in PANELS]
            assert [c.get_label() for c in axes[0].containers] == [
                DISPLAY[label] for label in ORDER
            ]
        finally:
            plt.close(rebuilt)

    def test_the_kind_is_registered_under_the_name_the_spec_stores(self) -> None:
        """The one-line version of the defect: the spec's `kind` and the
        registry's key must be the same string."""
        assert resolve_figure("severity_panels") is severity_panels


class TestWhatIsDrawnIsWhatWasGiven:
    def test_every_point_of_every_panel_comes_from_its_own_rows(self) -> None:
        table = _table()
        figure = severity_panels(_context(table))
        try:
            for axis, (key, _) in zip(figure.get_axes(), PANELS, strict=True):
                panel = table[table["panel"] == key]
                drawn = {c.get_label(): c for c in axis.containers}
                for contender in ORDER:
                    rows = panel[panel["contender"] == contender].sort_values(
                        "axis_label"
                    )
                    _, values = drawn[DISPLAY[contender]].lines[0].get_data()
                    assert list(values) == pytest.approx(
                        sorted(rows["aggregate"].tolist())
                    )
        finally:
            plt.close(figure)

    def test_the_spread_is_off_unless_declared(self) -> None:
        plain = severity_panels(_context(_table()))
        try:
            assert sum(len(a.collections) for a in plain.get_axes()) == 0
        finally:
            plt.close(plain)
        barred = severity_panels(_context(_table(), draw_spread=True))
        try:
            assert sum(len(a.collections) for a in barred.get_axes()) >= len(PANELS)
        finally:
            plt.close(barred)


class TestTheRefusals:
    def test_a_declared_panel_with_no_rows_is_refused(self) -> None:
        table = _table()
        with pytest.raises(SpecificationError, match="measured, and flat"):
            severity_panels(_context(table[table["panel"] != "told"]))

    def test_a_series_missing_from_one_panel_is_refused(self) -> None:
        table = _table()
        maimed = table[
            ~((table["panel"] == "told") & (table["contender"] == "recurrent"))
        ]
        with pytest.raises(SpecificationError, match="recurrent"):
            severity_panels(_context(maimed))

    def test_a_table_without_the_panel_column_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="panel"):
            severity_panels(_context(_table().drop(columns=["panel"])))

    def test_declaring_no_panels_is_refused(self) -> None:
        context = _context(_table())
        with pytest.raises(SpecificationError, match="no panels"):
            severity_panels(
                FigureContext(
                    figure_id=context.figure_id,
                    table=context.table,
                    config={"panels": []},
                    profile=context.profile,
                    series_order=context.series_order,
                    roles=context.roles,
                    display_names=context.display_names,
                )
            )


class TestTheHighlightArrow:
    """`highlight` points one series out of the band, in every panel.

    The figure draws seven curves that overlap for most of the axis, and a
    reader asked to find the proposed one has to match legend entries against
    line styles. The arrows answer that -- but only if they survive a rebuild,
    which is why they are config keys read by the renderer rather than
    something a tool draws afterwards.

    The label is a plain text and each arrow is an annotation with no text of
    its own, so one label can serve several arrows.
    """

    @staticmethod
    def _arrows(axis: Any) -> list[Any]:
        """The annotations. A plain `Text` has no `arrow_patch` attribute at
        all -- only `Annotation` does -- so this is a getattr and not a
        dotted access."""
        return [child for child in axis.texts if getattr(child, "arrow_patch", None)]

    @staticmethod
    def _labels(axis: Any, text: str) -> list[Any]:
        return [
            child
            for child in axis.texts
            if not getattr(child, "arrow_patch", None) and child.get_text() == text
        ]

    def test_every_panel_gets_a_label_and_one_arrow(self) -> None:
        figure = severity_panels(_context(_table(), highlight="proposed"))
        for axis in figure.axes:
            assert len(self._labels(axis, DISPLAY["proposed"])) == 1
            assert len(self._arrows(axis)) == 1, "no targets means one anchor"
        plt.close(figure)

    def test_one_arrow_per_declared_target(self) -> None:
        """The author's request: three arrows, to three named positions."""
        figure = severity_panels(
            _context(_table(), highlight="proposed", highlight_at=[5.0, 20.0, 45.0])
        )
        for axis in figure.axes:
            arrows = self._arrows(axis)
            assert len(arrows) == 3
            assert sorted(arrow.xy[0] for arrow in arrows) == [5.0, 20.0, 45.0]
            assert len(self._labels(axis, DISPLAY["proposed"])) == 1, "one label"
        plt.close(figure)

    def test_an_arrow_starts_outside_the_label_it_leaves(self) -> None:
        """The shaft must not cross its own text, in any direction.

        `shrinkA` is measured in points from the annotation's anchor, and the
        anchor is the CENTRE of a centred label -- so a single constant is
        either too small sideways or absurd vertically. A label is far wider
        than it is tall, and the author saw the consequence on the informed
        panel: the arrow to 45 deg ran over the word it started from.

        Asserted as a relation between the two directions rather than against a
        number, because the number is a font metric and would re-baseline every
        time the style did.
        """
        figure = severity_panels(
            _context(_table(), highlight="proposed", highlight_at=[5.0, 20.0, 45.0])
        )
        figure.canvas.draw()
        for axis in figure.axes:
            label = self._labels(axis, DISPLAY["proposed"])[0]
            box = label.get_window_extent(figure.canvas.get_renderer())
            centre = axis.transAxes.transform(label.get_position())
            for arrow in self._arrows(axis):
                shrink = arrow.arrowprops["shrinkA"]
                end = axis.transData.transform(arrow.xy)
                dx, dy = end[0] - centre[0], end[1] - centre[1]
                length = (dx * dx + dy * dy) ** 0.5
                # Where the shaft begins, in display pixels.
                start_x = centre[0] + dx / length * shrink * figure.dpi / 72.0
                start_y = centre[1] + dy / length * shrink * figure.dpi / 72.0
                assert not box.contains(start_x, start_y), (
                    f"the shaft to {arrow.xy[0]} begins inside its own label"
                )
        plt.close(figure)

    def test_no_highlight_means_no_arrows(self) -> None:
        figure = severity_panels(_context(_table()))
        for axis in figure.axes:
            assert not self._arrows(axis)
        plt.close(figure)

    def test_the_arrow_wears_the_series_encoding(self) -> None:
        """Same colour as the curve it points at -- the reader matches by eye."""
        figure = severity_panels(_context(_table(), highlight="proposed"))
        axis = figure.axes[0]
        container = next(
            c for c in axis.containers if c.get_label() == DISPLAY["proposed"]
        )
        drawn = container.lines[0]
        expected = matplotlib.colors.to_rgb(drawn.get_color())
        assert self._labels(axis, DISPLAY["proposed"])[0].get_color() == (
            drawn.get_color()
        )
        patch = self._arrows(axis)[0].arrow_patch
        assert matplotlib.colors.to_rgb(patch.get_edgecolor()[:3]) == expected
        plt.close(figure)

    def test_the_head_is_filled_rather_than_dashed(self) -> None:
        """A dashed `->` head is dashed to its tip and stops reading as a head.

        `-|>` draws a filled triangle, so the shaft can carry the series'
        linestyle while the head stays solid.
        """
        figure = severity_panels(_context(_table(), highlight="proposed"))
        for axis in figure.axes:
            for arrow in self._arrows(axis):
                style = arrow.arrow_patch.get_arrowstyle()
                assert isinstance(style, ArrowStyle.CurveFilledB), (
                    "the head must be a filled triangle (-|>), not the open "
                    f"curve (->) that the linestyle dashes: got {type(style).__name__}"
                )
        plt.close(figure)

    def test_the_label_and_every_arrow_stay_inside_the_panel(self) -> None:
        """The third panel's label left the frame when it was placed by an
        offset in points from a data anchor: where a curve sits says nothing
        about how much room is left beyond it."""
        figure = severity_panels(
            _context(_table(), highlight="proposed", highlight_at=[5.0, 20.0, 45.0])
        )
        for axis in figure.axes:
            label = self._labels(axis, DISPLAY["proposed"])[0]
            fx, fy = label.get_position()
            assert 0.0 < fx < 1.0 and 0.0 < fy < 1.0, "the label left the axes"
            low, high = axis.get_ylim()
            left, right = axis.get_xlim()
            for arrow in self._arrows(axis):
                x, y = arrow.xy
                assert left <= x <= right and low <= y <= high
        plt.close(figure)

    def test_the_anchor_is_where_the_curve_is_most_separated(self) -> None:
        """The fallback when no target is named -- not the middle position,
        which is where the curves converge.

        The table is built so that `proposed` sits on top of `baseline` at
        every angle but the last, where it is far away. Anchoring by position
        rather than by separation would point into the overlap.
        """
        table = _table()
        overlap = table["contender"].isin(("baseline", "proposed"))
        table.loc[overlap & (table["axis_label"] != "45"), "aggregate"] = 9.0
        table.loc[
            (table["contender"] == "proposed") & (table["axis_label"] == "45"),
            "aggregate",
        ] = 3.0
        figure = severity_panels(_context(table, highlight="proposed"))
        arrow = self._arrows(figure.axes[0])[0]
        assert arrow.xy[0] == pytest.approx(45.0), (
            "the arrow must land where the series separates, not at the middle"
        )
        plt.close(figure)

    def test_a_highlight_that_is_not_drawn_is_refused(self) -> None:
        with pytest.raises(SpecificationError, match="not one of the series"):
            severity_panels(_context(_table(), highlight="nonexistent"))

    def test_a_target_the_panel_never_measured_is_refused(self) -> None:
        """An arrow to an angle nobody ran points at nothing."""
        with pytest.raises(SpecificationError, match="did not"):
            severity_panels(
                _context(_table(), highlight="proposed", highlight_at=[33.0])
            )

    def test_the_arrows_survive_a_rebuild_from_the_artifacts(
        self, tmp_path: Path
    ) -> None:
        """The defect this whole suite exists for, in its newest form.

        Annotations added by the composing tool rather than by the renderer
        would draw once and vanish the moment anyone reopened the figure.
        """
        table = _table()
        context = _context(table, highlight="proposed", highlight_at=[20.0, 45.0])
        profile = resolve_profile("ieee-2col")
        with profile_context(profile):
            figure = resolve_figure("severity_panels")(context)
            write_figure_artifacts(
                figure,
                directory=tmp_path,
                figure_id="severity",
                table=table,
                spec={
                    "kind": "severity_panels",
                    "config": dict(context.config),
                    "series_order": list(ORDER),
                    "roles": {"baseline": Role.BASELINE.value},
                    "display_names": dict(DISPLAY),
                },
                profile=profile,
            )
        plt.close(figure)

        rebuilt = figure_from_artifacts(tmp_path / "severity.spec.json")
        for axis in rebuilt.axes:
            assert len(self._arrows(axis)) == 2, "the rebuild dropped an arrow"
            assert len(self._labels(axis, DISPLAY["proposed"])) == 1
        plt.close(rebuilt)
