"""Acceptance for the panelled-conditions renderer (Annex 03 §B.6).

The defect this suite exists for was found by the author trying to REOPEN the
figure: it had been drawn inside a tool under a `kind` no renderer answered to,
so the store held four artifacts that could not be rebuilt while every other
figure could. The asymmetry is invisible until someone opens the file — so the
central test here is the round trip, not the drawing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
