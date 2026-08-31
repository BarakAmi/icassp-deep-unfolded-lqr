"""Reloading a figure from its two raw files (`figure_from_artifacts`).

D6's argument for storing a data-plus-spec pair rather than a pickled `Figure`
is that a pickle silently fails to load across matplotlib versions and can never
be re-styled. `rebuild_figure` already exercised half of that — it finds a figure
by id, re-renders it and writes the four artifacts back in place, which is what a
command line wants. What was missing is the half a person wants: hand it the two
files and get the live figure back, writing nothing.

**The two must go through one code path.** A figure someone inspected and a
figure the store holds have to be the same figure, so both call `_draw`, and the
tests below assert the equivalence rather than trusting it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mbl.present import figure_from_artifacts
from mbl.present.artifacts import write_figure_artifacts
from mbl.present.profiles import resolve_profile
from mbl.present import registry
from mbl.present.registry import FigureContext
from mbl.present.runner import rebuild_figure
from mbl.spec.contender import Role
from mbl.spec.errors import SpecificationError
from mbl.present.axis_scaling import axis_scaling

ROLES = {
    "unfolded_alpha": Role.CONTENDER,
    "neural": Role.CONTENDER,
    "truncated_riccati": Role.BASELINE,
}
DISPLAY = {"neural": "GRU", "truncated_riccati": "Truncated-Riccati"}


@pytest.fixture(autouse=True)
def _close_every_figure():
    yield
    plt.close("all")


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
        "across_seed_spread": 0.0,
        "interval_low": float(value),
        "interval_high": float(value),
        "n_seeds": 5,
        "n_trajectories": 32768,
    }


SPEC: dict[str, Any] = {
    "id": "fig",
    "kind": "axis_scaling",
    "source": "cost",
    "study": "test/study",
    "study_id": "0" * 16,
    "config": {"title": "a title", "xlabel": "J"},
    "series_order": list(ROLES),
    "roles": {label: role.value for label, role in ROLES.items()},
    "display_names": dict(DISPLAY),
    "rendered_at_profile": "ieee-2col",
}


@pytest.fixture
def stored(tmp_path: Path) -> Path:
    """A real four-artifact bundle, written by the writer that writes them."""
    table = _table()
    figure = axis_scaling(
        FigureContext(
            figure_id="fig",
            table=table,
            config=dict(SPEC["config"]),
            profile=resolve_profile("ieee-2col"),
            series_order=list(ROLES),
            roles=dict(ROLES),
            display_names=dict(DISPLAY),
        )
    )
    try:
        write_figure_artifacts(
            figure,
            directory=tmp_path,
            figure_id="fig",
            table=table,
            spec=SPEC,
            profile=resolve_profile("ieee-2col"),
        )
    finally:
        plt.close(figure)
    return tmp_path / "fig.spec.json"


class TestTheDrawAppliesTheProfileToMatplotlib:
    """`_draw` must open the profile, not merely hand it to the renderer.

    The distinction was invisible until Figure 2 was rebuilt and came back in
    the wrong face. `axis_scaling` and `grouped_bars` each open
    `profile_context` themselves; `severity_panels` does not, and relied on its
    caller. `tools/render_severity_figure.py` opens one, so the tool was
    right — and `_draw` did not, so `mbl figure rebuild` was wrong, silently,
    exit code 0.

    Fixing only `severity_panels` would leave the next renderer free to make
    the same omission, so the guarantee is placed in `_draw` and asserted here
    against a renderer that deliberately does **not** protect itself. A probe
    renderer is used rather than a real one precisely because every real one
    would pass this test without `_draw` doing anything at all.
    """

    @pytest.fixture
    def probe_kind(self) -> Any:
        """Register a renderer that records the rcParams it was drawn under."""
        seen: dict[str, Any] = {}

        def _probe(context: FigureContext) -> Any:
            seen["font.family"] = plt.rcParams["font.family"]
            seen["font.size"] = plt.rcParams["font.size"]
            seen["pdf.fonttype"] = plt.rcParams["pdf.fonttype"]
            return plt.figure()

        registry._REGISTRY["_probe_profile"] = _probe
        try:
            yield seen
        finally:
            registry._REGISTRY.pop("_probe_profile", None)

    def test_the_renderer_runs_under_the_profile_s_settings(
        self, stored: Path, probe_kind: dict[str, Any]
    ) -> None:
        spec_path = stored.parent / "probe.spec.json"
        spec_path.write_text(json.dumps({**SPEC, "kind": "_probe_profile"}))
        (stored.parent / "probe.data.parquet").write_bytes(
            (stored.parent / "fig.data.parquet").read_bytes()
        )
        figure_from_artifacts(spec_path, style="ieee-2col")
        profile = resolve_profile("ieee-2col")
        assert probe_kind["font.family"] == [profile.font_family], (
            f"the renderer drew under font.family={probe_kind['font.family']}; "
            f"the profile asks for {profile.font_family}. `_draw` ran outside "
            "`profile_context`."
        )
        assert probe_kind["font.size"] == profile.base_font_pt
        assert probe_kind["pdf.fonttype"] == 42

    def test_the_settings_do_not_leak_past_the_draw(self, stored: Path) -> None:
        """`rc_context`, not a global update — the reason `profile_context`
        exists. A notebook that reloaded one figure at `ieee-2col` must not
        silently restyle every figure drawn after it."""
        before = plt.rcParams["font.family"], plt.rcParams["pdf.fonttype"]
        figure_from_artifacts(stored)
        assert (plt.rcParams["font.family"], plt.rcParams["pdf.fonttype"]) == before


class TestItRendersFromTheTwoFilesAlone:
    def test_it_returns_a_figure(self, stored: Path) -> None:
        figure = figure_from_artifacts(stored)
        assert figure.axes

    def test_it_draws_every_series_the_table_carries(self, stored: Path) -> None:
        figure = figure_from_artifacts(stored)
        drawn = {
            str(line.get_label()).split(" (")[0]
            for axes in figure.axes
            for line in axes.get_lines()
            if not str(line.get_label()).startswith("_")
        }
        assert drawn == {DISPLAY.get(k, k) for k in ROLES}, drawn

    def test_it_needs_no_study_no_analysis_and_no_store(self, stored: Path) -> None:
        """The property D6 is about. Everything but the two files is deleted
        and the figure still renders — including the PDF and PNG, so a caller
        cannot be secretly reading the rendered image."""
        for suffix in (".pdf", ".png"):
            (stored.parent / f"fig{suffix}").unlink()
        assert sorted(p.name for p in stored.parent.iterdir()) == [
            "fig.data.parquet",
            "fig.spec.json",
        ]
        assert figure_from_artifacts(stored).axes

    def test_it_writes_nothing(self, stored: Path) -> None:
        """Looking at a figure may not change the artifact of record."""
        before = {p.name: p.read_bytes() for p in stored.parent.iterdir()}
        figure_from_artifacts(stored)
        after = {p.name: p.read_bytes() for p in stored.parent.iterdir()}
        assert before == after


class TestTheProfile:
    def test_the_default_is_the_profile_it_was_STORED_at(self, stored: Path) -> None:
        """Not the project default. Reloading a figure to look at it must
        reproduce the stored artifact, or the thing inspected is not the thing
        on disk — and `thesis` is 6.0 in against `ieee-2col`'s 7.16."""
        figure = figure_from_artifacts(stored)
        assert figure.get_size_inches()[0] == pytest.approx(
            resolve_profile("ieee-2col").width_in
        )

    def test_an_explicit_style_re_styles_it(self, stored: Path) -> None:
        figure = figure_from_artifacts(stored, style="ieee-1col")
        assert figure.get_size_inches()[0] == pytest.approx(
            resolve_profile("ieee-1col").width_in
        )

    def test_an_unknown_style_is_refused_by_name(self, stored: Path) -> None:
        with pytest.raises(KeyError, match="ieee-3col"):
            figure_from_artifacts(stored, style="ieee-3col")


class TestItAgreesWithTheCommandThatWritesTheStore:
    def test_the_same_spec_and_data_give_the_same_marks(self, stored: Path) -> None:
        """`rebuild_figure` and `figure_from_artifacts` share `_draw`; this
        asserts the consequence rather than the arrangement. A figure a person
        inspected and a figure the store holds must be the same figure."""
        store = stored.parent.parent.parent
        (store / "studies").mkdir(parents=True, exist_ok=True)
        target = store / "studies" / "aaaaaaaaaaaaaaaa" / "figures"
        target.mkdir(parents=True, exist_ok=True)
        for path in stored.parent.iterdir():
            (target / path.name).write_bytes(path.read_bytes())

        outcome = rebuild_figure(store=store, figure_id="fig", style="ieee-2col")
        assert outcome.profile == "ieee-2col"
        rebuilt = figure_from_artifacts(target / "fig.spec.json")
        direct = figure_from_artifacts(stored)
        assert _marks(rebuilt) == _marks(direct)


class TestARefusalAnAuthorCanActOn:
    def test_a_missing_spec_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError, match="no figure specification"):
            figure_from_artifacts(tmp_path / "absent.spec.json")

    def test_a_missing_data_file_is_not_a_silent_empty_figure(
        self, stored: Path
    ) -> None:
        (stored.parent / "fig.data.parquet").unlink()
        with pytest.raises(Exception):
            figure_from_artifacts(stored)

    def test_an_unregistered_kind_is_refused(self, stored: Path) -> None:
        spec = json.loads(stored.read_text())
        spec["kind"] = "not_a_renderer"
        stored.write_text(json.dumps(spec))
        with pytest.raises(SpecificationError, match="not_a_renderer"):
            figure_from_artifacts(stored)


def _marks(figure: Any) -> list[tuple[str, tuple[float, ...]]]:
    return sorted(
        (
            str(line.get_label()),
            tuple(round(float(value), 12) for value in line.get_ydata()),
        )
        for axes in figure.axes
        for line in axes.get_lines()
        if not str(line.get_label()).startswith("_")
    )
