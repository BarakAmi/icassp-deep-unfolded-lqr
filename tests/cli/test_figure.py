"""Acceptance tests for `mbl figure` — slice Phase C, step C3.

Written before the implementation. Two verbs, and the second is the one D6
exists to make possible:

**`render`** draws every figure a study declares from the analysis tables
already in the store. It trains nothing, evaluates nothing and analyses
nothing — asserted by making the alternatives raise.

**`rebuild`** re-renders one stored figure at another style profile from
`<id>.spec.json` and `<id>.data.parquet` **alone**. The checkpoint the slice
plan sets for this phase is that it must work with the store's model directory
**renamed away**; this suite goes further and removes the study document, the
analyses directory and the models directory together, because the claim is
that the pair survives all three.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from mbl.cli.app import main

from ..runner.test_producer import _write

DECLARATIONS = """
[[analyses]]
id = "cost_by_depth"
kind = "cost_vs_axis"

[[figures]]
id = "fig_cost_vs_depth"
kind = "axis_scaling"
source = "cost_by_depth"
xlabel = "unrolling depth $J$"

[[figures]]
id = "fig_second"
kind = "axis_scaling"
source = "cost_by_depth"
title = "A second figure, so `--only` has something to exclude"
"""


def _document(directory: Path, /, **overrides: Any) -> Path:
    path = _write(directory, **overrides)
    path.write_text(path.read_text() + DECLARATIONS)
    return path


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    """A store holding a run, its analysis and nothing else."""
    document = _document(tmp_path / "doc")
    store = tmp_path / "store"
    assert (
        main(["--store", str(store), "run", str(document), "--tier", "standard"]) == 0
    )
    assert main(["--store", str(store), "analyse", str(document)]) == 0
    return document, store


def _render(document: Path, store: Path, *extra: str) -> int:
    return main(["--store", str(store), "figure", "render", str(document), *extra])


def _figures(store: Path) -> Path:
    return next((store / "studies").glob("*/figures"))


class TestRender:
    def test_it_writes_four_artifacts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document, store = _prepare(tmp_path)

        assert _render(document, store) == 0

        out = capsys.readouterr().out
        assert "fig_cost_vs_depth (axis_scaling)" in out
        stems = sorted({path.name.split(".")[0] for path in _figures(store).iterdir()})
        assert stems == ["fig_cost_vs_depth", "fig_second"]
        assert sorted(
            path.name
            for path in _figures(store).iterdir()
            if path.name.startswith("fig_cost_vs_depth")
        ) == [
            "fig_cost_vs_depth.data.parquet",
            "fig_cost_vs_depth.pdf",
            "fig_cost_vs_depth.png",
            "fig_cost_vs_depth.spec.json",
        ]

    def test_only_renders_exactly_what_it_names(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mutation testing dropped `only` on the way through and nothing
        # failed, because the fixture declared one figure -- so "render only
        # this one" and "render them all" were the same command. The document
        # declares two now.
        document, store = _prepare(tmp_path)

        assert _render(document, store, "--only", "fig_second") == 0

        out = capsys.readouterr().out
        assert "fig_second" in out
        assert "fig_cost_vs_depth" not in out
        stems = {path.name.split(".")[0] for path in _figures(store).iterdir()}
        assert stems == {"fig_second"}

    def test_the_style_reaches_the_render(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document, store = _prepare(tmp_path)

        assert _render(document, store, "--style", "ieee-2col") == 0

        assert "at style ieee-2col" in capsys.readouterr().out
        spec = json.loads((_figures(store) / "fig_cost_vs_depth.spec.json").read_text())
        assert spec["rendered_at_profile"] == "ieee-2col"

    def test_an_unknown_style_is_refused_by_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document, store = _prepare(tmp_path)
        assert _render(document, store, "--style", "ieee-3col") != 0
        assert "ieee-2col" in capsys.readouterr().out

    def test_rendering_neither_trains_nor_analyses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mbl.analysis.runner as analysis_runner
        import mbl.runner.producer as producer

        document, store = _prepare(tmp_path)

        def detonate(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("mbl figure reached the runner or the analysis")

        monkeypatch.setattr(producer, "run_study", detonate)
        monkeypatch.setattr(producer, "default_synthesize", detonate)
        monkeypatch.setattr(analysis_runner, "run_analyses", detonate)

        assert _render(document, store) == 0

    def test_a_figure_whose_analysis_was_never_run_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        main(["--store", str(store), "run", str(document), "--tier", "standard"])

        assert _render(document, store) != 0
        assert "mbl analyse" in capsys.readouterr().out

    def test_a_study_declaring_no_figures_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _write(tmp_path / "doc")
        store = tmp_path / "store"
        main(["--store", str(store), "run", str(document), "--tier", "standard"])

        assert _render(document, store) == 0
        assert "declares no figures" in capsys.readouterr().out

    def test_an_absent_store_is_refused_before_anything_is_read(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _document(tmp_path / "doc")
        assert _render(document, tmp_path / "absent") != 0
        assert "no store at" in capsys.readouterr().out
        assert not (tmp_path / "absent").exists()


class TestRebuild:
    def test_it_restyles_from_the_pair_alone(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The slice plan's checkpoint, and then some.

        The plan asks for the model directory renamed away. The claim is
        stronger than that: the spec and the data survive the study document,
        the analyses and the models all being gone at once, so all three go.
        """
        document, store = _prepare(tmp_path)
        assert _render(document, store) == 0

        document.unlink()
        shutil.rmtree(store / "models")
        for path in _figures(store).parent.glob("analyses/*"):
            path.unlink()

        code = main(
            [
                "--store",
                str(store),
                "figure",
                "rebuild",
                "fig_cost_vs_depth",
                "--style",
                "ieee-2col",
            ]
        )

        assert code == 0
        assert "at style ieee-2col" in capsys.readouterr().out
        spec = json.loads((_figures(store) / "fig_cost_vs_depth.spec.json").read_text())
        assert spec["rendered_at_profile"] == "ieee-2col"

    def test_the_rebuilt_pdf_is_a_different_width(self, tmp_path: Path) -> None:
        # The property `--style` sells, asserted on the artifact rather than on
        # the message: a re-style that produced an identical file would satisfy
        # every assertion about the spec.
        document, store = _prepare(tmp_path)
        _render(document, store, "--style", "ieee-1col")
        narrow = (_figures(store) / "fig_cost_vs_depth.png").stat().st_size

        main(
            [
                "--store",
                str(store),
                "figure",
                "rebuild",
                "fig_cost_vs_depth",
                "--style",
                "talk",
            ]
        )
        wide = (_figures(store) / "fig_cost_vs_depth.png").stat().st_size

        assert narrow != wide

    def test_the_data_artifact_is_unchanged_by_a_restyle(self, tmp_path: Path) -> None:
        import pandas as pd

        document, store = _prepare(tmp_path)
        _render(document, store)
        before = pd.read_parquet(_figures(store) / "fig_cost_vs_depth.data.parquet")

        main(
            [
                "--store",
                str(store),
                "figure",
                "rebuild",
                "fig_cost_vs_depth",
                "--style",
                "neurips",
            ]
        )
        after = pd.read_parquet(_figures(store) / "fig_cost_vs_depth.data.parquet")

        pd.testing.assert_frame_equal(before, after)

    def test_an_unknown_figure_is_refused_by_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, store = _prepare(tmp_path)
        code = main(["--store", str(store), "figure", "rebuild", "typo"])
        assert code != 0
        assert "typo" in capsys.readouterr().out


class TestRebuildReconstructsTheWholeContext:
    """What the spec must carry for a rebuild to draw the same figure.

    Mutation testing removed the roles and the declaration order from the
    reconstructed context and every test still passed — because all of them
    asserted that files appeared, and none asserted what was *in* them. The
    context is captured here instead, which is the only place the claim is
    checkable without re-reading a PDF.
    """

    def _capture(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        from mbl.present import registry as registry_module

        seen: list[Any] = []
        original = registry_module._REGISTRY["axis_scaling"]

        def capturing(context: Any) -> Any:
            seen.append(context)
            return original(context)

        monkeypatch.setitem(registry_module._REGISTRY, "axis_scaling", capturing)
        return seen

    def test_the_roles_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mbl.spec.contender import Role

        document, store = _prepare(tmp_path)
        _render(document, store)
        seen = self._capture(monkeypatch)

        assert (
            main(["--store", str(store), "figure", "rebuild", "fig_cost_vs_depth"]) == 0
        )

        assert seen, "the renderer was never reached"
        assert seen[-1].roles["baseline"] is Role.BASELINE

    def test_the_declaration_order_survives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without it the encodings are re-derived from the table, so a rebuild
        # repaints a figure that was already published.
        document, store = _prepare(tmp_path)
        _render(document, store)
        seen = self._capture(monkeypatch)

        main(["--store", str(store), "figure", "rebuild", "fig_cost_vs_depth"])

        assert tuple(seen[-1].series_order) == ("unfolded_a", "baseline")

    def test_a_rebuild_that_loses_separability_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """The gate runs on a rebuild too, before anything is overwritten."""
        from matplotlib import pyplot as plt

        from mbl.present import registry as registry_module

        document, store = _prepare(tmp_path)
        _render(document, store)
        before = (_figures(store) / "fig_cost_vs_depth.pdf").read_bytes()

        def unseparable(context: Any) -> Any:
            figure, axes = plt.subplots()
            for label in ("a", "b"):
                axes.plot([1, 2], [1, 2], label=label, color="#2a78d6", linestyle="-")
            return figure

        monkeypatch.setitem(registry_module._REGISTRY, "axis_scaling", unseparable)

        code = main(["--store", str(store), "figure", "rebuild", "fig_cost_vs_depth"])

        assert code != 0
        assert "greyscale" in capsys.readouterr().out
        assert (_figures(store) / "fig_cost_vs_depth.pdf").read_bytes() == before


class TestAnAmbiguousIdIsAQuestion:
    def test_two_studies_holding_one_figure_id_are_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Picking the first would make the answer depend on directory
        # iteration order, which is exactly the class of silent choice this
        # architecture removes everywhere else.
        document, store = _prepare(tmp_path)
        _render(document, store)
        original = _figures(store)
        twin = store / "studies" / ("f" * 16) / "figures"
        twin.mkdir(parents=True)
        for path in original.iterdir():
            shutil.copy2(path, twin / path.name)

        assert (
            main(["--store", str(store), "figure", "rebuild", "fig_cost_vs_depth"]) != 0
        )
        assert "2 studies" in capsys.readouterr().out
