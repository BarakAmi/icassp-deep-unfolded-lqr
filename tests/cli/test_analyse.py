"""Acceptance tests for `mbl analyse` — slice Phase B, step B3.

Written before the implementation. Tier 6 has existed since B2 and has been
reachable only from Python; this is the entry point that makes a table
producible by a person, and it is the same shape as `mbl run`: the decision
lives in `cli/analyse.py` and is returned as `(ok, message)`, because
`cli/app.py` is the only module on this surface allowed to print.

**The property this phase must not lose is that analysing does not run.**
`mbl analyse` reads a study document to learn what varies and reads the store
for the numbers; it trains nothing, evaluates nothing, and constructs no
controller. Asserting that by timing would be asserting a duration, so it is
asserted by making the alternative *raise*: `run_study` and `build_controller`
are both monkeypatched to detonate, and the command still succeeds.

The second property is the exit code. A study whose measurements are missing
must **fail**, not print a cheerful empty table — the analysis is the last step
before a figure, and a table that silently omits a contender is the plausible
and wrong output this whole re-architecture exists to make impossible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mbl.cli.app import main

from ..runner.test_producer import _write

#: The fixture's shape, named rather than repeated: two depths for the swept
#: contender plus one flat baseline.
EXPECTED_ROWS = 3

ANALYSES = """
[[analyses]]
id = "cost_by_depth"
kind = "cost_vs_axis"
"""


def _document(directory: Path, /, **overrides: Any) -> Path:
    """The producer fixture's document, plus a declared analysis."""
    path = _write(directory, **overrides)
    path.write_text(path.read_text() + ANALYSES)
    return path


def _run(document: Path, store: Path, *extra: str) -> int:
    return main(
        ["--store", str(store), "run", str(document), "--tier", "standard", *extra]
    )


def _analyse(document: Path, store: Path, *extra: str) -> int:
    return main(["--store", str(store), "analyse", str(document), *extra])


class TestTheHappyPath:
    def test_it_writes_the_table_and_the_sidecar(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        assert _run(document, store) == 0

        assert _analyse(document, store) == 0

        out = capsys.readouterr().out
        assert "cost_by_depth" in out
        assert f"{EXPECTED_ROWS} row" in out
        written = list((store / "studies").rglob("cost_by_depth.parquet"))
        assert len(written) == 1
        sidecar = written[0].with_suffix(".json")
        assert json.loads(sidecar.read_text())["kind"] == "cost_vs_axis"

    def test_the_message_names_the_kind_and_the_row_count_as_a_pair(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Asserted as a pair rather than as two substrings: "cost_vs_axis" and
        # "3" both appear in a message that has them the wrong way round, which
        # is the trap slice Phase A paid for.
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)
        _analyse(document, store)

        assert "cost_by_depth (cost_vs_axis): 3 rows" in capsys.readouterr().out

    def test_re_analysing_replaces_and_still_succeeds(self, tmp_path: Path) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)

        assert _analyse(document, store) == 0
        assert _analyse(document, store) == 0

    def test_a_named_analysis_may_be_selected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)

        assert _analyse(document, store, "--only", "cost_by_depth") == 0
        assert "cost_by_depth" in capsys.readouterr().out


class TestAnalysingDoesNotRun:
    def test_it_neither_trains_nor_builds_a_controller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pinned by making the alternative raise, not by timing it.

        A duration assertion is a measurement of the machine. Detonating
        `run_study` and `build_controller` makes "no training happened" a
        property of what is reachable from the command's inputs.
        """
        import mbl.runner.producer as producer

        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)

        def detonate(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("mbl analyse reached the runner")

        monkeypatch.setattr(producer, "run_study", detonate)
        monkeypatch.setattr(producer, "default_synthesize", detonate)

        assert _analyse(document, store) == 0

    def test_it_runs_with_the_model_directory_removed(self, tmp_path: Path) -> None:
        """The destructive form of the same claim.

        An analysis that reached for weights could not run once the
        checkpoints were gone. They are deleted, and it still succeeds.
        """
        import shutil

        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)
        shutil.rmtree(store / "models")

        assert _analyse(document, store) == 0


class TestTheRefusals:
    def test_a_study_with_no_measurements_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Not an empty table and a zero exit. The analysis is the last step
        # before a figure, and a silently short table is the plausible-and-
        # wrong output this architecture exists to make impossible.
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)
        # A different tier's models are different models, so its measurements
        # are absent while the store itself is real.
        assert _analyse(document, store, "--tier", "smoke") != 0
        assert "no measurement" in capsys.readouterr().out

    def test_a_subsetted_run_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`require_analysable_measurement`, reached from a command line.

        Four depths, so `smoke`'s cap of two actually truncates the axis --
        with the fixture's two-value axis nothing is dropped and the run is
        correctly not stamped.
        """
        document = _document(tmp_path / "doc", depths=[1, 2, 4, 8])
        store = tmp_path / "store"
        assert (
            main(
                [
                    "--store",
                    str(store),
                    "run",
                    str(document),
                    "--tier",
                    "smoke",
                ]
            )
            == 0
        )

        assert _analyse(document, store, "--tier", "smoke") != 0
        assert "axis_subset" in capsys.readouterr().out

    def test_a_study_declaring_no_analyses_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _write(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)

        assert _analyse(document, store) == 0
        assert "declares no analyses" in capsys.readouterr().out

    def test_selecting_an_undeclared_analysis_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)

        assert _analyse(document, store, "--only", "typo") != 0
        assert "typo" in capsys.readouterr().out

    def test_an_absent_store_is_refused_before_anything_is_read(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `mbl analyse` READS the store, so it must not be exempted from
        # main()'s existing-store check the way `mbl run` is -- a mistyped
        # --store would otherwise create an empty one and report it as empty.
        document = _document(tmp_path / "doc")

        assert _analyse(document, tmp_path / "absent") != 0
        assert "no store at" in capsys.readouterr().out
        assert not (tmp_path / "absent").exists()

    def test_a_malformed_document_is_refused_with_the_grammar_s_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        _run(document, store)
        document.write_text(document.read_text().replace('kind = "cost_vs_axis"', ""))

        assert _analyse(document, store) != 0
        assert "analyses[0]" in capsys.readouterr().out


class TestTheTierIsHonoured:
    def test_the_tier_decides_which_measurements_are_read(self, tmp_path: Path) -> None:
        """A tier changes every identifier (F2a), so it changes which
        measurements exist. Analysing at the wrong tier must not quietly find
        a different study's numbers -- it must find none and say so.

        The fixture's axis has two values and `smoke` caps an axis at two, so
        this run is correctly **not** stamped as subsetted; the subsetting
        refusal is exercised separately, on a four-value axis. Getting that
        wrong is what the first version of this test did.
        """
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        assert (
            main(["--store", str(store), "run", str(document), "--tier", "smoke"]) == 0
        )

        assert _analyse(document, store, "--tier", "smoke") == 0
        assert _analyse(document, store, "--tier", "standard") != 0


class TestTheOverridesReachTheTable:
    """`--set` is the third editing level, and mutation testing found it
    entirely untested on this command.

    Two mutants survived the first pass: dropping `overrides` on the way to
    `plan_run`, and re-parsing `--set` with a second little parser instead of
    the one `mbl run` uses. Both are the same defect seen from two sides -- a
    table computed for a different study than the run produced -- and neither
    is visible in the message, because the message names the study id, which
    an override does not change.
    """

    def test_a_table_computed_under_the_run_s_overrides_succeeds(
        self, tmp_path: Path
    ) -> None:
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        override = "--set", "training.batch.effective_size=64"
        assert _run(document, store, *override) == 0

        assert _analyse(document, store, *override) == 0

    def test_the_same_table_without_them_finds_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The other half, and the one that makes the test above mean anything:
        # a batch size is part of every `ModelID`, so analysing without the
        # override looks for measurements that were never produced.
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        assert _run(document, store, "--set", "training.batch.effective_size=64") == 0

        assert _analyse(document, store) != 0
        assert "no measurement" in capsys.readouterr().out

    def test_the_value_is_parsed_as_toml_and_not_as_text(self, tmp_path: Path) -> None:
        # `--set training.seeds=2` must reach the grammar as the integer 2.
        # A second parser that returned the string "2" would be refused by the
        # per-path coercion, far from the flag that caused it -- which is
        # exactly why this command shares `mbl run`'s parser rather than
        # growing its own.
        document = _document(tmp_path / "doc")
        store = tmp_path / "store"
        override = "--set", "training.seeds=2"
        assert _run(document, store, *override) == 0

        assert _analyse(document, store, *override) == 0
