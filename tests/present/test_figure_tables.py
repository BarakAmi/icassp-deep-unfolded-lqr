"""Every figure carries its table, and the table is the figure re-read (§B.1.2).

"A curve answers *which*; it does not answer *by how much*. ... a reader who
has to measure a figure with a ruler has been handed a picture instead of a
result." The author's findings document specifies a table beside every one of
the four figures, and nothing under `src/` emitted one.

Two objects now share the word "table" and §B.1.2 keeps them apart. §A.6's
**tidy table** is the data — `<id>.data.parquet`, machine-read. The
**presented table** is what a reader meets — `<id>.table.tex` and
`<id>.table.md` — and it is a *rendering* of the tidy table in exactly the
sense the PDF is: same input, second output channel, no computation of its
own. "A presented table that computed anything would be a third statement
about one study, and the two that already exist disagree often enough."

The four clauses are each mechanically checkable and each has a class here.

1. **One frame.** Both are rendered from `<id>.data.parquet` and nothing else.
2. **One row set.** Every mark a reader can see has a row, and every row is
   drawn. **This one was already false**: the renderer selected
   `config["series"]` internally while the runner froze the *unselected*
   analysis table, so a figure drawing one contender of two froze six rows for
   three drawn marks — and `<id>.data.parquet` claimed to be "the exact values
   plotted" while it was not. Measured before the fix.
3. **One set of numbers.** A table value is the frame's value *formatted*,
   never recomputed, at a declared precision.
4. **One set of names.** Rows are named by the display name; the join key is
   machinery and is never printed.

And the table is **optional at the artifact level**: a figure declaring none
emits exactly the four artifacts of §B.1, because "a fifth *required* member
would mark every already-rendered figure incomplete and stop replay resolving
until the whole store was re-rendered".
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mbl.present.artifacts import FIGURE_SUFFIXES
from mbl.present.runner import rebuild_figure, render_figures
from mbl.present.tables import (
    ABSENT_CELL,
    MACHINERY_COLUMNS,
    TABLE_FORMATS,
    render_tables,
)
from mbl.spec.analysis import AnalysisSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.figure import FigureSpec

TABLE_SUFFIXES = tuple(f".table.{name}" for name in sorted(TABLE_FORMATS))


def _study_with_figure(tmp_path: Path, **figure_overrides: Any):
    """The runner fixture, with a declarable table."""
    from mbl.analysis import run_analyses
    from mbl.experiments import DEFAULT_SPEC_BINDINGS
    from mbl.runner.producer import run_study
    from mbl.spec.loader import load_study

    from ..runner.test_producer import _write

    document = load_study(_write(tmp_path / "doc"), bindings=DEFAULT_SPEC_BINDINGS)
    study = dataclasses.replace(
        document.study,
        analyses=(AnalysisSpec(id="cost_by_depth", kind="cost_vs_axis"),),
        figures=(
            FigureSpec(
                id="fig_cost_vs_depth",
                kind="axis_scaling",
                source="cost_by_depth",
                **figure_overrides,
            ),
        ),
    )
    store = tmp_path / "store"
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    run_analyses(study, store=store)
    return study, store


def _figures_dir(study: Any, store: Path) -> Path:
    return store / "studies" / str(study.study_id) / "figures"


class TestTheTableIsOptionalAtTheArtifactLevel:
    def test_a_figure_declaring_none_emits_exactly_four(self, tmp_path: Path) -> None:
        study, store = _study_with_figure(tmp_path)
        render_figures(study, store=store)

        directory = _figures_dir(study, store)
        written = sorted(path.name for path in directory.iterdir())
        assert written == sorted(f"fig_cost_vs_depth{s}" for s in FIGURE_SUFFIXES)

    def test_a_figure_declaring_one_emits_four_plus_one_per_format(
        self, tmp_path: Path
    ) -> None:
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        render_figures(study, store=store)

        directory = _figures_dir(study, store)
        written = sorted(path.name for path in directory.iterdir())
        assert written == sorted(
            f"fig_cost_vs_depth{s}" for s in (*FIGURE_SUFFIXES, *TABLE_SUFFIXES)
        )

    def test_the_four_artifact_contract_still_reports_four(
        self, tmp_path: Path
    ) -> None:
        """§B.1's `FigureArtifacts.paths` is what the completeness check a
        notebook runs counts. A table is emitted beside the four and is
        "**never** counted as part of the four"."""
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        outcomes = render_figures(study, store=store)
        assert len(outcomes[0].artifacts.paths) == 4

    def test_an_unserialisable_declaration_is_refused_where_it_is_written(
        self,
    ) -> None:
        with pytest.raises(SpecificationError):
            FigureSpec(id="f", kind="axis_scaling", source="a", table={"p": object()})


class TestClauseOneOneFrame:
    def test_the_table_rebuilds_with_the_analysis_deleted(self, tmp_path: Path) -> None:
        """The clause, executed. If the emitter reached for
        `analyses/<source>.parquet` it could not do this — and a re-run
        analysis would silently make the table disagree with the PDF beside
        it, with nothing in the artifact set able to detect it."""
        import shutil

        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        render_figures(study, store=store)
        directory = _figures_dir(study, store)
        before = (directory / "fig_cost_vs_depth.table.md").read_text()

        shutil.rmtree(store / "studies" / str(study.study_id) / "analyses")
        rebuild_figure(store=store, figure_id="fig_cost_vs_depth")

        assert (directory / "fig_cost_vs_depth.table.md").read_text() == before

    def test_the_table_is_derivable_from_the_frozen_parquet_alone(
        self, tmp_path: Path
    ) -> None:
        # Independent recomputation: the emitter is re-run from the artifact
        # on disk, with no store and no study in reach, and must agree.
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        render_figures(study, store=store)
        directory = _figures_dir(study, store)

        frame = pd.read_parquet(directory / "fig_cost_vs_depth.data.parquet")
        spec = json.loads((directory / "fig_cost_vs_depth.spec.json").read_text())
        again = render_tables(
            frame,
            declaration=spec["table"],
            display_names=spec["display_names"],
            declarations=spec["declarations"],
        )
        for name, text in again.items():
            stored = directory / f"fig_cost_vs_depth.table.{name}"
            assert stored.read_text() == text, name


class TestClauseTwoOneRowSet:
    def test_a_selecting_figure_freezes_only_what_it_drew(self, tmp_path: Path) -> None:
        """The defect this clause names, measured before the fix.

        `axis_scaling` applied `config["series"]` internally and the runner
        froze the table it was handed, so a figure drawing one contender of
        two froze BOTH — six rows behind three drawn marks. The parquet is
        documented as "the exact values plotted", so the claim was false for
        the PDF as well as for the table.
        """
        study, store = _study_with_figure(
            tmp_path, config={"series": ["unfolded_a"]}, table={"precision": 4}
        )
        render_figures(study, store=store)
        directory = _figures_dir(study, store)

        frame = pd.read_parquet(directory / "fig_cost_vs_depth.data.parquet")
        assert sorted(set(frame["contender"])) == ["unfolded_a"]

    def test_every_table_row_is_a_frame_row_and_conversely(
        self, tmp_path: Path
    ) -> None:
        study, store = _study_with_figure(
            tmp_path, config={"series": ["unfolded_a"]}, table={"precision": 4}
        )
        render_figures(study, store=store)
        directory = _figures_dir(study, store)

        frame = pd.read_parquet(directory / "fig_cost_vs_depth.data.parquet")
        body = [
            line
            for line in (directory / "fig_cost_vs_depth.table.md")
            .read_text()
            .split("\n")
            if line.startswith("|") and not set(line) <= set("|-: ")
        ]
        assert len(body) == len(frame) + 1  # + the header row

    def test_selecting_a_series_the_analysis_lacks_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        # Moving the selection earlier must not move the guard: a figure
        # quietly missing the contender its caption is about is the plausible
        # and wrong output this architecture exists to prevent.
        study, store = _study_with_figure(tmp_path, config={"series": ["nope"]})
        with pytest.raises(SpecificationError, match="nope"):
            render_figures(study, store=store)


class TestClauseThreeOneSetOfNumbers:
    def test_every_numeric_cell_is_the_frames_value_formatted(
        self, tmp_path: Path
    ) -> None:
        """Compared element-wise against the parquet, which is the check
        §B.1.2 asks for by name."""
        study, store = _study_with_figure(
            tmp_path, table={"precision": 6, "columns": ["contender", "aggregate"]}
        )
        render_figures(study, store=store)
        directory = _figures_dir(study, store)

        frame = pd.read_parquet(directory / "fig_cost_vs_depth.data.parquet")
        text = (directory / "fig_cost_vs_depth.table.md").read_text()
        for value in frame["aggregate"]:
            assert f"{float(value):.6f}" in text

    def test_the_declared_precision_is_what_is_shown(self, tmp_path: Path) -> None:
        study, store = _study_with_figure(
            tmp_path, table={"precision": 2, "columns": ["contender", "aggregate"]}
        )
        render_figures(study, store=store)
        text = (_figures_dir(study, store) / "fig_cost_vs_depth.table.md").read_text()
        # Two decimals and no more: a value printed at the frame's full
        # float64 width would be a different presentation decision, silently.
        assert re.search(r"\|\s*\d+\.\d{2}\s*\|", text)
        assert not re.search(r"\|\s*\d+\.\d{3,}\s*\|", text)

    def test_an_absent_value_and_a_zero_are_printed_differently(self) -> None:
        """§A.3.2's two facts, which the figure deliberately cannot show.

        "the distinction is carried by the table, which prints `—` and `0.000`
        and means two different things." A one-seed row has no across-seed
        quantity at all; an analytic contender has a measured zero.
        """
        frame = pd.DataFrame(
            [
                {"contender": "learned", "aggregate": 1.0, "across_seed_spread": 0.25},
                {"contender": "analytic", "aggregate": 2.0, "across_seed_spread": 0.0},
                {
                    "contender": "one_seed",
                    "aggregate": 3.0,
                    "across_seed_spread": np.nan,
                },
            ]
        )
        rendered = render_tables(
            frame,
            declaration={"precision": 3},
            display_names={},
            declarations={},
        )["md"]
        rows = {
            line.split("|")[1].strip(): line
            for line in rendered.split("\n")
            if line.startswith("|")
        }
        assert "0.000" in rows["analytic"]
        assert ABSENT_CELL in rows["one_seed"]
        assert ABSENT_CELL not in rows["analytic"]

    def test_nothing_is_recomputed(self) -> None:
        """A negative control on the emitter's whole contract: hand it a frame
        whose `aggregate` disagrees with its own `per_seed_aggregate`, and the
        table must print the frame's number. An emitter that re-derived
        anything would silently correct — and disagree with the PDF."""
        frame = pd.DataFrame(
            [
                {
                    "contender": "a",
                    "aggregate": 99.0,
                    "per_seed_aggregate": [1.0, 2.0],
                }
            ]
        )
        rendered = render_tables(
            frame, declaration={"precision": 1}, display_names={}, declarations={}
        )["md"]
        assert "99.0" in rendered


class TestClauseFourOneSetOfNames:
    def test_rows_are_named_by_the_display_name(self) -> None:
        frame = pd.DataFrame([{"contender": "unfolded_alpha_p", "aggregate": 1.0}])
        rendered = render_tables(
            frame,
            declaration={"precision": 3},
            display_names={"unfolded_alpha_p": "Unfolded-$\\alpha$+P"},
            declarations={},
        )
        for text in rendered.values():
            assert "Unfolded-$\\alpha$+P" in text
            assert "unfolded_alpha_p" not in text

    def test_a_label_with_no_display_name_falls_back_to_itself(self) -> None:
        frame = pd.DataFrame([{"contender": "cocp", "aggregate": 1.0}])
        rendered = render_tables(
            frame, declaration={"precision": 3}, display_names={}, declarations={}
        )["md"]
        assert "cocp" in rendered

    def test_machinery_columns_are_not_shown_by_default(self) -> None:
        """The join key, the axis path and the role are plumbing. A reader
        moving between figure, table and caption must never meet two names for
        one contender."""
        frame = pd.DataFrame(
            [
                {
                    "contender": "a",
                    "role": "contender",
                    "axis_path": "contenders.*.config.num_iterations",
                    "axis_label": "2",
                    "aggregate": 1.0,
                }
            ]
        )
        rendered = render_tables(
            frame, declaration={"precision": 3}, display_names={}, declarations={}
        )["md"]
        for column in MACHINERY_COLUMNS:
            assert column not in rendered, column

    def test_an_author_may_ask_for_a_machinery_column_explicitly(self) -> None:
        # A default is not a prohibition: `columns` is the author's say, and a
        # role column is legitimate in a table that groups by it.
        frame = pd.DataFrame([{"contender": "a", "role": "bound", "aggregate": 1.0}])
        rendered = render_tables(
            frame,
            declaration={"precision": 3, "columns": ["contender", "role"]},
            display_names={},
            declarations={},
        )["md"]
        assert "bound" in rendered


class TestTheDeclarationsTravelWithTheTable:
    def test_the_spec_freezes_section_a4s_declarations(self, tmp_path: Path) -> None:
        """§B.1.2: they live in the analysis's sidecar, and a figure must
        rebuild with no analysis in reach — so they are frozen into
        `<id>.spec.json` at render time."""
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        render_figures(study, store=store)
        spec = json.loads(
            (_figures_dir(study, store) / "fig_cost_vs_depth.spec.json").read_text()
        )
        declarations = spec["declarations"]
        assert declarations["aggregate"] == "mean"
        assert declarations["interval"]["kind"] == "bootstrap_bca"
        assert declarations["interval"]["level"] == 0.95
        assert declarations["n_training_seeds"] == [2]
        assert declarations["n_evaluation_trajectories"] > 0

    def test_the_rendered_table_states_them(self, tmp_path: Path) -> None:
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        render_figures(study, store=store)
        directory = _figures_dir(study, store)
        for suffix in TABLE_SUFFIXES:
            text = (directory / f"fig_cost_vs_depth{suffix}").read_text()
            assert "mean" in text, suffix
            assert "95" in text, suffix

    def test_they_are_emitted_and_not_typed(self, tmp_path: Path) -> None:
        # §A.4: "emitted automatically from the specification, so they cannot
        # be omitted or misstated". Declaring a median must move the note.
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        study = dataclasses.replace(
            study,
            analyses=(
                AnalysisSpec(
                    id="cost_by_depth",
                    kind="cost_vs_axis",
                    config={"aggregate": "median"},
                ),
            ),
        )
        from mbl.analysis import run_analyses

        run_analyses(study, store=store)
        render_figures(study, store=store)
        text = (_figures_dir(study, store) / "fig_cost_vs_depth.table.md").read_text()
        assert "median" in text


class TestBothFormatsAreWellFormed:
    def test_a_snake_case_header_is_humanised_rather_than_escaped(self) -> None:
        """`across_seed_spread` reaches a reader as "across seed spread".

        Humanising also removes the commonest LaTeX hazard a column name
        carries, which is why the escape below needs a header that survives
        it — a test using underscores would assert nothing about escaping at
        all, and would look like it did.
        """
        frame = pd.DataFrame([{"contender": "a", "across_seed_spread": 1.0}])
        tex = render_tables(
            frame, declaration={"precision": 3}, display_names={}, declarations={}
        )["tex"]
        assert "across seed spread" in tex
        assert "across_seed_spread" not in tex

    def test_latex_escapes_a_derived_header_but_not_a_display_name(self) -> None:
        """Two provenances, two rules, and getting them the same way round is
        the trap.

        A column header comes from the DATA and any LaTeX special in it breaks
        compilation. A display name is author-written LaTeX
        (`Unfolded-$\\alpha$+P`, and the TOML single-quoting trap is why) —
        escaping it would print the markup instead of rendering it.
        """
        frame = pd.DataFrame([{"contender": "a", "cost % of floor": 1.0}])
        tex = render_tables(
            frame,
            declaration={"precision": 3},
            display_names={"a": "Unfolded-$\\alpha$+P"},
            declarations={},
        )["tex"]
        assert r"cost \% of floor" in tex
        assert "Unfolded-$\\alpha$+P" in tex
        assert r"Unfolded-\$" not in tex

    def test_latex_is_a_balanced_tabular(self) -> None:
        frame = pd.DataFrame([{"contender": "a", "aggregate": 1.0}])
        tex = render_tables(
            frame, declaration={"precision": 3}, display_names={}, declarations={}
        )["tex"]
        assert tex.count(r"\begin{tabular}") == 1
        assert tex.count(r"\end{tabular}") == 1
        assert r"\toprule" in tex and r"\bottomrule" in tex

    def test_markdown_has_one_alignment_row(self) -> None:
        frame = pd.DataFrame([{"contender": "a", "aggregate": 1.0}])
        md = render_tables(
            frame, declaration={"precision": 3}, display_names={}, declarations={}
        )["md"]
        alignment = [
            line for line in md.split("\n") if set(line) <= set("|-: ") and "|" in line
        ]
        assert len(alignment) == 1

    def test_every_row_has_the_header_s_column_count(self) -> None:
        frame = pd.DataFrame(
            [{"contender": "a", "aggregate": 1.0, "across_seed_spread": np.nan}]
        )
        md = render_tables(
            frame, declaration={"precision": 3}, display_names={}, declarations={}
        )["md"]
        rows = [line for line in md.split("\n") if line.startswith("|")]
        widths = {line.count("|") for line in rows}
        assert len(widths) == 1, rows


class TestAtomicity:
    def test_a_table_that_cannot_be_written_leaves_the_previous_four(
        self, tmp_path: Path
    ) -> None:
        """§B.1's staging promise, extended to the fifth and sixth files. A
        render that fails halfway must not leave three new artifacts beside
        two old ones."""
        study, store = _study_with_figure(tmp_path, table={"precision": 4})
        render_figures(study, store=store)
        directory = _figures_dir(study, store)
        first = (directory / "fig_cost_vs_depth.pdf").read_bytes()

        broken = dataclasses.replace(
            study,
            figures=(
                FigureSpec(
                    id="fig_cost_vs_depth",
                    kind="axis_scaling",
                    source="cost_by_depth",
                    table={"precision": 4, "columns": ["no_such_column"]},
                ),
            ),
        )
        with pytest.raises(SpecificationError, match="no_such_column"):
            render_figures(broken, store=store)
        assert (directory / "fig_cost_vs_depth.pdf").read_bytes() == first


class TestTheDeclarationReachesTheField:
    """The seventh inert declaration, prevented.

    `_figures` sweeps every key that is not `id`, `kind` or `source` into
    `config`. `[figures.table]` would therefore have parsed cleanly, landed in
    `config["table"]`, and been ignored — a document declaring a table, a
    render emitting four artifacts, and nothing anywhere saying why. This
    project has shipped six of those and caught them each time by perturbing
    the declaration rather than by reading the loader.
    """

    def test_a_declared_table_reaches_the_field_and_not_the_config(
        self, tmp_path: Path
    ) -> None:
        from mbl.experiments import DEFAULT_SPEC_BINDINGS
        from mbl.spec.loader import load_study

        from ..runner.test_producer import _write

        path = _write(tmp_path / "doc")
        path.write_text(
            path.read_text()
            + '\n[[analyses]]\nid = "cost_by_depth"\nkind = "cost_vs_axis"\n'
            + "\n[[figures]]\n"
            'id = "f"\nkind = "axis_scaling"\nsource = "cost_by_depth"\n'
            "\n[figures.table]\nprecision = 5\n"
            'caption = "Cost against depth."\n',
            encoding="utf-8",
        )
        study = load_study(path, bindings=DEFAULT_SPEC_BINDINGS).study

        figure = study.figures[-1]
        assert figure.table == {"precision": 5, "caption": "Cost against depth."}
        assert "table" not in figure.config

    def test_it_takes_effect_rather_than_being_echoed_back(
        self, tmp_path: Path
    ) -> None:
        """Perturbation, not invocation: the declared precision must MOVE the
        emitted bytes. Asserting that the field holds what was written would
        hold equally with the emitter never consulting it."""
        study, store = _study_with_figure(
            tmp_path, table={"precision": 1, "columns": ["contender", "aggregate"]}
        )
        render_figures(study, store=store)
        coarse = (_figures_dir(study, store) / "fig_cost_vs_depth.table.md").read_text()

        study = dataclasses.replace(
            study,
            figures=(
                dataclasses.replace(
                    study.figures[0],
                    table={"precision": 6, "columns": ["contender", "aggregate"]},
                ),
            ),
        )
        render_figures(study, store=store)
        fine = (_figures_dir(study, store) / "fig_cost_vs_depth.table.md").read_text()
        assert coarse != fine

    def test_a_table_that_is_not_a_table_is_refused(self, tmp_path: Path) -> None:
        from mbl.experiments import DEFAULT_SPEC_BINDINGS
        from mbl.spec.loader import load_study

        from ..runner.test_producer import _write

        path = _write(tmp_path / "doc")
        path.write_text(
            path.read_text() + '\n[[figures]]\nid = "f"\nkind = "axis_scaling"\n'
            'source = "cost_by_depth"\ntable = 4\n',
            encoding="utf-8",
        )
        with pytest.raises(SpecificationError, match="table"):
            load_study(path, bindings=DEFAULT_SPEC_BINDINGS)


class TestPrecisionIsDeclaredAtTheGranularityTheDataHas:
    """One number for a whole table is the wrong granularity.

    Found by emitting the tracked study's real table: at `precision = 4` an
    unrolling depth of J = 1 printed as `1.0000`. The axis is integer-valued
    and the cost is not, and §B.1.2 clause 3 already calls the displayed
    precision *a declared presentation decision* — a single number simply
    cannot express a decision the data has two of.
    """

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"contender": "a", "axis_value": 1.0, "aggregate": 2.10604}]
        )

    def test_a_single_number_still_applies_to_every_column(self) -> None:
        md = render_tables(
            self._frame(),
            declaration={"precision": 4},
            display_names={},
            declarations={},
        )["md"]
        assert "1.0000" in md and "2.1060" in md

    def test_a_per_column_declaration_is_honoured(self) -> None:
        md = render_tables(
            self._frame(),
            declaration={"precision": {"axis_value": 0, "default": 4}},
            display_names={},
            declarations={},
        )["md"]
        assert "| 1 |" in md, md
        assert "2.1060" in md
        assert "1.0000" not in md

    def test_a_precision_for_a_column_nobody_prints_is_refused(self) -> None:
        """A declaration that reaches nothing is this project's recurring
        defect, and a per-column map is a new place for one to hide."""
        with pytest.raises(SpecificationError, match="no_such_column"):
            render_tables(
                self._frame(),
                declaration={
                    "columns": ["contender", "aggregate"],
                    "precision": {"no_such_column": 2},
                },
                display_names={},
                declarations={},
            )

    def test_a_negative_precision_is_refused_in_either_form(self) -> None:
        for declaration in ({"precision": -1}, {"precision": {"aggregate": -1}}):
            with pytest.raises(SpecificationError, match="precision"):
                render_tables(
                    self._frame(),
                    declaration=declaration,
                    display_names={},
                    declarations={},
                )

    def test_the_trajectory_count_is_labelled_as_the_total_it_is(self) -> None:
        """The sidecar sums `n_trajectories` over every row, so the number is
        a study total. Printed bare, a reader takes it for the n behind the
        row they are looking at — 4 423 680 against a per-point 163 840 on the
        tracked study."""
        md = render_tables(
            self._frame(),
            declaration={"precision": 3},
            display_names={},
            declarations={"n_evaluation_trajectories": 4423680},
        )["md"]
        assert "evaluation trajectories (total): 4423680" in md
