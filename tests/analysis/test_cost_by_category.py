"""Acceptance tests for `cost_by_category` — ICASSP Phase D, the analysis half.

Written before the implementation. Annex 03 §A.6.1 is the specification and it
was written from measurements taken with these files open:

* a swept axis whose values are **names** already materialises and already
  reaches identity — six points, six distinct `ModelID`s, the value carried
  through as a Python string;
* what fails is the reduction, and it fails **as a crash**:
  `float("learned_step_size")` raises a bare `ValueError` out of the statistic,
  which an author cannot tell from a defect *in* the statistic.

So there are two obligations here and they are separate. `cost_vs_axis` must
**refuse** a non-numeric axis by name, and the categorical case gets its own
kind because the *figure* differs and not the statistic — a category has no
position, so a line drawn through two of them asserts a rate of change that
does not exist.

**The checkpoint class is independent recomputation**, as it is for
`cost_vs_axis`: one category's aggregate is recomputed by hand from the stored
per-trajectory costs and compared with what the analysis emits. A test that
only checked the schema would pass on an analysis that reduced the wrong rows.

**Two properties carry the rest**, and each has a mutant waiting for it:

* **Order is the study's declared order, never the sorted one.** The fixture
  declares its categories in an order that is *not* alphabetical and *not* the
  order of the measured aggregates, so a mutant that sorts either way is
  killed. A figure whose bar order followed its own data would reorder itself
  whenever a seed moved.
* **`axis_kind` is emitted so the pairing can be checked rather than trusted.**
  A categorical table rendered by `axis_scaling` would draw a line through
  positions 0, 1, 2 and read as a trend — plausible, and wrong.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mbl.analysis.cost_by_category import (
    AXIS_KIND_COLUMN,
    CATEGORICAL,
    _one_degradation,
    cost_by_category,
)
from mbl.analysis.reduction import Declared, axis_key
from mbl.analysis.cost_vs_axis import cost_vs_axis
from mbl.analysis.registry import AnalysisContext, registered_analyses, resolve_analysis
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.runner.producer import run_study
from mbl.spec.analysis import AnalysisSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study
from mbl.spec.problem import GeneratorProvenance, ProblemSpec
from mbl.spec.study import Composition, SweepAxis
from mbl.store.content_store import MeasurementStore

from ..runner import test_producer as tp
from ..runner.test_producer import _write

ANALYSIS = AnalysisSpec(id="cost_by_variant", kind="cost_by_category")

#: The sweep the fixture replaces its numeric one with. A real, identity-
#: bearing, string-valued field of the unfolded recipe — measured to give one
#: distinct `ModelID` per (value, seed) — rather than an invented axis, so
#: this suite is about the analysis and not about whether the grammar can
#: carry a string.
CATEGORY_PATH = "contenders.*.config.kind"

#: **Declared in an order that is neither alphabetical nor sorted by the
#: measured aggregate.** `"learned_step_size"` sorts *before*
#: `"learned_step_size_and_matrix"`, so a mutant that sorts the categories
#: alphabetically reverses this and is killed by
#: `test_the_category_order_is_the_declared_one`.
CATEGORIES = ("learned_step_size_and_matrix", "learned_step_size")

#: The fixture's shape: two categories for the swept contender, plus one
#: contender the axis does not apply to.
EXPECTED_ROWS = 3


def _study(directory: Path, /, **overrides: Any):
    return load_study(
        _write(directory, **overrides), bindings=DEFAULT_SPEC_BINDINGS
    ).study


def _categorical(study: Any, values: tuple[str, ...] = CATEGORIES):
    """The same study, swept over a STRING-valued axis instead of the depth.

    `dataclasses.replace` rather than a second document, and rather than a
    textual edit of the fixture's TOML: the sweep is applied at
    `materialise()`, so replacing it here produces exactly the points a
    document declaring it would.
    """
    return dataclasses.replace(
        study,
        sweep=(
            SweepAxis(path=CATEGORY_PATH, values=values, applies_to=("unfolded_a",)),
        ),
    )


def _run(directory: Path, store: Path, /, **overrides: Any):
    study = _categorical(_study(directory, **overrides))
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    return study


def _context(study: Any, store: Path, spec: AnalysisSpec = ANALYSIS) -> AnalysisContext:
    return AnalysisContext(
        study=study,
        study_id=str(study.study_id),
        spec=spec,
        measurements=MeasurementStore(store),
    )


def _row(table: pd.DataFrame, contender: str, category: str) -> pd.Series:
    rows = table[(table["contender"] == contender) & (table["axis_label"] == category)]
    assert len(rows) == 1, f"{contender}/{category}: {len(rows)} rows, expected 1"
    return rows.iloc[0]


class TestTheRegistry:
    def test_the_kind_resolves_by_its_annex_name(self) -> None:
        assert resolve_analysis("cost_by_category") is cost_by_category
        assert "cost_by_category" in registered_analyses()


class TestTheDefectItExistsFor:
    def test_a_string_axis_crashes_cost_vs_axis_today(self, tmp_path: Path) -> None:
        """The measurement §A.6.1 was written from, pinned as a test.

        Not `pytest.raises(ValueError)` — the point is that the failure is
        **refused by name** now, so this asserts the refusal and names the
        kind an author is supposed to reach for. A bare `ValueError` escaping
        a statistic is indistinguishable from a defect in the statistic.
        """
        study = _run(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError, match="cost_by_category"):
            cost_vs_axis(
                _context(
                    study,
                    tmp_path / "store",
                    AnalysisSpec(id="numeric", kind="cost_vs_axis"),
                )
            )

    def test_the_refusal_names_the_offending_value(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError, match="learned_step_size"):
            cost_vs_axis(
                _context(
                    study,
                    tmp_path / "store",
                    AnalysisSpec(id="numeric", kind="cost_vs_axis"),
                )
            )

    def test_a_numeric_axis_still_goes_through_cost_vs_axis(
        self, tmp_path: Path
    ) -> None:
        """The refusal must not have closed the door it was standing beside."""
        study = _study(tmp_path / "doc")
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        table = cost_vs_axis(
            _context(
                study, tmp_path / "store", AnalysisSpec(id="n", kind="cost_vs_axis")
            )
        ).table
        assert sorted(table[table["contender"] == "unfolded_a"]["axis_value"]) == [
            2.0,
            4.0,
        ]


class TestTheTidyTable:
    def test_one_row_per_contender_per_category(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        assert len(table) == EXPECTED_ROWS
        assert set(table["contender"]) == {"unfolded_a", "baseline"}

    def test_the_category_travels_as_a_name(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        swept = table[table["contender"] == "unfolded_a"]
        assert set(swept["axis_label"]) == set(CATEGORIES)

    def test_the_category_order_is_the_declared_one(self, tmp_path: Path) -> None:
        """§A.6.1: declared order, never sorted — and the fixture is chosen so
        the two differ. `axis_value` is the position, so this is the column a
        renderer places bars from."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        positions = {
            str(row["axis_label"]): float(row["axis_value"])
            for _, row in table.iterrows()
            if str(row["axis_label"])
        }
        assert positions == {CATEGORIES[0]: 0.0, CATEGORIES[1]: 1.0}
        assert sorted(CATEGORIES) != list(CATEGORIES), (
            "the fixture must declare its categories out of alphabetical order, "
            "or a mutant that sorts them cannot be killed"
        )

    def test_the_order_does_not_follow_the_measured_values(
        self, tmp_path: Path
    ) -> None:
        """A figure whose bar order followed its own data would reorder itself
        whenever a seed moved, and two runs could not be compared.

        Declaring the same two categories the other way round must move the
        positions and nothing else — which is only true if the position comes
        from the declaration.
        """
        study = _run(tmp_path / "doc", tmp_path / "store")
        forward = cost_by_category(_context(study, tmp_path / "store")).table
        reversed_study = _categorical(study, values=tuple(reversed(CATEGORIES)))
        backward = cost_by_category(_context(reversed_study, tmp_path / "store")).table

        for category in CATEGORIES:
            first = _row(forward, "unfolded_a", category)
            second = _row(backward, "unfolded_a", category)
            assert float(first["axis_value"]) != float(second["axis_value"])
            assert float(first["aggregate"]) == pytest.approx(
                float(second["aggregate"])
            )

    def test_a_contender_the_axis_misses_carries_no_category(
        self, tmp_path: Path
    ) -> None:
        """Annex 03 §A.6.1: it keeps the null `axis_value` that makes it a
        reference level rather than a bar."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        flat = table[table["contender"] == "baseline"]
        assert len(flat) == 1
        assert bool(np.isnan(float(flat.iloc[0]["axis_value"])))
        assert str(flat.iloc[0]["axis_label"]) == ""

    def test_the_table_declares_that_its_axis_is_categorical(
        self, tmp_path: Path
    ) -> None:
        """So `axis_scaling` can refuse it rather than draw a trend through it."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        assert AXIS_KIND_COLUMN in table.columns
        assert set(table[AXIS_KIND_COLUMN]) == {CATEGORICAL}

    def test_it_carries_the_three_dispersions_of_A_3_2(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        for column in (
            "within_seed_spread",
            "across_seed_spread",
            "interval_low",
            "interval_high",
            "seeds",
            "per_seed_aggregate",
            "n_seeds",
            "n_trajectories",
            "role",
        ):
            assert column in table.columns, column


class TestTheArithmetic:
    def test_the_aggregate_is_recomputed_by_hand(self, tmp_path: Path) -> None:
        """The checkpoint: mean over trajectories within a seed, then over
        seeds — §A.3's order, performed independently of the analysis."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        store = tmp_path / "store"
        table = cost_by_category(_context(study, store)).table
        measurements = MeasurementStore(store)

        category = CATEGORIES[0]
        per_seed = []
        for point in study.materialise():
            if point.contender.resolved_label != "unfolded_a":
                continue
            if point.axis_values.get(CATEGORY_PATH) != category:
                continue
            record = measurements.get(str(point.measurement_id))
            assert record.samples is not None
            per_seed.append(float(np.mean(record.samples["trajectory_cost"])))

        assert len(per_seed) == len(study.training.seeds)
        row = _row(table, "unfolded_a", category)
        assert float(row["aggregate"]) == pytest.approx(float(np.mean(per_seed)))
        assert float(row["across_seed_spread"]) == pytest.approx(
            float(np.std(np.asarray(per_seed), ddof=1))
        )
        assert int(row["n_seeds"]) == len(per_seed)

    def test_two_categories_are_not_pooled(self, tmp_path: Path) -> None:
        """The failure a `float()` crash was hiding: if the category were
        dropped from the group key, both categories would reduce into one row
        and the across-seed spread would be manufactured from four points that
        are not four seeds."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        swept = table[table["contender"] == "unfolded_a"]
        assert len(swept) == len(CATEGORIES)
        assert set(swept["n_seeds"]) == {len(study.training.seeds)}


class TestTheGuardsItInherits:
    #: The axis must be NAMED to reach the pooling at all: `axis_path` already
    #: refuses a two-axis study that defaults it. So the defect lives precisely
    #: where an author has answered that question and reasonably believes they
    #: have said what they meant.
    NAMED = AnalysisSpec(
        id="cost_by_variant",
        kind="cost_by_category",
        config={"axis": CATEGORY_PATH},
    )

    def _two_axis(self, directory: Path, store: Path):
        study = dataclasses.replace(
            _categorical(_study(directory)),
            training=dataclasses.replace(_study(directory).training, seeds=(0,)),
        )
        study = dataclasses.replace(
            study,
            sweep=(
                study.sweep[0],
                SweepAxis(
                    path="contenders.*.config.step_size_init",
                    values=(0.1, 0.2),
                    applies_to=("unfolded_a",),
                ),
            ),
        )
        run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
        return study

    def test_defaulting_the_axis_is_already_refused(self, tmp_path: Path) -> None:
        """Asserted so the guard below is not credited with its work."""
        study = self._two_axis(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError, match="cannot be defaulted"):
            cost_by_category(_context(study, tmp_path / "store"))

    def test_a_second_swept_axis_is_refused_by_name(self, tmp_path: Path) -> None:
        """`require_replicates`, shared rather than re-derived: a second axis
        lands in the group and fabricates the error bar."""
        study = self._two_axis(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError, match="distinct training seed"):
            cost_by_category(_context(study, tmp_path / "store", self.NAMED))

    def test_an_unrun_study_is_refused_rather_than_returning_nothing(
        self, tmp_path: Path
    ) -> None:
        study = _categorical(_study(tmp_path / "doc"))
        with pytest.raises(SpecificationError, match="no measurement"):
            cost_by_category(_context(study, tmp_path / "store"))


class TestANumericAxisMayBeDeclaredCategorical:
    def test_the_kind_does_not_require_string_values(self, tmp_path: Path) -> None:
        """§A.6.1's mechanism clause: restricting the kind to string axes would
        make the renderer's reach depend on what today's campaign sweeps."""
        study = _study(tmp_path / "doc")
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        table = cost_by_category(_context(study, tmp_path / "store")).table

        swept = table[table["contender"] == "unfolded_a"]
        assert sorted(swept["axis_label"]) == ["2", "4"]
        assert sorted(swept["axis_value"]) == [0.0, 1.0]
        assert set(table[AXIS_KIND_COLUMN]) == {CATEGORICAL}


class TestTheSidecar:
    def test_it_states_the_declarations_A_4_requires(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        sidecar = cost_by_category(_context(study, tmp_path / "store")).sidecar

        assert sidecar["kind"] == "cost_by_category"
        assert sidecar["aggregation_order"] == [
            "evaluation_trajectories",
            "training_seeds",
        ]
        assert sidecar["across_seed_spread_over"] == "training_seeds"
        assert sidecar["axis"] == CATEGORY_PATH

    def test_it_states_the_axis_kind_and_the_declared_categories(
        self, tmp_path: Path
    ) -> None:
        """So a reader of the sidecar alone can tell what the x positions mean
        and in what order they were declared."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        sidecar = cost_by_category(_context(study, tmp_path / "store")).sidecar

        assert sidecar["axis_kind"] == CATEGORICAL
        assert list(sidecar["categories"]) == list(CATEGORIES)


# --------------------------------------------------------------------------
# A spec-valued axis: the label is read here, so the refusal lives here
# --------------------------------------------------------------------------

SHIFTED = "rotated.npz"
CONDITIONS = ("nominal", "rotated 30 deg")


def _shift_document(directory: Path, *, labels: bool = True) -> Path:
    """The producer fixture swept over its EVALUATION problem.

    Figure 2's Experiment 1 exactly: the same models scored on a second plant,
    zero extra trainings. Written as one document because Annex 01 §2.5.1 made
    `evaluation.problem` sweepable; it needed four documents before.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tp._problem_file(directory, "problem.npz", seed=0)
    tp._problem_file(directory, SHIFTED, seed=1)
    text = tp.STUDY.format(**tp.DEFAULTS)
    head, marker, _ = text.partition("[[sweep]]")
    assert marker
    block = 'path = "evaluation.problem"\nvalues = ["problem.npz", "rotated.npz"]'
    if labels:
        block += f'\nlabels = ["{CONDITIONS[0]}", "{CONDITIONS[1]}"]'
    (directory / "study.toml").write_text(f"{head}[[sweep]]\n{block}\n")
    return directory / "study.toml"


def _shift_study(directory: Path, store: Path, *, labels: bool = True):
    study = load_study(
        _shift_document(directory, labels=labels), bindings=DEFAULT_SPEC_BINDINGS
    ).study
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    return study


class TestASpecValuedAxisNeedsItsLabels:
    """Annex 01 §2.5.1, corrected: required where READ, not where declared."""

    def test_an_unlabelled_spec_axis_is_refused_naming_the_axis(
        self, tmp_path: Path
    ) -> None:
        study = _shift_study(tmp_path / "doc", tmp_path / "store", labels=False)
        with pytest.raises(SpecificationError, match="evaluation.problem"):
            cost_by_category(_context(study, tmp_path / "store"))

    def test_the_refusal_says_what_to_add(self, tmp_path: Path) -> None:
        study = _shift_study(tmp_path / "doc", tmp_path / "store", labels=False)
        with pytest.raises(SpecificationError, match="labels"):
            cost_by_category(_context(study, tmp_path / "store"))

    def test_a_labelled_axis_uses_the_label_as_the_category(
        self, tmp_path: Path
    ) -> None:
        study = _shift_study(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        assert set(table["axis_label"]) == set(CONDITIONS)

    def test_the_label_never_leaks_a_repr(self, tmp_path: Path) -> None:
        """The whole reason labels exist: `str(ProblemSpec)` is a repr."""
        study = _shift_study(tmp_path / "doc", tmp_path / "store")
        table = cost_by_category(_context(study, tmp_path / "store")).table

        assert not any("ProblemSpec" in str(value) for value in table["axis_label"])


class TestPercentageDegradation:
    """The author's decision, 2026-08-07: Figure 2 plots percentage degradation
    relative to the nominal plant.

    `D_s = 100 * (c_s(shifted) - c_s(nominal)) / c_s(nominal)`, per contender
    per SEED, with the figure drawing the mean of `D_s`.

    **The pairing is the part that can be got wrong.** It is the relative
    change of the mean, not the mean of per-trajectory relative changes; and it
    is paired per seed, which is what makes an analytic contender's degradation
    carry a spread of exactly 0.0 rather than a number combined from two
    independently reduced columns.
    """

    RELATIVE = AnalysisSpec(
        id="degradation",
        kind="cost_by_category",
        config={"relative_to": CONDITIONS[0]},
    )

    def _tables(self, tmp_path: Path):
        study = _shift_study(tmp_path / "doc", tmp_path / "store")
        absolute = cost_by_category(_context(study, tmp_path / "store")).table
        relative = cost_by_category(
            _context(study, tmp_path / "store", self.RELATIVE)
        ).table
        return study, absolute, relative

    def test_the_reference_category_is_not_drawn(self, tmp_path: Path) -> None:
        """A bar at exactly 0 % has no length and would spend a fill slot and a
        legend row on ink that says nothing."""
        _, _, relative = self._tables(tmp_path)

        assert set(relative["axis_label"]) == {CONDITIONS[1]}

    def test_the_value_is_recomputed_by_hand_from_the_absolute_table(
        self, tmp_path: Path
    ) -> None:
        """The checkpoint. Independent of the implementation: the per-seed
        aggregates the absolute table already carries are paired here by hand.
        """
        _, absolute, relative = self._tables(tmp_path)

        for contender in set(relative["contender"]):
            nominal = _row(absolute, contender, CONDITIONS[0])
            shifted = _row(absolute, contender, CONDITIONS[1])
            expected = [
                100.0 * (after - before) / before
                for before, after in zip(
                    nominal["per_seed_aggregate"],
                    shifted["per_seed_aggregate"],
                    strict=True,
                )
            ]
            row = _row(relative, contender, CONDITIONS[1])
            assert float(row["aggregate"]) == pytest.approx(float(np.mean(expected)))
            assert list(row["per_seed_aggregate"]) == pytest.approx(expected)

    def test_it_is_not_the_ratio_of_the_two_means(self, tmp_path: Path) -> None:
        """A negative control that can fail. Pairing per seed and dividing the
        two column means are different numbers, and the second is the one a
        reader would get from the absolute table with a calculator.
        """
        _, absolute, relative = self._tables(tmp_path)
        differences = []
        for contender in set(relative["contender"]):
            nominal = _row(absolute, contender, CONDITIONS[0])
            shifted = _row(absolute, contender, CONDITIONS[1])
            unpaired = (
                100.0
                * (float(shifted["aggregate"]) - float(nominal["aggregate"]))
                / float(nominal["aggregate"])
            )
            paired = float(_row(relative, contender, CONDITIONS[1])["aggregate"])
            differences.append(abs(paired - unpaired))
        assert max(differences) > 0.0, (
            "the fixture's seeds must disagree, or this test cannot tell the "
            "paired form from the unpaired one"
        )

    def test_the_spread_is_of_the_per_seed_degradations(self, tmp_path: Path) -> None:
        _, _, relative = self._tables(tmp_path)
        for _, row in relative.iterrows():
            assert float(row["across_seed_spread"]) == pytest.approx(
                float(np.std(np.asarray(row["per_seed_aggregate"]), ddof=1))
            )

    def test_the_unit_is_declared_in_the_sidecar(self, tmp_path: Path) -> None:
        study = _shift_study(tmp_path / "doc", tmp_path / "store")
        sidecar = cost_by_category(
            _context(study, tmp_path / "store", self.RELATIVE)
        ).sidecar

        assert sidecar["relative_to"] == CONDITIONS[0]
        assert sidecar["quantity_unit"] == "percent"

    def test_an_unknown_reference_is_refused_naming_the_categories(
        self, tmp_path: Path
    ) -> None:
        study = _shift_study(tmp_path / "doc", tmp_path / "store")
        spec = AnalysisSpec(
            id="degradation",
            kind="cost_by_category",
            config={"relative_to": "no such condition"},
        )
        with pytest.raises(SpecificationError, match=CONDITIONS[0]):
            cost_by_category(_context(study, tmp_path / "store", spec))

    def test_a_contender_with_no_category_is_refused_rather_than_dropped(
        self, tmp_path: Path
    ) -> None:
        """Its degradation is undefined, and a figure quietly missing the
        contender its caption is about is the plausible-and-wrong output this
        architecture exists to prevent."""
        study = _shift_study(tmp_path / "doc", tmp_path / "store")
        narrowed = dataclasses.replace(
            study,
            sweep=(dataclasses.replace(study.sweep[0], applies_to=("unfolded_a",)),),
        )
        with pytest.raises(SpecificationError, match="baseline"):
            cost_by_category(_context(narrowed, tmp_path / "store", self.RELATIVE))

    def test_without_the_declaration_the_table_is_absolute(
        self, tmp_path: Path
    ) -> None:
        """The negative control: the transform happens only when asked for."""
        _, absolute, _ = self._tables(tmp_path)

        assert set(absolute["axis_label"]) == set(CONDITIONS)
        assert (absolute["aggregate"] > 1.0).all()


# --------------------------------------------------------------------------
# Three properties no end-to-end assertion can reach, each found by a mutant
# --------------------------------------------------------------------------


class TestPropertiesTheGrammarCannotProduce:
    """Unit tests, and the reason each one is a unit test is the point.

    A mutant that survives an end-to-end suite is a question, and for these
    three the answer was not "tune the test": it was that the case cannot be
    reached through a study document at all, or that a *second* rule repaired
    it before the assertion looked. Both are reasons to check the function
    rather than the figure.
    """

    def test_two_specs_differing_only_in_provenance_are_ONE_category(self) -> None:
        """Why `axis_key` uses the `ProblemID` and not `str`.

        **The obvious justification was measured and refuted.** `str` was
        expected to collide two *different* plants through NumPy's array
        summarisation; it does not, even at n = 50 (2 500 elements) and even
        under `set_printoptions(threshold=4)`. The real difference runs the
        other way: `provenance` is excluded from `problem_id` **by
        construction** and included in the repr, so two records of one plant
        are one category by identity and two by text.
        """
        source = _problem_spec()
        recorded = dataclasses.replace(
            source,
            provenance=GeneratorProvenance(generator="author_nominal", params={"n": 4}),
        )

        assert str(source) != str(recorded), "the fixture must differ as text"
        assert source.problem_id == recorded.problem_id
        assert axis_key(source) == axis_key(recorded)

    def test_an_unpaired_seed_is_refused(self) -> None:
        """Unreachable through a document, so checked on the function.

        Both categories of one contender come from the same `training.seeds`,
        so `grouped` can never hand this a group whose seeds disagree — which
        is why a mutant deleting the guard survived the whole suite. The guard
        stays because the pairing is the claim: an unpaired difference would
        compare one seed's shift with another seed's nominal, and nothing
        downstream could detect it.
        """
        row = {
            "contender": "unfolded_a",
            "axis_label": "rotated",
            "seeds": [0, 7],
            "per_seed_aggregate": [11.0, 12.0],
        }
        spec = _declared_for(aggregate="mean")
        with pytest.raises(SpecificationError, match="seed 7"):
            _one_degradation(row, {0: 10.0}, "nominal", spec)

    def test_a_zero_reference_is_refused(self) -> None:
        spec = _declared_for(aggregate="mean")
        row = {
            "contender": "unfolded_a",
            "axis_label": "rotated",
            "seeds": [0],
            "per_seed_aggregate": [11.0],
        }
        with pytest.raises(SpecificationError, match="exactly zero"):
            _one_degradation(row, {0: 0.0}, "nominal", spec)


def _problem_spec() -> ProblemSpec:
    """One frozen plant, loaded from a file the producer fixture writes."""
    directory = Path(tempfile.mkdtemp())
    tp._problem_file(directory, "problem.npz", seed=0)
    return ProblemSpec.load(directory / "problem.npz")


def _declared_for(*, aggregate: str) -> Declared:
    return Declared(
        axis_path="evaluation.problem",
        quantity="trajectory_cost",
        aggregate=aggregate,
        dispersion="std",
        level=0.95,
        resamples=100,
    )


class TestAZippedCompanionAxis:
    """Annex 01 §2.5's blind/aware zip, as acceptance (the D24 follow-up).

    A zipped group is ONE effective axis. Before the composite key, the two
    rehost modes of one plant landed in one group — keyed by the plant's
    `ProblemID` alone — and were refused as repeated seeds, with a blame
    message naming the zipped companion as though it were an independent
    pooling axis. The category is now the whole position, so one plant scored
    blind and aware is two categories, and the pooling refusal stays armed for
    a genuinely independent (product-composed) second axis — which the pinned
    refusal test above keeps honest.
    """

    def _zipped(self, directory: Path, *, rehost: tuple[str, ...]) -> Any:
        study = _study(directory)
        tp._problem_file(directory, "rotated.npz", seed=1)
        nominal = ProblemSpec.load(directory / "problem.npz")
        rotated = ProblemSpec.load(directory / "rotated.npz")
        return dataclasses.replace(
            study,
            sweep=(
                SweepAxis(
                    path="evaluation.problem",
                    values=(nominal, rotated, rotated),
                    labels=("nominal", "shifted blind", "shifted aware"),
                ),
                SweepAxis(
                    path="evaluation.rehost",
                    values=rehost,
                    compose=Composition.ZIP,
                ),
            ),
        )

    def test_the_two_modes_of_one_plant_are_two_categories(
        self, tmp_path: Path
    ) -> None:
        study = self._zipped(tmp_path / "doc", rehost=("blind", "blind", "aware"))
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        table = cost_by_category(
            _context(
                study,
                tmp_path / "store",
                AnalysisSpec(
                    id="by_plant",
                    kind="cost_by_category",
                    config={"axis": "evaluation.problem"},
                ),
            )
        ).table

        # One row per contender per POSITION: 2 contenders x 3 positions --
        # the very shape the value-keyed grouping could not produce.
        assert len(table) == 6
        assert set(table["axis_label"]) == {"nominal", "shifted blind", "shifted aware"}
        for contender in ("unfolded_a", "baseline"):
            assert _row(table, contender, "nominal")["axis_value"] == 0.0
            assert _row(table, contender, "shifted blind")["axis_value"] == 1.0
            assert _row(table, contender, "shifted aware")["axis_value"] == 2.0

        # The two modes carry genuinely different numbers for the analytic
        # contender -- blind keeps the nominal gains, aware re-solves on the
        # shifted plant -- so the categories are not two labels on one column.
        blind = _row(table, "baseline", "shifted blind")["aggregate"]
        aware = _row(table, "baseline", "shifted aware")["aggregate"]
        assert blind != aware

    def test_degradation_pairs_across_the_zipped_group(self, tmp_path: Path) -> None:
        study = self._zipped(tmp_path / "doc", rehost=("blind", "blind", "aware"))
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        table = cost_by_category(
            _context(
                study,
                tmp_path / "store",
                AnalysisSpec(
                    id="degradation",
                    kind="cost_by_category",
                    config={
                        "axis": "evaluation.problem",
                        "relative_to": "nominal",
                    },
                ),
            )
        ).table

        # The nominal position is the denominator and is dropped; the two
        # shifted positions survive per contender, paired per seed -- the
        # analytic contender's degradations therefore carry a spread of
        # exactly 0.0 in both modes.
        assert len(table) == 4
        assert set(table["axis_label"]) == {"shifted blind", "shifted aware"}
        for category in ("shifted blind", "shifted aware"):
            assert _row(table, "baseline", category)["across_seed_spread"] == 0.0

    def test_a_contender_the_zipped_axis_skips_still_lands_in_its_category(
        self, tmp_path: Path
    ) -> None:
        """The merged Figure 2's own failure (2026-08-10): an `applies_to` on
        one member of the zipped group writes nothing into the excluded
        contender's points, so its key holds `None` in that component and the
        declared-tuple lookup raised KeyError after a four-hour run. The hole
        must fill by subset match — here the analytic contender's
        `(shifted, blind, None)` belongs to the one position agreeing on what
        it does carry."""
        directory = tmp_path / "doc"
        study = self._zipped(directory, rehost=("blind", "blind", "aware"))
        rotated = ProblemSpec.load(directory / "rotated.npz")
        study = dataclasses.replace(
            study,
            sweep=(
                *study.sweep,
                SweepAxis(
                    path="training.problem",
                    values=(None, None, rotated),
                    applies_to=("unfolded_a",),
                    compose=Composition.ZIP,
                ),
            ),
        )
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        table = cost_by_category(
            _context(
                study,
                tmp_path / "store",
                AnalysisSpec(
                    id="by_plant",
                    kind="cost_by_category",
                    config={"axis": "evaluation.problem"},
                ),
            )
        ).table
        # Both contenders land one row in EVERY position — including the
        # analytic one at the position whose world axis skipped it.
        assert len(table) == 6
        for contender in ("unfolded_a", "baseline"):
            assert _row(table, contender, "shifted aware")["axis_value"] == 2.0

    def test_a_hole_that_fills_ambiguously_is_refused(self, tmp_path: Path) -> None:
        """Two positions that differ ONLY on the skipped axis would resolve
        the hole by declaration order; refused by name instead."""
        directory = tmp_path / "doc"
        study = self._zipped(directory, rehost=("blind", "blind", "blind"))
        tp._problem_file(directory, "second_world.npz", seed=2)
        rotated = ProblemSpec.load(directory / "rotated.npz")
        second = ProblemSpec.load(directory / "second_world.npz")
        # Positions 2 and 3 now differ only in the world -- two DISTINCT
        # worlds behind one (plant, rehost) face -- which the analytic
        # contender does not carry, so its hole matches both.
        study = dataclasses.replace(
            study,
            sweep=(
                *study.sweep,
                SweepAxis(
                    path="training.problem",
                    values=(None, rotated, second),
                    applies_to=("unfolded_a",),
                    compose=Composition.ZIP,
                ),
            ),
        )
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        with pytest.raises(SpecificationError, match="unambiguous"):
            cost_by_category(
                _context(
                    study,
                    tmp_path / "store",
                    AnalysisSpec(
                        id="by_plant",
                        kind="cost_by_category",
                        config={"axis": "evaluation.problem"},
                    ),
                )
            )

    def test_a_duplicate_position_is_refused_by_name(self, tmp_path: Path) -> None:
        """Two identical positions are one category declared twice. The old
        value-keyed maps were last-wins on repeats; the refusal replaces the
        silent relabelling."""
        study = self._zipped(tmp_path / "doc", rehost=("blind", "aware", "aware"))
        with pytest.raises(SpecificationError, match="more than once"):
            cost_by_category(
                _context(
                    study,
                    tmp_path / "store",
                    AnalysisSpec(
                        id="by_plant",
                        kind="cost_by_category",
                        config={"axis": "evaluation.problem"},
                    ),
                )
            )
