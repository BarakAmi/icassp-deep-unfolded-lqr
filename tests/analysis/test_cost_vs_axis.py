"""Acceptance tests for `cost_vs_axis` — slice Phase B, step B2b.

Written before the implementation. The checkpoint class is **independent
recomputation**: the aggregate and the interval for one contender at one depth
are recomputed by hand from the stored per-trajectory costs and compared with
what the analysis emits. Before step B0 that comparison had one operand — a
`smoke` measurement held a single row — which is why the payload came first.

Three properties beyond the arithmetic carry the phase:

* **`require_analysable_measurement` gets its call site here.** E2 built the
  guard and recorded that Stage 4 owed it a caller. The negative control is
  the discharge: a measurement produced under a subsetted `smoke` run must be
  **refused**, and one produced without subsetting must not — a guard verified
  only in the passing direction is a guard nobody has tested.

* **A depth-invariant contender has no axis value.** Its row carries a null
  one, and `role` is what a figure draws it by. Placing it at some depth it
  never had would turn a reference line into a curve.

* **One training seed has no across-seed interval.** Not a narrow one: none.
  The analysis emits null and says why in the sidecar, and the within-seed
  dispersion travels under a different name so no figure can render it as an
  across-seed band by accident.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mbl.analysis import run_analyses
from mbl.analysis.cost_vs_axis import cost_vs_axis
from mbl.analysis.registry import AnalysisContext, registered_analyses, resolve_analysis
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.runner.producer import run_study
from mbl.spec.analysis import AnalysisSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study
from mbl.spec.study import SweepAxis
from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE
from mbl.store.content_store import MeasurementStore
from mbl.store.study_artifacts import StudyArtifactStore

from ..runner.test_producer import _write

ANALYSIS = AnalysisSpec(id="cost_by_depth", kind="cost_vs_axis")

#: The fixture's shape, named so a count below never agrees with a stale
#: literal: two depths for the swept contender plus one flat baseline.
EXPECTED_ROWS = 3


def _study(directory: Path, /, **overrides: Any):
    return load_study(
        _write(directory, **overrides), bindings=DEFAULT_SPEC_BINDINGS
    ).study


def _run(directory: Path, store: Path, /, **overrides: Any):
    study = _study(directory, **overrides)
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    return study


def _context(study: Any, store: Path, spec: AnalysisSpec = ANALYSIS) -> AnalysisContext:
    return AnalysisContext(
        study=study,
        study_id=str(study.study_id),
        spec=spec,
        measurements=MeasurementStore(store),
    )


def _samples(measurements: MeasurementStore, point: Any) -> pd.Series:
    """One point's stored per-trajectory costs, read independently.

    A named helper rather than a chain in each test: `MeasurementRecord
    .samples` is `DataFrame | None`, and narrowing it once with a real
    assertion beats silencing the type at every call site.
    """
    record = measurements.get(str(point.measurement_id))
    assert record.samples is not None, f"{point.measurement_id} stored no samples"
    return record.samples["trajectory_cost"]


def _with(study: Any, **overrides: Any):
    return dataclasses.replace(study, **overrides)


def _one_seed(study: Any):
    """The same study at one training seed.

    Its points are a subset of the two-seed study's, so the store already
    holds every measurement -- which keeps this about the interval rather
    than about running the fixture twice.
    """
    return _with(study, training=dataclasses.replace(study.training, seeds=(0,)))


class TestTheRegistry:
    def test_the_kind_resolves_by_its_annex_name(self) -> None:
        assert resolve_analysis("cost_vs_axis") is cost_vs_axis
        assert "cost_vs_axis" in registered_analyses()

    def test_an_unregistered_kind_names_what_is_available(self) -> None:
        with pytest.raises(SpecificationError, match="cost_vs_axis"):
            resolve_analysis("no_such_analysis")


class TestTheTidyTable:
    def test_one_row_per_contender_per_axis_point(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table

        assert len(table) == EXPECTED_ROWS
        assert set(table["contender"]) == {"unfolded_a", "baseline"}

    def test_a_swept_contender_carries_its_axis_value(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table

        swept = table[table["contender"] == "unfolded_a"]
        assert sorted(swept["axis_value"]) == [2.0, 4.0]
        assert set(swept["axis_path"]) == {"contenders.*.config.num_iterations"}

    def test_a_depth_invariant_contender_carries_none(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table

        flat = table[table["contender"] == "baseline"]
        assert len(flat) == 1
        assert bool(np.isnan(flat["axis_value"].iloc[0]))
        assert flat["axis_path"].iloc[0] == ""
        assert flat["role"].iloc[0] == "baseline"

    def test_every_row_states_its_counts(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table

        # The fixture declares two training seeds and two evaluation batches
        # of sixteen; §A.4 requires both counts to be reported.
        assert set(table["n_seeds"]) == {2}
        assert set(table["n_trajectories"]) == {2 * 2 * 16}


class TestIndependentRecomputation:
    def test_the_aggregate_is_the_mean_of_the_per_seed_means(
        self, tmp_path: Path
    ) -> None:
        """The checkpoint. Recomputed by hand, in §A.3's declared order.

        First over evaluation trajectories within a seed, then over training
        seeds -- and computed here from the STORED samples rather than from
        any intermediate the analysis produced, so the two share no code.
        """
        store = tmp_path / "store"
        study = _run(tmp_path / "doc", store)
        table = cost_vs_axis(_context(study, store)).table
        measurements = MeasurementStore(store)

        for _, row in table.iterrows():
            points = [
                point
                for point in study.materialise()
                if point.contender.resolved_label == row["contender"]
                and (
                    np.isnan(row["axis_value"])
                    or point.contender.config.get("num_iterations")
                    == int(row["axis_value"])
                )
            ]
            per_seed = [float(_samples(measurements, point).mean()) for point in points]
            assert row["aggregate"] == pytest.approx(
                float(np.mean(per_seed)), rel=1e-12
            )

    def test_the_within_seed_spread_is_the_mean_of_the_per_seed_stds(
        self, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        study = _run(tmp_path / "doc", store)
        table = cost_vs_axis(_context(study, store)).table
        measurements = MeasurementStore(store)

        row = table[table["contender"] == "baseline"].iloc[0]
        points = [
            point
            for point in study.materialise()
            if point.contender.resolved_label == "baseline"
        ]
        spreads = [float(_samples(measurements, point).std(ddof=1)) for point in points]
        assert row["within_seed_spread"] == pytest.approx(
            float(np.mean(spreads)), rel=1e-12
        )

    def test_the_aggregate_is_not_the_pooled_mean(self, tmp_path: Path) -> None:
        """The order is fixed and it must be *observable* that it is.

        Pooling every trajectory of every seed into one mean is the obvious
        wrong implementation, and with equal batch sizes it gives the same
        answer -- so this asserts the ORDER by comparing against a case where
        the two differ, built by truncating one seed's samples.
        """
        seeds = [np.array([1.0, 3.0]), np.array([10.0])]
        pooled = float(np.mean(np.concatenate(seeds)))
        ordered = float(np.mean([float(batch.mean()) for batch in seeds]))

        assert ordered == pytest.approx(6.0)
        assert pooled == pytest.approx(14.0 / 3.0)
        assert ordered != pytest.approx(pooled)


class TestTheIntervalIsHonest:
    def test_two_seeds_produce_an_across_seed_interval(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table

        assert table["interval_low"].notna().all()
        assert (table["interval_low"] <= table["aggregate"]).all()
        assert (table["aggregate"] <= table["interval_high"]).all()

    def test_one_seed_produces_no_interval_at_all(self, tmp_path: Path) -> None:
        """Not a narrow interval. None.

        Substituting the across-trajectory interval -- always available,
        always narrower -- would be undetectable once it has been drawn as a
        band, which is why it is refused rather than defaulted.
        """
        store = tmp_path / "store"
        study = _one_seed(_run(tmp_path / "doc", store))
        output = cost_vs_axis(_context(study, store))

        assert output.table["interval_low"].isna().all()
        assert output.table["interval_high"].isna().all()
        assert output.sidecar["interval"]["kind"] is None
        assert "does not exist" in output.sidecar["interval"]["reason"]

    def test_one_seed_still_reports_the_within_seed_spread(
        self, tmp_path: Path
    ) -> None:
        # Under a different name, so no figure can render it as a band.
        store = tmp_path / "store"
        study = _one_seed(_run(tmp_path / "doc", store))
        output = cost_vs_axis(_context(study, store))

        assert (output.table["within_seed_spread"] > 0.0).all()
        assert output.sidecar["within_seed_spread_kind"] == "std"
        # Asserted structurally rather than as a substring: the reason text
        # names the column on purpose, so `"within_seed_spread" not in str(...)`
        # fails for the wrong reason. What must hold is that the interval block
        # carries no dispersion VALUE that a renderer could mistake for one.
        assert set(output.sidecar["interval"]) == {
            "kind",
            "level",
            "resamples",
            "over",
            "reason",
        }


class TestTheSidecarStatesWhatSectionA4Requires:
    def test_it_reports_the_aggregation_order(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        sidecar = cost_vs_axis(_context(study, tmp_path / "store")).sidecar

        assert sidecar["aggregation_order"] == [
            "evaluation_trajectories",
            "training_seeds",
        ]
        assert sidecar["interval"]["over"] == "training_seeds"

    def test_it_reports_the_aggregate_and_the_counts(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        sidecar = cost_vs_axis(_context(study, tmp_path / "store")).sidecar

        assert sidecar["aggregate"] == "mean"
        assert sidecar["n_training_seeds"] == [2]
        assert sidecar["n_evaluation_trajectories"] == EXPECTED_ROWS * 2 * 2 * 16
        assert sidecar["study_id"] == str(study.study_id)

    def test_a_median_pairs_with_an_interquartile_range(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        spec = AnalysisSpec(id="a", kind="cost_vs_axis", config={"aggregate": "median"})
        output = cost_vs_axis(_context(study, tmp_path / "store", spec))

        assert output.sidecar["aggregate"] == "median"
        assert output.sidecar["within_seed_spread_kind"] == "iqr"

    def test_a_declared_median_is_actually_computed(self, tmp_path: Path) -> None:
        """Mutation testing said the sidecar was the only thing being checked.

        Asserting that the declaration is echoed back is not asserting that it
        took effect -- an implementation that always computed a mean passed
        the test above. Trajectory cost is right-skewed, so the two differ,
        and the median is recomputed here from the stored samples.
        """
        store = tmp_path / "store"
        study = _run(tmp_path / "doc", store)
        spec = AnalysisSpec(id="a", kind="cost_vs_axis", config={"aggregate": "median"})
        table = cost_vs_axis(_context(study, store, spec)).table
        measurements = MeasurementStore(store)

        row = table[table["contender"] == "baseline"].iloc[0]
        points = [
            point
            for point in study.materialise()
            if point.contender.resolved_label == "baseline"
        ]
        per_seed = [float(_samples(measurements, p).median()) for p in points]
        assert row["aggregate"] == pytest.approx(float(np.median(per_seed)), rel=1e-12)

        means = [float(_samples(measurements, p).mean()) for p in points]
        assert float(np.median(per_seed)) != pytest.approx(
            float(np.mean(means)), rel=1e-9
        ), "the fixture's cost is symmetric enough that this test cannot fail"

    def test_an_unknown_aggregate_is_refused(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        spec = AnalysisSpec(id="a", kind="cost_vs_axis", config={"aggregate": "mode"})
        with pytest.raises(SpecificationError, match="mode"):
            cost_vs_axis(_context(study, tmp_path / "store", spec))


class TestTheAxisIsNamedRatherThanGuessed:
    def test_one_swept_axis_may_be_defaulted(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        assert len(cost_vs_axis(_context(study, tmp_path / "store")).table)

    def test_an_axis_the_study_does_not_sweep_is_refused(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        spec = AnalysisSpec(
            id="a", kind="cost_vs_axis", config={"axis": "training.seeds"}
        )
        with pytest.raises(SpecificationError, match="training.seeds"):
            cost_vs_axis(_context(study, tmp_path / "store", spec))

    def test_a_study_with_no_sweep_cannot_default_an_axis(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError, match="cannot be defaulted"):
            cost_vs_axis(_context(_with(study, sweep=()), tmp_path / "store"))

    def test_two_swept_axes_cannot_default_either(self, tmp_path: Path) -> None:
        """The case that matters, and the one mutation testing found missing.

        Zero axes cannot be defaulted for the trivial reason that there is
        nothing to pick. TWO is where picking is possible and wrong: taking
        the first would pick the x axis by declaration order, and the table
        would silently be against a different variable than the author meant.
        """
        study = _run(tmp_path / "doc", tmp_path / "store")
        second = SweepAxis(path="evaluation.protocol.n_batches", values=(1, 2))
        with pytest.raises(SpecificationError, match="2 swept axes"):
            cost_vs_axis(
                _context(_with(study, sweep=(*study.sweep, second)), tmp_path / "store")
            )


class TestTheNegativeControlTheGuardExistsFor:
    def test_a_subsetted_measurement_is_refused(self, tmp_path: Path) -> None:
        """The discharge of E2's recorded obligation.

        Four depths, so `smoke`'s cap of two actually truncates the axis --
        with a two-value axis nothing is dropped, the run is correctly NOT
        stamped, and this test would assert the refusal of a run that was
        never subsetted.
        """
        document = load_study(
            _write(tmp_path / "doc", depths=[1, 2, 4, 8]),
            bindings=DEFAULT_SPEC_BINDINGS,
        )
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "smoke")
        assert resolved.axis_subset, "smoke truncated nothing; the stamp is absent"
        store = tmp_path / "store"
        run_study(
            resolved.study,
            store=store,
            bindings=DEFAULT_SPEC_BINDINGS,
            provenance=resolved.provenance,
        )

        with pytest.raises(SpecificationError, match="axis_subset"):
            cost_vs_axis(_context(resolved.study, store))

    def test_an_unsubsetted_measurement_is_not_refused(self, tmp_path: Path) -> None:
        """The other half. A gate verified only in the failing direction is
        a gate that might refuse everything."""
        document = load_study(_write(tmp_path / "doc"), bindings=DEFAULT_SPEC_BINDINGS)
        resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "standard")
        assert not resolved.axis_subset
        store = tmp_path / "store"
        run_study(
            resolved.study,
            store=store,
            bindings=DEFAULT_SPEC_BINDINGS,
            provenance=resolved.provenance,
        )

        assert len(cost_vs_axis(_context(resolved.study, store)).table)


class TestItReadsTheStoreAndNothingElse:
    def test_a_missing_measurement_is_refused_by_name(self, tmp_path: Path) -> None:
        # An analysis reads what was produced and never produces it; silently
        # dropping the point would emit a table that is short by one row and
        # says nothing about it.
        study = _study(tmp_path / "doc")
        with pytest.raises(SpecificationError, match="no measurement"):
            cost_vs_axis(_context(study, tmp_path / "store"))

    def test_it_runs_with_the_model_directory_removed(self, tmp_path: Path) -> None:
        """The property the identity split exists for, asserted destructively.

        An analysis that reached for weights could not run once the
        checkpoints were gone -- so they are removed, and it still runs.
        """
        import shutil

        store = tmp_path / "store"
        study = _run(tmp_path / "doc", store)
        shutil.rmtree(store / "models")

        assert len(cost_vs_axis(_context(study, store)).table) == EXPECTED_ROWS


class TestTheRunnerWritesIntoTheStore:
    def test_it_writes_the_parquet_and_the_sidecar(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        study = _with(_run(tmp_path / "doc", store), analyses=(ANALYSIS,))

        outcomes = run_analyses(study, store=store)

        assert [o.analysis_id for o in outcomes] == ["cost_by_depth"]
        assert outcomes[0].rows == EXPECTED_ROWS
        stored = StudyArtifactStore(store).get_analysis(
            str(study.study_id), "cost_by_depth"
        )
        pd.testing.assert_frame_equal(
            stored.table, cost_vs_axis(_context(study, store)).table
        )
        assert stored.sidecar["kind"] == "cost_vs_axis"

    def test_the_sidecar_on_disk_is_plain_json(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        study = _with(_run(tmp_path / "doc", store), analyses=(ANALYSIS,))
        run_analyses(study, store=store)

        path = (
            store / "studies" / str(study.study_id) / "analyses" / "cost_by_depth.json"
        )
        assert json.loads(path.read_text(encoding="utf-8"))["aggregate"] == "mean"

    def test_a_study_declaring_nothing_runs_nothing(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        study = _run(tmp_path / "doc", store)
        assert run_analyses(study, store=store) == ()

    def test_selecting_an_undeclared_analysis_is_refused(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        study = _with(_run(tmp_path / "doc", store), analyses=(ANALYSIS,))
        with pytest.raises(SpecificationError, match="typo"):
            run_analyses(study, store=store, only=["typo"])

    def test_re_running_replaces_rather_than_refusing(self, tmp_path: Path) -> None:
        # The property that makes an analysis re-runnable: measurements do not
        # change, the declaration does.
        store = tmp_path / "store"
        study = _with(_run(tmp_path / "doc", store), analyses=(ANALYSIS,))
        run_analyses(study, store=store)

        median = AnalysisSpec(
            id="cost_by_depth", kind="cost_vs_axis", config={"aggregate": "median"}
        )
        run_analyses(_with(study, analyses=(median,)), store=store)

        stored = StudyArtifactStore(store).get_analysis(
            str(study.study_id), "cost_by_depth"
        )
        assert stored.sidecar["aggregate"] == "median"


class TestTheSpreadTheFigureDraws:
    """Annex 03 §A.3.2 — the third dispersion, and the one drawn.

    Mutation testing put this class here. The renderer's suite covered rule 2
    (absent and zero are different facts) against hand-built frames, so every
    mutation of the ANALYSIS half survived: emitting no column at all, emitting
    `0.0` where §A.3.2 requires `NaN`, dropping the audit trail, and leaving
    the sidecar silent about which dispersion the bars are. Four mutants, one
    gap — the half that produces the number was never asked.
    """

    def test_it_is_the_dispersion_of_the_per_seed_aggregates(
        self, tmp_path: Path
    ) -> None:
        """Independent recomputation, from the STORED samples.

        `within_seed_spread` and `across_seed_spread` are two reductions of
        one set of numbers and a renderer cannot tell them apart, so the check
        that matters is which one the column actually holds.
        """
        study = _run(tmp_path / "doc", tmp_path / "store")
        context = _context(study, tmp_path / "store")
        table = cost_vs_axis(context).table

        measurements = MeasurementStore(tmp_path / "store")
        for _, row in table.iterrows():
            per_seed = [
                float(np.mean(_samples(measurements, point)))
                for point in study.materialise()
                if point.contender.resolved_label == row["contender"]
                and (
                    np.isnan(row["axis_value"])
                    or float(
                        point.axis_values[row["axis_path"]]
                        if row["axis_path"]
                        else np.nan
                    )
                    == row["axis_value"]
                )
            ]
            expected = float(np.std(np.asarray(per_seed), ddof=1))
            assert row["across_seed_spread"] == pytest.approx(expected, rel=1e-12), row[
                "contender"
            ]

    def test_it_is_not_the_within_seed_spread(self, tmp_path: Path) -> None:
        # The two columns must not share a name AND must not share a value:
        # a within-seed dispersion drawn as an across-seed bar overstates the
        # retraining variability by the trajectory count's square root.
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table
        assert not np.allclose(table["across_seed_spread"], table["within_seed_spread"])

    def test_one_seed_has_no_across_seed_spread_at_all(self, tmp_path: Path) -> None:
        """`NaN`, not `0.0` — §A.3.2's rule 2 at the source.

        `_spread` answers `0.0` for a single value, which is right for "how
        spread are these trajectories" and wrong for "how spread are these
        seeds". A zero here would claim a measured absence of training
        variance where no such measurement was made, and the figure would
        then be correct to draw nothing for the wrong reason.
        """
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(_one_seed(study), tmp_path / "store")).table

        assert set(table["n_seeds"]) == {1}
        assert bool(table["across_seed_spread"].isna().all())
        # And the two facts stay distinguishable: the within-seed dispersion
        # is still a real number on the same rows.
        assert bool(table["within_seed_spread"].notna().all())

    def test_every_row_carries_the_reduction_behind_its_bar(
        self, tmp_path: Path
    ) -> None:
        """§A.3.2: `seeds` and `per_seed_aggregate` travel in the frame.

        A table reporting the per-seed values behind an error bar must read
        them from `<id>.data.parquet`; obtaining them in the emitter would
        reach past the frozen frame and reopen the drift §B.1.2 closes.
        """
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table

        for _, row in table.iterrows():
            seeds = list(row["seeds"])
            per_seed = np.asarray(row["per_seed_aggregate"], dtype=np.float64)
            assert len(seeds) == int(row["n_seeds"])
            assert len(per_seed) == int(row["n_seeds"])
            # The row's own headline numbers must be derivable from them, or
            # the audit trail is beside the result rather than behind it.
            assert float(np.mean(per_seed)) == pytest.approx(
                row["aggregate"], rel=1e-12
            )
            assert float(np.std(per_seed, ddof=1)) == pytest.approx(
                row["across_seed_spread"], rel=1e-12
            )

    def test_a_median_pairs_the_spread_with_an_interquartile_range(
        self, tmp_path: Path
    ) -> None:
        # A standard deviation about a median is a statistic of nothing, and
        # asserting the sidecar alone would not catch a spread that stayed a
        # standard deviation while the sidecar said `iqr`.
        study = _run(tmp_path / "doc", tmp_path / "store")
        spec = AnalysisSpec(id="a", kind="cost_vs_axis", config={"aggregate": "median"})
        output = cost_vs_axis(_context(study, tmp_path / "store", spec))

        assert output.sidecar["across_seed_spread_kind"] == "iqr"
        assert output.sidecar["across_seed_spread_over"] == "training_seeds"
        row = output.table.iloc[0]
        per_seed = np.asarray(row["per_seed_aggregate"], dtype=np.float64)
        quartiles = np.percentile(per_seed, [25.0, 75.0])
        assert row["across_seed_spread"] == pytest.approx(
            float(quartiles[1] - quartiles[0]), rel=1e-12
        )

    def test_the_sidecar_names_the_drawn_quantity(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store")
        sidecar = cost_vs_axis(_context(study, tmp_path / "store")).sidecar

        assert sidecar["across_seed_spread_kind"] == "std"
        assert sidecar["across_seed_spread_over"] == "training_seeds"
        # And it stays distinct from the interval's own declarations, which
        # §A.3.2 says must not be confused with it.
        assert sidecar["interval"]["kind"] == "bootstrap_bca"


class TestASecondSweptAxisIsRefusedRatherThanPooled:
    """A group must be replicates, and nothing else (§A.3).

    `_grouped` keys on `(contender, axis_value)` for the ONE declared axis, so
    on a two-axis study every group silently collects the points of the other
    axis as well. That is not a blurred mean — it **fabricates the error bar**.
    Measured before the guard, on a study with ONE training seed and a second
    swept axis: the groups came back `seeds = [0, 0]`, and `cost_vs_axis`
    reported `n_seeds = 2` with an `across_seed_spread` computed across two
    points that are not two seeds, under a sidecar declaring
    `across_seed_spread_over = "training_seeds"`.

    §A.3 names training seeds as the across-seed aggregation unit and §A.3.2
    makes that spread the quantity a figure DRAWS. A number manufactured from
    a second axis is therefore drawn as retraining variability, which is a
    claim nobody made and nothing downstream could detect — the frame, the
    figure and the table would all agree with each other and all be wrong.

    The detector is the one fact a replicate group cannot violate: its seeds
    are distinct.
    """

    def _two_axis(self, directory: Path, store: Path):
        """The fixture's study with one seed and a second swept axis."""
        study = _study(directory)
        study = _with(
            study,
            training=dataclasses.replace(study.training, seeds=(0,)),
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

    #: The axis must be NAMED to reach the pooling at all: `_axis_path`
    #: already refuses a two-axis study that defaults it ("`axis` cannot be
    #: defaulted; name the one this table is against"). So the defect lives
    #: precisely where an author has answered that question and reasonably
    #: believes they have said what they meant.
    NAMED = AnalysisSpec(
        id="cost_by_depth",
        kind="cost_vs_axis",
        config={"axis": "contenders.*.config.num_iterations"},
    )

    def test_defaulting_the_axis_is_already_refused(self, tmp_path: Path) -> None:
        # The guard that exists, asserted so the new one is not credited with
        # its work — and so the test below is known to reach further.
        study = self._two_axis(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError, match="cannot be defaulted"):
            cost_vs_axis(_context(study, tmp_path / "store"))

    def test_the_pooled_group_is_refused_by_name(self, tmp_path: Path) -> None:
        study = self._two_axis(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError) as raised:
            cost_vs_axis(_context(study, tmp_path / "store", self.NAMED))
        message = str(raised.value)
        # The message has to name what an author must edit: the contender, the
        # axis value whose group is not replicates, and the OTHER axis that is
        # being folded into it.
        assert "unfolded_a" in message
        assert "step_size_init" in message
        assert "seed" in message

    def test_the_guard_fires_before_any_number_is_produced(
        self, tmp_path: Path
    ) -> None:
        """A refusal after the aggregate is computed would still let a caller
        catch it and read a frame built from pooled groups."""
        study = self._two_axis(tmp_path / "doc", tmp_path / "store")
        with pytest.raises(SpecificationError):
            cost_vs_axis(_context(study, tmp_path / "store", self.NAMED)).table

    def test_a_single_axis_study_with_many_seeds_is_untouched(
        self, tmp_path: Path
    ) -> None:
        """The negative control, and the one that matters: the campaign's own
        shape is one axis over five seeds, whose groups are exactly
        replicates. A guard that refused this would refuse every real study."""
        study = _run(tmp_path / "doc", tmp_path / "store")
        table = cost_vs_axis(_context(study, tmp_path / "store")).table
        assert len(table) == EXPECTED_ROWS
        assert set(table["n_seeds"]) == {2}
        for _, row in table.iterrows():
            assert len(set(row["seeds"])) == len(row["seeds"])

    def test_one_seed_and_one_axis_is_still_fine(self, tmp_path: Path) -> None:
        # A one-seed group has one point, so distinctness is trivially true —
        # and the across-seed spread is `NaN`, not a fabricated number.
        study = _study(tmp_path / "doc")
        study = _with(study, training=dataclasses.replace(study.training, seeds=(0,)))
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        table = cost_vs_axis(_context(study, tmp_path / "store")).table
        assert set(table["n_seeds"]) == {1}
        assert bool(table["across_seed_spread"].isna().all())
