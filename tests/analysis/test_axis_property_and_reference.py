"""Acceptance for `axis_property` and `relative_to_contender` (Figure-3 plan
Phase 3; Annex 03 §A.6.1 as amended 2026-08-08).

`axis_property` makes a problem-valued axis numeric by projecting each
position's frozen plant onto an intrinsic property (`state_dim` first);
`relative_to_contender` makes one contender the per-position denominator,
paired per training seed. Both are refused when inert, in every direction the
annex names. Written before the implementation, in `test_cost_vs_axis`'s
idiom: real producer runs into a tmp store, no mocks.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mbl.analysis.cost_by_category import cost_by_category
from mbl.analysis.cost_vs_axis import cost_vs_axis
from mbl.analysis.registry import AnalysisContext
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.runner.producer import run_study
from mbl.spec.analysis import AnalysisSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.store.content_store import MeasurementStore

_producer = importlib.import_module("tests.runner.test_producer")


def _wider_plant(directory: Path, name: str, seed: int) -> Path:
    """Same horizon as the fixture's plant, two state dimensions wider — so a
    root sweep crosses shapes without touching the contender's horizon."""
    n = _producer.N + 2
    rng = np.random.default_rng(seed)
    spec = ProblemSpec(
        ProblemData(
            system={
                "A": rng.normal(size=(n, n)),
                "B": rng.normal(size=(n, _producer.M)),
            },
            cost={
                "Q": np.repeat(np.eye(n)[None], _producer.HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(_producer.M)[None], _producer.HORIZON, axis=0),
            },
            horizon=_producer.HORIZON,
            control_bound=0.5,
        )
    )
    spec.save(directory / name)
    return directory / name


def _document(directory: Path, *, second: str, applies_to: str = "") -> Path:
    """The producer fixture's study with its sweep replaced by a two-plant
    root `problem` axis."""
    directory.mkdir(parents=True, exist_ok=True)
    _producer._problem_file(directory, "problem.npz", seed=0)
    if second == "wider.npz":
        _wider_plant(directory, second, seed=1)
    else:
        _producer._problem_file(directory, second, seed=1)
    text = _producer.STUDY.format(**_producer.DEFAULTS)
    head, marker, _ = text.partition("[[sweep]]")
    assert marker, "the producer fixture's sweep block moved"
    sweep = (
        "[[sweep]]\n"
        'path = "problem"\n'
        f'values = ["problem.npz", "{second}"]\n'
        'labels = ["first", "second"]\n'
    )
    if applies_to:
        sweep += f"applies_to = [{applies_to}]\n"
    (directory / "study.toml").write_text(f"{head}{sweep}")
    return directory / "study.toml"


def _run(directory: Path, store: Path, **kwargs: Any) -> Any:
    study = load_study(
        _document(directory, **kwargs), bindings=DEFAULT_SPEC_BINDINGS
    ).study
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    return study


def _spec(**config: Any) -> AnalysisSpec:
    return AnalysisSpec(id="scaling", kind="cost_vs_axis", config=config)


def _context(study: Any, store: Path, spec: AnalysisSpec) -> AnalysisContext:
    return AnalysisContext(
        study=study,
        study_id=str(study.study_id),
        spec=spec,
        measurements=MeasurementStore(store),
    )


class TestAxisProperty:
    def test_the_projected_value_is_the_plants_state_dim(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        output = cost_vs_axis(
            _context(study, tmp_path / "store", _spec(axis_property="state_dim"))
        )
        values = set(output.table["axis_value"].dropna())
        assert values == {float(_producer.N), float(_producer.N + 2)}

    def test_absent_on_a_problem_axis_is_refused_naming_the_key(
        self, tmp_path: Path
    ) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        with pytest.raises(SpecificationError, match="axis_property"):
            cost_vs_axis(_context(study, tmp_path / "store", _spec()))

    def test_an_unknown_property_is_refused_naming_the_set(
        self, tmp_path: Path
    ) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        with pytest.raises(SpecificationError, match="state_dim"):
            cost_vs_axis(
                _context(study, tmp_path / "store", _spec(axis_property="volume"))
            )

    def test_declared_on_a_scalar_axis_is_refused_as_inert(
        self, tmp_path: Path
    ) -> None:
        """The fixture's own depth axis is numeric already; a projection
        declared over it reads nothing and is the inert-declaration defect."""
        directory = tmp_path / "doc"
        directory.mkdir(parents=True, exist_ok=True)
        _producer._problem_file(directory, "problem.npz", seed=0)
        (directory / "study.toml").write_text(
            _producer.STUDY.format(**_producer.DEFAULTS)
        )
        study = load_study(
            directory / "study.toml", bindings=DEFAULT_SPEC_BINDINGS
        ).study
        run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)
        with pytest.raises(SpecificationError, match="axis_property"):
            cost_vs_axis(
                _context(study, tmp_path / "store", _spec(axis_property="state_dim"))
            )

    def test_two_positions_projecting_onto_one_value_are_refused(
        self, tmp_path: Path
    ) -> None:
        """Two distinct plants with one state dimension would put two rows at
        one x; a line through them asserts a rate of change that does not
        exist, so the projection must be injective over the axis."""
        study = _run(tmp_path / "doc", tmp_path / "store", second="second.npz")
        with pytest.raises(SpecificationError, match="state_dim"):
            cost_vs_axis(
                _context(study, tmp_path / "store", _spec(axis_property="state_dim"))
            )

    def test_the_categorical_kind_refuses_it(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        spec = AnalysisSpec(
            id="scaling",
            kind="cost_by_category",
            config={"axis_property": "state_dim"},
        )
        with pytest.raises(SpecificationError, match="axis_property"):
            cost_by_category(_context(study, tmp_path / "store", spec))


class TestRelativeToContender:
    def test_the_reference_rows_are_exactly_one(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        output = cost_vs_axis(
            _context(
                study,
                tmp_path / "store",
                _spec(axis_property="state_dim", relative_to_contender="baseline"),
            )
        )
        reference = output.table[output.table["contender"] == "baseline"]
        assert len(reference) == 2
        assert (reference["aggregate"] == 1.0).all()

    def test_the_ratio_is_paired_per_seed_and_recomputable_by_hand(
        self, tmp_path: Path
    ) -> None:
        """The row's per-seed values must equal the two absolute per-seed
        aggregates divided seed by seed — the same pairing §A.4 declares,
        recomputed here from the ABSOLUTE table rather than from the store, so
        the assertion is about the normalisation alone."""
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        absolute = cost_vs_axis(
            _context(study, tmp_path / "store", _spec(axis_property="state_dim"))
        ).table
        ratio = cost_vs_axis(
            _context(
                study,
                tmp_path / "store",
                _spec(axis_property="state_dim", relative_to_contender="baseline"),
            )
        ).table

        for _, row in ratio.iterrows():
            if row["contender"] == "baseline":
                continue
            base = absolute[
                (absolute["contender"] == row["contender"])
                & (absolute["axis_value"] == row["axis_value"])
            ].iloc[0]
            anchor = absolute[
                (absolute["contender"] == "baseline")
                & (absolute["axis_value"] == row["axis_value"])
            ].iloc[0]
            paired = dict(zip(anchor["seeds"], anchor["per_seed_aggregate"]))
            expected = [
                value / paired[seed]
                for seed, value in zip(base["seeds"], base["per_seed_aggregate"])
            ]
            assert list(row["per_seed_aggregate"]) == expected
            assert row["aggregate"] == np.asarray(expected).mean()

    def test_the_pairing_is_per_seed_not_the_column_mean(self, tmp_path: Path) -> None:
        """A deterministic reference has identical per-seed values, so paired
        division and dividing by the column mean coincide on it — the exact
        blindness fig2's categorical suite recorded. A TRAINED reference's
        per-seed values differ, and only the paired division reproduces them;
        a mutant dividing by the anchor's mean survives the baseline-reference
        test and dies here."""
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        absolute = cost_vs_axis(
            _context(study, tmp_path / "store", _spec(axis_property="state_dim"))
        ).table
        anchor_rows = absolute[absolute["contender"] == "unfolded_a"]
        spreads = [
            np.std(row["per_seed_aggregate"]) for _, row in anchor_rows.iterrows()
        ]
        assert max(spreads) > 0.0, "the trained reference must vary across seeds"

        ratio = cost_vs_axis(
            _context(
                study,
                tmp_path / "store",
                _spec(axis_property="state_dim", relative_to_contender="unfolded_a"),
            )
        ).table
        for _, row in ratio.iterrows():
            base = absolute[
                (absolute["contender"] == row["contender"])
                & (absolute["axis_value"] == row["axis_value"])
            ].iloc[0]
            anchor = absolute[
                (absolute["contender"] == "unfolded_a")
                & (absolute["axis_value"] == row["axis_value"])
            ].iloc[0]
            paired = dict(zip(anchor["seeds"], anchor["per_seed_aggregate"]))
            expected = [
                value / paired[seed]
                for seed, value in zip(base["seeds"], base["per_seed_aggregate"])
            ]
            assert list(row["per_seed_aggregate"]) == expected

    def test_an_unknown_reference_is_refused_naming_the_cast(
        self, tmp_path: Path
    ) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        with pytest.raises(SpecificationError, match="no_such"):
            cost_vs_axis(
                _context(
                    study,
                    tmp_path / "store",
                    _spec(axis_property="state_dim", relative_to_contender="no_such"),
                )
            )

    def test_a_position_without_the_reference_is_refused(self, tmp_path: Path) -> None:
        """`applies_to` can exclude the reference from the swept positions;
        a ratio against a denominator that was never scored there must refuse
        rather than drop the position."""
        study = _run(
            tmp_path / "doc",
            tmp_path / "store",
            second="wider.npz",
            applies_to='"unfolded_a"',
        )
        with pytest.raises(SpecificationError, match="baseline"):
            cost_vs_axis(
                _context(
                    study,
                    tmp_path / "store",
                    _spec(axis_property="state_dim", relative_to_contender="baseline"),
                )
            )

    def test_the_sidecar_states_the_normalisation(self, tmp_path: Path) -> None:
        study = _run(tmp_path / "doc", tmp_path / "store", second="wider.npz")
        output = cost_vs_axis(
            _context(
                study,
                tmp_path / "store",
                _spec(axis_property="state_dim", relative_to_contender="baseline"),
            )
        )
        sidecar = str(output.sidecar)
        assert "baseline" in sidecar
        assert "state_dim" in sidecar
