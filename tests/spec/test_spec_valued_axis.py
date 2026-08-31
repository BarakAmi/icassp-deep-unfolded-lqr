"""Acceptance tests for a sweep axis whose values are whole specifications.

Annex 01 §2.5.1, written before the implementation. Two paths address a whole
`ProblemSpec` rather than a scalar inside one, and until now neither could be
swept: measured, the document **loads** and `materialise()` then dies with
`AttributeError: 'str' object has no attribute 'problem_id'`.

**This suite is also Phase D's own acceptance, and that is the point.** The
plan states it as store arithmetic — *"swapping `[evaluation.problem]` gives
'trained 0, reused 3' with measurements 3 → 6, while swapping `[problem]`
gives 'would train 3, reused 0'"* — which until now needed four separate
documents to observe. On one axis it becomes a property of `materialise()`:
the measurement-level path must leave **every** `ModelID` untouched while
doubling the measurements, and the model-level path must move every one.
Asserting the ModelIDs are the *same set* as the unswept study is stronger
than counting them, because a count is equal for two disjoint sets.

Three further properties carry the rest:

* **A label is presentation and takes no part in any identifier.** The
  standing 73-line identity golden must not move, and two studies differing
  only in their labels must derive one `StudyID` — the same contract a
  contender's `display` has.
* **A spec-valued axis without labels is REFUSED**, because `str(ProblemSpec)`
  is a repr and a filename would name a category
  `icassp_n4m2_N100_u0p1_s0_rotA30` in a paper. A refusal an author can act on
  beats an unreadable figure.
* **Nested problem paths were never broken and must stay working.**
  `problem.data.control_bound` materialises today, so a fix aimed at the two
  whole-spec paths that also caught the nested ones would be a regression.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import load_study
from mbl.spec.problem import ProblemSpec

_producer = importlib.import_module("tests.runner.test_producer")

#: The two paths whose value is a whole `ProblemSpec` (Annex 01 §2.5.1).
MEASUREMENT_LEVEL = "evaluation.problem"
MODEL_LEVEL = "problem"

NOMINAL, SHIFTED = "problem.npz", "rotated.npz"
LABELS = ("nominal", "rotated 30 deg")


def _document(directory: Path, sweep: str) -> Path:
    """The producer fixture's study with its sweep block replaced.

    Two frozen problems are written beside it, because D19 puts problem data
    next to the study that uses it and the loader resolves an axis value the
    same way it resolves `[problem].path`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _producer._problem_file(directory, NOMINAL, seed=0)
    _producer._problem_file(directory, SHIFTED, seed=1)
    text = _producer.STUDY.format(**_producer.DEFAULTS)
    head, marker, tail = text.partition("[[sweep]]")
    assert marker, "the producer fixture's sweep block moved"
    # The WHOLE block is replaced, `applies_to` included. Leaving the fixture's
    # `applies_to = ["unfolded_a"]` in place silently narrowed the axis to one
    # contender and made the point counts below disagree with the arithmetic --
    # the code was right and the fixture was not.
    (directory / "study.toml").write_text(f"{head}[[sweep]]\n{sweep}\n")
    return directory / "study.toml"


def _study(directory: Path, sweep: str) -> Any:
    return load_study(_document(directory, sweep), bindings=DEFAULT_SPEC_BINDINGS).study


def _unswept(directory: Path) -> Any:
    """The same study with no axis at all — the baseline the counts are against."""
    directory.mkdir(parents=True, exist_ok=True)
    _producer._problem_file(directory, NOMINAL, seed=0)
    text = _producer.STUDY.format(**_producer.DEFAULTS)
    head = text.split("[[sweep]]")[0]
    (directory / "study.toml").write_text(head)
    return load_study(directory / "study.toml", bindings=DEFAULT_SPEC_BINDINGS).study


def _sweep(path: str, *, labels: bool = True, values: str | None = None) -> str:
    declared = values or f'["{NOMINAL}", "{SHIFTED}"]'
    block = f'path = "{path}"\nvalues = {declared}'
    if labels:
        block += f'\nlabels = ["{LABELS[0]}", "{LABELS[1]}"]'
    return block


def _ids(study: Any) -> tuple[set[str], set[str]]:
    points = study.materialise()
    return (
        {str(point.model_id) for point in points},
        {str(point.measurement_id) for point in points},
    )


class TestTheDefectItExistsFor:
    def test_both_whole_spec_paths_are_swept_rather_than_crashing(
        self, tmp_path: Path
    ) -> None:
        """Measured before this existed: `AttributeError: 'str' object has no
        attribute 'problem_id'`, out of `materialise()` on both paths."""
        for index, path in enumerate((MEASUREMENT_LEVEL, MODEL_LEVEL)):
            study = _study(tmp_path / f"doc{index}", _sweep(path))
            assert study.materialise(), path

    def test_a_nested_problem_path_still_works(self, tmp_path: Path) -> None:
        """The regression guard. This was never broken — a fix aimed at the two
        whole-spec paths that also caught the nested ones would be one."""
        base = len(_unswept(tmp_path / "base").materialise())
        study = _study(
            tmp_path / "doc",
            'path = "problem.data.control_bound"\nvalues = [0.1, 0.2]',
        )
        # Against the unswept count rather than a literal: the earlier 6 was
        # measured on a fixture that still carried `applies_to = ["unfolded_a"]`
        # and is not a property of this path at all.
        assert len(study.materialise()) == 2 * base


class TestPhaseDsOwnAcceptance:
    """The plan's store arithmetic, as a property of one document.

    *"Swapping `[evaluation.problem]` gives 'trained 0, reused 3' with
    measurements 3 → 6, while swapping `[problem]` gives 'would train 3,
    reused 0'."*
    """

    def test_the_measurement_level_path_moves_no_model_id(self, tmp_path: Path) -> None:
        base_models, base_measurements = _ids(_unswept(tmp_path / "base"))
        models, measurements = _ids(
            _study(tmp_path / "shifted", _sweep(MEASUREMENT_LEVEL))
        )

        # The SAME SET, not merely the same count: two disjoint sets of three
        # would satisfy a count and would be three retrainings.
        assert models == base_models
        assert len(measurements) == 2 * len(base_measurements)
        assert base_measurements < measurements

    def test_the_model_level_path_moves_every_model_id(self, tmp_path: Path) -> None:
        base_models, _ = _ids(_unswept(tmp_path / "base"))
        models, measurements = _ids(_study(tmp_path / "trained", _sweep(MODEL_LEVEL)))

        assert len(models) == 2 * len(base_models)
        assert base_models < models
        assert len(measurements) == len(models)

    def test_the_two_paths_are_not_the_same_experiment(self, tmp_path: Path) -> None:
        """Experiment 1 and Experiment 2 differ in exactly this, and the
        grammar decides it from the path without an author saying which."""
        evaluation, _ = _ids(_study(tmp_path / "eval", _sweep(MEASUREMENT_LEVEL)))
        training, _ = _ids(_study(tmp_path / "train", _sweep(MODEL_LEVEL)))

        assert evaluation != training


class TestTheValueIsARealSpec:
    def test_the_axis_value_carries_a_problem_id(self, tmp_path: Path) -> None:
        study = _study(tmp_path / "doc", _sweep(MEASUREMENT_LEVEL))
        # A LIST, not a set: `ProblemSpec` holds NumPy arrays in dicts and is
        # deliberately unhashable, which is the same reason `materialise`
        # deduplicates on derived identifiers rather than on points.
        values = [
            point.axis_values[MEASUREMENT_LEVEL]
            for point in study.materialise()
            if MEASUREMENT_LEVEL in point.axis_values
        ]

        assert values
        for value in values:
            assert isinstance(value, ProblemSpec)
            assert str(value.problem_id)

    def test_the_two_problems_are_actually_different(self, tmp_path: Path) -> None:
        """Or the axis sweeps one plant twice and every count above is a
        coincidence of the fixture."""
        study = _study(tmp_path / "doc", _sweep(MEASUREMENT_LEVEL))
        ids = {
            str(value.problem_id)
            for value in [
                point.axis_values[MEASUREMENT_LEVEL]
                for point in study.materialise()
                if MEASUREMENT_LEVEL in point.axis_values
            ]
        }
        assert len(ids) == 2

    def test_the_evaluation_spec_is_the_one_that_moved(self, tmp_path: Path) -> None:
        """Not merely recorded in `axis_values`: the value has to reach the
        spec the identifier is derived from."""
        study = _study(tmp_path / "doc", _sweep(MEASUREMENT_LEVEL))
        scored = {
            str(point.evaluation.problem.problem_id) for point in study.materialise()
        }
        assert len(scored) == 2

    def test_a_missing_file_is_refused_naming_it(self, tmp_path: Path) -> None:
        with pytest.raises(SpecificationError, match="absent.npz"):
            _study(
                tmp_path / "doc",
                _sweep(MEASUREMENT_LEVEL, values=f'["{NOMINAL}", "absent.npz"]'),
            )


class TestLabels:
    def test_the_grammar_does_not_require_them(self, tmp_path: Path) -> None:
        """The correction, as a test rather than as a comment.

        Enforcing this in `SweepAxis.__post_init__` was tried and refuted: the
        grammar's own suite builds a spec-valued `evaluation.problem` axis
        purely to classify it, and a producer-only sweep never shows a value to
        anybody. The refusal belongs in the analysis that READS a label
        (Annex 03 §A.6.1), and `tests/analysis/test_cost_by_category.py` owns
        it. Requiring a label to materialise a point would put a figure's
        concern inside the identity grammar.
        """
        study = _study(tmp_path / "doc", _sweep(MEASUREMENT_LEVEL, labels=False))
        assert study.sweep[0].labels == ()
        assert study.materialise()

    def test_they_are_carried_on_the_axis_in_declaration_order(
        self, tmp_path: Path
    ) -> None:
        study = _study(tmp_path / "doc", _sweep(MEASUREMENT_LEVEL))
        assert study.sweep[0].labels == LABELS

    def test_a_length_mismatch_is_refused(self, tmp_path: Path) -> None:
        block = (
            f'path = "{MEASUREMENT_LEVEL}"\n'
            f'values = ["{NOMINAL}", "{SHIFTED}"]\n'
            'labels = ["only one"]'
        )
        with pytest.raises(SpecificationError, match="labels"):
            _study(tmp_path / "doc", block)

    def test_a_scalar_axis_needs_none(self, tmp_path: Path) -> None:
        """`2` and `4` are their own names, so requiring labels everywhere
        would be a tax on every study in the repository."""
        study = _study(
            tmp_path / "doc",
            'path = "contenders.*.config.num_iterations"\nvalues = [2, 4]',
        )
        assert study.sweep[0].labels == ()

    def test_a_scalar_axis_may_still_declare_them(self, tmp_path: Path) -> None:
        study = _study(
            tmp_path / "doc",
            'path = "contenders.*.config.num_iterations"\n'
            "values = [2, 4]\n"
            'labels = ["shallow", "deep"]',
        )
        assert study.sweep[0].labels == ("shallow", "deep")


class TestLabelsTakeNoPartInIdentity:
    """The contract a contender's `display` has (Annex 01 §2.2.1): renaming
    what a reader is shown must not retrain anything or orphan a result."""

    def _relabelled(self, tmp_path: Path) -> tuple[Any, Any]:
        block = _sweep(MEASUREMENT_LEVEL)
        other = block.replace(f'"{LABELS[0]}", "{LABELS[1]}"', '"A", "B"')
        return (
            _study(tmp_path / "one", block),
            _study(tmp_path / "two", other),
        )

    def test_the_study_id_does_not_move(self, tmp_path: Path) -> None:
        first, second = self._relabelled(tmp_path)
        assert first.sweep[0].labels != second.sweep[0].labels
        assert str(first.study_id) == str(second.study_id)

    def test_no_model_or_measurement_id_moves(self, tmp_path: Path) -> None:
        first, second = self._relabelled(tmp_path)
        assert _ids(first) == _ids(second)

    def test_the_signature_does_not_mention_them(self, tmp_path: Path) -> None:
        """Asserted on the emitted signature rather than only on the hash: a
        hash that happened to collide would pass the two tests above."""
        study, _ = self._relabelled(tmp_path)
        assert "labels" not in str(study.get_signature()["sweep"])


class TestTheEvaluationFollowsTheSweptProblem:
    """Phase 1 of the Figure-3 plan: sweeping the root `problem` moves the
    WHOLE study problem — the matched case at every position.

    Measured before the fix (probe, 2026-08-08): on a two-plant root sweep the
    second point's `evaluation.problem` stayed the DECLARED nominal and its
    batch spec still said `state_dim = 4` — an n-plant model scored on the
    4-plant instance, refused (at best) by a shape error deep in a rollout.
    """

    def test_the_evaluation_problem_is_the_points_problem(self, tmp_path: Path) -> None:
        study = _study(tmp_path / "doc", _sweep(MODEL_LEVEL))
        points = study.materialise()
        assert points
        for point in points:
            assert str(point.evaluation.problem.problem_id) == str(
                point.problem.problem_id
            )

    def test_the_batch_dimensions_follow_a_different_shape(
        self, tmp_path: Path
    ) -> None:
        """The trap the rehost plan recorded as out of scope, now closed: the
        protocol is no longer frozen at the declared plant's dimensions."""
        directory = tmp_path / "doc"
        directory.mkdir(parents=True, exist_ok=True)
        _producer._problem_file(directory, NOMINAL, seed=0)
        _wider_problem_file(directory, "wider.npz", seed=1)
        text = _producer.STUDY.format(**_producer.DEFAULTS)
        head, marker, _ = text.partition("[[sweep]]")
        assert marker
        sweep = _sweep(MODEL_LEVEL, values=f'["{NOMINAL}", "wider.npz"]')
        (directory / "study.toml").write_text(f"{head}[[sweep]]\n{sweep}\n")
        study = load_study(
            directory / "study.toml", bindings=DEFAULT_SPEC_BINDINGS
        ).study

        dims = {
            point.problem.state_dim: point.evaluation.protocol.batch_spec.state_dim
            for point in study.materialise()
        }
        # Two shapes on the axis, and each point's protocol draws its own.
        assert dims == {
            _producer.N: _producer.N,
            _producer.N + 2: _producer.N + 2,
        }
        horizons = {
            point.evaluation.protocol.batch_spec.horizon
            for point in study.materialise()
            if point.problem.state_dim == _producer.N + 2
        }
        assert horizons == {_producer.HORIZON + 3}

    def test_an_explicitly_shifted_evaluation_stays_pinned(
        self, tmp_path: Path
    ) -> None:
        """Following is the MATCHED default's behaviour only. A document that
        pins a shifted evaluation problem keeps it at every axis position."""
        directory = tmp_path / "doc"
        directory.mkdir(parents=True, exist_ok=True)
        _producer._problem_file(directory, NOMINAL, seed=0)
        _producer._problem_file(directory, SHIFTED, seed=1)
        _producer._problem_file(directory, "pinned.npz", seed=2)
        text = _producer.STUDY.format(**_producer.DEFAULTS)
        head, marker, _ = text.partition("[[sweep]]")
        assert marker
        head = head.replace(
            "[evaluation]",
            '[evaluation.problem]\npath = "pinned.npz"\n\n[evaluation]',
            1,
        )
        (directory / "study.toml").write_text(
            f"{head}[[sweep]]\n{_sweep(MODEL_LEVEL)}\n"
        )
        study = load_study(
            directory / "study.toml", bindings=DEFAULT_SPEC_BINDINGS
        ).study

        points = study.materialise()
        pinned = {str(point.evaluation.problem.problem_id) for point in points}
        assert len(pinned) == 1
        assert pinned != {str(point.problem.problem_id) for point in points}

    def test_the_matched_position_reuses_the_unswept_measurement(
        self, tmp_path: Path
    ) -> None:
        """Following must be an identity no-op where nothing moved: the axis
        position at the declared plant derives exactly the unswept study's
        MeasurementIDs, or the follow perturbed the matched case."""
        _, base_measurements = _ids(_unswept(tmp_path / "base"))
        _, swept_measurements = _ids(_study(tmp_path / "swept", _sweep(MODEL_LEVEL)))
        assert base_measurements < swept_measurements


def _wider_problem_file(directory: Path, name: str, seed: int) -> Path:
    """A plant two state dimensions wider and three steps longer than the
    fixture's, so a followed batch spec is measurably different."""
    import numpy as np

    from mbl.spec.problem import ProblemData

    n = _producer.N + 2
    horizon = _producer.HORIZON + 3
    rng = np.random.default_rng(seed)
    spec = ProblemSpec(
        ProblemData(
            system={
                "A": rng.normal(size=(n, n)),
                "B": rng.normal(size=(n, _producer.M)),
            },
            cost={
                "Q": np.repeat(np.eye(n)[None], horizon + 1, axis=0),
                "R": np.repeat(np.eye(_producer.M)[None], horizon, axis=0),
            },
            horizon=horizon,
            control_bound=0.5,
        )
    )
    spec.save(directory / name)
    return directory / name
