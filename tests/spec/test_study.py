"""Acceptance tests for `StudySpec` and the sweep algebra — Stage 2 Phase E1.

Written before the implementation. The checkpoint class is **independent
recomputation**: the sweep algebra *claims* which identity level each axis
perturbs, and this suite checks that claim by deriving the identifiers at every
point of a materialised sweep and counting them.

Parent §4.3 is the specification:

| Axis touches | Level perturbed | Cost |
|---|---|---|
| recipe parameter, training plan, train seed | `ModelID` | one training per point |
| evaluation problem, evaluation protocol | `MeasurementID` only | one cheap rollout per point |

The second row is the economic claim the whole re-architecture rests on. It is
verified by *counting*, not by assertion: a study sweeping three depths across
four evaluation points must produce **three** distinct `ModelID`s and twelve
`MeasurementID`s, not twelve of each. If the classifier were wrong in the
optimistic direction the count would be too low and results would collide; in
the pessimistic direction the count would be too high and the sweep would cost
four times what it should.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from mbl.applications.factories import GaussianBatchSpec
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import EvaluationProtocol
from mbl.spec.contender import ContenderSpec
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import AxisLevel, Composition, StudySpec, SweepAxis
from mbl.spec.tiers import PERMITTED_TIER_PATHS
from mbl.spec.training import BatchPlan, TrainingSpec
from mbl.store.ids import STUDY_KEYS_EXCLUDED_FROM_STUDY_ID

from .test_contender import FAMILIES, REGISTRY

N, M, HORIZON = 4, 2, 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)


def _problem(seed: int = 0) -> ProblemSpec:
    rng = np.random.default_rng(seed)
    return ProblemSpec(
        ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
        )
    )


def _contender(family: str, label: str) -> ContenderSpec:
    return ContenderSpec(
        family=family, config=FAMILIES[family].config, label=label, registry=REGISTRY
    )


def _protocol(n_batches: int = 2, seed: int = 0) -> EvaluationProtocol:
    return EvaluationProtocol(
        batch_spec=GaussianBatchSpec(
            state_dim=N,
            horizon=HORIZON,
            batch_size=64,
            seed=seed,
            process_noise_std=0.5,
        ),
        n_batches=n_batches,
    )


def _single(**overrides: Any) -> StudySpec:
    """A one-contender study.

    The level checkpoints below count points, and `materialise` yields one per
    (combination x contender x seed) -- an execution unit, which is what a
    runner consumes. Isolating the axis to one contender keeps those counts
    about the axis rather than about how many contenders happen to be declared;
    `applies_to` and the whole-study counts are exercised separately.
    """
    fields: dict[str, Any] = {"contenders": (_contender("unfolded", "unfolded_a"),)}
    fields.update(overrides)
    return _study(**fields)


def _study(**overrides: Any) -> StudySpec:
    fields: dict[str, Any] = {
        "id": "probe/depth_scaling",
        "problem": _problem(),
        "contenders": (
            _contender("unfolded", "unfolded_a"),
            _contender("truncated_riccati", "baseline"),
        ),
        "training": TrainingSpec(
            data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
            plan=TrainingPlan(
                optimizer=OptimizerSpec(name="adam", learning_rate=1e-3), epochs=3
            ),
            batch=BatchPlan(effective_size=256),
            ctx=CTX,
        ),
        "evaluation": EvaluationSpec(problem=_problem(), protocol=_protocol(), ctx=CTX),
    }
    fields.update(overrides)
    return StudySpec(**fields)


DEPTH_AXIS = SweepAxis(
    path="contenders.*.config.num_iterations",
    values=(2, 4, 8),
    applies_to=("unfolded_a",),
)
BATCH_AXIS = SweepAxis(path="training.batch.effective_size", values=(128, 256))
SEED_AXIS = SweepAxis(path="training.seeds", values=((0,), (1,)))
EVAL_BATCHES_AXIS = SweepAxis(
    path="evaluation.protocol.n_batches", values=(2, 4, 8, 16)
)
EVAL_PROBLEM_AXIS = SweepAxis(
    path="evaluation.problem", values=(_problem(1), _problem(2))
)


# --------------------------------------------------------------------------
# Classification — every root, not one
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        (SweepAxis(path="problem", values=(_problem(1),)), AxisLevel.MODEL),
        (DEPTH_AXIS, AxisLevel.MODEL),
        (BATCH_AXIS, AxisLevel.MODEL),
        (SEED_AXIS, AxisLevel.MODEL),
        (EVAL_BATCHES_AXIS, AxisLevel.MEASUREMENT),
        (EVAL_PROBLEM_AXIS, AxisLevel.MEASUREMENT),
    ],
    ids=[
        "problem",
        "contender config",
        "training batch",
        "train seeds",
        "eval protocol",
        "eval problem",
    ],
)
def test_each_axis_is_classified_by_what_it_touches(
    axis: SweepAxis, expected: AxisLevel
) -> None:
    """Parent §4.3's table, as a test. Every root the grammar accepts is
    covered, because a classifier verified on one root is a classifier nobody
    checked."""
    assert axis.level is expected


def test_an_unknown_axis_root_is_refused_with_the_valid_ones() -> None:
    """A misspelled path is the most likely authoring error in a sweep, and it
    must not silently classify as anything."""
    with pytest.raises(SpecificationError) as error:
        SweepAxis(path="contender.config.depth", values=(1, 2)).level  # noqa: B018
    message = str(error.value)
    assert "contender.config.depth" in message
    for root in ("problem", "contenders", "training", "evaluation"):
        assert root in message


# --------------------------------------------------------------------------
# THE CHECKPOINT — the level claimed is the level actually perturbed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis", [DEPTH_AXIS, BATCH_AXIS, SEED_AXIS], ids=["depth", "batch", "seeds"]
)
def test_a_model_level_axis_costs_one_training_per_point(axis: SweepAxis) -> None:
    """Claimed MODEL, so every point must be a distinct model."""
    study = _single(sweep=(axis,))
    points = study.materialise()
    assert len(points) == len(axis.values)
    assert len({point.model_id for point in points}) == len(axis.values)


@pytest.mark.parametrize(
    "axis",
    [EVAL_BATCHES_AXIS, EVAL_PROBLEM_AXIS],
    ids=["eval protocol", "eval problem"],
)
def test_a_measurement_level_axis_costs_no_training(axis: SweepAxis) -> None:
    """Claimed MEASUREMENT, so the whole sweep is ONE model.

    This is the economic claim the re-architecture rests on: an out-of-
    distribution study over ten shifted problems is one training and ten
    rollouts, not ten trainings.
    """
    study = _single(sweep=(axis,))
    points = study.materialise()
    assert len(points) == len(axis.values)
    assert len({point.model_id for point in points}) == 1
    assert len({point.measurement_id for point in points}) == len(axis.values)


def test_the_dedup_claim_is_counted_over_a_mixed_sweep() -> None:
    """Parent §4.3: "a model appearing at ten evaluation points is trained once".

    Three depths crossed with four evaluation points is twelve measurements and
    **three** models. Counting it is the only way to see the difference between
    a correct classifier and one that treats every axis as model-level, which
    would cost four times the training and pass any test that only checked the
    measurements were distinct.
    """
    study = _single(sweep=(DEPTH_AXIS, EVAL_BATCHES_AXIS))
    points = study.materialise()

    assert len(points) == len(DEPTH_AXIS.values) * len(EVAL_BATCHES_AXIS.values)
    assert len({point.model_id for point in points}) == len(DEPTH_AXIS.values)
    assert len({point.measurement_id for point in points}) == len(points)


def test_every_contender_is_swept_when_applies_to_is_empty() -> None:
    """The default is "every contender", and the point count has to reflect it."""
    axis = SweepAxis(path="training.batch.effective_size", values=(64, 128, 256))
    points = _study(sweep=(axis,)).materialise()
    # Two contenders, three values, and the axis names neither -- so it moves
    # both, and every one of the six points is its own model.
    assert len(points) == 6
    assert len({point.model_id for point in points}) == 6


def test_applies_to_leaves_the_other_contenders_alone() -> None:
    """A depth axis applying to the unfolded contender must not perturb the
    analytic baseline, which has no depth at all -- and the baseline's model
    must therefore be *one* model across the whole sweep."""
    study = _study(sweep=(DEPTH_AXIS,))
    per_contender: dict[str, set[str]] = {}
    for point in study.materialise():
        per_contender.setdefault(point.contender.resolved_label, set()).add(
            point.model_id
        )
    assert len(per_contender["unfolded_a"]) == len(DEPTH_AXIS.values)
    assert len(per_contender["baseline"]) == 1


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def _axis(path: str, values: tuple[Any, ...], **extra: Any) -> SweepAxis:
    return SweepAxis(path=path, values=values, applies_to=("unfolded_a",), **extra)


DEPTHS_3 = ("contenders.*.config.num_iterations", (2, 4, 8))
BATCHES_3 = ("evaluation.protocol.n_batches", (2, 4, 8))


def test_product_crosses_the_axes() -> None:
    study = _single(sweep=(DEPTH_AXIS, EVAL_BATCHES_AXIS))
    assert len(study.materialise()) == 3 * 4


def test_zip_pairs_an_axis_with_the_one_before_it() -> None:
    """`zip` is what expresses a diagonal -- "depth 2 at 2 batches, depth 4 at
    4" -- which a product cannot say at any length."""
    study = _single(
        sweep=(
            _axis(*DEPTHS_3),
            _axis(*BATCHES_3, compose=Composition.ZIP),
        )
    )
    assert len(study.materialise()) == 3


def test_zip_over_axes_of_different_lengths_is_refused() -> None:
    """Silently truncating to the shortest is how a sweep quietly stops
    covering what its author wrote down."""
    short = _axis("evaluation.protocol.n_batches", (2, 4), compose=Composition.ZIP)
    with pytest.raises(SpecificationError, match="length"):
        _single(sweep=(_axis(*DEPTHS_3), short)).materialise()


# --------------------------------------------------------------------------
# Mixed composition — the case a per-STUDY rule cannot express at all
# --------------------------------------------------------------------------


def test_a_zipped_pair_products_with_an_independent_axis() -> None:
    """Annex 01 §2.5's worked example, and the reason `compose` belongs to the
    axis. Three depths, crossed with a control bound that carries a coupled
    companion: 3 x 3 = **9** points, not 27.

    A study forced to choose one composition for every axis at once cannot say
    this. Producting everything gives 27 and needs post-hoc filtering -- the
    exact thing `Compose.ZIP` exists to avoid -- while zipping everything
    collapses it to 3 and discards the product with depth.
    """
    study = _single(
        sweep=(
            _axis(*DEPTHS_3),
            _axis("evaluation.protocol.n_batches", (2, 4, 8)),
            _axis(
                "evaluation.protocol.batch_spec.batch_size",
                (32, 64, 128),
                compose=Composition.ZIP,
            ),
        )
    )
    points = study.materialise()
    assert len(points) == 3 * 3
    #: The coupling is the point: the companion never varies independently, so
    #: only the three declared pairings appear, never all nine.
    pairs = {
        (
            point.evaluation.protocol.n_batches,
            point.evaluation.protocol.batch_spec.batch_size,
        )
        for point in points
    }
    assert pairs == {(2, 32), (4, 64), (8, 128)}


def test_two_independent_zipped_pairs_are_two_groups() -> None:
    """A ZIP axis joins the group *immediately* before it, so a later PRODUCT
    axis opens a new group rather than extending the old one. Without that, a
    second coupled pair would be zipped into the first and the study would
    silently lose a dimension."""
    study = _single(
        sweep=(
            _axis(*DEPTHS_3),
            _axis(*BATCHES_3, compose=Composition.ZIP),
            _axis("training.batch.effective_size", (128, 256)),
            # Not `training.plan.epochs`: since G-2 that path is the study's
            # template, refused as an axis because it varies nothing.
            _axis("training.data.seed", (1, 2), compose=Composition.ZIP),
        )
    )
    assert len(study.materialise()) == 3 * 2


def test_the_first_axis_composition_is_unread() -> None:
    """It has nothing to join. A leading ZIP that meant anything would have to
    mean "zip with nothing", which is either an error or a no-op -- and making
    it an error would refuse a study that is perfectly well formed."""
    leading_zip = _single(
        sweep=(_axis(*DEPTHS_3, compose=Composition.ZIP), _axis(*BATCHES_3))
    )
    leading_product = _single(sweep=(_axis(*DEPTHS_3), _axis(*BATCHES_3)))
    assert leading_zip.study_id == leading_product.study_id
    assert len(leading_zip.materialise()) == 3 * 3


def test_composition_is_part_of_the_study_identity() -> None:
    """Two studies over the same axes composed differently are different
    studies: one is a grid and the other a diagonal, and they do not contain
    the same points."""
    grid = _single(sweep=(_axis(*DEPTHS_3), _axis(*BATCHES_3)))
    diagonal = _single(
        sweep=(_axis(*DEPTHS_3), _axis(*BATCHES_3, compose=Composition.ZIP))
    )
    assert grid.study_id != diagonal.study_id


def test_no_point_is_emitted_twice() -> None:
    """An axis that does not apply to a contender is a no-op for it, so the
    naive expansion emits that contender once per axis value -- byte-identical
    points, same model, same measurement, differing in nothing.

    Found by running the algebra over NB04's real shape rather than a fixture:
    six contenders over eight depths, of which only three carry a depth,
    expanded to 48 points of which **21 were exact repeats**. Emitting them
    would have the runner score the analytic baseline eight times to get eight
    identical numbers. The suite could not see it, because its own fixtures
    counted distinct *models* for the unswept contender and never counted its
    points.
    """
    study = _study(sweep=(DEPTH_AXIS,))
    points = study.materialise()
    keys = [(str(point.model_id), str(point.measurement_id)) for point in points]
    assert len(keys) == len(set(keys)), "materialise emitted a duplicate point"
    # Three depths for the contender the axis names, one for the other.
    assert len(points) == len(DEPTH_AXIS.values) + 1


def test_the_saving_is_the_whole_point_of_applies_to() -> None:
    """Stated as the arithmetic a reader can check: an axis over `k` of `n`
    contenders at `v` values costs `k*v + (n-k)` trainings, not `n*v`."""
    values = (2, 4, 8)
    axis = SweepAxis(
        path="contenders.*.config.num_iterations",
        values=values,
        applies_to=("unfolded_a",),
    )
    points = _study(sweep=(axis,)).materialise()
    naive = 2 * len(values)
    assert len(points) == 1 * len(values) + 1
    assert len(points) < naive


def test_a_study_with_no_sweep_is_one_point_per_contender() -> None:
    """The degenerate sweep is a valid study, not an error: most studies are a
    fixed comparison."""
    points = _study().materialise()
    assert len(points) == 2
    assert len({point.model_id for point in points}) == 2


# --------------------------------------------------------------------------
# Identity of the study itself
# --------------------------------------------------------------------------


def test_renaming_a_study_does_not_orphan_its_results() -> None:
    """Annex 01 §5 rule 1. The `id` is what the study is filed under, not what
    it is."""
    assert _study(id="a/b").study_id == _study(id="c/d").study_id


def test_changing_the_study_content_changes_its_identity() -> None:
    assert _study().study_id != _study(sweep=(DEPTH_AXIS,)).study_id


def test_the_id_is_emitted_under_the_name_tier_4_drops() -> None:
    """A by-name contract with Tier 4, exactly as `seeds` is: renaming the key
    would silently put the study's name back into its identity, and the
    behavioural test above would still pass."""
    tree = _study().get_signature()
    for key in STUDY_KEYS_EXCLUDED_FROM_STUDY_ID:
        assert key in tree, f"{key!r} is stripped by study_id but never emitted"


# --------------------------------------------------------------------------
# Degeneracy
# --------------------------------------------------------------------------


def test_an_axis_with_no_values_is_refused() -> None:
    with pytest.raises(SpecificationError, match="values"):
        SweepAxis(path="training.batch.effective_size", values=())


def test_applies_to_naming_an_unknown_contender_is_refused() -> None:
    """The authoring error this catches is a typo in a contender label, which
    would otherwise sweep nothing and silently produce a one-point study."""
    axis = SweepAxis(
        path="contenders.*.config.num_iterations",
        values=(2, 4),
        applies_to=("unfolded_typo",),
    )
    with pytest.raises(SpecificationError) as error:
        _study(sweep=(axis,)).materialise()
    assert "unfolded_typo" in str(error.value)
    assert "unfolded_a" in str(error.value)


def test_an_unresolvable_path_is_refused_naming_the_field() -> None:
    axis = SweepAxis(path="training.batch.no_such_field", values=(1, 2))
    with pytest.raises(SpecificationError, match="no_such_field"):
        _study(sweep=(axis,)).materialise()


def test_duplicate_contender_labels_are_refused() -> None:
    """Labels index every result, so two contenders sharing one would make the
    study's own output ambiguous."""
    duplicate = _contender("truncated_riccati", "baseline")
    with pytest.raises(SpecificationError, match="baseline"):
        _study(contenders=(duplicate, _contender("riccati", "baseline")))


def test_a_study_with_no_contenders_is_refused() -> None:
    with pytest.raises(SpecificationError, match="contender"):
        _study(contenders=())


def test_a_study_is_frozen() -> None:
    study = _study()
    with pytest.raises(dataclasses.FrozenInstanceError):
        study.id = "renamed"  # type: ignore[misc]  # the mutation attempt is the test


# --------------------------------------------------------------------------
# The study's default plan — Stage 2 Phase G-2
# --------------------------------------------------------------------------
#
# Annex 01 §2.3.4: `TrainingSpec.plan` is the study's default, materialised
# into each contender that accepts one when the study is built. The home is
# `StudySpec.__post_init__` and not the loader, because E3 pinned that neither
# surface is privileged -- a loader-only step would give a study composed in
# Python different contenders from the same study written in TOML.


def _plan(epochs: int) -> TrainingPlan:
    return TrainingPlan(
        optimizer=OptimizerSpec(name="adam", learning_rate=1e-3), epochs=epochs
    )


def _plan_of(study: StudySpec, label: str) -> Any:
    config = dict(next(c for c in study.contenders if c.resolved_label == label).config)
    return config.get("plan")


def _without_plan(label: str) -> ContenderSpec:
    return ContenderSpec(
        family="unfolded",
        config={k: v for k, v in FAMILIES["unfolded"].config.items() if k != "plan"},
        label=label,
        registry=REGISTRY,
    )


def test_the_study_plan_is_materialised_when_the_study_is_built() -> None:
    """A contender that declares no plan gets the study's, and is *resolvable*
    afterwards -- which it is not before, since `plan` is a required field of
    every family that takes one."""
    study = _study(
        contenders=(_without_plan("unfolded_a"), _contender("truncated_riccati", "b")),
        training=replace(_study().training, plan=_plan(41)),
    )
    assert _plan_of(study, "unfolded_a").epochs == 41
    assert _plan_of(study, "b") is None
    assert study.contenders[0].resolve().plan.epochs == 41


def test_materialising_equals_spelling_the_plan_out() -> None:
    """Independent recomputation. If a default is really a default, the study
    that omits it and the study that writes it into the contender are one
    study -- the same `StudyID`, the same points, the same identifiers."""
    plan = _plan(41)
    omitted = _study(
        contenders=(_without_plan("unfolded_a"),),
        training=replace(_study().training, plan=plan),
    )
    spelled = _study(
        contenders=(
            ContenderSpec(
                family="unfolded",
                config={**FAMILIES["unfolded"].config, "plan": plan},
                label="unfolded_a",
                registry=REGISTRY,
            ),
        ),
        training=replace(_study().training, plan=plan),
    )
    assert omitted.study_id == spelled.study_id
    assert [str(p.model_id) for p in omitted.materialise()] == [
        str(p.model_id) for p in spelled.materialise()
    ]


def test_two_studies_differing_only_in_the_template_are_one_study() -> None:
    """**The negative control G-2 exists for.** Every contender here declares
    its own plan, so the template governs nothing -- and before G-2 it moved
    every `ModelID` and the `StudyID` regardless, which is two identifiers for
    one model in a store whose premise is that they are the same thing."""
    five = _study(training=replace(_study().training, plan=_plan(5)))
    five_hundred = _study(training=replace(_study().training, plan=_plan(500)))
    assert five.study_id == five_hundred.study_id
    assert [str(p.model_id) for p in five.materialise()] == [
        str(p.model_id) for p in five_hundred.materialise()
    ]


def test_two_studies_differing_in_a_contender_s_own_plan_are_two() -> None:
    """Anti-vacuity for the control above, and the property that must survive
    it: the plan that *executes* is still identity, through the contender."""

    def _built(epochs: int) -> StudySpec:
        return _study(
            contenders=(
                ContenderSpec(
                    family="unfolded",
                    config={
                        **FAMILIES["unfolded"].config,
                        "plan": _plan(epochs),
                    },
                    label="unfolded_a",
                    registry=REGISTRY,
                ),
            )
        )

    assert _built(5).study_id != _built(500).study_id
    assert [str(p.model_id) for p in _built(5).materialise()] != [
        str(p.model_id) for p in _built(500).materialise()
    ]


def test_an_axis_over_the_study_plan_is_refused() -> None:
    """The side effect that fails *silently* if it is not refused.

    After materialisation the template reaches nobody, so every value of such
    an axis derives identical identifiers, `materialise` deduplicates them, and
    an eight-value sweep expands to one point while the document says it swept
    eight. Refused naming the path the author meant.
    """
    with pytest.raises(SpecificationError, match=r"contenders\.\*\.config\.plan"):
        SweepAxis(path="training.plan.epochs", values=(5, 50, 500))
    with pytest.raises(SpecificationError, match=r"contenders\.\*\.config\.plan"):
        SweepAxis(path="training.plan", values=(_plan(5), _plan(50)))


def test_an_axis_over_the_plan_that_executes_is_accepted() -> None:
    """Anti-vacuity: the refusal above must be about the template, not about
    the word `plan`. This is the axis the message points at, and it works."""
    study = _single(sweep=(_axis("contenders.*.config.plan.epochs", (5, 50, 500)),))
    assert len(study.materialise()) == 3
    assert len({str(p.model_id) for p in study.materialise()}) == 3


def test_a_tier_may_still_write_the_template() -> None:
    """A tier writing `training.plan.*` keeps a resolved document from
    declaring a 200-epoch plan while running 5 (F2a). That is a declaration
    being kept honest, not a claim to vary something -- so it stays permitted
    where the axis does not, and it must still change no identifier."""
    assert "training.plan.*" in PERMITTED_TIER_PATHS
    written = replace(_study(), training=replace(_study().training, plan=_plan(5)))
    assert written.study_id == _study().study_id


def test_an_unlabelled_contender_is_materialised_before_it_is_named() -> None:
    """Ordering inside `__post_init__`, and it is load-bearing.

    A contender's label defaults to its recipe's, which means *resolving* it —
    and a contender that declares no plan cannot be resolved until it has one.
    So materialisation runs before the uniqueness check that reads every label,
    not after. Every document sets a label explicitly, which is exactly why
    this case needs a test of its own: the Python surface does not have to.
    """
    unlabelled = ContenderSpec(
        family="unfolded",
        config={k: v for k, v in FAMILIES["unfolded"].config.items() if k != "plan"},
        registry=REGISTRY,
    )
    study = _study(
        contenders=(unlabelled,), training=replace(_study().training, plan=_plan(41))
    )
    assert study.contenders[0].resolved_label == "unfolded_learned_step_size"
    assert _plan_of(study, "unfolded_learned_step_size").epochs == 41


# --------------------------------------------------------------------------
# `StudyPoint.axis_values` — what an analysis labels an axis with (B2a)
# --------------------------------------------------------------------------


def test_a_point_records_the_value_the_axis_bound_it_at() -> None:
    study = _single(sweep=(DEPTH_AXIS,))

    bound = {
        point.contender.config["num_iterations"]: point.axis_values
        for point in study.materialise()
    }
    assert set(bound) == {2, 4, 8}
    for depth, values in bound.items():
        assert values == {DEPTH_AXIS.path: depth}


def test_a_study_with_no_sweep_records_nothing() -> None:
    for point in _study().materialise():
        assert point.axis_values == {}


def test_an_axis_an_applies_to_excludes_contributes_no_entry() -> None:
    """The depth-invariant case, which is the reason this field exists.

    A contender the axis does not apply to has no depth. An analysis draws it
    as a flat reference; recording some depth it never had would place a
    horizontal line at one arbitrary point of the axis and call it a curve.
    """
    study = _study(sweep=(DEPTH_AXIS,))

    by_label = {
        point.contender.resolved_label: point.axis_values
        for point in study.materialise()
    }
    assert by_label["baseline"] == {}
    assert DEPTH_AXIS.path in by_label["unfolded_a"]


def test_every_axis_of_a_product_is_recorded() -> None:
    study = _single(sweep=(DEPTH_AXIS, EVAL_BATCHES_AXIS))

    recorded = [point.axis_values for point in study.materialise()]
    assert len(recorded) == len(DEPTH_AXIS.values) * len(EVAL_BATCHES_AXIS.values)
    for values in recorded:
        assert set(values) == {DEPTH_AXIS.path, EVAL_BATCHES_AXIS.path}


def test_the_recorded_value_is_the_one_written_into_the_spec() -> None:
    """Recorded, not re-derived — and the two must agree.

    The alternative design read the value back out of the spec tree by path.
    This asserts the equivalence on the axis where reading back is possible,
    which is what makes the mapping trustworthy on the axes where it is not.
    """
    for point in _single(sweep=(DEPTH_AXIS,)).materialise():
        assert (
            point.axis_values[DEPTH_AXIS.path]
            == point.contender.config["num_iterations"]
        )


def test_recording_the_axis_moves_no_identifier() -> None:
    """`axis_values` is a label, never identity.

    Every identifier is derived from the four specs, into which the value has
    already been written. Two points that differ only in this mapping would be
    the same work, and `materialise` deduplicates on exactly that.
    """
    study = _single(sweep=(DEPTH_AXIS,))
    points = study.materialise()

    relabelled = [
        dataclasses.replace(point, axis_values={"nonsense": 1}) for point in points
    ]
    assert [str(p.model_id) for p in relabelled] == [str(p.model_id) for p in points]
    assert [str(p.measurement_id) for p in relabelled] == [
        str(p.measurement_id) for p in points
    ]
