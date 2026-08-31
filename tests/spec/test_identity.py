"""Acceptance tests for the identity split — Stage 2 Phase D.

Written before the implementation. This is the phase the whole re-architecture
turns on, so the checkpoint is **negative control plus structural**, and the
structural half carries most of the weight.

The defect being closed is one dictionary literal in
`experiments/experiment.py`, which keys a model on
``{problem, contender, evaluation, compute_context}``. Because the evaluation
protocol is in there, raising an evaluation batch count, changing an evaluation
seed or adding a metric invalidates every trained model and retrains it.

The corrected shape is two identifiers instead of one:

* `ModelID` = H(problem, contender, training, ctx, seed) — **no evaluation term
  of any kind**;
* `MeasurementID` = H(model, eval_problem, eval_protocol).

The consequence is that distribution shift stops being a subsystem: a
measurement whose evaluation problem differs from the model's training problem
*is* a shifted result, at the cost of one rollout and zero training.

Both halves are asserted, and neither is sufficient alone. Behaviourally, every
way an evaluation can differ must leave `ModelID` untouched and move
`MeasurementID`. Structurally, `derive_model_id` must have **no parameter**
through which an evaluation could be passed at all — because a behavioural test
only covers the ways of differing that this module happened to think of, while
the absence of a parameter covers every way there will ever be.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.core.utils.signing import Signable
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import EvaluationProtocol
from mbl.applications.factories import GaussianBatchSpec
from mbl.applications.recipes.unfolded import UnfoldedKind
from mbl.spec.contender import ContenderSpec
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.identity import derive_measurement_id, derive_model_id
from mbl.spec.problem import GeneratorProvenance, ProblemData, ProblemSpec
from mbl.spec.training import BatchPlan, TrainingSpec
from mbl.store.ids import MeasurementID, ModelID

from .test_contender import FAMILIES, REGISTRY

EVERY_FAMILY = sorted(FAMILIES)

N, M, HORIZON = 4, 2, 6


def _problem(seed: int = 0, generator: str = "generate_marginally_stable_system"):
    rng = np.random.default_rng(seed)
    return ProblemSpec(
        data=ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
        ),
        provenance=GeneratorProvenance(generator=generator, params={"seed": seed}),
    )


def _ctx(precision: Precision = Precision.FLOAT64) -> ComputeContext:
    return ComputeContext(backend=Backend.TORCH, device="cpu", precision=precision)


def _contender(family: str = "riccati") -> ContenderSpec:
    return ContenderSpec(
        family=family, config=FAMILIES[family].config, registry=REGISTRY
    )


def _training(**overrides: Any) -> TrainingSpec:
    fields: dict[str, Any] = {
        "data": DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
        "plan": TrainingPlan(
            optimizer=OptimizerSpec(name="adam", learning_rate=1e-3), epochs=3
        ),
        "batch": BatchPlan(effective_size=256),
        "ctx": _ctx(),
    }
    fields.update(overrides)
    return TrainingSpec(**fields)


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


def _evaluation(**overrides: Any) -> EvaluationSpec:
    fields: dict[str, Any] = {
        "problem": _problem(),
        "protocol": _protocol(),
        "ctx": _ctx(),
    }
    fields.update(overrides)
    return EvaluationSpec(**fields)


def _leaves(tree: Any) -> list[str]:
    if isinstance(tree, dict):
        return [key for k, v in tree.items() for key in (k, *_leaves(v))]
    if isinstance(tree, (list, tuple)):
        return [key for item in tree for key in _leaves(item)]
    return []


#: Every way an evaluation can differ, as (name, kwargs) pairs. Parametrised
#: rather than written out once, because "changing the evaluation costs no
#: training" has to hold for every axis of the evaluation, not the one axis a
#: single example happens to exercise.
EVALUATION_VARIATIONS: list[tuple[str, dict[str, Any]]] = [
    ("more batches", {"protocol": _protocol(n_batches=8)}),
    ("a different evaluation seed", {"protocol": _protocol(seed=99)}),
    ("a different evaluation problem", {"problem": _problem(seed=7)}),
    ("a different scoring precision", {"ctx": _ctx(Precision.FLOAT32)}),
    ("an added metric", {"metrics": ("expected_cost", "violation_rate")}),
    (
        "protocol and problem together",
        {"protocol": _protocol(n_batches=8), "problem": _problem(seed=7)},
    ),
]
VARIATION_IDS = [name for name, _ in EVALUATION_VARIATIONS]


# --------------------------------------------------------------------------
# THE SPLIT — the regression test for the defect this whole plan exists to fix
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
@pytest.mark.parametrize(
    "variation", [kwargs for _, kwargs in EVALUATION_VARIATIONS], ids=VARIATION_IDS
)
def test_changing_the_evaluation_costs_no_training(
    family: str, variation: dict[str, Any]
) -> None:
    """Every way of evaluating differently, for every contender family.

    This is the direct regression test for parent §2.2. Under the old cache key
    each of these variations invalidated the trained model; under the split none
    of them may touch it.
    """
    problem, contender, training = _problem(), _contender(family), _training()
    baseline = derive_model_id(problem, contender, training, seed=0)

    shifted_evaluation = dataclasses.replace(_evaluation(), **variation)
    assert derive_model_id(problem, contender, training, seed=0) == baseline
    # ... and the measurement it produces is a genuinely different measurement.
    assert derive_measurement_id(baseline, shifted_evaluation) != derive_measurement_id(
        baseline, _evaluation()
    )


@pytest.mark.parametrize(
    "variation", [kwargs for _, kwargs in EVALUATION_VARIATIONS], ids=VARIATION_IDS
)
def test_every_evaluation_difference_moves_the_measurement(
    variation: dict[str, Any],
) -> None:
    """The other half of the split, and the reason the first half is not
    vacuous: an implementation that ignored the evaluation entirely would pass
    the test above and be useless."""
    model = derive_model_id(_problem(), _contender(), _training(), seed=0)
    varied = dataclasses.replace(_evaluation(), **variation)
    assert derive_measurement_id(model, varied) != derive_measurement_id(
        model, _evaluation()
    )


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_two_models_scored_alike_are_two_measurements(family: str) -> None:
    """A measurement is *of a model*, and the model must reach its identity.

    Without this, an implementation that hashed only the evaluation would give
    every contender in a study one shared `MeasurementID` -- so the store would
    hold a single result where there should be one per contender, and the
    second publication would surface as a content conflict rather than as the
    collision it is. Checked across families and across seeds of one family,
    since those are the two ways a study produces many models under one
    protocol.
    """
    problem, training, evaluation = _problem(), _training(), _evaluation()
    riccati = derive_model_id(problem, _contender("riccati"), training, seed=0)
    other = derive_model_id(problem, _contender(family), training, seed=0)
    seeded = derive_model_id(problem, _contender(family), training, seed=1)

    measurements = {
        derive_measurement_id(model, evaluation) for model in (riccati, other, seeded)
    }
    expected = len({riccati, other, seeded})
    assert len(measurements) == expected


def test_declaring_metrics_in_a_different_order_is_one_measurement() -> None:
    """Two spellings of one evaluation must collide rather than diverge.

    The same argument as the contender registry's `resolve()`-then-sign: an
    author who writes the metrics in a different order has not specified a
    different measurement, and paying for a second rollout to discover that
    would be absurd.
    """
    model = derive_model_id(_problem(), _contender(), _training(), seed=0)
    forwards = _evaluation(metrics=("expected_cost", "violation_rate"))
    backwards = _evaluation(metrics=("violation_rate", "expected_cost"))
    assert derive_measurement_id(model, forwards) == derive_measurement_id(
        model, backwards
    )
    # ... while a genuinely different metric set is a different measurement.
    assert derive_measurement_id(model, forwards) != derive_measurement_id(
        model, _evaluation(metrics=("expected_cost",))
    )


def test_a_shifted_evaluation_needs_no_separate_mechanism() -> None:
    """Annex 01 §4.2b, stated as a test: a measurement whose evaluation problem
    differs from the model's training problem *is* a distribution-shift result.
    There is no flag, no subsystem and no second code path -- just a different
    `ProblemSpec` in the evaluation."""
    training_problem = _problem(seed=0)
    model = derive_model_id(training_problem, _contender(), _training(), seed=0)

    nominal = _evaluation(problem=training_problem)
    shifted = _evaluation(problem=_problem(seed=7))

    assert derive_measurement_id(model, nominal) != derive_measurement_id(
        model, shifted
    )
    # Both are measurements OF THE SAME MODEL: nothing retrains.
    assert derive_model_id(training_problem, _contender(), _training(), seed=0) == model


# --------------------------------------------------------------------------
# Structural: what cannot be expressed at all
# --------------------------------------------------------------------------


def test_derive_model_id_has_no_evaluation_parameter() -> None:
    """The structural half, and the stronger one.

    A behavioural test covers the ways of differing this module thought of; the
    absence of a parameter covers every way there will ever be. This is the
    same assertion `tests/store/test_ids.py` makes of `model_id`, carried up to
    the grammar layer that calls it -- because a correct `model_id` reached
    through a `derive_model_id` that accepted an evaluation would be no better
    than the defect it replaces.
    """
    parameters = set(inspect.signature(derive_model_id).parameters)
    assert parameters == {"problem", "contender", "training", "seed"}
    forbidden = {"evaluation", "eval_problem", "eval_protocol", "metrics", "protocol"}
    assert not (parameters & forbidden)


def test_the_evaluation_signature_carries_no_problem() -> None:
    """§9.3b.2: a spec omits exactly what the Tier-4 function takes on its own
    axis. `measurement_id` has an `eval_problem` parameter, so carrying the
    problem in the tree as well would hash it twice."""
    tree = _evaluation().get_signature()
    assert "system" not in _leaves(tree)
    assert "cost" not in _leaves(tree)
    assert "problem" not in tree


def test_the_evaluation_signature_does_carry_its_context() -> None:
    """The same rule, the other way. `measurement_id` has **no** `ctx`
    parameter, so the evaluation context must ride inside the tree -- and it
    must, because scoring at float32 and float64 produces different numbers,
    which is the whole reason D20 gave the evaluation its own context."""
    tree = _evaluation(ctx=_ctx(Precision.FLOAT32)).get_signature()
    assert "precision" in _leaves(tree)
    assert tree["ctx"]["precision"] == "float32"


def test_the_training_and_evaluation_contexts_are_independent() -> None:
    """A contender trained at float64 and scored at float32 is one model with a
    distinct measurement -- not two models."""
    problem, contender = _problem(), _contender()
    model = derive_model_id(
        problem, contender, _training(ctx=_ctx(Precision.FLOAT64)), seed=0
    )
    scored_single = derive_measurement_id(
        model, _evaluation(ctx=_ctx(Precision.FLOAT32))
    )
    scored_double = derive_measurement_id(
        model, _evaluation(ctx=_ctx(Precision.FLOAT64))
    )
    assert scored_single != scored_double
    assert (
        derive_model_id(
            problem, contender, _training(ctx=_ctx(Precision.FLOAT64)), seed=0
        )
        == model
    )


# --------------------------------------------------------------------------
# What DOES move a ModelID — so the tests above cannot pass on a constant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("problem", {"problem": _problem(seed=3)}),
        ("training batch", {"training": _training(batch=BatchPlan(1024))}),
        ("training precision", {"training": _training(ctx=_ctx(Precision.FLOAT32))}),
        ("seed", {"seed": 5}),
    ],
)
def test_the_training_side_does_move_the_model_id(
    name: str, kwargs: dict[str, Any]
) -> None:
    """Without this the split tests would hold for a `derive_model_id` that
    returned a constant."""
    base_args: dict[str, Any] = {
        "problem": _problem(),
        "contender": _contender(),
        "training": _training(),
        "seed": 0,
    }
    assert derive_model_id(**{**base_args, **kwargs}) != derive_model_id(**base_args)


def test_a_different_contender_moves_the_model_id() -> None:
    problem, training = _problem(), _training()
    ids = {
        derive_model_id(problem, _contender(family), training, seed=0)
        for family in ("riccati", "truncated_riccati")
    }
    assert len(ids) == 2


def test_provenance_still_does_not_reach_the_model_id() -> None:
    """D19 survives composition: relabelling how a problem was authored must
    not orphan the models trained on it."""
    contender, training = _contender(), _training()
    assert derive_model_id(
        _problem(generator="hand_specified"), contender, training, seed=0
    ) == derive_model_id(_problem(), contender, training, seed=0)


# --------------------------------------------------------------------------
# The precision obligation §9.2b recorded, discharged here
# --------------------------------------------------------------------------


def test_derive_model_id_refuses_cocp_at_float32() -> None:
    """The first of the two call sites `require_supported_precision` needs.

    `derive_model_id` is the choke point where a contender family and a
    training context are both in hand, which is exactly why the check lives
    here rather than on `TrainingSpec`.
    """
    with pytest.raises(SpecificationError, match="float64"):
        derive_model_id(
            _problem(),
            _contender("cocp"),
            _training(ctx=_ctx(Precision.FLOAT32)),
            seed=0,
        )


def test_derive_model_id_accepts_cocp_at_float64() -> None:
    """The negative control for the check above: a rule that refused everything
    would pass the test above and break every study."""
    identifier = derive_model_id(
        _problem(), _contender("cocp"), _training(ctx=_ctx(Precision.FLOAT64)), seed=0
    )
    assert len(identifier) == 16


@pytest.mark.parametrize("family", EVERY_FAMILY)
@pytest.mark.parametrize("precision", list(Precision))
def test_only_the_declared_families_are_refused(
    family: str, precision: Precision
) -> None:
    """Every family against every precision, because a rule that applies to one
    family is verified for all of them or not at all."""
    from mbl.spec.training import PRECISION_REQUIREMENTS

    required = PRECISION_REQUIREMENTS.get(family)
    args = (_problem(), _contender(family), _training(ctx=_ctx(precision)))
    if required is None or required is precision:
        assert len(derive_model_id(*args, seed=0)) == 16
    else:
        with pytest.raises(SpecificationError):
            derive_model_id(*args, seed=0)


# --------------------------------------------------------------------------
# The types, and the shapes they produce
# --------------------------------------------------------------------------


def test_the_derived_identifiers_are_their_own_types() -> None:
    """A `MeasurementID` passed where a `ModelID` is meant would silently
    associate a measurement with the wrong model; the two are indistinguishable
    as bare strings, which is why Tier 4 subclasses `str`."""
    model = derive_model_id(_problem(), _contender(), _training(), seed=0)
    measurement = derive_measurement_id(model, _evaluation())
    assert isinstance(model, ModelID)
    assert isinstance(measurement, MeasurementID)


def test_the_evaluation_tree_survives_strict_identity_derivation() -> None:
    json.dumps(_evaluation().get_signature(), sort_keys=True)


def test_an_evaluation_spec_is_frozen() -> None:
    evaluation = _evaluation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluation.metrics = ()  # type: ignore[misc]  # the mutation attempt is the test


def test_an_evaluation_needs_at_least_one_metric() -> None:
    """An evaluation that measures nothing is a specification error, not an
    expensive rollout whose result is discarded."""
    with pytest.raises(SpecificationError, match="metric"):
        _evaluation(metrics=())


def test_duplicate_metrics_are_refused() -> None:
    with pytest.raises(SpecificationError, match="metric"):
        _evaluation(metrics=("expected_cost", "expected_cost"))


def test_the_protocol_is_held_structurally() -> None:
    """Tier 3 holds the protocol as `Signable`, which is what keeps `spec/` off
    `experiments/` and `applications/` (plan §9.3b.1)."""
    assert isinstance(_protocol(), Signable)
    assert _evaluation().get_signature()["protocol"] == _protocol().get_signature()


# --------------------------------------------------------------------------
# Renaming a contender never orphans its models (Phase G-1, plan §8.2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_renaming_a_contender_leaves_its_model_id_alone(family: str) -> None:
    """Annex 01 §5 rule 1, one level below the study it was written for.

    Phase B pinned the opposite as today's behaviour and said the phase that
    changed it would decide deliberately; this is that phase. Verified for
    **every** registered family rather than one, because the two that derive a
    label from `kind` are exactly the ones an argument about labels could get
    wrong.
    """
    spec = ContenderSpec(
        family=family, config=FAMILIES[family].config, registry=REGISTRY
    )
    named = replace(spec, label="alpha")
    renamed = replace(spec, label="beta")

    assert named.get_signature() != renamed.get_signature(), (
        "the label must still be visible in the recipe's own tree; the legacy "
        "cache key seven notebooks run on is that tree, and Tier 4 is what "
        "drops the key"
    )
    assert derive_model_id(_problem(), named, _training(), seed=0) == derive_model_id(
        _problem(), renamed, _training(), seed=0
    )


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_exclusion_loses_no_distinction_that_family_relies_on(
    family: str,
) -> None:
    """The §8.2 argument, as a measurement rather than an assertion.

    Dropping `label` is only safe because no family distinguishes two
    configurations by label alone: six default it to a constant, and the two
    that derive one derive it from `kind`, which is signed on its own account.
    So the *default-labelled* spec and an explicitly-labelled one must agree,
    while a genuinely different configuration must not.
    """
    spec = ContenderSpec(
        family=family, config=FAMILIES[family].config, registry=REGISTRY
    )
    default = derive_model_id(_problem(), spec, _training(), seed=0)
    explicit = derive_model_id(
        _problem(), replace(spec, label="anything"), _training(), seed=0
    )
    assert default == explicit

    varied = ContenderSpec(
        family=family,
        config={**FAMILIES[family].config, **FAMILIES[family].perturbation},
        registry=REGISTRY,
    )
    assert derive_model_id(_problem(), varied, _training(), seed=0) != default, (
        "a real configuration change stopped being a different model, so the "
        "equality above holds for the wrong reason"
    )


def test_two_contenders_differing_only_in_kind_are_still_two_models() -> None:
    """The load-bearing half of §8.2's argument, isolated.

    `unfolded` derives its default label from `kind`, so an implementation that
    dropped `label` *and* somehow lost `kind` would collapse two genuinely
    different policies into one identifier. `kind` is signed in its own right;
    this is what says so.
    """
    config = dict(FAMILIES["unfolded"].config)
    alpha = ContenderSpec(family="unfolded", config=config, registry=REGISTRY)
    alpha_p = ContenderSpec(
        family="unfolded",
        config={**config, "kind": UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX},
        registry=REGISTRY,
    )
    assert alpha.resolve().label != alpha_p.resolve().label, (
        "this family no longer derives its label from `kind`, so this test is "
        "no longer about what it says it is"
    )
    assert derive_model_id(_problem(), alpha, _training(), seed=0) != derive_model_id(
        _problem(), alpha_p, _training(), seed=0
    )
