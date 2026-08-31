"""Acceptance tests for the training distribution — Stage 2 Phase F1.

Written before the implementation. The checkpoint class is **negative control**,
and it closes a defect that has been live since Phase C: `TrainingSpec` had no
training distribution at all, so **two studies training under different
process-noise levels derived the same `ModelID`**.

That is worse than it sounds. The legacy key this grammar replaces *does*
distinguish them — the training batch spec reaches it through `evaluation` —
so the identity split as built had lost a term the defective thing it replaces
still carries. And a producer had nowhere to read training data from at all,
which is why this had to be closed before Phase F2 rather than with it.

The two directions are asserted together and neither is sufficient alone:

* changing the **training** distribution must move `ModelID` — the defect;
* changing the **evaluation** distribution must still leave every `ModelID`
  untouched — the thing the whole re-architecture exists for, and the property
  a careless fix would break by putting the distribution somewhere shared.

Annex 01 §2.3's `DataSpec` deliberately carries **no batch size**. The count
lives in `BatchPlan`, and a test here pins the reason: a tier writing
`training.batch.effective_size` rebuilds the `TrainingSpec`, so a consistency
guard between two homes for one quantity would fire on every tier application.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.identity import derive_measurement_id, derive_model_id
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE
from mbl.spec.training import BatchPlan, TrainingSpec

from .test_contender import FAMILIES, REGISTRY
from .test_study import _contender, _protocol

EVERY_FAMILY = sorted(FAMILIES)
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
            control_bound=0.5,
        )
    )


def _training(**overrides: Any) -> TrainingSpec:
    fields: dict[str, Any] = {
        "data": DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
        "plan": TrainingPlan(
            optimizer=OptimizerSpec(name="adam", learning_rate=1e-3), epochs=3
        ),
        "batch": BatchPlan(effective_size=256),
        "ctx": CTX,
    }
    fields.update(overrides)
    return TrainingSpec(**fields)


def _model(family: str, **overrides: Any) -> str:
    return str(
        derive_model_id(
            _problem(), _contender(family, family), _training(**overrides), seed=0
        )
    )


# --------------------------------------------------------------------------
# THE CHECKPOINT — the term the split had lost
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_training_noise_level_is_part_of_the_model(family: str) -> None:
    """The defect, closed. A model fitted on trajectories with twice the process
    noise is a different model; before this phase the two shared a `ModelID`,
    so the second would silently be served the first from the store.

    Every family, not one: a distribution reaching identity for the unfolded
    contenders and not for COCP would be a store that is right about some of
    its contents.
    """
    quiet = _model(family)
    loud = _model(
        family,
        data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 1.0}),
    )
    assert quiet != loud


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_training_stream_is_part_of_the_model(family: str) -> None:
    """The seed of the training distribution is not the training seed. One
    picks the trajectories, the other initialises the weights, and two models
    fitted on different draws are two models."""
    assert _model(family) != _model(
        family,
        data=DataSpec(kind="gaussian", seed=99, params={"process_noise_std": 0.5}),
    )


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_distribution_family_is_part_of_the_model(family: str) -> None:
    assert _model(family) != _model(
        family, data=DataSpec(kind="laplace", seed=1, params={"process_noise_std": 0.5})
    )


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_evaluation_distribution_still_reaches_no_model(family: str) -> None:
    """The other direction, and the one the whole re-architecture exists for. A
    fix that put the training distribution somewhere both specs share would
    close the defect above and reopen the one Stage 2 was written to close."""
    model = derive_model_id(_problem(), _contender(family, family), _training(), seed=0)
    measurements = {
        str(
            derive_measurement_id(
                model,
                EvaluationSpec(
                    problem=_problem(), protocol=_protocol(seed=seed), ctx=CTX
                ),
            )
        )
        for seed in (0, 7, 11)
    }
    assert len(measurements) == 3, "three evaluations must be three measurements"


def test_the_two_distributions_are_independently_settable() -> None:
    """Parent §2.2's structural fix, as a test: `TrainingSpec.data` and the
    evaluation's own distribution are different objects, so a study can train
    on one and score on another. That is what makes a robust-training study
    ordinary rather than a subsystem."""
    training = _training(
        data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 2.0})
    )
    evaluation = EvaluationSpec(problem=_problem(), protocol=_protocol(seed=5), ctx=CTX)
    model = derive_model_id(_problem(), _contender("unfolded", "u"), training, seed=0)
    assert derive_measurement_id(model, evaluation)
    assert training.data.params["process_noise_std"] == 2.0


# --------------------------------------------------------------------------
# The count has ONE home — and the reason is D15, not taste
# --------------------------------------------------------------------------


def test_the_distribution_carries_no_batch_size() -> None:
    """Structural, because a behavioural test would only cover the spellings
    this module thought of. The count belongs to `BatchPlan`; a second home
    would be free to disagree with the first."""
    import dataclasses

    assert {f.name for f in dataclasses.fields(DataSpec)} == {"kind", "seed", "params"}
    assert "batch_size" not in DataSpec().params


def test_a_tier_can_still_scale_the_batch_of_a_study_that_declares_data() -> None:
    """The regression the rejected design would have caused, pinned as a test.

    Coupling `data.batch_size` to `batch.effective_size` by validation is the
    obvious way to keep one quantity consistent across two homes -- and it is
    incompatible with D15: a tier writing `training.batch.effective_size`
    rebuilds the whole `TrainingSpec`, so the guard would fire on every tier
    application and the whitelist's most-used path would be unusable.
    """
    from mbl.spec.study import StudySpec, SweepAxis

    study = StudySpec(
        id="probe/tiered",
        problem=_problem(),
        contenders=(_contender("unfolded", "unfolded_a"),),
        training=_training(),
        evaluation=EvaluationSpec(problem=_problem(), protocol=_protocol(), ctx=CTX),
        sweep=(
            SweepAxis(
                path="contenders.*.config.num_iterations",
                values=(2, 4),
                applies_to=("unfolded_a",),
            ),
        ),
    )
    resolved = DEFAULT_TIER_CATALOGUE.resolve(study, "publication").study
    assert resolved.training.batch.effective_size == 8192
    assert resolved.training.data == study.training.data, (
        "a tier may scale the batch; it may never touch the distribution"
    )


def test_a_tier_may_not_write_the_training_distribution() -> None:
    """`training.data.*` is content, not effort. The noise level a model was
    trained at is the study's claim, and D15 forbids a tier from editing it --
    the same line §2.3.2 draws between the evaluation batch size, which is
    permitted, and the evaluation noise level, which is not."""
    from mbl.spec.tiers import Tier

    with pytest.raises(SpecificationError) as error:
        Tier(name="cheap", overrides={"training.data.params": {}})
    assert "training.data.params" in str(error.value)


# --------------------------------------------------------------------------
# Validation, and the signature
# --------------------------------------------------------------------------


def test_an_unnamed_distribution_is_refused() -> None:
    with pytest.raises(SpecificationError) as error:
        DataSpec(kind="")
    assert "kind" in str(error.value)


def test_the_signature_survives_the_strict_json_check() -> None:
    """`model_id` refuses anything `json.dumps` cannot represent, and `params`
    is the one field here an author fills freely."""
    spec = _training(
        data=DataSpec(
            kind="gaussian",
            seed=3,
            params={"process_noise_std": 0.5, "initial_state_std": 1.0},
        )
    )
    assert len(_model("unfolded", data=spec.data)) == 16


def test_a_parameter_not_serialisable_is_refused_at_construction() -> None:
    """Better here than at the first `model_id`, which is several frames and one
    phase away from the declaration that caused it."""
    with pytest.raises(SpecificationError) as error:
        DataSpec(params={"sampler": object()})
    assert "sampler" in str(error.value)


def test_parameters_are_sorted_in_the_signature() -> None:
    """Two authors listing the same parameters in a different order must
    collide rather than diverge -- the rule `metrics` and `gates` already
    obey."""
    forward = DataSpec(params={"a": 1.0, "b": 2.0}).get_signature()
    backward = DataSpec(params={"b": 2.0, "a": 1.0}).get_signature()
    assert list(forward["params"]) == list(backward["params"]) == ["a", "b"]


def test_the_registry_is_still_the_real_one() -> None:
    """Anti-vacuity: every identifier above is derived through the live recipe
    registry, so a fixture that stopped resolving would compare empty trees."""
    assert set(FAMILIES) == set(REGISTRY.available())
