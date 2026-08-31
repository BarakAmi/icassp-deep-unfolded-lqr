"""Acceptance tests for `derive_semantic_name` — Stage 2 Phase F3.

Written before the implementation. A content hash is correct and unusable:
nobody recognises `74ad4e8f36bfbb15`. Annex 02 §3 gives every stored record a
deterministic, human-legible name derived from what it is, and until this phase
the producer wrote `semantic_name: ""` — so `mbl models tree` rendered the
problem as its raw `ProblemID` and `—` in the depth and training columns for
every contender, including the three that carry a depth and the three that
train.

**The name is a label, never an identity**, which is why this module sits beside
the grammar rather than in `spec/identity.py`, and why its checkpoint class is
*independent recomputation* rather than negative control: nothing breaks if a
name is wrong, and everything downstream becomes unreadable, so what must be
verified is that the projection reports what the specification actually says.

Two properties carry the phase:

* **Deterministic and lossy in a stated order** — problem, then contender, then
  training, so a truncated display still separates two otherwise similar
  models, and the digest is what guarantees uniqueness. Two models differing
  only in something the name omits must share a name and differ in the digest.
* **Every descriptor comes from the specification**, never from a table this
  module keeps. Reading a family's own keys (`kind`, `num_iterations`) is
  permitted *here* precisely because a miss costs description and not
  correctness — the distinction the plan's §F3 states, against three earlier
  phases where the same move was rejected.
"""

from __future__ import annotations

import re
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
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.naming import derive_semantic_name
from mbl.spec.problem import GeneratorProvenance, ProblemData, ProblemSpec
from mbl.spec.study import StudySpec
from mbl.spec.training import BatchPlan, TrainingSpec
from mbl.store.naming import DIGEST_SEPARATOR, EMPTY_SEGMENT, split_semantic_name

from .test_contender import FAMILIES, REGISTRY

N, M, HORIZON = 4, 2, 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)


def _problem(*, provenance: bool = True, u_max: float | None = 0.5) -> ProblemSpec:
    rng = np.random.default_rng(0)
    return ProblemSpec(
        data=ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
            control_bound=u_max,
        ),
        provenance=(
            GeneratorProvenance(generator="LQRProblemFactory", params={"seed": 7})
            if provenance
            else None
        ),
    )


def _protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        batch_spec=GaussianBatchSpec(
            state_dim=N, horizon=HORIZON, batch_size=64, seed=0, process_noise_std=0.5
        ),
        n_batches=2,
    )


def _study(**overrides: Any) -> StudySpec:
    fields: dict[str, Any] = {
        "id": "box_lqr/depth_scaling",
        "problem": _problem(),
        "contenders": (
            ContenderSpec(
                family="unfolded",
                config=FAMILIES["unfolded"].config,
                label="unfolded_alpha",
                registry=REGISTRY,
            ),
            ContenderSpec(
                family="truncated_riccati",
                config=FAMILIES["truncated_riccati"].config,
                label="baseline",
                registry=REGISTRY,
            ),
        ),
        "training": TrainingSpec(
            data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
            plan=TrainingPlan(
                optimizer=OptimizerSpec(name="adam", learning_rate=1e-2), epochs=200
            ),
            batch=BatchPlan(effective_size=8192),
            ctx=CTX,
        ),
        "evaluation": EvaluationSpec(problem=_problem(), protocol=_protocol(), ctx=CTX),
    }
    fields.update(overrides)
    return StudySpec(**fields)


def _point(study: StudySpec, label: str) -> Any:
    return next(
        point
        for point in study.materialise()
        if point.contender.resolved_label == label
    )


def _name(study: StudySpec, label: str) -> str:
    point = _point(study, label)
    return derive_semantic_name(point, study)


def _parts(study: StudySpec, label: str) -> tuple[str, str, str, str]:
    return split_semantic_name(_name(study, label))


# -- the shape ---------------------------------------------------------------


def test_the_name_has_the_annexed_shape() -> None:
    """`problem/contender/training#digest` — asserted through the store's own
    splitter rather than by string surgery here, so a change to the separators
    cannot make this suite disagree with the module that reads names back."""
    problem, contender, training, digest = _parts(_study(), "unfolded_alpha")
    assert problem and contender and training and digest
    assert digest == str(_point(_study(), "unfolded_alpha").model_id)[: len(digest)]


def test_the_problem_segment_reports_what_the_specification_says() -> None:
    """Independent recomputation: every field is read off the study and the
    problem, so the name is checked against the source rather than against a
    string this file also wrote."""
    study = _study()
    problem, _, _, _ = _parts(study, "unfolded_alpha")
    # Asserted WHOLE, not field by field. `box_lqr/depth_scaling` starts with
    # `box_lqr` and its slug's first `-` component is `box_lqr` too, so both a
    # prefix check and a component check hold while the study path leaks into
    # every name as `box_lqr-depth_scaling-n4m2-…`. Only the complete segment,
    # recomputed here from the fixture's own constants, can see that.
    assert problem == f"box_lqr-n{N}m{M}-N{HORIZON}-u0.5-s7", (
        "every field of the problem segment, in the annex's order: the study "
        "id's first path segment, the dimensions, the horizon, the box bound, "
        "and the seed the problem was generated from"
    )


def test_a_problem_with_no_provenance_still_names() -> None:
    """D19 makes provenance optional, so a hand-specified problem has no seed
    to render. The segment must lose that one field and keep the rest, rather
    than falling back to a digest."""
    study = _study(problem=_problem(provenance=False))
    problem, _, _, _ = _parts(study, "unfolded_alpha")
    assert problem.split("-")[0] == "box_lqr" and f"n{N}m{M}" in problem
    # NOT `"s7" not in problem`: `s0` is not `s7` either, so that assertion held
    # while a hand-specified problem was being named as one seeded zero.
    assert not any(re.fullmatch(r"s\d+", part) for part in problem.split("-")), (
        f"{problem!r} carries a seed for a problem that was never generated; a "
        "problem nobody seeded and a problem seeded zero are different things, "
        "and only one of them can be regenerated"
    )


def test_an_unconstrained_problem_omits_the_bound() -> None:
    """`u_max=None` is a problem with no box, and rendering it as `u0` would
    name it as one with a zero bound — a different problem entirely."""
    study = _study(problem=_problem(u_max=None))
    problem, _, _, _ = _parts(study, "unfolded_alpha")
    assert "u" not in problem.split("-")[-1] or "s7" in problem
    assert "u0" not in problem


# -- the contender segment ---------------------------------------------------


def test_the_contender_segment_carries_kind_and_depth() -> None:
    """Both are read from the *resolved* recipe, so a default that the study
    never spelled out still appears."""
    _, contender, _, _ = _parts(_study(), "unfolded_alpha")
    assert contender.startswith("unfolded")
    assert "learned_step_size" in contender, (
        "the kind is slugged as declared; abbreviating it to `a`/`aP` would "
        "need a per-family table, which is a second source of truth"
    )
    assert contender.endswith("J3"), "the depth is `num_iterations`"


def test_a_contender_with_neither_kind_nor_depth_names_from_its_family() -> None:
    """Three of NB04's six carry no depth and four no kind. A missing key costs
    description and never correctness, which is the whole reason reading a
    family's own keys is permitted here and was refused in E3, G-2 and G-3."""
    _, contender, _, _ = _parts(_study(), "baseline")
    assert contender == "truncated_riccati"


# -- the training segment ----------------------------------------------------


def test_the_training_segment_reports_the_plan_that_executes() -> None:
    """The contender's own plan, not the study's template: G-2 made the study
    plan a default materialised into the contender, and the name must report
    what the model was actually fitted with."""
    _, _, training, _ = _parts(_study(), "unfolded_alpha")
    assert "adam" in training
    assert "lr1e-3" in training, (
        "the contender's plan learns at 1e-3 and the study template at 1e-2, so "
        "this field alone distinguishes which of the two the name reports"
    )
    assert "ep3" in training, "the contender's own plan runs 3 epochs, not 200"
    assert "b8192" in training
    assert "seed0" in training


def test_a_layerwise_schedule_omits_the_epoch_count() -> None:
    """`LayerwiseTrainingPlan` has `warmup_epochs_per_layer` and
    `refinement_epochs` and no `epochs` at all. Summing them would invent a
    number the specification does not state; the field is omitted instead."""
    study = _study(
        contenders=(
            ContenderSpec(
                family="unfolded_warmstart",
                config=FAMILIES["unfolded_warmstart"].config,
                label="flagship",
                registry=REGISTRY,
            ),
        )
    )
    _, _, training, _ = _parts(study, "flagship")
    assert "adam" in training and "seed0" in training
    assert "ep" not in training


def test_an_analytic_contender_still_reports_its_seed() -> None:
    """It has no fitting procedure, and its `ModelID` still has a seed axis, so
    the segment must not collapse to the empty placeholder — two replicates of
    an analytic contender are two models and must be two names."""
    _, _, training, _ = _parts(_study(), "baseline")
    assert training != EMPTY_SEGMENT
    assert training == "seed0"


def test_two_seeds_are_two_names() -> None:
    """The consequence of the test above, stated as the property that matters."""
    study = _study(
        training=replace(_study().training, seeds=(0, 1)),
    )
    names = {
        derive_semantic_name(point, study)
        for point in study.materialise()
        if point.contender.resolved_label == "baseline"
    }
    assert len(names) == 2


# -- determinism, and what the digest is for ---------------------------------


def test_the_name_is_deterministic() -> None:
    """It can be typed and predicted, which is the point of having one."""
    assert _name(_study(), "unfolded_alpha") == _name(_study(), "unfolded_alpha")


def test_two_models_the_name_cannot_separate_share_it_and_differ_in_the_digest() -> (
    None
):
    """Annex 02 §3's third property, and the one that makes the other two safe.

    The training *distribution* is signed into `ModelID` and appears nowhere in
    a name, so two studies differing only in their process-noise level are two
    models with one name — which is correct, and is why the name is never the
    identity.
    """
    quiet = _study()
    loud = _study(
        training=replace(
            quiet.training,
            data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 5.0}),
        )
    )
    quiet_name = _name(quiet, "unfolded_alpha")
    loud_name = _name(loud, "unfolded_alpha")

    assert quiet_name != loud_name, "the digest must separate them"
    assert quiet_name.split(DIGEST_SEPARATOR)[0] == loud_name.split(DIGEST_SEPARATOR)[0]
    assert (
        _point(quiet, "unfolded_alpha").model_id
        != _point(loud, "unfolded_alpha").model_id
    )


def test_the_name_is_never_the_identity() -> None:
    """Structural: `derive_semantic_name` is not in `spec.identity`, and the
    store's own naming module opens by saying a name is a label. A projection
    filed beside `derive_model_id` invites exactly the confusion that costs a
    store its meaning."""
    import mbl.spec.identity as identity

    assert not hasattr(identity, "derive_semantic_name")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_registered_family_names(family: str) -> None:
    """Coverage derived from the registry, so a family added later cannot drift
    out while this suite stays green — the risk being precisely the family
    whose config shape nobody checked."""
    study = _study(
        contenders=(
            ContenderSpec(
                family=family,
                config=FAMILIES[family].config,
                label=f"probe_{family}",
                registry=REGISTRY,
            ),
        )
    )
    problem, contender, training, digest = _parts(study, f"probe_{family}")
    assert problem and contender and training and len(digest) >= 6
    assert contender.startswith(family)


def test_no_family_defaults_a_name_bearing_key() -> None:
    """Why reading the *resolved* recipe is currently indistinguishable from
    reading the raw config — measured, and pinned so it cannot change silently.

    `_contender_name` reads `kind` and `num_iterations` out of the resolved
    signature so that a default the study never spelled out still appears. No
    registered family defaults either: every family that has them requires
    them. So the two sources agree today and a mutant swapping one for the
    other survives — an equivalent mutant by construction, not a weak test.

    Keeping the signature read is still right, because it is the source that
    stays correct when a family later defaults one. This test is what makes
    that day visible instead of silent.
    """
    import dataclasses

    defaulted = {
        f"{family}.{field.name}"
        for family in FAMILIES
        for field in dataclasses.fields(REGISTRY.get(family))
        if field.name in {"kind", "num_iterations"}
        and field.default is not dataclasses.MISSING
    }
    assert not defaulted, (
        f"{sorted(defaulted)} now default a key the semantic name renders, so "
        "reading the resolved recipe and reading the raw config have stopped "
        "agreeing. That is the case `_contender_name` was written for -- add a "
        "test that a contender omitting it is still named with it"
    )
