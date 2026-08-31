"""Acceptance tests for `TrainingSpec` and `BatchPlan` — Stage 2 Phase C, D20.

Written before the implementation. The checkpoint class is **negative control
over every instance**: D20 draws a line through the middle of the training
declaration, and a line is only worth drawing if both sides of it are checked.

D20, restated as the three things that must hold:

* **Effective batch size is scientific content.** Change it and you have a
  different model.
* **The microbatch is not.** It is chosen by the runner from whatever memory is
  available, so two hosts that split the same effective batch differently must
  agree on the identifier. If they did not, a tier would have to be re-tuned per
  problem size and D15's rule that a tier never changes scientific content would
  fall.
* **Precision is content, and COCP requires float64.** A specification pairing
  that family with float32 must be refused, not quietly run at the wrong
  precision.

Every claim is checked for **every family the recipe registry knows**, with the
family list derived *from* the registry, because the risk is precisely the
family nobody thought about. Two of the three are also checked *structurally* —
`microbatch` must appear nowhere in the signature tree, and `seeds` must appear
under exactly the name Tier 4 drops — since a behavioural test cannot see a key
that was never emitted at all, and a defaulted value shifts every derived
identifier uniformly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.core.utils.signing import Signable
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.spec.contender import ContenderSpec
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.training import (
    PRECISION_REQUIREMENTS,
    BatchPlan,
    TrainingSpec,
    require_supported_precision,
)
from mbl.store.ids import TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID, model_id

from .test_contender import FAMILIES, REGISTRY

EVERY_FAMILY = sorted(FAMILIES)

_OPTIMIZER = OptimizerSpec(name="adam", learning_rate=1e-3)
PLAN = TrainingPlan(optimizer=_OPTIMIZER, epochs=3)
LAYERWISE = LayerwiseTrainingPlan(
    optimizer=_OPTIMIZER, warmup_epochs_per_layer=2, refinement_epochs=1
)

#: A `ProblemID`-shaped stand-in. This suite is about the training leg; a real
#: problem would add a second failure mode to every assertion about the first.
STAND_IN_PROBLEM_ID = "0" * 16


def _ctx(precision: Precision = Precision.FLOAT64) -> ComputeContext:
    return ComputeContext(backend=Backend.TORCH, device="cpu", precision=precision)


def _training(**overrides: Any) -> TrainingSpec:
    fields: dict[str, Any] = {
        "data": DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
        "plan": PLAN,
        "batch": BatchPlan(effective_size=256),
        "ctx": _ctx(),
    }
    fields.update(overrides)
    return TrainingSpec(**fields)


def _model_id(family: str, training: TrainingSpec, seed: int = 0) -> str:
    """The identifier a contender/training pair derives, through Tier 4."""
    contender = ContenderSpec(
        family=family, config=FAMILIES[family].config, registry=REGISTRY
    )
    return model_id(
        problem=STAND_IN_PROBLEM_ID,
        contender=contender.get_signature(),
        training=training.get_signature(),
        ctx=training.ctx.get_signature(),
        seed=seed,
    )


def _leaves(tree: Any) -> list[str]:
    """Every key anywhere in a nested signature tree."""
    if isinstance(tree, dict):
        return [key for k, v in tree.items() for key in (k, *_leaves(v))]
    if isinstance(tree, (list, tuple)):
        return [key for item in tree for key in _leaves(item)]
    return []


# --------------------------------------------------------------------------
# D20, the resource half: the microbatch is not scientific content
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
@pytest.mark.parametrize("microbatch", [None, 32, 64, 128, 256])
def test_the_microbatch_never_reaches_the_identifier(
    family: str, microbatch: int | None
) -> None:
    """Every split of one effective batch is the same model.

    Parametrised over splits that genuinely differ — 8, 4, 2 and 1
    accumulation steps, plus "the runner decides" — rather than over one
    specimen, because a single alternative cannot distinguish "the microbatch
    is excluded" from "the microbatch happens not to matter here".
    """
    baseline = _training(batch=BatchPlan(effective_size=256))
    split = _training(batch=BatchPlan(effective_size=256, microbatch=microbatch))
    assert _model_id(family, split) == _model_id(family, baseline)


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_effective_batch_size_does_reach_the_identifier(family: str) -> None:
    """The other half of the line. Without this, the test above would hold for
    an implementation that dropped the batch declaration entirely."""
    small = _training(batch=BatchPlan(effective_size=256))
    large = _training(batch=BatchPlan(effective_size=512))
    assert _model_id(family, small) != _model_id(family, large)


def test_the_microbatch_appears_nowhere_in_the_signature_tree() -> None:
    """Structural, because behaviour cannot see an omitted key.

    A microbatch that entered the tree with a *constant* value would shift
    every derived identifier uniformly, so the behavioural test above would
    still pass while D20 was broken for anyone who later varied it.
    """
    tree = _training(batch=BatchPlan(effective_size=256, microbatch=32)).get_signature()
    assert "microbatch" not in _leaves(tree)
    assert "accumulation_steps" not in _leaves(tree)
    assert "effective_size" in _leaves(tree)


def test_accumulation_steps_covers_the_batch_without_exceeding_it() -> None:
    """`ceil`, not `floor`: a plan that dropped the remainder would silently
    train on fewer trajectories than the specification declares."""
    assert BatchPlan(effective_size=256, microbatch=64).accumulation_steps == 4
    assert BatchPlan(effective_size=256, microbatch=100).accumulation_steps == 3
    assert BatchPlan(effective_size=256, microbatch=256).accumulation_steps == 1
    assert BatchPlan(effective_size=256, microbatch=512).accumulation_steps == 1
    # `None` means the runner chooses, which is one whole-batch pass.
    assert BatchPlan(effective_size=256).accumulation_steps == 1


@pytest.mark.parametrize(
    ("effective", "microbatch"),
    [(0, None), (-1, None), (256, 0), (256, -8)],
)
def test_a_batch_plan_refuses_nonsense(effective: int, microbatch: int | None) -> None:
    with pytest.raises(SpecificationError):
        BatchPlan(effective_size=effective, microbatch=microbatch)


# --------------------------------------------------------------------------
# Precision is content, and COCP requires float64
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_precision_changes_the_identifier(family: str) -> None:
    """A float32 model and a float64 model are different models, correctly:
    precision genuinely changes the trained weights."""
    single = _training(ctx=_ctx(Precision.FLOAT32))
    double = _training(ctx=_ctx(Precision.FLOAT64))
    assert _model_id(family, single) != _model_id(family, double)


@pytest.mark.parametrize("family", EVERY_FAMILY)
@pytest.mark.parametrize("precision", list(Precision))
def test_every_family_is_checked_against_its_precision_requirement(
    family: str, precision: Precision
) -> None:
    """The rule applies to every family, so it is verified for every family.

    A family with no declared requirement must accept both precisions; a family
    with one must accept exactly that precision and refuse the other, loudly
    enough that the message names what to change.
    """
    required = PRECISION_REQUIREMENTS.get(family)
    if required is None or required is precision:
        require_supported_precision(family, precision)
        return
    with pytest.raises(SpecificationError) as error:
        require_supported_precision(family, precision)
    message = str(error.value)
    assert family in message
    assert required.value in message
    assert precision.value in message


def test_the_cocp_family_is_the_one_that_requires_float64() -> None:
    """The requirement table is the point of the rule, so it is pinned rather
    than merely consulted: COCP's solver is ill-conditioned at float32, and a
    table that quietly emptied would make every check above vacuous."""
    assert PRECISION_REQUIREMENTS["cocp"] is Precision.FLOAT64
    assert set(PRECISION_REQUIREMENTS) <= set(REGISTRY.available())


def test_the_precision_requirement_table_names_only_real_families() -> None:
    """A misspelled family name would silently require nothing of anybody."""
    unknown = set(PRECISION_REQUIREMENTS) - set(REGISTRY.available())
    assert not unknown, f"requirements declared for unregistered families: {unknown}"


# --------------------------------------------------------------------------
# Seeds: declared as a collection, dropped from the model identifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_declaring_more_seeds_does_not_change_an_existing_model(
    family: str,
) -> None:
    """A five-seed study is five models; adding a sixth must train one, not
    six. The declared *collection* therefore cannot appear in any individual
    model's identity."""
    one = _training(seeds=(0,))
    many = _training(seeds=(0, 1, 2, 3, 4, 5))
    assert _model_id(family, many, seed=0) == _model_id(family, one, seed=0)


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_individual_seed_does_change_the_identifier(family: str) -> None:
    """The other half: two seeds of one study are two models."""
    training = _training(seeds=(0, 1))
    assert _model_id(family, training, seed=0) != _model_id(family, training, seed=1)


def test_seeds_are_emitted_under_the_name_tier_4_drops() -> None:
    """Structural, and it is a contract between two modules.

    `model_id` strips `TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID` from the training
    tree by name. Emitting the collection under any other key would leave it in
    the identifier, and the behavioural test above would still pass on the day
    someone renamed the field.
    """
    tree = _training(seeds=(0, 1, 2)).get_signature()
    for key in TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID:
        assert key in tree, f"{key!r} is stripped by model_id but never emitted"
    assert tree["seeds"] == [0, 1, 2]


# --------------------------------------------------------------------------
# The shape of the specification itself
# --------------------------------------------------------------------------


def test_the_compute_context_is_not_hashed_twice() -> None:
    """`model_id` takes `ctx` on its own axis, so a training tree carrying it
    too would give one value two homes that could disagree (plan §9.2b)."""
    tree = _training(ctx=_ctx(Precision.FLOAT32)).get_signature()
    assert "backend" not in _leaves(tree)
    assert "precision" not in _leaves(tree)


def test_a_layerwise_plan_is_accepted_through_the_protocol() -> None:
    """Tier 3 holds the plan as `Signable`, not as a concrete type — which is
    what keeps `spec/` off `engine/`, and is also the only way both plan types
    fit: they share no concrete field.

    What the tier needs of a plan changed in Phase G-2: it hands the plan out
    rather than signing it, so the two trees below are now *equal*. `Signable`
    is kept because it remains the narrowest structural description of an
    object this tier may not construct — and because the object still has to be
    signable, inside the contender it lands in.
    """
    assert isinstance(PLAN, Signable)
    assert isinstance(LAYERWISE, Signable)
    assert (
        _training(plan=PLAN).get_signature()
        == _training(plan=LAYERWISE).get_signature()
    )


def test_the_study_plan_is_a_default_and_never_identity() -> None:
    """G-2, and the reason it is a re-baseline.

    `TrainableRecipe.build_engine` trains for the plan in the *contender*.
    `TrainingSpec.plan` was signed anyway, so two studies differing only in a
    template that governs nothing derived different `ModelID`s -- two
    identifiers for one model, in a store whose whole premise is that they are
    the same thing. It is a default now (Annex 01 §2.3.4), materialised into
    each contender, and identity comes from the effective plan once.
    """
    tree = _training().get_signature()
    assert "plan" not in tree, (
        "the study's template is in the training signature, so it is in every "
        "ModelID derived from it"
    )
    assert "plan" not in _leaves(tree), "it survives one level down"


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_two_templates_derive_one_model(family: str) -> None:
    """The negative control, over every registered family. Five epochs and five
    hundred must be the same model when the contender says what it trains
    for -- which every contender does, after materialisation."""
    five = _training(plan=TrainingPlan(optimizer=_OPTIMIZER, epochs=5))
    five_hundred = _training(plan=TrainingPlan(optimizer=_OPTIMIZER, epochs=500))
    assert five.plan is not five_hundred.plan, "the fixture varies nothing"
    assert _model_id(family, five) == _model_id(family, five_hundred)


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_training_tree_survives_strict_identity_derivation(family: str) -> None:
    """`model_id` refuses anything `json.dumps` cannot represent, deliberately.
    A tuple of seeds, an enum in a plan, or a stray numpy scalar would each
    break identity derivation at the first call rather than at review."""
    json.dumps(_training().get_signature(), sort_keys=True)
    assert len(_model_id(family, _training())) == 16


def test_a_training_spec_is_frozen() -> None:
    import dataclasses

    training = _training()
    with pytest.raises(dataclasses.FrozenInstanceError):
        training.seeds = (9,)  # type: ignore[misc]  # the mutation attempt is the test


def test_an_empty_seed_collection_is_refused() -> None:
    """A study that declares no seeds trains nothing, which is a specification
    error rather than a silently empty run."""
    with pytest.raises(SpecificationError, match="seed"):
        _training(seeds=())


def test_duplicate_seeds_are_refused() -> None:
    """Two identical seeds are one model published twice — which the store
    would surface as a content conflict, far from the declaration that caused
    it."""
    with pytest.raises(SpecificationError, match="seed"):
        _training(seeds=(0, 1, 0))
