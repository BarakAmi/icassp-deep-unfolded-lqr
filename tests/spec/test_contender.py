"""Acceptance tests for `ContenderSpec` — Stage 2 Phase B.

Written before the implementation. The checkpoint class is **independent
recomputation**: the grammar must *re-express* today's contenders, never
redefine them, so every claim below is checked against what the recipe registry
produces today rather than against a tree this module writes down.

Three properties carry the phase, and each is verified for **every family the
registry knows**, with the family list derived *from* the registry so a family
added later cannot drift out of coverage while the suite stays green:

* the signature tree equals the resolved recipe's own `get_signature()`;
* a default spelled out explicitly signs identically to the same default
  omitted -- the `resolve()`-then-sign behaviour, which is the whole reason two
  spellings of one effective configuration share a cache identity;
* the tree survives `model_id`'s strict JSON check, which refuses anything
  `json.dumps` cannot represent. Recipe signatures carry enum members and pass
  today only because all three of them are `StrEnum`s.

The registry itself is **injected**, never imported by Tier 3 (plan §8.1): the
grammar may not reach into `applications`/`models`, and a deferred import would
reintroduce that dependency while hiding it from the static boundary test.
`experiments.ContenderSpec` remains as the Tier-4 binding that supplies the
default registry, and is asserted here to be the same value.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from mbl.applications.recipes.base import build_default_recipe_registry
from mbl.applications.recipes.unfolded import UnfoldedKind
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import ContenderSpec as BoundContenderSpec
from mbl.spec.contender import ContenderSpec, Role
from mbl.spec.errors import SpecificationError
from mbl.store.ids import model_id

REGISTRY = build_default_recipe_registry()

_OPTIMIZER = OptimizerSpec(name="adam", learning_rate=1e-3)
_PLAN = TrainingPlan(optimizer=_OPTIMIZER, epochs=3)
_SCHEDULE = LayerwiseTrainingPlan(
    optimizer=_OPTIMIZER, warmup_epochs_per_layer=2, refinement_epochs=1
)

#: A `ProblemID`-shaped stand-in: this suite tests the contender leg of
#: `model_id`, and a real problem would add a second failure mode to every
#: assertion about the first.
STAND_IN_PROBLEM_ID = "0" * 16


@dataclass(frozen=True)
class Family:
    """One registered family, with enough to exercise it.

    Attributes:
        config: The minimal configuration -- every required field, nothing
            defaulted, so `test_spelling_a_default_signs_identically` has
            defaults left to spell.
        perturbation: One *scientific* field changed away from what `config`
            resolves to. Asserted to be a real change, so the negative
            direction cannot pass vacuously.
    """

    config: dict[str, Any]
    perturbation: dict[str, Any] = field(default_factory=dict)


#: Every family the recipe registry knows. `test_the_family_table_is_the
#: _registry` asserts this *is* the registry's own list rather than a sample --
#: the risk being precisely the family whose defaults changed.
FAMILIES: dict[str, Family] = {
    "riccati": Family({"horizon": 6}, {"horizon": 7}),
    "truncated_riccati": Family({"horizon": 6}, {"horizon": 7}),
    "cocp": Family({"plan": _PLAN}, {"solver_eps": 1e-6}),
    "cocp_exact": Family({"plan": _PLAN}, {"use_linear_term": True}),
    "cocp_exact_lower_bound": Family(
        {"process_noise_std": 0.1}, {"process_noise_std": 0.25}
    ),
    "cocp_lower_bound": Family({"process_noise_std": 0.1}, {"process_noise_std": 0.25}),
    "neural": Family({"hidden_dim": 8, "plan": _PLAN}, {"hidden_dim": 16}),
    "unfolded": Family(
        {
            "kind": UnfoldedKind.LEARNED_STEP_SIZE,
            "plan": _PLAN,
            "num_iterations": 3,
            "step_size_init": 0.1,
            "step_size_max": 1.0,
            "horizon": 6,
        },
        {"num_iterations": 5},
    ),
    "unfolded_fixed": Family(
        {
            "num_iterations": 3,
            "step_size_init": 0.1,
            "step_size_max": 1.0,
            "horizon": 6,
        },
        {"step_size_init": 0.25},
    ),
    "unfolded_warmstart": Family(
        {
            "kind": UnfoldedKind.LEARNED_STEP_SIZE,
            "schedule": _SCHEDULE,
            "num_iterations": 3,
            "step_size_init": 0.1,
            "step_size_max": 1.0,
            "horizon": 6,
        },
        {"num_iterations": 5},
    ),
}

EVERY_FAMILY = sorted(FAMILIES)


def _spec(family: str, **overrides: Any) -> ContenderSpec:
    """A Tier-3 spec for `family` with the registry injected."""
    config = {**FAMILIES[family].config, **overrides.pop("config", {})}
    return ContenderSpec(family=family, config=config, registry=REGISTRY, **overrides)


def _recipe_directly(family: str, **extra: Any) -> Any:
    """The expected recipe, built through the registry the way today's code
    does -- the independent half of `independent recomputation`."""
    return REGISTRY.create(family, **{**FAMILIES[family].config, **extra})


def _defaults_left_unspelled(family: str) -> dict[str, Any]:
    """Every defaulted field of `family`'s recipe that `config` omits, at its
    declared default value."""
    recipe_cls = REGISTRY.get(family)
    declared = FAMILIES[family].config
    spelled: dict[str, Any] = {}
    for recipe_field in dataclasses.fields(recipe_cls):
        if recipe_field.name in declared:
            continue
        if recipe_field.default is not dataclasses.MISSING:
            spelled[recipe_field.name] = recipe_field.default
        elif recipe_field.default_factory is not dataclasses.MISSING:
            spelled[recipe_field.name] = recipe_field.default_factory()
    return spelled


# --------------------------------------------------------------------------
# The parametrisation is itself under test
# --------------------------------------------------------------------------


def test_the_family_table_is_the_registry() -> None:
    """The coverage guard. Every test below parametrises over `FAMILIES`, so a
    family registered later would be silently unverified -- and it is exactly
    the newest family, whose defaults nobody has checked, that this phase is
    most likely to get wrong."""
    assert set(FAMILIES) == set(REGISTRY.available())


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_each_family_leaves_a_default_unspelled(family: str) -> None:
    """Negative control for `test_spelling_a_default_signs_identically`: if a
    family's `config` already named every defaulted field, that test would
    compare a spec against itself and could never fail."""
    assert _defaults_left_unspelled(family), (
        f"{family}'s config spells every default, so the resolve()-then-sign "
        "test degenerates to an identity"
    )


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_each_perturbation_is_a_real_change(family: str) -> None:
    """Negative control for `test_changing_a_configured_value_changes_the
    _signature`: a perturbation equal to the resolved value would make that
    test pass no matter what the implementation did."""
    recipe = _recipe_directly(family)
    for name, value in FAMILIES[family].perturbation.items():
        assert getattr(recipe, name) != value, (
            f"{family}.{name} already resolves to {value!r}"
        )


# --------------------------------------------------------------------------
# THE CHECKPOINT — independent recomputation, over every family
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_signature_equals_the_resolved_recipes_own(family: str) -> None:
    """The grammar re-expresses today's contenders rather than redefining
    them. Compared against the recipe built straight through the registry --
    not against a tree written down here, which would only assert that this
    module and the implementation agree."""
    assert _spec(family).get_signature() == _recipe_directly(family).get_signature()


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_a_label_override_reaches_the_resolved_recipe(family: str) -> None:
    """The one place `resolve()` does work beyond forwarding: the override has
    to arrive as a constructor argument, so that a family deriving its label in
    `__post_init__` still sees it."""
    spec = _spec(family, label="renamed")
    assert (
        spec.get_signature()
        == _recipe_directly(family, label="renamed").get_signature()
    )
    assert spec.resolved_label == "renamed"
    assert spec.resolve().label == "renamed"


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_spelling_a_default_signs_identically(family: str) -> None:
    """Two spellings of one effective configuration share one identity -- the
    entire purpose of resolving before signing. Every default the family has is
    spelled at once, not one representative."""
    implicit = _spec(family)
    explicit = _spec(family, config=_defaults_left_unspelled(family))
    assert implicit.get_signature() == explicit.get_signature()


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_changing_a_configured_value_changes_the_signature(family: str) -> None:
    """Without this the equality tests above would hold for an implementation
    that returned a constant."""
    perturbed = _spec(family, config=FAMILIES[family].perturbation)
    assert perturbed.get_signature() != _spec(family).get_signature()


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_signature_survives_strict_identity_derivation(family: str) -> None:
    """`model_id` refuses a tree `json.dumps` cannot represent, deliberately:
    `compute_signature_digest` passes `default=str` and would otherwise hash an
    object's address. Recipe signatures carry `UnfoldedKind`,
    `ControlInitMethod` and `SequenceModelType` members and pass only because
    all three are `StrEnum`s -- a family added with a plain `Enum`, a `Path` or
    a NumPy scalar breaks identity derivation, and Phase F is too late to find
    that out."""
    signature = _spec(family).get_signature()
    json.dumps(signature, sort_keys=True)  # the same refusal, stated locally
    identifier = model_id(
        problem=STAND_IN_PROBLEM_ID,
        contender=signature,
        training={},
        ctx={},
        seed=0,
    )
    assert len(identifier) == 16


# --------------------------------------------------------------------------
# What is NOT in the signature, and what is
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
@pytest.mark.parametrize("role", list(Role))
def test_role_never_reaches_the_signature(family: str, role: Role) -> None:
    """`role` classifies a contender for presentation -- policy against bound,
    baseline against reference. Reclassifying one must not retrain it. Every
    role is checked against every family, because a rule that applies N times
    is verified N times."""
    assert _spec(family, role=role).get_signature() == _spec(family).get_signature()
    assert "role" not in _spec(family, role=role).get_signature()


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_renaming_a_contender_changes_its_signature_tree(family: str) -> None:
    """`label` stays in the recipe's own tree, and that is deliberate.

    Phase B pinned this and said the phase that changed it would decide
    deliberately. **Phase G-1 decided**, and it decided one tier up: `label` is
    dropped by `model_id`, by name, exactly as `seeds` and a study's `id` are
    (`CONTENDER_KEYS_EXCLUDED_FROM_MODEL_ID`) -- so renaming a contender no
    longer orphans its models, which is what §8.2 asked for.

    The *tree* keeps its label, which this still pins, and the reason is
    unchanged: `Experiment.contender_content_digest` is that tree, seven
    notebooks still run on it, and Phase B's checkpoint is equality with the
    trees they already have. Excluding it here rather than in Tier 4 would
    re-key every legacy cache entry for a rename this stage exists to make
    free. See `tests/spec/test_identity.py` for the `ModelID` half."""
    assert (
        _spec(family, label="renamed").get_signature() != _spec(family).get_signature()
    )


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_registry_is_not_part_of_what_a_contender_is(family: str) -> None:
    """An injected collaborator, not content: two specs differing only in which
    registry object resolved them are the same specification, and a registry in
    the `repr` would make every failure message unreadable."""
    config = FAMILIES[family].config
    injected = ContenderSpec(family=family, config=config, registry=REGISTRY)
    bare = ContenderSpec(family=family, config=config)
    assert injected == bare
    assert "ControllerRegistry" not in repr(injected)


# --------------------------------------------------------------------------
# Resolution: precedence, and the errors when it cannot happen
# --------------------------------------------------------------------------


def test_from_recipe_wraps_a_composed_recipe() -> None:
    """The bypass path a `CaseStudy` declaration uses: no registry involved,
    and the spec resolves to that exact object."""
    recipe = _recipe_directly("riccati")
    spec = ContenderSpec.from_recipe(recipe)
    assert spec.resolve() is recipe
    assert spec.get_signature() == recipe.get_signature()
    assert spec.family == "riccati"


def test_an_explicit_registry_argument_wins() -> None:
    """The precedence contract: an argument at the call site beats the field,
    so a caller holding a different registry never has to rebuild the spec."""

    class _Rejecting:
        def create(self, name: str, **kwargs: Any) -> Any:
            raise AssertionError("the injected registry should not be consulted")

    spec = ContenderSpec(family="riccati", config={"horizon": 6}, registry=_Rejecting())
    assert spec.resolve(REGISTRY).horizon == 6


def test_a_composed_recipe_beats_every_registry() -> None:
    """`from_recipe` means *this* object, not "resolve the family again" -- so
    no registry may substitute another for it, whether injected or passed at
    the call site. The Tier-4 binding below always passes one, which is exactly
    the path that would re-resolve a `CaseStudy`'s recipes into constructor
    errors if this precedence were the other way round."""
    recipe = _recipe_directly("riccati", label="explicit")
    spec = ContenderSpec(
        family="riccati", config={"horizon": 99}, recipe=recipe, registry=REGISTRY
    )
    assert spec.resolve() is recipe
    assert spec.resolve(REGISTRY) is recipe
    assert BoundContenderSpec.from_recipe(recipe).resolve() is recipe


def test_without_a_registry_it_says_which_field_is_missing() -> None:
    """Tier 3 has no default registry to fall back on -- importing one is the
    dependency this tier exists to avoid -- so the failure has to name the way
    out rather than surfacing as an `AttributeError` on `None`."""
    spec = ContenderSpec(family="riccati", config={"horizon": 6})
    with pytest.raises(SpecificationError, match="registry"):
        spec.resolve()


def test_an_unknown_family_names_the_available_ones() -> None:
    """A misspelled family is the most likely authoring error in a YAML study,
    and a bare `KeyError` from inside the registry does not say what to write
    instead."""
    spec = ContenderSpec(family="ricatti", config={"horizon": 6}, registry=REGISTRY)
    with pytest.raises(SpecificationError) as error:
        spec.resolve()
    assert "ricatti" in str(error.value)
    assert "riccati" in str(error.value)


def test_a_registry_that_builds_something_other_than_a_recipe_is_refused() -> None:
    """A registry is a `name -> constructor` map and cannot promise what it
    builds. The gap is closed at resolution, not at the first
    `get_signature()` call -- which, under a study runner, is an hour in."""

    class _WrongProduct:
        def create(self, name: str, **kwargs: Any) -> Any:
            return 42

    spec = ContenderSpec(family="riccati", registry=_WrongProduct())
    with pytest.raises(SpecificationError, match="not a recipe"):
        spec.resolve()


def test_a_contender_needs_a_family_or_a_recipe() -> None:
    with pytest.raises(SpecificationError, match="family name or a composed recipe"):
        ContenderSpec()


def test_a_contender_is_frozen() -> None:
    spec = _spec("riccati")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.family = "neural"  # type: ignore[misc]  # the mutation attempt is the test


# --------------------------------------------------------------------------
# The Tier-4 binding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_tier_4_binding_supplies_the_default_registry(family: str) -> None:
    """`experiments.ContenderSpec` is the same value with this project's
    registry already in hand, which is what keeps the ~30 existing call sites
    and the seven notebooks working unchanged."""
    bound = BoundContenderSpec(family=family, config=FAMILIES[family].config)
    assert bound.get_signature() == _spec(family).get_signature()
    assert bound.resolved_label == _spec(family).resolved_label


def test_the_tier_4_binding_still_honours_an_explicit_registry() -> None:
    """Its default must be a default, not an override: `run_experiment` and the
    OOD rehost path both pass a registry explicitly.

    The registry here is deliberately a *different* one, and deliberately
    falsy. Passing the built-in registry back in would prove nothing -- the
    binding's default is that same object, so the assertion would hold whether
    or not the argument was consulted. And a registry is a container, so an
    implementation that tests it for truth rather than for `None` would discard
    an empty one and silently resolve through the built-in registry instead.
    """

    class _Alternative:
        """A second registry: empty by `len`, and answering distinguishably."""

        def __len__(self) -> int:
            return 0

        def create(self, name: str, **kwargs: Any) -> Any:
            return _recipe_directly("riccati", label="from-the-other-registry")

    spec = BoundContenderSpec(family="riccati", config={"horizon": 6})
    assert spec.resolve(_Alternative()).label == "from-the-other-registry"
    assert spec.resolve().label == "analytic"


def test_the_tier_4_binding_is_a_tier_3_contender() -> None:
    """Not a parallel type: whatever Phase D's identity functions accept, the
    notebooks' specs already are. `from_recipe` has to preserve the class too,
    or a `CaseStudy`'s contenders would come back as registry-less Tier-3
    values that only happen to work because a composed recipe short-circuits
    resolution."""
    assert issubclass(BoundContenderSpec, ContenderSpec)
    wrapped = BoundContenderSpec.from_recipe(_recipe_directly("riccati"))
    assert type(wrapped) is BoundContenderSpec


def test_naming_a_contender_does_not_require_resolving_it() -> None:
    """`resolved_label` short-circuits on an explicit label, which is not only
    a saved construction: `Experiment.__post_init__` reads every contender's
    label to check uniqueness, and a spec that was named explicitly must not
    need a registry to answer."""
    spec = ContenderSpec(family="riccati", config={"horizon": 6}, label="named")
    assert spec.resolved_label == "named"
    with pytest.raises(SpecificationError):
        spec.resolve()


# --------------------------------------------------------------------------
# The study's default plan — Stage 2 Phase G-2
# --------------------------------------------------------------------------
#
# `TrainingSpec.plan` is the study's default, materialised into each contender
# that accepts one (Annex 01 §2.3.4), so identity comes from the *effective*
# plan once rather than from a template and a plan that could disagree.
#
# Which families it reaches is written out as a literal below, deliberately.
# Deriving it from `inspect.signature` would restate the implementation and
# agree with it however wrong it was; and because the family list itself comes
# from the registry, a family added later that takes a plan makes this suite
# fail until somebody decides what it should do.

#: The families whose constructor takes a `plan`, by inspection of the source.
TAKES_A_PLAN = {"unfolded", "neural", "cocp", "cocp_exact"}

_DEFAULT_PLAN = TrainingPlan(
    optimizer=OptimizerSpec(name="sgd", learning_rate=0.5), epochs=41
)


@pytest.mark.parametrize("family", EVERY_FAMILY)
def test_the_default_plan_reaches_exactly_the_families_that_take_one(
    family: str,
) -> None:
    """Both directions in one parametrisation, over every registered family.

    A family that takes no plan must be left *untouched* rather than given one
    it would reject: `truncated_riccati` solves in closed form, and writing a
    fitting procedure onto it invents one. `unfolded_warmstart` is the
    load-bearing case -- it declares a layer-wise `schedule`, which an
    end-to-end plan is not, so skipping it is the plan-vs-schedule rule rather
    than an accident.
    """
    without = {
        key: value for key, value in FAMILIES[family].config.items() if key != "plan"
    }
    spec = ContenderSpec(family=family, config=without, registry=REGISTRY)
    given = spec.with_default_plan(_DEFAULT_PLAN)

    if family in TAKES_A_PLAN:
        assert dict(given.config)["plan"] is _DEFAULT_PLAN
        assert given.resolve().get_signature()["plan"] == _DEFAULT_PLAN.get_signature()
    else:
        assert "plan" not in dict(given.config), (
            f"{family} was given a fitting procedure it never declared"
        )
        assert dict(given.config) == without


def test_a_contender_that_declares_its_own_plan_keeps_it() -> None:
    """A default is a default. This is the case every learned contender of
    NB04 is in, and the one where the two homes used to disagree."""
    spec = _spec("unfolded")
    given = spec.with_default_plan(_DEFAULT_PLAN)
    assert dict(given.config)["plan"] is _PLAN
    assert given.get_signature() == spec.get_signature()


def test_a_layerwise_family_keeps_its_schedule() -> None:
    """Anti-vacuity for the skip above: `unfolded_warmstart` is skipped because
    it takes no `plan`, not because it takes nothing."""
    spec = _spec("unfolded_warmstart")
    given = spec.with_default_plan(_DEFAULT_PLAN)
    assert dict(given.config)["schedule"] is _SCHEDULE
    assert "plan" not in dict(given.config)


def test_a_composed_recipe_is_never_given_a_plan() -> None:
    """`from_recipe` names *that object*. It is already built, with whatever
    plan it was built with, and there is no configuration to inject into --
    the same precedence that makes a composed recipe beat every registry.

    **The registry is supplied explicitly**, and that is the whole test.
    `from_recipe` leaves it `None`, so a spec built that way comes back
    unchanged whatever the composed-recipe guard does -- the assertion would
    hold with the guard deleted. The Tier-4 binding every notebook uses always
    supplies one, which is the case this must cover.
    """
    recipe = _recipe_directly("unfolded")
    spec = ContenderSpec(family="unfolded", recipe=recipe, registry=REGISTRY)
    assert spec.registry is not None, "the guard under test would not be reached"
    assert spec.with_default_plan(_DEFAULT_PLAN) == spec
    assert "plan" not in dict(spec.with_default_plan(_DEFAULT_PLAN).config)
    assert spec.with_default_plan(_DEFAULT_PLAN).resolve() is recipe
    # And the `from_recipe` spelling, which reaches the same place by the
    # registry-less route.
    assert (
        ContenderSpec.from_recipe(recipe).with_default_plan(_DEFAULT_PLAN).resolve()
        is recipe
    )


def test_a_contender_with_no_registry_is_left_alone() -> None:
    """Tier 3 holds no default registry, so a spec that was never injected one
    cannot be asked what its family accepts. It must come back unchanged and
    fail later with `resolve()`'s message, which names the real problem."""
    spec = ContenderSpec(family="unfolded", config={"num_iterations": 3})
    assert spec.with_default_plan(_DEFAULT_PLAN) == spec


def test_the_default_plan_signs_as_though_it_had_been_written_out() -> None:
    """Independent recomputation, and the whole claim of "a default": the
    materialised contender and the same plan spelled into the config by hand
    must be the same specification, key for key."""
    omitted = ContenderSpec(
        family="unfolded",
        config={k: v for k, v in FAMILIES["unfolded"].config.items() if k != "plan"},
        registry=REGISTRY,
    ).with_default_plan(_DEFAULT_PLAN)
    spelled = ContenderSpec(
        family="unfolded",
        config={**FAMILIES["unfolded"].config, "plan": _DEFAULT_PLAN},
        registry=REGISTRY,
    )
    assert omitted.get_signature() == spelled.get_signature()
    assert model_id(
        problem=STAND_IN_PROBLEM_ID,
        contender=omitted.get_signature(),
        training={},
        ctx={},
        seed=0,
    ) == model_id(
        problem=STAND_IN_PROBLEM_ID,
        contender=spelled.get_signature(),
        training={},
        ctx={},
        seed=0,
    )
