"""One competing controller family, as data (Stage 2 Phase B).

A contender is a `(family, config)` pair **resolved through a recipe registry
before it is signed**, so that two spellings of one effective configuration
share a single identity: omitting a defaulted field and spelling it out at its
default must produce the same `ModelID`, or the store would hold the same model
twice and the second run would retrain it. That behaviour predates the grammar
and is kept exactly; this module changes where it lives, not what it does.

**The registry is injected, never imported.** Tier 3 may reach into `core` and
`store` and nothing above them, and today's registry lives at Tier 4 --
`build_default_recipe_registry()` imports every recipe module, and with it the
engine, the models and the persistence layer. Importing it here would hand the
grammar precisely the dependency that made `experiments.Experiment`
un-reusable, and a deferred import inside `resolve()` would do the same while
hiding it from the static boundary test.

So the two things this module needs are declared as structural `Protocol`s and
supplied by the caller. `experiments.ContenderSpec` is the Tier-4 binding that
supplies this project's own registry as the default, which is why the notebooks
and the study modules see no change at all.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import Any, Protocol, Self, runtime_checkable

from ..core.utils.signing import Signable
from .errors import SpecificationError

#: The constructor parameter a study's default training plan is offered to.
#: A family declaring it accepts the study's plan; `unfolded_warmstart`
#: declares a layer-wise `schedule` instead and is therefore skipped, which is
#: the plan-vs-schedule rule of Annex 01 §2.3.4 falling out of the mechanism
#: rather than being written down twice.
PLAN_PARAMETER = "plan"


@runtime_checkable
class Recipe(Protocol):
    """What Tier 3 needs *of* a recipe: a name, a label, and a signature.

    Deliberately far narrower than `applications.recipes.base.ModelRecipe`,
    which also knows how to build controllers and wire engines. The grammar
    derives identity; it never executes anything.

    The members are read-only properties rather than attributes so that a
    recipe declaring `family` as a `ClassVar` -- which every one of them does --
    satisfies the protocol.
    """

    @property
    def family(self) -> str:
        """The registry name of this recipe's family."""
        ...

    @property
    def label(self) -> str:
        """The per-instance name."""
        ...

    def get_signature(self) -> dict[str, Any]:
        """This recipe's specification-time signature tree."""
        ...


class RecipeRegistry(Protocol):
    """What Tier 3 needs *of* a registry: name plus keyword arguments in, a
    composed recipe out — and the constructor itself, to ask what it takes.

    `name` is deliberately *not* positional-only. Declaring it so would be the
    looser contract in the abstract, but `ControllerRegistry.create` forwards
    `**kwargs` to a recipe constructor and therefore cannot accept a `name`
    keyword meant for the recipe -- so a positional-only protocol would promise
    callers something the only implementation cannot honour.
    """

    def create(self, name: str, **kwargs: Any) -> Any:
        """Construct the recipe registered under `name`."""
        ...

    def get(self, name: str) -> Callable[..., Any]:
        """The constructor registered under `name`.

        Needed by `ContenderSpec.with_default_plan`, which has to know whether
        a family accepts a training plan before it offers one. Asked of the
        constructor rather than of a table in this tier, for the reason the
        loader's unknown-family pre-check was deleted in Phase E3: a table
        restating what a family accepts is a second source of truth, free to
        drift from what `create` actually takes.
        """
        ...


class Role(StrEnum):
    """What a contender *is*, for presentation and analysis (Annex 01 §2.2).

    Never part of any identity -- reclassifying a contender must not retrain
    it -- and load-bearing precisely because the distinction has been got wrong
    before: a bound rendered as though it were an attained policy, and an upper
    bound presented as a floor because its identifier said "lower_bound".
    Bounds are drawn with the reference-line grammar; policies are not.
    """

    #: A learned or solved policy under test.
    CONTENDER = "contender"
    #: An attained, non-learned policy (truncated Riccati).
    BASELINE = "baseline"
    #: An exact attained optimum where one exists (Riccati, unconstrained).
    REFERENCE = "reference"
    #: A computed bound that is *not* an attained policy (the LQR/SDP floors).
    BOUND = "bound"


@dataclass(frozen=True)
class ContenderSpec:
    """One competing controller family as data.

    Attributes:
        family: The registry name of the recipe family (``"riccati"``,
            ``"unfolded"``, ...). Ignored when `recipe` is given.
        config: Keyword arguments for the registry's recipe constructor; values
            may themselves be signable specs (e.g. a `TrainingPlan`).
        label: Optional per-study instance name; `None` keeps the recipe's own
            default label. **This is the join key** -- series selection,
            `applies_to`, the colour registry, gate grouping and the CLI's
            `--contender` all match on it exactly -- and it is never what a
            reader is shown (Annex 01 §2.2.1).
        display: Optional reader-facing name (``"Unfolded-$\alpha$+P"``);
            `None` falls back to `resolved_label`. Excluded from the signature
            by the same mechanism as `role`: it never reaches the recipe.
        recipe: An already-composed recipe, bypassing registry resolution
            (`from_recipe`).
        role: The presentation classification. Excluded from the signature.
        registry: The injected registry `resolve` looks `family` up in.
            Excluded from equality and from `repr`: it is a collaborator, not
            content, and two specs differing only in which registry object
            resolved them are the same specification.
    """

    family: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None
    display: str | None = None
    recipe: Recipe | None = None
    role: Role = Role.CONTENDER
    registry: RecipeRegistry | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.recipe is None and not self.family:
            raise SpecificationError(
                "ContenderSpec needs either a registry family name or a "
                "composed recipe."
            )
        if self.display is not None and self.display.startswith("_"):
            # Not a style objection. Matplotlib hides an artist whose label
            # starts with an underscore, and §B.4's greyscale gate skips such
            # artists as chrome -- so the series would vanish from the legend
            # AND from the check that the legend is separable in print, with
            # nothing raising anywhere.
            raise SpecificationError(
                f"contender display name {self.display!r} begins with '_', "
                "which matplotlib treats as 'do not draw this in the legend' "
                "and the greyscale gate treats as chrome. The series would "
                "disappear from both, silently."
            )

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> Self:
        """Wrap an already-composed recipe.

        Args:
            recipe: The composed recipe.

        Returns:
            A spec resolving to exactly `recipe`, of whichever `ContenderSpec`
            class this was called on.
        """
        return cls(family=recipe.family, recipe=recipe)

    def resolve(self, registry: RecipeRegistry | None = None) -> Recipe:
        """The concrete recipe this spec names.

        A composed `recipe` always wins: it names *that object*, so no registry
        may substitute another for it. Failing that, an explicit argument beats
        the injected `registry` field, so a caller holding a different registry
        never has to rebuild the spec.

        Args:
            registry: The registry to resolve through; overrides the field.

        Returns:
            The composed recipe, with the label override applied when set.

        Raises:
            SpecificationError: If no registry is available, or if `family` is
                not registered in the one that is.
        """
        if self.recipe is not None:
            return self.recipe
        if registry is None:
            registry = self.registry
        if registry is None:
            raise SpecificationError(
                f"ContenderSpec(family={self.family!r}) has no registry to "
                "resolve through: the specification tier holds no default, "
                "because importing one would give the grammar a dependency on "
                "the application layer. Pass registry=... to the constructor "
                "or to resolve(), or use mbl.experiments.ContenderSpec, which "
                "supplies this project's own."
            )
        kwargs = dict(self.config)
        if self.label is not None:
            kwargs["label"] = self.label
        try:
            composed = registry.create(self.family, **kwargs)
        except KeyError as error:
            raise SpecificationError(
                f"ContenderSpec names an unregistered family {self.family!r}: {error}"
            ) from error
        return _as_recipe(composed)

    def with_default_plan(self, plan: Signable) -> Self:
        """This contender, given the study's plan if it takes one and has none.

        Annex 01 §2.3.4: `TrainingSpec.plan` is the study's **default**, and
        this is where "default" is executed. A contender that declares its own
        keeps it; a family whose constructor takes no `plan` is left untouched,
        because writing a fitting procedure onto a closed-form solve invents
        one.

        Four cases come back unchanged, and each is a decision:

        * a **composed** recipe (`from_recipe`) names *that object*, already
          built, with no configuration to inject into;
        * a contender that **declares** a plan -- a default is a default;
        * a family whose constructor does not take one, asked of the
          constructor itself so that no table in this tier restates what a
          family accepts;
        * a spec with **no registry**, which cannot be asked. It comes back
          unchanged and fails later with `resolve()`'s message, which names the
          real problem rather than reporting a missing plan.

        Args:
            plan: The study's default plan.

        Returns:
            This spec, or a copy carrying the plan.
        """
        if self.recipe is not None or PLAN_PARAMETER in self.config:
            return self
        registry = self.registry
        if registry is None:
            return self
        try:
            constructor = registry.get(self.family)
        except (KeyError, AttributeError):
            # An unregistered family is `resolve()`'s to report, with a message
            # that lists what *is* registered.
            return self
        if PLAN_PARAMETER not in inspect.signature(constructor).parameters:
            return self
        return dataclasses.replace(self, config={**self.config, PLAN_PARAMETER: plan})

    @cached_property
    def resolved_label(self) -> str:
        """The effective instance label (the recipe's, unless overridden)."""
        return self.label if self.label is not None else self.resolve().label

    @property
    def resolved_display(self) -> str:
        """What a reader is shown, falling back to the join key.

        The fallback is deliberate and its refusal lives elsewhere: Annex 04's
        rule binds on what a reader *sees*, so a study with no figure and no
        chapter may leave this undeclared, while a rendered figure that shows a
        raw identifier is refused by the reader-facing check. Making the field
        mandatory here instead would be a rule justified by there being one
        study document in the repository today.
        """
        return self.display if self.display is not None else self.resolved_label

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the **resolved** recipe's signature tree.

        Resolving first is the point: registry defaults participate, so two
        spellings of one effective configuration share one cache identity.
        Reconstructing a tree from `config` instead would also lose the
        per-family overrides that are load-bearing -- `FixedUnfoldedRecipe` and
        its siblings omit the control-init fields when the method is `COLD`, to
        keep existing identifiers valid.

        `role` and `registry` do not appear, because neither is content.

        Returns:
            The resolved recipe's signature tree.
        """
        return self.resolve().get_signature()


def _as_recipe(value: Any) -> Recipe:
    """Narrow a registry's untyped product to the protocol Tier 3 relies on.

    A registry is a `name -> constructor` map and cannot promise more than
    `Any` about what it builds, so the gap is closed once, here, and it fails
    at resolution rather than at the first `get_signature()` call -- which,
    under a study runner, could be an hour into the run.

    Args:
        value: Whatever the registry constructed.

    Returns:
        `value`, typed as a `Recipe`.

    Raises:
        SpecificationError: If it is not one.
    """
    if not isinstance(value, Recipe):
        raise SpecificationError(
            f"the registry produced {type(value).__name__}, which is not a "
            "recipe: a contender needs family, label and get_signature()"
        )
    return value
