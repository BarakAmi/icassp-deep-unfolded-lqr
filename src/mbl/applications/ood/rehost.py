"""The horizon-rehost seam (docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md
Sec 1.4/3.5): a nominally-trained artifact cannot simply be rolled out at a
different horizon `N`, because two families cache horizon-length internal
state --

* `UnfoldedController` invalidates a per-forward cache of BATCHED HORIZON
  gradient matrices at every rollout boundary
  (`models.unfolded.base.UnfoldedController.get_control_policy`), built from
  a Riccati stack solved for the training horizon.
* `TruncatedRiccatiController`/`SynthesizedTruncatedRiccatiController` hold
  `P_arr`/`K_arr` solved for a FIXED horizon
  (`models.analytic.truncated_riccati`).

`rehost_at_horizon` dispatches per family, per the Horizon-OOD law (NB06
plan Sec 1.4): **analytic structure is recomputed at the test horizon;
learned parameters are frozen and transplanted.**

* `truncated_riccati` -- has no learned parameter at all; "rehosting" IS
  just a fresh `TruncatedRiccatiSynthesizer` solve against `target_problem`
  at `horizon` (the family's whole behavior is "solve exactly for whatever
  problem/horizon it is given").
* `unfolded` / `unfolded_warmstart` -- rebuilds a fresh `UnfoldedController`
  at `horizon` via the SAME recipe (`dataclasses.replace(recipe,
  horizon=...)` then `recipe.build_controller`, never a hand-rolled
  reconstruction of `UnfoldedBuildSpec` -- this guarantees the rebuilt
  controller's `alpha_max`/`init_method`/etc. are IDENTICAL to the ones the
  source was trained under, which matters: `StepSizeParameter.load` clips
  the loaded value into ``(tol, alpha_max - tol)``, so a mismatched
  `alpha_max` would silently corrupt the transplant), then transplants every
  learned parameter via `UnfoldedParameter.load` -- exact and total, since
  `alpha` (`(K, m)`) and the Riccati-replacing matrix (`(n, n)`) are both
  horizon-free by shape.
* `neural`, `cocp`, `cocp_lower_bound` -- the GRU is recurrent (no
  horizon-length object) and the COCP families solve a per-step QP (also no
  horizon-length object); rehosting is a pure identity, returning the SAME
  artifact.
"""

import dataclasses

from ..recipes.base import ModelRecipe, TrainedControllerArtifact
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import ComputeContext
from ...experiments.experiment import ContenderSpec
from ...models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer
from ...models.lifecycle import SynthesizedController
from ...models.unfolded.base import UnfoldedController

#: Recipe families whose synthesized artifact is `TrainedControllerArtifact`
#: wrapping an `UnfoldedController` -- see module docstring's unfolded branch.
_UNFOLDED_FAMILIES = frozenset({"unfolded", "unfolded_warmstart"})


def _rehost_unfolded(
    source_artifact: TrainedControllerArtifact,
    recipe: ModelRecipe,
    target_problem: OptimalControlProblem,
    horizon: int,
    ctx: ComputeContext,
) -> TrainedControllerArtifact:
    """Rebuild at `horizon` via the source recipe (unchanged except for
    `horizon`), then transplant every learned parameter from the source
    controller (see module docstring)."""
    # `dataclasses.replace` requires a concrete dataclass instance; `recipe`
    # is typed at its abstract `ModelRecipe` base (mypy cannot see that every
    # `_UNFOLDED_FAMILIES` member is one), but both `UnfoldedRecipe` and
    # `WarmStartUnfoldedRecipe` are frozen dataclasses with a `horizon`
    # field, so this holds for every recipe this branch is ever called with.
    rebuilt_recipe = dataclasses.replace(recipe, horizon=horizon)  # type: ignore[type-var]
    rebuilt_controller = rebuilt_recipe.build_controller(target_problem, ctx)
    assert isinstance(rebuilt_controller, UnfoldedController)
    source_controller = source_artifact.controller
    assert isinstance(source_controller, UnfoldedController)

    for name, parameter in rebuilt_controller.config.parameters.items():
        parameter.load(source_controller.config.parameters[name].get_numpy())

    return TrainedControllerArtifact(
        controller=rebuilt_controller,
        context=ctx,
        synthesizer_signature=source_artifact.synthesizer_signature,
        provenance=source_artifact.provenance,
    )


def rehost_at_horizon(
    artifact: SynthesizedController,
    spec: ContenderSpec,
    target_problem: OptimalControlProblem,
    *,
    horizon: int,
    ctx: ComputeContext,
) -> SynthesizedController:
    """Produce the artifact `evaluate_under_shift` should roll out at
    `horizon`, per `spec`'s family (see module docstring for the per-family
    dispatch and why each branch is correct).

    Args:
        artifact: The nominally-trained (or frozen) artifact.
        spec: The `ContenderSpec` `artifact` was synthesized from -- its
            family selects the dispatch branch, and (for the unfolded
            families) its resolved recipe is the single source of truth for
            every construction hyperparameter the rebuild must match exactly.
        target_problem: The problem to roll out against at the new horizon
            (typically `applications.ood.perturbations.PerturbedLQRProblemFactory`
            with ``horizon_override=horizon``).
        horizon: The target horizon.
        ctx: The compute context for any fresh solve/build this dispatch
            performs.

    Returns:
        The artifact to evaluate at `horizon`: freshly solved
        (`truncated_riccati`), rebuilt-and-transplanted (`unfolded`/
        `unfolded_warmstart`), or `artifact` itself, unchanged (every other
        family).

    Raises:
        TypeError: If `spec` resolves to `truncated_riccati` and
            `target_problem` declares no box constraint.
    """
    recipe = spec.resolve()
    if recipe.family == "truncated_riccati":
        constraints = target_problem.constraints or []
        if not constraints:
            raise TypeError(
                "Rehosting a truncated_riccati contender requires "
                "target_problem.constraints[0] (its box constraint)."
            )
        return TruncatedRiccatiSynthesizer(horizon, constraints[0]).synthesize(
            target_problem, ctx
        )
    if recipe.family in _UNFOLDED_FAMILIES:
        assert isinstance(artifact, TrainedControllerArtifact)
        return _rehost_unfolded(artifact, recipe, target_problem, horizon, ctx)
    return artifact
