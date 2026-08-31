"""The control-bound (actuator-limit) shift for zero-shot OOD evaluation
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.6/3.9): Axis C. Train
under one `u_max`; deploy under `multiplier * u_max` -- the most
operationally realistic shift in the notebook (de-rated/upgraded actuators,
safety margins), and the only axis that perturbs the FEASIBLE SET rather
than the dynamics or the noise.

Three protocols share this module's machinery:

* **C-blind** -- the controller keeps its trained belief about `u_max`; the
  PLANT enforces the true (possibly tighter) bound. `CommandRecorder.wrap`
  is the mechanism: it records what the controller ACTUALLY ASKED FOR (the
  commanded, pre-clip value) while the wrapped policy returns what the
  plant APPLIES (the clipped value) -- cost/saturation are charged on the
  applied control (what the plant really did), while the feasibility audit
  is computed against the commanded control (what the controller believed
  was fine). These are genuinely different quantities under de-rating
  (`multiplier < 1`), and conflating them would hide the entire phenomenon.
  At `multiplier >= 1` the external clip provably never binds (every
  architecture in this project already projects/squashes its raw output to
  at most the TRAINED `u_max` internally, so a looser external bound is
  never reached) -- the exact-null acceptance test for that branch.
* **C-aware** -- the controller is TOLD the new bound: `rehost_at_bound`
  re-parameterizes its projection/squash at the shifted bound while
  transplanting every learned parameter untouched (the Constraint-OOD law,
  the bound analogue of `rehost.rehost_at_horizon`'s horizon law).
* **C-null** -- a joint `(u_max, sigma, sigma_0)` scale by the SAME
  multiplier is an exact homogeneity: the optimal cost scales by
  `multiplier**2` and the suboptimality gap is exactly invariant. The
  homogeneous contenders (`truncated_riccati`, and the C-aware-rehosted
  unfolded families) are `multiplier**2`-equivariant to machine precision;
  the GRU is not (its `tanh` nonlinearity is not positively homogeneous).
  This is the one EXACT (non-statistical) correctness check available
  anywhere in NB06 -- see `workbench.ood_sweep.run_constraint_ood_sweep`'s
  `"null"` protocol and `OODSweepResult.normalized_cost_band`.

`rehost_at_bound`'s per-family dispatch, unlike `rehost.rehost_at_horizon`'s,
has NO silent identity fallback: every family in this project's contender
set has ITS OWN box constraint baked in somewhere (the unfolded families'
inner-iteration projection, COCP/COCP-LB's QP constraint, the GRU's
`tanh`-squash scale) -- unlike a horizon, which several families genuinely
carry no notion of at all. An unrecognized family therefore raises rather
than silently returning the untouched artifact, which would silently
mis-report a C-aware point as if the bound had actually moved.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from ..recipes.base import ModelRecipe, TrainedControllerArtifact
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import ComputeContext
from ...core.system.system import BatchedStateSpaceVector, ControlPolicy
from ...experiments.experiment import ContenderSpec
from ...models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer
from ...models.lifecycle import SynthesizedController
from ...models.unfolded.base import UnfoldedController

#: Recipe families whose synthesized artifact is `TrainedControllerArtifact`
#: wrapping an `UnfoldedController` -- mirrors `rehost._UNFOLDED_FAMILIES`
#: (kept as its own copy here rather than importing that private name
#: across modules; the two are declaratively coupled, not a shared object).
_UNFOLDED_FAMILIES = frozenset({"unfolded", "unfolded_warmstart"})

#: Recipe families whose synthesized artifact is a plain `torch.nn.Module`
#: (COCP/COCP-LB's `COCPController`, the GRU's `NeuralPolicy`) with no
#: per-parameter clipping/validation on load -- a whole-module
#: `state_dict()`/`load_state_dict()` transplant is exact and total for all
#: three, whether the module's parameters were arrived at by gradient
#: training (`cocp`, `neural`) or a closed-form SDP solve (`cocp_lower_bound`
#: -- its cost-to-go is a FIXED function of the nominal problem, not learned
#: in the gradient sense, but the rehost mechanism is identical: transplant
#: that fixed cost-to-go into a QP re-parameterized at the new bound).
_STATE_DICT_FAMILIES = frozenset({"cocp", "cocp_lower_bound", "neural"})


@dataclass(frozen=True)
class ControlBoundShift:
    """One control-bound shift specification (Axis C, NB06 plan Sec 2.6).

    Attributes:
        multiplier: Scale factor applied to the nominal `u_max`; ``1.0`` is
            the nominal bound itself (the degeneracy anchor -- every
            protocol must reproduce the nominal evaluation exactly there).
    """

    multiplier: float

    def bound(self, nominal_u_max: float) -> float:
        """The shifted bound: ``multiplier * nominal_u_max``."""
        return self.multiplier * nominal_u_max

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member.

        Returns:
            ``{"type": "ControlBoundShift", "multiplier":}``.
        """
        return {"type": type(self).__name__, "multiplier": self.multiplier}


class CommandRecorder:
    """Records a rollout's COMMANDED (pre-clip) controls as a side effect
    while the policy it wraps returns the APPLIED (clipped) control (module
    docstring's C-blind mechanism). Mutable and stateful across calls
    WITHIN one rollout -- construct a fresh instance per batch, mirroring
    `SynthesizedController.make_policy`'s own freshness law.

    Attributes:
        commanded: Every wrapped call's raw (pre-clip) output, in call
            order -- one entry per rollout time step, each shape
            ``(batch, control_dim)``.
    """

    def __init__(self) -> None:
        self.commanded: list[torch.Tensor] = []

    def wrap(self, policy: ControlPolicy, *, bound: float) -> ControlPolicy:
        """Wrap `policy` so every call's raw output is recorded before
        being clipped to ``[-bound, bound]``.

        Args:
            policy: The underlying policy -- already internally bounded at
                its OWN trained `u_max` (every family in this project
                projects/squashes internally), so `bound` here is the
                PLANT's own (possibly tighter) limit, not a redundant
                re-clip of an unconstrained signal.
            bound: The bound the plant enforces.

        Returns:
            The wrapped `ControlPolicy`: same signature, clipped output,
            with every commanded value appended to `self.commanded`.
        """

        def wrapped(t: int, x: BatchedStateSpaceVector) -> BatchedStateSpaceVector:
            u = cast(torch.Tensor, policy(t, x))
            self.commanded.append(u.detach())
            return u.clamp(-bound, bound)

        return wrapped


def _rehost_unfolded_at_bound(
    source_artifact: TrainedControllerArtifact,
    recipe: ModelRecipe,
    target_problem: OptimalControlProblem,
    ctx: ComputeContext,
) -> TrainedControllerArtifact:
    """Rebuild against `target_problem` (whose box constraint carries the
    shifted bound -- unlike the horizon rehost, `recipe` itself needs NO
    `dataclasses.replace`: the bound is read off the PROBLEM at build time
    (`applications.recipes.unfolded.build_unfolded_controller` reads
    `problem.constraints`), never carried as a recipe/build-spec field),
    then transplant every learned parameter from the source controller --
    identical mechanism to `rehost._rehost_unfolded`."""
    rebuilt_controller = recipe.build_controller(target_problem, ctx)
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


def _rehost_via_state_dict(
    source_artifact: TrainedControllerArtifact,
    recipe: ModelRecipe,
    target_problem: OptimalControlProblem,
    ctx: ComputeContext,
) -> TrainedControllerArtifact:
    """Rebuild against `target_problem` (COCP/COCP-LB's QP layer and the
    GRU's squash both read their bound fresh from `target_problem
    .constraints[0]` at construction/forward time respectively), then
    transplant every learned tensor via a whole-module `state_dict()` --
    exact and total, since neither `COCPController` (`P_sqrt`/`q`, plain
    `nn.Parameter`s) nor `NeuralPolicy` (its backbone's ordinary weights)
    apply any per-parameter clipping on load, unlike the unfolded family's
    `alpha_max`-clipped step size."""
    rebuilt_controller = recipe.build_controller(target_problem, ctx)
    source_controller = source_artifact.controller
    assert isinstance(rebuilt_controller, nn.Module)
    assert isinstance(source_controller, nn.Module)
    rebuilt_controller.load_state_dict(source_controller.state_dict())

    return TrainedControllerArtifact(
        controller=rebuilt_controller,
        context=ctx,
        synthesizer_signature=source_artifact.synthesizer_signature,
        provenance=source_artifact.provenance,
    )


def rehost_at_bound(
    artifact: SynthesizedController,
    spec: ContenderSpec,
    target_problem: OptimalControlProblem,
    *,
    horizon: int,
    ctx: ComputeContext,
) -> SynthesizedController:
    """Produce the artifact `experiments.zero_shot.evaluate_under_shift`
    should roll out under `target_problem`'s (shifted) bound, per `spec`'s
    family (module docstring's per-family dispatch).

    Args:
        artifact: The nominally-trained (or frozen) artifact.
        spec: The `ContenderSpec` `artifact` was synthesized from -- its
            family selects the dispatch branch.
        target_problem: The problem to roll out against at the new bound
            (typically `applications.ood.perturbations.PerturbedLQRProblemFactory`
            with ``u_max_override=<the shifted bound>``).
        horizon: `target_problem`'s own horizon (Axis C never shifts the
            horizon, but `truncated_riccati`'s fresh solve needs it
            explicitly, matching `rehost.rehost_at_horizon`'s own contract).
        ctx: The compute context for any fresh solve/build this dispatch
            performs.

    Returns:
        The artifact to evaluate at the shifted bound: freshly solved
        (`truncated_riccati`), rebuilt-and-transplanted (every other
        family).

    Raises:
        TypeError: If `spec` resolves to `truncated_riccati` and
            `target_problem` declares no box constraint.
        ValueError: If `spec` resolves to a family this dispatch does not
            recognize (deliberately no silent identity fallback -- see
            module docstring).
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
        return _rehost_unfolded_at_bound(artifact, recipe, target_problem, ctx)
    if recipe.family in _STATE_DICT_FAMILIES:
        assert isinstance(artifact, TrainedControllerArtifact)
        return _rehost_via_state_dict(artifact, recipe, target_problem, ctx)
    raise ValueError(
        f"rehost_at_bound: unrecognized family {recipe.family!r} -- every "
        "family in this project's contender set has its own box constraint "
        "baked in somewhere, so there is deliberately no silent identity "
        "fallback (module docstring)."
    )
