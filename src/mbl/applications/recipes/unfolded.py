"""Recipes for the deep-unfolded (unrolled gradient-descent) controller
family, in its three benchmark configurations -- previously constructed by
two ~60-line verbatim-duplicated blocks inside each app's `build_models`:

    fixed:                         fixed step-size, true Riccati P, nothing learned
    learned_step_size:             learned per-iteration step-size, true Riccati P
    learned_step_size_and_matrix:  learned per-iteration step-size + a single
                                    learned time-invariant matrix replacing Riccati P

The construction honors the problem's own constraints (a box-constrained
problem yields constraint-projected refinements; an unconstrained one
yields the plain refinements), so one recipe serves both case studies.
"""

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
import torch

from .base import (
    accept_enum_values,
    AnalyticRecipe,
    EngineHarness,
    ModelRecipe,
    TrainableRecipe,
    TrainedControllerArtifact,
    register_recipe,
)
from ..rollout import RolloutModel
from ...core.kernels import time_invariant_slice
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import ComputeContext
from ...engine.callbacks import (
    Callback,
    ParameterSnapshotCallback,
    StructuredTrainingLogCallback,
    TrainingStateCheckpointCallback,
)
from ...engine.engine import Engine
from ...engine.runner import Runner, TrainingPhase
from ...engine.strategy import LayerwiseGradientDescentStrategy, StepExecution
from ...engine.training_plan import LayerwiseTrainingPlan, TrainingPlan
from ...models.analytic.riccati import (
    finite_horizon_riccati,
    get_lqr_gradient_matrices,
)
from ...models.iterative.initializers import (
    ControlInitMethod,
    build_control_initializer,
)
from ...models.base import Controller
from ...models.guards import require_linear_quadratic
from ...models.iterative.refinement import GradientDescentRefinement
from ...models.lifecycle import Synthesizer
from ...models.unfolded.base import UnfoldedController, UnfoldingConfig
from ...models.unfolded.iterative_refinement import (
    PerIterationRiccatiRefinement,
    RiccatiRefinement,
    ScalarModulatedRiccatiRefinement,
    StepSizeRefinement,
)
from ...models.unfolded.layerwise import LayerFreezeBuilder
from ...models.unfolded.parameters import (
    UnfoldedParameter,
    require_saturating_bound,
    MatrixModulationParameter,
    MatrixModulationParameterConfig,
    PerIterationRiccatiMatrixParameter,
    PerIterationRiccatiMatrixParameterConfig,
    RiccatiMatrixParameter,
    RiccatiMatrixParameterConfig,
    StepSizeParameter,
    StepSizeParameterConfig,
)

logger = logging.getLogger(__name__)


class UnfoldedKind(StrEnum):
    """Which of the benchmark unfolded configurations to build."""

    FIXED = "fixed"
    LEARNED_STEP_SIZE = "learned_step_size"
    LEARNED_STEP_SIZE_AND_MATRIX = "learned_step_size_and_matrix"
    #: R1.5 (NB05_LTV_BOX_CONSTRAINED_PLAN.md Sec 14.9): one shared learned
    #: matrix P rescaled per unfolding iteration by a learned positive
    #: scalar c_j -- P^(j) = c_j * P. The parameter-matched control for
    #: LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX below.
    LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX = (
        "learned_step_size_and_scalar_modulated_matrix"
    )
    #: R2 (NB05_LTV_BOX_CONSTRAINED_PLAN.md Sec 14.1-14.10): J fully
    #: independent learned matrices P^(0), ..., P^(J-1), one per unfolding
    #: iteration.
    LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX = (
        "learned_step_size_and_per_iteration_matrix"
    )


#: R1.5's modulation-scale reparameterization upper bound (NB05 plan Sec
#: 14.9): P^(j) = c_j * P with c_j in (0, MATRIX_MODULATION_MAX). Not a
#: notebook-exposed knob -- generous headroom around the c_j = 1 identity
#: (which reproduces R1's P exactly) without leaving the reparameterization
#: unbounded.
MATRIX_MODULATION_MAX = 5.0


@dataclass(frozen=True)
class UnfoldedBuildSpec:
    """The configuration axis of one unfolded-controller construction.

    Attributes:
        kind: Which benchmark configuration to build.
        num_iterations: Unfolding (inner refinement) iteration count.
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        horizon: The rollout horizon.
        init_method: How ``u^(0)`` is seeded before refinement
            (`ControlInitMethod`); defaults to the classical cold start.
        random_init_std: Std of the Gaussian draw for
            `ControlInitMethod.RANDOMIZED`; ignored otherwise.
        random_init_seed: Seed for the reproducible, device-native generator
            of `ControlInitMethod.RANDOMIZED`; ignored otherwise. `None` --
            the default -- **inherits the training replicate** through the
            ambient global RNG (Annex 01 §2.3.3); a declared value is a pin,
            fixing the draw across every replicate.
    """

    kind: UnfoldedKind
    num_iterations: int
    step_size_init: float
    step_size_max: float
    horizon: int
    init_method: ControlInitMethod = ControlInitMethod.COLD
    random_init_std: float = 1.0
    random_init_seed: int | None = None


def build_unfolded_controller(
    problem: OptimalControlProblem,
    ctx: ComputeContext,
    spec: UnfoldedBuildSpec,
    *,
    riccati_problem: OptimalControlProblem | None = None,
) -> UnfoldedController:
    """The single home of the unfolded-controller construction (kills the
    M1 duplication): Riccati-derived gradient coefficient stacks, the
    step-size/matrix parameters for `spec.kind`, and the matching refinement,
    with the problem's own constraints applied during refinement.

    Args:
        problem: The `OptimalControlProblem` to solve.
        ctx: The compute context (single dtype/device authority).
        spec: The configuration axis (see `UnfoldedBuildSpec`).
        riccati_problem: The plant the OFFLINE Riccati recursion solves, when
            it is not `problem` (Annex 01 §2.4.1's aware rebuild): the
            cost-to-go stack is an offline artifact and stays derived from
            the plant the model was told, while the per-step gradient
            coefficients — the online expressions — form from the frozen
            stack and `problem`'s own matrices. `None`, the matched default,
            is every caller before the rehost existed. Only the kinds whose
            construction solves a Riccati recursion read it; the learned-P
            kinds carry no offline stack to freeze.

    Returns:
        The constructed `UnfoldedController`.
    """
    kind, num_iterations, horizon = spec.kind, spec.num_iterations, spec.horizon
    step_size_init, step_size_max = spec.step_size_init, spec.step_size_max
    dtype, device = ctx.torch_dtype, ctx.torch_device
    system, cost = require_linear_quadratic(problem)
    n = system.dimensions.state_dim
    m = system.dimensions.control_dim
    # Per-step densification (NB05_LTV_BOX_CONSTRAINED_PLAN.md Sec 1.3 Rule 3),
    # not `system.A_t.array`/`system.B_t.array`: `.array` is TimeSeriesMatrix's
    # compressed UNIQUE-matrix bank (shape (D, ...), D the distinct-slice
    # count), which only coincides with the per-step stack when the system is
    # time-invariant (D == 1, stored 2D). For a genuinely time-varying system
    # (D > 1) `.array` is neither in time order nor of length `horizon`, so
    # every downstream broadcast against a (horizon, ...)-shaped stack (P_arr,
    # M_stack, C_stack below) would silently misalign or outright fail to
    # broadcast. `A_t[k]` is the only correct per-step access -- the same
    # idiom `RiccatiSynthesizer.synthesize` already uses. For a time-invariant
    # system this reduces to `horizon` copies of the same 2D matrix, which
    # every consumer below already broadcast identically before this change.
    A = np.stack([system.A_t[k] for k in range(horizon)])
    B = np.stack([system.B_t[k] for k in range(horizon)])
    R = cost.R
    constraints = list(problem.constraints or [])

    step_size = StepSizeParameter(
        StepSizeParameterConfig(
            name="step_size",
            num_iterations=num_iterations,
            action_dim=m,
            alpha_init=step_size_init,
            alpha_max=step_size_max,
            dtype=dtype,
            device=device,
        )
    )
    initializer = build_control_initializer(
        spec.init_method,
        control_dim=m,
        dtype=dtype,
        device=device,
        random_std=spec.random_init_std,
        seed=spec.random_init_seed,
    )

    if kind is UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX:
        riccati_matrix = RiccatiMatrixParameter(
            RiccatiMatrixParameterConfig(
                name="riccati_matrix",
                num_iterations=num_iterations,
                state_dim=n,
                dtype=dtype,
                device=device,
            )
        )
        # A/B are already densified to (horizon, n, n)/(horizon, n, m) above
        # (no `.expand` needed -- that used to paper over `.array`'s wrong
        # shape, and silently assumed time-invariance in the process).
        A_stack = torch.tensor(A, dtype=dtype, device=device)
        B_stack = torch.tensor(B, dtype=dtype, device=device)
        # already time-stacked: (horizon, m, m)
        R_stack = torch.tensor(R, dtype=dtype, device=device)
        parameters: dict[str, UnfoldedParameter[Any]] = {
            "step_size": step_size,
            "riccati_matrix": riccati_matrix,
        }
        refinement: GradientDescentRefinement = RiccatiRefinement(
            step_size=step_size,
            num_iterations=num_iterations,
            learnable_parameters={"riccati_matrix": riccati_matrix},
            static_parameters={"A": A_stack, "B": B_stack, "R": R_stack},
            constraints=constraints,
        )
    elif kind is UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX:
        # R1.5 (NB05 plan Sec 14.9): the SAME shared-matrix parameter as the
        # LEARNED_STEP_SIZE_AND_MATRIX branch above (registered under the
        # identical "riccati_matrix" key, so WarmStartUnfoldedRecipe's
        # `has_matrix` check and its whole-tensor `train_matrix` freeze both
        # keep working unchanged), plus a new per-iteration positive scalar
        # that rescales it: P^(j) = c_j * P.
        riccati_matrix = RiccatiMatrixParameter(
            RiccatiMatrixParameterConfig(
                name="riccati_matrix",
                num_iterations=num_iterations,
                state_dim=n,
                dtype=dtype,
                device=device,
            )
        )
        matrix_modulation = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=num_iterations,
                modulation_init=1.0,
                modulation_max=MATRIX_MODULATION_MAX,
                dtype=dtype,
                device=device,
            )
        )
        A_stack = torch.tensor(A, dtype=dtype, device=device)
        B_stack = torch.tensor(B, dtype=dtype, device=device)
        R_stack = torch.tensor(R, dtype=dtype, device=device)
        parameters = {
            "step_size": step_size,
            "riccati_matrix": riccati_matrix,
            "matrix_modulation": matrix_modulation,
        }
        refinement = ScalarModulatedRiccatiRefinement(
            step_size=step_size,
            num_iterations=num_iterations,
            learnable_parameters={
                "riccati_matrix": riccati_matrix,
                "matrix_modulation": matrix_modulation,
            },
            static_parameters={"A": A_stack, "B": B_stack, "R": R_stack},
            constraints=constraints,
        )
    elif kind is UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX:
        # R2 (NB05 plan Sec 14.1-14.10): J fully independent learned
        # matrices, still registered under "riccati_matrix" for the same
        # `has_matrix`/`train_matrix` reasons as the branch above.
        per_iteration_matrix = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix",
                num_iterations=num_iterations,
                state_dim=n,
                dtype=dtype,
                device=device,
            )
        )
        A_stack = torch.tensor(A, dtype=dtype, device=device)
        B_stack = torch.tensor(B, dtype=dtype, device=device)
        R_stack = torch.tensor(R, dtype=dtype, device=device)
        parameters = {
            "step_size": step_size,
            "riccati_matrix": per_iteration_matrix,
        }
        refinement = PerIterationRiccatiRefinement(
            step_size=step_size,
            num_iterations=num_iterations,
            learnable_parameters={"riccati_matrix": per_iteration_matrix},
            static_parameters={"A": A_stack, "B": B_stack, "R": R_stack},
            constraints=constraints,
        )
    else:
        if kind is UnfoldedKind.FIXED:
            step_size.get_raw().requires_grad_(False)
        if riccati_problem is not None:
            offline_system, offline_cost = require_linear_quadratic(riccati_problem)
            P_arr, _ = finite_horizon_riccati(offline_system, offline_cost, horizon)
        else:
            P_arr, _ = finite_horizon_riccati(system, cost, horizon)
        M_stack, C_stack = get_lqr_gradient_matrices(P_arr, A, B, R)
        parameters = {"step_size": step_size}
        refinement = StepSizeRefinement(
            step_size=step_size,
            num_iterations=num_iterations,
            static_parameters={
                "M_stack": torch.tensor(2 * M_stack, dtype=dtype, device=device),
                "C_stack": torch.tensor(2 * C_stack, dtype=dtype, device=device),
            },
            constraints=constraints,
        )

    return UnfoldedController(
        problem,
        UnfoldingConfig(
            parameters=parameters,
            control_initializer=initializer,
            iterative_refinement=refinement,
            horizon=horizon,
        ),
    )


#: The init-selection recipe fields folded into a recipe signature only when
#: NON-default (see `_unfolded_recipe_signature`).
_INIT_SIGNATURE_FIELDS = ("init_method", "random_init_std", "random_init_seed")


def _unfolded_recipe_signature(
    init_method: ControlInitMethod,
    random_init_std: float,
    random_init_seed: int | None,
    base_signature: dict[str, Any],
) -> dict[str, Any]:
    """A recipe signature that folds the control-init selection in ONLY when
    it is not the default cold start, so introducing this knob leaves every
    existing cold-start run's digest (and the golden masters) byte-identical,
    while ``warm``/``randomized`` become correctly-distinct cache entries. A
    randomized draw additionally contributes its ``std``/``seed`` -- the two
    values that change its realized initial control -- while ``warm`` (which
    has none) contributes only the method name. Digests are key-sorted
    (`core.utils.signing.compute_signature_digest`), so dropping and re-adding
    keys never perturbs a cold digest.

    Args:
        init_method: The recipe's selected control-init method.
        random_init_std: The recipe's randomized-draw std.
        random_init_seed: The recipe's randomized-draw seed, or `None` for
            "inherit the training replicate". Both are signed, and they are
            genuinely different models: one varies with the replicate and one
            does not.
        base_signature: The recipe's field-derived base signature
            (`ModelRecipe.get_signature`), which already carries the three
            init fields verbatim -- they are stripped here and re-added
            conditionally.

    Returns:
        The adjusted signature mapping.
    """
    signature = {
        key: value
        for key, value in base_signature.items()
        if key not in _INIT_SIGNATURE_FIELDS
    }
    if init_method is ControlInitMethod.COLD:
        return signature
    signature["init_method"] = init_method.value
    if init_method is ControlInitMethod.RANDOMIZED:
        signature["random_init_std"] = random_init_std
        signature["random_init_seed"] = random_init_seed
    return signature


def unfolded_convergence_parameters(
    controller: UnfoldedController,
    problem: OptimalControlProblem,
    *,
    horizon: int,
) -> dict:
    """Learned parameters + problem matrices needed to replay this unfolded
    model's per-iteration control convergence later (see
    `src.viz.plots.convergence`), saved via `ParameterSnapshotCallback`
    since the controller's structured parameters live in a plain
    `UnfoldingConfig` dataclass (its own ``state_dict`` checkpoints only
    what `as_module` has registered).

    This is called BEFORE training starts (from `extra_callbacks`, at engine
    build time), so the learned entries ("step_size"/"riccati_matrix") are
    bound as the parameter's own ``.get`` bound method -- a zero-argument
    callable `ParameterSnapshotCallback` re-invokes at `on_train_end` time --
    rather than the result of calling `.get()` here. `.get()` returns a
    reparameterization derived from the raw `nn.Parameter` (e.g.
    ``sigmoid(rho) * alpha_max``), a fresh tensor computed once, not a live
    reference to `rho` -- calling it eagerly here would silently freeze the
    UNTRAINED initial value into the snapshot forever, however much training
    subsequently happens.

    Args:
        controller: The (about to be trained) `UnfoldedController`.
        problem: The problem whose matrices the replay needs.
        horizon: The Riccati horizon for the true-P variants.

    Returns:
        Also includes "A_t"/"B_t" (the full per-step ``(horizon, ...)``
        dynamics stacks, additive alongside "A"/"B" -- see the LTV plan's
        Sec 6.4). Full contract:
        Name -> (live tensor/array, or zero-argument callable) mapping to
        snapshot at train end -- the learned parameters are callables
        (see above); the static problem matrices ("A"/"B"/"R"/"P") are the
        arrays themselves, since they never change during training.
    """
    system, cost = require_linear_quadratic(problem)
    parameters: dict[str, Any] = {
        "step_size": controller.config.parameters["step_size"].get,
        # NB05_LTV_BOX_CONSTRAINED_PLAN.md Sec 6.4: `A_t[0]`/`B_t[0]`, not
        # `.array` (the compressed unique-matrix bank, wrong both in order
        # and in length under LTV) -- bit-identical to the old `.array` read
        # for a time-invariant system (both ARE the same 2D matrix), and the
        # honest "the system AT t=0" reading otherwise. "A_t"/"B_t" carry the
        # FULL per-step stack alongside it, additively -- existing consumers
        # that only ever read "A"/"B" (e.g. `unfolding_landscape_inputs`,
        # every NB03/NB04 cached artifact) are unaffected; a caller that
        # wants the true per-step dynamics reads these new keys instead.
        "A": system.A_t[0],
        "B": system.B_t[0],
        "A_t": np.stack([system.A_t[k] for k in range(horizon)]),
        "B_t": np.stack([system.B_t[k] for k in range(horizon)]),
        "R": time_invariant_slice(cost.R),
    }
    if "riccati_matrix" in controller.config.parameters:
        parameters["riccati_matrix"] = controller.config.parameters[
            "riccati_matrix"
        ].get
    else:
        P_arr, _ = finite_horizon_riccati(system, cost, horizon)
        parameters["P"] = P_arr[0]
    return parameters


def unfolded_parameter_log_summaries(
    controller: UnfoldedController,
) -> dict[str, Callable[[], str]]:
    """Zero-argument formatters for `StructuredTrainingLogCallback`'s
    per-epoch "Learned Parameters" column.

    Each callable re-reads `controller.config.parameters` and reparameterizes
    it (`UnfoldedParameter.get_numpy`) fresh every time it is invoked, so the
    narrated value tracks live training progress rather than a value frozen
    at recipe-build time (before any epoch has run).

    Both learned quantities are displayed IN FULL, each on its own line(s)
    (NB03 Phase B, directive 2): `step_size` as its explicit per-iteration
    ``num_iterations x action_dim`` array, and `riccati_matrix` as the FULL
    ``state_dim x state_dim`` learned matrix replacing the Riccati P (not just
    its Frobenius norm) -- so a reader can inspect exactly which matrix the
    contender converged to, and how far it sits from the true Riccati P. The
    multi-line matrix string is indented per-line by
    `engine.callbacks.render_parameter_block`.

    Args:
        controller: The (about to be trained) `UnfoldedController`.

    Returns:
        name -> zero-arg callable returning that parameter's current
        formatted value (possibly multi-line); empty if `controller` has no
        learnable parameters matching either of the two known shapes.
    """
    summaries: dict[str, Callable[[], str]] = {}
    if "step_size" in controller.config.parameters:
        step_size = controller.config.parameters["step_size"]
        summaries["step_size (alpha)"] = lambda: np.array2string(
            step_size.get_numpy(), precision=4, separator=","
        )
    if "riccati_matrix" in controller.config.parameters:
        riccati_matrix = controller.config.parameters["riccati_matrix"]
        summaries["P (full learned matrix)"] = lambda: np.array2string(
            riccati_matrix.get_numpy(), precision=4, separator=",", suppress_small=True
        )
    return summaries


@register_recipe("unfolded")
@dataclass(frozen=True)
class UnfoldedRecipe(TrainableRecipe):
    """A gradient-trained unfolded configuration (`learned_step_size` or
    `learned_step_size_and_matrix`).

    Attributes:
        kind: Which learned configuration to build.
        plan: This family's declarative `TrainingPlan` (T3.b).
        num_iterations: Unfolding iteration count.
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        horizon: The rollout horizon.
        label: Instance name; defaults to ``"unfolded_<kind>"``.
        log_first_epochs: ``a`` of the dynamic narration schedule (directive
            2) forwarded to `StructuredTrainingLogCallback`.
        log_every_epochs: ``b`` of the dynamic narration schedule.
    """

    family: ClassVar[str] = "unfolded"

    kind: UnfoldedKind
    plan: TrainingPlan
    num_iterations: int
    step_size_init: float
    step_size_max: float
    horizon: int
    label: str = ""
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    init_method: ControlInitMethod = ControlInitMethod.COLD
    random_init_std: float = 1.0
    random_init_seed: int | None = None

    def __post_init__(self) -> None:
        """Default the label from `kind`, reject the fixed configuration
        (that one never trains -- use `FixedUnfoldedRecipe`), and refuse a step
        size the reparameterization could only reach by saturating.

        The step-size refusal lives here, at the recipe, because this is the
        tier that fires **while the document is open**: loading and
        materialising a study constructs the recipes and zero
        `StepSizeParameter`s, so the Tier-2 guard alone would not report a
        typo until training had started.

        Raises:
            ValueError: If `kind` is `UnfoldedKind.FIXED`, or `step_size_init`
                does not lie strictly inside ``(0, step_size_max)``.
        """
        accept_enum_values(self, kind=UnfoldedKind, init_method=ControlInitMethod)
        if self.kind is UnfoldedKind.FIXED:
            raise ValueError(
                "UnfoldedRecipe trains its parameters; the fixed configuration "
                "is FixedUnfoldedRecipe."
            )
        require_saturating_bound(
            type(self).__name__, "step_size", self.step_size_init, self.step_size_max
        )
        if not self.label:
            object.__setattr__(self, "label", f"unfolded_{self.kind.value}")

    def get_signature(self) -> dict[str, Any]:
        """See `ModelRecipe.get_signature`; the control-init selection folds in
        only when non-default (`_unfolded_recipe_signature`), so a cold-start
        recipe's digest -- and the golden masters -- are untouched by this knob.
        """
        return _unfolded_recipe_signature(
            self.init_method,
            self.random_init_std,
            self.random_init_seed,
            super().get_signature(),
        )

    def _build_spec(self) -> UnfoldedBuildSpec:
        return UnfoldedBuildSpec(
            kind=self.kind,
            num_iterations=self.num_iterations,
            step_size_init=self.step_size_init,
            step_size_max=self.step_size_max,
            horizon=self.horizon,
            init_method=self.init_method,
            random_init_std=self.random_init_std,
            random_init_seed=self.random_init_seed,
        )

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> UnfoldedController:
        """See `ModelRecipe.build_controller`."""
        return build_unfolded_controller(problem, ctx, self._build_spec())

    def build_rehosted_controller(
        self,
        offline_problem: OptimalControlProblem,
        online_problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> TrainedControllerArtifact:
        """Annex 01 §2.4.1: the offline Riccati stack stays the told plant's;
        the per-step gradient coefficients form from it with the given
        matrices. The learned-P kinds carry no offline stack, and for them
        this reduces to the base construction on the online plant."""
        controller = build_unfolded_controller(
            online_problem, ctx, self._build_spec(), riccati_problem=offline_problem
        )
        return TrainedControllerArtifact(
            controller=controller,
            context=ctx,
            synthesizer_signature={
                "type": "RehostedRebuild",
                "recipe": self.get_signature(),
            },
        )

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> list[Callback]:
        """The convergence-replay parameter snapshot plus the structured,
        per-epoch training-log narration (family-owned, no app involvement)."""
        assert isinstance(controller, UnfoldedController)
        return [
            ParameterSnapshotCallback(
                unfolded_convergence_parameters(
                    controller, problem, horizon=self.horizon
                )
            ),
            StructuredTrainingLogCallback(
                unfolded_parameter_log_summaries(controller),
                log_first_epochs=self.log_first_epochs,
                log_every_epochs=self.log_every_epochs,
                unfolding_depth=self.num_iterations,
            ),
        ]


@register_recipe("unfolded_fixed")
@dataclass(frozen=True)
class FixedUnfoldedRecipe(AnalyticRecipe):
    """The frozen unfolded configuration: fixed step size, true Riccati P,
    nothing learned -- evaluated, never trained.

    Attributes:
        num_iterations: Unfolding iteration count.
        step_size_init: The (fixed) per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        horizon: The rollout horizon.
        label: Instance name; defaults to ``"unfolded_fixed"``.
    """

    family: ClassVar[str] = "unfolded_fixed"

    num_iterations: int
    step_size_init: float
    step_size_max: float
    horizon: int
    label: str = "unfolded_fixed"
    init_method: ControlInitMethod = ControlInitMethod.COLD
    random_init_std: float = 1.0
    random_init_seed: int | None = None

    def __post_init__(self) -> None:
        """Accept `init_method` as its string value (`accept_enum_values`), and
        refuse a step size outside its own bound.

        This family is the analytic PGD, whose step size is the whole of its
        behaviour: it never trains, so a silently clamped step is not an
        initialization that gradient descent would move away from, it is the
        controller.

        Raises:
            ValueError: If `step_size_init` does not lie strictly inside
                ``(0, step_size_max)``.
        """
        accept_enum_values(self, init_method=ControlInitMethod)
        require_saturating_bound(
            type(self).__name__, "step_size", self.step_size_init, self.step_size_max
        )

    def get_signature(self) -> dict[str, Any]:
        """See `ModelRecipe.get_signature`; the control-init selection folds in
        only when non-default (`_unfolded_recipe_signature`), so a cold-start
        recipe's digest -- and the golden masters -- are untouched by this knob.
        """
        return _unfolded_recipe_signature(
            self.init_method,
            self.random_init_std,
            self.random_init_seed,
            super().get_signature(),
        )

    def _build_spec(self) -> UnfoldedBuildSpec:
        return UnfoldedBuildSpec(
            kind=UnfoldedKind.FIXED,
            num_iterations=self.num_iterations,
            step_size_init=self.step_size_init,
            step_size_max=self.step_size_max,
            horizon=self.horizon,
            init_method=self.init_method,
            random_init_std=self.random_init_std,
            random_init_seed=self.random_init_seed,
        )

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> UnfoldedController:
        """See `ModelRecipe.build_controller`."""
        return build_unfolded_controller(problem, ctx, self._build_spec())

    def build_rehosted_controller(
        self,
        offline_problem: OptimalControlProblem,
        online_problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> TrainedControllerArtifact:
        """Annex 01 §2.4.1: P frozen from the told plant, and the step stays
        the declared literal — an offline artifact by the same law, which is
        why no per-plant step is recomputed here."""
        controller = build_unfolded_controller(
            online_problem, ctx, self._build_spec(), riccati_problem=offline_problem
        )
        return TrainedControllerArtifact(
            controller=controller,
            context=ctx,
            synthesizer_signature={
                "type": "RehostedRebuild",
                "recipe": self.get_signature(),
            },
        )


@register_recipe("unfolded_warmstart")
@dataclass(frozen=True)
class WarmStartUnfoldedRecipe(ModelRecipe):
    """A greedy layer-wise ("warm-start") trained unfolded configuration
    (NB03 blueprint v2 SS3, the flagship build): reuses
    `build_unfolded_controller` verbatim (identical construction to
    `UnfoldedRecipe`) but wires its engine from a `LayerwiseTrainingPlan`
    schedule instead of a flat `TrainingPlan` -- one `TrainingPhase` per
    unfolding iteration (each running its OWN, freshly-built
    `LayerwiseGradientDescentStrategy`/optimizer -- the anti-momentum-
    carryover law, SS7.1), then an optional trailing end-to-end refinement
    phase.

    Subclasses `ModelRecipe` directly rather than `TrainableRecipe`:
    `TrainableRecipe` declares a single flat ``plan: TrainingPlan`` field
    that its own `build_engine`/`build_synthesizer` (and the reused
    `EngineTrainedSynthesizer`) read directly, and this family's training
    specification is a whole multi-phase `LayerwiseTrainingPlan` SCHEDULE,
    not a single plan -- a different shape, not a narrower one. Both
    `build_engine` and `build_synthesizer` are therefore implemented here
    directly (see `WarmStartSynthesizer`); `Runner` itself needs no
    changes -- it already accepts any ``Sequence[TrainingPhase]``.

    Attributes:
        kind: Which learned configuration to build (`FIXED` is rejected --
            see `__post_init__`, the same guard `UnfoldedRecipe` applies).
        schedule: This family's declarative `LayerwiseTrainingPlan`.
        num_iterations: Unfolding iteration count (one warm-up phase each).
        step_size_init: Initial per-iteration step size.
        step_size_max: Step-size reparameterization upper bound.
        horizon: The rollout horizon.
        label: Instance name; defaults to ``"unfolded_warmstart_<kind>"``.
        log_first_epochs: ``a`` of the dynamic narration schedule (directive
            2) forwarded to `StructuredTrainingLogCallback`.
        log_every_epochs: ``b`` of the dynamic narration schedule.
    """

    family: ClassVar[str] = "unfolded_warmstart"

    kind: UnfoldedKind
    schedule: LayerwiseTrainingPlan
    num_iterations: int
    step_size_init: float
    step_size_max: float
    horizon: int
    label: str = ""
    log_first_epochs: int = 10
    log_every_epochs: int = 20
    init_method: ControlInitMethod = ControlInitMethod.COLD
    random_init_std: float = 1.0
    random_init_seed: int | None = None

    def __post_init__(self) -> None:
        """Default the label from `kind`, and reject the fixed configuration
        (mirrors `UnfoldedRecipe.__post_init__`: a warm-start schedule trains
        parameters; `FixedUnfoldedRecipe` already serves the frozen case).

        Raises:
            ValueError: If `kind` is `UnfoldedKind.FIXED`, or `step_size_init`
                does not lie strictly inside ``(0, step_size_max)``.
        """
        accept_enum_values(self, kind=UnfoldedKind, init_method=ControlInitMethod)
        if self.kind is UnfoldedKind.FIXED:
            raise ValueError(
                "WarmStartUnfoldedRecipe trains its parameters; the fixed "
                "configuration is FixedUnfoldedRecipe."
            )
        require_saturating_bound(
            type(self).__name__, "step_size", self.step_size_init, self.step_size_max
        )
        if not self.label:
            object.__setattr__(self, "label", f"unfolded_warmstart_{self.kind.value}")

    def get_signature(self) -> dict[str, Any]:
        """See `ModelRecipe.get_signature`; the control-init selection folds in
        only when non-default (`_unfolded_recipe_signature`), so a cold-start
        recipe's digest -- and the golden masters -- are untouched by this knob.
        """
        return _unfolded_recipe_signature(
            self.init_method,
            self.random_init_std,
            self.random_init_seed,
            super().get_signature(),
        )

    def build_rehosted_controller(
        self,
        offline_problem: OptimalControlProblem,
        online_problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> TrainedControllerArtifact:
        """Annex 01 §2.4.1, identical split to `UnfoldedRecipe`'s."""
        controller = build_unfolded_controller(
            online_problem,
            ctx,
            UnfoldedBuildSpec(
                kind=self.kind,
                num_iterations=self.num_iterations,
                step_size_init=self.step_size_init,
                step_size_max=self.step_size_max,
                horizon=self.horizon,
                init_method=self.init_method,
                random_init_std=self.random_init_std,
                random_init_seed=self.random_init_seed,
            ),
            riccati_problem=offline_problem,
        )
        return TrainedControllerArtifact(
            controller=controller,
            context=ctx,
            synthesizer_signature={
                "type": "RehostedRebuild",
                "recipe": self.get_signature(),
            },
        )

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> UnfoldedController:
        """See `ModelRecipe.build_controller`; identical construction to
        `UnfoldedRecipe` (the shared `build_unfolded_controller` home)."""
        return build_unfolded_controller(
            problem,
            ctx,
            UnfoldedBuildSpec(
                kind=self.kind,
                num_iterations=self.num_iterations,
                step_size_init=self.step_size_init,
                step_size_max=self.step_size_max,
                horizon=self.horizon,
                init_method=self.init_method,
                random_init_std=self.random_init_std,
                random_init_seed=self.random_init_seed,
            ),
        )

    def build_engine(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> Engine:
        """Overrides `TrainableRecipe.build_engine` entirely (never calls
        ``super()``): compiles `self.schedule` into `PhaseSpec`s, builds ONE
        `LayerwiseGradientDescentStrategy` per phase -- each with its own
        freshly-constructed optimizer (SS7.1's anti-momentum-carryover law)
        -- and assembles them into the `Runner`'s ``phases`` sequence.
        """
        assert isinstance(controller, UnfoldedController)
        rollout = RolloutModel(controller)
        module = controller.as_module()
        has_matrix = "riccati_matrix" in controller.config.parameters
        freezer = LayerFreezeBuilder()
        phase_specs = self.schedule.compile(self.num_iterations, has_matrix=has_matrix)
        strategies = [
            LayerwiseGradientDescentStrategy(
                rollout,
                module,
                self.schedule.optimizer,
                freezer.build(spec.activation, controller.config.parameters),
                execution=StepExecution(
                    loss_reduction=self.schedule.resolve_loss_reduction(),
                    gradient_clip_norm=self.schedule.gradient_clip_norm,
                    # From the HARNESS, never the schedule: the schedule is
                    # signed into `ModelID` and D20 excludes the microbatch
                    # (Annex 06 §4.3).
                    microbatch=harness.microbatch,
                ),
            )
            for spec in phase_specs
        ]
        phases = [
            TrainingPhase(strategy, epochs=spec.epochs, name=spec.name)
            for strategy, spec in zip(strategies, phase_specs, strict=True)
        ]
        batch_sampler, distributions = self._make_sampler(harness)
        callbacks = self._assemble_callbacks(
            controller, problem, harness, distributions
        )
        # The FINAL phase's optimizer is the meaningful resumable state
        # (SS3.4's resumability note): earlier phases' optimizers are
        # deliberately discarded once their own phase ends.
        callbacks.append(
            TrainingStateCheckpointCallback(module, strategies[-1].optimizer)
        )
        config = self.schedule.training_config(
            batch_size=self._effective_batch_size(harness), log_every=harness.log_every
        )
        return Runner(
            model=rollout,
            tracker=harness.tracker,
            config=config,
            batch_sampler=batch_sampler,
            phases=phases,
            callbacks=callbacks,
        )

    def build_synthesizer(
        self, problem: OptimalControlProblem, harness: EngineHarness
    ) -> Synthesizer:
        """The engine-backed synthesizer (T3.a), schedule-aware sibling of
        `EngineTrainedSynthesizer` (see `WarmStartSynthesizer`)."""
        return WarmStartSynthesizer(recipe=self, harness=harness)

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> list[Callback]:
        """The same convergence-replay parameter snapshot and structured
        training-log narration `UnfoldedRecipe` attaches -- family-owned, no
        app involvement."""
        assert isinstance(controller, UnfoldedController)
        return [
            ParameterSnapshotCallback(
                unfolded_convergence_parameters(
                    controller, problem, horizon=self.horizon
                )
            ),
            StructuredTrainingLogCallback(
                unfolded_parameter_log_summaries(controller),
                log_first_epochs=self.log_first_epochs,
                log_every_epochs=self.log_every_epochs,
                unfolding_depth=self.num_iterations,
            ),
        ]


@dataclass(frozen=True)
class WarmStartSynthesizer:
    """The layer-wise-schedule-aware sibling of `EngineTrainedSynthesizer`
    (T3.a): identical offline-phase mechanics (build controller -> build
    engine -> train -> freeze into a `TrainedControllerArtifact`), but reads
    its training provenance from ``recipe.schedule`` (a
    `LayerwiseTrainingPlan`) rather than ``recipe.plan`` (a flat
    `TrainingPlan`) -- `WarmStartUnfoldedRecipe` has no single flat plan for
    `EngineTrainedSynthesizer`'s own ``recipe.plan.get_signature()``
    provenance line to read, so this sibling exists rather than forcing
    `WarmStartUnfoldedRecipe` to fake a `TrainableRecipe.plan: TrainingPlan`
    it does not structurally have. Stateless (frozen): every call trains a
    fresh controller.

    Attributes:
        recipe: The owning `WarmStartUnfoldedRecipe`.
        harness: The execution resources; construct with a
            `NullExperimentTracker`-backed harness for untracked synthesis.
    """

    recipe: WarmStartUnfoldedRecipe
    harness: EngineHarness

    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext | None = None
    ) -> TrainedControllerArtifact:
        """See `EngineTrainedSynthesizer.synthesize`; identical mechanics,
        with provenance sourced from ``recipe.schedule`` instead of
        ``recipe.plan``.

        Args:
            problem: The `OptimalControlProblem` to train against.
            ctx: Optional context override; defaults to the harness context.

        Returns:
            The frozen `TrainedControllerArtifact`.
        """
        effective_ctx = ctx if ctx is not None else self.harness.ctx
        harness = (
            self.harness
            if effective_ctx == self.harness.ctx
            else dataclasses.replace(self.harness, ctx=effective_ctx)
        )
        logger.info("Synthesis (warm-start training) started for %r", self.recipe.label)
        controller = self.recipe.build_controller(problem, effective_ctx)
        engine = self.recipe.build_engine(controller, problem, harness)
        final_metrics = dict(engine.train())
        logger.info("Synthesis finished for %r: %s", self.recipe.label, final_metrics)
        return TrainedControllerArtifact(
            controller=controller,
            context=effective_ctx,
            synthesizer_signature=self.get_signature(),
            provenance={
                "training": self.recipe.schedule.get_signature(),
                "final_metrics": final_metrics,
            },
        )

    def get_signature(self) -> dict[str, Any]:
        """Specification-time configuration only (C1 law): the recipe.

        Returns:
            ``{"type": "WarmStartSynthesizer", "recipe": {...}}``.
        """
        return {"type": type(self).__name__, "recipe": self.recipe.get_signature()}
