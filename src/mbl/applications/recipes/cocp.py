"""Recipes for the COCP (convex-optimization control policy) family: the
gradient-trained one-step-QP policy seeded at the unconstrained DARE
solution, and its frozen variant fixed at the box-constrained SDP's
cost-to-go (a strong non-learned reference whose theoretical bound is
logged as declared provenance -- the sanctioned replacement for the M10
``cocp_lower_bound.lower_bound_value = ...`` attribute smuggling)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.linalg import solve_discrete_are, sqrtm

from .base import AnalyticRecipe, EngineHarness, TrainableRecipe, register_recipe
from ...core.constraint.box_constraint import BoxConstraint
from ...core.kernels import time_invariant_slice
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import ComputeContext
from ...engine.callbacks import (
    Callback,
    ParameterSnapshotCallback,
    StructuredTrainingLogCallback,
)
from ...engine.training_plan import TrainingPlan
from ...models.base import Controller
from ...models.constrained.cocp import COCPConfig, COCPController
from ...models.constrained.lower_bound import solve_box_constrained_lower_bound
from ...models.constrained.solver_resolution import (
    DEFAULT_COCP_BACKEND,
    COCPCanaryInstance,
    require_validated_cocp_solver,
)
from ...models.constrained.solver_spec import SUPPORTED_SOLVERS, COCPSolverSpec
from ...models.guards import require_linear_quadratic


def require_declarable_backend(name: str, family: str) -> None:
    """Refuse a solver backend no `COCPSolverSpec` could be built with (D17).

    A free function with two call sites — one per COCP family — rather than a
    check inside `build_controller`, because *where* it fires is the whole
    point: a recipe is constructed while the study document is still open, so a
    mistyped backend is refused with the contender's own position rather than as
    a `ValueError` an hour into a run.

    Args:
        name: The declared backend.
        family: The declaring family, for the message.

    Raises:
        ValueError: If `name` is not one `COCPSolverSpec` supports. `ValueError`
            deliberately, not a spec-tier error: `applications` sits below the
            grammar, and the loader already wraps a recipe's `ValueError` into a
            `SpecificationError` naming `contenders[i]`.
    """
    if name not in SUPPORTED_SOLVERS:
        raise ValueError(
            f"{family} declares solver_backend={name!r}, which does not exist; "
            f"supported: {sorted(SUPPORTED_SOLVERS)}. The backend is part of "
            "what a model IS (D17), so it is declared and validated, never "
            "guessed from what happens to be installed"
        )


def _declared_solver(
    name: str, device: str, eps: float, max_iters: int
) -> COCPSolverSpec:
    """The solver spec a contender declared. Not resolved: constructed."""
    return COCPSolverSpec(name=name, device=device, eps=eps, max_iters=max_iters)


def _box_constraint(problem: OptimalControlProblem) -> BoxConstraint:
    """The COCP family's fail-fast gate: the problem's first constraint,
    narrowed to the `BoxConstraint` its one-step QP is built around.

    Raises:
        TypeError: If the problem declares no constraints, or the first is
            not a `BoxConstraint`.
    """
    constraints = problem.constraints or []
    if not constraints or not isinstance(constraints[0], BoxConstraint):
        raise TypeError(
            "The COCP family requires problem.constraints[0] to be a BoxConstraint."
        )
    return constraints[0]


def cocp_convergence_parameters(
    controller: COCPController, problem: OptimalControlProblem
) -> dict:
    """COCP's learned cost-to-go (P_sqrt, q) + the problem matrices/box
    bound needed to re-solve its one-step QP later for a convergence
    comparison against the unfolded models.

    Args:
        controller: The `COCPController` whose live parameters to snapshot.
        problem: The problem whose matrices the replay needs.

    Returns:
        The name -> live tensor/array mapping to snapshot at train end.
    """
    system, cost = require_linear_quadratic(problem)
    return {
        "P_sqrt": controller.P_sqrt,
        "q": controller.q,
        "A": system.A_t.array,
        "B": system.B_t.array,
        "R": time_invariant_slice(cost.R),
        "u_max": np.asarray(controller.constraint.u_max),
    }


def cocp_parameter_log_summaries(
    controller: COCPController,
) -> dict[str, Callable[[], str]]:
    """Zero-argument formatters for `StructuredTrainingLogCallback`'s
    per-epoch "Learned Parameters" column -- COCP's own learned cost-to-go
    ``(P_sqrt, q)``, mirroring `applications.recipes.unfolded
    .unfolded_parameter_log_summaries`'s identical convention: each callable
    re-reads the LIVE controller fresh at every logged epoch (never a value
    snapshotted once before training starts), so the narrated matrix/vector
    tracks live training progress. Previously COCP trained with NO per-epoch
    narration at all, unlike every other learned contender -- leaving its
    (often much longer, QP-per-timestep) training run with no progress
    visibility.

    Args:
        controller: The (about to be trained) `COCPController`.

    Returns:
        name -> zero-arg callable returning that parameter's current
        formatted value.
    """
    return {
        "P_sqrt (learned cost-to-go sqrt)": lambda: np.array2string(
            controller.P_sqrt.detach().cpu().numpy(),
            precision=4,
            separator=",",
            suppress_small=True,
        ),
        "q (learned linear term)": lambda: np.array2string(
            controller.q.detach().cpu().numpy(), precision=4, separator=","
        ),
    }


@register_recipe("cocp")
@dataclass(frozen=True)
class COCPRecipe(TrainableRecipe):
    """The gradient-trained one-step convex-QP policy, seeded at the
    unconstrained DARE solution's matrix square root.

    COCP solves one QP per (batch element, time step) via cvxpylayers/SCS --
    orders of magnitude more expensive per sample than the other families --
    so it declares its own (smaller) batch/horizon overrides.

    Attributes:
        plan: This family's declarative `TrainingPlan` (T3.b).
        solver_eps: Solver convergence tolerance, translated into the
            resolved solver's own vocabulary (`COCPSolverSpec.to_solver_args`).
        solver_max_iters: Solver iteration cap, translated the same way.
        batch_size: Optional training/evaluation batch override.
        horizon: Optional training/evaluation horizon override.
        label: Instance name; defaults to ``"cocp"``.
        solver_backend: **Which solver produces this model's gradients**
            (D17). Declared, never resolved: it is part of what the model IS
            and is signed into its `ModelID`, and the canary that used to
            *choose* it now only validates the declaration and refuses one
            this machine cannot honour. Defaults to the measured default
            (`DEFAULT_COCP_BACKEND`, MOREAU) -- see Annex 01 §5 rule 5 for
            why a default here has to be a measurement.
        solver_device: The device the declared solver runs on --
            DELIBERATELY independent of `ctx.device` (the REST of the
            experiment's device), defaulting to ``"cpu"`` because that is
            the pairing the measurement supports: MOREAU-CPU's gradients
            matched DIFFCP/SCS to ~1e-7, MOREAU-CUDA's disagreed by up to
            ~50%, seed-dependent (NB04 COCP/viz refinement plan Sec 1.0.6).
            Declaring ``"cuda"`` is allowed and is refused by the validator
            on the machines where that finding holds -- loudly, rather than
            by the silent fall-back to DIFFCP/SCS this used to perform.
        log_first_epochs: ``a`` of the dynamic narration schedule (directive
            2) forwarded to `StructuredTrainingLogCallback` -- same
            convention as `applications.recipes.unfolded.UnfoldedRecipe`.
            COCP has no unfolding depth, so its narrated header omits that
            field.
        log_every_epochs: ``b`` of the dynamic schedule.
    """

    family: ClassVar[str] = "cocp"

    plan: TrainingPlan
    solver_eps: float = 1e-8
    solver_max_iters: int = 10000
    batch_size: int | None = None
    horizon: int | None = None
    label: str = "cocp"
    solver_backend: str = DEFAULT_COCP_BACKEND
    solver_device: str = "cpu"
    log_first_epochs: int = 10
    log_every_epochs: int = 20

    def __post_init__(self) -> None:
        """Refuse a backend that does not exist, while the document is open."""
        require_declarable_backend(self.solver_backend, self.family)

    def batch_overrides(self) -> tuple[int | None, int | None]:
        """This family's smaller training scale (see class docstring)."""
        return self.batch_size, self.horizon

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> COCPController:
        """See `ModelRecipe.build_controller`; seeds the learnable cost-to-go
        at the unconstrained DARE solution and builds the QP layer with the
        solver this contender **declared**.

        Nothing is resolved here (D17). The declaration is validated against
        DIFFCP on this problem's own shape and the run is refused if this
        machine cannot honour it -- never switched to whatever does work, which
        is how one study came to hold two models whose gradients were produced
        by different solvers under identifiers that said nothing about it.
        """
        system, cost = require_linear_quadratic(problem)
        A = system.A_t.array
        B = system.B_t.array
        Q_2d = time_invariant_slice(cost.Q)
        R_2d = np.asarray(time_invariant_slice(cost.R))
        P_are = solve_discrete_are(A, B, Q_2d, R_2d)
        P_sqrt_init = np.real(sqrtm(P_are))
        u_max = float(np.asarray(_box_constraint(problem).u_max))
        n = A.shape[0]
        solver = _declared_solver(
            self.solver_backend,
            self.solver_device,
            self.solver_eps,
            self.solver_max_iters,
        )
        require_validated_cocp_solver(
            COCPCanaryInstance(
                A=A,
                B=B,
                R=R_2d,
                u_max=u_max,
                x=np.ones(n),
                P_sqrt=P_sqrt_init,
                q=np.zeros(n),
            ),
            solver,
        )
        return COCPController(
            problem,
            _box_constraint(problem),
            COCPConfig(
                P_sqrt_init=P_sqrt_init,
                dtype=ctx.torch_dtype,
                solver=solver,
            ),
        )

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> list[Callback]:
        """The QP-replay parameter snapshot plus the structured, per-epoch
        training-log narration (family-owned) -- previously missing
        entirely, unlike every other learned contender, leaving COCP's
        (often much longer, one-QP-solve-per-timestep) training run with no
        progress visibility at all."""
        assert isinstance(controller, COCPController)
        return [
            ParameterSnapshotCallback(cocp_convergence_parameters(controller, problem)),
            StructuredTrainingLogCallback(
                cocp_parameter_log_summaries(controller),
                log_first_epochs=self.log_first_epochs,
                log_every_epochs=self.log_every_epochs,
            ),
        ]


@register_recipe("cocp_lower_bound")
@dataclass(frozen=True)
class COCPLowerBoundRecipe(AnalyticRecipe):
    """The same QP policy, fixed (never trained) at the box-constrained
    SDP's cost-to-go -- evaluated as a strong non-learned reference, with
    the SDP's theoretical bound value logged as declared provenance.

    Scored via the shared problem cost (whose Q/R are time-stacked to the
    MAIN horizon), so -- unlike `COCPRecipe` -- only the batch size may be
    reduced here, never the horizon.

    Attributes:
        process_noise_std: The evaluation noise level; the SDP bound is
            computed for ``W = process_noise_std**2 * I``.
        solver_eps: Solver convergence tolerance, translated into the
            resolved solver's own vocabulary (`COCPSolverSpec.to_solver_args`).
        solver_max_iters: Solver iteration cap, translated the same way.
        batch_size: Optional evaluation batch override.
        label: Instance name; defaults to ``"cocp_lower_bound"``.
        solver_backend: The declared solver (D17) -- see `COCPRecipe`'s
            identical field. This family is the reason the decision exists:
            it canaries at the SDP's cost-to-go while `COCPRecipe` canaries
            at the DARE's, and under the old resolve-and-fall-back the two
            families of one study were observed choosing different solvers.
        solver_device: The device the declared solver runs on -- see
            `COCPRecipe`'s identical field for the full rationale
            (deliberately independent of `ctx.device`, defaulting to
            ``"cpu"``).
    """

    family: ClassVar[str] = "cocp_lower_bound"

    process_noise_std: float
    solver_eps: float = 1e-8
    solver_max_iters: int = 10000
    batch_size: int | None = None
    label: str = "cocp_lower_bound"
    solver_backend: str = DEFAULT_COCP_BACKEND
    solver_device: str = "cpu"

    def __post_init__(self) -> None:
        """Refuse a backend that does not exist, while the document is open."""
        require_declarable_backend(self.solver_backend, self.family)

    def batch_overrides(self) -> tuple[int | None, int | None]:
        """Batch-size-only override (see class docstring)."""
        return self.batch_size, None

    def _solve_lower_bound(
        self, problem: OptimalControlProblem
    ) -> tuple[float, np.ndarray]:
        """The box-constrained SDP: its optimal value is the theoretical
        cost lower bound, its solution matrix the frozen cost-to-go seed."""
        system, cost = require_linear_quadratic(problem)
        A = system.A_t.array
        B = system.B_t.array
        Q_2d = np.asarray(time_invariant_slice(cost.Q))
        R_2d = np.asarray(time_invariant_slice(cost.R))
        W = self.process_noise_std**2 * np.eye(A.shape[0])
        u_max = float(np.asarray(_box_constraint(problem).u_max))
        return solve_box_constrained_lower_bound(A, B, Q_2d, R_2d, W, u_max)

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> COCPController:
        """See `ModelRecipe.build_controller`; the QP's cost-to-go is the SDP
        solution and both learnable parameters are frozen. The declared solver
        is validated exactly as `COCPRecipe` validates its own.

        **This family is why D17 exists.** Its canary query point is seeded from
        the SDP solution while `COCPRecipe`'s is seeded from the DARE solution,
        so under the old resolve-and-fall-back the two families of one study
        could -- and once did -- end up on different solvers. A declaration
        cannot: the query point now decides nothing.
        """
        _, P_lb = self._solve_lower_bound(problem)
        system, cost = require_linear_quadratic(problem)
        A = system.A_t.array
        B = system.B_t.array
        R_2d = np.asarray(time_invariant_slice(cost.R))
        P_sqrt_init = np.real(sqrtm(P_lb))
        u_max = float(np.asarray(_box_constraint(problem).u_max))
        n = A.shape[0]
        solver = _declared_solver(
            self.solver_backend,
            self.solver_device,
            self.solver_eps,
            self.solver_max_iters,
        )
        require_validated_cocp_solver(
            COCPCanaryInstance(
                A=A,
                B=B,
                R=R_2d,
                u_max=u_max,
                x=np.ones(n),
                P_sqrt=P_sqrt_init,
                q=np.zeros(n),
            ),
            solver,
        )
        controller = COCPController(
            problem,
            _box_constraint(problem),
            COCPConfig(
                P_sqrt_init=P_sqrt_init,
                dtype=ctx.torch_dtype,
                solver=solver,
            ),
        )
        controller.P_sqrt.requires_grad_(False)
        controller.q.requires_grad_(False)
        return controller

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> list[Callback]:
        """Log the SDP's theoretical bound alongside the achieved simulated
        cost (declared provenance, replacing the M10 attribute smuggling),
        plus the QP-replay parameter snapshot.

        The SDP is deliberately re-solved here rather than smuggled from
        `build_controller` through shared mutable state: the two hooks stay
        independent and the recipe stays frozen; the solve is small (one
        semidefinite program in ``n``) next to the evaluation rollout.
        """
        assert isinstance(controller, COCPController)
        lower_bound_value, _ = self._solve_lower_bound(problem)
        harness.tracker.log_params({"sdp_lower_bound_value": lower_bound_value})
        return [
            ParameterSnapshotCallback(cocp_convergence_parameters(controller, problem))
        ]

    def provenance(self, problem: OptimalControlProblem) -> Mapping[str, float]:
        """This family's declared synthesis provenance (M10 replacement):
        the SDP bound, exposed for artifact consumers.

        Returns:
            ``{"sdp_lower_bound_value": <float>}``.
        """
        lower_bound_value, _ = self._solve_lower_bound(problem)
        return {"sdp_lower_bound_value": lower_bound_value}
