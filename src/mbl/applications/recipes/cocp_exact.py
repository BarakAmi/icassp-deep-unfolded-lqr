"""Recipe for the exact convex policy: the same one-step QP, solved as a box-QP.

Additive by construction. `applications.recipes.cocp` and the two families it
registers are untouched, so every stored `cocp`/`cocp_lower_bound` model keeps
its identifier; `register_recipe` raises on a duplicate name, which is the
structural guarantee that these two cannot reach each other.

Two fields the cvxpylayers recipe carries are deliberately absent. There is no
``solver_backend``: this family solves no cone program, so there is nothing to
declare and nothing for a machine to fail to honour. And there is no
``solver_device``: nothing here leaves torch, so the family follows the run's
`ComputeContext` like every other contender instead of carrying a second,
independently declared device.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.linalg import solve_discrete_are, sqrtm

from .base import AnalyticRecipe, EngineHarness, TrainableRecipe, register_recipe
from .cocp import _box_constraint
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
from ...models.constrained.box_lagrangian import (
    BoxLQRData,
    DualBound,
    infinite_horizon_box_bound,
)
from ...models.constrained.cocp_exact import ExactCOCPConfig, ExactCOCPController
from ...models.guards import require_linear_quadratic


def exact_cocp_convergence_parameters(
    controller: ExactCOCPController, problem: OptimalControlProblem
) -> dict:
    """The learned cost-to-go plus the plant a later replay needs to re-solve.

    Args:
        controller: The controller whose live parameters to snapshot.
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


def exact_cocp_parameter_log_summaries(
    controller: ExactCOCPController,
) -> dict[str, Callable[[], str]]:
    """Per-epoch narration formatters, re-read live at every logged epoch.

    Args:
        controller: The (about to be trained) controller.

    Returns:
        name -> zero-arg callable returning that parameter's formatted value.
    """
    summaries: dict[str, Callable[[], str]] = {
        "P_sqrt (learned cost-to-go sqrt)": lambda: np.array2string(
            controller.P_sqrt.detach().cpu().numpy(),
            precision=4,
            separator=",",
            suppress_small=True,
        ),
    }
    if controller.q is not None:
        summaries["q (learned linear term)"] = lambda: np.array2string(
            controller.q.detach().cpu().numpy(),  # type: ignore[union-attr]
            precision=4,
            separator=",",
        )
    return summaries


@register_recipe("cocp_exact")
@dataclass(frozen=True)
class ExactCOCPRecipe(TrainableRecipe):
    """The gradient-trained convex policy, differentiated through its own KKT
    system rather than through a differentiable cone program.

    Attributes:
        plan: This family's declarative `TrainingPlan` (T3.b).
        use_linear_term: Whether the policy carries the linear cost-to-go term.
            **Declared, and signed into `ModelID`**: measured, it changes the
            cost by 0.0002 % on a problem symmetric about the origin and by
            +3.9 % once a drift breaks that symmetry, so it is a modelling
            choice a study states rather than a constant the code picks.
            Defaults off, matching the formulation the COCP paper publishes
            for box-constrained LQR.
        batch_size: Optional training/evaluation batch override. Unlike the
            cvxpylayers family this exists for symmetry rather than necessity
            -- the QP is no longer what makes this contender expensive.
        horizon: Optional training/evaluation horizon override.
        label: Instance name; defaults to ``"cocp_exact"``.
        log_first_epochs: ``a`` of the dynamic narration schedule.
        log_every_epochs: ``b`` of the dynamic narration schedule.
    """

    family: ClassVar[str] = "cocp_exact"

    plan: TrainingPlan
    use_linear_term: bool = False
    batch_size: int | None = None
    horizon: int | None = None
    label: str = "cocp_exact"
    log_first_epochs: int = 10
    log_every_epochs: int = 20

    def batch_overrides(self) -> tuple[int | None, int | None]:
        """This family's optional training-scale overrides.

        Returns:
            ``(batch_size, horizon)`` as declared.
        """
        return self.batch_size, self.horizon

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> ExactCOCPController:
        """See `ModelRecipe.build_controller`.

        Seeds the learnable cost-to-go at the unconstrained DARE solution's
        matrix square root -- the same seed the cvxpylayers family uses, so the
        two differ in how they solve and not in where they start.

        Args:
            problem: The problem to control.
            ctx: The run's compute context; supplies the working precision, and
                the device, since this family declares neither itself.

        Returns:
            The constructed `ExactCOCPController`.
        """
        system, cost = require_linear_quadratic(problem)
        A = system.A_t.array
        B = system.B_t.array
        Q_2d = time_invariant_slice(cost.Q)
        R_2d = np.asarray(time_invariant_slice(cost.R))
        P_are = solve_discrete_are(A, B, Q_2d, R_2d)
        controller = ExactCOCPController(
            problem,
            _box_constraint(problem),
            ExactCOCPConfig(
                P_sqrt_init=np.real(sqrtm(P_are)),
                use_linear_term=self.use_linear_term,
                dtype=ctx.torch_dtype,
            ),
        )
        # THE RECIPE MOVES THE MODULE, as every other family's does. Saying
        # "this family follows `ctx`" is not the same as doing it: the
        # controller builds its parameters and its plant buffers with a dtype
        # and no device, so without this they stay on the CPU while the rollout
        # runs on the card and the first matmul fails. Found by a CUDA study,
        # not by the unit test that was meant to cover it -- that test called
        # `.to("cuda")` itself and so verified a condition it had created.
        return controller.to(device=ctx.torch_device)

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> list[Callback]:
        """The replay snapshot plus this family's per-epoch narration.

        Args:
            controller: The controller being trained.
            problem: The problem it controls.
            harness: The engine harness (unused here).

        Returns:
            The family-owned callbacks.
        """
        assert isinstance(controller, ExactCOCPController)
        return [
            ParameterSnapshotCallback(
                exact_cocp_convergence_parameters(controller, problem)
            ),
            StructuredTrainingLogCallback(
                exact_cocp_parameter_log_summaries(controller),
                log_first_epochs=self.log_first_epochs,
                log_every_epochs=self.log_every_epochs,
            ),
        ]


@register_recipe("cocp_exact_lower_bound")
@dataclass(frozen=True)
class ExactCOCPLowerBoundRecipe(AnalyticRecipe):
    """The same convex policy, frozen at the box-aware dual bound's cost-to-go.

    A strong non-learned reference, and the family whose *synthesis* was the
    expensive half. `COCPLowerBoundRecipe` reaches its seed through a
    semidefinite program that took **4.9 hours at n = 100 and came back flagged
    inaccurate**; the same bound is the Lagrangian dual of its own constraint
    and takes **6.3 seconds**, certified by weak duality and feasible by
    construction. Its per-step QP is then work unit 1's exact box-QP, so
    nothing about this contender is expensive any more.

    **The naming is the right way round here, and it was not before.** The
    bound is a *lower* bound on the constrained optimum; this policy's attained
    cost is an *upper* one, because the policy is real, feasible and
    box-respecting. `provenance` reports the bound; the evaluation reports the
    attainment; they sit either side of the truth and a caption must not swap
    them.

    Attributes:
        process_noise_std: The evaluation noise level; the bound is computed for
            ``W = process_noise_std**2 * I``.
        use_linear_term: Whether the frozen policy carries a linear cost-to-go
            term. The dual bound supplies no such term -- it produces ``P``
            alone -- so declaring this leaves ``q`` frozen at zero, and it
            exists only so the field means the same thing in both families.
        batch_size: Optional evaluation batch override.
        label: Instance name; defaults to ``"cocp_exact_lower_bound"``.
    """

    family: ClassVar[str] = "cocp_exact_lower_bound"

    process_noise_std: float
    use_linear_term: bool = False
    batch_size: int | None = None
    label: str = "cocp_exact_lower_bound"

    def batch_overrides(self) -> tuple[int | None, int | None]:
        """Batch-size only: this family is scored on the shared problem cost,
        whose ``Q``/``R`` are stacked to the main horizon.

        Returns:
            ``(batch_size, None)``.
        """
        return self.batch_size, None

    def _bound(self, problem: OptimalControlProblem) -> DualBound:
        """The box-aware dual bound for this problem at the declared noise.

        Args:
            problem: The problem to bound.

        Returns:
            The `DualBound`; its `value` is the theoretical floor and its
            `cost_to_go` the frozen seed.
        """
        system, cost = require_linear_quadratic(problem)
        A = system.A_t.array
        return infinite_horizon_box_bound(
            BoxLQRData(
                A=A,
                B=system.B_t.array,
                Q=np.asarray(time_invariant_slice(cost.Q)),
                R=np.asarray(time_invariant_slice(cost.R)),
                W=self.process_noise_std**2 * np.eye(A.shape[0]),
                u_max=float(np.asarray(_box_constraint(problem).u_max)),
            )
        )

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> ExactCOCPController:
        """See `ModelRecipe.build_controller`; the cost-to-go is the bound's and
        both parameters are frozen.

        Args:
            problem: The problem to control.
            ctx: The run's compute context; supplies the working precision.

        Returns:
            The frozen `ExactCOCPController`.
        """
        bound = self._bound(problem)
        assert bound.cost_to_go is not None  # infinite-horizon bounds carry one
        controller = ExactCOCPController(
            problem,
            _box_constraint(problem),
            ExactCOCPConfig(
                P_sqrt_init=np.real(sqrtm(bound.cost_to_go)),
                use_linear_term=self.use_linear_term,
                dtype=ctx.torch_dtype,
            ),
        )
        controller.P_sqrt.requires_grad_(False)
        if controller.q is not None:
            controller.q.requires_grad_(False)
        return controller.to(device=ctx.torch_device)

    def extra_callbacks(
        self,
        controller: Controller,
        problem: OptimalControlProblem,
        harness: EngineHarness,
    ) -> list[Callback]:
        """Log the bound beside the attained cost, and snapshot for replay.

        Args:
            controller: The frozen controller.
            problem: The problem it controls.
            harness: The engine harness, whose tracker receives the bound.

        Returns:
            The family-owned callbacks.
        """
        assert isinstance(controller, ExactCOCPController)
        bound = self._bound(problem)
        harness.tracker.log_params(
            {
                "dual_lower_bound_value": bound.value,
                "dual_lower_bound_kkt_residual": bound.kkt_residual,
                "dual_lower_bound_lmi_slack": bound.lmi_slack,
            }
        )
        return [
            ParameterSnapshotCallback(
                exact_cocp_convergence_parameters(controller, problem)
            )
        ]

    def provenance(self, problem: OptimalControlProblem) -> Mapping[str, float]:
        """This family's declared synthesis provenance.

        Args:
            problem: The problem whose bound to report.

        Returns:
            The bound and the two residuals that certify it -- a value without
            them is a number, not a bound.
        """
        bound = self._bound(problem)
        return {
            "dual_lower_bound_value": bound.value,
            "dual_lower_bound_kkt_residual": bound.kkt_residual,
            "dual_lower_bound_lmi_slack": bound.lmi_slack,
        }
