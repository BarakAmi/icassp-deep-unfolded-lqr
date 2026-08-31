"""Engine-free, whole-horizon analytical gradient-descent LQR solver (Phase
2.5): the interactive research lab's primary instrument. Refines the entire
open-loop control sequence ``U`` over macro-iterations, driven by an
injectable `SweepStrategy` (Jacobi/Gauss-Seidel -- `models.iterative.sweeps`),
using the closed-form LOCAL cost-to-go gradient ``grad_u = 2(R + BᵀP_{k+1}B)u
+ 2(BᵀP_{k+1}A)x`` -- already implemented in `models.unfolded
.iterative_refinement.StepSizeRefinement` and reused here verbatim (never
reimplemented) via `build_riccati_gd_refinement`.

No `Runner`, `TrainingConfig`, or `TrainingStrategy` is involved anywhere in
this module: `.solve()` is a plain, native method that natively captures
`U_history`/`J_history` into a plain `OptimizationResult`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from .riccati import finite_horizon_riccati, get_lqr_gradient_matrices
from ...core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from ..base import Config
from ..guards import require_linear_quadratic
from ..iterative.initializers import ControlInitializer
from ..iterative.refinement import GradientDescentRefinement
from ..iterative.result import OptimizationResult
from ..iterative.step_size import StepSizeProvider
from ..iterative.sweeps import JacobiSweep, SweepContext, SweepStrategy
from ..samplers import Distribution, ZeroDistribution
from ..unfolded.iterative_refinement import StepSizeRefinement
from ...core.constraint.constraint import Constraint
from ...core.cost.quadratic_cost import QuadraticCost
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy
from ...core.utils import ensure_positive_integer


@dataclass(frozen=True)
class SolveSpec:
    """The complete specification of one `AnalyticalIterativeGDController
    .solve` run (T3.f): horizon, disturbance distributions, starting rule,
    iteration/tolerance budget, batch size, and the optional sweep-topology/
    refinement/constraint overrides — the sixteen-argument signature of the
    pre-S4 solver dissolved into one frozen value object.

    Attributes:
        horizon: The control horizon ``T``.
        x0_sampler: Draws the initial state, called as
            ``x0_sampler(batch_size, state_dim)``.
        noise_sampler: Draws the process noise, called as
            ``noise_sampler(batch_size, T, state_dim)``.
        control_initializer: Proposes ``U^(0)``.
        max_iters: Hard cap on the number of macro-iterations (sweeps).
        tolerance: Optional ``epsilon`` on ``|J^(i) - J^(i-1)|``; ``None``
            runs exactly `max_iters` sweeps.
        batch_size: Number of parallel trajectories.
        measurement_noise_sampler: Draws the measurement noise; defaults to
            `models.samplers.ZeroDistribution`.
        refinement: Overrides the default `build_riccati_gd_refinement`-built
            refinement.
        sweep_strategy: The update topology; defaults to `JacobiSweep`.
        constraints: Optional feasible-set projections forwarded to the
            default refinement (ignored if `refinement` is given explicitly).
    """

    horizon: int
    x0_sampler: Distribution
    noise_sampler: Distribution
    control_initializer: ControlInitializer
    max_iters: int
    tolerance: float | None = None
    batch_size: int = 1
    measurement_noise_sampler: Distribution | None = None
    refinement: GradientDescentRefinement | None = None
    sweep_strategy: SweepStrategy | None = None
    constraints: Sequence[Constraint] | None = None


def build_riccati_gd_refinement(
    problem: OptimalControlProblem,
    step_size: StepSizeProvider,
    spec: SolveSpec,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> StepSizeRefinement:
    """Build the DEFAULT local-gradient refinement for
    `AnalyticalIterativeGDController`: the closed-form local cost-to-go
    gradient, computed via the exact same `finite_horizon_riccati`/
    `get_lqr_gradient_matrices` machinery the Riccati controller itself uses
    (reused verbatim, never reimplemented), fed into the already-tested
    `StepSizeRefinement` -- so the analytical solver and the unfolded
    ``unfolded_fixed`` model provably share one gradient implementation.

    Args:
        problem: The problem being solved; its cost must have genuinely
            time-stacked ``Q``/``R`` (`finite_horizon_riccati`'s own
            precondition).
        step_size: The (fixed or learnable) per-macro-iteration step size.
        spec: The `SolveSpec` this refinement will be driven under -- supplies
            the horizon, the macro-iteration count (``max_iters``), and any
            feasible-set `constraints`.
        dtype: dtype the gradient-coefficient matrices are cast to.
        device: device the gradient-coefficient matrices are cast to.

    Returns:
        A `StepSizeRefinement` with static ``M_stack``/``C_stack`` derived
        from the exact Riccati cost-to-go (``M_stack = 2(R + BᵀPB)``,
        ``C_stack = 2(BᵀPA)``).
    """
    system, cost = require_linear_quadratic(problem)
    P_arr, _ = finite_horizon_riccati(system, cost, spec.horizon)
    A, B, R = system.A_t.array, system.B_t.array, cost.R
    M_stack, C_stack = get_lqr_gradient_matrices(P_arr, A, B, R)
    return StepSizeRefinement(
        step_size=step_size,
        num_iterations=spec.max_iters,
        static_parameters={
            "M_stack": torch.as_tensor(2 * M_stack, dtype=dtype, device=device),
            "C_stack": torch.as_tensor(2 * C_stack, dtype=dtype, device=device),
        },
        constraints=spec.constraints,
    )


def _build_cost_evaluator(
    cost: QuadraticCost, dtype: torch.dtype, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """Thin adapter over the dual-backend quadratic kernel (T2.d): the
    batch-mean scalar cost of a realized trajectory under `QuadraticCost`'s
    own ``include_terminal_cost``/``is_time_averaged`` conventions -- read
    off `cost` itself, never passed separately, so this can never silently
    disagree with the problem's declared cost. The per-sample kernel
    reduction reproduces this evaluator's pre-S2 operation sequence exactly
    (the frozen ``J_history`` goldens ride it).

    Args:
        cost: The problem's `QuadraticCost`.
        dtype: dtype the cost matrices are cast to.
        device: device the cost matrices are cast to.

    Returns:
        ``(X, U) -> float``: the batch-mean scalar cost of a realized
        trajectory, ``X`` shape ``(batch, T+1, n)``, ``U`` shape
        ``(batch, T, m)``.
    """
    Q = torch.as_tensor(time_invariant_slice(cost.Q), dtype=dtype, device=device)
    R = torch.as_tensor(time_invariant_slice(cost.R), dtype=dtype, device=device)

    def evaluate_cost(X: torch.Tensor, U: torch.Tensor) -> float:
        per_sample = total_quadratic_cost(
            Q,
            R,
            X,
            U,
            conventions=cost.conventions,
            reduction=CostReduction.PER_SAMPLE,
        )
        return float(per_sample.mean())

    return evaluate_cost


class AnalyticalIterativeGDController:
    """Engine-free, whole-horizon analytical gradient-descent LQR solver --
    the interactive notebook lab's primary instrument. Directly optimizes the
    open-loop control sequence ``U`` via macro-iterations of a pluggable
    `SweepStrategy` (Jacobi/Gauss-Seidel) over the shared, closed-form local
    Bellman gradient. No `Runner`/`TrainingConfig`/`TrainingStrategy`
    anywhere in this class.

    Satisfies `models.base.Controller` only AFTER `solve()` has been called
    (mirroring the fact that `problem` itself is a `solve()` argument, not
    known at construction time).

    Attributes:
        step_size: The default per-macro-iteration step size.
        problem: ``None`` until `solve()` is called; then the solved
            `OptimalControlProblem`.
        config: Always ``None`` -- this controller has no learnable
            hyperparameters to log (mirroring `RiccatiController`).
    """

    def __init__(
        self,
        step_size: StepSizeProvider,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
            step_size: The default per-macro-iteration step size, used to
                build the default refinement (`build_riccati_gd_refinement`)
                when `solve()` is not given an explicit `refinement`.
            dtype: dtype every tensor this solver builds is cast to.
            device: device every tensor this solver builds is cast to.
        """
        self.step_size = step_size
        self.dtype = dtype
        self.device = device
        self.problem: OptimalControlProblem | None = None
        self.config: Config | None = None
        self._U_final: torch.Tensor | None = None

    def solve(
        self,
        problem: OptimalControlProblem,
        spec: SolveSpec,
    ) -> OptimizationResult:
        """Solve the LQR problem by direct signal-space gradient descent.

        Every experimental knob rides the frozen `spec` (T3.f): its samplers
        are called once (a fixed realization, not resampled per iteration) to
        draw the deterministic evaluation batch; its `control_initializer`
        proposes ``U^(0)``; its `sweep_strategy` selects the Jacobi/
        Gauss-Seidel update topology; its `refinement` overrides the default
        Riccati-derived local gradient (e.g. to inject constraints via
        `build_riccati_gd_refinement` yourself, or a wholly different
        gradient source).

        Args:
            problem: The `OptimalControlProblem` to solve; its cost must have
                genuinely time-stacked ``Q``/``R`` (`finite_horizon_riccati`'s
                precondition).
            spec: The frozen `SolveSpec` bundling horizon, distributions,
                starting rule, iteration/tolerance budget, batch size, and
                the optional topology/refinement/constraint overrides.

        Returns:
            The `OptimizationResult`: index-aligned ``(U_history, J_history)``
            for ``i = 0 .. iterations_run``, plus the final trajectory and a
            reproducibility signature.

        Raises:
            ValueError: If ``spec.max_iters`` is not a positive ``int``, or
                ``spec.tolerance`` is given and not positive.
        """
        horizon, max_iters, tolerance = spec.horizon, spec.max_iters, spec.tolerance
        batch_size, control_initializer = spec.batch_size, spec.control_initializer
        x0_sampler, noise_sampler = spec.x0_sampler, spec.noise_sampler

        ensure_positive_integer(max_iters, "max_iters")
        if tolerance is not None and tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}.")

        system, cost = require_linear_quadratic(problem)
        state_dim = system.dimensions.state_dim
        measurement_noise_sampler = spec.measurement_noise_sampler or ZeroDistribution()
        sweep_strategy = spec.sweep_strategy or JacobiSweep()
        refinement = spec.refinement or build_riccati_gd_refinement(
            problem,
            self.step_size,
            spec,
            dtype=self.dtype,
            device=self.device,
        )

        x0 = torch.as_tensor(
            x0_sampler(batch_size, state_dim), dtype=self.dtype, device=self.device
        )
        w = torch.as_tensor(
            noise_sampler(batch_size, horizon, state_dim),
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.as_tensor(
            measurement_noise_sampler(batch_size, horizon, state_dim),
            dtype=self.dtype,
            device=self.device,
        )

        def rollout(
            policy: ControlPolicy,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # This solver authors x0/w/v as tensors, so the rollout is
            # torch-native end to end.
            return cast(
                "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                problem.system.run(policy, x0, w, v),
            )

        evaluate_cost = _build_cost_evaluator(cost, self.dtype, self.device)

        with torch.no_grad():
            prev_u: torch.Tensor | None = None

            def proposal_policy(t: int, y: torch.Tensor) -> torch.Tensor:
                nonlocal prev_u
                u = control_initializer(t, y, prev_u)
                prev_u = u.detach()
                return u

            X, _, U = rollout(cast("ControlPolicy", proposal_policy))
            J = evaluate_cost(X, U)

            U_snapshots = [U.detach().cpu().numpy().copy()]
            J_values = [J]
            converged = False

            ctx = SweepContext(
                refinement=refinement,
                rollout=rollout,
                evaluate_cost=evaluate_cost,
                horizon=horizon,
            )
            for i in range(1, max_iters + 1):
                result = sweep_strategy.sweep(i - 1, U, X, ctx)
                J_prev = J
                U, X, J = result.U_next, result.X_next, result.J_next
                U_snapshots.append(U.detach().cpu().numpy().copy())
                J_values.append(J)
                if tolerance is not None and abs(J - J_prev) < tolerance:
                    converged = True
                    break

        self.problem = problem
        self.config = None
        self._U_final = U

        signature = {
            "problem": problem.get_signature(),
            "control_initializer": control_initializer.get_signature(),
            "x0_sampler": x0_sampler.get_signature(),
            "noise_sampler": noise_sampler.get_signature(),
            "measurement_noise_sampler": measurement_noise_sampler.get_signature(),
            "refinement": refinement.get_signature(),
            "sweep_strategy": sweep_strategy.get_signature(),
        }

        return OptimizationResult(
            U_history=np.stack(U_snapshots),
            J_history=np.asarray(J_values, dtype=float),
            U_final=U.detach().cpu().numpy(),
            J_final=float(J),
            X_final=X.detach().cpu().numpy(),
            iterations_run=len(J_values) - 1,
            converged=converged,
            signature=signature,
        )

    def get_control_policy(self) -> ControlPolicy:
        """Return the open-loop policy replaying the converged `U_final`.

        Returns:
            A `ControlPolicy`: ``lambda t, y: self._U_final[:, t]``.

        Raises:
            RuntimeError: If `solve()` has not been called yet.
        """
        if self._U_final is None:
            raise RuntimeError("solve() must be called before get_control_policy().")
        U_final = self._U_final
        return lambda t, y: U_final[:, t]

    def get_signature(self) -> dict:
        """`Signable` member: this controller's own type.

        The full solve provenance (problem/initializer/samplers/refinement/
        sweep-strategy) lives on `OptimizationResult.signature` instead --
        this method exists only for `Controller` protocol conformance.

        Returns:
            ``{"type": "AnalyticalIterativeGDController"}``.
        """
        return {"type": type(self).__name__}
