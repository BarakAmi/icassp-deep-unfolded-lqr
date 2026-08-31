"""Update-sweep topologies (Phase 2.5 addendum): the Jacobi (simultaneous)
vs. Gauss-Seidel (sequential) dichotomy for one macro-iteration of a
whole-horizon analytical solve, exposed as an injectable notebook knob
(`models.analytic.iterative_gd.AnalyticalIterativeGDController.solve`'s
``sweep_strategy`` argument).

Both topologies drive the IDENTICAL single-timestep kernel --
`refinement.pre_iteration_hook` / `refinement.refine_step` (the closed-form
local Bellman gradient, `models.iterative.refinement.GradientDescentRefinement`)
-- and differ ONLY in where the per-timestep state comes from and when the
update takes effect. Deliberately named `SweepStrategy` (not `*TrainingStrategy`)
to avoid any confusion with `engine.strategy.TrainingStrategy`: this module has
no relationship to, and no dependency on, the ML engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

from .refinement import GradientDescentRefinement
from ...core.system.state_space_system import ControlPolicy


@dataclass(frozen=True)
class SweepContext:
    """Everything one macro-iteration needs, assembled once per `.solve()`
    call by the controller -- strategies stay free of problem/system
    knowledge and only sequence operations over these injected capabilities.

    Attributes:
        refinement: The shared local-gradient kernel (`pre_iteration_hook`/
            `refine_step`), reused identically by every topology.
        rollout: Closure over `problem.system.run` with the fixed sampled
            ``(x0, w, v)`` batch already bound -- ``policy -> (X, Y, U)``.
        evaluate_cost: Closure over the problem's cost convention --
            ``(X, U) -> float``, the batch-mean scalar cost of a realized
            trajectory.
        horizon: ``T``, the control horizon.
    """

    refinement: GradientDescentRefinement
    rollout: Callable[[ControlPolicy], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    evaluate_cost: Callable[[torch.Tensor, torch.Tensor], float]
    horizon: int


@dataclass(frozen=True)
class SweepResult:
    """One macro-iteration's output.

    Attributes:
        U_next: The updated control sequence ``U^(i)``, shape ``(batch, T, m)``.
        X_next: The state trajectory REALIZED BY `U_next` (the load-bearing
            postcondition every `SweepStrategy` must satisfy: `X_next`/
            `J_next` are never a stale or slice-recomputed value -- they are
            exactly what rolling `U_next` forward produces), shape
            ``(batch, T+1, n)``.
        J_next: ``== evaluate_cost(X_next, U_next)`` -- the exact cost of
            `U_next`, by construction.
    """

    U_next: torch.Tensor
    X_next: torch.Tensor
    J_next: float


class SweepStrategy(ABC):
    """One macro-iteration of an analytical whole-horizon solve:
    ``(U^(i-1), X^(i-1)) -> (U^(i), X^(i), J^(i))``. Stateless; every
    concrete strategy is constructed with no arguments -- the trivially
    swappable notebook knob."""

    @abstractmethod
    def sweep(
        self,
        iteration_index: int,
        U_prev: torch.Tensor,
        X_prev: torch.Tensor,
        ctx: SweepContext,
    ) -> SweepResult:
        """Advance one macro-iteration.

        Args:
            iteration_index: This macro-iteration's index (forwarded to
                `ctx.refinement.refine_step` to select the step size).
            U_prev: The previous iterate ``U^(i-1)``, shape ``(batch, T, m)``.
            X_prev: The state trajectory realized by `U_prev`, shape
                ``(batch, T+1, n)``.
            ctx: This solve's shared capabilities.

        Returns:
            The `SweepResult` for this macro-iteration.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        ...

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: this sweep strategy's own type -- lets a
        solve's provenance record (`models.iterative.OptimizationResult
        .signature`) capture which topology produced a given run.

        Returns:
            ``{"type": <class name>}``.
        """
        return {"type": type(self).__name__}


class JacobiSweep(SweepStrategy):
    """Simultaneous (classical gradient-descent) topology: every timestep's
    local gradient is computed against the *previous* iterate's realized
    states `X_prev`, and every ``u_k`` is updated at once before the update
    takes effect anywhere. The updated sequence is then replayed once
    (open-loop) to obtain the trajectory `X_next` it actually realizes --
    which is exactly the `X_prev` the *next* Jacobi sweep needs, so this
    topology costs exactly one `rollout` call per macro-iteration, the same
    as `GaussSeidelSweep`."""

    def sweep(
        self,
        iteration_index: int,
        U_prev: torch.Tensor,
        X_prev: torch.Tensor,
        ctx: SweepContext,
    ) -> SweepResult:
        updated: list[torch.Tensor] = []
        for k in range(ctx.horizon):
            x_k = X_prev[:, k]
            args = ctx.refinement.pre_iteration_hook(k, x_k)
            u_k, _ = ctx.refinement.refine_step(iteration_index, U_prev[:, k], *args)
            updated.append(u_k)
        U_next = torch.stack(updated, dim=1)  # (batch, T, m), simultaneous application

        def replay_policy(t: int, y: torch.Tensor) -> torch.Tensor:
            return U_next[:, t]

        X_next, _, U_realized = ctx.rollout(cast("ControlPolicy", replay_policy))
        J_next = ctx.evaluate_cost(X_next, U_realized)
        return SweepResult(U_next=U_realized, X_next=X_next, J_next=J_next)


class GaussSeidelSweep(SweepStrategy):
    """Sequential topology: each timestep's local gradient is evaluated
    against the *live*, already-updated state ``x_k`` inside one forward
    rollout (reflecting the just-updated ``u_0 .. u_{k-1}``), and the update
    takes effect immediately, shaping ``x_{k+1}``. Self-consistent by
    construction: the single rollout call both applies the update and
    realizes its trajectory, so `X_next`/`J_next` are exactly `U_next`'s own
    trajectory/cost with no second rollout."""

    def sweep(
        self,
        iteration_index: int,
        U_prev: torch.Tensor,
        X_prev: torch.Tensor,
        ctx: SweepContext,
    ) -> SweepResult:
        def policy(t: int, x_t: torch.Tensor) -> torch.Tensor:
            args = ctx.refinement.pre_iteration_hook(t, x_t)
            u_t, _ = ctx.refinement.refine_step(iteration_index, U_prev[:, t], *args)
            return u_t

        X_next, _, U_next = ctx.rollout(cast("ControlPolicy", policy))
        J_next = ctx.evaluate_cost(X_next, U_next)
        return SweepResult(U_next=U_next, X_next=X_next, J_next=J_next)
