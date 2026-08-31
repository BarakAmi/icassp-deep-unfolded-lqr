"""Recipes for the closed-form Riccati families: the exact finite-horizon
optimum (`RiccatiRecipe`) and its clipped-onto-the-box variant
(`TruncatedRiccatiRecipe`). Both are `AnalyticRecipe`s (one evaluation
epoch, no optimizer) whose `Synthesizer`s are the direct solver-backed
two-phase implementations from Stage S2."""

from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np

from .base import AnalyticRecipe, EngineHarness, register_recipe
from ...core.kernels.riccati import gains_for_cost_to_go
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import Backend, ComputeContext
from ...core.kernels.riccati import riccati_recursion
from ...models.analytic.riccati import (
    RiccatiController,
    RiccatiSynthesizer,
    SynthesizedRiccatiController,
)
from ...models.analytic.truncated_riccati import (
    SynthesizedTruncatedRiccatiController,
    TruncatedRiccatiController,
    TruncatedRiccatiSynthesizer,
)
from ...models.guards import require_linear_quadratic


def _frozen_stack_and_informed_gains(
    offline_problem: OptimalControlProblem,
    online_problem: OptimalControlProblem,
    horizon: int,
    ctx: ComputeContext,
) -> tuple[np.ndarray, np.ndarray]:
    """The two halves of an informed Riccati controller (Annex 01 §2.4.1).

    The cost-to-go stack is the OFFLINE artifact — solved from the plant the
    controller was told, never re-solved online — and the gains are the
    ONLINE expressions, re-formed per step from that frozen stack and the
    matrices the controller is handed:
    ``K'_k = (R_k + B_k^T P_{k+1} B_k)^{-1} B_k^T P_{k+1} A_k``.

    Every operand is authored onto `ctx`'s substrate before any solve — the
    same single ingress conversion `RiccatiSynthesizer.synthesize` performs —
    because the blind reuse path re-derives through that synthesizer, and the
    NumPy and ctx routes differ by one ULP (the documented reuse hazard). On
    a matched pair this then reproduces the recursion's own gains bit for
    bit, which the no-op anchor asserts.
    """
    offline_system, offline_cost = require_linear_quadratic(offline_problem)
    online_system, online_cost = require_linear_quadratic(online_problem)
    A_off = ctx.asarray(np.stack([offline_system.A_t[k] for k in range(horizon)]))
    B_off = ctx.asarray(np.stack([offline_system.B_t[k] for k in range(horizon)]))
    P_arr, _ = riccati_recursion(
        A_off, B_off, ctx.asarray(offline_cost.Q), ctx.asarray(offline_cost.R), horizon
    )
    A_on = ctx.asarray(np.stack([online_system.A_t[k] for k in range(horizon)]))
    B_on = ctx.asarray(np.stack([online_system.B_t[k] for k in range(horizon)]))
    K_arr = gains_for_cost_to_go(P_arr, A_on, B_on, ctx.asarray(online_cost.R), horizon)
    return cast(np.ndarray, P_arr), cast(np.ndarray, K_arr)


@register_recipe("riccati")
@dataclass(frozen=True)
class RiccatiRecipe(AnalyticRecipe):
    """The unconstrained finite-horizon LQR optimum via backward Riccati
    recursion.

    Attributes:
        horizon: The finite horizon length ``N``.
        label: Instance name; defaults to the historical ``"analytic"``.
    """

    family: ClassVar[str] = "riccati"
    SAMPLER_BACKEND: ClassVar[Backend] = Backend.NUMPY

    horizon: int
    label: str = "analytic"

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> RiccatiController:
        """See `ModelRecipe.build_controller`; the solve happens here (the
        legacy single-phase construction the engine's `AnalyticalStrategy`
        consumes)."""
        return RiccatiController(problem, self.horizon)

    def build_synthesizer(
        self, problem: OptimalControlProblem, harness: EngineHarness
    ) -> RiccatiSynthesizer:
        """The direct solver-backed two-phase synthesizer (T2.a, Stage S2)."""
        return RiccatiSynthesizer(self.horizon)

    def build_rehosted_controller(
        self,
        offline_problem: OptimalControlProblem,
        online_problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> SynthesizedRiccatiController:
        """Annex 01 §2.4.1: P frozen from the told plant, gains re-formed
        per step from the given matrices — never a second recursion."""
        P_arr, K_arr = _frozen_stack_and_informed_gains(
            offline_problem, online_problem, self.horizon, ctx
        )
        return SynthesizedRiccatiController(
            P_arr=P_arr,
            K_arr=K_arr,
            context=ctx,
            synthesizer_signature={
                "type": "RehostedRebuild",
                "recipe": self.get_signature(),
            },
        )


@register_recipe("truncated_riccati")
@dataclass(frozen=True)
class TruncatedRiccatiRecipe(AnalyticRecipe):
    """The unconstrained Riccati optimum, saturated onto the problem's box
    constraint at every step -- the cheap non-learnable baseline for
    box-constrained problems.

    Attributes:
        horizon: The finite horizon length ``N``.
        label: Instance name; defaults to ``"truncated_riccati"``.
    """

    family: ClassVar[str] = "truncated_riccati"
    SAMPLER_BACKEND: ClassVar[Backend] = Backend.NUMPY

    horizon: int
    label: str = "truncated_riccati"

    def build_controller(
        self, problem: OptimalControlProblem, ctx: ComputeContext
    ) -> TruncatedRiccatiController:
        """See `ModelRecipe.build_controller`; the constraint is read off the
        problem (the recipe holds no copy of it)."""
        constraints = problem.constraints or []
        if not constraints:
            raise TypeError(
                "TruncatedRiccatiRecipe requires the problem to declare its "
                "box constraint as problem.constraints[0]."
            )
        return TruncatedRiccatiController(problem, self.horizon, constraints[0])

    def build_synthesizer(
        self, problem: OptimalControlProblem, harness: EngineHarness
    ) -> TruncatedRiccatiSynthesizer:
        """The direct solver-backed two-phase synthesizer (T2.a, Stage S2)."""
        constraints = problem.constraints or []
        if not constraints:
            raise TypeError(
                "TruncatedRiccatiRecipe requires the problem to declare its "
                "box constraint as problem.constraints[0]."
            )
        return TruncatedRiccatiSynthesizer(self.horizon, constraints[0])

    def build_rehosted_controller(
        self,
        offline_problem: OptimalControlProblem,
        online_problem: OptimalControlProblem,
        ctx: ComputeContext,
    ) -> SynthesizedTruncatedRiccatiController:
        """Annex 01 §2.4.1: P frozen from the told plant, gains re-formed
        per step from the given matrices, the projection read off the plant
        actually deployed against — never a second recursion."""
        constraints = online_problem.constraints or []
        if not constraints:
            raise TypeError(
                "TruncatedRiccatiRecipe requires the problem to declare its "
                "box constraint as problem.constraints[0]."
            )
        P_arr, K_arr = _frozen_stack_and_informed_gains(
            offline_problem, online_problem, self.horizon, ctx
        )
        return SynthesizedTruncatedRiccatiController(
            P_arr=P_arr,
            K_arr=K_arr,
            constraint=constraints[0],
            context=ctx,
            synthesizer_signature={
                "type": "RehostedRebuild",
                "recipe": self.get_signature(),
            },
        )
