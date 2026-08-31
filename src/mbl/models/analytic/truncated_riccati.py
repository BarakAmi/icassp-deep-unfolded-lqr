"""Non-learnable Riccati-plus-saturation baseline for box-constrained LQR."""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import torch

from ...core.optimal_control_problem import Constraint, OptimalControlProblem
from ...core.runtime import Backend, ComputeContext
from ...core.system.state_space_system import ControlPolicy
from ..base import Config
from ..guards import check_controllability, require_linear_quadratic
from ..lifecycle import ALL_BACKENDS, SynthesizerBase
from ..registry import register_model
from .riccati import (
    RiccatiSynthesizer,
    finite_horizon_riccati,
    get_riccati_control_policy,
)


@register_model("truncated_riccati")
class TruncatedRiccatiController:
    """The optimal unconstrained Riccati feedback policy, clipped onto a
    feasible control set at every time step -- a cheap, non-learnable
    baseline for a box-constrained problem: ignores the constraint while
    solving for P/K, then saturates the resulting control input.

    LEGACY single-phase shape (see `RiccatiController`): retained this
    stage for the application layer; new orchestration targets
    `TruncatedRiccatiSynthesizer` (Stage S3 recipes).

    Attributes:
        problem: The `OptimalControlProblem` this controller solves.
        horizon: The finite horizon length ``N``.
        constraint: The feasible-set `Constraint` the policy is projected onto.
        config: Always ``None`` -- this controller has no learnable
            hyperparameters to log.
        P_arr: Cost-to-go matrices, shape ``(N+1, n, n)``, ignoring `constraint`.
        K_arr: Feedback gains, shape ``(N, m, n)``, ignoring `constraint`.
    """

    def __init__(
        self,
        problem: OptimalControlProblem,
        horizon: int,
        constraint: Constraint,
    ) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` to solve (constraint-unaware
                during the Riccati solve).
            horizon: The finite horizon length ``N``.
            constraint: The feasible-set projection applied at every step of
                the resulting policy.

        Raises:
            ValueError: (from `finite_horizon_riccati`) if ``(A, B, Q, R)`` are
                dimensionally incoherent, or `problem.cost`'s ``Q``/``R`` are
                not genuinely time-stacked with enough slices (Phase 1A
                fail-fast guards).

        Warns:
            UserWarning: If ``(A, B)`` is not controllable (diagnostic only;
                see `models.guards.check_controllability`).
        """
        self.problem = problem
        self.horizon = horizon
        self.constraint = constraint
        self.config: Config | None = None
        system, cost = require_linear_quadratic(problem)
        check_controllability(system.A_t[0], system.B_t[0])
        self.P_arr, self.K_arr = finite_horizon_riccati(system, cost, horizon)

    def get_control_policy(self) -> ControlPolicy:
        """Return the unconstrained Riccati feedback policy, projected onto
        `self.constraint` at every step.

        Returns:
            A `ControlPolicy`: ``(t, x_t) -> constraint(-K_t x_t)``.
        """
        unconstrained_policy = get_riccati_control_policy(self.K_arr)

        def policy(
            t: int, x: "np.ndarray | torch.Tensor"
        ) -> "np.ndarray | torch.Tensor":
            u = unconstrained_policy(t, x)
            return self.constraint(u)

        return policy

    def get_signature(self) -> dict:
        """`Signable` member: type + horizon + this controller's own
        constraint (distinct from any `problem.constraints`).

        The solved `problem`'s signature is intentionally omitted --
        `ProblemSignatureCallback` already logs it once at the root
        (``problem.*``); embedding it again here would just duplicate every
        key under ``controller.problem.*``.

        Returns:
            ``{"type": "TruncatedRiccatiController", "horizon":,
            "constraint": ...}``.
        """
        return {
            "type": type(self).__name__,
            "horizon": self.horizon,
            "constraint": self.constraint.get_signature(),
        }


@dataclass(frozen=True)
class SynthesizedTruncatedRiccatiController:
    """Frozen offline artifact of `TruncatedRiccatiSynthesizer`: the
    unconstrained Riccati solution plus the feasible-set projection the
    online policy applies per step.

    Attributes:
        P_arr: Cost-to-go matrices ``(N+1, n, n)``, constraint-unaware.
        K_arr: Feedback gains ``(N, m, n)``, constraint-unaware.
        constraint: The feasible-set `Constraint` projected at every step.
        context: The effective `ComputeContext` (residency law).
        synthesizer_signature: The producing synthesizer's signature.
    """

    P_arr: np.ndarray
    K_arr: np.ndarray
    constraint: Constraint
    context: ComputeContext
    synthesizer_signature: dict[str, Any]

    def make_policy(self) -> ControlPolicy:
        """A fresh saturated-feedback closure ``u_t = proj(-K_t x_t)`` per
        call (freshness law).

        Returns:
            A batched `ControlPolicy` projecting through `constraint`.
        """
        unconstrained_policy = get_riccati_control_policy(self.K_arr)
        constraint = self.constraint

        def policy(
            t: int, x: "np.ndarray | torch.Tensor"
        ) -> "np.ndarray | torch.Tensor":
            u = unconstrained_policy(t, x)
            return constraint(u)

        return policy

    def get_signature(self) -> dict[str, Any]:
        """Synthesizer signature + synthesis provenance (C1 law: no
        ``problem.*``, no live values).

        Returns:
            ``{"type":, "synthesizer":, "compute_context":}``.
        """
        return {
            "type": type(self).__name__,
            "synthesizer": dict(self.synthesizer_signature),
            "compute_context": self.context.get_signature(),
        }


@dataclass(frozen=True)
class TruncatedRiccatiSynthesizer(SynthesizerBase):
    """Two-phase (T2.a) synthesizer for the truncated-Riccati baseline:
    delegates the unconstrained solve to `RiccatiSynthesizer` (the
    recursion mathematics lives exactly once) and attaches the feasible-set
    projection to the frozen artifact.

    Declared envelope (T1.e): the full backend set. Originally NumPy-only
    -- deliberately conservative, since the online phase projects every
    control through an arbitrary injected `Constraint`, and that protocol
    (`core.constraint.constraint.Constraint`) guarantees nothing about torch
    support in general. Widened (NB04, box-constrained benchmarking) now
    that the condition the original docstring named has been met: the one
    concrete `Constraint` this tree ships, `BoxConstraint`, IS backend-aware
    (its `__call__` dispatches on `isinstance(u, torch.Tensor)` and clamps
    natively either way), and the delegated solve
    (`RiccatiSynthesizer`) already supports every backend -- so a torch
    `ComputeContext` produces a torch-native `P_arr`/`K_arr` and
    `get_riccati_control_policy` already handles the mixed torch-state/
    NumPy-gain case explicitly, meaning nothing here ever silently
    round-trips through NumPy per step (the exact T1.f failure the original
    restriction guarded against). A future `Constraint` implementation that
    is NOT backend-aware would need to re-narrow this ClassVar (or wrap
    itself defensively) -- this widening is verified for `BoxConstraint`
    specifically, not a blanket claim about the `Constraint` protocol.

    Attributes:
        horizon: The finite horizon length ``N``.
        constraint: The feasible-set projection applied by the policy.
    """

    SUPPORTED_BACKENDS: ClassVar[frozenset[Backend]] = ALL_BACKENDS

    horizon: int
    constraint: Constraint

    def synthesize(
        self, problem: OptimalControlProblem, ctx: ComputeContext | None = None
    ) -> SynthesizedTruncatedRiccatiController:
        """Solve the unconstrained problem, then freeze gains + projection.

        Args:
            problem: The `OptimalControlProblem` (constraint-unaware during
                the Riccati solve).
            ctx: The injected compute context; ``None`` inherits the module
                default. Must lie inside the NumPy-only envelope.

        Returns:
            The frozen `SynthesizedTruncatedRiccatiController`.

        Raises:
            UnsupportedBackendError: If `ctx` requests the torch backend
                (outside this family's declared envelope).
            ValueError: If the Riccati preconditions fail.

        Warns:
            UserWarning: If ``(A, B)`` is not controllable (diagnostic only).
        """
        ctx = self._resolve_context(ctx)
        unconstrained = RiccatiSynthesizer(self.horizon).synthesize(problem, ctx)
        return SynthesizedTruncatedRiccatiController(
            P_arr=cast(np.ndarray, unconstrained.P_arr),
            K_arr=cast(np.ndarray, unconstrained.K_arr),
            constraint=self.constraint,
            context=ctx,
            synthesizer_signature=self.get_signature(),
        )

    def get_signature(self) -> dict[str, Any]:
        """Specification-time configuration only (C1 law): type + horizon +
        this synthesizer's own constraint (distinct from any
        ``problem.constraints``).

        Returns:
            ``{"type":, "horizon":, "constraint":}``.
        """
        return super().get_signature() | {
            "horizon": self.horizon,
            "constraint": self.constraint.get_signature(),
        }
