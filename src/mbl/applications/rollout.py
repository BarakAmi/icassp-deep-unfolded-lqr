"""Adapts any torch-based Controller into the (X, Y, U, cost) forward
contract GradientDescentStrategy expects, with `cost` staying part of the
live autograd graph.

UnfoldedController already implements this (X, Y, U, cost) shape itself, but
its `cost` is computed via the numpy-only QuadraticCost (converting X/U with
to_numpy first) -- correct for reporting/evaluation, but not
backpropagatable, so it cannot drive real training through
GradientDescentStrategy (whose .step() calls `loss.backward()`). NeuralPolicy
has the opposite gap: it maps a state sequence to controls directly and never
simulates the system at all. RolloutModel covers both: it always runs the
rollout itself via `controller.get_control_policy()` and scores it through
the torch path of the dual-backend quadratic kernel, so the returned `cost`
remains differentiable regardless of which Controller it wraps.
"""

from typing import cast

import torch
from torch import nn

from ..core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from ..core.profiling import profiled
from ..core.optimal_control_problem import OptimalControlProblem
from ..models.base import Controller
from ..models.guards import require_linear_quadratic


class RolloutModel(nn.Module):
    """Wraps `controller` as a full-horizon, differentiable rollout callable.

    `controller` is assigned as a submodule when it is itself an nn.Module
    (e.g. NeuralPolicy), so `RolloutModel(controller).parameters()` correctly
    exposes the controller's own trainable weights to an optimizer. Controllers
    whose learnable parameters live elsewhere (e.g. UnfoldedController's, held
    in its UnfoldingConfig rather than registered as submodule attributes)
    still need their optimizer built from those parameters directly.

    `problem` is the WORLD this rollout runs in — dynamics and loss — and it
    defaults to the controller's own, which every caller before Annex 01
    §2.3.5 (D25) implicitly assumed: the model a contender holds and the
    world it trains in were one object by construction, welded exactly here.
    A declared training world hands a different plant, and the controller's
    internal model deliberately does not follow it — the loss is computed on
    what the data does, not on what the contender believes.
    """

    def __init__(
        self, controller: Controller, problem: OptimalControlProblem | None = None
    ) -> None:
        super().__init__()
        self.controller = controller
        self.problem = problem if problem is not None else controller.problem
        _, cost = require_linear_quadratic(self.problem)
        # The declared cost conventions this rollout trains against (C2 fix:
        # captured from the cost itself, honored unconditionally in
        # _differentiable_cost -- flipping either flag now changes the
        # training loss exactly as it changes the evaluated objective).
        self._conventions = cost.conventions
        # This problem's cost is time-invariant even when Q/R are stored
        # time-stacked (RiccatiController requires 3D Q/R) -- any slice is
        # the same matrix. Registered as (non-trainable) buffers so the static
        # cost matrices are materialized as tensors exactly once (not rebuilt
        # from numpy on every forward) and automatically follow .to(device/dtype).
        Q_2d = time_invariant_slice(cost.Q)
        R_2d = time_invariant_slice(cost.R)
        # Keep the cost matrices' native (numpy) dtype here; _differentiable_cost
        # casts to the incoming trajectory's dtype/device per call (a no-op when
        # they already match, which is the common case).
        self.register_buffer("_Q", torch.as_tensor(Q_2d), persistent=False)
        self.register_buffer("_R", torch.as_tensor(R_2d), persistent=False)

    @profiled("rollout.forward")
    def forward(
        self,
        initial_state: torch.Tensor,
        process_noise: torch.Tensor | None = None,
        measurement_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy = self.controller.get_control_policy()
        assert process_noise is not None and measurement_noise is not None
        # A RolloutModel rollout is torch-native end to end (autograd law).
        X, Y, U = cast(
            "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
            self.problem.system.run(
                policy, initial_state, process_noise, measurement_noise
            ),
        )
        cost = self._differentiable_cost(X, U)
        return X, Y, U, cost

    def _differentiable_cost(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """**Per-trajectory** quadratic trajectory cost under the problem's
        DECLARED conventions, differentiable end to end: the torch path of the
        dual-backend kernel's `PER_SAMPLE` reduction, honoring
        `include_terminal_cost`/`is_time_averaged` (C2 fix -- the pre-S2
        body hardcoded mean stage costs with no terminal term, silently
        training against a different objective whenever either flag was
        flipped). Q/R are cached buffers; cast per call only if the incoming
        trajectory dtype differs (e.g. float32 NeuralPolicy vs float64
        buffer).

        **The batch axis is preserved, and that is the point.** This returned
        `BATCH_MEAN` until 2026-08-04, i.e. a 0-dimensional tensor -- and
        `GradientDescentStrategy` then applied the study's declared
        `TrainingPlan.loss_reduction` to it. `torch.mean` and `torch.sum` are
        both the identity on a scalar, so `sum` and `mean` trained
        *identically* while `loss_reduction` was signed into `ModelID`: two
        studies differing only in that word derived two identifiers for one
        model. Reducing here is a decision that belongs to the caller, and
        leaving it here left the seam inert (Annex 06 §4.3's correction).

        A caller that wants the old scalar takes `.mean()`, which agrees with
        `BATCH_MEAN` to 1.34 ULP -- measured across batch sizes 8..8192 in
        both float32 and float64, the two being documented as distinct float
        association orders rather than different quantities.
        """
        Q = self._Q.to(dtype=X.dtype, device=X.device)
        R = self._R.to(dtype=U.dtype, device=U.device)
        return cast(
            torch.Tensor,
            total_quadratic_cost(
                Q,
                R,
                X,
                U,
                conventions=self._conventions,
                reduction=CostReduction.PER_SAMPLE,
            ),
        )
