"""Two `CostOracle` adapters, scoring a landscape grid by fundamentally
different means:

- `make_rollout_cost_oracle`: rolls a *batch* of full-horizon, open-loop
  control sequences through `problem.system` from a fixed reference
  initial-state/noise point, and scores each sample with the same ``(Q, R,
  include_terminal_cost, is_time_averaged)`` cost definition `problem.cost`
  declares -- but preserving the batch axis, which `applications.rollout
  .RolloutModel.forward`'s own cost does not (see `types.CostOracle`'s
  docstring: reducing to one scalar would collapse an entire grid evaluation
  to one number). Its downstream (post-``timestep``) continuation is held
  open-loop at a stale baseline rather than re-optimized against whatever
  perturbed state a grid point induces -- an approximation of the true local
  cost-to-go, not a reproduction of it.

- `make_local_cost_to_go_oracle`: the EXACT one-step Bellman cost-to-go at a
  frozen state (`models.analytic.riccati.evaluate_local_cost_to_go`), never
  propagating past that one step -- so a landscape built from it has no
  downstream continuation to get wrong, and its numerical minimum coincides
  exactly with the analytical Riccati optimum (the "displaced star" fix).

Kept torch-native throughout (the torch path of the dual-backend quadratic
kernel, evaluated under `torch.no_grad()`) so the oracle never forces a
host<->device round-trip per grid chunk -- load-bearing once a
`RolloutModel` (e.g. wrapping an Unfolded network) runs on GPU.
"""

import numpy as np
from typing import cast

import torch

from .types import CostOracle
from ...core.cost.quadratic_cost import QuadraticCost
from ...core.system.state_space_system import ControlPolicy
from ...core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.utils import to_numpy
from ...models.analytic.riccati import LocalCostToGoModel, evaluate_local_cost_to_go

type Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""``(initial_state, process_noise, measurement_noise)``, each batch-1 --
the fixed, deterministic reference point (Phase 2A Decision 13.1: batch 1,
zero noise) every grid sample is broadcast against."""


def make_rollout_cost_oracle(
    problem: OptimalControlProblem,
    reference_batch: Batch,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> CostOracle:
    """Build a per-sample `CostOracle` scoring open-loop control sequences
      against `problem`'s cost, broadcasting the fixed `reference_batch`
      across the grid batch.

      Mirrors `problem.cost`'s own ``Q``/``R`` (reduced to their time-invariant
      2D slice exactly as `applications.rollout.RolloutModel.__init__` does --
      a genuinely time-varying cost is out of scope here, same limitation
      `RolloutModel` already has) and its `include_terminal_cost`/
      `is_time_averaged` flags -- read off `problem.cost` itself, never as
      separate arguments, the same discipline
      `models.analytic.riccati.compute_theoretical_expected_cost` follows, so
      the landscape always matches whichever cost variant a given run used.

      When `problem.cost.include_terminal_cost` is ``False`` and
      `problem.cost.is_time_averaged` is ``True`` (`core.cost.quadratic_cost
      .QuadraticCost`'s own defaults), this oracle's per-sample output
      averages to `RolloutModel.forward`'s own per-trajectory cost (batch-
    reduced by the caller since Annex 06 §4.3's correction) for an
      identical batch of reference controls -- the acceptance test this
      adapter is built against (`tests/applications/visualization
      /test_oracles.py`).

      Args:
          reference_batch: ``(x0, w, v)``, each batch-1 -- the fixed,
              deterministic rollout point every grid sample is broadcast
              against.
          dtype: dtype grid batches are cast to before rollout.
          device: device grid batches are cast to before rollout.

      Returns:
          A `CostOracle`: ``(batch, T, m) -> (batch,)``.
    """
    x0_ref, w_ref, v_ref = reference_batch
    cost = problem.cost
    if not isinstance(cost, QuadraticCost):
        raise TypeError(
            f"make_rollout_cost_oracle requires a QuadraticCost, got "
            f"{type(cost).__name__}."
        )
    Q_t = torch.as_tensor(time_invariant_slice(cost.Q), dtype=dtype, device=device)
    R_t = torch.as_tensor(time_invariant_slice(cost.R), dtype=dtype, device=device)

    x0_ref = x0_ref.to(dtype=dtype, device=device)
    w_ref = w_ref.to(dtype=dtype, device=device)
    v_ref = v_ref.to(dtype=dtype, device=device)

    @torch.no_grad()
    def cost_oracle(U: torch.Tensor) -> torch.Tensor:
        """Score a batch of full-horizon open-loop control sequences.

        Args:
            U: candidate control batch, shape ``(batch, T, m)``.

        Returns:
            Per-sample scalar cost, shape ``(batch,)``.
        """
        batch = U.shape[0]
        U = U.to(dtype=dtype, device=device)
        x0 = x0_ref.expand(batch, *x0_ref.shape[1:])
        w = w_ref.expand(batch, *w_ref.shape[1:])
        v = v_ref.expand(batch, *v_ref.shape[1:])

        def policy(t: int, y: torch.Tensor) -> torch.Tensor:
            return U[:, t]

        X, _, U_out = problem.system.run(cast("ControlPolicy", policy), x0, w, v)

        # The dual-backend kernel's per-sample reduction (T2.d) -- the
        # batch axis survives (see `types.CostOracle`) and the declared
        # terminal/averaging flags are honored by construction.
        return cast(
            torch.Tensor,
            total_quadratic_cost(
                Q_t,
                R_t,
                X,
                U_out,
                conventions=cost.conventions,
                reduction=CostReduction.PER_SAMPLE,
            ),
        )

    return cost_oracle


def make_local_cost_to_go_oracle(
    x_star: np.ndarray,
    *,
    model: LocalCostToGoModel,
    timestep: int,
) -> CostOracle:
    """Build a `CostOracle` plugging the EXACT closed-form one-step Bellman
    cost-to-go (`models.analytic.riccati.evaluate_local_cost_to_go`) into
    `grids.GridProjector`'s existing ``(P, T, m)`` batch contract, so every
    landscape renderer built for `make_rollout_cost_oracle` (Assets 3-5)
    works completely unchanged for this oracle too.

    Only the `timestep` slice of each candidate row is ever read -- the
    local Bellman cost-to-go is, by definition, a function of the frozen
    state `x_star` and that one instant's control alone, genuinely
    independent of every other ``(t, j)`` entry `grids.GridProjector.project`
    fills the row with. Whatever `reference_U` a caller passes to `project`
    therefore only needs to match the problem's ``(T, m)`` shape -- its
    values away from `timestep` are never read.

    Args:
        x_star: the frozen reference state ``x_{t*}``, shape ``(n,)``.
        model: the one-timestep model slice (`LocalCostToGoModel`) evaluated
            at `timestep` -- system/cost matrices, Riccati continuation
            ``P_{timestep+1}``, and noise covariance.
        timestep: which timestep slice of a candidate ``(T, m)`` row to
            read as ``u`` -- must match the `SliceSpec.timestep` the
            resulting oracle is projected over.

    Returns:
        A `CostOracle`: ``(batch, T, m) -> (batch,)``.
    """

    def cost_oracle(U: torch.Tensor) -> torch.Tensor:
        U_np = to_numpy(U)[:, timestep, :]
        cost = evaluate_local_cost_to_go(x_star, U_np, model)
        return torch.as_tensor(cost, dtype=U.dtype, device=U.device)

    return cost_oracle
