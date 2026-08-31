"""The `ORACLE` model-access mode's per-family resynchronization
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.2/5.1): re-point a
controller's internal model at a freshly-perturbed plant, in place, between
epochs -- the counterpart to `NOMINAL` mode, which leaves the internal model
untouched while the simulated plant drifts underneath it (NB07 plan Sec 2.3's
taxonomy).

Dispatch is purely by controller/refinement TYPE, not by recipe or family
name (a deliberate simplification of the plan's ``resync_to_plant(artifact,
recipe, ...)`` sketch): the internal representation that needs rewriting is
uniquely determined by the controller's own class, so no recipe object is
needed to make the dispatch decision.
"""

from typing import Any

import numpy as np
import torch

from ...core.optimal_control_problem import OptimalControlProblem
from ...core.runtime import ComputeContext
from ...models.analytic.riccati import finite_horizon_riccati, get_lqr_gradient_matrices
from ...models.analytic.truncated_riccati import TruncatedRiccatiController
from ...models.constrained.cocp import COCPController
from ...models.guards import require_linear_quadratic
from ...models.neural.nerual import NeuralPolicy
from ...models.unfolded.base import UnfoldedController
from ...models.unfolded.iterative_refinement import (
    RiccatiRefinement,
    StepSizeRefinement,
)


class UnsupportedResyncError(TypeError):
    """Raised when `resync_to_plant` is asked to `ORACLE`-resynchronize a
    controller family that has no cheap internal-model resynchronizer (e.g.
    COCP -- NB07 plan Sec 4.2's declared `NOMINAL`-only families)."""


def supports_oracle_resync(controller: Any) -> bool:
    """Whether `resync_to_plant` can resynchronize `controller` without
    raising -- `False` only for `COCPController` (NB07 plan Sec 4.2's
    declared `NOMINAL`-only exception; both `cocp` and `cocp_lower_bound`
    are backed by this type). A sweep driving one axis under `ORACLE` reads
    this before attaching the resync callback to a given contender, so that
    one declared exception downgrades to `NOMINAL` for itself instead of
    raising and aborting every other contender's run.

    Args:
        controller: The live controller a `ModelAccess.ORACLE` sweep would
            otherwise resynchronize every epoch.

    Returns:
        `True` for every controller type `resync_to_plant` itself handles
        without raising (`UnfoldedController`, `TruncatedRiccatiController`,
        `NeuralPolicy`); `False` for `COCPController`.
    """
    return not isinstance(controller, COCPController)


def resync_to_plant(
    controller: Any, perturbed_problem: OptimalControlProblem, ctx: ComputeContext
) -> None:
    """Rewrite `controller`'s internal model in place so it matches
    `perturbed_problem` -- the `ORACLE` mode's whole mechanism (NB07 plan
    Sec 4.2/5.1). A no-op for controller families with no internal model
    (the GRU); raises for families with no cheap resynchronizer (COCP).

    Args:
        controller: The live controller to resynchronize (mutated in place).
        perturbed_problem: The plant the controller's internal model should
            now describe.
        ctx: The compute context (dtype/device authority for any rebuilt
            tensor).

    Raises:
        UnsupportedResyncError: If `controller` is a `COCPController` (no
            cheap resynchronizer exists -- COCP is `NOMINAL`-only, NB07 plan
            Sec 13 decision 2) or an otherwise unrecognized controller type.
    """
    if isinstance(controller, UnfoldedController):
        _resync_unfolded(controller, perturbed_problem, ctx)
        return
    if isinstance(controller, TruncatedRiccatiController):
        _resync_truncated_riccati(controller, perturbed_problem)
        return
    if isinstance(controller, NeuralPolicy):
        return  # No internal model to synchronize -- the structural asymmetry under study.
    if isinstance(controller, COCPController):
        raise UnsupportedResyncError(
            "COCPController has no cheap ORACLE resynchronizer (rebuilding "
            "its per-step QP is as expensive as a fresh synthesis); COCP is "
            "declared NOMINAL-only (NB07 plan Sec 4.2, Sec 13 decision 2)."
        )
    raise UnsupportedResyncError(
        f"No ORACLE resync is defined for controller type {type(controller).__name__!r}."
    )


def _resync_unfolded(
    controller: UnfoldedController,
    perturbed_problem: OptimalControlProblem,
    ctx: ComputeContext,
) -> None:
    """Dispatch on the controller's refinement strategy: `StepSizeRefinement`
    (no learned matrix -- re-solve Riccati and rewrite its gradient-
    coefficient stacks) or `RiccatiRefinement` (learned matrix -- rewrite
    only the static ``A``/``B`` the refinement recomputes its gradient
    coefficients from every rollout; the learned matrix itself is NEVER
    touched, since it is the parameter being trained)."""
    refinement = controller.config.iterative_refinement
    if isinstance(refinement, RiccatiRefinement):
        _resync_riccati_refinement(refinement, perturbed_problem, ctx)
    elif isinstance(refinement, StepSizeRefinement):
        _resync_step_size_refinement(
            refinement, controller.config.horizon, perturbed_problem, ctx
        )
    else:
        raise UnsupportedResyncError(
            f"No ORACLE resync is defined for refinement type "
            f"{type(refinement).__name__!r}."
        )


def _resync_riccati_refinement(
    refinement: RiccatiRefinement,
    perturbed_problem: OptimalControlProblem,
    ctx: ComputeContext,
) -> None:
    """Overwrite `refinement.static_parameters["A"]`/``["B"]`` with the
    perturbed plant's per-step stacks; ``"R"`` (the cost) is never perturbed
    (Sec 2.1's hard scope constraint) and is left untouched. The refinement's
    own gradient-coefficient cache invalidates automatically at the next
    rollout (`on_rollout_start`), rebuilding ``M``/``C`` from these fresh
    tensors and the CURRENT (possibly mid-training) learned matrix -- this is
    the "automatic" propagation NB07 plan Sec 2.3 describes."""
    system, _ = require_linear_quadratic(perturbed_problem)
    horizon = refinement.static_parameters["A"].shape[0]
    dtype, device = ctx.torch_dtype, ctx.torch_device
    A = np.stack([system.A_t[k] for k in range(horizon)])
    B = np.stack([system.B_t[k] for k in range(horizon)])
    # Whole-mapping reassignment, not indexed mutation: `static_parameters`
    # is read fresh via attribute access at every call site (never a
    # dict-identity captured at construction), so swapping in a new mapping
    # is equivalent to in-place mutation and keeps `Mapping`'s read-only
    # typing honest.
    refinement.static_parameters = {
        **refinement.static_parameters,
        "A": torch.tensor(A, dtype=dtype, device=device),
        "B": torch.tensor(B, dtype=dtype, device=device),
    }


def _resync_step_size_refinement(
    refinement: StepSizeRefinement,
    horizon: int,
    perturbed_problem: OptimalControlProblem,
    ctx: ComputeContext,
) -> None:
    """Re-solve the finite-horizon Riccati recursion for `perturbed_problem`
    and overwrite `refinement.static_parameters["M_stack"]`/``["C_stack"]``
    -- the learned step sizes must now serve gradient steps derived from the
    perturbed plant's own (re-solved) cost-to-go."""
    system, cost = require_linear_quadratic(perturbed_problem)
    dtype, device = ctx.torch_dtype, ctx.torch_device
    A = np.stack([system.A_t[k] for k in range(horizon)])
    B = np.stack([system.B_t[k] for k in range(horizon)])
    R = cost.R
    P_arr, _ = finite_horizon_riccati(system, cost, horizon)
    M_stack, C_stack = get_lqr_gradient_matrices(P_arr, A, B, R)
    refinement.static_parameters = {
        **refinement.static_parameters,
        "M_stack": torch.tensor(2 * M_stack, dtype=dtype, device=device),
        "C_stack": torch.tensor(2 * C_stack, dtype=dtype, device=device),
    }


def _resync_truncated_riccati(
    controller: TruncatedRiccatiController, perturbed_problem: OptimalControlProblem
) -> None:
    """Re-solve `P_arr`/`K_arr` for `perturbed_problem` -- the saturated-gain
    baseline's entire "internal model" is this pair of arrays."""
    system, cost = require_linear_quadratic(perturbed_problem)
    controller.problem = perturbed_problem
    controller.P_arr, controller.K_arr = finite_horizon_riccati(
        system, cost, controller.horizon
    )
