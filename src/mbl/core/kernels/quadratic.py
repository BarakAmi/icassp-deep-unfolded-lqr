"""The one quadratic-objective kernel (T2.d) — dual-backend, flag-honoring.

Consolidates the five pre-S2 quadratic cost implementations
(`QuadraticCost.__call__`, `RolloutModel._differentiable_cost`,
`iterative_gd._build_cost_evaluator`, `oracles.make_rollout_cost_oracle`,
and the plotting-layer cumulative average) into one module: the per-step
quadratic form, the cumulative per-step cost curve, and the scalar/
per-sample trajectory totals all live here exactly once, and every entry
point honors the cost's declared ``include_terminal_cost`` /
``is_time_averaged`` flags (the C2 fix — no consumer can drift again).

Bit-stability heritage (§7.1 golden masters):

* `cumulative_quadratic_cost` reproduces `QuadraticCost.__call__`'s exact
  NumPy operation sequence (einsum → slice-sum → cumsum → optional
  terminal tail-add → optional elapsed-step division).
* `total_quadratic_cost(reduction=PER_SAMPLE)` reproduces the
  signal-space-GD evaluator's and the rollout oracle's exact torch
  sequence (einsum → per-sample time sums → optional terminal → optional
  ``/T``) — the frozen ``J_history`` goldens ride it.
* `total_quadratic_cost(reduction=BATCH_MEAN)` uses the streamed-mean
  formulation (`mean` over batch×time stage costs first): the standard,
  numerically well-scaled training-loss form, and bit-identical to the
  pre-S2 training loss under the default ``(False, True)`` flags — the
  frozen ``J_opt`` golden rides it.

The two reductions are the same mathematical objective (the batch
expectation of the per-sample total); they differ only in floating-point
association order, and a kernel test pins their agreement at
precision-appropriate tolerance.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ..runtime import shape_checked
from .dispatch import array_namespace

StateTrajectory = Float[np.ndarray, "batch timep1 n"] | Float[Tensor, "batch timep1 n"]
"""Batched state trajectory ``X``: one more step than the control horizon."""

ControlTrajectory = Float[np.ndarray, "batch time m"] | Float[Tensor, "batch time m"]
"""Batched control trajectory ``U`` over the ``time``-step horizon."""

StateCostMatrix = (
    Float[np.ndarray, "timep1 n n"]
    | Float[np.ndarray, "n n"]
    | Float[Tensor, "timep1 n n"]
    | Float[Tensor, "n n"]
)
"""Running state cost ``Q``: time-invariant, or one slice per state step."""

ControlCostMatrix = (
    Float[np.ndarray, "time m m"]
    | Float[np.ndarray, "m m"]
    | Float[Tensor, "time m m"]
    | Float[Tensor, "m m"]
)
"""Control cost ``R``: time-invariant, or one slice per control step."""

CostCurve = Float[np.ndarray, "batch time"] | Float[Tensor, "batch time"]
"""Per-sample cumulative cost at every elapsed step ``k = 1..time``."""


@dataclass(frozen=True)
class CostConventions:
    """The two declared conventions of a quadratic objective, as one value.

    Every kernel entry point takes this bundle instead of two loose booleans,
    so the pair can never be threaded half-way (the drift C2 documented).
    `QuadraticCost.conventions` exposes a cost's own declared pair, making
    ``conventions=cost.conventions`` the canonical call-site form.

    Attributes:
        include_terminal_cost: Add the terminal-state term ``x_N^T Q x_N``.
        is_time_averaged: Divide totals/curve entries by the elapsed steps.
    """

    include_terminal_cost: bool
    is_time_averaged: bool


class CostReduction(StrEnum):
    """How `total_quadratic_cost` reduces the per-sample trajectory totals.

    Both reductions evaluate the same declared objective; they are distinct
    *named* floating-point association orders (the `LossReduction` seam the
    blueprint names for T2.d), each carrying one golden-master heritage —
    see the module docstring.
    """

    PER_SAMPLE = "per_sample"
    """No batch reduction: one total per trajectory, shape ``(batch,)``."""

    BATCH_MEAN = "batch_mean"
    """Batch expectation as a scalar, streamed-mean association order —
    the differentiable training-loss form."""


def time_invariant_slice(mat: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """Reduce a possibly time-stacked cost matrix to its time-invariant slice.

    The canonical home (T2.e) of the ``mat[0] if mat.ndim == 3 else mat``
    idiom previously copy-pasted across ≥6 call sites. Only meaningful for
    costs that are genuinely time-invariant even when stored time-stacked
    (the Riccati preconditions require 3D storage); a truly time-varying
    stack has no single slice, and callers own that judgement.

    Args:
        mat: Cost matrix, shape ``(d, d)`` or ``(N, d, d)``.

    Returns:
        The ``(d, d)`` slice (``mat`` itself when already 2D — never a copy).
    """
    return mat[0] if mat.ndim == 3 else mat


def quadratic_form(
    mat: np.ndarray | Tensor, arr: np.ndarray | Tensor
) -> np.ndarray | Tensor:
    """Per-step, per-batch-element quadratic form ``arr_t^T @ mat_t @ arr_t``.

    The single numeric primitive under every quadratic cost evaluation —
    the pre-S2 NumPy ``core.cost.quadratic_form`` einsum and its private
    torch twin (`oracles._torch_quadratic_form`) merged into one
    dual-backend implementation. Encodes this project's 2D-vs-3D
    (time-invariant vs. time-stacked) cost-matrix convention.

    Args:
        mat: Cost matrix, shape ``(N, d, d)`` (time-stacked) or ``(d, d)``
            (time-invariant, broadcast across all ``N`` steps).
        arr: Trajectory array, shape ``(batch, N, d)``, same backend as
            `mat`.

    Returns:
        The per-step, per-batch-element quadratic form, shape
        ``(batch, N)``, on the operands' backend and device.
    """
    xp = array_namespace(mat, arr)
    if mat.ndim == 2:
        return cast("np.ndarray | Tensor", xp.einsum("btn,nk,btk->bt", arr, mat, arr))
    return cast("np.ndarray | Tensor", xp.einsum("btn,tnk,btk->bt", arr, mat, arr))


def _terminal_quadratic_form(
    mat: np.ndarray | Tensor, x_final: np.ndarray | Tensor
) -> np.ndarray | Tensor:
    """Terminal-state cost ``x_N^T @ mat @ x_N`` per batch element, ``(batch,)``."""
    xp = array_namespace(mat, x_final)
    return cast("np.ndarray | Tensor", xp.einsum("bn,nk,bk->b", x_final, mat, x_final))


@shape_checked
def cumulative_quadratic_cost(
    Q: StateCostMatrix,
    R: ControlCostMatrix,
    X: StateTrajectory,
    U: ControlTrajectory,
    *,
    conventions: CostConventions,
) -> CostCurve:
    """Per-sample cumulative quadratic cost at every elapsed step.

    The curve convention `QuadraticCost.__call__` has always exposed,
    reproduced operation-for-operation (golden-master bit-stability on the
    NumPy path) and now executable natively on either backend: entry ``k-1``
    holds ``sum_{t=0}^{k-1} (x_t^T Q_t x_t + u_t^T R_t u_t)`` for
    ``k = 1..N``, with the terminal term ``x_k^T Q_k x_k`` folded into every
    entry when `include_terminal_cost`, and each entry divided by its
    elapsed step count ``k`` when `is_time_averaged`.

    Args:
        Q: Running state cost, shape ``(n, n)`` or ``(N+1, n, n)``.
        R: Control cost, shape ``(m, m)`` or ``(N, m, m)``.
        X: State trajectory, shape ``(batch, N+1, n)``.
        U: Control trajectory, shape ``(batch, N, m)``.
        conventions: The declared terminal/averaging convention pair; the
            terminal term is folded into each entry, and each entry ``k-1``
            is divided by its elapsed step count ``k``.

    Returns:
        The per-sample cumulative cost curve, shape ``(batch, N)``.
    """
    xp = array_namespace(Q, R, X, U)
    cost_X = quadratic_form(Q, X)  # (batch, N+1)
    cost_U = quadratic_form(R, U)  # (batch, N)

    stage_costs = cost_X[:, :-1] + cost_U
    J_cum = xp.cumsum(stage_costs, 1)  # (batch, N)

    if conventions.include_terminal_cost:
        # Adds x_k^T Q_k x_k for k = 1, ..., N (the whole lookahead tail,
        # matching QuadraticCost's historical curve semantics exactly).
        J_cum = J_cum + cost_X[:, 1:]

    if conventions.is_time_averaged:
        k_steps: np.ndarray | Tensor
        if xp is torch:
            k_steps = torch.arange(1, X.shape[1], device=J_cum.device)
        else:
            k_steps = np.arange(1, X.shape[1])  # [1, 2, ..., N]
        J_cum = J_cum / k_steps
    return cast("np.ndarray | Tensor", J_cum)


@shape_checked
def total_quadratic_cost(
    Q: StateCostMatrix,
    R: ControlCostMatrix,
    X: StateTrajectory,
    U: ControlTrajectory,
    *,
    conventions: CostConventions,
    reduction: CostReduction = CostReduction.PER_SAMPLE,
) -> np.ndarray | Tensor:
    """Whole-trajectory quadratic cost under the declared cost flags.

    The scalar convention every torch consumer previously reimplemented —
    now defined once, differentiable (pure tensor ops on the torch path,
    no detach/item), and honoring `include_terminal_cost` /
    `is_time_averaged` unconditionally (the C2 fix). The time-invariant
    matrix convention applies: time-stacked `Q`/`R` are reduced via
    `time_invariant_slice` (a genuinely time-varying total is out of scope,
    exactly as it was for every consolidated consumer).

    Args:
        Q: Running state cost, shape ``(n, n)`` or a time-invariant stack.
        R: Control cost, shape ``(m, m)`` or a time-invariant stack.
        X: State trajectory, shape ``(batch, N+1, n)``.
        U: Control trajectory, shape ``(batch, N, m)``.
        conventions: The declared terminal/averaging convention pair.
        reduction: `CostReduction.PER_SAMPLE` for one total per trajectory
            (shape ``(batch,)``), `CostReduction.BATCH_MEAN` for the scalar
            batch expectation in streamed-mean order (see `CostReduction`).

    Returns:
        Per-sample totals ``(batch,)`` under ``PER_SAMPLE``; a scalar
        (0-d tensor on torch, NumPy scalar on NumPy) under ``BATCH_MEAN``.
    """
    Q_2d = time_invariant_slice(Q)
    R_2d = time_invariant_slice(R)
    horizon = U.shape[1]

    cost_x = quadratic_form(Q_2d, X[:, :-1])  # (batch, N)
    cost_u = quadratic_form(R_2d, U)  # (batch, N)

    if reduction is CostReduction.PER_SAMPLE:
        total = cost_x.sum(1) + cost_u.sum(1)  # (batch,)
        if conventions.include_terminal_cost:
            total = total + _terminal_quadratic_form(Q_2d, X[:, -1])
        if conventions.is_time_averaged:
            total = total / horizon
        return total

    # BATCH_MEAN: streamed-mean association order — divide-early over the
    # batch*time stage costs (well-scaled for large ensembles, and the
    # bit-heritage of the pre-S2 training loss under default flags).
    total = cost_x.mean() + cost_u.mean()
    if not conventions.is_time_averaged:
        total = total * horizon
    if conventions.include_terminal_cost:
        terminal = _terminal_quadratic_form(Q_2d, X[:, -1]).mean()
        total = total + (
            terminal / horizon if conventions.is_time_averaged else terminal
        )
    return total
