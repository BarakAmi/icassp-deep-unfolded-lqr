"""Finite-horizon LQR Riccati mathematics, expressed once per operation
against the dual-backend kernel surface (T1.e) — the recursion no longer
belongs to any substrate.

`compute_lqr_gradient_matrices` moved here verbatim from
``models.lqr_gradient`` (which now re-exports it): it was already the
tree's proof that one implementation can serve both frameworks (``.mT``
and ``@`` are defined identically by NumPy and torch), and the kernel
layer is where that discipline now lives. `riccati_recursion` generalizes
``finite_horizon_riccati``'s loop the same way, with the executing
namespace resolved once through `dispatch.array_namespace` — on NumPy
inputs it performs literally the pre-S2 operation sequence (golden-master
bit-stability), on torch inputs it runs natively on the tensors' device.

Preconditions (dimensional coherence, genuinely time-stacked cost
matrices, controllability diagnostics) stay with the callers in
``models``: the kernel is pure mathematics over already-validated,
already-materialized arrays.
"""

from typing import Any, cast

import numpy as np
from jaxtyping import Float
from torch import Tensor

from .dispatch import array_namespace

CostToGoStack = Float[np.ndarray, "timep1 n n"] | Float[Tensor, "timep1 n n"]
"""Cost-to-go matrices ``P_k`` for ``k = 0..N``."""

GainStack = Float[np.ndarray, "time m n"] | Float[Tensor, "time m n"]
"""Feedback gains ``K_k`` (``u_k = -K_k x_k``) for ``k = 0..N-1``."""


def compute_lqr_gradient_matrices(
    P_next: "np.ndarray | Tensor",
    A: "np.ndarray | Tensor",
    B: "np.ndarray | Tensor",
    R: "np.ndarray | Tensor",
) -> "tuple[np.ndarray | Tensor, np.ndarray | Tensor]":
    """Compute ``M = R + B^T P_next B`` and ``C = B^T P_next A``.

    Reused by the Riccati recursion below, the analytic Riccati solver, and
    the unfolded gradient-descent refinement strategies, so the underlying
    formula is defined exactly once. ``.mT`` (transpose of the trailing two
    axes) and ``@`` (batched matrix multiplication over any leading axes)
    are defined identically by NumPy arrays and torch tensors, so a single
    implementation serves a single time step (2D) or a full time-stacked
    horizon (3D, leading time axis) on either backend.

    Args:
        P_next: Cost-to-go matrix (or matrices) ``P_{k+1}``, shape
            ``(n, n)`` or ``(N, n, n)``.
        A: State transition matrix (or matrices), shape ``(n, n)`` or
            ``(N, n, n)``.
        B: Control input matrix (or matrices), shape ``(n, m)`` or
            ``(N, n, m)``.
        R: Control cost matrix (or matrices), shape ``(m, m)`` or
            ``(N, m, m)``.

    Returns:
        ``M``: shape ``(m, m)`` or ``(N, m, m)``.
        ``C``: shape ``(m, n)`` or ``(N, m, n)``.
    """
    BtP = B.mT @ P_next
    M = R + BtP @ B
    C = BtP @ A
    return M, C


def gains_for_cost_to_go(
    P_arr: Any,
    A: Any,
    B: Any,
    R: "np.ndarray | Tensor",
    horizon: int,
) -> "np.ndarray | Tensor":
    """Feedback gains re-formed from a FROZEN cost-to-go stack.

    ``K_k = (R_k + B_k^T P_{k+1} B_k)^{-1} B_k^T P_{k+1} A_k`` — the online
    half of the Riccati control law (Annex 01 §2.4.1). The recursion that
    produced ``P_arr`` is an offline computation; forming a gain from the
    frozen stack and the matrices currently believed is a per-step solver
    expression, which is why an informed controller may do it with the plant
    it is handed while its ``P_arr`` stays the one its offline phase solved.
    On the plant `P_arr` was solved for, this reproduces `riccati_recursion`'s
    own gains bit for bit — the same expressions on the same operands.

    Args:
        P_arr: Frozen cost-to-go stack ``(N+1, n, n)`` (the offline artifact).
        A: Per-step state matrices, indexable ``A[k] -> (n, n)``.
        B: Per-step control matrices, indexable ``B[k] -> (n, m)``.
        R: Control cost stack, shape ``(>=horizon, m, m)``.
        horizon: The finite horizon length ``N``.

    Returns:
        ``K_arr``: feedback gains, shape ``(N, m, n)``.
    """
    xp = array_namespace(R)
    gains: list[Any] = []
    for k in range(horizon):
        M, C = compute_lqr_gradient_matrices(P_arr[k + 1], A[k], B[k], R[k])
        gains.append(xp.linalg.solve(M, C))
    return cast("np.ndarray | Tensor", xp.stack(gains))


def gradient_lipschitz_constant(
    P: Any,
    A: Any,
    B: Any,
    R: "np.ndarray | Tensor",
) -> float:
    """The Lipschitz constant of the per-step LQR gradient over the horizon.

    ``L = max_t lambda_max(2 (R_t + B_t^T P_{t+1} B_t))`` — the largest
    eigenvalue of exactly the gradient-coefficient tensor the unfolded
    refinement consumes as ``2 * M``. ``1/L`` is the step projected gradient
    descent converges fastest at, and ``2/L`` the step at which it stops
    converging at all.

    **The routine is `eigvalsh` by name, and that is a contract rather than a
    detail.** ``eigvals``, ``norm(., 2)``, ``svd`` and the closed 2x2 form
    each land one ULP away on the campaign's n = 4 plant, and one ULP of a
    declared ``step_size_init`` moves the `ModelID`. This kernel is the single
    home of the computation: the authoring tool, `models/analytic`'s wrapper
    and the `step_is_inverse_lipschitz` gate all resolve to these exact
    operations, which is what makes the gate's exact-equality comparison
    meaningful.

    Args:
        P: Cost-to-go matrices ``P[k]``, shape ``(N+1, n, n)`` — offset
            internally to ``P[1:]``, exactly as the gradient stacks are.
        A: State transition matrices, shape ``(N, n, n)``.
        B: Control input matrices, shape ``(N, n, m)``.
        R: Control cost matrices, shape ``(N, m, m)``.

    Returns:
        ``L``, strictly positive because ``R`` is positive definite.
    """
    M, _ = compute_lqr_gradient_matrices(P[1:], A, B, R)
    return float(np.linalg.eigvalsh(2.0 * np.asarray(M)).max())


def riccati_recursion(
    A: Any,
    B: Any,
    Q: "np.ndarray | Tensor",
    R: "np.ndarray | Tensor",
    horizon: int,
) -> tuple[CostToGoStack, GainStack]:
    """Backward Riccati recursion for the finite-horizon LQR problem.

    One algorithm, N substrates: the executing backend is resolved from
    `Q`/`R` (the cost stacks — always true arrays) through the kernel
    layer's single dispatch site, never assumed. `A`/`B` only need per-step
    indexing (``A[k] -> (n, n)``), so dense stacks and duck-typed
    containers (``TimeSeriesMatrix``) both work — but must index onto the
    same substrate `Q`/`R` occupy (T1.f: no implicit conversion inside the
    loop).

    Args:
        A: Per-step state transition matrices, indexable ``A[k]``,
            ``k = 0..horizon-1``, each ``(n, n)``.
        B: Per-step control input matrices, indexable ``B[k]``, each
            ``(n, m)``.
        Q: Running state cost stack, shape ``(>=horizon+1, n, n)``.
        R: Control cost stack, shape ``(>=horizon, m, m)``.
        horizon: The finite horizon length ``N``.

    Returns:
        ``P_arr``: cost-to-go matrices ``P_k``, shape ``(N+1, n, n)``.
        ``K_arr``: feedback gains ``K_k``, shape ``(N, m, n)``.
    """
    xp = array_namespace(Q, R)
    P_list: list[Any] = [None] * (horizon + 1)
    K_list: list[Any] = [None] * horizon

    P = Q[horizon]
    P_list[horizon] = P

    for k in reversed(range(horizon)):
        A_k, B_k = A[k], B[k]
        M, C = compute_lqr_gradient_matrices(P, A_k, B_k, R[k])
        AtP = A_k.mT @ P
        K = xp.linalg.solve(M, C)
        P = Q[k] + AtP @ (A_k - B_k @ K)
        P_list[k], K_list[k] = P, K

    return xp.stack(P_list), xp.stack(K_list)
