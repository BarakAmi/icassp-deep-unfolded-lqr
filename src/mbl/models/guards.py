"""Control-theory domain guards.

These encode the *mathematical* preconditions of the LQR machinery, kept
separate from the structural (dimension/shape) guards in ``core.utils`` because
they depend on the system matrices and on control-theoretic facts:

* Finite-horizon backward Riccati is well-posed for *any* ``(A, B)`` provided
  ``R`` is positive definite (so ``M_k = R + Bᵀ P_{k+1} B`` is invertible) and
  the cost stacks cover the horizon. Stabilizability is **not** required, so
  controllability here is a *diagnostic* (:func:`check_controllability`, warns).
* The infinite-horizon DARE, by contrast, requires ``(A, B)`` stabilizable for a
  stabilizing solution to exist -- so :func:`ensure_stabilizable_for_dare` is a
  **hard gate** (raises), ready for a future DARE solver.

Rank tests use SVD with explicit tolerances (never exact equality) to avoid
false stability failures on finite-precision matrices. Expensive checks are
content-cached so a static ``(A, B)`` is analysed exactly once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..core.cost.quadratic_cost import QuadraticCost
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.system.linear_system import LinearSystem
from ..core.utils.guards import GuardTier, content_cached, guard


def _trailing_2d(matrix: np.ndarray) -> tuple[int, int]:
    """Last two dims of a matrix or a time-stack of matrices.

    Args:
        matrix: A matrix, shape ``(..., rows, cols)``.

    Returns:
        ``(rows, cols)``, i.e. ``matrix.shape[-2:]``.
    """
    return matrix.shape[-2], matrix.shape[-1]


@guard(GuardTier.DOMAIN)
def ensure_lqr_dimensional_coherence(
    *,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> None:
    """Assert the LQR data agree on ``(n, m)``: ``A:(n,n)``, ``B:(n,m)``,
    ``Q:(...,n,n)``, ``R:(...,m,m)``. This is the genuinely missing
    precondition -- ``Q``/``R`` silently mis-sized against ``(A, B)`` otherwise
    only surfaces mid-recursion.

    Args:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        Q: Running state cost, shape ``(n, n)`` or ``(..., n, n)``.
        R: Control cost, shape ``(m, m)`` or ``(..., m, m)``.

    Raises:
        ValueError: If `A` is not square, or `B`/`Q`/`R` disagree with the
            ``(n, m)`` implied by `A`/`B`. Skipped (a no-op) when
            `GuardTier.DOMAIN` is bypassed for performance.
    """
    n, n2 = _trailing_2d(A)
    if n != n2:
        raise ValueError(f"A must be square (n, n), got trailing shape {(n, n2)}.")
    bn, m = _trailing_2d(B)
    if bn != n:
        raise ValueError(f"B must have shape (n={n}, m), got trailing shape {(bn, m)}.")
    qn, qn2 = _trailing_2d(Q)
    if (qn, qn2) != (n, n):
        raise ValueError(f"Q must have trailing shape (n={n}, n={n}), got {(qn, qn2)}.")
    rm, rm2 = _trailing_2d(R)
    if (rm, rm2) != (m, m):
        raise ValueError(f"R must have trailing shape (m={m}, m={m}), got {(rm, rm2)}.")


@guard(GuardTier.DOMAIN)
def ensure_riccati_horizon_coherence(
    *, horizon: int, Q: np.ndarray, R: np.ndarray
) -> None:
    """The backward recursion indexes ``Q[horizon]`` and ``R[k]`` for
    ``k = 0..horizon-1``, so both must be genuinely time-stacked (3D) and long
    enough. A 2D (time-invariant) ``Q`` would make ``Q[horizon]`` index rows,
    silently corrupting the terminal cost.

    Args:
        horizon: The finite horizon length ``N``.
        Q: Running state cost; must be time-stacked, shape ``(>=N+1, n, n)``.
        R: Control cost; must be time-stacked, shape ``(>=N, m, m)``.

    Raises:
        ValueError: If `Q` is not 3D or has fewer than ``horizon + 1`` slices,
            or `R` is not 3D or has fewer than `horizon` slices. Skipped (a
            no-op) when `GuardTier.DOMAIN` is bypassed for performance.
    """
    if Q.ndim != 3 or Q.shape[0] < horizon + 1:
        raise ValueError(
            f"Riccati requires a time-stacked Q of length >= horizon+1="
            f"{horizon + 1}, got shape {Q.shape}."
        )
    if R.ndim != 3 or R.shape[0] < horizon:
        raise ValueError(
            f"Riccati requires a time-stacked R of length >= horizon="
            f"{horizon}, got shape {R.shape}."
        )


def _svd_rank(matrix: np.ndarray, rtol: float, atol: float) -> int:
    """Numerically robust rank: count singular values above a relative+absolute
    tolerance, rather than testing exact equality.

    Args:
        matrix: The matrix to rank, shape ``(rows, cols)``.
        rtol: Relative tolerance, scaled by the largest singular value and the
            matrix's largest dimension.
        atol: Absolute tolerance floor.

    Returns:
        The number of singular values exceeding
        ``max(atol, rtol * largest_singular_value * max(matrix.shape))``.
    """
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0
    threshold = max(atol, rtol * singular[0] * max(matrix.shape))
    return int(np.count_nonzero(singular > threshold))


@content_cached()
def _controllability_rank(
    A: np.ndarray, B: np.ndarray, rtol: float, atol: float
) -> int:
    """Rank of the controllability matrix ``[B, AB, ..., A^(n-1) B]``.

    Args:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        rtol: Relative tolerance forwarded to `_svd_rank`.
        atol: Absolute tolerance forwarded to `_svd_rank`.

    Returns:
        The controllability matrix's numerical rank (``== n`` iff controllable).
    """
    n = A.shape[-1]
    blocks = [B]
    for _ in range(1, n):
        blocks.append(A @ blocks[-1])
    ctrb = np.concatenate(blocks, axis=1)  # (n, n*m)
    return _svd_rank(ctrb, rtol, atol)


@dataclass(frozen=True)
class ControllabilityReport:
    """Result of a `check_controllability` diagnostic.

    Attributes:
        controllable: Whether ``rank == state_dim``.
        rank: The computed numerical rank of the controllability matrix.
        state_dim: The state dimension ``n``.
    """

    controllable: bool
    rank: int
    state_dim: int


@guard(GuardTier.EXPENSIVE)
def check_controllability(
    A: np.ndarray,
    B: np.ndarray,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    raise_on_fail: bool = False,
) -> None:
    """Finite-horizon **diagnostic**: SVD-rank of the controllability matrix
    ``[B, AB, ..., Aⁿ⁻¹B]``. Warns (does not raise) when ``(A, B)`` is not
    controllable, since finite-horizon Riccati is still well-posed; escalate
    with ``raise_on_fail=True``. Not a precondition of the finite-horizon solve
    (see module docstring).

    Args:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        rtol: Relative tolerance forwarded to `_svd_rank`.
        atol: Absolute tolerance forwarded to `_svd_rank`.
        raise_on_fail: If ``True``, raise instead of warning when
            uncontrollable.

    Raises:
        ValueError: If `raise_on_fail` is ``True`` and ``(A, B)`` is not
            controllable. Skipped (a no-op) when `GuardTier.EXPENSIVE` is
            bypassed for performance.

    Warns:
        UserWarning: If `raise_on_fail` is ``False`` (default) and ``(A, B)``
            is not controllable.
    """
    n = A.shape[-1]
    rank = _controllability_rank(A, B, rtol, atol)
    report = ControllabilityReport(controllable=rank == n, rank=rank, state_dim=n)
    if not report.controllable:
        message = (
            f"(A, B) is not controllable (rank {rank} < state dim {n}). "
            "Finite-horizon Riccati remains well-posed, but the closed loop may "
            "not be stabilizable for an infinite-horizon / DARE formulation."
        )
        if raise_on_fail:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)


@content_cached()
def _is_stabilizable(A: np.ndarray, B: np.ndarray, rtol: float, atol: float) -> bool:
    """Discrete-time PBH test: every mode with ``|lambda| >= 1`` (unstable or
    marginal) must be controllable, i.e. ``rank([A - lambda I | B]) == n``.

    Args:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        rtol: Relative tolerance forwarded to `_svd_rank`, and used to decide
            whether an eigenvalue counts as "stable" (``|lambda| < 1 - rtol``).
        atol: Absolute tolerance forwarded to `_svd_rank`.

    Returns:
        ``True`` iff every unstable/marginal mode of `A` is controllable.
    """
    n = A.shape[-1]
    identity = np.eye(n)
    for eigenvalue in np.linalg.eigvals(A):
        if abs(eigenvalue) < 1.0 - rtol:
            continue  # strictly stable mode: irrelevant to stabilizability
        pencil = np.hstack([A - eigenvalue * identity, B]).astype(complex)
        if _svd_rank(pencil, rtol, atol) < n:
            return False
    return True


@guard(GuardTier.EXPENSIVE)
def ensure_stabilizable_for_dare(
    A: np.ndarray,
    B: np.ndarray,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Infinite-horizon **hard gate**: raise unless ``(A, B)`` is stabilizable,
    the precondition for the DARE to admit a stabilizing solution. Intended for
    a future infinite-horizon solver; the finite-horizon path must not call
    this (it would wrongly reject solvable finite-horizon problems).

    Args:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
        rtol: Relative tolerance (see `_is_stabilizable`).
        atol: Absolute tolerance (see `_is_stabilizable`).

    Raises:
        ValueError: If ``(A, B)`` is not stabilizable. Skipped (a no-op) when
            `GuardTier.EXPENSIVE` is bypassed for performance.
    """
    if not _is_stabilizable(A, B, rtol, atol):
        raise ValueError(
            "(A, B) is not stabilizable: an unstable/marginal mode is "
            "uncontrollable, so the infinite-horizon DARE has no stabilizing "
            "solution."
        )


def require_linear_quadratic(
    problem: "OptimalControlProblem",
) -> "tuple[LinearSystem, QuadraticCost]":
    """Narrow a problem's abstract ``system``/``cost`` to the concrete
    `LinearSystem`/`QuadraticCost` pair the LQR machinery requires — the one
    fail-fast gate every Riccati/GD/COCP controller shares, instead of each
    reaching into ``problem.system.A_t`` and crashing with an `AttributeError`
    deep inside a solve when handed a non-linear problem.

    Args:
        problem: The `OptimalControlProblem` to narrow.

    Returns:
        ``(problem.system, problem.cost)``, typed concretely.

    Raises:
        TypeError: If the system is not a `LinearSystem` or the cost is not a
            `QuadraticCost`.
    """
    system, cost = problem.system, problem.cost
    if not isinstance(system, LinearSystem):
        raise TypeError(
            f"This controller requires a LinearSystem, got {type(system).__name__}."
        )
    if not isinstance(cost, QuadraticCost):
        raise TypeError(
            f"This controller requires a QuadraticCost, got {type(cost).__name__}."
        )
    return system, cost
