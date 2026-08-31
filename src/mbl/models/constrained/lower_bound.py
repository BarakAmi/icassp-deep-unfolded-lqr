"""SDP relaxation lower bound on the achievable average cost of a
box-constrained (infinity-norm |u| <= u_max) LQR problem -- a convex
relaxation/duality bound, not an attained policy, used as a theoretical
reference for the COCP model's trained/untrained performance.

THIRD-PARTY ATTRIBUTION
-----------------------
This implementation is **derived from the code released with**:

    Akshay Agrawal, Shane Barratt, Stephen Boyd and Bartolomeo Stellato,
    "Learning convex optimization control policies", 2019.
    https://github.com/cvxgrp/cocp -- Apache License 2.0.

**Changes were made.** The bound is expressed against this project's
`ProblemSpec` conventions, returns the achieving `P` alongside the value so it
can be frozen into a controller, and raises rather than returning `None` when
the solver reports no optimal point. Apache-2.0 §4 requires both the notice
above and this statement of modification; they are kept here, in the file that
carries the derived work, rather than only in a repository-level NOTICE, so
that the attribution survives the file being read on its own.

A caution that belongs beside the citation rather than buried: the quantity
this returns is a bound on the *infinite-horizon average* cost, and it is not a
lower bound on the finite-horizon quantity this project's figures draw. See
`docs/research/box_constrained_lower_bounds/`.
"""

import cvxpy as cp
import numpy as np


def solve_box_constrained_lower_bound(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    u_max: float,
) -> tuple[float, np.ndarray]:
    """Solve the SDP relaxation of the box-constrained infinite-horizon LQR
    problem, returning (lower_bound, P_lb): P_lb is the matrix achieving the
    bound, usable as a cost-to-go in a COCPController (e.g. via its matrix
    square root) as a strong non-learned reference policy.

    Args:
        A, B: state/control system matrices, shape (n, n)/(n, m).
        Q, R: quadratic state/control cost matrices, shape (n, n)/(m, m).
        W: process noise covariance, shape (n, n).
        u_max: the scalar infinity-norm bound on the control input.

    Returns:
        (lower_bound, P_lb): the SDP's optimal value (a lower bound on the
        true achievable average cost under the box constraint) and the
        corresponding P matrix.

    Raises:
        RuntimeError: If the SDP solver fails to find a feasible/optimal
            solution (``P.value is None``); the message includes the solver's
            reported status.
    """
    n, m = A.shape[0], B.shape[1]
    P = cp.Variable((n, n), PSD=True)
    R_relaxed = cp.Variable((m, m), PSD=True)
    lam = cp.Variable(m, nonneg=True)

    objective = cp.trace(P @ W) - (u_max**2) * cp.sum(lam)
    constraints = [
        R_relaxed - R << cp.diag(lam),
        P >> 0,
        R_relaxed >> 0,
        lam >= 0,
        cp.bmat(
            [
                [R_relaxed + B.T @ P @ B, B.T @ P @ A],
                [A.T @ P @ B, Q + A.T @ P @ A - P],
            ]
        )
        >> 0,
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    result = problem.solve()

    if P.value is None:
        raise RuntimeError(
            f"Box-constrained lower-bound SDP failed to solve (status={problem.status})."
        )
    return float(result), P.value
