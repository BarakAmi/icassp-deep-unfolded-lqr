"""The box-constrained LQR's lower bounds, as the small concave problems they are.

`lower_bound.py` states the box-aware bound as a semidefinite program, ported
from the COCP paper's own code and correct as written. It is also the Lagrangian
dual of the box constraint, and saying so collapses it: with the objective
monotone in ``P`` the relaxed control weight sits at its upper bound
``R + diag(lambda)``, and for a fixed weight the program's large LMI is a
Riccati inequality whose *maximal* solution is the stabilising DARE solution.
What remains is a maximisation over ``lambda >= 0`` alone -- **30 variables at
n = 100 against the semidefinite form's 5545** -- whose every evaluation is one
DARE solve.

Measured, that is **4.9 hours to under 7 seconds** at n = 100, agreeing with an
interior-point solver to thirteen significant figures at sizes where one can be
run, while the shipped first-order path returns points that violate the
program's own ``P >= 0`` constraint by 1752.

**Two conventions, and they must never be mixed.**
`infinite_horizon_box_bound` bounds the steady-state average cost;
`finite_horizon_box_bound` bounds the finite-horizon time-average this project
actually *evaluates*, which is a cheaper quantity because a short trajectory
from a low-energy start has not reached the stationary distribution and no
terminal term is charged. The infinite-horizon bound does not bound any figure's
drawn curve, and a document quoting either must say which it is in.

**Both are certified at every iterate.** Weak duality holds for *any*
``lambda >= 0``, so a value returned before convergence is still a valid bound --
early stopping is safe, and feasibility is not something a solver has to be
trusted for.

**What these bound, and what they do not.** They bound box-*feasible* policies.
An unconstrained controller may cost less and routinely does; it is not a
competitor for the constrained optimum, and a figure drawing one alongside this
floor has to say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import minimize

from ..guards import ensure_stabilizable_for_dare

#: Multipliers at or below this count as inactive, where complementary
#: slackness asks only for an inequality. **Not cosmetic**: measured with an
#: equality on the inactive set, a converged n = 100 solve reports a KKT
#: residual of 4.2e-03 where the truth is 5.2e-13, because 24 of its 30
#: multipliers are legitimately zero.
_INACTIVE = 1e-9


#: Iteration cap for the concave maximisation. A module constant rather than a
#: parameter: the argument gate is six, and an iteration budget is the least
#: interesting thing a caller could be given control of.
_MAX_ITER = 2000

#: Gradient tolerance for the same maximisation. **Measured, not chosen.** At
#: n = 100 a tolerance of 1e-14 and one of 1e-12 agree on the value to every
#: printed digit, on the KKT residual (5.17e-13), and on the iteration count
#: (10) -- and cost **20.2 s against 6.3 s**, the difference going entirely into
#: line-search evaluations chasing a gradient norm that is already at the noise
#: floor. Loosening further to 1e-10 saves another 1.6 s and gives up two
#: decades of stationarity, which is the trade this value declines.
_GRADIENT_TOL = 1e-12


@dataclass(frozen=True)
class BoxLQRData:
    """The box-constrained LQR a bound is taken over.

    A named bundle rather than seven loose arrays, because both bounds need the
    same six things and one of them needs two more.

    Attributes:
        A: State transition, ``(n, n)``.
        B: Control input, ``(n, m)``.
        Q: State cost, ``(n, n)``.
        R: Control cost, ``(m, m)``; must be positive definite.
        W: Process-noise covariance, ``(n, n)``.
        u_max: The scalar infinity-norm bound on the control.
    """

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    Q: NDArray[np.float64]
    R: NDArray[np.float64]
    W: NDArray[np.float64]
    u_max: float


@dataclass(frozen=True)
class DualBound:
    """A lower bound on the box-constrained cost, and the evidence for it.

    Attributes:
        value: The bound. Valid for **any** non-negative multipliers by weak
            duality, so this is certified whether or not the maximisation
            converged -- `kkt_residual` says how tight it is, never whether it
            holds.
        multipliers: The multipliers achieving it: ``(m,)`` for the
            infinite-horizon bound, ``(horizon, m)`` for the finite-horizon one.
        cost_to_go: The ``P`` achieving the bound, ``(n, n)`` -- what a frozen
            policy is seeded with. `None` for the finite-horizon bound, whose
            cost-to-go is a time-varying sequence rather than one matrix.
        kkt_residual: Stationarity of the dual problem, with complementary
            slackness respected on the inactive set.
        lmi_slack: Least eigenvalue of the program's large LMI at `cost_to_go`;
            non-negative to numerical precision by construction, and reported
            so that "by construction" is checkable. `nan` where there is no
            single LMI to evaluate.
    """

    value: float
    multipliers: NDArray[np.float64]
    cost_to_go: NDArray[np.float64] | None
    kkt_residual: float
    lmi_slack: float


def _dual_at(
    data: BoxLQRData, multipliers: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """The relaxed problem's value, its control second moments, and its ``P``.

    Args:
        data: The box-constrained LQR.
        multipliers: The multipliers ``lambda``, ``(m,)``.

    Returns:
        ``(tr(P W), diag(E[u u^T]), P)``.
    """
    A, B, R, W = data.A, data.B, data.R, data.W
    relaxed = R + np.diag(multipliers)
    P = solve_discrete_are(A, B, data.Q, relaxed)
    gain = np.linalg.solve(relaxed + B.T @ P @ B, B.T @ P @ A)
    closed_loop = A - B @ gain
    state_covariance = solve_discrete_lyapunov(closed_loop, W)
    control_second_moment = np.diag(gain @ state_covariance @ gain.T).copy()
    return float(np.trace(P @ W)), control_second_moment, P


def _kkt_residual(
    multipliers: NDArray[np.float64],
    second_moment: NDArray[np.float64],
    u_max: float,
) -> float:
    """Dual stationarity, with complementary slackness on the inactive set.

    Args:
        multipliers: The multipliers ``lambda``.
        second_moment: The corresponding ``E[u_i^2]``.
        u_max: The scalar infinity-norm bound.

    Returns:
        The largest violation: an equality where a multiplier is active, and
        only an upper inequality where it is not.
    """
    bound = u_max**2
    active = multipliers > _INACTIVE
    equality = (
        float(np.max(np.abs(second_moment[active] - bound))) if active.any() else 0.0
    )
    inequality = (
        float(np.max(np.maximum(second_moment[~active] - bound, 0.0)))
        if (~active).any()
        else 0.0
    )
    return max(equality, inequality)


def _lmi_slack(
    data: BoxLQRData, P: NDArray[np.float64], multipliers: NDArray[np.float64]
) -> float:
    """Least eigenvalue of the semidefinite program's large LMI at ``P``.

    Args:
        data: The box-constrained LQR.
        P: The cost-to-go to evaluate at.
        multipliers: The multipliers ``lambda``.

    Returns:
        The least eigenvalue, symmetrised before the decomposition.
    """
    A, B, Q = data.A, data.B, data.Q
    relaxed = data.R + np.diag(multipliers)
    block = np.block(
        [
            [relaxed + B.T @ P @ B, B.T @ P @ A],
            [A.T @ P @ B, Q + A.T @ P @ A - P],
        ]
    )
    return float(np.min(np.linalg.eigvalsh((block + block.T) / 2.0)))


def infinite_horizon_box_bound(data: BoxLQRData) -> DualBound:
    """The box-aware bound on the steady-state average cost.

    The semidefinite program of `lower_bound.solve_box_constrained_lower_bound`,
    reduced to its dual (see the module docstring) and solved in seconds rather
    than hours.

    Args:
        data: The box-constrained LQR to bound.

    Returns:
        The `DualBound`, whose `cost_to_go` seeds a frozen policy.

    Raises:
        ValueError: If ``(A, B)`` is not stabilisable, so the DARE has no
            stabilising solution and the reduction's premise does not hold.
    """
    # The reduction rests on the maximal solution of a Riccati inequality being
    # the STABILISING DARE solution, which needs this. Assumed rather than
    # checked would be exactly the kind of premise that fails quietly.
    ensure_stabilizable_for_dare(data.A, data.B)
    bound = data.u_max**2

    def negated(multipliers: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value, second_moment, _ = _dual_at(data, multipliers)
        return (
            -(value - bound * float(multipliers.sum())),
            -(second_moment - bound),
        )

    m = data.B.shape[1]
    result = minimize(
        negated,
        np.zeros(m),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * m,
        options={"maxiter": _MAX_ITER, "ftol": 1e-16, "gtol": _GRADIENT_TOL},
    )
    multipliers = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    value, second_moment, P = _dual_at(data, multipliers)
    return DualBound(
        value=value - bound * float(multipliers.sum()),
        multipliers=multipliers,
        cost_to_go=P,
        kkt_residual=_kkt_residual(multipliers, second_moment, data.u_max),
        lmi_slack=_lmi_slack(data, P, multipliers),
    )


def _finite_horizon_dual(
    data: BoxLQRData,
    initial_covariance: NDArray[np.float64],
    multipliers: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """The relaxed finite-horizon value and its per-step control second moments.

    The inner problem is a finite-horizon LQR with a **time-varying** control
    weight and **zero terminal cost** -- a backward Riccati recursion, not a
    DARE -- because that is the convention every figure in this project
    evaluates: a time-average over ``N`` steps with no terminal term charged.

    Args:
        data: The box-constrained LQR.
        initial_covariance: Covariance of ``x_0``, ``(n, n)``.
        multipliers: The multipliers, ``(horizon, m)``.

    Returns:
        ``(value, second_moments)`` with `second_moments` shaped
        ``(horizon, m)``.
    """
    A, B, Q, R, W = data.A, data.B, data.Q, data.R, data.W
    horizon = multipliers.shape[0]
    n = A.shape[0]

    P = np.zeros((n, n))
    costs_to_go: list[NDArray[np.float64]] = [P]
    gains: list[NDArray[np.float64]] = []
    for step in range(horizon - 1, -1, -1):
        relaxed = R + np.diag(multipliers[step])
        gain = np.asarray(
            np.linalg.solve(relaxed + B.T @ P @ B, B.T @ P @ A), dtype=np.float64
        )
        stepped = Q + A.T @ P @ A - A.T @ P @ B @ gain
        P = np.asarray((stepped + stepped.T) / 2.0, dtype=np.float64)
        costs_to_go.append(P)
        gains.append(gain)
    costs_to_go.reverse()
    gains.reverse()

    value = float(np.trace(costs_to_go[0] @ initial_covariance)) + sum(
        float(np.trace(costs_to_go[step] @ W)) for step in range(1, horizon)
    )

    covariance = initial_covariance.copy()
    second_moments = np.empty_like(multipliers)
    for step in range(horizon):
        gain = gains[step]
        second_moments[step] = np.diag(gain @ covariance @ gain.T)
        closed_loop = A - B @ gain
        covariance = closed_loop @ covariance @ closed_loop.T + W
    return value, second_moments


def finite_horizon_box_bound(
    data: BoxLQRData,
    initial_covariance: NDArray[np.float64],
    horizon: int,
    *,
    time_varying: bool = True,
) -> DualBound:
    """The box-aware bound **in the convention the figures evaluate**.

    `infinite_horizon_box_bound` bounds the steady-state average cost, which is
    not what any figure here draws: a 100-step time-average from a low-energy
    start, with no terminal term, is a genuinely cheaper quantity, and the
    steady-state bound sits above it. Measured at n = 100 the two differ by
    0.60 -- enough that drawing the wrong one as a floor puts it above every
    curve in the figure, including a provably optimal controller's.

    Args:
        data: The box-constrained LQR.
        initial_covariance: Covariance of ``x_0``, ``(n, n)``.
        horizon: The number of steps the time-average runs over.
        time_varying: Whether the multipliers may differ per step. Tighter, at
            ``horizon`` times the variables; both are concave and both are
            certified.

    Returns:
        The `DualBound`. `cost_to_go` is `None` -- the finite-horizon inner
        problem has a *sequence* of cost-to-go matrices, not one, so there is
        nothing here for a frozen policy to be seeded with.
    """
    m = data.B.shape[1]
    bound = data.u_max**2
    shape = (horizon, m) if time_varying else (1, m)

    def expand(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        block = flat.reshape(shape)
        return block if time_varying else np.repeat(block, horizon, axis=0)

    def negated(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        multipliers = expand(flat)
        value, second_moments = _finite_horizon_dual(
            data, initial_covariance, multipliers
        )
        objective = (value - bound * float(multipliers.sum())) / horizon
        gradient = (second_moments - bound) / horizon
        collapsed = gradient if time_varying else gradient.sum(axis=0, keepdims=True)
        return -objective, -collapsed.ravel()

    result = minimize(
        negated,
        np.zeros(int(np.prod(shape))),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * int(np.prod(shape)),
        options={"maxiter": _MAX_ITER, "ftol": 1e-16, "gtol": _GRADIENT_TOL},
    )
    multipliers = expand(np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None))
    value, second_moments = _finite_horizon_dual(data, initial_covariance, multipliers)
    return DualBound(
        value=(value - bound * float(multipliers.sum())) / horizon,
        multipliers=multipliers,
        cost_to_go=None,
        kkt_residual=_kkt_residual(
            multipliers.ravel(), second_moments.ravel(), data.u_max
        ),
        lmi_slack=float("nan"),
    )
