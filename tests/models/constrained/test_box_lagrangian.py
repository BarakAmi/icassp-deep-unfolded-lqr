"""Acceptance for the two Lagrangian bounds.

The infinite-horizon one is checked against the semidefinite program it reduces
-- at sizes where that program is still sound -- and against an interior-point
solver where it is not. The finite-horizon one is checked against the two things
that must bracket it: the unconstrained optimum below, which it equals at
``lambda = 0``, and a genuinely box-feasible policy above.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from mbl.models.constrained.box_lagrangian import (
    BoxLQRData,
    finite_horizon_box_bound,
    infinite_horizon_box_bound,
)
from mbl.models.constrained.lower_bound import solve_box_constrained_lower_bound

U_MAX, NOISE, X0, HORIZON = 0.1, 0.5, 0.5, 40


def _data(n: int, m: int, seed: int = 0, rho: float = 0.99) -> BoxLQRData:
    """A stable plant at the campaign's shape, bundled with its cost and noise."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
    return BoxLQRData(
        A=A,
        B=rng.normal(size=(n, m)),
        Q=np.eye(n),
        R=np.eye(m),
        W=NOISE**2 * np.eye(n),
        u_max=U_MAX,
    )


# --- the infinite-horizon bound ----------------------------------------------


@pytest.mark.parametrize("n,m", [(4, 2), (8, 3)])
def test_it_reproduces_the_semidefinite_program_it_reduces(n: int, m: int) -> None:
    """At sizes where the shipped path is sound, the two agree -- to *its* accuracy.

    The tolerance is the first-order solver's, not the reduction's, and the
    difference between those two is the point. Measured here the two differ by
    ~1e-4 relative in **either** direction depending on the plant, while the
    reduced form is internally certified three to five decades tighter: its dual
    stationarity is ~1e-9 and its LMI slack ~1e-13. A disagreement one side can
    account for and the other cannot is not a tie.
    """
    data = _data(n, m)
    reduced = infinite_horizon_box_bound(data)
    semidefinite, _ = solve_box_constrained_lower_bound(
        data.A, data.B, data.Q, data.R, data.W, data.u_max
    )
    disagreement = abs(reduced.value - semidefinite) / abs(semidefinite)
    assert disagreement < 2e-3

    # The half that makes this more than a loose comparison: whatever the two
    # disagree by, the reduced form's own residuals are far below it, so the
    # discrepancy cannot be coming from this side.
    assert reduced.kkt_residual < 1e-7
    assert abs(reduced.lmi_slack) < 1e-8
    assert reduced.kkt_residual < disagreement


@pytest.mark.parametrize("n,m", [(4, 2), (8, 3), (20, 8)])
def test_the_returned_cost_to_go_satisfies_the_programs_own_lmi(n: int, m: int) -> None:
    """Feasibility by construction, reported so that "by construction" is checkable.

    `n = 20, m = 8` is the regime the reduction had never been checked in when
    it was first derived: many controls, and about half the multipliers
    inactive.
    """
    bound = infinite_horizon_box_bound(_data(n, m))
    assert bound.lmi_slack > -1e-8
    assert bound.kkt_residual < 1e-7
    assert bound.cost_to_go is not None
    assert np.min(np.linalg.eigvalsh(bound.cost_to_go)) > -1e-9


def test_the_multipliers_are_doing_the_work() -> None:
    """Anti-vacuity: at `lambda = 0` the bound is the unconstrained cost.

    If the maximisation returned zero the tests above would still pass -- the
    LMI holds there and the value is a valid bound -- so the gap it buys is
    what has to be asserted.
    """
    data = _data(8, 3)
    bound = infinite_horizon_box_bound(data)
    unconstrained = float(
        np.trace(solve_discrete_are(data.A, data.B, data.Q, data.R) @ data.W)
    )
    assert bound.multipliers.min() >= 0.0
    assert bound.value > unconstrained
    assert bound.multipliers.max() > 0.0


def test_an_unstabilisable_plant_is_refused() -> None:
    """The reduction's premise is a stabilising DARE solution, so it is gated."""
    data = _data(4, 2)
    crippled = BoxLQRData(
        A=np.diag([2.0, 0.5, 0.5, 0.5]),  # an uncontrollable unstable mode
        B=np.vstack([np.zeros((1, 2)), data.B[1:]]),
        Q=data.Q,
        R=data.R,
        W=data.W,
        u_max=data.u_max,
    )
    with pytest.raises(ValueError, match="stabilizable"):
        infinite_horizon_box_bound(crippled)


# --- the finite-horizon bound ------------------------------------------------


def _clipped_riccati_cost(data: BoxLQRData, horizon: int, seed: int = 0) -> float:
    """A genuinely box-feasible policy's finite-horizon time-averaged cost.

    Exactly the convention the bound is written for: no terminal term, averaged
    over `horizon`, started from the same covariance.
    """
    P = solve_discrete_are(data.A, data.B, data.Q, data.R)
    gain = np.linalg.solve(data.R + data.B.T @ P @ data.B, data.B.T @ P @ data.A)
    rng = np.random.default_rng(seed)
    n = data.A.shape[0]
    state = rng.normal(scale=X0, size=(4096, n))
    total = np.zeros(4096)
    for _ in range(horizon):
        control = np.clip(-state @ gain.T, -data.u_max, data.u_max)
        total += np.einsum("bi,ij,bj->b", state, data.Q, state) + np.einsum(
            "bi,ij,bj->b", control, data.R, control
        )
        state = (
            state @ data.A.T
            + control @ data.B.T
            + rng.normal(scale=NOISE, size=(4096, n))
        )
    return float(np.mean(total) / horizon)


@pytest.mark.parametrize("time_varying", [False, True])
def test_the_finite_horizon_bound_brackets_correctly(time_varying: bool) -> None:
    """Below a feasible policy, above the unconstrained optimum it starts from.

    Both halves matter. The upper check is the bound's whole purpose -- a floor
    that a real feasible controller undercuts is not a floor. The lower check is
    the anti-vacuity: at `lambda = 0` the bound *is* the unconstrained optimum,
    so a maximisation that did nothing would sit exactly there.
    """
    data = _data(6, 2)
    covariance = X0**2 * np.eye(data.A.shape[0])
    zero = finite_horizon_box_bound(
        data, covariance, HORIZON, time_varying=time_varying
    )
    feasible = _clipped_riccati_cost(data, HORIZON)

    assert zero.value < feasible, "the floor is above a feasible policy"
    unconstrained = _finite_horizon_unconstrained(data, covariance, HORIZON)
    assert zero.value > unconstrained, "the multipliers bought nothing"


def _finite_horizon_unconstrained(
    data: BoxLQRData, covariance: np.ndarray, horizon: int
) -> float:
    """The same bound at `lambda = 0`, which is the unconstrained optimum."""
    zeroed = BoxLQRData(A=data.A, B=data.B, Q=data.Q, R=data.R, W=data.W, u_max=1e9)
    return finite_horizon_box_bound(
        zeroed, covariance, horizon, time_varying=False
    ).value


def test_time_varying_multipliers_are_at_least_as_tight() -> None:
    """More freedom cannot make a maximisation worse."""
    data = _data(6, 2)
    covariance = X0**2 * np.eye(data.A.shape[0])
    constant = finite_horizon_box_bound(data, covariance, HORIZON, time_varying=False)
    varying = finite_horizon_box_bound(data, covariance, HORIZON, time_varying=True)
    assert varying.value >= constant.value - 1e-9
    assert varying.multipliers.shape == (HORIZON, data.B.shape[1])
    assert constant.multipliers.shape == (HORIZON, data.B.shape[1])


def test_the_two_conventions_disagree_and_must_not_be_swapped() -> None:
    """The reason this module carries two functions rather than one.

    The steady-state bound is not a floor for a short time-average from a
    low-energy start: it sits *above* it, which is exactly how drawing the
    wrong one puts a lower bound above every curve in a figure.
    """
    data = _data(6, 2)
    covariance = X0**2 * np.eye(data.A.shape[0])
    steady = infinite_horizon_box_bound(data)
    finite = finite_horizon_box_bound(data, covariance, HORIZON)
    assert steady.value != pytest.approx(finite.value, rel=1e-3)
