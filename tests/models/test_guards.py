import warnings

import numpy as np
import pytest

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.riccati import finite_horizon_riccati
from mbl.models.guards import (
    check_controllability,
    ensure_lqr_dimensional_coherence,
    ensure_riccati_horizon_coherence,
    ensure_stabilizable_for_dare,
)


def test_lqr_dimensional_coherence_accepts_consistent_data() -> None:
    n, m = 2, 1
    ensure_lqr_dimensional_coherence(
        A=np.eye(n),
        B=np.zeros((n, m)),
        Q=np.zeros((3, n, n)),
        R=np.zeros((3, m, m)),
    )


def test_lqr_dimensional_coherence_rejects_mismatched_B() -> None:
    with pytest.raises(ValueError, match="B must have shape"):
        ensure_lqr_dimensional_coherence(
            A=np.eye(2), B=np.zeros((3, 1)), Q=np.eye(2), R=np.eye(1)
        )


def test_lqr_dimensional_coherence_rejects_mismatched_R() -> None:
    with pytest.raises(ValueError, match="R must have trailing shape"):
        ensure_lqr_dimensional_coherence(
            A=np.eye(2), B=np.zeros((2, 1)), Q=np.eye(2), R=np.eye(2)
        )


def test_riccati_horizon_coherence_rejects_time_invariant_q() -> None:
    with pytest.raises(ValueError, match="time-stacked Q"):
        ensure_riccati_horizon_coherence(horizon=3, Q=np.eye(2), R=np.zeros((3, 1, 1)))


def test_riccati_horizon_coherence_rejects_too_short_stacks() -> None:
    with pytest.raises(ValueError, match="time-stacked R"):
        ensure_riccati_horizon_coherence(
            horizon=5, Q=np.zeros((6, 2, 2)), R=np.zeros((3, 1, 1))
        )


def _uncontrollable_pair() -> tuple[np.ndarray, np.ndarray]:
    # ctrb([B, AB]) is rank-deficient: A scales both coordinates equally, so
    # B and AB are colinear.
    A = 0.9 * np.eye(2)
    B = np.array([[1.0], [0.5]])
    return A, B


def test_check_controllability_silent_for_controllable_pair() -> None:
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.array([[1.0], [0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_controllability(A, B)  # must not warn or raise


def test_check_controllability_warns_for_uncontrollable_pair() -> None:
    A, B = _uncontrollable_pair()
    with pytest.warns(UserWarning, match="not controllable"):
        check_controllability(A, B)


def test_check_controllability_can_escalate_to_raise() -> None:
    A, B = _uncontrollable_pair()
    with pytest.raises(ValueError, match="not controllable"):
        check_controllability(A, B, raise_on_fail=True)


def test_finite_horizon_riccati_solves_uncontrollable_problem() -> None:
    """Regression lock (plan §0.1): an uncontrollable but *finite-horizon* LQR
    still has a well-defined solution -- the solver must NOT gate on
    controllability/stabilizability."""
    A, B = _uncontrollable_pair()
    horizon, n, m = 4, 2, 1
    system = LinearSystem.fully_observable(
        np.repeat(A[None], horizon, axis=0), np.repeat(B[None], horizon, axis=0)
    )
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R)

    P_arr, K_arr = finite_horizon_riccati(system, cost, horizon)
    assert P_arr.shape == (horizon + 1, n, n)
    assert K_arr.shape == (horizon, m, n)
    assert np.all(np.isfinite(P_arr))


def test_stabilizable_for_dare_hard_gates_unstable_uncontrollable_mode() -> None:
    A = np.array([[2.0]])  # unstable, |lambda| = 2 >= 1
    B = np.array([[0.0]])  # uncontrollable
    with pytest.raises(ValueError, match="not stabilizable"):
        ensure_stabilizable_for_dare(A, B)


def test_stabilizable_for_dare_accepts_controllable_pair() -> None:
    A = np.array([[2.0]])
    B = np.array([[1.0]])  # controllable -> stabilizable
    ensure_stabilizable_for_dare(A, B)
