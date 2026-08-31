import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from mbl.models.constrained.lower_bound import solve_box_constrained_lower_bound


def _stable_system():
    A = np.array([[0.9, 0.0], [0.0, 0.85]])
    B = np.array([[1.0], [0.5]])
    Q = np.eye(2)
    R = np.eye(1)
    W = 0.25 * np.eye(2)
    return A, B, Q, R, W


def test_lower_bound_returns_finite_value_and_psd_matrix() -> None:
    A, B, Q, R, W = _stable_system()

    lower_bound, P_lb = solve_box_constrained_lower_bound(A, B, Q, R, W, u_max=1.0)

    assert np.isfinite(lower_bound)
    symmetric_P = (P_lb + P_lb.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_P)
    assert np.all(eigvals >= -1e-6)


def test_lower_bound_is_non_decreasing_as_u_max_shrinks() -> None:
    """The SDP maximizes a pointwise max of functions affine (with a
    non-positive slope) in u_max**2 over a u_max-independent feasible set, so
    a smaller control budget can only raise (never lower) the bound -- a
    tighter constraint makes the problem harder, never easier."""
    A, B, Q, R, W = _stable_system()

    loose_bound, _ = solve_box_constrained_lower_bound(A, B, Q, R, W, u_max=5.0)
    tight_bound, _ = solve_box_constrained_lower_bound(A, B, Q, R, W, u_max=0.5)

    assert tight_bound >= loose_bound - 1e-6


def test_lower_bound_approaches_the_unconstrained_riccati_cost_for_large_u_max() -> (
    None
):
    """As u_max -> infinity the box constraint stops binding, so the SDP's
    optimal value should converge to trace(P_are @ W), the standard
    unconstrained-LQR average cost."""
    A, B, Q, R, W = _stable_system()
    P_are = solve_discrete_are(A, B, Q, R)
    unconstrained_cost = np.trace(P_are @ W)

    lower_bound, _ = solve_box_constrained_lower_bound(A, B, Q, R, W, u_max=100.0)

    assert lower_bound == pytest.approx(unconstrained_cost, rel=1e-2)


def test_lower_bound_raises_a_clear_error_when_infeasible() -> None:
    """An unstabilizable system (B all-zero) can't satisfy the Riccati LMI at
    any finite P -- the SDP should fail, and that failure should surface as a
    clear RuntimeError rather than silently returning P_lb=None."""
    A = np.array([[1.5, 0.0], [0.0, 1.2]])
    B = np.zeros((2, 1))
    Q = np.eye(2)
    R = np.eye(1)
    W = np.eye(2)

    with pytest.raises(RuntimeError, match="failed to solve"):
        solve_box_constrained_lower_bound(A, B, Q, R, W, u_max=1.0)
