import numpy as np
import pytest

from mbl.core.cost.quadratic_cost import QuadraticCost


def test_quadratic_cost_basic_no_terminal() -> None:
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(1))

    x = np.zeros((2, 4, 2))
    u = np.ones((2, 3, 1))

    out = cost(x, u)
    assert out.shape == (3,)
    assert np.allclose(out, np.ones(3))


def test_quadratic_cost_is_time_averaged_false_returns_raw_cumulative_cost() -> None:
    """Same setup as test_quadratic_cost_basic_no_terminal, but with
    is_time_averaged=False: the per-step division by (k+1) must not happen,
    leaving the raw cumulative sum [1, 2, 3] instead of [1, 1, 1]."""
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(1), is_time_averaged=False)

    x = np.zeros((2, 4, 2))
    u = np.ones((2, 3, 1))

    out = cost(x, u)
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_quadratic_cost_is_time_averaged_false_with_terminal_cost() -> None:
    """Same setup as test_quadratic_cost_terminal_cost_affects_final_prefix_average,
    but raw (is_time_averaged=False): [0.0, 2.0] * [1, 2] == [0.0, 2.0]
    (terminal cost always reuses Q -- no separate Qf)."""
    cost = QuadraticCost(
        Q=np.eye(2),
        R=np.eye(1),
        include_terminal_cost=True,
        is_time_averaged=False,
    )

    x = np.zeros((1, 3, 2))
    x[:, -1, :] = 1.0
    u = np.zeros((1, 2, 1))

    out = cost(x, u)
    assert out.shape == (2,)
    assert np.allclose(out, [0.0, 2.0])


@pytest.mark.parametrize("include_terminal_cost", [True, False])
def test_quadratic_cost_is_time_averaged_true_equals_raw_divided_by_elapsed_steps(
    include_terminal_cost: bool,
) -> None:
    """The defining relationship between the two conventions (Phase 1F):
    the time-averaged curve is exactly the raw cumulative curve divided by
    the elapsed step count (k+1), regardless of the terminal-cost convention."""
    rng = np.random.default_rng(0)
    n, m, batch, horizon = 3, 2, 5, 6
    x = rng.standard_normal((batch, horizon + 1, n))
    u = rng.standard_normal((batch, horizon, m))
    Q, R = np.eye(n), np.eye(m)

    averaged = QuadraticCost(Q=Q, R=R, include_terminal_cost=include_terminal_cost)(
        x, u
    )
    raw = QuadraticCost(
        Q=Q,
        R=R,
        include_terminal_cost=include_terminal_cost,
        is_time_averaged=False,
    )(x, u)

    k_steps = np.arange(1, horizon + 1)
    assert np.allclose(averaged, raw / k_steps)


def test_quadratic_cost_terminal_cost_affects_final_prefix_average() -> None:
    """The terminal cost term always reuses Q (its own final slice) -- there
    is no separate terminal cost matrix."""
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(1), include_terminal_cost=True)

    x = np.zeros((1, 3, 2))
    x[:, -1, :] = 1.0
    u = np.zeros((1, 2, 1))

    out = cost(x, u)
    assert out.shape == (2,)
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], 1.0)


def test_quadratic_cost_rejects_horizon_mismatch() -> None:
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(1))
    x = np.zeros((1, 5, 2))
    u = np.zeros((1, 3, 1))

    with pytest.raises(ValueError, match="horizon"):
        cost(x, u)


def test_quadratic_cost_rejects_non_square_matrices() -> None:
    with pytest.raises(ValueError, match="square"):
        QuadraticCost(Q=np.zeros((2, 3)), R=np.eye(1))

    with pytest.raises(ValueError, match="square"):
        QuadraticCost(Q=np.eye(2), R=np.zeros((1, 2)))


def test_quadratic_cost_get_signature_reports_type_flags_and_hashes() -> None:
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(1), include_terminal_cost=True)
    signature = cost.get_signature()

    assert signature == {
        "type": "QuadraticCost",
        "include_terminal_cost": True,
        "is_time_averaged": True,
        "Q_hash": signature["Q_hash"],
        "R_hash": signature["R_hash"],
    }
    assert signature["Q_hash"].startswith("sha256:")
    assert signature["R_hash"].startswith("sha256:")


def test_quadratic_cost_get_signature_omits_state_dim_and_control_dim() -> None:
    """Dimensions are the system's concern (LinearSystem.get_signature()
    already reports them); QuadraticCost signs neither, even though it
    exposes both as plain attributes (see the sibling test below)."""
    signature = QuadraticCost(Q=np.eye(2), R=np.eye(1)).get_signature()
    assert "state_dim" not in signature
    assert "control_dim" not in signature


def test_quadratic_cost_exposes_state_dim_and_control_dim_eagerly() -> None:
    """state_dim/control_dim must be set at construction time (not deferred
    to the first __call__), since ensure_system_cost_dims_match reads them
    at OptimalControlProblem construction time, before any rollout runs."""
    cost = QuadraticCost(Q=np.eye(3), R=np.eye(2))
    assert cost.state_dim == 3
    assert cost.control_dim == 2


def test_quadratic_cost_get_signature_reports_is_time_averaged_false() -> None:
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(1), is_time_averaged=False)
    assert cost.get_signature()["is_time_averaged"] is False


def test_quadratic_cost_get_signature_hash_changes_with_matrix_content() -> None:
    sig1 = QuadraticCost(Q=np.eye(2), R=np.eye(1)).get_signature()
    sig2 = QuadraticCost(Q=np.eye(2) * 3.0, R=np.eye(1)).get_signature()
    assert sig1["Q_hash"] != sig2["Q_hash"]
    assert sig1["R_hash"] == sig2["R_hash"]
