import numpy as np
import pytest

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.riccati import (
    RiccatiController,
    compute_state_covariance_trajectory,
    compute_theoretical_expected_cost,
    LocalCostToGoModel,
    evaluate_local_cost_to_go,
    finite_horizon_riccati,
    get_lqr_gradient_matrices,
    get_riccati_control_policy,
    get_riccati_control_trajectory,
)


def _scalar_riccati_reference(
    A: float, B: float, Q: float, R: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent, plain-Python reference implementation of the scalar Riccati recursion,
    used to cross-check finite_horizon_riccati without reusing its own machinery."""
    P = [0.0] * (horizon + 1)
    K = [0.0] * horizon
    P[horizon] = Q
    for k in reversed(range(horizon)):
        M = R + B * B * P[k + 1]
        C = B * A * P[k + 1]
        K[k] = C / M
        P[k] = Q + A * P[k + 1] * (A - B * K[k])
    return np.array(P).reshape(horizon + 1, 1, 1), np.array(K).reshape(horizon, 1, 1)


def _build_scalar_problem(
    A: float, B: float, Q: float, R: float, horizon: int
) -> OptimalControlProblem:
    system = LinearSystem.fully_observable(np.array([[A]]), np.array([[B]]))
    Q_stack = np.repeat(np.array([[[Q]]]), horizon + 1, axis=0)
    R_stack = np.repeat(np.array([[[R]]]), horizon, axis=0)
    cost = QuadraticCost(Q=Q_stack, R=R_stack, include_terminal_cost=True)
    return OptimalControlProblem(system=system, cost=cost)


def test_finite_horizon_riccati_matches_independent_scalar_reference() -> None:
    A, B, Q, R, horizon = 1.2, 0.8, 2.0, 0.5, 5
    problem = _build_scalar_problem(A, B, Q, R, horizon)

    P_arr, K_arr = finite_horizon_riccati(problem.system, problem.cost, horizon)
    expected_P, expected_K = _scalar_riccati_reference(A, B, Q, R, horizon)

    assert np.allclose(P_arr, expected_P)
    assert np.allclose(K_arr, expected_K)


def test_get_lqr_gradient_matrices_solve_recovers_riccati_gains() -> None:
    """K_k = M_k^-1 C_k, where M, C come from get_lqr_gradient_matrices, must equal the
    feedback gains already produced by the Riccati recursion (they are the same formula)."""
    A, B, Q, R, horizon = 0.9, 1.1, 1.0, 1.0, 4
    problem = _build_scalar_problem(A, B, Q, R, horizon)

    P_arr, K_arr = finite_horizon_riccati(problem.system, problem.cost, horizon)
    A_arr = np.repeat(np.array([[[A]]]), horizon, axis=0)
    B_arr = np.repeat(np.array([[[B]]]), horizon, axis=0)
    R_arr = problem.cost.R

    M_arr, C_arr = get_lqr_gradient_matrices(P_arr, A_arr, B_arr, R_arr)
    solved_K = np.linalg.solve(M_arr, C_arr)

    assert np.allclose(solved_K, K_arr)


def test_riccati_controller_get_control_policy_matches_manual_gain_application() -> (
    None
):
    A, B, Q, R, horizon = 1.0, 1.0, 1.0, 1.0, 3
    problem = _build_scalar_problem(A, B, Q, R, horizon)

    controller = RiccatiController(problem, horizon)
    assert controller.problem is problem
    assert controller.config is None

    policy = controller.get_control_policy()
    x = np.array([[2.0]])
    u = policy(0, x)
    assert np.allclose(u, -controller.K_arr[0] @ x[0])


def test_get_riccati_control_policy_and_trajectory_are_consistent() -> None:
    horizon, n, m = 3, 1, 1
    K_arr = np.ones((horizon, m, n)) * 0.5
    X = np.array([[[1.0], [0.5], [0.25], [0.125]]])  # (batch=1, horizon+1, n)

    policy = get_riccati_control_policy(K_arr)
    manual_controls = np.stack([policy(t, X[:, t]) for t in range(horizon)], axis=1)

    trajectory_controls = get_riccati_control_trajectory(X, K_arr)
    assert np.allclose(manual_controls, trajectory_controls)


def test_finite_horizon_riccati_rejects_non_square_q() -> None:
    with pytest.raises(ValueError, match="square"):
        QuadraticCost(Q=np.zeros((2, 3)), R=np.eye(1))


def test_compute_state_covariance_trajectory_matches_manual_matrix_powers_with_zero_noise() -> (
    None
):
    """With zero process noise, Sigma_t is exactly M^t @ Sigma_0 @ (M^t)^T for
    the (here, time-invariant) closed-loop matrix M = A - BK -- an
    independent, hand-derivable reference distinct from the recursive
    implementation itself."""
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.array([[1.0], [0.5]])
    K = np.array([[0.2, 0.1]])
    horizon = 4
    system = LinearSystem.fully_observable(
        np.repeat(A[None], horizon, axis=0), np.repeat(B[None], horizon, axis=0)
    )
    K_arr = np.repeat(K[None], horizon, axis=0)
    cov_0 = np.diag([2.0, 3.0])
    zero_noise = np.zeros((2, 2))

    Sigma = compute_state_covariance_trajectory(system, K_arr, zero_noise, cov_0)

    M = A - B @ K
    expected = cov_0
    for t in range(horizon + 1):
        assert np.allclose(Sigma[t], expected)
        expected = M @ expected @ M.T


@pytest.mark.parametrize("include_terminal_cost", [False, True])
def test_compute_theoretical_expected_cost_matches_large_batch_monte_carlo(
    include_terminal_cost,
) -> None:
    """The defining property under test: the theoretical (closed-form) cost
    must match QuadraticCost's own empirical Monte-Carlo cost (evaluated on
    a real, large-batch system rollout) to within statistical tolerance --
    exactly the sanity check the research notebook built on this function
    performs. Parametrized over both of QuadraticCost's terminal-cost
    variants, since the theoretical formula must mirror each exactly."""
    rng = np.random.default_rng(0)
    n, m, horizon = 2, 1, 6
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.array([[1.0], [0.5]])
    system = LinearSystem.fully_observable(
        np.repeat(A[None], horizon, axis=0), np.repeat(B[None], horizon, axis=0)
    )
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=include_terminal_cost)
    _, K_arr = finite_horizon_riccati(system, cost, horizon)

    process_noise_std = 0.5
    process_noise_cov = process_noise_std**2 * np.eye(n)
    initial_state_cov = np.eye(n)

    theoretical = compute_theoretical_expected_cost(
        system, cost, K_arr, process_noise_cov, initial_state_cov
    )

    batch = 20_000
    x0 = rng.normal(size=(batch, n))
    w = rng.normal(scale=process_noise_std, size=(batch, horizon, n))
    v = np.zeros((batch, horizon, n))
    policy = get_riccati_control_policy(K_arr)
    X, _, U = system.run(policy, x0, w, v)
    empirical = cost(X, U)

    assert theoretical.shape == (horizon,)
    assert np.allclose(theoretical, empirical, rtol=0.05, atol=0.02)


@pytest.mark.parametrize("include_terminal_cost", [False, True])
def test_compute_theoretical_expected_cost_mirrors_is_time_averaged_false(
    include_terminal_cost,
) -> None:
    """Same defining property as the is_time_averaged=True (default) case
    above, now for the raw-cumulative-cost variant (Phase 1F): the
    theoretical curve must still match a large-batch Monte-Carlo empirical
    curve computed with is_time_averaged=False on the same cost."""
    rng = np.random.default_rng(0)
    n, m, horizon = 2, 1, 6
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.array([[1.0], [0.5]])
    system = LinearSystem.fully_observable(
        np.repeat(A[None], horizon, axis=0), np.repeat(B[None], horizon, axis=0)
    )
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(
        Q=Q, R=R, include_terminal_cost=include_terminal_cost, is_time_averaged=False
    )
    _, K_arr = finite_horizon_riccati(system, cost, horizon)

    process_noise_std = 0.5
    process_noise_cov = process_noise_std**2 * np.eye(n)
    initial_state_cov = np.eye(n)

    theoretical = compute_theoretical_expected_cost(
        system, cost, K_arr, process_noise_cov, initial_state_cov
    )

    batch = 20_000
    x0 = rng.normal(size=(batch, n))
    w = rng.normal(scale=process_noise_std, size=(batch, horizon, n))
    v = np.zeros((batch, horizon, n))
    policy = get_riccati_control_policy(K_arr)
    X, _, U = system.run(policy, x0, w, v)
    empirical = cost(X, U)

    assert theoretical.shape == (horizon,)
    assert np.allclose(theoretical, empirical, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("include_terminal_cost", [False, True])
def test_compute_theoretical_expected_cost_time_averaged_equals_raw_divided_by_elapsed_steps(
    include_terminal_cost,
) -> None:
    """The same algebraic identity QuadraticCost.__call__ itself satisfies
    (Phase 1F): the time-averaged theoretical curve is exactly the raw
    theoretical curve divided by the elapsed step count (k+1)."""
    n, m, horizon = 2, 1, 5
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.array([[1.0], [0.5]])
    system = LinearSystem.fully_observable(
        np.repeat(A[None], horizon, axis=0), np.repeat(B[None], horizon, axis=0)
    )
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    averaged_cost = QuadraticCost(Q=Q, R=R, include_terminal_cost=include_terminal_cost)
    raw_cost = QuadraticCost(
        Q=Q,
        R=R,
        include_terminal_cost=include_terminal_cost,
        is_time_averaged=False,
    )
    _, K_arr = finite_horizon_riccati(system, averaged_cost, horizon)

    process_noise_cov = 0.25 * np.eye(n)
    initial_state_cov = np.eye(n)

    averaged = compute_theoretical_expected_cost(
        system, averaged_cost, K_arr, process_noise_cov, initial_state_cov
    )
    raw = compute_theoretical_expected_cost(
        system, raw_cost, K_arr, process_noise_cov, initial_state_cov
    )

    assert np.allclose(averaged, raw / np.arange(1, horizon + 1))


def test_evaluate_local_cost_to_go_minimizer_matches_riccati_gain() -> None:
    """The core landscape-alignment acceptance test: a fine brute-force grid
    search over candidate controls u, scored by `evaluate_local_cost_to_go`
    at a FROZEN state x, must land its argmin exactly at u* = -K_t x -- the
    mathematical guarantee that a landscape built from this closed-form
    function can never exhibit a numerically "displaced" optimum the way a
    rollout-based surrogate cost (whose downstream continuation is held
    open-loop rather than re-optimized) can."""
    n, m, horizon = 2, 2, 5
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n)) * 0.3
    B = rng.normal(size=(n, m))
    system = LinearSystem.fully_observable(A, B)
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R)
    P_arr, K_arr = finite_horizon_riccati(system, cost, horizon)

    t = 2
    x = np.array([1.3, -0.7])
    process_noise_cov = 0.1 * np.eye(n)
    u_star = -K_arr[t] @ x

    grid = np.linspace(-3.0, 3.0, 121)
    U1, U2 = np.meshgrid(grid, grid)
    U_batch = np.stack([U1.ravel(), U2.ravel()], axis=-1)

    V = evaluate_local_cost_to_go(
        x,
        U_batch,
        LocalCostToGoModel(
            A=A,
            B=B,
            Q=Q[t],
            R=R[t],
            P_next=P_arr[t + 1],
            process_noise_cov=process_noise_cov,
        ),
    )
    argmin = np.argmin(V)
    grid_spacing = 6.0 / 120

    assert abs(U_batch[argmin, 0] - u_star[0]) <= grid_spacing
    assert abs(U_batch[argmin, 1] - u_star[1]) <= grid_spacing


def test_evaluate_local_cost_to_go_matches_manual_expansion_with_zero_noise() -> None:
    """Hand-expanded reference (zero process noise, so the closed-form value
    is an ordinary deterministic quadratic form) for one specific (x, u)
    pair, independent of `compute_lqr_gradient_matrices`."""
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.eye(2)
    R = 0.5 * np.eye(2)
    P_next = np.array([[2.0, 0.3], [0.3, 1.5]])
    x = np.array([1.0, -2.0])
    u = np.array([[0.5, 0.25]])

    expected_next_state = A @ x + B @ u[0]
    expected = (
        x @ Q @ x + u[0] @ R @ u[0] + expected_next_state @ P_next @ expected_next_state
    )

    result = evaluate_local_cost_to_go(
        x,
        u,
        LocalCostToGoModel(
            A=A, B=B, Q=Q, R=R, P_next=P_next, process_noise_cov=np.zeros((2, 2))
        ),
    )

    assert result == pytest.approx([expected])


def test_evaluate_local_cost_to_go_process_noise_adds_constant_offset() -> None:
    """Process noise contributes only tr(P_next Sigma_w) -- an additive
    constant independent of u -- so two evaluations differing only in
    `process_noise_cov` must differ by exactly that trace, for every u."""
    A = np.array([[0.9, 0.1], [0.0, 0.85]])
    B = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    P_next = np.array([[2.0, 0.3], [0.3, 1.5]])
    x = np.array([0.5, 1.0])
    U = np.array([[0.1, -0.2], [1.0, 2.0], [-3.0, 0.5]])
    Sigma_w = np.array([[0.3, 0.0], [0.0, 0.2]])

    zero_noise = evaluate_local_cost_to_go(
        x,
        U,
        LocalCostToGoModel(
            A=A, B=B, Q=Q, R=R, P_next=P_next, process_noise_cov=np.zeros((2, 2))
        ),
    )
    with_noise = evaluate_local_cost_to_go(
        x,
        U,
        LocalCostToGoModel(
            A=A, B=B, Q=Q, R=R, P_next=P_next, process_noise_cov=Sigma_w
        ),
    )

    assert np.allclose(with_noise - zero_noise, np.trace(P_next @ Sigma_w))


def test_riccati_controller_get_signature_reports_type_and_horizon_only() -> None:
    A, B, Q, R, horizon = 1.0, 1.0, 1.0, 1.0, 3
    problem = _build_scalar_problem(A, B, Q, R, horizon)
    controller = RiccatiController(problem, horizon)

    signature = controller.get_signature()

    assert signature == {"type": "RiccatiController", "horizon": horizon}


def test_riccati_controller_get_signature_is_identical_across_different_problems() -> (
    None
):
    """`ProblemSignatureCallback` logs the problem's signature once at the
    root (``problem.*``); the controller's own signature must not re-embed
    it (Phase 1E: this used to duplicate every key under
    ``controller.problem.*``), so two controllers over different problems
    but the same horizon must sign identically."""
    horizon = 3
    problem_1 = _build_scalar_problem(1.0, 1.0, 1.0, 1.0, horizon)
    problem_2 = _build_scalar_problem(1.0, 1.0, 2.0, 1.0, horizon)  # different Q

    sig_1 = RiccatiController(problem_1, horizon).get_signature()
    sig_2 = RiccatiController(problem_2, horizon).get_signature()

    assert sig_1 == sig_2
    assert "problem" not in sig_1
