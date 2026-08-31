from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are, sqrtm

from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.engine.strategy import GradientDescentStrategy
from mbl.models.base import Controller
from mbl.models.constrained.cocp import COCPConfig, COCPController
from mbl.models.constrained.solver_spec import COCPSolverSpec

DTYPE = torch.float64


def _build_problem(state_dim: int = 2, control_dim: int = 1):
    A = np.array([[0.9, 0.0], [0.0, 0.85]])[:state_dim, :state_dim]
    B = np.ones((state_dim, control_dim)) * 0.5
    system = LinearSystem.fully_observable(A, B)
    cost = QuadraticCost(Q=np.eye(state_dim), R=np.eye(control_dim))
    return OptimalControlProblem(system=system, cost=cost), A, B


def _build_cocp(u_max: float = 1.0, state_dim: int = 2, control_dim: int = 1):
    problem, A, B = _build_problem(state_dim, control_dim)
    P_are = solve_discrete_are(A, B, problem.cost.Q, problem.cost.R)
    P_sqrt_init = np.real(sqrtm(P_are))
    constraint = BoxConstraint(u_max=u_max)
    config = COCPConfig(P_sqrt_init=P_sqrt_init, dtype=DTYPE)
    controller = COCPController(problem, constraint, config)
    return controller, problem


def test_cocp_controller_satisfies_controller_protocol() -> None:
    controller, _ = _build_cocp()
    assert isinstance(controller, Controller)


def test_cocp_control_policy_respects_the_box_constraint() -> None:
    u_max = 0.2  # tight, so the QP's constraint is very likely to bind
    controller, _ = _build_cocp(u_max=u_max)
    policy = controller.get_control_policy()

    batch = 4
    y = torch.full(
        (batch, 2), 10.0, dtype=DTYPE
    )  # large state -> large desired control
    u = policy(0, y)

    assert u.shape == (batch, 1)
    assert torch.all(u.abs() <= u_max + 1e-4)


def test_cocp_control_policy_matches_unconstrained_lqr_for_small_states() -> None:
    """With a generous bound and a small state, the QP shouldn't saturate --
    the resulting control should closely match the unconstrained LQR gain."""
    state_dim, control_dim = 2, 1
    controller, problem = _build_cocp(
        u_max=1000.0, state_dim=state_dim, control_dim=control_dim
    )
    A = problem.system.A_t.array
    B = problem.system.B_t.array
    P_are = solve_discrete_are(A, B, problem.cost.Q, problem.cost.R)
    K_lqr = -np.linalg.solve(problem.cost.R + B.T @ P_are @ B, B.T @ P_are @ A)

    policy = controller.get_control_policy()
    x = torch.tensor([[0.01, -0.01]], dtype=DTYPE)
    u = policy(0, x)

    expected = x.numpy() @ K_lqr.T
    assert np.allclose(u.detach().numpy(), expected, atol=1e-3)


def test_cocp_rollout_gradients_flow_back_through_the_cvxpylayer() -> None:
    """The core requirement: differentiating a rollout cost through the QP's
    KKT conditions must produce real, finite, non-zero gradients on P_sqrt
    and q -- proving the cvxpylayers graph survives an actual multi-step
    System.run() rollout, not just a single solve."""
    controller, _ = _build_cocp(u_max=0.5)
    rollout = RolloutModel(controller)

    batch, horizon, state_dim = 3, 3, 2
    initial_state = torch.ones(batch, state_dim, dtype=DTYPE)
    process_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)
    measurement_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)

    _, _, _, cost = rollout(initial_state, process_noise, measurement_noise)
    # One cost per trajectory since Annex 06 §4.3's correction; backward needs
    # a scalar, and which scalar is the caller's decision.
    cost.mean().backward()

    assert controller.P_sqrt.grad is not None
    assert torch.isfinite(controller.P_sqrt.grad).all()
    assert (controller.P_sqrt.grad != 0).any()

    assert controller.q.grad is not None
    assert torch.isfinite(controller.q.grad).all()


def test_cocp_trains_end_to_end_through_gradient_descent_strategy() -> None:
    """Runs COCP through the actual engine machinery (GradientDescentStrategy,
    not a hand-rolled backward() call), proving P_sqrt/q genuinely update
    during the engine's training loop."""
    controller, _ = _build_cocp(u_max=0.5)
    rollout = RolloutModel(controller)
    optimizer = torch.optim.Adam(rollout.parameters(), lr=0.05)
    strategy = GradientDescentStrategy(rollout, optimizer)

    batch, horizon, state_dim = 3, 3, 2
    batch_data = (
        torch.ones(batch, state_dim, dtype=DTYPE),
        torch.zeros(batch, horizon, state_dim, dtype=DTYPE),
        torch.zeros(batch, horizon, state_dim, dtype=DTYPE),
    )

    P_sqrt_before = controller.P_sqrt.detach().clone()
    q_before = controller.q.detach().clone()

    metrics = strategy.step(MagicMock(), batch_data)

    assert np.isfinite(metrics["loss"])
    assert not torch.equal(controller.P_sqrt.detach(), P_sqrt_before)
    assert not torch.equal(controller.q.detach(), q_before)


def test_frozen_cocp_lower_bound_parameters_never_change() -> None:
    """The "COCP lower bound" baseline: COCPController seeded from the
    box-constrained SDP's P_lb and never trained -- its parameters must stay
    bit-for-bit identical through a training-shaped call."""
    controller, _ = _build_cocp(u_max=0.5)
    controller.P_sqrt.requires_grad_(False)
    controller.q.requires_grad_(False)

    rollout = RolloutModel(controller)
    batch, horizon, state_dim = 2, 2, 2
    initial_state = torch.ones(batch, state_dim, dtype=DTYPE)
    process_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)
    measurement_noise = torch.zeros(batch, horizon, state_dim, dtype=DTYPE)

    P_sqrt_before = controller.P_sqrt.detach().clone()
    q_before = controller.q.detach().clone()

    with torch.no_grad():
        rollout(initial_state, process_noise, measurement_noise)

    assert torch.equal(controller.P_sqrt.detach(), P_sqrt_before)
    assert torch.equal(controller.q.detach(), q_before)


def test_cocp_get_signature_reports_type_hashes_dtype_and_constraint() -> None:
    """`problem` is intentionally absent from this controller's own signature
    (Phase 1E): `ProblemSignatureCallback` already logs it once at the root."""
    controller, _ = _build_cocp(u_max=1.0)

    signature = controller.get_signature()

    assert signature["type"] == "COCPController"
    assert signature["P_sqrt_init_hash"].startswith("sha256:")
    assert signature["q_init_hash"] is None  # q_init defaults to zeros(n), not given
    assert signature["dtype"] == DTYPE
    assert signature["solver"] == controller.config.solver.get_signature()
    assert signature["constraint"] == controller.constraint.get_signature()
    assert "problem" not in signature


def test_cocp_get_signature_is_unaffected_by_training() -> None:
    """Signs config.P_sqrt_init (the seed), never the live, possibly-trained
    self.P_sqrt -- the signature must not change after a gradient step."""
    controller, _ = _build_cocp(u_max=0.5)
    before = controller.get_signature()

    with torch.no_grad():
        controller.P_sqrt.add_(1.0)

    assert controller.get_signature() == before


def test_cocp_parameters_are_placed_on_the_configured_solver_device() -> None:
    controller, _ = _build_cocp(u_max=0.5)
    assert controller.P_sqrt.device == torch.device(controller.config.solver.device)
    assert controller.q.device == torch.device(controller.config.solver.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cocp_control_policy_accepts_a_state_on_a_different_device_than_the_solver() -> (
    None
):
    """Regression: `config.solver.device` (MOREAU-cuda is excluded from
    auto-resolution, so this is CPU in practice) need not match the
    experiment's own `ComputeContext.device` -- when the engine rollout runs
    on CUDA (this project's real auto-selected default whenever a GPU is
    available) while COCP's resolved solver stays CPU-only, the policy must
    still work: `x`/`P_sqrt`/`q` are cast to the SAME device for the QP
    solve, and the result is cast back to the caller's own device/dtype so
    it composes with the rest of the (CUDA-resident) rollout. Confirms the
    real crash this project hit (`RuntimeError: Expected all tensors to be
    on the same device`) is fixed, without changing batch size, horizon, or
    which states are solved -- only where the solve's tensors briefly live."""
    controller, _ = _build_cocp(u_max=0.5)  # solver.device == "cpu" (the default)
    assert controller.P_sqrt.device == torch.device("cpu")
    policy = controller.get_control_policy()

    batch_size, state_dim = 4, 2
    y = torch.randn(batch_size, state_dim, dtype=DTYPE, device="cuda")

    u = policy(0, y)

    assert u.device == y.device
    assert u.dtype == y.dtype
    assert torch.isfinite(u).all()


def test_build_qp_layer_default_solver_matches_explicit_diffcp_spec() -> None:
    """No `solver=` argument must reproduce this method's pre-existing
    (pre-COCPSolverSpec) behavior exactly -- both resolve to "DIFFCP"."""
    problem, A, B = _build_problem()
    R = problem.cost.R
    default_layer = COCPController.build_qp_layer(A, B, R, 0.5)
    explicit_layer = COCPController.build_qp_layer(
        A, B, R, 0.5, COCPSolverSpec(name="DIFFCP")
    )

    x = torch.tensor([[10.0], [10.0]], dtype=DTYPE)
    P_sqrt = torch.tensor(np.eye(2), dtype=DTYPE)
    q = torch.zeros(2, dtype=DTYPE)
    (u_default,) = default_layer(x, P_sqrt, q)
    (u_explicit,) = explicit_layer(x, P_sqrt, q)
    assert torch.allclose(u_default, u_explicit)
