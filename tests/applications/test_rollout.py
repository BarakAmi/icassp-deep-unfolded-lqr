import numpy as np
import pytest
import torch

from mbl.applications.rollout import RolloutModel
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.riccati import RiccatiController


def _build_rollout(horizon: int = 4) -> RolloutModel:
    n, m = 2, 1
    system = LinearSystem.fully_observable(
        np.repeat(np.eye(n)[None], horizon, axis=0),
        np.repeat(np.array([[1.0], [0.5]])[None], horizon, axis=0),
    )
    Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
    R = np.repeat(2.0 * np.eye(m)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R)
    problem = OptimalControlProblem(system=system, cost=cost)
    # (A=I, B) is deliberately minimal and hence uncontrollable; the
    # EXPENSIVE-tier guard is expected to flag it, so assert the warning
    # rather than letting it leak into the suite's output.
    with pytest.warns(UserWarning, match="not controllable"):
        controller = RiccatiController(problem, horizon)
    return RolloutModel(controller)


def test_cost_matrices_are_registered_as_buffers() -> None:
    rollout = _build_rollout()
    buffers = dict(rollout.named_buffers())
    assert "_Q" in buffers
    assert "_R" in buffers
    assert isinstance(rollout._Q, torch.Tensor)


def test_cost_buffers_are_non_persistent_and_absent_from_state_dict() -> None:
    """V-3: the static cost matrices must not enter the checkpoint state_dict
    (they were not attributes before, and are reconstructible from the problem),
    preserving the existing ModelCheckpoint contract."""
    rollout = _build_rollout()
    assert "_Q" not in rollout.state_dict()
    assert "_R" not in rollout.state_dict()


def test_cost_buffers_are_not_optimizer_parameters() -> None:
    """Buffers must be excluded from .parameters() so an optimizer never tries
    to train the fixed cost matrices."""
    rollout = _build_rollout()
    assert all(
        p is not rollout._Q and p is not rollout._R for p in rollout.parameters()
    )


def test_cost_buffers_follow_dtype_conversion() -> None:
    rollout = _build_rollout()
    assert rollout._Q.dtype == torch.float64  # native numpy dtype preserved
    rollout.to(dtype=torch.float32)
    assert rollout._Q.dtype == torch.float32
    assert rollout._R.dtype == torch.float32


def test_differentiable_cost_matches_manual_quadratic_form() -> None:
    rollout = _build_rollout()
    torch.manual_seed(0)
    X = torch.randn(3, 5, 2, dtype=torch.float64)
    U = torch.randn(3, 4, 1, dtype=torch.float64)

    cost = rollout._differentiable_cost(X, U)

    Q = rollout._Q.to(dtype=X.dtype)
    R = rollout._R.to(dtype=U.dtype)
    # `.mean(dim=1)` averages over time and KEEPS the batch axis: the rollout
    # returns one cost per trajectory (Annex 06 §4.3's correction), so this
    # compares b values rather than one, which is the stronger check.
    expected = torch.einsum("bti,ij,btj->bt", X[:, :-1], Q, X[:, :-1]).mean(
        dim=1
    ) + torch.einsum("bti,ij,btj->bt", U, R, U).mean(dim=1)
    assert cost.shape == expected.shape == (X.shape[0],)
    assert torch.allclose(cost, expected)
