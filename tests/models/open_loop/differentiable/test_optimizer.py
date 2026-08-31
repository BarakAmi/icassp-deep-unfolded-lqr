import torch

from mbl.models.open_loop.differentiable.optimizer import AnalyticalGradientDescent
from mbl.models.iterative.step_size import StepSizeSchedule

M, T, m = 4, 3, 2


def _make_leaf(batch: int = 1) -> torch.Tensor:
    U0 = torch.full((batch, T, m), 5.0, dtype=torch.float64)
    return U0.clone().detach().requires_grad_(True)


def test_step_applies_scalar_step_size_update() -> None:
    U = _make_leaf()
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1, dtype=torch.float64),
        num_iterations=M,
        horizon=T,
        control_dim=m,
    )
    optimizer = AnalyticalGradientDescent([U], schedule)
    U.grad = torch.full_like(U, 2.0)

    before = U.detach().clone()
    optimizer.step()

    assert torch.allclose(U.detach(), before - 0.1 * 2.0)


def test_step_skips_parameters_with_no_gradient() -> None:
    U = _make_leaf()
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1, dtype=torch.float64),
        num_iterations=M,
        horizon=T,
        control_dim=m,
    )
    optimizer = AnalyticalGradientDescent([U], schedule)
    assert U.grad is None

    before = U.detach().clone()
    optimizer.step()  # must not raise despite grad being None

    assert torch.equal(U.detach(), before)


def test_iteration_counter_advances_and_selects_the_right_row() -> None:
    """A 2D (per-iteration) schedule proves the internal counter tracks the
    epoch index with no external syncing: step k must use raw[k], not raw[0]
    every time."""
    raw = torch.tensor([[0.5, 0.5], [0.1, 0.1], [0.01, 0.01], [0.0, 0.0]])
    schedule = StepSizeSchedule(raw=raw, num_iterations=M, horizon=T, control_dim=m)
    U = _make_leaf()
    optimizer = AnalyticalGradientDescent([U], schedule)

    assert optimizer._iteration == 0
    U.grad = torch.ones_like(U)
    before = U.detach().clone()
    optimizer.step()
    assert torch.allclose(U.detach(), before - raw[0].reshape(1, m))
    assert optimizer._iteration == 1

    U.grad = torch.ones_like(U)
    before = U.detach().clone()
    optimizer.step()
    assert torch.allclose(U.detach(), before - raw[1].reshape(1, m))
    assert optimizer._iteration == 2


def test_multi_dim_step_size_broadcasts_per_timestep_and_component() -> None:
    """A 3D (per-iteration-per-timestep) schedule must broadcast a distinct
    step size per (t, component) pair against a (batch, T, m) gradient."""
    raw = torch.arange(M * T * m, dtype=torch.float64).reshape(M, T, m) * 0.01
    schedule = StepSizeSchedule(raw=raw, num_iterations=M, horizon=T, control_dim=m)
    U = _make_leaf(batch=3)
    optimizer = AnalyticalGradientDescent([U], schedule)

    grad = torch.ones_like(U)
    U.grad = grad
    before = U.detach().clone()
    optimizer.step()

    expected = before - raw[0].unsqueeze(0) * grad
    assert torch.allclose(U.detach(), expected)


def test_step_is_in_place_and_preserves_leaf_identity() -> None:
    U = _make_leaf()
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1, dtype=torch.float64),
        num_iterations=M,
        horizon=T,
        control_dim=m,
    )
    optimizer = AnalyticalGradientDescent([U], schedule)
    tensor_id = id(U)

    U.grad = torch.ones_like(U)
    optimizer.step()

    assert id(U) == tensor_id
    assert U.requires_grad is True
    assert U.is_leaf is True


def test_step_runs_under_no_grad_and_does_not_track_the_update() -> None:
    U = _make_leaf()
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1, dtype=torch.float64),
        num_iterations=M,
        horizon=T,
        control_dim=m,
    )
    optimizer = AnalyticalGradientDescent([U], schedule)
    U.grad = torch.ones_like(U)

    optimizer.step()

    assert U.grad_fn is None  # still a leaf; the update was not autograd-tracked


def test_zero_grad_clears_gradients_between_steps() -> None:
    U = _make_leaf()
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1, dtype=torch.float64),
        num_iterations=M,
        horizon=T,
        control_dim=m,
    )
    optimizer = AnalyticalGradientDescent([U], schedule)
    U.grad = torch.ones_like(U)

    optimizer.zero_grad()

    assert U.grad is None or torch.all(U.grad == 0)
