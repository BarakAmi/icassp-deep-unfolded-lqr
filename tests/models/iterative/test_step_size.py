import pytest
import torch

from mbl.models.iterative.step_size import StepSizeProvider, StepSizeSchedule

M, T, m = 5, 3, 2


def _grad_like() -> torch.Tensor:
    """A stand-in gradient tensor, shape (batch, T, m)."""
    return torch.ones(4, T, m, dtype=torch.float64)


def test_schedule_satisfies_step_size_provider_protocol() -> None:
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1), num_iterations=M, horizon=T, control_dim=m
    )
    assert isinstance(schedule, StepSizeProvider)


def test_0d_schedule_broadcasts_to_1x1_regardless_of_iteration() -> None:
    schedule = StepSizeSchedule(
        raw=torch.tensor(0.1), num_iterations=M, horizon=T, control_dim=m
    )
    for i in (0, 1, M - 1):
        alpha = schedule.for_iteration(i)
        assert alpha.shape == (1, 1)
        assert torch.allclose(alpha, torch.tensor([[0.1]], dtype=alpha.dtype))
    # Must broadcast cleanly against a (batch, T, m) gradient.
    assert (schedule.for_iteration(0) * _grad_like()).shape == (4, T, m)


def test_1d_schedule_broadcasts_to_1xm_regardless_of_iteration() -> None:
    raw = torch.tensor([0.1, 0.2])
    schedule = StepSizeSchedule(raw=raw, num_iterations=M, horizon=T, control_dim=m)
    for i in (0, 1, M - 1):
        alpha = schedule.for_iteration(i)
        assert alpha.shape == (1, m)
        assert torch.allclose(alpha, raw.reshape(1, m))
    assert (schedule.for_iteration(0) * _grad_like()).shape == (4, T, m)


def test_2d_schedule_selects_iteration_row() -> None:
    raw = torch.arange(M * m, dtype=torch.float64).reshape(M, m)
    schedule = StepSizeSchedule(raw=raw, num_iterations=M, horizon=T, control_dim=m)
    for i in range(M):
        alpha = schedule.for_iteration(i)
        assert alpha.shape == (1, m)
        assert torch.allclose(alpha, raw[i].reshape(1, m))
    assert (schedule.for_iteration(0) * _grad_like()).shape == (4, T, m)


def test_3d_schedule_selects_iteration_slice() -> None:
    raw = torch.arange(M * T * m, dtype=torch.float64).reshape(M, T, m)
    schedule = StepSizeSchedule(raw=raw, num_iterations=M, horizon=T, control_dim=m)
    for i in range(M):
        alpha = schedule.for_iteration(i)
        assert alpha.shape == (T, m)
        assert torch.allclose(alpha, raw[i])
    assert (schedule.for_iteration(0) * _grad_like()).shape == (4, T, m)


def test_rejects_a_learnable_raw_tensor() -> None:
    with pytest.raises(ValueError, match="require grad"):
        StepSizeSchedule(
            raw=torch.tensor(0.1, requires_grad=True),
            num_iterations=M,
            horizon=T,
            control_dim=m,
        )


def test_rejects_5d_raw_tensor() -> None:
    with pytest.raises(ValueError, match="0D-3D"):
        StepSizeSchedule(
            raw=torch.zeros(1, 1, 1, 1),
            num_iterations=M,
            horizon=T,
            control_dim=m,
        )


def test_rejects_1d_raw_with_wrong_control_dim() -> None:
    with pytest.raises(ValueError, match="rank 1"):
        StepSizeSchedule(
            raw=torch.zeros(m + 1), num_iterations=M, horizon=T, control_dim=m
        )


def test_rejects_2d_raw_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        StepSizeSchedule(
            raw=torch.zeros(M, m + 1), num_iterations=M, horizon=T, control_dim=m
        )


def test_rejects_3d_raw_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        StepSizeSchedule(
            raw=torch.zeros(M, T + 1, m), num_iterations=M, horizon=T, control_dim=m
        )


@pytest.mark.parametrize("field", ["num_iterations", "horizon", "control_dim"])
def test_rejects_non_positive_dimensions(field: str) -> None:
    kwargs = {"num_iterations": M, "horizon": T, "control_dim": m}
    kwargs[field] = 0
    with pytest.raises(ValueError):
        StepSizeSchedule(raw=torch.tensor(0.1), **kwargs)
