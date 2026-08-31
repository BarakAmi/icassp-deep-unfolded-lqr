import numpy as np
import pytest
import torch

from mbl.core.utils.guards import GUARDS, guards_disabled
from mbl.core.utils.shapes import enforce_tensor_shapes


@pytest.fixture(autouse=True)
def _restore_guard_state():
    enabled, max_tier = GUARDS.enabled, GUARDS.max_tier
    yield
    GUARDS.enabled, GUARDS.max_tier = enabled, max_tier


def test_valid_shapes_pass_through_numpy() -> None:
    @enforce_tensor_shapes(x=("B", "n"), returns=("B", "m"))
    def project(x: np.ndarray) -> np.ndarray:
        return x[:, :1]

    out = project(np.ones((4, 3)))
    assert out.shape == (4, 1)


def test_wrong_rank_is_rejected() -> None:
    @enforce_tensor_shapes(x=("B", "n"))
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="must be 2D"):
        identity(np.ones((4, 3, 2)))


def test_named_axis_mismatch_across_arguments_is_rejected() -> None:
    @enforce_tensor_shapes(x=("B", "n"), y=("B", "m"))
    def combine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x

    combine(np.ones((4, 2)), np.ones((4, 5)))  # batch B agrees -> ok
    with pytest.raises(ValueError, match="mismatch across arguments"):
        combine(np.ones((4, 2)), np.ones((7, 5)))  # B: 4 vs 7


def test_offset_label_checks_trajectory_length() -> None:
    @enforce_tensor_shapes(u=("B", "T", "m"), returns=("B", "T+1", "n"))
    def rollout(u: np.ndarray) -> np.ndarray:
        batch, horizon, _ = u.shape
        return np.zeros((batch, horizon + 1, 2))

    rollout(np.zeros((3, 5, 1)))  # returns (3, 6, 2): 6 == 5 + 1 -> ok

    @enforce_tensor_shapes(u=("B", "T", "m"), returns=("B", "T+1", "n"))
    def bad_rollout(u: np.ndarray) -> np.ndarray:
        batch, horizon, _ = u.shape
        return np.zeros((batch, horizon, 2))  # wrong: T, not T+1

    with pytest.raises(ValueError, match="mismatch across arguments"):
        bad_rollout(np.zeros((3, 5, 1)))


def test_autograd_graph_is_preserved() -> None:
    """The decorator must only read shapes -- never detach/convert -- so a
    grad-requiring tensor keeps its graph and gradients still flow."""
    weight = torch.nn.Parameter(torch.randn(3, 2))

    @enforce_tensor_shapes(x=("B", "n"), returns=("B", "m"))
    def linear(x: torch.Tensor) -> torch.Tensor:
        return x @ weight

    x = torch.randn(4, 3)
    out = linear(x)
    assert out.requires_grad
    out.sum().backward()
    assert weight.grad is not None
    assert torch.count_nonzero(weight.grad) > 0


def test_decorator_returns_original_function_when_tier_inactive() -> None:
    """Mandate: when the SHAPE tier is inactive at decoration time, return the
    original, unwrapped function (zero overhead, jit/compile friendly)."""

    def raw(x: np.ndarray) -> np.ndarray:
        return x

    with guards_disabled():
        wrapped = enforce_tensor_shapes(x=("B", "n"))(raw)

    assert wrapped is raw
