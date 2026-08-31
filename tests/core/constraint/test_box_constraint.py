import numpy as np
import pytest
import torch

from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.core.constraint.constraint import Constraint


def test_box_constraint_satisfies_the_constraint_protocol() -> None:
    assert isinstance(BoxConstraint(u_max=1.0), Constraint)


def test_box_constraint_rejects_non_positive_bound() -> None:
    with pytest.raises(ValueError, match="u_max"):
        BoxConstraint(u_max=0.0)
    with pytest.raises(ValueError, match="u_max"):
        BoxConstraint(u_max=-1.0)


def test_box_constraint_rejects_u_min_not_strictly_less_than_u_max() -> None:
    with pytest.raises(ValueError, match="u_min"):
        BoxConstraint(u_max=1.0, u_min=1.0)
    with pytest.raises(ValueError, match="u_min"):
        BoxConstraint(u_max=1.0, u_min=2.0)


def test_box_constraint_defaults_u_min_to_negative_u_max() -> None:
    constraint = BoxConstraint(u_max=2.0)
    assert constraint.u_min == -2.0


def test_call_clips_numpy_array_elementwise() -> None:
    constraint = BoxConstraint(u_max=2.0)
    u = np.array([[-5.0, 0.5, 3.0], [1.0, -1.5, 2.5]])

    clipped = constraint(u)

    assert isinstance(clipped, np.ndarray)
    assert np.array_equal(clipped, np.array([[-2.0, 0.5, 2.0], [1.0, -1.5, 2.0]]))


def test_call_clips_torch_tensor_elementwise_and_preserves_dtype() -> None:
    constraint = BoxConstraint(u_max=1.5)
    u = torch.tensor([[-3.0, 0.2, 4.0]], dtype=torch.float64)

    clipped = constraint(u)

    assert isinstance(clipped, torch.Tensor)
    assert clipped.dtype == torch.float64
    assert torch.equal(clipped, torch.tensor([[-1.5, 0.2, 1.5]], dtype=torch.float64))


def test_call_preserves_torch_autograd_graph_within_bounds() -> None:
    """Values already inside the box must pass through clamp with gradients
    intact -- essential for training a controller whose output only
    occasionally saturates."""
    constraint = BoxConstraint(u_max=5.0)
    u = torch.tensor([0.5, -0.5], dtype=torch.float64, requires_grad=True)

    clipped = constraint(u)
    clipped.sum().backward()

    assert u.grad is not None
    assert torch.equal(u.grad, torch.ones_like(u))


def test_call_supports_asymmetric_bounds() -> None:
    constraint = BoxConstraint(u_max=2.0, u_min=-0.5)
    u = np.array([-5.0, 5.0, 0.0])

    assert np.array_equal(constraint(u), np.array([-0.5, 2.0, 0.0]))


def test_box_constraint_supports_per_dimension_bounds() -> None:
    constraint = BoxConstraint(u_max=np.array([1.0, 3.0]))
    u = np.array([2.0, 2.0])

    assert np.array_equal(constraint(u), np.array([1.0, 2.0]))


def test_box_constraint_get_signature_reports_scalar_bounds_directly() -> None:
    signature = BoxConstraint(u_max=1.5).get_signature()
    assert signature == {"type": "BoxConstraint", "u_max": 1.5, "u_min": -1.5}


def test_box_constraint_get_signature_inlines_small_array_valued_bounds() -> None:
    """Per-dimension bounds are typically small (control_dim is usually a
    handful of actuators) -- these should stay human-readable/diffable as a
    plain list, not be opaquely hashed."""
    signature = BoxConstraint(u_max=np.array([1.0, 3.0])).get_signature()
    assert signature == {
        "type": "BoxConstraint",
        "u_max": [1.0, 3.0],
        "u_min": [-1.0, -3.0],
    }


def test_box_constraint_get_signature_hashes_large_array_valued_bounds() -> None:
    """Once the per-dimension bound array is large enough that inlining it
    would stop being human-readable, fall back to a content hash."""
    signature = BoxConstraint(u_max=np.ones(20)).get_signature()
    assert signature["type"] == "BoxConstraint"
    assert signature["u_max"].startswith("sha256:")
    assert signature["u_min"].startswith("sha256:")


def test_box_constraint_without_get_signature_does_not_satisfy_constraint_protocol() -> (
    None
):
    """Regression lock for the mandatory-Signable fix: a Constraint-shaped
    object missing get_signature() must no longer satisfy the protocol."""

    class _LegacyConstraint:
        def __call__(self, u):
            return u

    assert not isinstance(_LegacyConstraint(), Constraint)


def test_optimal_control_problem_accepts_a_box_constraint() -> None:
    from mbl.core.cost.quadratic_cost import QuadraticCost
    from mbl.core.optimal_control_problem import OptimalControlProblem
    from mbl.core.system.linear_system import LinearSystem

    system = LinearSystem.fully_observable(np.eye(2), np.eye(2))
    cost = QuadraticCost(Q=np.eye(2), R=np.eye(2))
    problem = OptimalControlProblem(
        system=system, cost=cost, constraints=[BoxConstraint(u_max=1.0)]
    )

    assert len(problem.constraints) == 1
