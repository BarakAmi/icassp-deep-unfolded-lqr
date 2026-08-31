import torch

from mbl.models.iterative.step_size import StepSizeSchedule
from mbl.models.unfolded.iterative_refinement import (
    RiccatiRefinement,
    StepSizeRefinement,
)


class _ConstantParam:
    """Minimal stand-in for an UnfoldedParameter: only needs a `.get()` method."""

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def get(self) -> torch.Tensor:
        return self._value


def test_riccati_refinement_matches_manual_gradient_descent_reference() -> None:
    """Independent, from-scratch gradient-descent reference (not reusing
    compute_lqr_gradient_matrices) that RiccatiRefinement's output must match, given
    the factor-of-2 gradient scaling it applies on top of the shared M/C formula."""
    dtype = torch.float64
    A = torch.tensor([[0.9, 0.0], [0.0, 0.9]], dtype=dtype)
    B = torch.tensor([[1.0], [0.0]], dtype=dtype)
    R = torch.tensor([[1.0]], dtype=dtype)
    P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=dtype)
    y = torch.tensor([[1.0, 0.0]], dtype=dtype)
    num_iterations = 3
    alpha = 0.1

    refinement = RiccatiRefinement(
        step_size=StepSizeSchedule(
            raw=torch.tensor([alpha], dtype=dtype),
            num_iterations=num_iterations,
            horizon=1,
            control_dim=1,
        ),
        num_iterations=num_iterations,
        learnable_parameters={"riccati_matrix": _ConstantParam(P)},
        static_parameters={
            "A": A.unsqueeze(0),
            "B": B.unsqueeze(0),
            "R": R.unsqueeze(0),
        },
    )

    u0 = torch.zeros((1, 1), dtype=dtype)
    u_out = refinement(t=0, y=y, u=u0)

    # Independent reference: grad = 2*(R + B^T P B) u + 2*(B^T P A) y, computed by hand.
    BtP = B.T @ P
    M2 = 2 * (R + BtP @ B)
    C2 = 2 * (BtP @ A)
    y_Ct = y @ C2.T

    u_ref = u0
    for _ in range(num_iterations):
        grad = u_ref @ M2 + y_Ct
        u_ref = u_ref - alpha * grad

    assert torch.allclose(u_out, u_ref)
    assert torch.isclose(u_out, torch.tensor([[-0.5616]], dtype=dtype), atol=1e-6)


def test_step_size_refinement_uses_precomputed_matrices_without_extra_scaling() -> None:
    dtype = torch.float64
    M = torch.tensor([[2.0]], dtype=dtype)
    C = torch.tensor([[1.0, 0.0]], dtype=dtype)
    y = torch.tensor([[1.0, 0.0]], dtype=dtype)
    alpha = 0.5

    refinement = StepSizeRefinement(
        step_size=StepSizeSchedule(
            raw=torch.tensor(alpha, dtype=dtype),
            num_iterations=1,
            horizon=1,
            control_dim=1,
        ),
        num_iterations=1,
        static_parameters={"M_stack": M.unsqueeze(0), "C_stack": C.unsqueeze(0)},
    )

    u0 = torch.zeros((1, 1), dtype=dtype)
    u_out = refinement(t=0, y=y, u=u0)

    y_Ct = y @ C.T
    grad = u0 @ M + y_Ct
    u_ref = u0 - alpha * grad

    assert torch.allclose(u_out, u_ref)


class _MutableParam:
    """Stand-in UnfoldedParameter whose returned value can be swapped, to probe
    RiccatiRefinement's per-rollout cache invalidation."""

    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def get(self) -> torch.Tensor:
        return self._value

    def set(self, value: torch.Tensor) -> None:
        self._value = value


def _manual_gradient_matrices(P, A, B, R, y):
    BtP = B.T @ P
    M2 = 2 * (R + BtP @ B)
    C2 = 2 * (BtP @ A)
    return M2, y @ C2.T


def test_riccati_refinement_batched_precompute_matches_per_step_reference() -> None:
    """V-2: the batched horizon precompute must reproduce, for every time step,
    the per-step (P, A_t, B_t, R_t) gradient matrices exactly."""
    dtype = torch.float64
    A = torch.stack([0.9 * torch.eye(2, dtype=dtype), 0.8 * torch.eye(2, dtype=dtype)])
    B = torch.stack(
        [
            torch.tensor([[1.0], [0.0]], dtype=dtype),
            torch.tensor([[0.5], [0.5]], dtype=dtype),
        ]
    )
    R = torch.stack(
        [torch.tensor([[1.0]], dtype=dtype), torch.tensor([[2.0]], dtype=dtype)]
    )
    P = torch.tensor([[2.0, 0.3], [0.3, 2.0]], dtype=dtype)
    y = torch.tensor([[1.0, -0.5]], dtype=dtype)

    refinement = RiccatiRefinement(
        step_size=StepSizeSchedule(
            raw=torch.full((2, 1), 0.1, dtype=dtype),
            num_iterations=2,
            horizon=1,
            control_dim=1,
        ),
        num_iterations=2,
        learnable_parameters={"riccati_matrix": _ConstantParam(P)},
        static_parameters={"A": A, "B": B, "R": R},
    )

    for t in range(2):
        M, y_Ct = refinement.pre_iteration_hook(t, y)
        M_ref, y_Ct_ref = _manual_gradient_matrices(P, A[t], B[t], R[t], y)
        assert torch.allclose(M, M_ref)
        assert torch.allclose(y_Ct, y_Ct_ref)


def test_riccati_refinement_gradient_flows_to_riccati_matrix() -> None:
    """V-2 must not sever autograd: gradients still reach the learnable P."""
    dtype = torch.float64
    P = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=dtype, requires_grad=True)
    refinement = RiccatiRefinement(
        step_size=StepSizeSchedule(
            raw=torch.tensor(0.1, dtype=dtype),
            num_iterations=1,
            horizon=1,
            control_dim=1,
        ),
        num_iterations=1,
        learnable_parameters={"riccati_matrix": _ConstantParam(P)},
        static_parameters={
            "A": (0.9 * torch.eye(2, dtype=dtype)).unsqueeze(0),
            "B": torch.tensor([[1.0], [0.0]], dtype=dtype).unsqueeze(0),
            "R": torch.tensor([[1.0]], dtype=dtype).unsqueeze(0),
        },
    )
    u_out = refinement(
        t=0, y=torch.tensor([[1.0, 0.5]], dtype=dtype), u=torch.zeros(1, 1, dtype=dtype)
    )
    u_out.sum().backward()
    assert P.grad is not None
    assert torch.isfinite(P.grad).all()


def test_riccati_refinement_cache_is_invalidated_on_rollout_start() -> None:
    """V-2 correctness guard: within a rollout the horizon matrices are cached,
    but on_rollout_start must invalidate them so a changed P is picked up."""
    dtype = torch.float64
    param = _MutableParam(torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=dtype))
    refinement = RiccatiRefinement(
        step_size=StepSizeSchedule(
            raw=torch.tensor(0.1, dtype=dtype),
            num_iterations=1,
            horizon=1,
            control_dim=1,
        ),
        num_iterations=1,
        learnable_parameters={"riccati_matrix": param},
        static_parameters={
            "A": (0.9 * torch.eye(2, dtype=dtype)).unsqueeze(0),
            "B": torch.tensor([[1.0], [0.0]], dtype=dtype).unsqueeze(0),
            "R": torch.tensor([[1.0]], dtype=dtype).unsqueeze(0),
        },
    )
    y = torch.tensor([[1.0, 0.0]], dtype=dtype)

    M_first, _ = refinement.pre_iteration_hook(0, y)
    param.set(torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=dtype))
    M_stale, _ = refinement.pre_iteration_hook(0, y)  # cached -> unchanged
    assert torch.allclose(M_first, M_stale)

    refinement.on_rollout_start()  # rollout boundary invalidates the cache
    M_fresh, _ = refinement.pre_iteration_hook(0, y)
    assert not torch.allclose(M_first, M_fresh)
