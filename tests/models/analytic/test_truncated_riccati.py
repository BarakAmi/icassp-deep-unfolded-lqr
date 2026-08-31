import numpy as np
import torch

from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.riccati import finite_horizon_riccati
from mbl.models.analytic.truncated_riccati import (
    TruncatedRiccatiController,
    TruncatedRiccatiSynthesizer,
)
from mbl.models.base import Controller


def _build_problem(horizon: int) -> OptimalControlProblem:
    system = LinearSystem.fully_observable(np.eye(2) * 0.9, np.eye(2))
    cost = QuadraticCost(
        Q=np.repeat(np.eye(2)[None], horizon + 1, axis=0),
        R=np.repeat(np.eye(2)[None], horizon, axis=0),
    )
    return OptimalControlProblem(system=system, cost=cost)


def test_truncated_riccati_satisfies_controller_protocol() -> None:
    horizon = 4
    problem = _build_problem(horizon)
    controller = TruncatedRiccatiController(
        problem, horizon, constraint=BoxConstraint(u_max=1.0)
    )
    assert isinstance(controller, Controller)


def test_truncated_riccati_matches_true_riccati_recursion() -> None:
    """Clipping must never change the underlying P/K solution -- only the
    output of the resulting policy."""
    horizon = 4
    problem = _build_problem(horizon)
    controller = TruncatedRiccatiController(
        problem, horizon, constraint=BoxConstraint(u_max=1.0)
    )
    expected_P, expected_K = finite_horizon_riccati(
        problem.system, problem.cost, horizon
    )

    assert np.allclose(controller.P_arr, expected_P)
    assert np.allclose(controller.K_arr, expected_K)


def test_truncated_riccati_policy_never_exceeds_the_box_bound() -> None:
    horizon = 4
    u_max = 0.2  # deliberately tight, so the unconstrained policy WILL saturate
    problem = _build_problem(horizon)
    controller = TruncatedRiccatiController(
        problem, horizon, constraint=BoxConstraint(u_max=u_max)
    )
    policy = controller.get_control_policy()

    x = np.full((5, 2), 10.0)  # large state -> large unconstrained control
    for t in range(horizon):
        u = policy(t, x)
        assert np.all(np.abs(u) <= u_max + 1e-9)


def test_truncated_riccati_policy_matches_unclipped_riccati_within_bounds() -> None:
    """With a generous bound, small states shouldn't saturate at all -- the
    truncated policy must then equal the plain Riccati policy exactly."""
    horizon = 4
    problem = _build_problem(horizon)
    controller = TruncatedRiccatiController(
        problem, horizon, constraint=BoxConstraint(u_max=1000.0)
    )
    policy = controller.get_control_policy()

    x = np.full((3, 2), 0.01)
    for t in range(horizon):
        u = policy(t, x)
        expected = -x @ controller.K_arr[t].T
        assert np.allclose(u, expected)


def test_truncated_riccati_get_signature_reports_type_horizon_and_constraint() -> None:
    """`problem` is intentionally absent from this controller's own signature
    (Phase 1E): `ProblemSignatureCallback` already logs it once at the root."""
    horizon = 4
    problem = _build_problem(horizon)
    constraint = BoxConstraint(u_max=1.0)
    controller = TruncatedRiccatiController(problem, horizon, constraint=constraint)

    signature = controller.get_signature()

    assert signature == {
        "type": "TruncatedRiccatiController",
        "horizon": horizon,
        "constraint": constraint.get_signature(),
    }


class TestTruncatedRiccatiSynthesizerBackendEnvelope:
    """`TruncatedRiccatiSynthesizer`'s declared envelope was widened (NB04,
    box-constrained benchmarking) from NumPy-only to the full backend set,
    now that its concrete `Constraint` (`BoxConstraint`) is verified
    backend-aware -- see the class docstring. These tests pin BOTH halves of
    that change: NumPy synthesis still behaves exactly as before (no
    regression), and a torch `ComputeContext` -- previously rejected with
    `UnsupportedBackendError` -- now succeeds and agrees numerically."""

    def _synthesize(self, ctx: ComputeContext, u_max: float = 0.2):
        horizon = 4
        problem = _build_problem(horizon)
        synthesizer = TruncatedRiccatiSynthesizer(
            horizon=horizon, constraint=BoxConstraint(u_max=u_max)
        )
        return synthesizer.synthesize(problem, ctx)

    def test_numpy_context_still_synthesizes(self) -> None:
        ctx = ComputeContext(
            backend=Backend.NUMPY, device="cpu", precision=Precision.FLOAT64
        )
        artifact = self._synthesize(ctx)
        assert isinstance(artifact.P_arr, np.ndarray)
        assert isinstance(artifact.K_arr, np.ndarray)

    def test_torch_context_now_synthesizes_without_raising(self) -> None:
        ctx = ComputeContext(
            backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64
        )
        artifact = self._synthesize(ctx)
        assert isinstance(artifact.P_arr, torch.Tensor)
        assert isinstance(artifact.K_arr, torch.Tensor)

    def test_torch_and_numpy_syntheses_agree_numerically(self) -> None:
        numpy_ctx = ComputeContext(
            backend=Backend.NUMPY, device="cpu", precision=Precision.FLOAT64
        )
        torch_ctx = ComputeContext(
            backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64
        )
        numpy_artifact = self._synthesize(numpy_ctx)
        torch_artifact = self._synthesize(torch_ctx)

        assert np.allclose(numpy_artifact.K_arr, torch_artifact.K_arr.numpy())

    def test_torch_context_policy_still_respects_the_box_bound(self) -> None:
        u_max = 0.2  # deliberately tight, so the unconstrained policy WILL saturate
        ctx = ComputeContext(
            backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64
        )
        artifact = self._synthesize(ctx, u_max=u_max)
        policy = artifact.make_policy()

        x = torch.full((5, 2), 10.0, dtype=torch.float64)
        for t in range(4):
            u = policy(t, x)
            assert torch.all(torch.abs(u) <= u_max + 1e-9)
