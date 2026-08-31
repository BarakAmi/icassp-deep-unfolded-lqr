"""Lifecycle-contract tests (REFACTOR_PLAN v3 §7.3, Stage S2 slice): the
two-phase `Synthesizer -> SynthesizedController -> policy` laws, verified
on the migrated analytic families (Riccati, truncated Riccati). Trained
families join in Stage S3, when their `synthesize` gains its engine
wiring; the `TrainableController` sub-contract's seam (`as_module`) is
verified on `NeuralPolicy` today.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.core.cost.quadratic_cost import QuadraticCost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import (
    Backend,
    ComputeContext,
    UnsupportedBackendError,
    default_compute_context,
)
from mbl.core.system.linear_system import LinearSystem
from mbl.models.analytic.riccati import RiccatiController, RiccatiSynthesizer
from mbl.models.analytic.truncated_riccati import (
    TruncatedRiccatiController,
    TruncatedRiccatiSynthesizer,
)
from mbl.models.lifecycle import (
    ALL_BACKENDS,
    SynthesizedController,
    Synthesizer,
    TrainableController,
    ensure_backend_supported,
    supported_backends,
)
from mbl.models.neural.nerual import NeuralConfig, NeuralPolicy

N_DIM, M_DIM, HORIZON = 3, 2, 10

TORCH_CPU = ComputeContext(backend=Backend.TORCH, device="cpu")


@pytest.fixture(scope="module")
def problem() -> OptimalControlProblem:
    """A seeded, marginally-scaled, non-trivial LQR instance."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((N_DIM, N_DIM))
    A /= np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.standard_normal((N_DIM, M_DIM))
    system = LinearSystem.fully_observable(A, B)
    cost = QuadraticCost(
        Q=np.repeat(np.eye(N_DIM)[None], HORIZON + 1, axis=0),
        R=np.repeat(np.eye(M_DIM)[None], HORIZON, axis=0),
    )
    return OptimalControlProblem(system=system, cost=cost)


@pytest.fixture(scope="module")
def rollout_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    x0 = rng.normal(size=(8, N_DIM))
    w = 0.1 * rng.normal(size=(8, HORIZON, N_DIM))
    v = np.zeros((8, HORIZON, N_DIM))
    return x0, w, v


class TestProtocolConformance:
    def test_analytic_synthesizers_satisfy_the_synthesizer_protocol(self, problem):
        assert isinstance(RiccatiSynthesizer(HORIZON), Synthesizer)
        assert isinstance(
            TruncatedRiccatiSynthesizer(HORIZON, BoxConstraint(u_max=1.0)),
            Synthesizer,
        )

    def test_artifacts_satisfy_the_synthesized_controller_protocol(self, problem):
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem)
        assert isinstance(artifact, SynthesizedController)
        truncated = TruncatedRiccatiSynthesizer(
            HORIZON, BoxConstraint(u_max=1.0)
        ).synthesize(problem)
        assert isinstance(truncated, SynthesizedController)

    def test_trainable_controller_is_a_synthesized_controller_sub_contract(self):
        """Structural containment: a TrainableController must carry the full
        artifact surface (context/make_policy/get_signature) PLUS as_module."""

        class _FullArtifact:
            context = default_compute_context()

            def make_policy(self):
                return lambda t, y: y

            def get_signature(self):
                return {"type": "_FullArtifact"}

            def as_module(self) -> nn.Module:
                return nn.Identity()

        class _ModuleOnly:
            def as_module(self) -> nn.Module:
                return nn.Identity()

        class _PlainArtifact:
            context = default_compute_context()

            def make_policy(self):
                return lambda t, y: y

            def get_signature(self):
                return {"type": "_PlainArtifact"}

        assert isinstance(_FullArtifact(), TrainableController)
        # The training seam alone is not an artifact...
        assert not isinstance(_ModuleOnly(), TrainableController)
        # ...and an artifact without the seam is not trainable -- the
        # sub-contract genuinely narrows in both directions.
        assert isinstance(_PlainArtifact(), SynthesizedController)
        assert not isinstance(_PlainArtifact(), TrainableController)

    def test_neural_policy_exposes_the_as_module_seam(self, problem):
        """T2.b groundwork: the engine-facing parameter boundary exists and
        is the policy's own module (backbone + head registered)."""
        policy = NeuralPolicy(
            problem, NeuralConfig(state_dim=N_DIM, control_dim=M_DIM, hidden_dim=8)
        )
        module = policy.as_module()
        assert module is policy
        assert isinstance(module, nn.Module)
        assert list(module.parameters()), "the seam must expose parameters"


class TestPhaseDisjointnessAndEquivalence:
    def test_synthesize_make_policy_rollout_end_to_end(self, problem, rollout_inputs):
        """§7.3(a): the two-phase path runs with no attribute smuggling and
        prices trajectories identically to the legacy single-phase path."""
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem)
        policy = artifact.make_policy()
        x0, w, v = rollout_inputs
        X, _, U = problem.system.run(policy, x0, w, v)
        assert X.shape == (8, HORIZON + 1, N_DIM)
        assert U.shape == (8, HORIZON, M_DIM)
        assert np.isfinite(problem.cost(X, U)).all()

    def test_two_phase_solution_is_bit_identical_to_the_legacy_controller(
        self, problem
    ):
        """The synthesizer and the legacy controller run the same kernel
        recursion: identical P/K on the default NumPy/float64 context."""
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem)
        legacy = RiccatiController(problem, HORIZON)
        assert np.array_equal(artifact.P_arr, legacy.P_arr)
        assert np.array_equal(artifact.K_arr, legacy.K_arr)

    def test_truncated_two_phase_matches_legacy_and_saturates(
        self, problem, rollout_inputs
    ):
        u_max = 0.05
        constraint = BoxConstraint(u_max=u_max)
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(problem)
        legacy = TruncatedRiccatiController(problem, HORIZON, constraint)
        assert np.array_equal(artifact.K_arr, legacy.K_arr)

        x0, w, v = rollout_inputs
        _, _, U = problem.system.run(artifact.make_policy(), x0, w, v)
        assert np.all(np.abs(U) <= u_max + 1e-15)


class TestStatelessnessAndFreshness:
    def test_synthesizer_is_frozen_and_unmutated_by_synthesis(self, problem):
        """§7.3: `synthesize` returns its artifact instead of mutating self
        -- structurally guaranteed (frozen dataclass) and behaviorally
        observed (identical state and results across repeat calls)."""
        synthesizer = RiccatiSynthesizer(HORIZON)
        before = dataclasses.asdict(synthesizer)
        first = synthesizer.synthesize(problem)
        second = synthesizer.synthesize(problem)
        assert dataclasses.asdict(synthesizer) == before
        assert np.array_equal(first.K_arr, second.K_arr)
        with pytest.raises(dataclasses.FrozenInstanceError):
            synthesizer.horizon = HORIZON + 1  # type: ignore[misc]  # the mutation attempt is the test

    def test_make_policy_returns_a_fresh_closure_per_call(self, problem):
        """§7.3(b): freshness law -- consecutive calls yield independent
        policy objects (no state can leak across rollouts)."""
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem)
        assert artifact.make_policy() is not artifact.make_policy()


class TestResidencyAndBackendFlexibility:
    def test_default_context_residency_is_numpy(self, problem):
        """§7.3(d): the artifact truthfully reports where its bytes live."""
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem)
        assert artifact.context == default_compute_context()
        assert isinstance(artifact.K_arr, np.ndarray)
        assert artifact.K_arr.dtype == np.float64

    def test_torch_context_residency_and_policy_substrate(self, problem):
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem, TORCH_CPU)
        assert artifact.context == TORCH_CPU
        assert isinstance(artifact.K_arr, torch.Tensor)
        assert artifact.K_arr.dtype == torch.float64

        policy = artifact.make_policy()
        y = torch.randn(6, N_DIM, dtype=torch.float64)
        u = policy(0, y)
        assert isinstance(u, torch.Tensor)
        assert u.shape == (6, M_DIM)
        torch.testing.assert_close(u, -y @ artifact.K_arr[0].T)

    def test_cross_backend_synthesis_agrees(self, problem):
        """§7.9 at the synthesizer level: NumPy/CPU vs torch/CPU produce the
        same controller within float64 tolerance."""
        numpy_artifact = RiccatiSynthesizer(HORIZON).synthesize(problem)
        torch_artifact = RiccatiSynthesizer(HORIZON).synthesize(problem, TORCH_CPU)
        np.testing.assert_allclose(
            torch_artifact.K_arr.numpy(),
            numpy_artifact.K_arr,
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            torch_artifact.P_arr.numpy(),
            numpy_artifact.P_arr,
            atol=1e-10,
            rtol=1e-10,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="host has no CUDA")
    def test_cuda_synthesis_agrees_and_reports_residency(self, problem):
        ctx = ComputeContext(backend=Backend.TORCH, device="cuda")
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem, ctx)
        assert artifact.K_arr.device.type == "cuda"
        assert artifact.context.device.startswith("cuda")
        reference = RiccatiSynthesizer(HORIZON).synthesize(problem)
        np.testing.assert_allclose(
            artifact.K_arr.cpu().numpy(), reference.K_arr, atol=1e-9, rtol=1e-9
        )

    def test_float32_context_governs_precision(self, problem):
        """Precision is the context's third axis: nothing but the injected
        context decides the working width."""
        ctx = ComputeContext(backend=Backend.TORCH, precision="float32")
        artifact = RiccatiSynthesizer(HORIZON).synthesize(problem, ctx)
        assert artifact.K_arr.dtype == torch.float32


class _NumpyOnlySynthesizer:
    """A minimal double declaring a narrow envelope -- exists ONLY to test
    `ensure_backend_supported`'s generic rejection mechanism in isolation,
    decoupled from any one real family's own envelope choice (which may
    legitimately widen over time, as `TruncatedRiccatiSynthesizer`'s did --
    see the next test and that class's own docstring). Coupling this
    contract's coverage to a specific real family is exactly what silently
    lost it the moment that family's envelope changed; a purpose-built
    double keeps the generic mechanism tested regardless."""

    SUPPORTED_BACKENDS = frozenset({Backend.NUMPY})


class TestDeclaredEnvelope:
    def test_riccati_declares_the_full_envelope(self):
        assert supported_backends(RiccatiSynthesizer(HORIZON)) == ALL_BACKENDS
        assert supported_backends(RiccatiSynthesizer) == ALL_BACKENDS

    def test_truncated_declares_the_full_envelope(self):
        """Widened (NB04, box-constrained benchmarking) from the original
        NumPy-only declaration: the one concrete `Constraint` this tree
        ships (`BoxConstraint`) is backend-aware, and the delegated Riccati
        solve already supports every backend -- see
        `TruncatedRiccatiSynthesizer`'s own docstring for the full
        reasoning and the condition that triggered the widening."""
        synthesizer = TruncatedRiccatiSynthesizer(HORIZON, BoxConstraint(u_max=1.0))
        assert supported_backends(synthesizer) == ALL_BACKENDS

    def test_outside_the_envelope_fails_at_ingress_with_the_typed_error(self):
        """T1.e: declared capability, not discovered failure -- a family
        that DOES declare a narrow envelope dies loudly at synthesis
        ingress, naming both sides."""
        with pytest.raises(UnsupportedBackendError, match="numpy") as excinfo:
            ensure_backend_supported(_NumpyOnlySynthesizer(), TORCH_CPU)
        assert "torch" in str(excinfo.value)
