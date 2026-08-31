"""Acceptance tests for `applications.ood.constraint_shift`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 2.6/3.9): Axis C.
`ControlBoundShift`'s bound arithmetic and signature; `CommandRecorder`'s
commanded-vs-applied split; `rehost_at_bound`'s per-family dispatch --
identity at the nominal bound, exact parameter transplant for every
learned/derived family, a fresh solve for `truncated_riccati`, and a raise
for an unrecognized family (deliberately NO silent identity fallback, unlike
the horizon rehost). `TestConstraintNullTheorem` is the module's most
important test: the exact `multiplier**2` homogeneity a joint
``(u_max, sigma, sigma_0)`` scale predicts for every homogeneous contender.
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.ood.constraint_shift import (
    CommandRecorder,
    ControlBoundShift,
    rehost_at_bound,
)
from mbl.applications.ood.perturbations import (
    DynamicsPerturbation,
    PerturbedLQRProblemFactory,
)
from mbl.applications.recipes import UnfoldedKind
from mbl.applications.recipes.base import TrainedControllerArtifact, null_harness
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import ContenderSpec
from mbl.experiments.zero_shot import evaluate_under_shift
from mbl.models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
STATE_DIM, CONTROL_DIM, HORIZON = 3, 2, 6
U_MAX = 0.3


def _base_factory(u_max: float = U_MAX) -> LQRProblemFactory:
    return LQRProblemFactory(
        state_dim=STATE_DIM,
        control_dim=CONTROL_DIM,
        horizon=HORIZON,
        seed=0,
        u_max=u_max,
    )


def _batches(
    problem: OptimalControlProblem,
    *,
    n_batches: int = 2,
    seed: int = 3,
    process_noise_std: float = 0.4,
    initial_state_std: float = 1.0,
    batch_size: int = 32,
):
    spec = GaussianBatchSpec(
        state_dim=STATE_DIM,
        horizon=HORIZON,
        batch_size=batch_size,
        seed=seed,
        process_noise_std=process_noise_std,
        initial_state_std=initial_state_std,
    )
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    return tuple(sampler() for _ in range(n_batches))


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=3)


def _unfolded_alpha_spec() -> ContenderSpec:
    return ContenderSpec(
        family="unfolded",
        config={
            "kind": UnfoldedKind.LEARNED_STEP_SIZE,
            "plan": _fast_plan(),
            "num_iterations": 3,
            "step_size_init": 0.05,
            "step_size_max": 0.8,
            "horizon": HORIZON,
        },
        label="unfolded_alpha",
    )


def _unfolded_alpha_p_spec() -> ContenderSpec:
    return ContenderSpec(
        family="unfolded_warmstart",
        config={
            "kind": UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            "schedule": LayerwiseTrainingPlan(
                optimizer=OptimizerSpec("adam", 0.05),
                warmup_epochs_per_layer=1,
                refinement_epochs=1,
                train_matrix_from="each",
            ),
            "num_iterations": 3,
            "step_size_init": 0.05,
            "step_size_max": 0.8,
            "horizon": HORIZON,
        },
        label="unfolded_alpha_p",
    )


def _neural_spec() -> ContenderSpec:
    return ContenderSpec(
        family="neural",
        config={"hidden_dim": 4, "plan": _fast_plan()},
        label="neural",
    )


def _synthesize(
    spec: ContenderSpec, problem: OptimalControlProblem
) -> TrainedControllerArtifact:
    recipe = spec.resolve()
    harness = null_harness(
        GaussianBatchSpec(
            state_dim=STATE_DIM,
            horizon=HORIZON,
            batch_size=32,
            seed=0,
            process_noise_std=0.4,
        ),
        CTX,
    )
    artifact = recipe.build_synthesizer(problem, harness).synthesize(problem, CTX)
    assert isinstance(artifact, TrainedControllerArtifact)
    return artifact


class TestControlBoundShift:
    def test_bound_scales_the_nominal_u_max(self) -> None:
        shift = ControlBoundShift(multiplier=0.5)
        assert shift.bound(0.4) == pytest.approx(0.2)

    def test_multiplier_one_reproduces_the_nominal_bound(self) -> None:
        shift = ControlBoundShift(multiplier=1.0)
        assert shift.bound(0.4) == pytest.approx(0.4)

    def test_signature_reflects_the_multiplier(self) -> None:
        assert (
            ControlBoundShift(0.5).get_signature()
            != ControlBoundShift(2.0).get_signature()
        )


class TestCommandRecorder:
    def test_wrap_clips_output_and_records_the_unclipped_value(self) -> None:
        def raw_policy(t: int, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x[:, :CONTROL_DIM], 5.0)

        recorder = CommandRecorder()
        wrapped = recorder.wrap(raw_policy, bound=1.0)
        x = torch.zeros(4, STATE_DIM)
        applied = wrapped(0, x)

        assert torch.all(applied == 1.0)
        assert len(recorder.commanded) == 1
        assert torch.all(recorder.commanded[0] == 5.0)

    def test_a_looser_bound_never_clips(self) -> None:
        def raw_policy(t: int, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x[:, :CONTROL_DIM], 0.3)

        recorder = CommandRecorder()
        wrapped = recorder.wrap(raw_policy, bound=10.0)
        applied = wrapped(0, torch.zeros(4, STATE_DIM))
        assert torch.allclose(applied, torch.full_like(applied, 0.3))

    def test_accumulates_across_multiple_calls(self) -> None:
        def raw_policy(t: int, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x[:, :CONTROL_DIM], float(t))

        recorder = CommandRecorder()
        wrapped = recorder.wrap(raw_policy, bound=1.0)
        for t in range(3):
            wrapped(t, torch.zeros(2, STATE_DIM))
        assert len(recorder.commanded) == 3


class TestIdentityAtNominalBound:
    @pytest.mark.parametrize("spec_fn", [_unfolded_alpha_spec, _unfolded_alpha_p_spec])
    def test_rehosting_to_its_own_bound_reproduces_nominal_cost(self, spec_fn) -> None:
        base = _base_factory()
        problem = base.build()
        spec = spec_fn()
        artifact = _synthesize(spec, problem)
        batches = _batches(problem)

        nominal_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=U_MAX
        )

        same_bound_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=U_MAX
        ).build()
        rehosted = rehost_at_bound(
            artifact, spec, same_bound_problem, horizon=HORIZON, ctx=CTX
        )
        rehosted_metrics, _ = evaluate_under_shift(
            rehosted, same_bound_problem, batches, u_max=U_MAX
        )

        assert rehosted_metrics.cost_mean == pytest.approx(
            nominal_metrics.cost_mean, rel=1e-9
        )


class TestParameterTransplantExactness:
    def test_step_size_transplants_exactly_for_unfolded_alpha(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = _unfolded_alpha_spec()
        artifact = _synthesize(spec, problem)

        shifted_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=0.1
        ).build()
        rehosted = rehost_at_bound(
            artifact, spec, shifted_problem, horizon=HORIZON, ctx=CTX
        )

        source_alpha = artifact.controller.config.parameters["step_size"].get_numpy()
        rehosted_alpha = rehosted.controller.config.parameters["step_size"].get_numpy()
        np.testing.assert_allclose(rehosted_alpha, source_alpha, rtol=1e-8, atol=1e-10)

    def test_step_size_and_matrix_both_transplant_exactly_for_unfolded_alpha_p(
        self,
    ) -> None:
        base = _base_factory()
        problem = base.build()
        spec = _unfolded_alpha_p_spec()
        artifact = _synthesize(spec, problem)

        shifted_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=0.6
        ).build()
        rehosted = rehost_at_bound(
            artifact, spec, shifted_problem, horizon=HORIZON, ctx=CTX
        )

        source_params = artifact.controller.config.parameters
        rehosted_params = rehosted.controller.config.parameters
        np.testing.assert_allclose(
            rehosted_params["step_size"].get_numpy(),
            source_params["step_size"].get_numpy(),
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            rehosted_params["riccati_matrix"].get_numpy(),
            source_params["riccati_matrix"].get_numpy(),
            rtol=1e-6,
            atol=1e-9,
        )

    def test_rehosted_unfolded_controller_reports_the_shifted_bound(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = _unfolded_alpha_spec()
        artifact = _synthesize(spec, problem)

        shifted_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=0.1
        ).build()
        rehosted = rehost_at_bound(
            artifact, spec, shifted_problem, horizon=HORIZON, ctx=CTX
        )
        assert rehosted.controller.problem.constraints[0].u_max == pytest.approx(0.1)

    def test_neural_weights_transplant_exactly_via_state_dict(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = _neural_spec()
        artifact = _synthesize(spec, problem)

        shifted_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=0.1
        ).build()
        rehosted = rehost_at_bound(
            artifact, spec, shifted_problem, horizon=HORIZON, ctx=CTX
        )

        source_state = artifact.controller.state_dict()
        rehosted_state = rehosted.controller.state_dict()
        assert source_state.keys() == rehosted_state.keys()
        for key in source_state:
            torch.testing.assert_close(rehosted_state[key], source_state[key])


class TestTruncatedRiccatiRehostAtBound:
    def test_resolves_fresh_riccati_at_the_shifted_bound(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )

        shifted_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=0.1
        ).build()
        rehosted = rehost_at_bound(
            artifact, spec, shifted_problem, horizon=HORIZON, ctx=CTX
        )
        direct = TruncatedRiccatiSynthesizer(
            HORIZON,
            shifted_problem.constraints[0],  # type: ignore[index]
        ).synthesize(shifted_problem, CTX)

        np.testing.assert_allclose(rehosted.P_arr, direct.P_arr)
        np.testing.assert_allclose(rehosted.K_arr, direct.K_arr)


class TestUnrecognizedFamilyRaises:
    def test_raises_value_error(self) -> None:
        base = _base_factory()
        problem = base.build()
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )

        # Monkeypatch-free: fabricate a spec that resolves to an unrecognized
        # family via a minimal stand-in recipe rather than mutating the
        # real registry.
        class _FakeRecipe:
            family = "not_a_real_family"

        class _FakeSpec:
            def resolve(self) -> object:
                return _FakeRecipe()

        with pytest.raises(ValueError, match="unrecognized family"):
            rehost_at_bound(artifact, _FakeSpec(), problem, horizon=HORIZON, ctx=CTX)


class TestConstraintBlindProtocol:
    """C-blind (NB06 plan Sec 2.6): the artifact is NOT rehosted; the plant
    enforces the shifted bound via `evaluate_under_shift`'s
    `applied_control_bound` (already unit-tested in
    `tests/experiments/test_zero_shot.py::TestAppliedControlBound` -- this
    class only checks the Axis-C-specific framing: a looser bound is an
    exact null, a tighter one produces a genuine, non-zero audit)."""

    def test_a_looser_shift_reproduces_the_nominal_artifact_unchanged(self) -> None:
        base = _base_factory()
        problem = base.build()
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )
        batches = _batches(problem)

        looser_bound = ControlBoundShift(multiplier=4.0).bound(U_MAX)
        nominal_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=U_MAX
        )
        blind_metrics, _ = evaluate_under_shift(
            artifact,
            problem,
            batches,
            u_max=looser_bound,
            applied_control_bound=looser_bound,
        )
        assert blind_metrics.cost_mean == pytest.approx(
            nominal_metrics.cost_mean, rel=1e-12
        )


class TestConstraintNullTheorem:
    """The exact homogeneity theorem (NB06 plan Sec 2.6/3.9): scaling
    `(u_max, sigma, sigma_0)` JOINTLY by `c` multiplies the optimal cost by
    exactly `c**2` for every contender whose policy is positively
    homogeneous (`truncated_riccati`, and the unfolded families once
    C-aware-rehosted: cold-start u^(0)=0, gradient linear in (x, u),
    projection commutes with scaling). `J / c**2` must therefore be flat to
    a tight tolerance -- the strongest, non-statistical correctness
    guarantee available anywhere in NB06."""

    @staticmethod
    def _cost_at_multiplier(
        artifact,
        base: LQRProblemFactory,
        spec: ContenderSpec,
        multiplier: float,
        *,
        rehost: bool,
    ) -> float:
        shift = ControlBoundShift(multiplier=multiplier)
        shifted_u_max = shift.bound(base.u_max)
        shifted_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), u_max_override=shifted_u_max
        ).build()
        artifact_to_use = artifact
        if rehost:
            artifact_to_use = rehost_at_bound(
                artifact, spec, shifted_problem, horizon=HORIZON, ctx=CTX
            )
        # Same seed at every multiplier, both noise sources scaled by the
        # SAME `c` -- this is what makes (x0, w) exactly `c` times the
        # multiplier=1 draw, which is what the homogeneity argument needs.
        batches = _batches(
            shifted_problem,
            batch_size=256,
            process_noise_std=0.4 * multiplier,
            initial_state_std=1.0 * multiplier,
        )
        metrics, _ = evaluate_under_shift(
            artifact_to_use, shifted_problem, batches, u_max=shifted_u_max
        )
        return metrics.cost_mean

    def test_truncated_riccati_is_exactly_c_squared_equivariant(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )

        normalized = [
            self._cost_at_multiplier(artifact, base, spec, c, rehost=True) / c**2
            for c in (0.5, 1.0, 2.0, 4.0)
        ]
        for value in normalized[1:]:
            assert value == pytest.approx(normalized[0], rel=1e-8)

    @pytest.mark.parametrize("spec_fn", [_unfolded_alpha_spec, _unfolded_alpha_p_spec])
    def test_c_aware_rehosted_unfolded_families_are_c_squared_equivariant(
        self, spec_fn
    ) -> None:
        base = _base_factory()
        problem = base.build()
        spec = spec_fn()
        artifact = _synthesize(spec, problem)

        normalized = [
            self._cost_at_multiplier(artifact, base, spec, c, rehost=True) / c**2
            for c in (0.5, 1.0, 2.0)
        ]
        for value in normalized[1:]:
            assert value == pytest.approx(normalized[0], rel=1e-6)
