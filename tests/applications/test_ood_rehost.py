"""Phase 3 acceptance tests for `applications.ood.rehost`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 1.4/3.5/8): the
Horizon-OOD law -- analytic structure (Truncated-Riccati's Riccati
recursion) is recomputed at the target horizon, learned parameters
(unfolded step size / Riccati-replacing matrix) are frozen and transplanted
exactly, and horizon-free families (neural, COCP, COCP-LB) rehost as a pure
identity. The identity-at-the-nominal-horizon test is the exact-reproduction
regression anchor: rehosting "to" the horizon a contender already trained at
must reproduce its own nominal cost bit-for-bit (up to the reparameterization
round-trip's own floating-point tolerance).
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.ood.perturbations import (
    DynamicsPerturbation,
    PerturbedLQRProblemFactory,
)
from mbl.applications.ood.rehost import rehost_at_horizon
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


def _base_factory() -> LQRProblemFactory:
    return LQRProblemFactory(
        state_dim=STATE_DIM,
        control_dim=CONTROL_DIM,
        horizon=HORIZON,
        seed=0,
        u_max=U_MAX,
    )


def _batches(problem: OptimalControlProblem, *, n_batches: int = 2, seed: int = 3):
    spec = GaussianBatchSpec(
        state_dim=STATE_DIM,
        horizon=problem.system.dimensions.state_dim and HORIZON,
        batch_size=32,
        seed=seed,
        process_noise_std=0.4,
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


class TestIdentityAtNominalHorizon:
    @pytest.mark.parametrize("spec_fn", [_unfolded_alpha_spec, _unfolded_alpha_p_spec])
    def test_rehosting_to_its_own_horizon_reproduces_nominal_cost(
        self, spec_fn
    ) -> None:
        base = _base_factory()
        problem = base.build()
        spec = spec_fn()
        artifact = _synthesize(spec, problem)
        batches = _batches(problem)

        nominal_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=U_MAX
        )

        same_horizon_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, same_horizon_problem, horizon=HORIZON, ctx=CTX
        )
        rehosted_metrics, _ = evaluate_under_shift(
            rehosted, same_horizon_problem, batches, u_max=U_MAX
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

        extended_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=4 * HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, extended_problem, horizon=4 * HORIZON, ctx=CTX
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

        extended_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=4 * HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, extended_problem, horizon=4 * HORIZON, ctx=CTX
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

    def test_rehosted_controller_reports_the_target_horizon(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = _unfolded_alpha_spec()
        artifact = _synthesize(spec, problem)

        extended_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=4 * HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, extended_problem, horizon=4 * HORIZON, ctx=CTX
        )
        assert rehosted.controller.config.horizon == 4 * HORIZON


class TestTruncatedRiccatiRehost:
    def test_resolves_fresh_riccati_at_the_target_horizon(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )

        extended_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=3 * HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, extended_problem, horizon=3 * HORIZON, ctx=CTX
        )

        assert rehosted.P_arr.shape[0] == 3 * HORIZON + 1
        assert rehosted.K_arr.shape[0] == 3 * HORIZON

    def test_matches_a_direct_synthesis_at_the_target_horizon(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
        constraint = problem.constraints[0]  # type: ignore[index]
        artifact = TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(
            problem, CTX
        )

        extended_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=2 * HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, extended_problem, horizon=2 * HORIZON, ctx=CTX
        )
        direct = TruncatedRiccatiSynthesizer(2 * HORIZON, constraint).synthesize(
            extended_problem, CTX
        )
        np.testing.assert_allclose(rehosted.P_arr, direct.P_arr)
        np.testing.assert_allclose(rehosted.K_arr, direct.K_arr)


class TestHorizonFreeFamiliesRehostAsIdentity:
    def test_neural_rehost_returns_the_same_object(self) -> None:
        base = _base_factory()
        problem = base.build()
        spec = ContenderSpec(
            family="neural",
            config={"hidden_dim": 4, "plan": _fast_plan()},
            label="neural",
        )
        harness = null_harness(
            GaussianBatchSpec(
                state_dim=STATE_DIM,
                horizon=HORIZON,
                batch_size=16,
                seed=0,
                process_noise_std=0.4,
            ),
            CTX,
        )
        artifact = (
            spec.resolve().build_synthesizer(problem, harness).synthesize(problem, CTX)
        )

        extended_problem = PerturbedLQRProblemFactory(
            base=base, perturbation=DynamicsPerturbation(), horizon_override=3 * HORIZON
        ).build()
        rehosted = rehost_at_horizon(
            artifact, spec, extended_problem, horizon=3 * HORIZON, ctx=CTX
        )
        assert rehosted is artifact
