"""Phase 3 acceptance tests for `experiments.shifted_evaluation`
(docs/planning/NB07_ROBUST_TRAINING_PLAN.md Sec 5.3/11): evaluating a nominal
artifact on the nominal problem reproduces `evaluate_synthesized_controller`'s
cost; the feasibility audit is <= tolerance for a projecting contender;
saturation rate is 1.0 under a deliberately over-tight `u_max` and 0.0 under
a deliberately loose one; trajectory capture returns the first batch only.
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    UnfoldedKind,
    build_unfolded_controller,
)
from mbl.applications.recipes.base import TrainedControllerArtifact
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments.evaluation import evaluate_synthesized_controller
from mbl.experiments.shifted_evaluation import evaluate_under_shift

_STATE_DIM = 4
_CONTROL_DIM = 2
_HORIZON = 20
_U_MAX = 0.5
_BATCH_SIZE = 64
_N_BATCHES = 4


def _ctx() -> ComputeContext:
    return ComputeContext(
        backend=Backend.TORCH,
        device="cpu",
        precision=Precision.from_torch_dtype(torch.float64),
    )


def _problem(u_max: float = _U_MAX):
    return LQRProblemFactory(
        state_dim=_STATE_DIM,
        control_dim=_CONTROL_DIM,
        horizon=_HORIZON,
        seed=0,
        u_max=u_max,
    ).build()


def _batches(problem, n_batches: int = _N_BATCHES):
    spec = GaussianBatchSpec(
        state_dim=_STATE_DIM,
        horizon=_HORIZON,
        batch_size=_BATCH_SIZE,
        seed=1,
        process_noise_std=0.5,
    )
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    return tuple(sampler() for _ in range(n_batches))


def _artifact(problem, ctx: ComputeContext) -> TrainedControllerArtifact:
    controller = build_unfolded_controller(
        problem,
        ctx,
        UnfoldedBuildSpec(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            num_iterations=6,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=_HORIZON,
        ),
    )
    return TrainedControllerArtifact(
        controller=controller, context=ctx, synthesizer_signature={"type": "test"}
    )


class TestReproducesNominalCost:
    def test_cost_mean_matches_evaluate_synthesized_controller(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem)

        nominal_metrics, _ = evaluate_synthesized_controller(artifact, problem, batches)
        shift_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=_U_MAX
        )

        assert shift_metrics.cost_mean == pytest.approx(
            nominal_metrics["eval_expected_cost"], rel=1e-9
        )


class TestFeasibilityAudit:
    def test_max_violation_within_tolerance_for_projecting_contender(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem)
        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=_U_MAX)
        assert metrics.max_violation <= 1e-9


class TestSaturationRate:
    def test_over_tight_bound_saturates_fully(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem)
        # An absurdly tight bound: every projected control sits at the bound.
        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=1e-6)
        assert metrics.saturation_rate == pytest.approx(1.0, abs=1e-6)

    def test_loose_bound_never_saturates(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem)
        # An absurdly loose bound relative to the projected controls.
        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=1e6)
        assert metrics.saturation_rate == 0.0


class TestPooledStatistics:
    def test_n_trajectories_is_n_batches_times_batch_size(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem, n_batches=3)
        metrics, arrays = evaluate_under_shift(artifact, problem, batches, u_max=_U_MAX)
        assert metrics.n_batches == 3
        assert metrics.n_trajectories == 3 * _BATCH_SIZE
        assert arrays["batch_costs"].shape == (3 * _BATCH_SIZE,)

    def test_median_and_iqr_are_well_ordered(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem)
        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=_U_MAX)
        assert metrics.cost_q25 <= metrics.cost_median <= metrics.cost_q75


class TestTrajectoryCapture:
    def test_capture_returns_first_batch_shapes(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem, n_batches=2)
        _, arrays = evaluate_under_shift(
            artifact, problem, batches, u_max=_U_MAX, capture_trajectories=True
        )
        assert arrays["trajectory_states"].shape == (
            _BATCH_SIZE,
            _HORIZON + 1,
            _STATE_DIM,
        )
        assert arrays["trajectory_controls"].shape == (
            _BATCH_SIZE,
            _HORIZON,
            _CONTROL_DIM,
        )

    def test_no_capture_by_default(self) -> None:
        ctx = _ctx()
        problem = _problem()
        artifact = _artifact(problem, ctx)
        batches = _batches(problem)
        _, arrays = evaluate_under_shift(artifact, problem, batches, u_max=_U_MAX)
        assert "trajectory_states" not in arrays


class TestForeignProblemEvaluation:
    """The whole point of the module: score an artifact trained on one
    problem against a DIFFERENT (perturbed) problem instance."""

    def test_evaluating_on_a_different_seed_problem_does_not_raise(self) -> None:
        ctx = _ctx()
        nominal_problem = _problem()
        artifact = _artifact(nominal_problem, ctx)
        foreign_problem = LQRProblemFactory(
            state_dim=_STATE_DIM,
            control_dim=_CONTROL_DIM,
            horizon=_HORIZON,
            seed=99,
            u_max=_U_MAX,
        ).build()
        batches = _batches(foreign_problem)
        metrics, _ = evaluate_under_shift(
            artifact, foreign_problem, batches, u_max=_U_MAX
        )
        assert np.isfinite(metrics.cost_mean)
