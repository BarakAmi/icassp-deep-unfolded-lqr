"""Phase 2 acceptance tests for `experiments.zero_shot`
(docs/planning/NB06_OOD_GENERALIZATION_PLAN.md Sec 3.2/8): evaluating a
synthesized artifact on its OWN nominal problem must reproduce
`evaluate_synthesized_controller`'s cost exactly (the shared-kernel law --
`evaluate_under_shift` is a peer orchestrator over the SAME cost primitives,
never a parallel reimplementation); the feasibility audit must read
(near-)zero for a projecting family; and the saturation-rate metric must
read 1.0 under a deliberately over-tight bound and 0.0 under a deliberately
loose one.
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.ood.noise import ExoticBatchSpec, NoiseFamily
from mbl.core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments import ContenderSpec, EvaluationProtocol, Experiment
from mbl.experiments.evaluation import evaluate_synthesized_controller
from mbl.experiments.zero_shot import (
    ShiftMetrics,
    evaluate_under_shift,
    synthesize_nominal_contenders,
)
from mbl.models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer
from mbl.models.guards import require_linear_quadratic

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
HORIZON = 10


def _problem(
    u_max: float, *, state_dim: int = 4, control_dim: int = 2, seed: int = 0
) -> OptimalControlProblem:
    return LQRProblemFactory(
        state_dim=state_dim,
        control_dim=control_dim,
        horizon=HORIZON,
        seed=seed,
        u_max=u_max,
    ).build()


def _truncated_riccati_artifact(problem: OptimalControlProblem):
    constraint = problem.constraints[0]  # type: ignore[index]
    return TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(problem, CTX)


def _batches(
    problem: OptimalControlProblem,
    *,
    n_batches: int = 3,
    batch_size: int = 64,
    seed: int = 1,
):
    spec = GaussianBatchSpec(
        state_dim=problem.system.dimensions.state_dim,
        horizon=HORIZON,
        batch_size=batch_size,
        seed=seed,
        process_noise_std=0.5,
    )
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
    return tuple(sampler() for _ in range(n_batches))


class TestReproducesNominalCostExactly:
    def test_cost_mean_matches_evaluate_synthesized_controller(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem)

        nominal_metrics, _ = evaluate_synthesized_controller(artifact, problem, batches)
        shift_metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.3)

        assert shift_metrics.cost_mean == pytest.approx(
            nominal_metrics["eval_expected_cost"], rel=1e-12
        )

    def test_batch_costs_array_matches_exactly(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem)

        _, nominal_arrays = evaluate_synthesized_controller(artifact, problem, batches)
        _, shift_arrays = evaluate_under_shift(artifact, problem, batches, u_max=0.3)

        np.testing.assert_allclose(
            shift_arrays["batch_costs"], nominal_arrays["eval_batch_costs"], rtol=1e-12
        )


class TestFeasibilityAudit:
    def test_max_violation_is_at_float_tolerance_for_a_projecting_family(self) -> None:
        problem = _problem(u_max=0.2)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.2)

        assert metrics.max_violation <= 1e-9

    def test_n_batches_is_recorded(self) -> None:
        problem = _problem(u_max=0.2)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=5)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.2)
        assert metrics.n_batches == 5


class TestSaturationRate:
    def test_saturates_fully_under_a_deliberately_over_tight_bound(self) -> None:
        # u_max effectively ~0 relative to the system's natural control
        # scale: the truncated-Riccati policy clips to the bound on
        # virtually every (t, batch, dim) entry.
        problem = _problem(u_max=1e-8)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=1e-8)
        assert metrics.saturation_rate == pytest.approx(1.0, abs=1e-6)

    def test_never_saturates_under_a_deliberately_loose_bound(self) -> None:
        # u_max effectively unbounded: the unconstrained Riccati optimum
        # never comes close to it.
        problem = _problem(u_max=1e6)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=1e6)
        assert metrics.saturation_rate == pytest.approx(0.0, abs=1e-9)


class TestTrajectoryCapture:
    def test_capture_trajectories_returns_stacked_arrays_of_the_right_shape(
        self,
    ) -> None:
        problem = _problem(u_max=0.3, state_dim=4, control_dim=2)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2, batch_size=16)

        _, arrays = evaluate_under_shift(
            artifact, problem, batches, u_max=0.3, capture_trajectories=True
        )

        assert arrays["trajectory_states"].shape == (2, 16, HORIZON + 1, 4)
        assert arrays["trajectory_controls"].shape == (2, 16, HORIZON, 2)

    def test_no_trajectory_arrays_when_not_requested(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2)

        _, arrays = evaluate_under_shift(artifact, problem, batches, u_max=0.3)
        assert "trajectory_states" not in arrays
        assert "trajectory_controls" not in arrays


class TestShiftMetricsRobustStatistics:
    def test_median_and_iqr_are_reported_and_bracket_correctly(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=8)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.3)
        assert isinstance(metrics, ShiftMetrics)
        assert metrics.cost_q25 <= metrics.cost_median <= metrics.cost_q75

    def test_q90_and_cvar10_bracket_correctly(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=8)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.3)
        assert metrics.cost_q75 <= metrics.cost_q90
        # CVaR at the 10% tail is the MEAN of the worst 10% -- it must be at
        # least as large as the point (q90) that defines that tail.
        assert metrics.cost_cvar10 >= metrics.cost_q90 - 1e-9


class TestPerTrajectoryStatisticsAreCorrectlyReduced:
    """Regression anchor for NB06 plan Sec 2.4/6/11 S3: `ShiftMetrics`'
    quantiles must come from the POOLED PER-TRAJECTORY cost distribution,
    never from `n_batches` per-batch MEANS -- verified here by independently
    re-deriving the pooled per-trajectory costs via the exact same kernel
    (`CostReduction.PER_SAMPLE`) `evaluate_under_shift` claims to use, never
    by re-invoking `evaluate_under_shift` itself."""

    def test_trajectory_costs_and_quantiles_match_an_independent_per_sample_reduction(
        self,
    ) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=3, batch_size=16)

        metrics, arrays = evaluate_under_shift(artifact, problem, batches, u_max=0.3)

        _, cost = require_linear_quadratic(problem)
        Q = torch.as_tensor(time_invariant_slice(cost.Q))
        R = torch.as_tensor(time_invariant_slice(cost.R))
        per_batch_reference: list[np.ndarray] = []
        for initial_state, process_noise, measurement_noise in batches:
            policy = artifact.make_policy()
            with torch.no_grad():
                X, _, U = problem.system.run(
                    policy, initial_state, process_noise, measurement_noise
                )
                per_sample = total_quadratic_cost(
                    Q.to(dtype=X.dtype, device=X.device),
                    R.to(dtype=U.dtype, device=U.device),
                    X,
                    U,
                    conventions=cost.conventions,
                    reduction=CostReduction.PER_SAMPLE,
                )
            per_batch_reference.append(per_sample.detach().cpu().numpy())
        reference = np.concatenate(per_batch_reference).astype(np.float64)

        np.testing.assert_allclose(arrays["trajectory_costs"], reference, rtol=1e-12)
        assert metrics.cost_median == pytest.approx(
            float(np.median(reference)), rel=1e-12
        )
        assert metrics.cost_q25 == pytest.approx(
            float(np.percentile(reference, 25)), rel=1e-12
        )
        assert metrics.cost_q75 == pytest.approx(
            float(np.percentile(reference, 75)), rel=1e-12
        )
        assert metrics.cost_q90 == pytest.approx(
            float(np.percentile(reference, 90)), rel=1e-12
        )

        # The property that actually distinguishes this from the OLD
        # (per-batch-mean) statistic: with batch_size=16 > 1 and a
        # non-degenerate within-batch cost spread, the median of BATCH
        # MEANS differs from the median of the POOLED per-trajectory costs.
        batch_mean_median = float(
            np.median([float(b.mean()) for b in per_batch_reference])
        )
        assert metrics.cost_median != pytest.approx(batch_mean_median, rel=1e-9)


class TestAppliedControlBound:
    """The C-blind mechanism (NB06 plan Sec 2.6/3.9): `applied_control_bound`
    wraps the policy so the PLANT rolls out on the clipped (applied) control
    while `max_violation` is judged on the COMMANDED (pre-clip) signal."""

    def test_none_reproduces_the_unwrapped_path_exactly(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2, batch_size=32)

        default_metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.3)
        explicit_none_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=0.3, applied_control_bound=None
        )
        assert explicit_none_metrics == default_metrics

    def test_a_looser_bound_never_binds_and_matches_the_unwrapped_rollout(
        self,
    ) -> None:
        # Truncated-Riccati is trained/saturated at u_max=0.3, so its
        # commanded |u| never exceeds 0.3 -- an external clip at 10.0 can
        # never activate, so the wrapped and unwrapped rollouts must be
        # identical: the exact-null acceptance test for Sec 2.6's c > 1 case.
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2, batch_size=32)

        unwrapped_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=10.0
        )
        wrapped_metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=10.0, applied_control_bound=10.0
        )
        assert wrapped_metrics.cost_mean == pytest.approx(
            unwrapped_metrics.cost_mean, rel=1e-12
        )
        assert wrapped_metrics.max_violation == pytest.approx(
            unwrapped_metrics.max_violation, abs=1e-9
        )

    def test_a_tighter_bound_clips_the_applied_control_and_raises_the_audit(
        self,
    ) -> None:
        # Trained/saturated at 0.3; an applied bound of 0.1 is tighter, so
        # the commanded signal (still <= 0.3) legitimately exceeds it --
        # the feasibility audit must therefore read POSITIVE (the first,
        # and only, setting in this notebook where it can).
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2, batch_size=32)

        metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=0.1, applied_control_bound=0.1
        )
        assert metrics.max_violation > 0.0
        # The APPLIED control is clipped to 0.1: saturation (measured
        # against u_max=0.1 here) should read very high.
        assert metrics.saturation_rate > 0.5

    def test_applied_signal_is_bounded_by_the_applied_control_bound(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=1, batch_size=32)

        _, arrays = evaluate_under_shift(
            artifact,
            problem,
            batches,
            u_max=0.1,
            applied_control_bound=0.1,
            capture_trajectories=True,
        )
        assert float(np.abs(arrays["trajectory_controls"]).max()) <= 0.1 + 1e-9


class TestDivergenceRate:
    def test_defaults_to_zero_when_not_requested(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.3)
        assert metrics.divergence_rate == 0.0

    def test_zero_when_the_threshold_is_never_exceeded(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2, batch_size=32)

        metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=0.3, divergence_reference_norm=1e6
        )
        assert metrics.divergence_rate == pytest.approx(0.0, abs=1e-9)

    def test_one_when_the_threshold_is_always_exceeded(self) -> None:
        problem = _problem(u_max=0.3)
        artifact = _truncated_riccati_artifact(problem)
        batches = _batches(problem, n_batches=2, batch_size=32)

        metrics, _ = evaluate_under_shift(
            artifact, problem, batches, u_max=0.3, divergence_reference_norm=1e-6
        )
        assert metrics.divergence_rate == pytest.approx(1.0, abs=1e-6)


class TestPooledStatisticsStableUnderCauchyNoise:
    """Regression anchor for the v1 defect (NB06 plan Sec 2.4/11 S3): under
    Cauchy process noise the cost has infinite expectation, and a median of
    per-batch MEANS would itself wander with batch geometry (a batch mean of
    Cauchy-driven costs does not concentrate). The pooled per-trajectory
    median must instead stay within a fixed, generous bound regardless of
    `batch_size` -- not an exact cross-configuration equality (the draws
    genuinely differ), a stability sanity check."""

    @staticmethod
    def _cauchy_batches(
        problem: OptimalControlProblem, *, batch_size: int, n_batches: int
    ):
        spec = ExoticBatchSpec(
            state_dim=problem.system.dimensions.state_dim,
            horizon=HORIZON,
            batch_size=batch_size,
            seed=7,
            process_noise_std=0.3,
            family=NoiseFamily.CAUCHY,
        )
        sampler, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64)
        return tuple(sampler() for _ in range(n_batches))

    @pytest.mark.parametrize("batch_size", [8, 128])
    def test_median_stays_within_a_generous_bound_regardless_of_batch_size(
        self, batch_size: int
    ) -> None:
        # A loose bound: isolates the STATISTIC under test from saturation
        # effects (the point here is Cauchy's tail, not the box).
        problem = _problem(u_max=10.0)
        artifact = _truncated_riccati_artifact(problem)
        batches = self._cauchy_batches(problem, batch_size=batch_size, n_batches=4)

        metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=10.0)

        assert np.isfinite(metrics.cost_median)
        assert 0.0 <= metrics.cost_median <= 1e4


def _nominal_experiment(tmp_batch_size: int = 32) -> Experiment:
    problem = LQRProblemFactory(
        state_dim=3, control_dim=2, horizon=6, seed=0, u_max=0.3
    )
    batch = GaussianBatchSpec(
        state_dim=3, horizon=6, batch_size=tmp_batch_size, seed=1, process_noise_std=0.4
    )
    return Experiment(
        name="zero_shot_synthesize_test",
        problem=problem,
        contenders=(
            ContenderSpec(family="truncated_riccati", config={"horizon": 6}),
            ContenderSpec(
                family="unfolded_fixed",
                config={
                    "num_iterations": 3,
                    "step_size_init": 0.05,
                    "step_size_max": 0.5,
                    "horizon": 6,
                },
            ),
        ),
        evaluation=EvaluationProtocol(batch_spec=batch, n_batches=2),
        ctx=CTX,
    )


class TestSynthesizeNominalContenders:
    def test_returns_one_artifact_per_contender_and_the_nominal_report(
        self, tmp_path
    ) -> None:
        experiment = _nominal_experiment()
        artifacts, report = synthesize_nominal_contenders(experiment, root=tmp_path)

        assert set(artifacts) == {"truncated_riccati", "unfolded_fixed"}
        assert set(report.results) == {"truncated_riccati", "unfolded_fixed"}

    def test_reevaluating_the_in_memory_artifact_on_the_nominal_problem_matches_the_cached_report(
        self, tmp_path
    ) -> None:
        # Both contenders here are deterministic (no gradient training), so
        # the SECOND (in-memory) synthesis must reproduce the FIRST
        # (cached) run's own cost exactly -- unlike a trained family, where
        # a second training pass need not be bit-identical.
        experiment = _nominal_experiment()
        artifacts, report = synthesize_nominal_contenders(experiment, root=tmp_path)
        problem = experiment.problem.build()
        batches = experiment.evaluation.build_batches(experiment.ctx)

        for label, artifact in artifacts.items():
            metrics, _ = evaluate_under_shift(artifact, problem, batches, u_max=0.3)
            assert metrics.cost_mean == pytest.approx(
                report.metric(label, "eval_expected_cost"), rel=1e-9
            )

    def test_second_call_is_served_from_cache_for_the_nominal_run(
        self, tmp_path
    ) -> None:
        experiment = _nominal_experiment()
        _, first_report = synthesize_nominal_contenders(experiment, root=tmp_path)
        _, second_report = synthesize_nominal_contenders(experiment, root=tmp_path)

        assert all(r.disposition == "fresh" for r in first_report.results.values())
        assert all(r.disposition == "hit" for r in second_report.results.values())
