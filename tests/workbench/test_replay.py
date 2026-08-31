"""NB03 acceptance test (A-6) for `workbench.replay`: `load_training_history`
returns an identical curve regardless of disposition (fresh vs. cache HIT),
and `None` for a non-learnable (analytic) contender -- executed end to end
against the real experiments/cache/recipe layers, never mocked."""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes import UnfoldedKind
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import (
    CachePolicy,
    ContenderResult,
    ContenderSpec,
    EvaluationProtocol,
    Experiment,
    run_experiment,
)
from mbl.models.analytic.riccati import LocalCostToGoModel
from mbl.models.constrained.cocp import COCPController
from mbl.workbench import (
    UnfoldingLandscapeInputs,
    cocp_reference_point,
    load_contender_artifacts,
    load_training_history,
    render_training_log_replay,
    unfolding_landscape_inputs,
)

HORIZON = 6
EPOCHS = 5
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
PROBLEM = LQRProblemFactory(state_dim=3, control_dim=2, horizon=HORIZON, seed=0)
BATCH = GaussianBatchSpec(
    state_dim=3, horizon=HORIZON, batch_size=8, seed=0, process_noise_std=0.3
)

RICCATI = ContenderSpec(family="riccati", config={"horizon": HORIZON})
TRAINED = ContenderSpec(
    family="unfolded",
    config={
        "kind": UnfoldedKind.LEARNED_STEP_SIZE,
        "plan": TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=EPOCHS),
        "num_iterations": 3,
        "step_size_init": 0.02,
        "step_size_max": 0.5,
        "horizon": HORIZON,
    },
)
TRAINED_LABEL = "unfolded_learned_step_size"
ANALYTIC_LABEL = "analytic"

MATRIX_TRAINED = ContenderSpec(
    family="unfolded",
    config={
        "kind": UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
        "plan": TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=EPOCHS),
        "num_iterations": 3,
        "step_size_init": 0.02,
        "step_size_max": 0.5,
        "horizon": HORIZON,
    },
)
MATRIX_TRAINED_LABEL = "unfolded_learned_step_size_and_matrix"


def _experiment() -> Experiment:
    return Experiment(
        name="replay_acceptance",
        problem=PROBLEM,
        contenders=(RICCATI, TRAINED),
        evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=1),
        ctx=CTX,
    )


def _experiment_with_matrix() -> Experiment:
    return Experiment(
        name="replay_acceptance_matrix",
        problem=PROBLEM,
        contenders=(RICCATI, MATRIX_TRAINED),
        evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=1),
        ctx=CTX,
    )


class TestLoadTrainingHistorySymmetry:
    def test_fresh_history_has_one_row_per_epoch(self, tmp_path) -> None:
        report = run_experiment(_experiment(), root=tmp_path)
        result = report.results[TRAINED_LABEL]
        assert result.disposition == "fresh"

        history = load_training_history(result)

        assert history is not None
        assert len(history) == EPOCHS
        assert list(history["epoch"]) == list(range(EPOCHS))
        assert "loss" in history.columns

    def test_cached_history_matches_the_fresh_one_exactly(self, tmp_path) -> None:
        experiment = _experiment()
        fresh_report = run_experiment(experiment, root=tmp_path)
        fresh_history = load_training_history(fresh_report.results[TRAINED_LABEL])
        assert fresh_history is not None

        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        cached_result = cached_report.results[TRAINED_LABEL]
        assert cached_result.disposition == "hit"
        cached_history = load_training_history(cached_result)

        assert cached_history is not None
        assert len(cached_history) == len(fresh_history)
        assert np.array_equal(
            fresh_history["loss"].to_numpy(), cached_history["loss"].to_numpy()
        )
        assert np.array_equal(
            fresh_history["epoch"].to_numpy(), cached_history["epoch"].to_numpy()
        )

    def test_analytic_contender_has_no_training_history_fresh_or_cached(
        self, tmp_path
    ) -> None:
        experiment = _experiment()
        fresh_report = run_experiment(experiment, root=tmp_path)
        assert load_training_history(fresh_report.results[ANALYTIC_LABEL]) is None

        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        assert load_training_history(cached_report.results[ANALYTIC_LABEL]) is None

    def test_missing_run_dir_and_arrays_is_no_history(self) -> None:
        result = ContenderResult(
            label="x",
            family="riccati",
            cache_key="k",
            content_digest="d",
            disposition="fresh",
            run_dir=None,
            metrics={},
        )
        assert load_training_history(result) is None


class TestRenderTrainingLogReplay:
    """Phase 1 directive 3 (Cache Replay Visibility): a cache HIT must print
    the identical training progression a fresh run printed live -- executed
    end to end, never mocked, mirroring `TestLoadTrainingHistorySymmetry`."""

    def test_fresh_replay_prints_one_row_per_logged_epoch(self, tmp_path) -> None:
        report = run_experiment(_experiment(), root=tmp_path)
        result = report.results[TRAINED_LABEL]
        assert result.disposition == "fresh"

        text = render_training_log_replay(result)

        assert f"[Cache Replay] {TRAINED_LABEL}" in text
        assert "disposition=fresh" in text
        assert text.count("Epoch:") == EPOCHS  # all EPOCHS < default warmup a=10
        assert "Cost (Loss):" in text
        # New multi-line format: each learned parameter on its own line, no
        # legacy "Learned Parameters:" prefix.
        assert "step_size (alpha) =" in text
        assert "Learned Parameters:" not in text

    def test_render_training_log_replay_flushes_the_trace_to_the_stream(
        self, tmp_path
    ) -> None:
        """Directive 2 (replay enforcement): the function itself flushes the
        trace to the stream (defaulting to stdout), so a cached notebook run
        is never a silent, blank cell -- the printed text equals the returned
        text."""
        import io

        report = run_experiment(_experiment(), root=tmp_path)
        buffer = io.StringIO()
        returned = render_training_log_replay(
            report.results[TRAINED_LABEL], stream=buffer
        )
        printed = buffer.getvalue()
        assert printed.rstrip("\n") == returned
        assert "Epoch:" in printed

    def test_cached_replay_reproduces_the_same_rows_as_the_fresh_one(
        self, tmp_path
    ) -> None:
        experiment = _experiment()
        fresh_report = run_experiment(experiment, root=tmp_path)
        fresh_rows = render_training_log_replay(
            fresh_report.results[TRAINED_LABEL]
        ).splitlines()[1:]  # drop the identity header (disposition differs)

        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        cached_result = cached_report.results[TRAINED_LABEL]
        assert cached_result.disposition == "hit"
        cached_rows = render_training_log_replay(cached_result).splitlines()[1:]

        assert cached_rows == fresh_rows

    def test_analytic_contender_has_no_training_log_fresh_or_cached(
        self, tmp_path
    ) -> None:
        experiment = _experiment()
        fresh_report = run_experiment(experiment, root=tmp_path)
        fresh_text = render_training_log_replay(fresh_report.results[ANALYTIC_LABEL])
        assert "never trained" in fresh_text

        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        cached_text = render_training_log_replay(cached_report.results[ANALYTIC_LABEL])
        assert "never trained" in cached_text

    def test_missing_run_dir_and_arrays_is_the_never_trained_notice(self) -> None:
        result = ContenderResult(
            label="x",
            family="riccati",
            cache_key="k",
            content_digest="d",
            disposition="fresh",
            run_dir=None,
            metrics={},
        )
        assert "never trained" in render_training_log_replay(result)


class TestLoadContenderArtifacts:
    def test_fresh_artifacts_include_disk_only_payloads(self, tmp_path) -> None:
        report = run_experiment(_experiment(), root=tmp_path)
        result = report.results[TRAINED_LABEL]
        assert "metrics_history" not in result.arrays  # the FRESH-path gap

        artifacts = load_contender_artifacts(result)

        assert "metrics_history" in artifacts
        assert "trajectory_states" in artifacts
        assert "eval_batch_costs" in artifacts  # still carries the in-memory payload

    def test_cached_artifacts_are_a_superset_too(self, tmp_path) -> None:
        experiment = _experiment()
        run_experiment(experiment, root=tmp_path)
        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        artifacts = load_contender_artifacts(cached_report.results[TRAINED_LABEL])
        assert {"metrics_history", "trajectory_states", "eval_batch_costs"} <= set(
            artifacts
        )

    def test_no_run_dir_falls_back_to_in_memory_arrays_only(self) -> None:
        result = ContenderResult(
            label="x",
            family="riccati",
            cache_key="k",
            content_digest="d",
            disposition="fresh",
            run_dir=None,
            metrics={},
            arrays={"eval_batch_costs": np.array([1.0, 2.0])},
        )
        artifacts = load_contender_artifacts(result)
        assert set(artifacts) == {"eval_batch_costs"}


class TestUnfoldingLandscapeInputs:
    """NB03 Phase 3 requirement 1's data bridge: `unfolding_landscape_inputs`
    reads a trained unfolded contender's persisted `ParameterSnapshotCallback`/
    `TrajectoryLoggingCallback` artifacts -- executed end to end against the
    real experiments/cache/recipe layers, never mocked, mirroring
    `TestLoadTrainingHistorySymmetry`."""

    def test_reads_step_size_and_true_p_fallback_for_learned_step_size_only(
        self, tmp_path
    ) -> None:
        report = run_experiment(_experiment(), root=tmp_path)
        result = report.results[TRAINED_LABEL]

        inputs = unfolding_landscape_inputs(result, t_star=2, sample_index=0)

        assert isinstance(inputs, UnfoldingLandscapeInputs)
        assert inputs.x_star.shape == (3,)  # state_dim
        assert inputs.step_size.shape == (3, 2)  # (J=num_iterations, control_dim)
        assert inputs.P_own.shape == (
            3,
            3,
        )  # the true-P fallback, state_dim x state_dim
        assert inputs.A.shape == (3, 3)
        assert inputs.B.shape == (3, 2)
        assert inputs.R.shape == (2, 2)

    def test_reads_learned_riccati_matrix_when_present(self, tmp_path) -> None:
        report = run_experiment(_experiment_with_matrix(), root=tmp_path)
        result = report.results[MATRIX_TRAINED_LABEL]

        inputs = unfolding_landscape_inputs(result, t_star=1, sample_index=0)

        assert inputs.P_own.shape == (3, 3)
        # RiccatiMatrixParameter.get() is a Cholesky product (L @ L.T) --
        # positive-semidefinite by construction, the meaningful contract
        # here (a specific numeric value would be an implementation detail).
        eigenvalues = np.linalg.eigvalsh(inputs.P_own)
        assert np.all(eigenvalues >= -1e-8)

    def test_x_star_matches_trajectory_states_at_t_star(self, tmp_path) -> None:
        report = run_experiment(_experiment(), root=tmp_path)
        result = report.results[TRAINED_LABEL]
        states = load_contender_artifacts(result)["trajectory_states"]

        inputs = unfolding_landscape_inputs(result, t_star=2, sample_index=1)

        assert np.array_equal(inputs.x_star, states[1, 2])

    def test_disposition_agnostic_between_fresh_and_cached(self, tmp_path) -> None:
        experiment = _experiment()
        fresh_report = run_experiment(experiment, root=tmp_path)
        fresh_inputs = unfolding_landscape_inputs(
            fresh_report.results[TRAINED_LABEL], t_star=1
        )

        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        cached_result = cached_report.results[TRAINED_LABEL]
        assert cached_result.disposition == "hit"
        cached_inputs = unfolding_landscape_inputs(cached_result, t_star=1)

        assert np.allclose(fresh_inputs.step_size, cached_inputs.step_size)
        assert np.allclose(fresh_inputs.P_own, cached_inputs.P_own)
        assert np.array_equal(fresh_inputs.x_star, cached_inputs.x_star)


COCP_U_MAX = 0.3
COCP_PROBLEM = LQRProblemFactory(
    state_dim=3, control_dim=2, horizon=HORIZON, seed=0, u_max=COCP_U_MAX
)
COCP_CONTENDER = ContenderSpec(
    family="cocp",
    config={
        # A real recipe/engine pass, deliberately small-scale (not
        # NB04's full batch=256/horizon=50): this test exercises
        # `cocp_reference_point`'s CONTRACT against genuinely trained,
        # persisted artifacts, not the training run's own convergence.
        "plan": TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=4),
        "batch_size": 16,
        "horizon": HORIZON,
    },
    label="cocp",
)


class TestCocpReferencePoint:
    """NB04 COCP/viz refinement plan Sec 2.2/3.4: `cocp_reference_point`
    re-solves a trained COCP contender's one-step QP at a frozen state --
    executed end to end against the real experiments/cache/recipe layers
    (a real `run_experiment` pass, the `COCPRecipe`'s own solver
    resolution included), never mocked, mirroring
    `TestUnfoldingLandscapeInputs`."""

    def _experiment(self) -> Experiment:
        return Experiment(
            name="cocp_reference_point_acceptance",
            problem=COCP_PROBLEM,
            contenders=(RICCATI, COCP_CONTENDER),
            evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=1),
            ctx=CTX,
        )

    def test_solved_control_is_feasible_and_finite(self, tmp_path) -> None:
        report = run_experiment(self._experiment(), root=tmp_path)
        result = report.results["cocp"]
        problem = COCP_PROBLEM.build()
        A = problem.system.A_t.array
        B = problem.system.B_t.array
        Q = problem.cost.Q[0]
        R = problem.cost.R[0]
        x_star = np.array([0.2, -0.1, 0.3])
        local_model = LocalCostToGoModel(
            A=A, B=B, Q=Q, R=R, P_next=np.eye(3), process_noise_cov=np.eye(3) * 0.01
        )

        reference = cocp_reference_point(result, x_star, local_model=local_model)

        assert reference.point.shape == (2,)
        assert np.all(np.isfinite(reference.point))
        assert np.isfinite(reference.value)
        assert np.max(np.abs(reference.point)) <= COCP_U_MAX + 1e-4

    def test_matches_an_independent_direct_qp_layer_solve(self, tmp_path) -> None:
        """Cross-check against a fresh `COCPController.build_qp_layer` solve
        fed the SAME persisted (P_sqrt, q, A, B, R, u_max) -- the exact
        "only have saved values" use case that method's own docstring
        names -- confirming `cocp_reference_point` reads the right
        artifacts and solves the right QP, not just "some QP"."""
        report = run_experiment(self._experiment(), root=tmp_path)
        result = report.results["cocp"]
        artifacts = load_contender_artifacts(result)
        problem = COCP_PROBLEM.build()
        A = problem.system.A_t.array
        B = problem.system.B_t.array
        Q = problem.cost.Q[0]
        R = problem.cost.R[0]
        x_star = np.array([0.2, -0.1, 0.3])
        local_model = LocalCostToGoModel(
            A=A, B=B, Q=Q, R=R, P_next=np.eye(3), process_noise_cov=np.eye(3) * 0.01
        )

        reference = cocp_reference_point(result, x_star, local_model=local_model)

        layer = COCPController.build_qp_layer(
            np.asarray(artifacts["parameter_A"]),
            np.asarray(artifacts["parameter_B"]),
            np.asarray(artifacts["parameter_R"]),
            float(np.asarray(artifacts["parameter_u_max"])),
        )
        dtype = torch.float64
        x = torch.as_tensor(x_star, dtype=dtype).reshape(1, -1, 1)
        P_sqrt = torch.as_tensor(
            np.asarray(artifacts["parameter_P_sqrt"]), dtype=dtype
        ).unsqueeze(0)
        q = torch.as_tensor(
            np.asarray(artifacts["parameter_q"]), dtype=dtype
        ).unsqueeze(0)
        with torch.no_grad():
            (u,) = layer(x, P_sqrt, q)
        direct_point = u.squeeze(-1).squeeze(0).numpy()

        assert np.allclose(reference.point, direct_point, atol=1e-5)

    def test_disposition_agnostic_between_fresh_and_cached(self, tmp_path) -> None:
        experiment = self._experiment()
        x_star = np.array([0.2, -0.1, 0.3])
        problem = COCP_PROBLEM.build()
        local_model = LocalCostToGoModel(
            A=problem.system.A_t.array,
            B=problem.system.B_t.array,
            Q=problem.cost.Q[0],
            R=problem.cost.R[0],
            P_next=np.eye(3),
            process_noise_cov=np.eye(3) * 0.01,
        )

        fresh_report = run_experiment(experiment, root=tmp_path)
        fresh = cocp_reference_point(
            fresh_report.results["cocp"], x_star, local_model=local_model
        )

        cached_report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("reuse")
        )
        cached_result = cached_report.results["cocp"]
        assert cached_result.disposition == "hit"
        cached = cocp_reference_point(cached_result, x_star, local_model=local_model)

        assert np.allclose(fresh.point, cached.point)
        assert fresh.value == pytest.approx(cached.value)

    def test_training_log_narrates_cocps_own_learned_parameters(self, tmp_path) -> None:
        """COCP previously trained with NO per-epoch narration at all,
        unlike every other learned contender (`applications.recipes.cocp
        .cocp_parameter_log_summaries`, wired into `COCPRecipe
        .extra_callbacks` alongside the existing parameter snapshot) --
        confirmed end to end via the SAME cache-replay mechanism
        `TestRenderTrainingLogReplay` already exercises for the unfolded
        family."""
        report = run_experiment(self._experiment(), root=tmp_path)
        result = report.results["cocp"]
        assert result.disposition == "fresh"

        text = render_training_log_replay(result)

        assert "[Cache Replay] cocp" in text
        assert "disposition=fresh" in text
        assert "Epoch:" in text
        assert "P_sqrt (learned cost-to-go sqrt)" in text
        assert "q (learned linear term)" in text
        assert "never trained" not in text
