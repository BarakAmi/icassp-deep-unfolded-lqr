"""Stage-S4 acceptance tests for `run_experiment` + `ExperimentCache` +
history registry + per-run logging (T3.d/T3.e/T3.j/T3.k), executed end to end
against the real recipe registry, engine, and persistence layers."""

import dataclasses
import logging

import numpy as np
import pytest

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import (
    CachePolicy,
    CacheReadOnlyMissError,
    ContenderSpec,
    EvaluationProtocol,
    Experiment,
    ExperimentHistoryRegistry,
    REGISTRY_FILENAME,
    RUN_LOG_FILENAME,
    code_provenance_stamp,
    run_experiment,
)
from mbl.experiments.cache import CACHE_KEY_PARAM, STAMP_PARAM
from mbl.persistence import load_run_metadata

HORIZON = 12
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
PROBLEM = LQRProblemFactory(state_dim=4, control_dim=2, horizon=HORIZON, seed=0)
BATCH = GaussianBatchSpec(
    state_dim=4, horizon=HORIZON, batch_size=64, seed=0, process_noise_std=0.5
)

RICCATI = ContenderSpec(family="riccati", config={"horizon": HORIZON})
UNFOLDED = ContenderSpec(
    family="unfolded_fixed",
    config={
        "num_iterations": 5,
        "step_size_init": 0.05,
        "step_size_max": 1.0,
        "horizon": HORIZON,
    },
)
TRAINED = ContenderSpec(
    family="neural",
    config={
        "hidden_dim": 8,
        "plan": TrainingPlan(optimizer=OptimizerSpec("adam", 1e-3), epochs=2),
    },
)


def _experiment(*contenders, name="s4_acceptance") -> Experiment:
    return Experiment(
        name=name,
        problem=PROBLEM,
        contenders=tuple(contenders) or (RICCATI, UNFOLDED),
        evaluation=EvaluationProtocol(batch_spec=BATCH, n_batches=2),
        ctx=CTX,
    )


def _run_dirs(root):
    return sorted(p for p in root.iterdir() if p.is_dir())


class TestFreshExecution:
    def test_computes_scores_and_persists_every_contender(self, tmp_path):
        report = run_experiment(_experiment(), root=tmp_path)

        assert set(report.results) == {"analytic", "unfolded_fixed"}
        for result in report.results.values():
            assert result.disposition == "fresh"
            assert result.metrics["eval_expected_cost"] > 0
            assert result.arrays["eval_batch_costs"].shape == (2,)
            metadata = load_run_metadata(_run_dirs(tmp_path)[0])
            assert CACHE_KEY_PARAM in metadata["params"]

        # The Riccati optimum must not lose to the fixed unfolded controller
        # on its own problem -- a sanity check that evaluation really scores
        # the synthesized policies.
        assert (
            report.metric("analytic", "eval_expected_cost")
            <= report.metric("unfolded_fixed", "eval_expected_cost") + 1e-9
        )

    def test_common_noise_realizations_across_contenders(self, tmp_path):
        """The same family under two labels must score IDENTICALLY: every
        contender is evaluated on the same materialized batches."""
        twin = dataclasses.replace(RICCATI, label="analytic_twin")
        report = run_experiment(_experiment(RICCATI, twin), root=tmp_path)
        np.testing.assert_array_equal(
            report.results["analytic"].arrays["eval_batch_costs"],
            report.results["analytic_twin"].arrays["eval_batch_costs"],
        )

    def test_trained_contender_carries_synthesis_final_metrics(self, tmp_path):
        report = run_experiment(_experiment(TRAINED), root=tmp_path)
        metrics = report.results["neural"].metrics
        assert "final_loss" in metrics
        assert "eval_expected_cost" in metrics

    def test_direct_solver_contender_now_carries_a_synthesis_profile(self, tmp_path):
        """A direct-solver contender (Riccati) never enters a `Runner`/
        `ProfilingCallback` training loop, so it previously had NO offline
        timing/memory metrics at all. The generic, family-agnostic wrapper in
        `_measure_synthesis` must measure it uniformly with every other
        family."""
        report = run_experiment(_experiment(RICCATI), root=tmp_path)
        metrics = report.results["analytic"].metrics
        assert "profile/synthesis_wall_time_s" in metrics
        assert metrics["profile/synthesis_wall_time_s"] >= 0.0


class TestCacheInterception:
    def test_second_execution_is_served_entirely_from_cache(self, tmp_path):
        first = run_experiment(_experiment(), root=tmp_path)
        n_dirs = len(_run_dirs(tmp_path))

        second = run_experiment(_experiment(), root=tmp_path)
        assert all(r.disposition == "hit" for r in second.results.values())
        assert len(_run_dirs(tmp_path)) == n_dirs  # no new run directories
        for label in first.results:
            assert (
                second.results[label].metrics["eval_expected_cost"]
                == first.results[label].metrics["eval_expected_cost"]
            )

    def test_adding_a_contender_computes_only_the_delta(self, tmp_path):
        run_experiment(_experiment(RICCATI), root=tmp_path)
        n_dirs = len(_run_dirs(tmp_path))

        report = run_experiment(_experiment(RICCATI, UNFOLDED), root=tmp_path)
        assert report.results["analytic"].disposition == "hit"
        assert report.results["unfolded_fixed"].disposition == "fresh"
        assert len(_run_dirs(tmp_path)) == n_dirs + 1

    def test_recompute_policy_forces_a_fresh_run(self, tmp_path):
        run_experiment(_experiment(RICCATI), root=tmp_path)
        n_dirs = len(_run_dirs(tmp_path))

        report = run_experiment(
            _experiment(RICCATI), root=tmp_path, policy=CachePolicy("recompute")
        )
        assert report.results["analytic"].disposition == "recompute"
        assert len(_run_dirs(tmp_path)) == n_dirs + 1

    def test_readonly_policy_serves_hits_but_never_computes(self, tmp_path):
        run_experiment(_experiment(RICCATI), root=tmp_path)
        report = run_experiment(
            _experiment(RICCATI), root=tmp_path, policy=CachePolicy("readonly")
        )
        assert report.results["analytic"].disposition == "hit"

        with pytest.raises(CacheReadOnlyMissError, match="unfolded_fixed"):
            run_experiment(
                _experiment(UNFOLDED), root=tmp_path, policy=CachePolicy("readonly")
            )

    def test_stamp_mismatch_is_a_loud_warning_logged_miss(
        self, tmp_path, monkeypatch, caplog
    ):
        """The v3 code-provenance law: same content signature under an older
        stamp is never served -- it recomputes, warns, and leaves the stale
        run on disk."""
        run_experiment(_experiment(RICCATI), root=tmp_path)
        n_dirs = len(_run_dirs(tmp_path))

        import mbl.experiments.runner as runner_module

        monkeypatch.setattr(
            runner_module,
            "code_provenance_stamp",
            lambda: "lqr-99.0.0/schema-99",
        )
        with caplog.at_level(logging.WARNING, logger="mbl.experiments.cache"):
            report = run_experiment(_experiment(RICCATI), root=tmp_path)

        assert report.results["analytic"].disposition == "fresh"
        assert any("provenance stamp" in record.message for record in caplog.records)
        # never deleted: both the stale and the fresh run remain readable
        assert len(_run_dirs(tmp_path)) == n_dirs + 1

    def test_content_change_misses_without_touching_other_entries(self, tmp_path):
        run_experiment(_experiment(RICCATI), root=tmp_path)
        changed = dataclasses.replace(
            _experiment(RICCATI), problem=dataclasses.replace(PROBLEM, seed=1)
        )
        report = run_experiment(changed, root=tmp_path)
        assert report.results["analytic"].disposition == "fresh"


class TestHistoryRegistry:
    def test_every_execution_appends_one_record_hits_included(self, tmp_path):
        run_experiment(_experiment(), root=tmp_path)
        run_experiment(_experiment(), root=tmp_path)

        registry = ExperimentHistoryRegistry(tmp_path)
        rows = registry.records()
        assert len(rows) == 2
        assert (tmp_path / REGISTRY_FILENAME).exists()

        fresh, hit = rows
        assert fresh["experiment_name"] == "s4_acceptance"
        assert fresh["state_dim"] == "4"
        assert fresh["control_dim"] == "2"
        assert fresh["horizon"] == str(HORIZON)
        assert fresh["provenance_stamp"] == code_provenance_stamp()
        assert "analytic=fresh" in fresh["dispositions"]
        assert "analytic=hit" in hit["dispositions"]
        assert "analytic" in fresh["final_costs"]
        assert fresh["experiment_signature"] == hit["experiment_signature"]

    def test_registry_is_machine_loadable_with_pandas(self, tmp_path):
        import pandas as pd

        run_experiment(_experiment(RICCATI), root=tmp_path)
        table = pd.read_csv(tmp_path / REGISTRY_FILENAME)
        assert list(table.columns)[0] == "experiment_signature"
        assert len(table) == 1

    def test_rebuild_regenerates_from_run_directories(self, tmp_path):
        run_experiment(_experiment(), root=tmp_path)
        registry = ExperimentHistoryRegistry(tmp_path)
        registry.path.unlink()  # lose the logbook -- loses nothing

        rebuilt = registry.rebuild()
        assert rebuilt == 2  # one record per cache-indexed run
        rows = registry.records()
        assert {row["experiment_name"] for row in rows} == {"s4_acceptance"}
        assert all(row["provenance_stamp"] == code_provenance_stamp() for row in rows)


class TestPerRunLogging:
    def test_every_fresh_run_writes_experiment_log_with_lifecycle_records(
        self, tmp_path
    ):
        run_experiment(_experiment(), root=tmp_path)
        for run_dir in _run_dirs(tmp_path):
            log_path = run_dir / RUN_LOG_FILENAME
            assert log_path.exists()
            content = log_path.read_text()
            assert "started" in content  # contender start
            assert "Online evaluation finished" in content
            assert "cache key" in content

    def test_a_failed_run_writes_its_own_traceback_before_release(self, tmp_path):
        """The T3.k failure contract: a dead run's folder explains itself."""

        class _ExplodingRecipe:
            family = "exploding"
            label = "exploding"

            def get_signature(self):
                return {"type": "_ExplodingRecipe", "family": self.family}

            def build_synthesizer(self, problem, harness):
                raise RuntimeError("synthetic synthesis failure")

        experiment = _experiment(
            ContenderSpec.from_recipe(_ExplodingRecipe()), name="failing"
        )
        with pytest.raises(RuntimeError, match="synthetic synthesis failure"):
            run_experiment(experiment, root=tmp_path)

        (run_dir,) = _run_dirs(tmp_path)
        content = (run_dir / RUN_LOG_FILENAME).read_text()
        assert "Traceback" in content
        assert "synthetic synthesis failure" in content

    def test_no_root_handler_leaks_after_execution(self, tmp_path):
        root_logger = logging.getLogger()
        before = list(root_logger.handlers)
        run_experiment(_experiment(RICCATI), root=tmp_path)
        assert root_logger.handlers == before


class TestArtifactFormats:
    def test_cached_runs_hold_no_pickle_bytes(self, tmp_path):
        """The T3.i law on the cache's own storage: every artifact of every
        experiment-layer run rides an approved, code-execution-free format."""
        run_experiment(_experiment(RICCATI, UNFOLDED, TRAINED), root=tmp_path)
        artifact_files = list(tmp_path.glob("*/artifacts/*"))
        assert artifact_files, "expected persisted artifacts"
        allowed = {".npz", ".npy", ".safetensors", ".csv", ".json", ".png"}
        for path in artifact_files:
            assert path.suffix in allowed, path

    def test_stamp_is_recorded_in_run_metadata(self, tmp_path):
        run_experiment(_experiment(RICCATI), root=tmp_path)
        (run_dir,) = _run_dirs(tmp_path)
        params = load_run_metadata(run_dir)["params"]
        assert params[STAMP_PARAM] == code_provenance_stamp()
