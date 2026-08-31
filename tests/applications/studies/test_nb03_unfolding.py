"""NB03 acceptance tests for the `Experiment` declaration (blueprint v2 SS2):
the four contenders resolve through `run_experiment` end to end, on the real
recipe/cache/history layers -- A-8's PIPELINE half.

The tight numerical convergence claim (the learned unfolded contenders
closing the gap to Theo-Riccati within `J` unfolding iterations, while
Standard-GD does not) is a production-scale, many-epoch empirical result
demonstrated in the notebook itself (blueprint SS10/SS11), not a fast CI
assertion -- training to genuine convergence takes far longer than a unit
test budget allows, and asserting a tight numerical bound at reduced
epoch/dimension counts would be either vacuous or flaky. This suite instead
proves the four contenders correctly EXECUTE, CACHE, and produce finite,
non-degenerate costs -- including confirming the flagship layer-wise
contender's multi-phase synthesis genuinely ran and moved its parameters.
"""

import math

import pytest
import torch

from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import (
    NB03Config,
    UnfoldedModelConfig,
    nb03_unfolding_experiment,
)
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import CachePolicy, run_experiment

_EXPECTED_LABELS_TO_FAMILIES = {
    "sim_riccati": "riccati",
    "standard_gd": "unfolded_fixed",
    "unfolded_alpha": "unfolded",
    "unfolded_alpha_p": "unfolded_warmstart",
}


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=4)


def _fast_schedule() -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=2,
        refinement_epochs=2,
        train_matrix_from="refinement",
    )


def _fast_config(**overrides: object) -> NB03Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=6,
        num_unfolding_iterations=3,
        step_size_init=0.02,
        step_size_max=0.5,
        batch_size=16,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.3,
        unfolded_alpha=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE,
            training_mode="end_to_end",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
        unfolded_alpha_p=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            training_mode="layerwise",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
    )
    defaults.update(overrides)
    return NB03Config(**defaults)


class TestExperimentDeclaration:
    def test_declares_four_contenders_with_expected_families(self) -> None:
        experiment = nb03_unfolding_experiment(_fast_config())
        families = {spec.resolved_label: spec.family for spec in experiment.contenders}
        assert families == _EXPECTED_LABELS_TO_FAMILIES

    def test_ctx_honors_the_configured_dtype(self) -> None:
        experiment = nb03_unfolding_experiment(_fast_config(dtype=torch.float32))
        assert experiment.ctx.precision.value == "float32"

    def test_ctx_defaults_to_cpu_device(self) -> None:
        experiment = nb03_unfolding_experiment(_fast_config())
        assert experiment.ctx.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_ctx_honors_the_configured_device(self) -> None:
        experiment = nb03_unfolding_experiment(_fast_config(device="cuda"))
        assert experiment.ctx.device == "cuda:0"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_changing_device_changes_the_experiment_signature(self) -> None:
        """Phase 1 directive 1's architecture rule: a CUDA run and a CPU run
        are fundamentally different profiling experiments, so the device
        must be folded into the cache signature -- never served
        interchangeably from the same cache entry."""
        cpu_experiment = nb03_unfolding_experiment(_fast_config(device="cpu"))
        cuda_experiment = nb03_unfolding_experiment(_fast_config(device="cuda"))

        assert cpu_experiment.signature_digest() != cuda_experiment.signature_digest()
        for spec in cpu_experiment.contenders:
            assert cpu_experiment.contender_content_digest(
                spec
            ) != cuda_experiment.contender_content_digest(spec)

    def test_evaluation_draws_multiple_batches_when_configured(self) -> None:
        experiment = nb03_unfolding_experiment(_fast_config(n_eval_batches=3))
        assert experiment.evaluation.n_batches == 3


class TestTrainingModeRouting:
    """Each learned model routes to the layer-wise or the end-to-end regime
    per its OWN `UnfoldedModelConfig.training_mode` -- a free per-model choice
    (not fixed to a particular model) -- always under the SAME label and the
    SAME learned kind; only the training family/schedule differs."""

    def _flagship(self, mode: str):
        config = _fast_config(
            unfolded_alpha_p=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                training_mode=mode,
                plan=_fast_plan(),
                schedule=_fast_schedule(),
            )
        )
        experiment = nb03_unfolding_experiment(config)
        (flagship,) = (
            spec
            for spec in experiment.contenders
            if spec.resolved_label == "unfolded_alpha_p"
        )
        return flagship

    def test_layerwise_mode_routes_to_the_warmstart_family(self) -> None:
        assert self._flagship("layerwise").family == "unfolded_warmstart"

    def test_end_to_end_mode_routes_to_the_flat_unfolded_family(self) -> None:
        flagship = self._flagship("end_to_end")
        assert flagship.family == "unfolded"
        # Same label AND same learned kind -- only the training regime changed.
        assert flagship.config["kind"] is UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX

    def test_the_two_modes_are_distinct_experiments(self) -> None:
        layerwise = nb03_unfolding_experiment(
            _fast_config(
                unfolded_alpha_p=UnfoldedModelConfig(
                    kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                    training_mode="layerwise",
                    plan=_fast_plan(),
                    schedule=_fast_schedule(),
                )
            )
        )
        end_to_end = nb03_unfolding_experiment(
            _fast_config(
                unfolded_alpha_p=UnfoldedModelConfig(
                    kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                    training_mode="end_to_end",
                    plan=_fast_plan(),
                    schedule=_fast_schedule(),
                )
            )
        )
        assert layerwise.signature_digest() != end_to_end.signature_digest()

    def test_unfolded_alpha_can_also_run_layerwise(self) -> None:
        """Directive: the layer-wise regime is NOT reserved for the flagship
        -- Unfolded-alpha (no learned matrix) can run it too, the layer-wise
        machinery simply skipping the matrix-activation step."""
        config = _fast_config(
            unfolded_alpha=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE,
                training_mode="layerwise",
                plan=_fast_plan(),
                schedule=_fast_schedule(),
            )
        )
        experiment = nb03_unfolding_experiment(config)
        (alpha,) = (
            spec
            for spec in experiment.contenders
            if spec.resolved_label == "unfolded_alpha"
        )
        assert alpha.family == "unfolded_warmstart"
        assert alpha.config["kind"] is UnfoldedKind.LEARNED_STEP_SIZE


class TestEndToEndExecution:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_every_contender_executes_on_cuda_without_a_device_mismatch(
        self, tmp_path
    ) -> None:
        """Regression: every contender's evaluation/training batches and the
        unfolded controller's own static tensors (A/B/R, M/C stacks, learned
        parameters) must all resolve onto the SAME device as the configured
        ComputeContext -- a torch device-mismatch RuntimeError anywhere in
        this path means the hardware transparency device selector (Phase 1
        directive 1) is cosmetic, not functional."""
        experiment = nb03_unfolding_experiment(_fast_config(device="cuda"))
        report = run_experiment(experiment, root=tmp_path)

        assert set(report.results) == set(_EXPECTED_LABELS_TO_FAMILIES)
        for label, result in report.results.items():
            cost = result.metrics["eval_expected_cost"]
            assert math.isfinite(cost) and cost > 0, (
                f"{label}: non-finite/non-positive expected cost {cost}"
            )

    def test_every_contender_executes_and_produces_a_finite_cost(
        self, tmp_path
    ) -> None:
        experiment = nb03_unfolding_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        assert set(report.results) == set(_EXPECTED_LABELS_TO_FAMILIES)
        for label, result in report.results.items():
            cost = result.metrics["eval_expected_cost"]
            assert math.isfinite(cost) and cost > 0, (
                f"{label}: non-finite/non-positive expected cost {cost}"
            )
            assert result.disposition == "fresh"
            assert result.family == _EXPECTED_LABELS_TO_FAMILIES[label]

    def test_second_run_is_served_entirely_from_cache(self, tmp_path) -> None:
        experiment = nb03_unfolding_experiment(_fast_config())
        run_experiment(experiment, root=tmp_path)

        report = run_experiment(experiment, root=tmp_path, policy=CachePolicy("reuse"))

        assert {r.disposition for r in report.results.values()} == {"hit"}

    def test_warmstart_contender_genuinely_trained(self, tmp_path) -> None:
        """A-8's pipeline proof for the flagship build specifically: the
        layer-wise contender's synthesis actually ran its compiled phases
        (a finite final training loss is recorded), not silently skipped."""
        experiment = nb03_unfolding_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        result = report.results["unfolded_alpha_p"]
        assert "final_loss" in result.metrics
        assert math.isfinite(result.metrics["final_loss"])

    def test_standard_unfolded_contender_genuinely_trained(self, tmp_path) -> None:
        experiment = nb03_unfolding_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        result = report.results["unfolded_alpha"]
        assert "final_loss" in result.metrics
        assert math.isfinite(result.metrics["final_loss"])

    def test_analytic_contenders_carry_no_training_loss(self, tmp_path) -> None:
        """Sim-Riccati and Standard-GD are non-learnable: their synthesis is
        construction only, so they must not report a spurious training loss."""
        experiment = nb03_unfolding_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        assert "final_loss" not in report.results["sim_riccati"].metrics
        assert "final_loss" not in report.results["standard_gd"].metrics
