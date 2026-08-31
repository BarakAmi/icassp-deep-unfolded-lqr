"""NB06 acceptance tests for the zero-shot-OOD `Experiment` declaration
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 3/4/8): the six
contenders resolve through `run_experiment` end to end on the real recipe/
cache/history layers, and `synthesize_nominal_contenders` (Phase 2) hands
back one in-memory artifact per contender ready for the OOD pass.

Numerical-closeness claims against production-scale training are
deliberately out of scope (mirrors `test_nb04_box_constrained.py`'s own
documented scope): this suite proves every contender executes, caches, and
produces a finite, box-feasible cost -- not that any of them has converged.
"""

import numpy as np
import pytest
import torch

from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import (
    NB06Config,
    UnfoldedModelConfig,
    build_nb06_config,
    nb06_ood_generalization_experiment,
)
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import run_experiment
from mbl.experiments.zero_shot import synthesize_nominal_contenders

_EXPECTED_LABELS_TO_FAMILIES = {
    "truncated_riccati": "truncated_riccati",
    "unfolded_alpha": "unfolded",
    "unfolded_alpha_p": "unfolded_warmstart",
    "cocp": "cocp",
    "neural": "neural",
    "cocp_lower_bound": "cocp_lower_bound",
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


def _fast_config(**overrides: object) -> NB06Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=6,
        u_max=0.3,
        num_unfolding_iterations=3,
        step_size_init=0.02,
        step_size_max=0.5,
        batch_size=16,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.5,
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
        cocp_plan=_fast_plan(),
        neural_hidden_dim=4,
        neural_plan=_fast_plan(),
    )
    defaults.update(overrides)
    return NB06Config(**defaults)


class TestExperimentDeclaration:
    def test_declares_six_contenders_with_expected_families(self) -> None:
        experiment = nb06_ood_generalization_experiment(_fast_config())
        families = {spec.resolved_label: spec.family for spec in experiment.contenders}
        assert families == _EXPECTED_LABELS_TO_FAMILIES

    def test_problem_carries_the_configured_box_constraint(self) -> None:
        experiment = nb06_ood_generalization_experiment(_fast_config(u_max=0.7))
        problem = experiment.problem.build()
        assert problem.constraints is not None
        assert len(problem.constraints) == 1

    def test_ctx_honors_the_configured_dtype(self) -> None:
        experiment = nb06_ood_generalization_experiment(
            _fast_config(dtype=torch.float32)
        )
        assert experiment.ctx.precision.value == "float32"

    def test_non_positive_u_max_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="u_max"):
            _fast_config(u_max=0.0)

    def test_neural_contender_carries_hidden_dim_and_init_seed(self) -> None:
        experiment = nb06_ood_generalization_experiment(
            _fast_config(neural_hidden_dim=8, neural_init_seed=3)
        )
        spec = next(s for s in experiment.contenders if s.resolved_label == "neural")
        assert spec.config["hidden_dim"] == 8
        assert spec.config["init_seed"] == 3


class TestBuildNb06Config:
    def test_flat_builder_matches_the_dataclass_constructor(self) -> None:
        via_builder = build_nb06_config(
            state_dim=3,
            control_dim=2,
            horizon=6,
            u_max=0.3,
            num_unfolding_iterations=3,
            step_size_init=0.02,
            step_size_max=0.5,
            batch_size=16,
            n_eval_batches=2,
            seed=0,
            process_noise_std=0.5,
            unfolded_alpha=UnfoldedModelConfig(kind=UnfoldedKind.LEARNED_STEP_SIZE),
            unfolded_alpha_p=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX
            ),
            cocp_plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=4),
            neural_hidden_dim=4,
            neural_plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=4),
        )
        assert via_builder.u_max == 0.3
        assert via_builder.state_dim == 3
        assert via_builder.neural_hidden_dim == 4


class TestRunExperimentEndToEnd:
    def test_every_contender_executes_and_scores_a_finite_box_feasible_cost(
        self, tmp_path
    ) -> None:
        experiment = nb06_ood_generalization_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        assert set(report.results) == set(_EXPECTED_LABELS_TO_FAMILIES)
        for label, result in report.results.items():
            cost = result.metrics["eval_expected_cost"]
            assert np.isfinite(cost), f"{label} produced a non-finite cost"
            assert cost >= 0.0

    def test_synthesize_nominal_contenders_returns_all_six_artifacts(
        self, tmp_path
    ) -> None:
        experiment = nb06_ood_generalization_experiment(_fast_config())
        artifacts, report = synthesize_nominal_contenders(experiment, root=tmp_path)

        assert set(artifacts) == set(_EXPECTED_LABELS_TO_FAMILIES)
        assert set(report.results) == set(_EXPECTED_LABELS_TO_FAMILIES)
        for artifact in artifacts.values():
            policy = artifact.make_policy()
            assert callable(policy)
