"""Phase 4 acceptance tests for `applications.studies.nb07_robust_training`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 5.5/11): six training-sweep
contenders exclude `cocp`; the nominal-anchor experiment declares seven and
actually runs through `run_experiment`; every condition builder's degeneracy
point reproduces the nominal instance; a tiny end-to-end sweep over the real
six contenders executes without error.
"""

from pathlib import Path

import numpy as np
import pytest

from mbl.applications.recipes.unfolded import UnfoldedKind
from mbl.applications.studies.nb03_unfolding import UnfoldedModelConfig
from mbl.applications.studies.nb07_robust_training import (
    NB07Config,
    build_nb07_config,
    nb07_compute_context,
    nb07_nominal_anchor_experiment,
    nominal_problem_and_batches,
    noise_scale_condition_builder,
    plant_additive_condition_builder,
    plant_rotation_condition_builder,
    recipes_at_for,
    six_training_sweep_contenders,
    train_horizon_condition_builder,
)
from mbl.core.system.linear_system import TimeSeriesMatrix
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import CachePolicy, run_experiment
from mbl.models.guards import require_linear_quadratic
from mbl.workbench.robust_training_sweep import (
    ModelAccess,
    RobustTrainingSpec,
    UncertaintyAxis,
    run_robust_training_sweep,
)

_TINY_HORIZON = 6


def _tiny_plan(epochs: int = 3) -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=epochs)


def _tiny_config(**overrides: object) -> NB07Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=_TINY_HORIZON,
        u_max=0.5,
        num_unfolding_iterations=2,
        step_size_init=0.05,
        step_size_max=1.0,
        batch_size=8,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.3,
        unfolded_alpha=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE, plan=_tiny_plan()
        ),
        unfolded_alpha_p=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
            training_mode="end_to_end",
            plan=_tiny_plan(),
        ),
        gru_hidden_dim=4,
        gru_plan=_tiny_plan(),
        cocp_plan=_tiny_plan(epochs=2),
        noise_scale_multipliers=(1.0, 2.0),
        noise_families=(),
        plant_additive_epsilons=(0.0, 0.2),
        plant_rotation_degrees=(0.0, 30.0),
        train_horizons=(_TINY_HORIZON, 2 * _TINY_HORIZON),
        sample_sizes=(8, 16),
        n_training_seeds=5,
    )
    defaults.update(overrides)
    return build_nb07_config(**defaults)


class TestSixContendersExcludeCocp:
    def test_exactly_six_labels_no_cocp(self) -> None:
        cfg = _tiny_config()
        recipes = six_training_sweep_contenders(cfg)
        assert set(recipes) == {
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "cocp_lower_bound",
            "neural",
        }
        assert "cocp" not in recipes

    def test_horizon_override_propagates(self) -> None:
        cfg = _tiny_config()
        recipes = six_training_sweep_contenders(cfg, horizon=99)
        assert recipes["truncated_riccati"].horizon == 99
        assert recipes["standard_pgd"].horizon == 99


class TestNominalAnchorExperiment:
    def test_declares_seven_contenders_including_cocp(self) -> None:
        cfg = _tiny_config()
        experiment = nb07_nominal_anchor_experiment(cfg)
        labels = {spec.resolved_label for spec in experiment.contenders}
        assert labels == {
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "cocp_lower_bound",
            "neural",
            "cocp",
        }

    def test_runs_end_to_end_at_tiny_scale(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        experiment = nb07_nominal_anchor_experiment(cfg)
        report = run_experiment(
            experiment, root=tmp_path, policy=CachePolicy("recompute")
        )
        for label, result in report.results.items():
            cost = result.metrics.get("eval_expected_cost")
            assert cost is not None and np.isfinite(cost), label


class TestConditionBuilderDegeneracy:
    """Each axis's degeneracy point must reproduce the nominal instance."""

    def test_noise_scale_multiplier_one_matches_nominal_process_std(self) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        condition_at = noise_scale_condition_builder(cfg, ctx)
        condition = condition_at(1.0, seed=7)
        x0, w, _ = condition.eval_batches[0]
        assert float(w.std()) == pytest.approx(cfg.process_noise_std, rel=0.2)
        assert condition.dr_distribution is None

    def test_plant_additive_zero_level_reproduces_nominal_plant(self) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        condition_at = plant_additive_condition_builder(cfg, ctx)
        condition = condition_at(0.0, seed=3)
        nominal_system, _ = require_linear_quadratic(condition.training_problem)
        eval_system, _ = require_linear_quadratic(condition.eval_problem)
        np.testing.assert_array_equal(eval_system.A_t.array, nominal_system.A_t.array)
        np.testing.assert_array_equal(eval_system.B_t.array, nominal_system.B_t.array)

    def test_plant_rotation_zero_degrees_reproduces_nominal_plant(self) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        condition_at = plant_rotation_condition_builder(cfg, ctx)
        condition = condition_at(0.0, seed=3)
        nominal_system, _ = require_linear_quadratic(condition.training_problem)
        eval_system, _ = require_linear_quadratic(condition.eval_problem)
        np.testing.assert_array_equal(eval_system.A_t.array, nominal_system.A_t.array)
        np.testing.assert_array_equal(eval_system.B_t.array, nominal_system.B_t.array)

    def test_plant_rotation_has_a_fixed_dr_distribution(self) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        condition_at = plant_rotation_condition_builder(cfg, ctx)
        condition = condition_at(30.0, seed=1)
        assert condition.dr_distribution is not None
        assert condition.dr_distribution.resample_per_epoch is False

    def test_plant_additive_resamples_per_epoch(self) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        condition_at = plant_additive_condition_builder(cfg, ctx)
        condition = condition_at(0.2, seed=1)
        assert condition.dr_distribution is not None
        assert condition.dr_distribution.resample_per_epoch is True

    def test_train_horizon_matches_the_level(self) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        condition_at = train_horizon_condition_builder(cfg, ctx)
        condition = condition_at(2 * _TINY_HORIZON, seed=0)
        system, _ = require_linear_quadratic(condition.training_problem)
        assert isinstance(system.A_t, TimeSeriesMatrix)
        x0, w, _ = condition.eval_batches[0]
        assert w.shape[1] == 2 * _TINY_HORIZON


class TestRecipesAtFor:
    def test_non_horizon_axis_returns_the_same_dict_every_level(self) -> None:
        cfg = _tiny_config()
        recipes_at = recipes_at_for(cfg, UncertaintyAxis.NOISE_SCALE)
        recipes_1 = recipes_at(1.0)
        recipes_2 = recipes_at(2.0)
        assert recipes_1 is recipes_2

    def test_horizon_axis_rebuilds_per_level(self) -> None:
        cfg = _tiny_config()
        recipes_at = recipes_at_for(cfg, UncertaintyAxis.TRAIN_HORIZON)
        recipes_a = recipes_at(_TINY_HORIZON)
        recipes_b = recipes_at(2 * _TINY_HORIZON)
        assert recipes_a["truncated_riccati"].horizon == _TINY_HORIZON
        assert recipes_b["truncated_riccati"].horizon == 2 * _TINY_HORIZON


class TestTinyEndToEndSweep:
    def test_noise_scale_sweep_over_the_real_six_contenders(
        self, tmp_path: Path
    ) -> None:
        cfg = _tiny_config()
        ctx = nb07_compute_context(cfg)
        nominal_problem, nominal_batches = nominal_problem_and_batches(cfg, ctx)
        condition_at = noise_scale_condition_builder(cfg, ctx)
        recipes_at = recipes_at_for(cfg, UncertaintyAxis.NOISE_SCALE)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0,),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        result = run_robust_training_sweep(
            recipes_at,
            condition_at,
            spec,
            ctx=ctx,
            u_max=cfg.u_max,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        assert set(result.points) == {
            "truncated_riccati",
            "standard_pgd",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "cocp_lower_bound",
            "neural",
        }
        for label, points in result.points.items():
            assert len(points) == 5
            for point in points:
                assert np.isfinite(point.trained_metrics.cost_mean), label
                assert np.isfinite(point.nominal_metrics.cost_mean), label
        # UF-alpha+P is the only contender with a learned P.
        assert result.points["unfolded_alpha_p"][0].learned_p is not None
        assert result.points["unfolded_alpha"][0].learned_p is None
