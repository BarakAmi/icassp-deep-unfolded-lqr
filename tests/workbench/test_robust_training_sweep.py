"""Phase 4 acceptance tests for `workbench.robust_training_sweep`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 5.4/11): `n_seeds < 5` raises;
a cache miss then hit reproduces identical points without retraining; the
fairness law (identical realizations at a given (level, seed)) holds by
construction of the condition builder.

Uses only cheap, non-learnable `AnalyticRecipe` contenders (Truncated-Riccati,
Standard-PGD) so the engine itself is exercised independent of the (slower)
learned families the NB07 study module wires in.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes.analytic import TruncatedRiccatiRecipe
from mbl.applications.recipes.base import EngineHarness
from mbl.applications.recipes.unfolded import FixedUnfoldedRecipe
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.experiments.experiment import EvaluationBatch
from mbl.persistence.null_tracker import NullExperimentTracker
from mbl.workbench.robust_training_sweep import (
    ModelAccess,
    RobustTrainingSpec,
    TrainingCondition,
    UncertaintyAxis,
    run_robust_training_sweep,
)

_STATE_DIM = 4
_CONTROL_DIM = 2
_HORIZON = 10
_U_MAX = 0.5
_BATCH_SIZE = 16
_N_EVAL_BATCHES = 2


def _ctx() -> ComputeContext:
    return ComputeContext(
        backend=Backend.TORCH,
        device="cpu",
        precision=Precision.from_torch_dtype(torch.float64),
    )


def _cheap_recipes(level: float) -> dict:
    del level
    return {
        "truncated_riccati": TruncatedRiccatiRecipe(horizon=_HORIZON),
        "standard_pgd": FixedUnfoldedRecipe(
            num_iterations=2,
            step_size_init=0.05,
            step_size_max=1.0,
            horizon=_HORIZON,
            label="standard_pgd",
        ),
    }


def _draw_batches(spec, ctx: ComputeContext, n: int) -> tuple[EvaluationBatch, ...]:
    sampler, _ = spec.build(Backend.TORCH, torch_dtype=ctx.torch_dtype)
    return tuple(sampler() for _ in range(n))


def _condition_at(level: float, seed: int) -> TrainingCondition:
    problem = LQRProblemFactory(
        state_dim=_STATE_DIM,
        control_dim=_CONTROL_DIM,
        horizon=_HORIZON,
        seed=0,
        u_max=_U_MAX,
    ).build()
    ctx = _ctx()
    train_spec = GaussianBatchSpec(
        state_dim=_STATE_DIM,
        horizon=_HORIZON,
        batch_size=_BATCH_SIZE,
        seed=seed,
        process_noise_std=0.5 * level,
    )
    eval_spec = GaussianBatchSpec(
        state_dim=_STATE_DIM,
        horizon=_HORIZON,
        batch_size=_BATCH_SIZE,
        seed=999,
        process_noise_std=0.5 * level,
    )
    return TrainingCondition(
        training_problem=problem,
        training_harness=EngineHarness(
            batch_spec=train_spec, tracker=NullExperimentTracker(), ctx=ctx
        ),
        eval_problem=problem,
        eval_batches=_draw_batches(eval_spec, ctx, _N_EVAL_BATCHES),
    )


def _nominal(ctx: ComputeContext) -> tuple[object, tuple[EvaluationBatch, ...]]:
    problem = LQRProblemFactory(
        state_dim=_STATE_DIM,
        control_dim=_CONTROL_DIM,
        horizon=_HORIZON,
        seed=0,
        u_max=_U_MAX,
    ).build()
    spec = GaussianBatchSpec(
        state_dim=_STATE_DIM,
        horizon=_HORIZON,
        batch_size=_BATCH_SIZE,
        seed=1234,
        process_noise_std=0.5,
    )
    return problem, _draw_batches(spec, ctx, _N_EVAL_BATCHES)


class TestIngressGate:
    def test_n_seeds_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="n_seeds"):
            RobustTrainingSpec(
                axis=UncertaintyAxis.NOISE_SCALE,
                levels=(1.0,),
                model_access=ModelAccess.NOMINAL,
                n_seeds=4,
                base_seed=0,
            )

    def test_minimum_seeds_is_accepted(self) -> None:
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0,),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        assert spec.n_seeds == 5


class TestSweepExecutesAndCaches:
    def test_produces_a_point_per_label_per_level_per_seed(
        self, tmp_path: Path
    ) -> None:
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0, 2.0),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        result = run_robust_training_sweep(
            _cheap_recipes,
            _condition_at,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        assert set(result.points) == {"truncated_riccati", "standard_pgd"}
        for label in result.points:
            assert len(result.points[label]) == 2 * 5  # 2 levels x 5 seeds
            for point in result.points[label]:
                assert point.disposition == "fresh"
                assert np.isfinite(point.trained_metrics.cost_mean)
                assert np.isfinite(point.nominal_metrics.cost_mean)

    def test_second_run_is_all_cache_hits(self, tmp_path: Path) -> None:
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0,),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        first = run_robust_training_sweep(
            _cheap_recipes,
            _condition_at,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        second = run_robust_training_sweep(
            _cheap_recipes,
            _condition_at,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        for label in second.points:
            for point in second.points[label]:
                assert point.disposition == "hit"
        # Cached numbers reproduce the freshly-computed ones exactly.
        for label in first.points:
            for fresh_point, hit_point in zip(
                first.points[label], second.points[label]
            ):
                assert fresh_point.trained_metrics.cost_mean == pytest.approx(
                    hit_point.trained_metrics.cost_mean
                )
                assert fresh_point.learning_curve == hit_point.learning_curve

    def test_readonly_policy_raises_on_miss(self, tmp_path: Path) -> None:
        from mbl.experiments.cache import CachePolicy, CacheReadOnlyMissError

        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0,),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        with pytest.raises(CacheReadOnlyMissError):
            run_robust_training_sweep(
                _cheap_recipes,
                _condition_at,
                spec,
                ctx=ctx,
                u_max=_U_MAX,
                nominal_problem=nominal_problem,
                nominal_batches=nominal_batches,
                root=tmp_path,
                policy=CachePolicy("readonly"),
            )


class TestOracleUnsupportedResyncException:
    """The plan's declared COCP exception (Sec 4.2): a contender whose
    controller has no ORACLE resynchronizer (COCP-backed) trains NOMINAL for
    itself instead of raising and aborting the whole sweep, while every
    resync-supporting contender in the SAME sweep still trains ORACLE."""

    def _recipes_at(self, level: float) -> dict:
        from mbl.applications.recipes.cocp import COCPLowerBoundRecipe

        del level
        return {
            "truncated_riccati": TruncatedRiccatiRecipe(horizon=_HORIZON),
            "cocp_lower_bound": COCPLowerBoundRecipe(process_noise_std=0.5),
        }

    def _condition_with_dr(self, level: float, seed: int) -> TrainingCondition:
        from mbl.applications.uncertainty.perturbations import (
            PerturbationDistribution,
            PerturbationKind,
            PlantPerturbation,
        )

        base = _condition_at(level, seed)
        distribution = PerturbationDistribution(
            perturbation=PlantPerturbation(
                kind=PerturbationKind.ADDITIVE, level=0.2, seed=seed
            ),
            resample_per_epoch=True,
        )
        return TrainingCondition(
            training_problem=base.training_problem,
            training_harness=base.training_harness,
            eval_problem=base.eval_problem,
            eval_batches=base.eval_batches,
            dr_distribution=distribution,
        )

    def test_cocp_lower_bound_downgrades_but_riccati_stays_oracle(
        self, tmp_path: Path
    ) -> None:
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.PLANT_ADDITIVE,
            levels=(0.2,),
            model_access=ModelAccess.ORACLE,
            n_seeds=5,
            base_seed=0,
        )
        result = run_robust_training_sweep(
            self._recipes_at,
            self._condition_with_dr,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        assert len(result.points["cocp_lower_bound"]) == 5
        for point in result.points["cocp_lower_bound"]:
            assert point.effective_model_access == ModelAccess.NOMINAL
        for point in result.points["truncated_riccati"]:
            assert point.effective_model_access == ModelAccess.ORACLE

    def test_effective_access_survives_a_cache_hit(self, tmp_path: Path) -> None:
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.PLANT_ADDITIVE,
            levels=(0.2,),
            model_access=ModelAccess.ORACLE,
            n_seeds=5,
            base_seed=0,
        )
        run_robust_training_sweep(
            self._recipes_at,
            self._condition_with_dr,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        second = run_robust_training_sweep(
            self._recipes_at,
            self._condition_with_dr,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        for point in second.points["cocp_lower_bound"]:
            assert point.disposition == "hit"
            assert point.effective_model_access == ModelAccess.NOMINAL
        for point in second.points["truncated_riccati"]:
            assert point.disposition == "hit"
            assert point.effective_model_access == ModelAccess.ORACLE


class TestNominalOverride:
    """The TRAIN_HORIZON axis's own crash reproduction: a contender sized
    for its OWN level's horizon has a gain/refinement stack of exactly that
    length, so scoring it against the sweep's GLOBAL nominal problem (a
    DIFFERENT horizon) indexes past the end of that stack.
    `nominal_override` supplies a horizon-matched reference instead."""

    # Deliberately SHORTER than _HORIZON=10 above: the crash this reproduces
    # only occurs when the controller's own (shorter) gain stack is indexed
    # past its end by a LONGER rollout -- the reverse mismatch (a longer-
    # horizon controller run for fewer steps than it has gains for) never
    # indexes out of bounds at all.
    _OTHER_HORIZON = 5

    def _recipes_at(self, level: float) -> dict:
        del level
        return {
            "truncated_riccati": TruncatedRiccatiRecipe(horizon=self._OTHER_HORIZON)
        }

    def _condition_at(self, level: float, seed: int) -> TrainingCondition:
        del level
        ctx = _ctx()
        problem = LQRProblemFactory(
            state_dim=_STATE_DIM,
            control_dim=_CONTROL_DIM,
            horizon=self._OTHER_HORIZON,
            seed=0,
            u_max=_U_MAX,
        ).build()
        train_spec = GaussianBatchSpec(
            state_dim=_STATE_DIM,
            horizon=self._OTHER_HORIZON,
            batch_size=_BATCH_SIZE,
            seed=seed,
            process_noise_std=0.5,
        )
        eval_spec = GaussianBatchSpec(
            state_dim=_STATE_DIM,
            horizon=self._OTHER_HORIZON,
            batch_size=_BATCH_SIZE,
            seed=999,
            process_noise_std=0.5,
        )
        eval_batches = _draw_batches(eval_spec, ctx, _N_EVAL_BATCHES)
        return TrainingCondition(
            training_problem=problem,
            training_harness=EngineHarness(
                batch_spec=train_spec, tracker=NullExperimentTracker(), ctx=ctx
            ),
            eval_problem=problem,
            eval_batches=eval_batches,
            nominal_override=(problem, eval_batches),
        )

    def test_horizon_mismatched_nominal_problem_would_crash_without_override(
        self, tmp_path: Path
    ) -> None:
        """Sanity check on the reproduction itself: WITHOUT `nominal_override`,
        scoring this horizon-mismatched artifact against the (shorter-horizon)
        global nominal problem raises -- confirming the test actually
        exercises the bug `nominal_override` fixes, not a vacuous scenario."""
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(
            ctx
        )  # _HORIZON=10, shorter than _OTHER_HORIZON
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.TRAIN_HORIZON,
            levels=(1.0,),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )

        def _condition_without_override(level: float, seed: int) -> TrainingCondition:
            return dataclasses.replace(
                self._condition_at(level, seed), nominal_override=None
            )

        with pytest.raises(IndexError):
            run_robust_training_sweep(
                self._recipes_at,
                _condition_without_override,
                spec,
                ctx=ctx,
                u_max=_U_MAX,
                nominal_problem=nominal_problem,
                nominal_batches=nominal_batches,
                root=tmp_path,
            )

    def test_nominal_override_avoids_the_crash_and_matches_trained_metrics(
        self, tmp_path: Path
    ) -> None:
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.TRAIN_HORIZON,
            levels=(1.0,),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        result = run_robust_training_sweep(
            self._recipes_at,
            self._condition_at,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )
        for point in result.points["truncated_riccati"]:
            assert np.isfinite(point.nominal_metrics.cost_median)
            # nominal_override reuses the SAME (problem, batches) as
            # eval_problem/eval_batches, so the two scored passes must agree.
            assert point.nominal_metrics.cost_median == pytest.approx(
                point.trained_metrics.cost_median
            )


class TestFairnessLaw:
    def test_identical_condition_at_same_level_seed_gives_identical_eval_batches(
        self,
    ) -> None:
        condition_a = _condition_at(1.0, 3)
        condition_b = _condition_at(1.0, 3)
        for batch_a, batch_b in zip(condition_a.eval_batches, condition_b.eval_batches):
            for tensor_a, tensor_b in zip(batch_a, batch_b):
                torch.testing.assert_close(tensor_a, tensor_b, atol=0, rtol=0)


class TestBandAggregation:
    def _result(self, tmp_path: Path):
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0, 2.0),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        return run_robust_training_sweep(
            _cheap_recipes,
            _condition_at,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )

    def test_cost_band_has_one_entry_per_level(self, tmp_path: Path) -> None:
        result = self._result(tmp_path)
        band = result.cost_band("truncated_riccati", on="trained")
        assert band.x == (1.0, 2.0)
        assert len(band.median) == len(band.q25) == len(band.q75) == 2
        for q25, median, q75 in zip(band.q25, band.median, band.q75):
            assert q25 <= median <= q75

    def test_cost_band_rejects_unknown_on(self, tmp_path: Path) -> None:
        result = self._result(tmp_path)
        with pytest.raises(ValueError, match="on"):
            result.cost_band("truncated_riccati", on="bogus")

    def test_nominal_band_differs_in_shape_not_error(self, tmp_path: Path) -> None:
        result = self._result(tmp_path)
        band = result.cost_band("standard_pgd", on="nominal")
        assert band.x == (1.0, 2.0)

    def test_saturation_band_values_in_unit_interval(self, tmp_path: Path) -> None:
        result = self._result(tmp_path)
        band = result.saturation_band("standard_pgd")
        for value in band.median:
            assert 0.0 <= value <= 1.0

    def test_learning_curve_band_has_one_entry_per_epoch(self, tmp_path: Path) -> None:
        result = self._result(tmp_path)
        band = result.learning_curve_band("truncated_riccati", 1.0)
        # AnalyticRecipe trains for exactly one epoch (AnalyticalStrategy).
        assert band.x == (1,)
        assert len(band.median) == 1

    def test_isotropy_band_is_empty_for_families_with_no_learned_p(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        band = result.isotropy_band("truncated_riccati")
        assert band.x == ()
        assert band.median == ()

    def test_spectral_norm_band_is_empty_for_families_with_no_learned_p(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        band = result.spectral_norm_band("truncated_riccati")
        assert band.x == ()
        assert band.median == ()

    def test_condition_number_band_is_empty_for_families_with_no_learned_p(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        band = result.condition_number_band("truncated_riccati")
        assert band.x == ()
        assert band.median == ()

    def test_eigenvalue_bands_is_empty_tuple_for_families_with_no_learned_p(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        assert result.eigenvalue_bands("truncated_riccati") == ()

    def test_principal_angle_band_is_empty_for_families_with_no_learned_p(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        band = result.principal_angle_band("truncated_riccati", np.eye(_STATE_DIM))
        assert band.x == ()
        assert band.median == ()


class TestSpectralBandsWithLearnedP:
    """`unfolded_alpha_p` is the one family that reports `learned_p` -- the
    non-trivial counterpart to `TestBandAggregation`'s empty-family checks
    above."""

    def _recipes_at(self, level: float) -> dict:
        from mbl.applications.recipes.unfolded import UnfoldedKind, UnfoldedRecipe
        from mbl.engine.training_plan import OptimizerSpec, TrainingPlan

        del level
        plan = TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=2)
        return {
            "unfolded_alpha_p": UnfoldedRecipe(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                plan=plan,
                label="unfolded_alpha_p",
                num_iterations=2,
                step_size_init=0.05,
                step_size_max=1.0,
                horizon=_HORIZON,
            ),
        }

    def _result(self, tmp_path: Path):
        ctx = _ctx()
        nominal_problem, nominal_batches = _nominal(ctx)
        spec = RobustTrainingSpec(
            axis=UncertaintyAxis.NOISE_SCALE,
            levels=(1.0, 2.0),
            model_access=ModelAccess.NOMINAL,
            n_seeds=5,
            base_seed=0,
        )
        return run_robust_training_sweep(
            self._recipes_at,
            _condition_at,
            spec,
            ctx=ctx,
            u_max=_U_MAX,
            nominal_problem=nominal_problem,
            nominal_batches=nominal_batches,
            root=tmp_path,
        )

    def test_spectral_norm_and_condition_number_bands_are_populated(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        norm_band = result.spectral_norm_band("unfolded_alpha_p")
        cond_band = result.condition_number_band("unfolded_alpha_p")
        assert norm_band.x == (1.0, 2.0)
        assert all(value > 0 for value in norm_band.median)
        assert cond_band.x == (1.0, 2.0)
        assert all(value >= 1.0 for value in cond_band.median)

    def test_eigenvalue_bands_has_one_band_per_state_dim(self, tmp_path: Path) -> None:
        result = self._result(tmp_path)
        bands = result.eigenvalue_bands("unfolded_alpha_p")
        assert len(bands) == _STATE_DIM
        for band in bands:
            assert band.x == (1.0, 2.0)

    def test_principal_angle_band_against_identity_is_in_valid_range(
        self, tmp_path: Path
    ) -> None:
        result = self._result(tmp_path)
        band = result.principal_angle_band("unfolded_alpha_p", np.eye(_STATE_DIM))
        assert band.x == (1.0, 2.0)
        for value in band.median:
            assert 0.0 <= value <= 90.0
