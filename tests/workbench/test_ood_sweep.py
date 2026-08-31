"""Phase 4 acceptance tests for `workbench.ood_sweep`
(docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md Sec 3.3-3.8/8): the
statistical-rigor ingress gate (`n_seeds < 5` raises), the stochastic-
fairness law (every contender at one (level, seed) point is evaluated on
byte-identical realizations), the SDP-floor cache (solved once per distinct
problem instance, not once per seed), per-contender level subsetting (the
sparse-grid mechanism for COCP/COCP-LB), and the horizon axis's rehost
integration.
"""

import numpy as np
import pytest

import mbl.workbench.ood_sweep as ood_sweep_module
from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.ood.noise import NoiseFamily
from mbl.applications.ood.perturbations import DynamicsPerturbationKind
from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import (
    NB06Config,
    UnfoldedModelConfig,
    nb06_ood_generalization_experiment,
)
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import ContenderSpec
from mbl.experiments.zero_shot import synthesize_nominal_contenders
from mbl.models.analytic.truncated_riccati import TruncatedRiccatiSynthesizer
from mbl.experiments import CachePolicy
from mbl.experiments.cache import CacheReadOnlyMissError
from mbl.workbench.ood_sweep import (
    MIN_OOD_SEEDS,
    ConstraintProtocol,
    OODAxis,
    OODSweepContext,
    OODSweepSpec,
    build_ood_degradation_table,
    deserialize_ood_sweep_result,
    ood_sweep_signature_digest,
    relabel_noise_family_by_distance,
    run_constraint_ood_sweep,
    run_dynamics_ood_sweep,
    run_horizon_ood_sweep,
    run_interaction_scale_bound_sweep,
    run_noise_family_ood_sweep,
    run_ood_depth_ablation,
    run_ood_sweep_cached,
    run_scale_ood_sweep,
    serialize_ood_sweep_result,
)

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
HORIZON = 6
U_MAX = 0.3


def _base_factory() -> LQRProblemFactory:
    return LQRProblemFactory(
        state_dim=3, control_dim=2, horizon=HORIZON, seed=0, u_max=U_MAX
    )


def _nominal_batch(seed: int = 1) -> GaussianBatchSpec:
    return GaussianBatchSpec(
        state_dim=3, horizon=HORIZON, batch_size=16, seed=seed, process_noise_std=0.4
    )


def _truncated_riccati_artifact(problem):
    constraint = problem.constraints[0]
    return TruncatedRiccatiSynthesizer(HORIZON, constraint).synthesize(problem, CTX)


def _single_contender_context(**spec_overrides: object) -> OODSweepContext:
    factory = _base_factory()
    problem = factory.build()
    artifact = _truncated_riccati_artifact(problem)
    spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
    return OODSweepContext(
        artifacts={"truncated_riccati": artifact},
        contender_specs={"truncated_riccati": spec},
        base_factory=factory,
        nominal_batch=_nominal_batch(),
        ctx=CTX,
        u_max=U_MAX,
    )


def _two_identical_contenders_context() -> OODSweepContext:
    """The SAME frozen artifact under two different labels -- the
    stochastic-fairness probe: if the two labels' sampled batches ever
    diverged, their costs (from an identical deterministic policy) would
    diverge too."""
    factory = _base_factory()
    problem = factory.build()
    artifact = _truncated_riccati_artifact(problem)
    spec = ContenderSpec(family="truncated_riccati", config={"horizon": HORIZON})
    return OODSweepContext(
        artifacts={"a": artifact, "b": artifact},
        contender_specs={"a": spec, "b": spec},
        base_factory=factory,
        nominal_batch=_nominal_batch(),
        ctx=CTX,
        u_max=U_MAX,
    )


class TestStatisticalRigorGate:
    def test_n_seeds_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="n_seeds"):
            OODSweepSpec(
                axis=OODAxis.SCALE, levels=(1.0, 2.0), n_seeds=MIN_OOD_SEEDS - 1
            )

    def test_n_seeds_at_minimum_is_accepted(self) -> None:
        spec = OODSweepSpec(axis=OODAxis.SCALE, levels=(1.0,), n_seeds=MIN_OOD_SEEDS)
        assert spec.n_seeds == MIN_OOD_SEEDS

    def test_empty_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="levels"):
            OODSweepSpec(axis=OODAxis.SCALE, levels=(), n_seeds=MIN_OOD_SEEDS)


class TestStochasticFairness:
    def test_two_identical_artifacts_score_identically_at_every_point(self) -> None:
        context = _two_identical_contenders_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)

        points_a = result.points["a"]
        points_b = result.points["b"]
        assert len(points_a) == len(points_b) == 3 * MIN_OOD_SEEDS
        for point_a, point_b in zip(points_a, points_b, strict=True):
            assert point_a.level == point_b.level
            assert point_a.seed == point_b.seed
            assert point_a.metrics.cost_mean == pytest.approx(
                point_b.metrics.cost_mean, rel=1e-12
            )


class TestFloorCaching:
    def test_floor_is_computed_once_per_level_not_once_per_seed(
        self, monkeypatch
    ) -> None:
        call_count = {"n": 0}
        original = ood_sweep_module.compute_box_constrained_floors

        def _counting_wrapper(*args: object, **kwargs: object):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            ood_sweep_module, "compute_box_constrained_floors", _counting_wrapper
        )
        context = _single_contender_context()
        levels = (0.5, 1.0, 2.0)
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=levels,
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        run_scale_ood_sweep(context, spec)

        # SCALE varies process_noise_std per level (genuinely distinct
        # floors) but not per seed (same floor reused across all
        # MIN_OOD_SEEDS draws at a fixed level) -- exactly len(levels) solves.
        assert call_count["n"] == len(levels)

    def test_horizon_axis_floor_is_shared_across_every_level(self, monkeypatch) -> None:
        # J_SDP/J_LQR are infinite-horizon-average quantities: horizon does
        # not enter the floor's own formula at all, so every horizon level
        # shares the SAME floor -- exactly one solve total.
        call_count = {"n": 0}
        original = ood_sweep_module.compute_box_constrained_floors

        def _counting_wrapper(*args: object, **kwargs: object):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            ood_sweep_module, "compute_box_constrained_floors", _counting_wrapper
        )
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.HORIZON,
            levels=(HORIZON, 2 * HORIZON, 3 * HORIZON),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        run_horizon_ood_sweep(context, spec)
        assert call_count["n"] == 1


class TestPerContenderLevelSubsetting:
    def test_sparse_contender_only_visits_its_own_levels(self) -> None:
        factory = _base_factory()
        problem = factory.build()
        artifact = _truncated_riccati_artifact(problem)
        spec_dense = ContenderSpec(
            family="truncated_riccati", config={"horizon": HORIZON}
        )
        context = OODSweepContext(
            artifacts={"dense": artifact, "sparse": artifact},
            contender_specs={"dense": spec_dense, "sparse": spec_dense},
            base_factory=factory,
            nominal_batch=_nominal_batch(),
            ctx=CTX,
            u_max=U_MAX,
        )
        dense_levels = (0.5, 1.0, 2.0, 4.0)
        sparse_levels = (0.5, 4.0)
        sweep_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=dense_levels,
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
            contender_levels={"sparse": sparse_levels},
        )
        result = run_scale_ood_sweep(context, sweep_spec)

        assert {p.level for p in result.points["dense"]} == set(dense_levels)
        assert {p.level for p in result.points["sparse"]} == set(sparse_levels)
        assert len(result.points["sparse"]) == len(sparse_levels) * MIN_OOD_SEEDS


class TestHorizonAxisRehost:
    def test_rehosting_at_the_nominal_horizon_reproduces_nominal_cost(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.HORIZON,
            levels=(HORIZON,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=2,
        )
        result = run_horizon_ood_sweep(context, spec)
        band = result.cost_band("truncated_riccati")
        assert band.x == (HORIZON,)
        assert np.isfinite(band.median[0])

    def test_extended_horizon_produces_a_finite_cost(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.HORIZON,
            levels=(4 * HORIZON,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_horizon_ood_sweep(context, spec)
        band = result.cost_band("truncated_riccati")
        assert np.isfinite(band.median[0])


class TestDynamicsAxis:
    def test_additive_perturbation_produces_finite_costs_and_a_floor(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.DYNAMICS_ADDITIVE,
            levels=(0.0, 0.1, 0.2),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_dynamics_ood_sweep(
            context, spec, kind=DynamicsPerturbationKind.ADDITIVE
        )
        band = result.cost_band("truncated_riccati")
        assert all(np.isfinite(band.median))
        gap_band = result.suboptimality_gap_band("truncated_riccati")
        assert all(np.isfinite(gap_band.median))

    def test_rotation_perturbation_produces_finite_costs(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.DYNAMICS_ROTATION,
            levels=(0.0, 15.0, 30.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_dynamics_ood_sweep(
            context, spec, kind=DynamicsPerturbationKind.ROTATION
        )
        band = result.cost_band("truncated_riccati")
        assert all(np.isfinite(band.median))


class TestNoiseFamilyAxis:
    def test_cauchy_has_no_floor_but_every_other_family_does(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.GAUSSIAN, NoiseFamily.UNIFORM, NoiseFamily.CAUCHY),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_noise_family_ood_sweep(context, spec)
        by_level = {p.level: p for p in result.points["truncated_riccati"]}
        cauchy_points = [
            p
            for p in result.points["truncated_riccati"]
            if p.level is NoiseFamily.CAUCHY
        ]
        gaussian_points = [
            p
            for p in result.points["truncated_riccati"]
            if p.level is NoiseFamily.GAUSSIAN
        ]
        assert all(p.floor is None for p in cauchy_points)
        assert all(p.floor is not None for p in gaussian_points)
        del by_level  # constructed for readability only, not asserted on directly

    def test_median_cost_stays_finite_even_for_cauchy(self) -> None:
        # The per-batch MEAN can be huge/undefined for Cauchy; the median
        # (the primary statistic) must stay finite and sane.
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.CAUCHY,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=4,
        )
        result = run_noise_family_ood_sweep(context, spec)
        for point in result.points["truncated_riccati"]:
            assert np.isfinite(point.metrics.cost_median)


class TestScaleNoiseOnlyVariant:
    def test_scale_initial_state_false_leaves_initial_std_at_nominal(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE_NOISE_ONLY,
            levels=(1.0, 4.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec, scale_initial_state=False)
        assert result.axis is OODAxis.SCALE_NOISE_ONLY
        band = result.cost_band("truncated_riccati")
        assert all(np.isfinite(band.median))


class TestConstraintAxis:
    """Axis C (NB06 plan Sec 2.6/3.9): all three protocols produce finite
    costs; BLIND at a looser multiplier is an exact null; AWARE at the
    nominal multiplier reproduces the nominal cost; the floor moves with
    the shifted bound (not frozen at the nominal `u_max`, the Sec 3.7 fix
    this axis specifically needed)."""

    def test_all_three_protocols_produce_finite_costs(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.CONSTRAINT_BLIND,
            levels=(0.5, 1.0, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        for protocol in ConstraintProtocol:
            result = run_constraint_ood_sweep(context, spec, protocol=protocol)
            band = result.cost_band("truncated_riccati")
            assert all(np.isfinite(band.median)), protocol

    def test_blind_at_a_looser_multiplier_is_an_exact_null(self) -> None:
        context = _single_contender_context()
        nominal_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        nominal_cost = (
            run_scale_ood_sweep(context, nominal_spec)
            .cost_band("truncated_riccati")
            .median[0]
        )

        looser_spec = OODSweepSpec(
            axis=OODAxis.CONSTRAINT_BLIND,
            levels=(4.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        blind_result = run_constraint_ood_sweep(
            context, looser_spec, protocol=ConstraintProtocol.BLIND
        )
        blind_cost = blind_result.cost_band("truncated_riccati").median[0]
        assert blind_cost == pytest.approx(nominal_cost, rel=1e-9)

    def test_aware_at_the_nominal_multiplier_reproduces_nominal_cost(self) -> None:
        context = _single_contender_context()
        nominal_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        nominal_cost = (
            run_scale_ood_sweep(context, nominal_spec)
            .cost_band("truncated_riccati")
            .median[0]
        )

        aware_spec = OODSweepSpec(
            axis=OODAxis.CONSTRAINT_AWARE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        aware_result = run_constraint_ood_sweep(
            context, aware_spec, protocol=ConstraintProtocol.AWARE
        )
        aware_cost = aware_result.cost_band("truncated_riccati").median[0]
        assert aware_cost == pytest.approx(nominal_cost, rel=1e-9)

    def test_floor_moves_with_the_shifted_bound(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.CONSTRAINT_AWARE,
            levels=(0.5, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_constraint_ood_sweep(
            context, spec, protocol=ConstraintProtocol.AWARE
        )
        floors_by_level = {
            point.level: point.floor for point in result.points["truncated_riccati"]
        }
        assert floors_by_level[0.5] is not None
        assert floors_by_level[2.0] is not None
        assert floors_by_level[0.5].j_sdp != pytest.approx(floors_by_level[2.0].j_sdp)

    def test_null_protocol_gives_a_flat_normalized_band_for_truncated_riccati(
        self,
    ) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.CONSTRAINT_NULL,
            levels=(0.5, 1.0, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=2,
        )
        result = run_constraint_ood_sweep(
            context, spec, protocol=ConstraintProtocol.NULL
        )
        normalized = result.normalized_cost_band("truncated_riccati", power=2.0)
        assert normalized.median[0] == pytest.approx(normalized.median[-1], rel=1e-6)


class TestOODSweepResultAggregation:
    def test_max_violation_is_at_float_tolerance_for_a_projecting_family(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0, 3.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        assert result.max_violation("truncated_riccati") <= 1e-9

    def test_saturation_band_is_bounded_in_unit_interval(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0, 5.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        band = result.saturation_band("truncated_riccati")
        assert all(0.0 <= v <= 1.0 for v in band.median)

    def test_tail_band_is_at_or_above_the_cost_band(self) -> None:
        # cost_q90 (a per-point 90th-percentile-of-trajectories quantity)
        # is >= that same point's median -- so the q90 band's median (a
        # median-of-q90s) must be >= the cost band's median-of-medians too.
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0, 3.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=2,
        )
        result = run_scale_ood_sweep(context, spec)
        cost_band = result.cost_band("truncated_riccati")
        tail_band = result.tail_band("truncated_riccati")
        for cost_m, tail_m in zip(cost_band.median, tail_band.median, strict=True):
            assert tail_m >= cost_m - 1e-9

    def test_cvar_band_is_at_or_above_the_tail_band(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0, 3.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=2,
        )
        result = run_scale_ood_sweep(context, spec)
        tail_band = result.tail_band("truncated_riccati")
        cvar_band = result.cvar_band("truncated_riccati")
        for tail_m, cvar_m in zip(tail_band.median, cvar_band.median, strict=True):
            assert cvar_m >= tail_m - 1e-9

    def test_divergence_band_is_zero_when_not_requested(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0, 3.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        band = result.divergence_band("truncated_riccati")
        assert all(v == 0.0 for v in band.median)

    def test_normalized_cost_band_divides_by_level_to_the_power(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        cost_band = result.cost_band("truncated_riccati")
        normalized = result.normalized_cost_band("truncated_riccati", power=2.0)
        for x, cost_m, norm_m in zip(
            cost_band.x, cost_band.median, normalized.median, strict=True
        ):
            assert norm_m == pytest.approx(cost_m / x**2.0)


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=3)


def _fast_schedule() -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=1,
        refinement_epochs=1,
        train_matrix_from="each",
    )


def _fast_nb06_config(**overrides: object) -> NB06Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=HORIZON,
        u_max=U_MAX,
        num_unfolding_iterations=3,
        step_size_init=0.02,
        step_size_max=0.5,
        batch_size=16,
        n_eval_batches=2,
        seed=0,
        process_noise_std=0.4,
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


class TestFullExperimentIntegration:
    """Exercises the sweep engine against a REAL six-contender NB06
    experiment (not a single hand-built artifact) -- in particular, that
    the learned unfolded contenders rehost correctly through the full
    `run_horizon_ood_sweep` path, not just via `rehost_at_horizon` directly
    (already covered by `test_ood_rehost.py`)."""

    def test_every_contender_produces_a_finite_cost_at_an_extended_horizon(
        self, tmp_path
    ) -> None:
        cfg = _fast_nb06_config()
        experiment = nb06_ood_generalization_experiment(cfg)
        artifacts, _ = synthesize_nominal_contenders(experiment, root=tmp_path)
        contender_specs = {spec.resolved_label: spec for spec in experiment.contenders}
        context = OODSweepContext(
            artifacts=artifacts,
            contender_specs=contender_specs,
            base_factory=LQRProblemFactory(
                state_dim=cfg.state_dim,
                control_dim=cfg.control_dim,
                horizon=cfg.horizon,
                seed=cfg.seed,
                u_max=cfg.u_max,
            ),
            nominal_batch=experiment.evaluation.batch_spec,
            ctx=experiment.ctx,
            u_max=cfg.u_max,
        )
        spec = OODSweepSpec(
            axis=OODAxis.HORIZON,
            levels=(HORIZON, 3 * HORIZON),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_horizon_ood_sweep(context, spec)

        assert set(result.points) == set(artifacts)
        for label in artifacts:
            band = result.cost_band(label)
            assert all(np.isfinite(band.median)), f"{label} produced a non-finite cost"

    def test_unfolded_contenders_rehost_to_the_extended_horizon_exactly(
        self, tmp_path
    ) -> None:
        cfg = _fast_nb06_config()
        experiment = nb06_ood_generalization_experiment(cfg)
        artifacts, _ = synthesize_nominal_contenders(experiment, root=tmp_path)
        contender_specs = {spec.resolved_label: spec for spec in experiment.contenders}
        target_horizon = 4 * HORIZON
        for label in ("unfolded_alpha", "unfolded_alpha_p"):
            rehosted = ood_sweep_module.rehost_at_horizon(
                artifacts[label],
                contender_specs[label],
                ood_sweep_module.PerturbedLQRProblemFactory(
                    base=LQRProblemFactory(
                        state_dim=cfg.state_dim,
                        control_dim=cfg.control_dim,
                        horizon=cfg.horizon,
                        seed=cfg.seed,
                        u_max=cfg.u_max,
                    ),
                    perturbation=ood_sweep_module.DynamicsPerturbation(),
                    horizon_override=target_horizon,
                ).build(),
                horizon=target_horizon,
                ctx=experiment.ctx,
            )
            assert rehosted.controller.config.horizon == target_horizon


class TestDepthAblation:
    def test_sweeps_every_depth_and_restricts_to_the_requested_labels(
        self, tmp_path
    ) -> None:
        base_config = _fast_nb06_config()
        scale_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0, 3.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        results = run_ood_depth_ablation(
            base_config,
            depths=(3, 4),
            scale_spec=scale_spec,
            root=tmp_path,
            labels=("unfolded_alpha", "unfolded_alpha_p"),
        )

        assert set(results) == {3, 4}
        for depth_result in results.values():
            assert set(depth_result.points) == {"unfolded_alpha", "unfolded_alpha_p"}
            for label in depth_result.points:
                band = depth_result.cost_band(label)
                assert all(np.isfinite(band.median))

    def test_defaults_to_every_contender_when_labels_is_none(self, tmp_path) -> None:
        base_config = _fast_nb06_config()
        scale_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        results = run_ood_depth_ablation(
            base_config, depths=(3,), scale_spec=scale_spec, root=tmp_path
        )
        # Every one of the six NB06 contender labels is present when
        # `labels` is not restricted.
        assert set(results[3].points) == {
            "truncated_riccati",
            "unfolded_alpha",
            "unfolded_alpha_p",
            "cocp",
            "neural",
            "cocp_lower_bound",
        }


class TestDegradationTable:
    def test_one_row_per_axis_contender_with_expected_columns(self) -> None:
        context = _single_contender_context()
        scale_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        horizon_spec = OODSweepSpec(
            axis=OODAxis.HORIZON,
            levels=(HORIZON, 2 * HORIZON),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        results = {
            OODAxis.SCALE: run_scale_ood_sweep(context, scale_spec),
            OODAxis.HORIZON: run_horizon_ood_sweep(context, horizon_spec),
        }
        table = build_ood_degradation_table(results)

        assert set(table["axis"]) == {"scale", "horizon"}
        assert set(table["contender"]) == {"truncated_riccati"}
        assert len(table) == 2
        for column in (
            "cost_mildest",
            "cost_severest",
            "relative_degradation",
            "saturation_mildest",
            "saturation_severest",
            "max_violation",
            "breakdown_level",
        ):
            assert column in table.columns

    def test_empty_results_gives_an_empty_table(self) -> None:
        table = build_ood_degradation_table({})
        assert len(table) == 0

    def test_noise_family_axis_with_raw_categorical_levels_does_not_raise(self) -> None:
        # Regression: OODAxis.NOISE_FAMILY's raw result (before
        # relabel_noise_family_by_distance) has NoiseFamily-member levels,
        # not floats -- build_ood_degradation_table must handle it, not
        # crash inside _breakdown_level.
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_noise_family_ood_sweep(context, spec)
        table = build_ood_degradation_table({OODAxis.NOISE_FAMILY: result})
        assert len(table) == 1
        assert np.isnan(table.iloc[0]["breakdown_level"])

    def test_breakdown_level_is_the_first_level_past_2x_the_mildest_gap(self) -> None:
        context = _single_contender_context()
        # The rotation axis is an exact isometry (NB06 plan Sec 2.3): the
        # floor is unchanged across levels, so the gap moves ONLY because
        # of the contender's own (im)perfect equivariance -- real, if
        # small, movement worth exercising here.
        spec = OODSweepSpec(
            axis=OODAxis.DYNAMICS_ROTATION,
            levels=(0.0, 15.0, 30.0, 45.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=2,
        )
        result = run_dynamics_ood_sweep(
            context, spec, kind=DynamicsPerturbationKind.ROTATION
        )
        table = build_ood_degradation_table({OODAxis.DYNAMICS_ROTATION: result})
        row = table.iloc[0]
        assert np.isnan(row["breakdown_level"]) or row["breakdown_level"] in (
            0.0,
            15.0,
            30.0,
            45.0,
        )


class TestBreakdownLevelHelper:
    """Direct, deterministic tests of `_breakdown_level` (NB06 plan Sec
    5.3/11 G11) against synthetic bands -- the integration test in
    `TestDegradationTable` exercises real dynamics and can't pin an exact
    expected value; this one can."""

    def test_first_level_exceeding_2x_the_mildest_gap(self) -> None:
        band = ood_sweep_module.CurveWithBand(
            x=(0.0, 1.0, 2.0, 3.0),
            median=(0.05, 0.06, 0.15, 0.20),
            q25=(0.0, 0.0, 0.0, 0.0),
            q75=(0.0, 0.0, 0.0, 0.0),
        )
        # baseline = 0.05, threshold = 0.10 -- first exceeded at x=2.0 (0.15).
        assert ood_sweep_module._breakdown_level(band) == pytest.approx(2.0)

    def test_never_exceeding_gives_nan(self) -> None:
        band = ood_sweep_module.CurveWithBand(
            x=(0.0, 1.0, 2.0),
            median=(0.05, 0.06, 0.07),
            q25=(0.0, 0.0, 0.0),
            q75=(0.0, 0.0, 0.0),
        )
        assert np.isnan(ood_sweep_module._breakdown_level(band))

    def test_empty_band_gives_nan(self) -> None:
        band = ood_sweep_module.CurveWithBand(x=(), median=(), q25=(), q75=())
        assert np.isnan(ood_sweep_module._breakdown_level(band))

    def test_near_zero_baseline_gives_nan_rather_than_a_spurious_breakdown(
        self,
    ) -> None:
        band = ood_sweep_module.CurveWithBand(
            x=(0.0, 1.0),
            median=(1e-15, 0.5),
            q25=(0.0, 0.0),
            q75=(0.0, 0.0),
        )
        assert np.isnan(ood_sweep_module._breakdown_level(band))

    def test_non_numeric_x_gives_nan_rather_than_raising(self) -> None:
        # OODAxis.NOISE_FAMILY's RAW (pre-relabel) levels are NoiseFamily
        # members, not floats -- build_ood_degradation_table must not crash
        # when it is handed that axis's suboptimality_gap_band directly
        # (the exact regression this notebook execution surfaced).
        band = ood_sweep_module.CurveWithBand(
            x=(NoiseFamily.GAUSSIAN, NoiseFamily.STUDENT_T),
            median=(0.05, 0.5),
            q25=(0.0, 0.0),
            q75=(0.0, 0.0),
        )
        assert np.isnan(ood_sweep_module._breakdown_level(band))


class TestRelabelNoiseFamilyByDistance:
    def test_levels_become_finite_floats_ordered_by_distance(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.GAUSSIAN, NoiseFamily.LAPLACE, NoiseFamily.CAUCHY),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_noise_family_ood_sweep(context, spec)

        relabeled = relabel_noise_family_by_distance(result, sigma=0.4)

        levels = [point.level for point in relabeled.points["truncated_riccati"]]
        assert all(isinstance(level, float) for level in levels)
        # Gaussian's own Hellinger distance to itself is exactly zero, and
        # Cauchy (the heaviest-tailed family here) is the farthest.
        by_family = {
            point.level: original.level
            for point, original in zip(
                relabeled.points["truncated_riccati"],
                result.points["truncated_riccati"],
                strict=True,
            )
        }
        gaussian_distance = next(
            d for d, family in by_family.items() if family is NoiseFamily.GAUSSIAN
        )
        cauchy_distance = next(
            d for d, family in by_family.items() if family is NoiseFamily.CAUCHY
        )
        assert gaussian_distance == pytest.approx(0.0, abs=1e-9)
        assert cauchy_distance > gaussian_distance

    def test_bands_still_aggregate_correctly_after_relabeling(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.GAUSSIAN, NoiseFamily.UNIFORM),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_noise_family_ood_sweep(context, spec)
        relabeled = relabel_noise_family_by_distance(result, sigma=0.4)

        band = relabeled.cost_band("truncated_riccati")
        assert len(band.x) == 2
        assert all(isinstance(x, float) for x in band.x)
        assert all(np.isfinite(band.median))


class TestSerializationRoundTrip:
    def test_scale_result_round_trips_exactly(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_scale_ood_sweep(context, spec)
        round_tripped = deserialize_ood_sweep_result(serialize_ood_sweep_result(result))

        assert round_tripped.axis is result.axis
        assert set(round_tripped.points) == set(result.points)
        for point, original in zip(
            round_tripped.points["truncated_riccati"],
            result.points["truncated_riccati"],
            strict=True,
        ):
            assert point.level == original.level
            assert point.seed == original.seed
            assert point.metrics == original.metrics
            assert point.floor is not None and original.floor is not None
            np.testing.assert_allclose(point.floor.p_sdp, original.floor.p_sdp)

    def test_noise_family_levels_round_trip_as_enum_members(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.GAUSSIAN, NoiseFamily.CAUCHY),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_noise_family_ood_sweep(context, spec)
        round_tripped = deserialize_ood_sweep_result(serialize_ood_sweep_result(result))

        levels = {point.level for point in round_tripped.points["truncated_riccati"]}
        assert levels == {NoiseFamily.GAUSSIAN, NoiseFamily.CAUCHY}
        assert all(isinstance(level, NoiseFamily) for level in levels)

    def test_cauchy_none_floor_round_trips_as_none(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.NOISE_FAMILY,
            levels=(NoiseFamily.CAUCHY,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_noise_family_ood_sweep(context, spec)
        round_tripped = deserialize_ood_sweep_result(serialize_ood_sweep_result(result))
        assert all(
            point.floor is None for point in round_tripped.points["truncated_riccati"]
        )


class TestOODSweepCachedPersistence:
    def test_second_call_replays_without_any_rollouts(
        self, tmp_path, monkeypatch
    ) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        first = run_ood_sweep_cached(run_scale_ood_sweep, context, spec, root=tmp_path)

        call_count = {"n": 0}
        original = ood_sweep_module.evaluate_under_shift

        def _counting_wrapper(*args: object, **kwargs: object):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ood_sweep_module, "evaluate_under_shift", _counting_wrapper)
        second = run_ood_sweep_cached(run_scale_ood_sweep, context, spec, root=tmp_path)

        assert call_count["n"] == 0
        assert set(second.points) == set(first.points)
        for point, original_point in zip(
            second.points["truncated_riccati"],
            first.points["truncated_riccati"],
            strict=True,
        ):
            assert point.metrics == original_point.metrics

    def test_a_different_spec_is_a_cache_miss(self, tmp_path) -> None:
        context = _single_contender_context()
        spec_a = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 1.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        spec_b = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(0.5, 2.0),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        assert ood_sweep_signature_digest(
            context, spec_a
        ) != ood_sweep_signature_digest(context, spec_b)

    def test_readonly_replays_a_hit(self, tmp_path) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        run_ood_sweep_cached(run_scale_ood_sweep, context, spec, root=tmp_path)
        result = run_ood_sweep_cached(
            run_scale_ood_sweep,
            context,
            spec,
            root=tmp_path,
            policy=CachePolicy(mode="readonly"),
        )
        assert set(result.points) == {"truncated_riccati"}

    def test_readonly_raises_on_a_miss(self, tmp_path) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        with pytest.raises(CacheReadOnlyMissError):
            run_ood_sweep_cached(
                run_scale_ood_sweep,
                context,
                spec,
                root=tmp_path,
                policy=CachePolicy(mode="readonly"),
            )

    def test_recompute_always_calls_sweep_fn(self, tmp_path, monkeypatch) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        call_count = {"n": 0}
        original = ood_sweep_module.evaluate_under_shift

        def _counting_wrapper(*args: object, **kwargs: object):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ood_sweep_module, "evaluate_under_shift", _counting_wrapper)
        run_ood_sweep_cached(
            run_scale_ood_sweep,
            context,
            spec,
            root=tmp_path,
            policy=CachePolicy(mode="recompute"),
        )
        first_count = call_count["n"]
        assert first_count > 0
        run_ood_sweep_cached(
            run_scale_ood_sweep,
            context,
            spec,
            root=tmp_path,
            policy=CachePolicy(mode="recompute"),
        )
        assert call_count["n"] == 2 * first_count


class TestInteractionScaleBoundSweep:
    def test_produces_finite_costs_at_every_pair(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.INTERACTION_SCALE_BOUND,
            levels=((1.0, 1.0), (1.0, 0.5), (2.0, 1.0), (2.0, 0.5)),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_interaction_scale_bound_sweep(context, spec)
        for point in result.points["truncated_riccati"]:
            assert np.isfinite(point.metrics.cost_median)

    def test_nominal_pair_reproduces_the_nominal_cost(self) -> None:
        context = _single_contender_context()
        nominal_scale_spec = OODSweepSpec(
            axis=OODAxis.SCALE,
            levels=(1.0,),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        nominal_cost = (
            run_scale_ood_sweep(context, nominal_scale_spec)
            .cost_band("truncated_riccati")
            .median[0]
        )

        spec = OODSweepSpec(
            axis=OODAxis.INTERACTION_SCALE_BOUND,
            levels=((1.0, 1.0),),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_interaction_scale_bound_sweep(context, spec)
        interaction_cost = result.cost_band("truncated_riccati").median[0]
        assert interaction_cost == pytest.approx(nominal_cost, rel=1e-9)

    def test_interaction_grid_pivots_correctly(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.INTERACTION_SCALE_BOUND,
            levels=((1.0, 1.0), (1.0, 0.5), (2.0, 1.0), (2.0, 0.5)),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_interaction_scale_bound_sweep(context, spec)
        scale_levels, bound_levels, grid = result.interaction_grid("truncated_riccati")

        assert scale_levels == (1.0, 2.0)
        assert bound_levels == (0.5, 1.0)
        assert grid.shape == (2, 2)
        assert np.all(np.isfinite(grid))
        # Tighter bound (0.5) at fixed scale should not give a LOWER cost
        # than the looser bound (1.0) -- a real, if loose, sanity check.
        assert grid[0, 0] >= grid[0, 1] - 1e-6  # scale=1.0: bound=0.5 vs 1.0
        assert grid[1, 0] >= grid[1, 1] - 1e-6  # scale=2.0: bound=0.5 vs 1.0

    def test_interaction_grid_has_nan_for_unswept_combinations(self) -> None:
        context = _single_contender_context()
        spec = OODSweepSpec(
            axis=OODAxis.INTERACTION_SCALE_BOUND,
            levels=((1.0, 1.0), (2.0, 0.5)),
            n_seeds=MIN_OOD_SEEDS,
            eval_batches_per_seed=1,
        )
        result = run_interaction_scale_bound_sweep(context, spec)
        _, _, grid = result.interaction_grid("truncated_riccati")
        assert np.isnan(grid).sum() == 2
