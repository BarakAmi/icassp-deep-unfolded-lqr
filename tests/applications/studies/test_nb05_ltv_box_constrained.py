"""Phase D acceptance tests for the NB05 experiment declaration
(docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 6.5/12): the six
contenders (the original four plus the P-resolution extension's R1.5/R2,
Sec 14) resolve through `run_experiment` end to end on the real recipe/
cache/history layers, in EVERY regime, every contender's REALIZED controls
stay within the box (the projection guarantee), and COCP/COCP-LB are
confirmed absent (the user's explicit scope decision, Sec 0.4/4.4).

Mirrors `test_nb04_box_constrained.py`'s structure closely; floor-specific
coverage lives in `test_nb05_ltv_floors.py`, not duplicated here.
"""

import math

import pytest

from mbl.applications.ltv_factories import LTVRegime
from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import (
    NB05Config,
    UnfoldedModelConfig,
    build_nb05_config,
    nb05_ltv_experiment,
)
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import CachePolicy, run_experiment

_EXPECTED_LABELS_TO_FAMILIES = {
    "truncated_riccati": "truncated_riccati",
    "standard_pgd": "unfolded_fixed",
    "unfolded_alpha": "unfolded",
    "unfolded_alpha_p": "unfolded_warmstart",
    "unfolded_alpha_p_scalarmod": "unfolded_warmstart",
    "unfolded_alpha_p_periter": "unfolded_warmstart",
}

_REGIMES = [
    (LTVRegime.PERIODIC, 3),
    (LTVRegime.BLOCK_CONSTANT, 3),
    (LTVRegime.FULLY_VARYING, None),
]


def _fast_plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=4)


def _fast_schedule() -> LayerwiseTrainingPlan:
    return LayerwiseTrainingPlan(
        optimizer=OptimizerSpec("adam", 0.05),
        warmup_epochs_per_layer=2,
        refinement_epochs=2,
        train_matrix_from="refinement",
    )


def _fast_config(**overrides: object) -> NB05Config:
    defaults: dict[str, object] = dict(
        state_dim=3,
        control_dim=2,
        horizon=6,
        u_max=0.3,
        regime=LTVRegime.PERIODIC,
        period_or_block=3,
        variation_strength=0.4,
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
        unfolded_alpha_p_scalarmod=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX,
            training_mode="layerwise",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
        unfolded_alpha_p_periter=UnfoldedModelConfig(
            kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX,
            training_mode="layerwise",
            plan=_fast_plan(),
            schedule=_fast_schedule(),
        ),
    )
    defaults.update(overrides)
    return NB05Config(**defaults)


class TestExperimentDeclaration:
    def test_declares_six_contenders_with_expected_families(self) -> None:
        experiment = nb05_ltv_experiment(_fast_config())
        families = {spec.resolved_label: spec.family for spec in experiment.contenders}
        assert families == _EXPECTED_LABELS_TO_FAMILIES

    def test_no_cocp_contenders_present(self) -> None:
        """The user's explicit scope decision (Sec 0.4/4.4): COCP/COCP-LB
        are dropped entirely, not merely unconfigured."""
        experiment = nb05_ltv_experiment(_fast_config())
        labels = {spec.resolved_label for spec in experiment.contenders}
        assert "cocp" not in labels
        assert "cocp_lower_bound" not in labels

    def test_problem_is_an_ltv_factory_carrying_the_configured_box(self) -> None:
        from mbl.applications.ltv_factories import LTVLQRProblemFactory

        experiment = nb05_ltv_experiment(_fast_config(u_max=0.7))
        assert isinstance(experiment.problem, LTVLQRProblemFactory)
        problem = experiment.problem.build()
        assert problem.constraints is not None
        assert len(problem.constraints) == 1

    def test_problem_carries_the_configured_regime(self) -> None:
        experiment = nb05_ltv_experiment(
            _fast_config(regime=LTVRegime.BLOCK_CONSTANT, period_or_block=2)
        )
        from mbl.applications.ltv_factories import LTVLQRProblemFactory

        assert isinstance(experiment.problem, LTVLQRProblemFactory)
        assert experiment.problem.regime is LTVRegime.BLOCK_CONSTANT
        assert experiment.problem.distinct_slice_count() == 3

    def test_ctx_honors_the_configured_dtype(self) -> None:
        import torch

        experiment = nb05_ltv_experiment(_fast_config(dtype=torch.float32))
        assert experiment.ctx.precision.value == "float32"

    def test_non_positive_u_max_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="u_max"):
            _fast_config(u_max=0.0)


class TestBuildNb05Config:
    def test_flat_builder_matches_the_dataclass_constructor(self) -> None:
        via_builder = build_nb05_config(
            state_dim=3,
            control_dim=2,
            horizon=6,
            u_max=0.3,
            regime=LTVRegime.PERIODIC,
            period_or_block=3,
            variation_strength=0.4,
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
            unfolded_alpha_p_scalarmod=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX
            ),
            unfolded_alpha_p_periter=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX
            ),
        )
        assert via_builder.u_max == 0.3
        assert via_builder.state_dim == 3
        assert via_builder.regime is LTVRegime.PERIODIC

    def test_fully_varying_needs_no_period_or_block(self) -> None:
        via_builder = build_nb05_config(
            state_dim=3,
            control_dim=2,
            horizon=6,
            u_max=0.3,
            regime=LTVRegime.FULLY_VARYING,
            variation_strength=0.4,
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
            unfolded_alpha_p_scalarmod=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_SCALAR_MODULATED_MATRIX
            ),
            unfolded_alpha_p_periter=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_PER_ITERATION_MATRIX
            ),
        )
        assert via_builder.period_or_block is None


class TestTrainingModeRouting:
    """Mirrors NB03/NB04's own routing test: each learned model routes to
    the layer-wise or end-to-end regime per its OWN `UnfoldedModelConfig
    .training_mode` -- a free per-model Control-Panel choice, no regime
    hardcoded (NB05 plan module docstring)."""

    def _flagship(self, mode: str):
        config = _fast_config(
            unfolded_alpha_p=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                training_mode=mode,
                plan=_fast_plan(),
                schedule=_fast_schedule(),
            )
        )
        experiment = nb05_ltv_experiment(config)
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
        assert flagship.config["kind"] is UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX


class TestEndToEndExecution:
    @pytest.mark.parametrize("regime,period_or_block", _REGIMES)
    def test_every_contender_executes_and_produces_a_finite_cost(
        self, tmp_path, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        experiment = nb05_ltv_experiment(
            _fast_config(regime=regime, period_or_block=period_or_block)
        )
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
        experiment = nb05_ltv_experiment(_fast_config())
        run_experiment(experiment, root=tmp_path)

        report = run_experiment(experiment, root=tmp_path, policy=CachePolicy("reuse"))

        assert {r.disposition for r in report.results.values()} == {"hit"}

    def test_warmstart_contender_genuinely_trained(self, tmp_path) -> None:
        experiment = nb05_ltv_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        result = report.results["unfolded_alpha_p"]
        assert "final_loss" in result.metrics
        assert math.isfinite(result.metrics["final_loss"])

    def test_standard_unfolded_contender_genuinely_trained(self, tmp_path) -> None:
        experiment = nb05_ltv_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        result = report.results["unfolded_alpha"]
        assert "final_loss" in result.metrics
        assert math.isfinite(result.metrics["final_loss"])

    def test_truncated_riccati_carries_no_training_loss(self, tmp_path) -> None:
        experiment = nb05_ltv_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        assert "final_loss" not in report.results["truncated_riccati"].metrics

    def test_zero_variation_strength_reproduces_nb04_style_lti_costs(
        self, tmp_path
    ) -> None:
        """A degenerate-regime end-to-end run (Sec 2.3's degeneracy law,
        exercised through the FULL experiment/cache stack this time, not
        just the factory in isolation)."""
        experiment = nb05_ltv_experiment(_fast_config(variation_strength=0.0))
        report = run_experiment(experiment, root=tmp_path)

        for result in report.results.values():
            assert math.isfinite(result.metrics["eval_expected_cost"])


class TestBoxConstraintSatisfaction:
    """The projection guarantee (NB05 plan Sec 9/12), across every regime:
    every contender's REALIZED controls must satisfy |u| <= u_max + eps.
    Mirrors `test_nb04_box_constrained.py`'s identical check."""

    @pytest.mark.parametrize("regime,period_or_block", _REGIMES)
    def test_every_contenders_realized_controls_stay_within_the_box(
        self, regime: LTVRegime, period_or_block: int | None
    ) -> None:
        import torch

        u_max = 0.3
        config = _fast_config(
            u_max=u_max, regime=regime, period_or_block=period_or_block
        )
        experiment = nb05_ltv_experiment(config)
        problem = experiment.problem.build()
        ctx = experiment.ctx

        batch = 5
        x0 = torch.full(
            (batch, config.state_dim),
            50.0,
            dtype=ctx.torch_dtype,
            device=ctx.torch_device,
        )
        w = torch.zeros(
            (batch, config.horizon, config.state_dim),
            dtype=ctx.torch_dtype,
            device=ctx.torch_device,
        )
        v = torch.zeros_like(w)

        for spec in experiment.contenders:
            controller = spec.resolve().build_controller(problem, ctx)
            policy = controller.get_control_policy()
            with torch.no_grad():
                _, _, controls = problem.system.run(policy, x0, w, v)
            controls_np = (
                controls.detach().cpu().numpy()
                if isinstance(controls, torch.Tensor)
                else controls
            )
            assert (abs(controls_np) <= u_max + 1e-6).all(), (
                f"{spec.resolved_label} ({regime}): a realized control exceeded "
                f"the box (u_max = {u_max})"
            )
