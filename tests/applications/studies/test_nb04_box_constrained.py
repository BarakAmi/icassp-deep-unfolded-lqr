"""NB04 acceptance tests for the box-constrained `Experiment` declaration
(docs/planning/03_studies/nb04_box_constrained/box_constrained_lqr_benchmark.md Sec 9, extended by
docs/planning/03_studies/nb04_box_constrained/cocp_integration_and_convergence_visualization.md Sec 3.2 for
COCP and docs/planning/03_studies/nb04_box_constrained/reference_bounds_and_figure_refinement.md
Sec 1.4 for COCP-LB): the six contenders resolve through `run_experiment`
end to end on the real recipe/cache/history layers, every contender's
REALIZED controls stay within the box (the projection guarantee), and the
two SDP floors are well-formed and correctly bracket each other -- proving
Phase A's acceptance criteria on the real stack, never mocked.

Numerical-closeness claims against production-scale training are deliberately
out of scope here, exactly as `test_nb03_unfolding.py` documents: training to
genuine convergence takes far longer than a unit-test budget allows. What
this suite proves is that every contender EXECUTES, CACHES, and produces a
finite, box-feasible cost, and that the two floors are mathematically
well-formed (J_LQR <= J_SDP always; both PSD, both finite).
"""

import math

import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are

from mbl.applications.recipes import UnfoldedKind
from mbl.applications.studies import (
    NB04Config,
    UnfoldedModelConfig,
    build_nb04_config,
    compute_box_constrained_floors,
    nb04_box_constrained_experiment,
)
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import CachePolicy, run_experiment

_EXPECTED_LABELS_TO_FAMILIES = {
    "truncated_riccati": "truncated_riccati",
    "standard_pgd": "unfolded_fixed",
    "unfolded_alpha": "unfolded",
    "unfolded_alpha_p": "unfolded_warmstart",
    "cocp": "cocp",
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


def _fast_config(**overrides: object) -> NB04Config:
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
    )
    defaults.update(overrides)
    return NB04Config(**defaults)


class TestExperimentDeclaration:
    def test_declares_six_contenders_with_expected_families(self) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config())
        families = {spec.resolved_label: spec.family for spec in experiment.contenders}
        assert families == _EXPECTED_LABELS_TO_FAMILIES

    def test_cocp_lower_bound_config_carries_process_noise_std(self) -> None:
        """COCP-LB's SDP needs `process_noise_std` explicitly (unlike COCP,
        it has no training batch to read a noise distribution from) --
        wired from the SAME `cfg.process_noise_std` every other contender's
        evaluation batch uses (reference-bounds plan Sec 1.4 step 1)."""
        config = _fast_config(process_noise_std=0.42)
        experiment = nb04_box_constrained_experiment(config)
        spec = next(
            s for s in experiment.contenders if s.resolved_label == "cocp_lower_bound"
        )
        assert spec.config["process_noise_std"] == pytest.approx(0.42)

    def test_problem_carries_the_configured_box_constraint(self) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config(u_max=0.7))
        problem = experiment.problem.build()
        assert problem.constraints is not None
        assert len(problem.constraints) == 1

    def test_ctx_honors_the_configured_dtype(self) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config(dtype=torch.float32))
        assert experiment.ctx.precision.value == "float32"

    def test_ctx_defaults_to_cpu_device(self) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config())
        assert experiment.ctx.device == "cpu"

    def test_evaluation_draws_multiple_batches_when_configured(self) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config(n_eval_batches=3))
        assert experiment.evaluation.n_batches == 3

    def test_non_positive_u_max_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="u_max"):
            _fast_config(u_max=0.0)


class TestBuildNb04Config:
    def test_flat_builder_matches_the_dataclass_constructor(self) -> None:
        via_builder = build_nb04_config(
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
        )
        assert via_builder.u_max == 0.3
        assert via_builder.state_dim == 3


class TestTrainingModeRouting:
    """Mirrors NB03's own routing test: each learned model routes to the
    layer-wise or end-to-end regime per its OWN `UnfoldedModelConfig
    .training_mode` -- a free per-model Control-Panel choice, exactly as in
    NB03 (the user's explicit directive for NB04: no regime is hardcoded)."""

    def _flagship(self, mode: str):
        config = _fast_config(
            unfolded_alpha_p=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                training_mode=mode,
                plan=_fast_plan(),
                schedule=_fast_schedule(),
            )
        )
        experiment = nb04_box_constrained_experiment(config)
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

    def test_unfolded_alpha_can_also_run_layerwise(self) -> None:
        config = _fast_config(
            unfolded_alpha=UnfoldedModelConfig(
                kind=UnfoldedKind.LEARNED_STEP_SIZE,
                training_mode="layerwise",
                plan=_fast_plan(),
                schedule=_fast_schedule(),
            )
        )
        experiment = nb04_box_constrained_experiment(config)
        (alpha,) = (
            spec
            for spec in experiment.contenders
            if spec.resolved_label == "unfolded_alpha"
        )
        assert alpha.family == "unfolded_warmstart"
        assert alpha.config["kind"] is UnfoldedKind.LEARNED_STEP_SIZE


class TestEndToEndExecution:
    def test_every_contender_executes_and_produces_a_finite_cost(
        self, tmp_path
    ) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config())
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
        experiment = nb04_box_constrained_experiment(_fast_config())
        run_experiment(experiment, root=tmp_path)

        report = run_experiment(experiment, root=tmp_path, policy=CachePolicy("reuse"))

        assert {r.disposition for r in report.results.values()} == {"hit"}

    def test_warmstart_contender_genuinely_trained(self, tmp_path) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        result = report.results["unfolded_alpha_p"]
        assert "final_loss" in result.metrics
        assert math.isfinite(result.metrics["final_loss"])

    def test_standard_unfolded_contender_genuinely_trained(self, tmp_path) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        result = report.results["unfolded_alpha"]
        assert "final_loss" in result.metrics
        assert math.isfinite(result.metrics["final_loss"])

    def test_truncated_riccati_carries_no_training_loss(self, tmp_path) -> None:
        experiment = nb04_box_constrained_experiment(_fast_config())
        report = run_experiment(experiment, root=tmp_path)

        assert "final_loss" not in report.results["truncated_riccati"].metrics


class TestBoxConstraintSatisfaction:
    """The projection guarantee (NB04 plan Sec 9): every contender's
    REALIZED controls must satisfy |u| <= u_max + eps -- the saturated
    Riccati baseline and every unfolded family alike.

    Checked by building each contender's controller directly from its
    resolved recipe and rolling it out on a deliberately large-magnitude
    state (forcing saturation), NOT via the persisted `trajectory_controls`
    artifact: that artifact only exists for contenders whose synthesis
    routes through the Engine/Runner/callback pipeline (the two TRAINED
    unfolded families). `truncated_riccati` (a direct two-phase synthesizer)
    and `standard_pgd` (a `FrozenControllerSynthesizer`, construction-only)
    never attach `TrajectoryLoggingCallback` -- mirroring NB03's own
    `sim_riccati`, whose notebook computes its comparison trajectory fresh
    for the identical reason ("Riccati is a deterministic, depth-invariant
    closed-form policy with no training run of its own to draw a persisted
    trajectory from")."""

    def test_every_contenders_realized_controls_stay_within_the_box(self) -> None:
        u_max = 0.3
        config = _fast_config(u_max=u_max)
        experiment = nb04_box_constrained_experiment(config)
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
                else np.asarray(controls)
            )
            assert np.all(np.abs(controls_np) <= u_max + 1e-6), (
                f"{spec.resolved_label}: a realized control exceeded the box "
                f"(max |u| = {np.abs(controls_np).max():.6f}, u_max = {u_max})"
            )


class TestBoxConstrainedFloors:
    """The two SDP lower bounds (NB04 plan Sec 2.2): well-formed (finite,
    PSD) and correctly bracketed, J_LQR <= J_SDP -- a fact that holds for
    ANY problem instance (dropping a constraint can only lower the optimal
    cost), independent of Monte-Carlo scale."""

    def test_j_lqr_matches_the_dare_trace_independently(self) -> None:
        config = _fast_config()
        problem = nb04_box_constrained_experiment(config).problem.build()

        floors = compute_box_constrained_floors(
            problem, process_noise_std=config.process_noise_std, u_max=config.u_max
        )

        A, B = problem.system.A_t[0], problem.system.B_t[0]
        Q, R = problem.cost.Q[0], problem.cost.R[0]
        P_expected = solve_discrete_are(A, B, Q, R)
        W = config.process_noise_std**2 * np.eye(config.state_dim)
        expected_j_lqr = float(np.trace(P_expected @ W))

        assert floors.j_lqr == pytest.approx(expected_j_lqr, rel=1e-8)

    def test_j_sdp_is_at_least_j_lqr(self) -> None:
        """The box-aware bound can never be looser than the unconstrained
        one: the SDP maximizes over a STRICT SUBSET of the unconstrained
        problem's feasible set (RESEARCH_PLAN.md Sec 5.1)."""
        config = _fast_config()
        problem = nb04_box_constrained_experiment(config).problem.build()

        floors = compute_box_constrained_floors(
            problem, process_noise_std=config.process_noise_std, u_max=config.u_max
        )

        assert floors.j_sdp >= floors.j_lqr - 1e-6

    def test_floors_are_finite_and_matrices_are_symmetric_psd(self) -> None:
        config = _fast_config()
        problem = nb04_box_constrained_experiment(config).problem.build()

        floors = compute_box_constrained_floors(
            problem, process_noise_std=config.process_noise_std, u_max=config.u_max
        )

        assert math.isfinite(floors.j_lqr)
        assert math.isfinite(floors.j_sdp)
        for P in (floors.p_lqr, floors.p_sdp):
            symmetric = (P + P.T) / 2
            eigvals = np.linalg.eigvalsh(symmetric)
            assert np.all(eigvals >= -1e-6)

    def test_j_sdp_approaches_j_lqr_as_u_max_grows(self) -> None:
        """As u_max -> infinity the box stops binding, so the two floors
        should converge (mirrors `test_lower_bound.py`'s own version of this
        property, exercised here through the NB04 wrapper)."""
        config = _fast_config(u_max=200.0)
        problem = nb04_box_constrained_experiment(config).problem.build()

        floors = compute_box_constrained_floors(
            problem, process_noise_std=config.process_noise_std, u_max=config.u_max
        )

        assert floors.j_sdp == pytest.approx(floors.j_lqr, rel=1e-2)


class TestCocpLowerBoundBracketsTheFloor:
    """The defining inequality of the reference-bounds plan (Sec 1.4/7.2):
    COCP-LB is a REAL, FEASIBLE policy (the SDP-seeded one-step QP), so its
    attained Monte-Carlo cost can never fall below the SDP's own proven
    floor -- ``J_SDP <= J*_C <= J_COCP-LB``. This is the single regression
    guard that fails loudly if the SDP, the P_lb seeding, or the evaluation
    protocol ever drift apart from each other."""

    def test_cocp_lower_bound_cost_is_at_least_j_sdp(self, tmp_path) -> None:
        config = _fast_config()
        experiment = nb04_box_constrained_experiment(config)
        problem = experiment.problem.build()

        floors = compute_box_constrained_floors(
            problem, process_noise_std=config.process_noise_std, u_max=config.u_max
        )
        report = run_experiment(experiment, root=tmp_path)
        cocp_lb_cost = report.results["cocp_lower_bound"].metrics["eval_expected_cost"]

        assert cocp_lb_cost >= floors.j_sdp - 1e-6, (
            f"COCP-LB's attained cost ({cocp_lb_cost:.6f}) fell below the "
            f"SDP floor J_SDP ({floors.j_sdp:.6f}) -- COCP-LB is a feasible "
            "policy and can never beat its own seeding bound."
        )
