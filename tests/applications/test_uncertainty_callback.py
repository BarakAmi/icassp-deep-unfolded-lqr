"""Phase 2 acceptance tests for `applications.uncertainty.callback`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 5.2/9/11): the plant the
rollout simulates at epoch k IS the drawn one (asserted on the live system,
not on the spec); `resample_per_epoch=False` gives a constant plant; `ORACLE`
resync reproduces a freshly-built controller's gradient-coefficient tensors
to 1e-12; `NOMINAL` leaves them bit-identical; Q/R buffers are provably
untouched; the nominal plant is restored at `on_train_end`.
"""

import numpy as np
import torch

from mbl.applications.factories import LQRProblemFactory
from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    UnfoldedKind,
    build_unfolded_controller,
)
from mbl.applications.rollout import RolloutModel
from mbl.applications.uncertainty.callback import DomainRandomizationCallback
from mbl.applications.uncertainty.perturbations import (
    PerturbationDistribution,
    PerturbationKind,
    PlantPerturbation,
)
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.config import TrainingConfig
from mbl.engine.context import RunContext
from mbl.persistence.null_tracker import NullExperimentTracker

_STATE_DIM = 4
_CONTROL_DIM = 2
_HORIZON = 20
_U_MAX = 0.5


def _ctx() -> ComputeContext:
    return ComputeContext(
        backend=Backend.TORCH,
        device="cpu",
        precision=Precision.from_torch_dtype(torch.float64),
    )


def _problem(seed: int = 0):
    return LQRProblemFactory(
        state_dim=_STATE_DIM,
        control_dim=_CONTROL_DIM,
        horizon=_HORIZON,
        seed=seed,
        u_max=_U_MAX,
    ).build()


def _spec(kind: UnfoldedKind) -> UnfoldedBuildSpec:
    return UnfoldedBuildSpec(
        kind=kind,
        num_iterations=6,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=_HORIZON,
    )


def _context(controller) -> RunContext:
    rollout = RolloutModel(controller)
    return RunContext(
        model=rollout,
        tracker=NullExperimentTracker(),
        config=TrainingConfig(batch_size=8),
    )


class TestPlantIsActuallyRewritten:
    def test_resample_per_epoch_changes_the_live_system(self) -> None:
        problem = _problem()
        controller = build_unfolded_controller(
            problem, _ctx(), _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        context = _context(controller)
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ADDITIVE, level=0.2, seed=1
                ),
                resample_per_epoch=True,
            ),
            ctx=_ctx(),
            resync_controller=False,
        )
        callback.on_train_start(context)
        nominal_A = context.model.problem.system.A_t.array.copy()

        context.epoch = 0
        callback.on_epoch_start(context)
        A_epoch0 = context.model.problem.system.A_t.array.copy()
        context.epoch = 1
        callback.on_epoch_start(context)
        A_epoch1 = context.model.problem.system.A_t.array.copy()

        assert not np.array_equal(nominal_A, A_epoch0)
        assert not np.array_equal(A_epoch0, A_epoch1)

    def test_fixed_perturbation_holds_the_same_plant(self) -> None:
        problem = _problem()
        controller = build_unfolded_controller(
            problem, _ctx(), _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        context = _context(controller)
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ROTATION, level=30.0, seed=1
                ),
                resample_per_epoch=False,
            ),
            ctx=_ctx(),
            resync_controller=False,
        )
        callback.on_train_start(context)
        context.epoch = 0
        callback.on_epoch_start(context)
        A_epoch0 = context.model.problem.system.A_t.array.copy()
        context.epoch = 1
        callback.on_epoch_start(context)
        A_epoch1 = context.model.problem.system.A_t.array.copy()
        np.testing.assert_array_equal(A_epoch0, A_epoch1)


class TestOracleResyncMatchesFreshBuild:
    def test_step_size_refinement(self) -> None:
        problem = _problem(seed=0)
        ctx = _ctx()
        controller = build_unfolded_controller(
            problem, ctx, _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        context = _context(controller)
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ADDITIVE, level=0.2, seed=7
                ),
                resample_per_epoch=True,
            ),
            ctx=ctx,
            resync_controller=True,
        )
        callback.on_train_start(context)
        context.epoch = 0
        callback.on_epoch_start(context)

        perturbed_A = context.model.problem.system.A_t.array
        perturbed_B = context.model.problem.system.B_t.array
        from mbl.core.system.linear_system import LinearSystem
        from mbl.core.optimal_control_problem import OptimalControlProblem

        perturbed_problem = OptimalControlProblem(
            system=LinearSystem.fully_observable(perturbed_A, perturbed_B),
            cost=problem.cost,
            constraints=problem.constraints,
        )
        fresh = build_unfolded_controller(
            perturbed_problem, ctx, _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )

        torch.testing.assert_close(
            controller.config.iterative_refinement.static_parameters["M_stack"],
            fresh.config.iterative_refinement.static_parameters["M_stack"],
            atol=1e-12,
            rtol=0,
        )


class TestNominalModeLeavesControllerUntouched:
    def test_step_size_refinement_unchanged(self) -> None:
        problem = _problem()
        ctx = _ctx()
        controller = build_unfolded_controller(
            problem, ctx, _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        context = _context(controller)
        before = controller.config.iterative_refinement.static_parameters[
            "M_stack"
        ].clone()
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ADDITIVE, level=0.2, seed=7
                ),
                resample_per_epoch=True,
            ),
            ctx=ctx,
            resync_controller=False,
        )
        callback.on_train_start(context)
        context.epoch = 0
        callback.on_epoch_start(context)
        after = controller.config.iterative_refinement.static_parameters["M_stack"]
        torch.testing.assert_close(before, after, atol=0, rtol=0)


class TestQRUntouched:
    def test_rollout_model_cost_buffers_never_change(self) -> None:
        problem = _problem()
        ctx = _ctx()
        controller = build_unfolded_controller(
            problem, ctx, _spec(UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX)
        )
        context = _context(controller)
        q_before = context.model._Q.clone()
        r_before = context.model._R.clone()
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ROTATION, level=30.0, seed=1
                ),
            ),
            ctx=ctx,
            resync_controller=True,
        )
        callback.on_train_start(context)
        for epoch in range(3):
            context.epoch = epoch
            callback.on_epoch_start(context)
        torch.testing.assert_close(q_before, context.model._Q, atol=0, rtol=0)
        torch.testing.assert_close(r_before, context.model._R, atol=0, rtol=0)


class TestNominalRestoredAtTrainEnd:
    def test_train_end_restores_nominal_plant(self) -> None:
        problem = _problem()
        ctx = _ctx()
        controller = build_unfolded_controller(
            problem, ctx, _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        context = _context(controller)
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ADDITIVE, level=0.3, seed=3
                ),
                resample_per_epoch=True,
            ),
            ctx=ctx,
            resync_controller=False,
        )
        callback.on_train_start(context)
        nominal_A = context.model.problem.system.A_t.array.copy()
        for epoch in range(3):
            context.epoch = epoch
            callback.on_epoch_start(context)
        assert not np.array_equal(nominal_A, context.model.problem.system.A_t.array)
        callback.on_train_end(context)
        np.testing.assert_array_equal(nominal_A, context.model.problem.system.A_t.array)


class TestHistoryRecorded:
    def test_one_history_row_per_epoch(self) -> None:
        problem = _problem()
        ctx = _ctx()
        controller = build_unfolded_controller(
            problem, ctx, _spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        context = _context(controller)
        callback = DomainRandomizationCallback(
            PerturbationDistribution(
                perturbation=PlantPerturbation(
                    kind=PerturbationKind.ROTATION, level=15.0, seed=1
                ),
                resample_per_epoch=False,
            ),
            ctx=ctx,
            resync_controller=False,
        )
        callback.on_train_start(context)
        for epoch in range(4):
            context.epoch = epoch
            callback.on_epoch_start(context)
        assert len(callback._history) == 4
        assert callback._history[0]["angle_degrees"] == 15.0
        callback.on_train_end(context)
        assert "domain_randomization_history" in context.tracker.artifact_names
