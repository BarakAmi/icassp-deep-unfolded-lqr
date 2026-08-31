"""Phase 1 acceptance tests for `applications.uncertainty.resync`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 5.1/9/11): ORACLE resync
reproduces a freshly-built controller's internal gradient-coefficient
tensors to 1e-12; the GRU's resync is a genuine no-op; COCP raises
(NOMINAL-only, Sec 13 decision 2).
"""

import numpy as np
import pytest
import torch

from mbl.applications.factories import LQRProblemFactory
from mbl.applications.recipes.cocp import COCPRecipe
from mbl.applications.recipes.neural import NeuralRecipe
from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    UnfoldedKind,
    build_unfolded_controller,
)
from mbl.applications.recipes.analytic import TruncatedRiccatiRecipe
from mbl.applications.uncertainty.resync import UnsupportedResyncError, resync_to_plant
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan

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


def _problem(seed: int):
    return LQRProblemFactory(
        state_dim=_STATE_DIM,
        control_dim=_CONTROL_DIM,
        horizon=_HORIZON,
        seed=seed,
        u_max=_U_MAX,
    ).build()


def _build_spec(kind: UnfoldedKind) -> UnfoldedBuildSpec:
    return UnfoldedBuildSpec(
        kind=kind,
        num_iterations=8,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=_HORIZON,
    )


class TestStepSizeRefinementResync:
    """The `unfolded_fixed`/`unfolded` (no-matrix) family: resync re-solves
    Riccati and must reproduce a fresh build's M_stack/C_stack exactly."""

    def test_oracle_resync_matches_fresh_build(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()

        controller = build_unfolded_controller(
            nominal_problem, ctx, _build_spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        resync_to_plant(controller, perturbed_problem, ctx)

        fresh = build_unfolded_controller(
            perturbed_problem, ctx, _build_spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )

        refinement = controller.config.iterative_refinement
        fresh_refinement = fresh.config.iterative_refinement
        torch.testing.assert_close(
            refinement.static_parameters["M_stack"],
            fresh_refinement.static_parameters["M_stack"],
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            refinement.static_parameters["C_stack"],
            fresh_refinement.static_parameters["C_stack"],
            atol=1e-12,
            rtol=0,
        )

    def test_nominal_mode_leaves_it_bit_identical(self) -> None:
        """NOMINAL mode is simply "never call resync" -- verified by omission:
        the static parameters must stay exactly what the nominal build produced."""
        nominal_problem = _problem(seed=0)
        ctx = _ctx()
        controller = build_unfolded_controller(
            nominal_problem, ctx, _build_spec(UnfoldedKind.LEARNED_STEP_SIZE)
        )
        before = controller.config.iterative_refinement.static_parameters[
            "M_stack"
        ].clone()
        # No resync call -- NOMINAL mode.
        after = controller.config.iterative_refinement.static_parameters["M_stack"]
        torch.testing.assert_close(before, after, atol=0, rtol=0)


class TestRiccatiRefinementResync:
    """The `unfolded_alpha_p` (learned-matrix) family: resync overwrites only
    A/B; the learned matrix itself is untouched."""

    def test_oracle_resync_updates_a_b_matches_fresh_build(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()

        controller = build_unfolded_controller(
            nominal_problem, ctx, _build_spec(UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX)
        )
        resync_to_plant(controller, perturbed_problem, ctx)

        fresh = build_unfolded_controller(
            perturbed_problem,
            ctx,
            _build_spec(UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX),
        )

        refinement = controller.config.iterative_refinement
        fresh_refinement = fresh.config.iterative_refinement
        torch.testing.assert_close(
            refinement.static_parameters["A"],
            fresh_refinement.static_parameters["A"],
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            refinement.static_parameters["B"],
            fresh_refinement.static_parameters["B"],
            atol=1e-12,
            rtol=0,
        )

    def test_learned_matrix_is_never_touched(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()
        controller = build_unfolded_controller(
            nominal_problem, ctx, _build_spec(UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX)
        )
        riccati_matrix_before = (
            controller.config.parameters["riccati_matrix"].get().clone().detach()
        )
        resync_to_plant(controller, perturbed_problem, ctx)
        riccati_matrix_after = (
            controller.config.parameters["riccati_matrix"].get().detach()
        )
        torch.testing.assert_close(
            riccati_matrix_before, riccati_matrix_after, atol=0, rtol=0
        )

    def test_r_is_never_perturbed(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()
        controller = build_unfolded_controller(
            nominal_problem, ctx, _build_spec(UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX)
        )
        r_before = controller.config.iterative_refinement.static_parameters["R"].clone()
        resync_to_plant(controller, perturbed_problem, ctx)
        r_after = controller.config.iterative_refinement.static_parameters["R"]
        torch.testing.assert_close(r_before, r_after, atol=0, rtol=0)


class TestTruncatedRiccatiResync:
    def test_oracle_resync_matches_fresh_solve(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()
        recipe = TruncatedRiccatiRecipe(horizon=_HORIZON)
        controller = recipe.build_controller(nominal_problem, ctx)
        resync_to_plant(controller, perturbed_problem, ctx)
        fresh = recipe.build_controller(perturbed_problem, ctx)
        np.testing.assert_allclose(controller.P_arr, fresh.P_arr, atol=1e-10)
        np.testing.assert_allclose(controller.K_arr, fresh.K_arr, atol=1e-10)


class TestNeuralResyncIsNoOp:
    def test_gru_resync_does_not_raise_and_changes_nothing(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()
        plan = TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=1)
        recipe = NeuralRecipe(hidden_dim=8, plan=plan, init_seed=0)
        controller = recipe.build_controller(nominal_problem, ctx)
        state_before = {k: v.clone() for k, v in controller.state_dict().items()}
        resync_to_plant(controller, perturbed_problem, ctx)
        state_after = controller.state_dict()
        for key, value in state_before.items():
            torch.testing.assert_close(value, state_after[key], atol=0, rtol=0)


class TestCOCPRaises:
    def test_oracle_resync_raises(self) -> None:
        nominal_problem = _problem(seed=0)
        perturbed_problem = _problem(seed=1)
        ctx = _ctx()
        plan = TrainingPlan(optimizer=OptimizerSpec("adam", 0.1), epochs=1)
        recipe = COCPRecipe(plan=plan)
        controller = recipe.build_controller(nominal_problem, ctx)
        with pytest.raises(UnsupportedResyncError):
            resync_to_plant(controller, perturbed_problem, ctx)
