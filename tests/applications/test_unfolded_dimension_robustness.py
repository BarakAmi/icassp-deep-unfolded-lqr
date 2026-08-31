"""NB03 Phase A directive 4 -- the 'm' factor: the whole unfolded backend
stack must be dimension-adaptive purely from the configured problem matrices.
Changing the control dimension ``m`` (and, defensively, the state dimension
``n``) must build and forward every unfolded kind with NO PyTorch
shape/broadcast mismatch -- the failure mode the notebook never previously
verified beyond its single hard-coded ``m=2``.

Kept forward-only and small: this is a shape/plumbing gate (does the tensor
algebra broadcast?), not a convergence assertion (that is the notebook's
production-scale job)."""

import numpy as np
import pytest
import torch

from mbl.applications.factories import GaussianBatchSpec, LQRProblemFactory
from mbl.applications.recipes import UnfoldedKind, UnfoldedRecipe
from mbl.applications.recipes.base import EngineHarness
from mbl.applications.recipes.unfolded import (
    UnfoldedBuildSpec,
    build_unfolded_controller,
)
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.models.analytic.riccati import RiccatiController
from mbl.persistence.null_tracker import NullExperimentTracker

CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)
HORIZON = 5
BATCH = 4

#: (state_dim n, control_dim m) pairs: m sweeps {1,2,3} at fixed n, plus two
#: rectangular n!=m instances so a hidden square-only assumption is caught.
_DIMS = [(4, 1), (4, 2), (4, 3), (3, 2), (5, 3)]
_KINDS = [
    UnfoldedKind.FIXED,
    UnfoldedKind.LEARNED_STEP_SIZE,
    UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
]


def _problem(n: int, m: int):
    return LQRProblemFactory(
        state_dim=n, control_dim=m, horizon=HORIZON, seed=0
    ).build()


def _batch(n: int):
    x0 = torch.randn(BATCH, n, dtype=torch.float64)
    w = 0.1 * torch.randn(BATCH, HORIZON, n, dtype=torch.float64)
    v = torch.zeros(BATCH, HORIZON, n, dtype=torch.float64)
    return x0, w, v


@pytest.mark.parametrize("n,m", _DIMS)
@pytest.mark.parametrize("kind", _KINDS)
def test_forward_pass_is_shape_adaptive(kind: UnfoldedKind, n: int, m: int) -> None:
    """Every unfolded kind builds and forwards at arbitrary ``(n, m)``,
    producing a finite ``(BATCH, HORIZON, m)`` control tensor."""
    controller = build_unfolded_controller(
        _problem(n, m),
        CTX,
        UnfoldedBuildSpec(
            kind=kind,
            num_iterations=3,
            step_size_init=0.02,
            step_size_max=0.3,
            horizon=HORIZON,
        ),
    )
    x0, w, v = _batch(n)
    with torch.no_grad():
        _, _, u, _ = controller.forward(x0, w, v)
    assert u.shape == (BATCH, HORIZON, m)
    assert torch.isfinite(u).all()


@pytest.mark.parametrize("n,m", _DIMS)
def test_learned_recipe_trains_end_to_end_at_any_m(n: int, m: int) -> None:
    """The trainable recipe path (build_controller -> build_engine -> train)
    also runs shape-clean at each ``m`` -- the engine/rollout/cost stack, not
    just the bare forward pass."""
    problem = _problem(n, m)
    recipe = UnfoldedRecipe(
        kind=UnfoldedKind.LEARNED_STEP_SIZE,
        plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=2),
        num_iterations=3,
        step_size_init=0.02,
        step_size_max=0.3,
        horizon=HORIZON,
    )
    harness = EngineHarness(
        batch_spec=GaussianBatchSpec(
            state_dim=n, horizon=HORIZON, batch_size=8, seed=0, process_noise_std=0.1
        ),
        tracker=NullExperimentTracker(),
        ctx=CTX,
    )
    controller = recipe.build_controller(problem, CTX)
    metrics = recipe.build_engine(controller, problem, harness).train()
    assert np.isfinite(metrics["final_loss"])


@pytest.mark.parametrize("n,m", _DIMS)
def test_riccati_gain_shape_matches_control_dim(n: int, m: int) -> None:
    """Sanity anchor for the baseline the unfolded controllers are compared
    against: the Riccati gain is ``(HORIZON, m, n)`` at every dimension."""
    riccati = RiccatiController(_problem(n, m), HORIZON)
    assert np.asarray(riccati.K_arr).shape == (HORIZON, m, n)
