"""Regression tests for the NB03 Phase A root-cause fix: the
`trajectory_controls`/`trajectory_states` artifact must be rolled out on a
FIXED, reproducible evaluation batch that is INDEPENDENT of training length,
so every contender's and every depth's trajectory is directly comparable and
the exact batch is reconstructible outside training.

Before the fix, `TrajectoryLoggingCallback` reused the advancing TRAINING
sampler, so the persisted trajectory was drawn only after every training step
had advanced the generator -- its contents therefore depended on the training
length (e.g. the layer-wise contender's ``K*warmup + refinement``, different
at every depth K). The notebook's separately-drawn Riccati baseline matched
none of them, and the "does u_k converge to Riccati as depth grows?" figures
showed a persistent ~0.37 relative error even for an essentially-converged
controller. These tests encode both halves: the determinism invariant, and
the convergence it restores.
"""

import numpy as np
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

STATE_DIM, CONTROL_DIM, HORIZON = 3, 2, 6
CTX = ComputeContext(backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64)


class _CapturingTracker(NullExperimentTracker):
    """`NullExperimentTracker` that additionally keeps saved artifact VALUES
    in memory (the base only records names) -- the trajectory-logging
    callback's ``trajectory_states`` is all these tests read back."""

    def __init__(self) -> None:
        super().__init__()
        self.artifacts: dict[str, object] = {}

    def save_artifact(self, name, obj):  # noqa: ANN001
        self.artifacts[name] = obj
        return super().save_artifact(name, obj)


def _problem():
    return LQRProblemFactory(
        state_dim=STATE_DIM, control_dim=CONTROL_DIM, horizon=HORIZON, seed=0
    ).build()


def _batch_spec() -> GaussianBatchSpec:
    return GaussianBatchSpec(
        state_dim=STATE_DIM,
        horizon=HORIZON,
        batch_size=8,
        seed=0,
        process_noise_std=0.3,
        initial_state_std=1.0,
    )


def _recipe(epochs: int) -> UnfoldedRecipe:
    return UnfoldedRecipe(
        kind=UnfoldedKind.LEARNED_STEP_SIZE,
        plan=TrainingPlan(optimizer=OptimizerSpec("adam", 0.05), epochs=epochs),
        num_iterations=3,
        step_size_init=0.02,
        step_size_max=0.5,
        horizon=HORIZON,
    )


def _trajectory_states_after_training(epochs: int) -> np.ndarray:
    problem = _problem()
    recipe = _recipe(epochs)
    tracker = _CapturingTracker()
    harness = EngineHarness(batch_spec=_batch_spec(), tracker=tracker, ctx=CTX)
    controller = recipe.build_controller(problem, CTX)
    recipe.build_engine(controller, problem, harness).train()
    return np.asarray(tracker.artifacts["trajectory_states"])


class TestTrajectoryBatchDeterminism:
    def test_batch_is_independent_of_training_length(self) -> None:
        """The persisted rollout's INITIAL states (``X[:, 0]`` == the sampled
        ``x0``) are byte-identical across two runs of different epoch counts:
        the trajectory batch no longer rides the advancing training
        generator."""
        short = _trajectory_states_after_training(epochs=2)
        long = _trajectory_states_after_training(epochs=8)
        assert np.array_equal(short[:, 0], long[:, 0])

    def test_batch_equals_an_independent_first_draw(self) -> None:
        """That fixed batch is exactly the first draw of a FRESH sampler built
        from the same (seeded) spec -- i.e. reconstructible outside training,
        the property the notebook's Riccati baseline relies on."""
        states = _trajectory_states_after_training(epochs=4)
        sampler, _ = _batch_spec().build(Backend.TORCH, torch_dtype=torch.float64)
        x0, _, _ = sampler()
        assert np.allclose(states[:, 0], x0.numpy())


class TestConvergenceRestoredOnSharedBatch:
    """The invariant the fix exists to serve: on the SHARED, reproducible
    eval batch, a deep (essentially converged) fixed unrolled controller's
    control matches Riccati's to machine precision -- the alignment the
    mismatched-batch bug destroyed."""

    def test_deep_fixed_unfolded_matches_riccati_on_shared_batch(self) -> None:
        problem = _problem()
        riccati = RiccatiController(problem, HORIZON)
        sampler, _ = _batch_spec().build(Backend.TORCH, torch_dtype=torch.float64)
        x0, w, _ = sampler()
        obs = problem.system.dimensions.observation_dim
        v_t = torch.zeros(x0.shape[0], HORIZON, obs, dtype=torch.float64)

        deep = build_unfolded_controller(
            problem,
            CTX,
            UnfoldedBuildSpec(
                kind=UnfoldedKind.FIXED,
                num_iterations=200,
                step_size_init=0.05,
                step_size_max=0.3,
                horizon=HORIZON,
            ),
        )
        with torch.no_grad():
            _, _, u_unfolded = problem.system.run(deep.get_control_policy(), x0, w, v_t)
            _, _, u_riccati = problem.system.run(
                riccati.get_control_policy(),
                x0.numpy(),
                w.numpy(),
                v_t.numpy(),
            )
        u_unfolded = np.asarray(u_unfolded)
        u_riccati = np.asarray(u_riccati)
        rel = np.linalg.norm(u_unfolded - u_riccati) / np.linalg.norm(u_riccati)
        assert rel < 1e-6
