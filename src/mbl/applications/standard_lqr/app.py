"""Standard LQR baseline case study: a fully-observable LTI system with
zero-mean Gaussian state noise and a quadratic cost, compared across five
model families -- the analytic Riccati optimum, an RNN policy, and three
unfolded (deep-unrolled) gradient-descent configurations:

    unfolded_fixed:                        fixed step-size, true Riccati P, nothing learned
    unfolded_learned_step_size:            learned per-iteration step-size, true Riccati P
    unfolded_learned_step_size_and_matrix: learned per-iteration step-size + a single
                                            learned time-invariant matrix replacing Riccati P

Since Stage S3 this module is a *declaration*: `standard_lqr_case_study`
composes the shared problem factory, batch spec, and per-family
`ModelRecipe`s into a `CaseStudy`, and `StandardLQRApp` is a thin binding
of that declaration to the generic `CaseStudyApplication` chassis. The
former ~400-line monolith body (problem construction, model recipes,
sampler factories, engine wiring, ``if name == ...`` routing -- M1/M2) is
gone; adding a model family appends one recipe here, editing nothing else.
"""

from collections.abc import Callable
from typing import Any

from .config import StandardLQRConfig
from ...persistence.tracker import ExperimentTracker
from ..case_study import CaseStudy, CaseStudyApplication
from ..factories import GaussianBatchSpec, LQRProblemFactory
from ..recipes import (
    NeuralRecipe,
    RiccatiRecipe,
    FixedUnfoldedRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
)
from ...core.runtime import Backend, ComputeContext, Precision
from ...engine.training_plan import OptimizerSpec, TrainingPlan


def standard_lqr_case_study(config: StandardLQRConfig) -> CaseStudy:
    """Compose the standard-LQR benchmark declaration from `config`.

    Args:
        config: The case study's frozen settings.

    Returns:
        The `CaseStudy` (problem factory + batch spec + five recipes).
    """
    ctx = ComputeContext(
        backend=Backend.TORCH,
        device="cpu",
        precision=Precision.from_torch_dtype(config.dtype),
    )
    # Per-family declarative training plans (T3.b): each family's optimizer
    # is BUILT from -- and its provenance logged from -- its own plan (C4).
    unfolded_plan = TrainingPlan(
        optimizer=OptimizerSpec("adam", config.unfolded_learning_rate),
        epochs=config.unfolded_epochs,
    )
    neural_plan = TrainingPlan(
        optimizer=OptimizerSpec("adam", config.neural_learning_rate),
        epochs=config.neural_epochs,
    )
    unfolded_shape: dict[str, Any] = {
        "num_iterations": config.num_unfolding_iterations,
        "step_size_init": config.step_size_init,
        "step_size_max": config.step_size_max,
        "horizon": config.horizon,
    }
    return CaseStudy(
        name="StandardLQR",
        problem=LQRProblemFactory(
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            horizon=config.horizon,
            seed=config.seed,
        ),
        batch_spec=GaussianBatchSpec(
            state_dim=config.state_dim,
            horizon=config.horizon,
            batch_size=config.batch_size,
            seed=config.seed,
            process_noise_std=config.process_noise_std,
        ),
        recipes=(
            RiccatiRecipe(horizon=config.horizon),
            NeuralRecipe(hidden_dim=config.neural_hidden_dim, plan=neural_plan),
            FixedUnfoldedRecipe(**unfolded_shape),
            UnfoldedRecipe(
                kind=UnfoldedKind.LEARNED_STEP_SIZE,
                plan=unfolded_plan,
                **unfolded_shape,
            ),
            UnfoldedRecipe(
                kind=UnfoldedKind.LEARNED_STEP_SIZE_AND_MATRIX,
                plan=unfolded_plan,
                **unfolded_shape,
            ),
        ),
        ctx=ctx,
    )


class StandardLQRApp(CaseStudyApplication):
    """The standard-LQR declaration bound to the generic chassis."""

    # A plain class attribute (not the chassis' instance property) so
    # class-level access (e.g. examples/run_baselines.py's sweep labels)
    # keeps yielding the string.
    application_name = "StandardLQR"

    def __init__(
        self,
        tracker_factory: Callable[[str], ExperimentTracker],
        config: StandardLQRConfig = StandardLQRConfig(),
    ) -> None:
        """
        Args:
            tracker_factory: Builds a fresh `ExperimentTracker` per model
                label (see `BaseApplication`).
            config: The case study's frozen settings.
        """
        self.config = config
        super().__init__(standard_lqr_case_study(config), tracker_factory)
