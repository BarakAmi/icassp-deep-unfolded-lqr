"""Box-constraint LQR baseline case study: a fully-observable LTI system
with zero-mean Gaussian state noise, a quadratic cost, and an infinity-norm
"box" bound on the control input (|u_t| <= u_max), compared across seven
model families:

    truncated_riccati:                     unconstrained Riccati optimum, clipped onto the box
    cocp:                                   trained one-step convex-QP policy (learnable cost-to-go)
    cocp_lower_bound:                       the same QP policy, fixed at the box-constrained SDP's
                                             cost-to-go (never trained) -- a strong non-learned reference
    neural:                                 RNN policy, automatically clipped via problem.constraints
    unfolded_fixed:                         fixed step-size, true Riccati P, nothing learned
    unfolded_learned_step_size:             learned per-iteration step-size, true Riccati P
    unfolded_learned_step_size_and_matrix:  learned per-iteration step-size + a single
                                             learned time-invariant matrix replacing Riccati P

Since Stage S3 this module is a *declaration* (see `standard_lqr/app.py`):
`box_constraint_lqr_case_study` composes the shared factories and
per-family `ModelRecipe`s; the generic `CaseStudyApplication` chassis
executes it. The former ~500-line monolith -- ≈85% a verbatim copy of
`StandardLQRApp` (M1), routed by string ladders (M2), with the SDP bound
smuggled onto a live module (M10) -- is gone.
"""

from collections.abc import Callable
from typing import Any

from .config import BoxConstraintLQRConfig
from ...persistence.tracker import ExperimentTracker
from ..case_study import CaseStudy, CaseStudyApplication
from ..factories import GaussianBatchSpec, LQRProblemFactory
from ..recipes import (
    COCPLowerBoundRecipe,
    COCPRecipe,
    FixedUnfoldedRecipe,
    NeuralRecipe,
    TruncatedRiccatiRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
)
from ...core.runtime import Backend, ComputeContext, Precision
from ...engine.training_plan import OptimizerSpec, TrainingPlan


def box_constraint_lqr_case_study(config: BoxConstraintLQRConfig) -> CaseStudy:
    """Compose the box-constrained LQR benchmark declaration from `config`.

    Args:
        config: The case study's frozen settings.

    Returns:
        The `CaseStudy` (problem factory + batch spec + seven recipes).
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
    cocp_plan = TrainingPlan(
        optimizer=OptimizerSpec("adam", config.cocp_learning_rate),
        epochs=config.cocp_epochs,
    )
    unfolded_shape: dict[str, Any] = {
        "num_iterations": config.num_unfolding_iterations,
        "step_size_init": config.step_size_init,
        "step_size_max": config.step_size_max,
        "horizon": config.horizon,
    }
    return CaseStudy(
        name="BoxConstraintLQR",
        problem=LQRProblemFactory(
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            horizon=config.horizon,
            seed=config.seed,
            u_max=config.u_max,
        ),
        batch_spec=GaussianBatchSpec(
            state_dim=config.state_dim,
            horizon=config.horizon,
            batch_size=config.batch_size,
            seed=config.seed,
            process_noise_std=config.process_noise_std,
        ),
        recipes=(
            TruncatedRiccatiRecipe(horizon=config.horizon),
            COCPRecipe(
                plan=cocp_plan,
                solver_eps=config.cocp_solver_eps,
                solver_max_iters=config.cocp_solver_max_iters,
                batch_size=config.cocp_batch_size,
                horizon=config.cocp_horizon,
            ),
            COCPLowerBoundRecipe(
                process_noise_std=config.process_noise_std,
                solver_eps=config.cocp_solver_eps,
                solver_max_iters=config.cocp_solver_max_iters,
                # Scored via the shared problem cost, whose Q/R are stacked
                # to the MAIN horizon -- only the batch size may shrink here.
                batch_size=config.cocp_batch_size,
            ),
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


class BoxConstraintLQRApp(CaseStudyApplication):
    """The box-constrained LQR declaration bound to the generic chassis."""

    # A plain class attribute (not the chassis' instance property) so
    # class-level access (e.g. examples/run_baselines.py's sweep labels)
    # keeps yielding the string.
    application_name = "BoxConstraintLQR"

    def __init__(
        self,
        tracker_factory: Callable[[str], ExperimentTracker],
        config: BoxConstraintLQRConfig = BoxConstraintLQRConfig(),
    ) -> None:
        """
        Args:
            tracker_factory: Builds a fresh `ExperimentTracker` per model
                label (see `BaseApplication`).
            config: The case study's frozen settings.
        """
        self.config = config
        super().__init__(box_constraint_lqr_case_study(config), tracker_factory)
