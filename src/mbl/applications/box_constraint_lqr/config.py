from dataclasses import dataclass

import torch

from ...core.utils import ensure_positive_integer
from ...models.base import Config


@dataclass(frozen=True)
class BoxConstraintLQRConfig(Config):
    """Settings for BoxConstraintLQRApp's LTI system, quadratic cost, noise,
    infinity-norm control bound, and per-model-family training
    hyperparameters.

    dtype is float64 throughout: cvxpylayers/SCS (used by the COCP model)
    needs double precision for numerically reliable solves, and the rest of
    this app matches it so no tensor ever needs a dtype conversion at a
    module boundary.
    """

    state_dim: int = 4
    control_dim: int = 2
    horizon: int = 50
    process_noise_std: float = 0.5
    u_max: float = 1.0
    batch_size: int = 256
    seed: int = 0

    num_unfolding_iterations: int = 10
    step_size_init: float = 0.05
    step_size_max: float = 1.0
    unfolded_learning_rate: float = 0.1
    unfolded_epochs: int = 100

    neural_hidden_dim: int = 64
    neural_learning_rate: float = 1e-3
    neural_epochs: int = 100

    cocp_learning_rate: float = 0.1
    cocp_epochs: int = 50
    cocp_solver_eps: float = 1e-8
    cocp_solver_max_iters: int = 10000
    # COCP solves one convex QP per (batch element, time step) via
    # cvxpylayers/SCS -- orders of magnitude more expensive per sample than
    # the gradient/matrix-multiply-based models. None means "use batch_size
    # / horizon unchanged"; override with smaller values to keep COCP
    # tractable at a batch_size/horizon chosen for the other models.
    cocp_batch_size: int | None = None
    cocp_horizon: int | None = None

    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        ensure_positive_integer(self.state_dim, "state_dim")
        ensure_positive_integer(self.control_dim, "control_dim")
        ensure_positive_integer(self.horizon, "horizon")
        ensure_positive_integer(self.batch_size, "batch_size")
        ensure_positive_integer(
            self.num_unfolding_iterations, "num_unfolding_iterations"
        )
        ensure_positive_integer(self.unfolded_epochs, "unfolded_epochs")
        ensure_positive_integer(self.neural_hidden_dim, "neural_hidden_dim")
        ensure_positive_integer(self.neural_epochs, "neural_epochs")
        ensure_positive_integer(self.cocp_epochs, "cocp_epochs")
        ensure_positive_integer(self.cocp_solver_max_iters, "cocp_solver_max_iters")
        if self.cocp_batch_size is not None:
            ensure_positive_integer(self.cocp_batch_size, "cocp_batch_size")
        if self.cocp_horizon is not None:
            ensure_positive_integer(self.cocp_horizon, "cocp_horizon")
        if self.process_noise_std < 0:
            raise ValueError("process_noise_std must be non-negative.")
        if self.u_max <= 0:
            raise ValueError("u_max must be strictly positive.")
        if self.step_size_init <= 0 or self.step_size_init >= self.step_size_max:
            raise ValueError("step_size_init must lie in (0, step_size_max).")
