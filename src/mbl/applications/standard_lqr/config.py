from dataclasses import dataclass

import torch

from ...core.utils import ensure_positive_integer
from ...models.base import Config


@dataclass(frozen=True)
class StandardLQRConfig(Config):
    """Settings for StandardLQRApp's LTI system, quadratic cost, noise, and
    per-model-family training hyperparameters.

    dtype is float64 throughout (matching this codebase's established
    convention for LinearSystem/torch interop -- mixed float32/float64
    matmuls between the numpy system matrices and torch parameters raise a
    dtype-mismatch RuntimeError).
    """

    state_dim: int = 4
    control_dim: int = 2
    horizon: int = 50
    process_noise_std: float = 0.5
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
        if self.process_noise_std < 0:
            raise ValueError("process_noise_std must be non-negative.")
        if self.step_size_init <= 0 or self.step_size_init >= self.step_size_max:
            raise ValueError("step_size_init must lie in (0, step_size_max).")
