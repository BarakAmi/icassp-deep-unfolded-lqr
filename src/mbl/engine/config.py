from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.utils import ensure_positive_integer
from ..models.base import Config


@dataclass(frozen=True)
class TrainingConfig(Config):
    """Run-level hyperparameters/metadata, logged via ExperimentTrackingCallback.

    Deliberately holds no `epochs` field: epoch counts are owned per-phase by
    TrainingPhase (a run can compose several phases, each with its own
    strategy and epoch count), so a single flat `epochs` here would be
    misleading. The optimizer instance itself is injected directly into
    whichever TrainingStrategy needs one (e.g. GradientDescentStrategy);
    optimizer_name/optimizer_kwargs here are reproducibility metadata only.

    From Stage S3 the optimizer fields are DERIVED, never hand-assembled:
    trainable runs obtain their config via `TrainingPlan.training_config`,
    so the logged learning rate/optimizer identity is the one the executed
    optimizer was built from (the C4 provenance law). Non-trainable
    (analytical) runs leave `learning_rate` as ``None`` -- they have no
    optimizer to describe, and `ExperimentTrackingCallback` omits the
    optimizer-only fields from their metadata anyway.

    Attributes:
        batch_size: Number of trajectories per training batch.
        learning_rate: Reproducibility metadata for the injected optimizer
            (the optimizer itself is constructed and owned by the caller);
            ``None`` for runs with no optimizer.
        optimizer_name: Reproducibility metadata naming the optimizer type.
        optimizer_kwargs: Reproducibility metadata for the optimizer's
            constructor kwargs.
        log_every: Log metrics every this many epochs (see
            `ExperimentTrackingCallback`).
        seed: Optional random seed, logged for reproducibility.
    """

    batch_size: int
    learning_rate: float | None = None
    optimizer_name: str = "adam"
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    log_every: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        """Fail-fast guards on the run-level hyperparameters.

        Raises:
            ValueError: If `batch_size` or `log_every` is not a positive
                integer, `learning_rate` is set but not positive, or `seed`
                is negative.
        """
        ensure_positive_integer(self.batch_size, "batch_size")
        ensure_positive_integer(self.log_every, "log_every")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")
