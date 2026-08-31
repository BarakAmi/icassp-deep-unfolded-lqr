"""Shared, mutable run state passed to every Callback hook and TrainingStrategy
call -- the same role as PyTorch Lightning's Trainer state / HF's TrainerState."""

from dataclasses import dataclass, field
from typing import Any

from .config import TrainingConfig
from ..persistence.tracker import ExperimentTracker


@dataclass
class RunContext:
    """Mutable state threaded through one `Runner` run: every `Callback` hook
    and `TrainingStrategy.step`/`evaluate_step` call receives the same
    instance, and may read or update it (e.g. `context.metrics`).

    Attributes:
        model: The model being trained/evaluated (an `nn.Module`, a
            `models.base.Controller`, or any object the active
            `TrainingStrategy` knows how to use).
        tracker: The `ExperimentTracker` this run logs params/metrics/artifacts to.
        config: This run's `TrainingConfig`.
        total_epochs: Total epoch count across all `TrainingPhase`s in the run.
        epoch: The current global epoch index (0-based, continuous across phases).
        phase: The name of the `TrainingPhase` currently running, set by
            `Runner.train` before each phase's first epoch. Written down for
            diagnosis and read by nothing that decides anything: a layer-wise
            family trains through one phase per unfolding iteration with a
            FRESH optimizer each time, and a curve without phase boundaries
            cannot tell "this layer never converged" from "the refinement undid
            it". Defaults to ``""`` for a context built outside `Runner`.
        batch: The current batch index within the epoch.
        metrics: The most recent step's scalar metrics (e.g. ``{"loss": ...}``),
            updated by `Runner.train` after each `TrainingStrategy.step` call.
        is_trainable: Whether any phase's `TrainingStrategy` actually updates
            learnable parameters (see `TrainingStrategy.trains_parameters`),
            derived once by `Runner._build_context`. Defaults to ``True`` so a
            `RunContext` built without going through `Runner` (e.g. directly
            in a test) keeps today's behavior; `ExperimentTrackingCallback`
            uses this to skip logging optimizer-specific metadata for
            non-trainable (e.g. `AnalyticalStrategy`) runs.
        stop_requested: Set by a `Callback` (e.g. `EarlyStoppingCallback`) to
            ask `Runner.train` to stop after the current epoch's
            ``on_epoch_end`` hooks have fired. Defaults to ``False``, so every
            existing run is byte-for-byte unaffected unless a callback opts
            in.
    """

    model: Any
    tracker: ExperimentTracker
    config: TrainingConfig
    total_epochs: int = 0
    epoch: int = 0
    phase: str = ""
    batch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    is_trainable: bool = True
    stop_requested: bool = False
