"""Lifecycle hooks the Runner fires around each phase/epoch/batch. Concrete
side effects (tracking, checkpointing, early stopping, ...) live here as
Callbacks instead of being hardcoded into the training loop -- new behavior is
added by writing a new Callback, not by editing the Runner (Open/Closed
Principle).
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from .context import RunContext
from .training_plan import TrainingState
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.runtime import ComputeContext
from ..core.utils import to_numpy
from ..core.profiling import (
    COLLECTOR,
    cpu_clock,
    current_peak_ram_bytes,
    peak_gpu_memory_bytes,
    reset_gpu_peak_memory,
    wall_clock,
)
from ..core.utils.signing import (
    Signable,
    compute_run_signature,
    flatten_signature,
)
from ..models.base import Controller

logger = logging.getLogger(__name__)

#: The artifact name `MetricsHistoryCallback` files the per-epoch history under,
#: and the one a reader asks a tracker for. Defined here, where the writer is:
#: `persistence.RecordingExperimentTracker` deliberately takes the name as an
#: argument rather than importing it, because `engine` already imports
#: `persistence.tracker` and the reverse edge would close a cycle.
METRICS_HISTORY = "metrics_history"


class Callback:
    """Base class: every hook defaults to a no-op, so subclasses override only
    what they need. Every hook receives the same `RunContext` instance the
    `Runner` is currently using, and may read or mutate it."""

    def on_train_start(self, context: RunContext) -> None:
        """Fired once, before the first epoch of `Runner.train`.

        Args:
            context: The current `RunContext`.
        """
        pass

    def on_epoch_start(self, context: RunContext) -> None:
        """Fired at the start of every epoch, before `context.epoch` batches.

        Args:
            context: The current `RunContext`.
        """
        pass

    def on_batch_start(self, context: RunContext) -> None:
        """Fired before each batch's `TrainingStrategy.step` call.

        Args:
            context: The current `RunContext`.
        """
        pass

    def on_batch_end(self, context: RunContext) -> None:
        """Fired after each batch's `TrainingStrategy.step` call, with
        `context.metrics` already updated.

        Args:
            context: The current `RunContext`.
        """
        pass

    def on_epoch_end(self, context: RunContext) -> None:
        """Fired at the end of every epoch.

        Args:
            context: The current `RunContext`.
        """
        pass

    def on_train_end(self, context: RunContext) -> None:
        """Fired once, after the last epoch of `Runner.train`.

        Args:
            context: The current `RunContext`.
        """
        pass


#: TrainingConfig fields that only mean something when a TrainingStrategy
#: actually trains parameters (see RunContext.is_trainable) -- logging these
#: for a non-trainable (e.g. AnalyticalStrategy) run is actively misleading,
#: not just extraneous (Phase 1D: this is exactly the flaw that made a
#: RiccatiController run's metadata.json report a fake "learning_rate": 1.0).
#: "seed" and "log_every" are included too (Phase 1E): the ground-truth seeds
#: are already logged per-distribution under ``sampling.*.seed``, and
#: "log_every" only governs a training loop's logging cadence.
_OPTIMIZER_ONLY_CONFIG_KEYS = frozenset(
    {"learning_rate", "optimizer_name", "optimizer_kwargs", "seed", "log_every"}
)


def _is_logging_epoch(context: RunContext) -> bool:
    """Whether `context.epoch` is on `context.config.log_every`'s cadence, or
    is the run's final epoch (always logged regardless of cadence) -- the
    shared rule `ExperimentTrackingCallback` and `StructuredTrainingLogCallback`
    both apply, factored out so the tracked metrics and the narrated log rows
    can never drift onto different epochs.

    Args:
        context: The current `RunContext`.

    Returns:
        ``True`` iff this epoch should be logged.
    """
    is_final_epoch = context.epoch == context.total_epochs - 1
    return context.epoch % context.config.log_every == 0 or is_final_epoch


def is_scheduled_log_epoch(
    epoch: int, total_epochs: int, *, first_epochs: int, every_epochs: int
) -> bool:
    """The dynamic narration schedule (NB03 Phase B, directive 2): log the
    first `first_epochs` epochs consecutively, then every `every_epochs`-th
    epoch, and ALWAYS the final epoch -- so a reader sees the fast early
    transient in full, a coarse trace through the long tail, and the final
    converged values, without drowning in every intermediate row.

    Args:
        epoch: The 0-indexed epoch just completed.
        total_epochs: The run's total epoch count (the final epoch is
            ``total_epochs - 1``).
        first_epochs: ``a`` -- how many leading epochs to log consecutively.
        every_epochs: ``b`` -- the cadence after the leading block (every
            ``b``-th epoch by absolute index).

    Returns:
        ``True`` iff `epoch` should be narrated under this schedule.
    """
    is_final = epoch == total_epochs - 1
    return epoch < first_epochs or epoch % every_epochs == 0 or is_final


def format_structured_log_row(
    epoch: int,
    cost: float,
    parameters_block: str,
    *,
    unfolding_depth: int | None = None,
) -> str:
    """The single formatter both the live `StructuredTrainingLogCallback` and
    the cached `workbench.replay.render_training_log_replay` render each row
    through, so a replayed trace is byte-identical to the live one (modulo the
    `logging` timestamp prefix). The header is one line; each learned
    parameter occupies its own line(s) below it (directive 2's readability
    requirement), already assembled into `parameters_block`.

    Args:
        epoch: The 0-indexed epoch; DISPLAYED 1-indexed (``Epoch: 1`` is the
            first epoch), the natural reading for a human-facing trace -- the
            persisted history keeps the raw 0-indexed value.
        cost: The reported training cost/loss.
        parameters_block: The pre-rendered, newline-separated per-parameter
            block (see `render_parameter_block`); ``""`` for a run with no
            learned parameters.
        unfolding_depth: The unrolled depth ``J`` this run trains at, shown at
            the HEAD of the row (before the epoch and cost); omitted when
            ``None`` (a run with no unrolling has no such depth).

    Returns:
        The multi-line row string (header alone if `parameters_block` empty).
    """
    header = ""
    if unfolding_depth is not None:
        header += f"Unfolding Depth: {unfolding_depth} | "
    header += f"Epoch: {epoch + 1:>4d} | Cost (Loss): {cost:.6f}"
    return f"{header}\n{parameters_block}" if parameters_block else header


def render_parameter_block(
    parameter_summaries: Mapping[str, Callable[[], str]],
) -> str:
    """Render each learned parameter on its OWN distinct, indented line(s)
    (directive 2): a scalar/vector summary inline after ``name =``; a
    multi-line value (e.g. a full matrix) on the following indented lines, so
    a `` P `` matrix prints in full and aligned rather than crammed onto one
    line.

    Each summary callable is invoked fresh here (never a value snapshotted at
    construction time), so the rendered block always reflects live training
    progress.

    Args:
        parameter_summaries: name -> zero-argument callable returning the
            parameter's current formatted value (possibly multi-line).

    Returns:
        The assembled block, or ``""`` when there are no summaries.
    """
    lines: list[str] = []
    for name, summary in parameter_summaries.items():
        value = summary()
        if "\n" in value:
            body = "\n".join(f"        {line}" for line in value.splitlines())
            lines.append(f"    {name} =\n{body}")
        else:
            lines.append(f"    {name} = {value}")
    return "\n".join(lines)


def _loggable_config(context: RunContext) -> dict[str, Any]:
    """`context.config`'s hyperparameters, omitting the optimizer-only fields
    (see `_OPTIMIZER_ONLY_CONFIG_KEYS`) when `context.is_trainable` is
    ``False`` -- the shared filter `ExperimentTrackingCallback` and
    `StructuredTrainingLogCallback` both apply, so a non-trainable run's
    persisted params and narrated hyperparameter line never disagree.

    Args:
        context: The current `RunContext`.

    Returns:
        The filtered hyperparameter mapping.
    """
    params = context.config.get_config()
    if not context.is_trainable:
        return {
            key: value
            for key, value in params.items()
            if key not in _OPTIMIZER_ONLY_CONFIG_KEYS
        }
    return dict(params)


class ExperimentTrackingCallback(Callback):
    """Logs the run's hyperparameters once at the start, then logs metrics every
    `log_every` epochs (always including the final epoch), via the
    ExperimentTracker on the RunContext."""

    def on_train_start(self, context: RunContext) -> None:
        """Log `context.config`'s hyperparameters via `context.tracker`,
        omitting optimizer-specific fields (and, for non-trainable runs,
        `log_every`/`seed`) when `context.is_trainable` is ``False`` --
        `batch_size` alone describes the rollout and is kept regardless.

        Args:
            context: The current `RunContext`.
        """
        context.tracker.log_params(_loggable_config(context))

    def on_epoch_end(self, context: RunContext) -> None:
        """Log `context.metrics` via `context.tracker` if this epoch is on the
        `log_every` cadence or is the final epoch.

        The `"epoch"` key itself is omitted when `context.is_trainable` is
        ``False``: an analytical (N=1 rollout) run has no epoch-level
        training curve, so logging an epoch index is redundant noise.

        Args:
            context: The current `RunContext`.
        """
        if _is_logging_epoch(context):
            metrics = (
                {"epoch": context.epoch, **context.metrics}
                if context.is_trainable
                else dict(context.metrics)
            )
            context.tracker.log_metrics(metrics)


class ComputeContextAnnouncementCallback(Callback):
    """Announces the resolved hardware/backend/precision triple once, at the
    start of every run -- hardware transparency (a reader of `experiment.log`
    must never have to guess which device produced a result). `ComputeContext`
    is already signature-relevant (`ComputeContext.get_signature`), so two
    runs differing only by device are already distinct cache entries; this
    callback makes that same fact visible in the run's own narrated log, not
    just in its signature.
    """

    def __init__(self, ctx: ComputeContext) -> None:
        """
        Args:
            ctx: The `ComputeContext` this run executes under.
        """
        self.ctx = ctx

    def on_train_start(self, context: RunContext) -> None:
        """Log the device/backend/precision this run resolved to.

        Args:
            context: The current `RunContext`.
        """
        logger.info(
            "[ComputeContext] Initiating training on DEVICE: %s "
            "(backend=%s, precision=%s)",
            self.ctx.device,
            self.ctx.backend_enum.value,
            self.ctx.precision_enum.value,
        )


class StructuredTrainingLogCallback(Callback):
    """Narrates the same per-epoch cadence `ExperimentTrackingCallback`
    persists as structured metrics (`_is_logging_epoch`), as an academic-
    grade, human-readable training trace: one hyperparameter-assumptions
    line at `on_train_start`, then one aligned row every logged epoch of the
    form ``Epoch: <n> | Cost (Loss): <loss> | Learned Parameters: <summary>``.
    Both land in ``experiment.log`` via the standard `logging` channel (the
    per-run binding in `experiments.run_logging`), so a reader can follow a
    training run's progress from the log alone, without cross-referencing
    ``metadata.json``.

    Every logged row is also accumulated in memory and persisted as an
    ``artifact_name`` DataFrame artifact at `on_train_end` --
    `workbench.replay.render_training_log_replay` replays these exact rows
    for a cache-hit session, so this callback's live console narration and a
    later replay are never a different experience.
    """

    def __init__(
        self,
        parameter_summaries: Mapping[str, Callable[[], str]],
        metric_key: str = "loss",
        artifact_name: str = "training_log_history",
        *,
        log_first_epochs: int = 10,
        log_every_epochs: int = 20,
        unfolding_depth: int | None = None,
    ) -> None:
        """
        Args:
            parameter_summaries: name -> zero-argument callable returning
                that learned parameter's CURRENT formatted value, evaluated
                fresh at every logged epoch (never a value snapshotted once
                at construction time, which would silently go stale as
                training proceeds).
            metric_key: The `context.metrics` key reported as "Cost (Loss)".
            artifact_name: The artifact name the accumulated rows are saved
                under at `on_train_end`.
            log_first_epochs: ``a`` of the dynamic schedule (directive 2) --
                the leading epochs narrated consecutively. This callback owns
                its OWN narration cadence (via `is_scheduled_log_epoch`),
                independent of `ExperimentTrackingCallback`'s metric-
                persistence cadence (`_is_logging_epoch`).
            log_every_epochs: ``b`` of the dynamic schedule -- the cadence
                after the leading block. The final epoch is ALWAYS narrated.
            unfolding_depth: The unrolled depth ``J`` this run trains at, shown
                in each narrated header (and stored per history row) beside the
                epoch and cost; ``None`` for a run with no unrolling.

        Raises:
            ValueError: If `log_first_epochs` is negative or `log_every_epochs`
                is not a positive integer.
        """
        if log_first_epochs < 0:
            raise ValueError(
                f"log_first_epochs must be non-negative, got {log_first_epochs}."
            )
        if log_every_epochs <= 0:
            raise ValueError(
                f"log_every_epochs must be a positive integer, got {log_every_epochs}."
            )
        self.parameter_summaries = parameter_summaries
        self.metric_key = metric_key
        self.artifact_name = artifact_name
        self.log_first_epochs = log_first_epochs
        self.log_every_epochs = log_every_epochs
        self.unfolding_depth = unfolding_depth
        self._history: list[dict[str, Any]] = []

    def on_train_start(self, context: RunContext) -> None:
        """Log this run's hyperparameter assumptions once.

        Args:
            context: The current `RunContext`.
        """
        logger.info("[Hyperparameters] %s", _loggable_config(context))

    def on_epoch_end(self, context: RunContext) -> None:
        """Log (and accumulate) one structured, multi-line row if this epoch
        is on THIS callback's own dynamic narration schedule
        (`is_scheduled_log_epoch`: first ``a``, then every ``b``, always the
        final epoch).

        Args:
            context: The current `RunContext`.

        Raises:
            ValueError: If `metric_key` is absent from `context.metrics` --
                a configured-key mismatch, diagnosed by name instead of a
                bare `KeyError` (mirrors `EarlyStoppingCallback`'s guard).
        """
        if not is_scheduled_log_epoch(
            context.epoch,
            context.total_epochs,
            first_epochs=self.log_first_epochs,
            every_epochs=self.log_every_epochs,
        ):
            return
        if self.metric_key not in context.metrics:
            raise ValueError(
                f"StructuredTrainingLogCallback watches metric "
                f"{self.metric_key!r}, but this run's strategy emits "
                f"{sorted(context.metrics)}. Configure metric_key to one of "
                "the emitted metrics."
            )
        cost = context.metrics[self.metric_key]
        parameters_block = render_parameter_block(self.parameter_summaries)
        # One `logging` record per row; the multi-line block rides inside it
        # (the timestamp prefix leads the header line, the parameter lines
        # continue below), so the live trace reads exactly like the replay.
        logger.info(
            "%s",
            format_structured_log_row(
                context.epoch,
                cost,
                parameters_block,
                unfolding_depth=self.unfolding_depth,
            ),
        )
        row: dict[str, Any] = {
            "epoch": context.epoch,
            "cost": cost,
            "learned_parameters": parameters_block,
        }
        if self.unfolding_depth is not None:
            row["unfolding_depth"] = self.unfolding_depth
        self._history.append(row)

    def on_train_end(self, context: RunContext) -> None:
        """Persist every logged row as one DataFrame artifact.

        Args:
            context: The current `RunContext`.
        """
        context.tracker.save_artifact(self.artifact_name, pd.DataFrame(self._history))


class ModelCheckpointCallback(Callback):
    """Saves the model's trained weights as an artifact once training
    completes.

    Only ``nn.Module`` models are checkpointed (their ``state_dict`` rides
    safetensors per the T3.i serialization policy). Non-module models —
    closed-form analytic controllers with no trained weights — are skipped
    with an INFO log: their solved parameters are persisted explicitly by
    `ParameterSnapshotCallback`/family callbacks instead, and the removed
    pickle fallback (which used to swallow whole controller objects here) is
    prohibited on the write path.
    """

    def __init__(self, artifact_name: str = "model") -> None:
        """
        Args:
            artifact_name: The artifact name to save the model under.
        """
        self.artifact_name = artifact_name

    def on_train_end(self, context: RunContext) -> None:
        """Save `context.model`'s weights via `context.tracker` when it is a
        torch module; log and skip otherwise.

        Args:
            context: The current `RunContext`.
        """
        if not isinstance(context.model, torch.nn.Module):
            logger.info(
                "Skipping model checkpoint %r: %s is not an nn.Module "
                "(no trained weights to persist; T3.i prohibits pickling "
                "whole controller objects).",
                self.artifact_name,
                type(context.model).__name__,
            )
            return
        path = context.tracker.save_artifact(self.artifact_name, context.model)
        logger.info(
            "Saved model checkpoint artifact %r to %s", self.artifact_name, path
        )


class TrainingStateCheckpointCallback(Callback):
    """Emits a resumable `TrainingState` snapshot once training completes
    (the T3.b resumability seam): epoch reached, trained module weights,
    and the optimizer's own state, persisted through the serializer
    registry like any other artifact.

    The module is the family's `TrainableController.as_module()` seam (T2.b)
    -- the same module the optimizer was built over -- so a later resume
    restores exactly the parameters that were being trained.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        artifact_name: str = "training_state",
    ) -> None:
        """
        Args:
            module: The trainable module whose ``state_dict`` is snapshotted.
            optimizer: The optimizer whose ``state_dict`` is snapshotted.
            artifact_name: The artifact name to save the snapshot under.
        """
        self.module = module
        self.optimizer = optimizer
        self.artifact_name = artifact_name

    def on_train_end(self, context: RunContext) -> None:
        """Snapshot ``(epoch, module state, optimizer state)`` and persist it.

        Args:
            context: The current `RunContext`.
        """
        state = TrainingState(
            epoch=context.epoch,
            module_state_dict=self.module.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )
        path = context.tracker.save_artifact(self.artifact_name, state)
        logger.info(
            "Saved training-state artifact %r (epoch %d) to %s",
            self.artifact_name,
            context.epoch,
            path,
        )


class MetricsHistoryCallback(Callback):
    """Accumulates every epoch's metrics, phase and wall-clock into a full
    per-epoch history, saved as a single DataFrame artifact once training
    completes.

    ExperimentTrackingCallback alone isn't enough to recover a training
    curve after the run ends: `ExperimentTracker.log_metrics` merges by key
    into one dict (`metadata.json`'s "metrics" field), so each call
    overwrites the previous epoch's values rather than accumulating a
    history -- only the last logged epoch survives. This callback captures
    every epoch instead, independent of `TrainingConfig.log_every`.

    Under the Tier-5 producer this frame becomes the model record's
    `training.parquet` (Annex 02 §2.2), which is what makes the columns worth
    naming: `epoch`, `phase`, `wall_time_s`, then the strategy's own metrics.
    """

    def __init__(self, artifact_name: str = METRICS_HISTORY) -> None:
        """
        Args:
            artifact_name: The artifact name to save the history DataFrame under.
        """
        self.artifact_name = artifact_name
        self._history: list[dict] = []
        self._started: float | None = None

    def on_epoch_start(self, context: RunContext) -> None:
        """Stamp this epoch's start.

        Args:
            context: The current `RunContext`.
        """
        self._started = wall_clock()

    def on_epoch_end(self, context: RunContext) -> None:
        """Append this epoch's row to the in-memory history.

        The elapsed time is measured from *this* callback's `on_epoch_start` to
        *this* callback's `on_epoch_end`, so it covers the strategy's step plus
        the callbacks that fire between the two -- what an epoch actually cost,
        rather than what its arithmetic cost.

        The stamp is cleared afterwards, and a row whose epoch never announced
        a start records ``None`` rather than a duration. A stamp left in place
        would make the *next* such epoch report the time since the previous one
        began: a number that is wrong and entirely plausible, which is the
        worse of the two failures.

        Args:
            context: The current `RunContext`.
        """
        elapsed = None if self._started is None else wall_clock() - self._started
        self._started = None
        self._history.append(
            {
                "epoch": context.epoch,
                "phase": context.phase,
                "wall_time_s": elapsed,
                **context.metrics,
            }
        )

    def on_train_end(self, context: RunContext) -> None:
        """Save the full per-epoch history as a `pandas.DataFrame` artifact.

        Args:
            context: The current `RunContext`.
        """
        context.tracker.save_artifact(self.artifact_name, pd.DataFrame(self._history))


class TrajectoryLoggingCallback(Callback):
    """Rolls out the trained model on one evaluation batch once training
    completes and saves the resulting state/observation/control trajectories
    as artifacts -- the raw, high-resolution data that scalar metrics alone
    can't reconstruct (e.g. for later trajectory or cumulative-cost-over-time
    plots).

    `rollout_fn`/`batch_sampler` are supplied by the caller (typically an
    Application's `build_engine`) rather than derived from `context.model`
    directly, since "how to roll out this model" and "how to sample an
    evaluation batch" are problem/model-specific concerns this Callback
    shouldn't need to know about.
    """

    def __init__(
        self,
        rollout_fn: Callable[[tuple], tuple],
        batch_sampler: Callable[[], tuple],
        artifact_prefix: str = "trajectory",
    ) -> None:
        """
        Args:
            rollout_fn: Maps a batch (as produced by `batch_sampler`) to a
                ``(states, observations, controls)`` tuple.
            batch_sampler: Called with no arguments to produce the evaluation
                batch to roll out.
            artifact_prefix: Prefix for the three saved artifact names
                (``f"{prefix}_states"``, etc.).
        """
        self.rollout_fn = rollout_fn
        self.batch_sampler = batch_sampler
        self.artifact_prefix = artifact_prefix

    def on_train_end(self, context: RunContext) -> None:
        """Sample a batch, roll it out via `rollout_fn`, and save the
        resulting states/observations/controls as three numpy artifacts.

        Args:
            context: The current `RunContext`.
        """
        batch = self.batch_sampler()
        states, observations, controls = self.rollout_fn(batch)
        context.tracker.save_artifact(
            f"{self.artifact_prefix}_states", to_numpy(states)
        )
        context.tracker.save_artifact(
            f"{self.artifact_prefix}_observations", to_numpy(observations)
        )
        context.tracker.save_artifact(
            f"{self.artifact_prefix}_controls", to_numpy(controls)
        )


class ParameterSnapshotCallback(Callback):
    """Saves named learnable-parameter values once training completes.

    Unlike `ModelCheckpointCallback`'s generic `state_dict` save, this
    captures parameters that live outside standard nn.Module submodule
    registration -- e.g. `UnfoldedController`'s, held in a plain
    `UnfoldingConfig` dataclass rather than as registered attributes -- and
    so wouldn't otherwise be recoverable from a checkpoint at all (its
    `state_dict()` is simply empty). Each entry of `parameters` is either:

    * the live tensor/array itself, mutated in place by an optimizer -- e.g.
      `COCPController.P_sqrt`, a registered `nn.Parameter` -- so reading it
      at `on_train_end` time naturally reflects the trained value without
      this callback needing to know when each update happens; or
    * a zero-argument callable returning the parameter's CURRENT value,
      evaluated fresh at `on_train_end` time -- required whenever the
      logged quantity is a REPARAMETERIZATION derived from a raw parameter
      (e.g. `StepSizeParameter.get`, ``sigmoid(rho) * alpha_max``): calling
      the transform once at construction time (before training even starts)
      would freeze the untrained value forever, since the returned tensor is
      a fresh computation, not a live reference to `rho` (the same class of
      staleness bug `StructuredTrainingLogCallback.parameter_summaries`
      already guards against).
    """

    def __init__(
        self,
        parameters: Mapping[str, Any | Callable[[], Any]],
        artifact_prefix: str = "parameter",
    ) -> None:
        """
        Args:
            parameters: Maps a name to either the *live* tensor/array to
                snapshot at `on_train_end` time (not a copy), or a
                zero-argument callable returning that value fresh at
                `on_train_end` time (see class docstring).
            artifact_prefix: Prefix for each saved artifact name
                (``f"{prefix}_{name}"``).
        """
        self.parameters = parameters
        self.artifact_prefix = artifact_prefix

    def on_train_end(self, context: RunContext) -> None:
        """Save each of `self.parameters`' current values as a numpy
        artifact, calling any callable entry to read its fresh value first.

        Args:
            context: The current `RunContext`.
        """
        for name, value in self.parameters.items():
            current = value() if callable(value) else value
            context.tracker.save_artifact(
                f"{self.artifact_prefix}_{name}", to_numpy(current)
            )


class ProfilingCallback(Callback):
    """Measures per-epoch latency (wall/CPU, and CUDA device time via the peak
    stats) and memory (peak process RAM, peak GPU VRAM), then persists an
    aggregated summary into ``metadata.json`` and the full per-epoch history as
    a ``profiling_history`` DataFrame artifact.

    Attached by default at the Runner scope (by an Application's ``build_engine``)
    so every run records its performance footprint using only cheap epoch-level
    measurements. Fine-grained block metrics from ``@profiled`` functions
    (opt-in via ``SC_PROFILE``) are drained from the process-global ``COLLECTOR``
    into the same ``profile/`` namespace when present.

    Per-epoch history is captured here (independent of ``TrainingConfig.log_every``)
    for the same reason as ``MetricsHistoryCallback``: ``log_metrics`` merges by
    key, so only a summary -- not a curve -- can live in ``metadata.json``.
    """

    def __init__(
        self,
        artifact_name: str = "profiling_history",
        metrics_prefix: str = "profile/",
    ) -> None:
        """
        Args:
            artifact_name: The artifact name to save the per-epoch profiling
                history DataFrame under.
            metrics_prefix: Namespace prefix for every logged metric key.
        """
        self.artifact_name = artifact_name
        self.metrics_prefix = metrics_prefix
        self._history: list[dict] = []
        self._train_wall_start = 0.0
        self._epoch_wall_start = 0.0
        self._epoch_cpu_start = 0.0

    def on_train_start(self, context: RunContext) -> None:
        """Reset the block-level `COLLECTOR` and start the whole-run timer.

        Args:
            context: The current `RunContext`.
        """
        COLLECTOR.reset()
        self._history = []
        self._train_wall_start = wall_clock()

    def on_epoch_start(self, context: RunContext) -> None:
        """Reset the CUDA peak-memory counter and start this epoch's timers.

        Args:
            context: The current `RunContext`.
        """
        reset_gpu_peak_memory()
        self._epoch_wall_start = wall_clock()
        self._epoch_cpu_start = cpu_clock()

    def on_epoch_end(self, context: RunContext) -> None:
        """Record this epoch's wall/CPU time and peak RAM/VRAM into the
        in-memory history.

        Args:
            context: The current `RunContext`.
        """
        record: dict[str, float] = {
            "epoch": context.epoch,
            "wall_time_s": wall_clock() - self._epoch_wall_start,
            "cpu_time_s": cpu_clock() - self._epoch_cpu_start,
        }
        peak_vram = peak_gpu_memory_bytes()
        if peak_vram is not None:
            record["peak_vram_mb"] = peak_vram / 1e6
        peak_ram = current_peak_ram_bytes()
        if peak_ram is not None:
            record["peak_ram_mb"] = peak_ram / 1e6
        self._history.append(record)

    def on_train_end(self, context: RunContext) -> None:
        """Log a summary (see `_summarize`) and save the full per-epoch
        history as a DataFrame artifact. A no-op if no epoch was recorded.

        For non-trainable runs (`context.is_trainable` ``False``, e.g. a
        single analytical rollout), the per-epoch aggregate statistics
        (``{prefix}epoch_*`` and ``{prefix}num_epochs_profiled``) are dropped
        from the logged summary -- with exactly one epoch, a mean/p50/p95
        over it is redundant noise, not a statistic. The aggregate scalars
        (whole-run wall time, peak RAM/VRAM) are always kept.

        Args:
            context: The current `RunContext`.
        """
        if not self._history:
            return
        summary = self._summarize()
        if not context.is_trainable:
            prefix = self.metrics_prefix
            summary = {
                key: value
                for key, value in summary.items()
                if not key.startswith(f"{prefix}epoch_")
                and key != f"{prefix}num_epochs_profiled"
            }
        context.tracker.log_metrics(summary)
        context.tracker.save_artifact(self.artifact_name, pd.DataFrame(self._history))

    def _summarize(self) -> dict[str, float]:
        """Aggregate the per-epoch history and drain `COLLECTOR`'s block-level
        results into one flat metrics dict.

        Returns:
            A dict of ``f"{self.metrics_prefix}..."``-prefixed scalar metrics:
            whole-run wall time, per-epoch wall-time mean/p50/p95, total CPU
            time, epoch count, peak RAM/VRAM (if sampled), and any `@profiled`
            block aggregates recorded via `COLLECTOR`.
        """
        prefix = self.metrics_prefix
        walls = np.array([r["wall_time_s"] for r in self._history], dtype=float)
        summary = {
            f"{prefix}train_wall_time_total_s": wall_clock() - self._train_wall_start,
            f"{prefix}epoch_wall_time_mean_s": float(walls.mean()),
            f"{prefix}epoch_wall_time_p50_s": float(np.percentile(walls, 50)),
            f"{prefix}epoch_wall_time_p95_s": float(np.percentile(walls, 95)),
            f"{prefix}epoch_cpu_time_total_s": float(
                sum(r["cpu_time_s"] for r in self._history)
            ),
            f"{prefix}num_epochs_profiled": float(len(self._history)),
        }
        ram = [r["peak_ram_mb"] for r in self._history if "peak_ram_mb" in r]
        if ram:
            summary[f"{prefix}peak_ram_mb"] = max(ram)
        vram = [r["peak_vram_mb"] for r in self._history if "peak_vram_mb" in r]
        if vram:
            summary[f"{prefix}peak_vram_mb"] = max(vram)
        # Drain opt-in (SC_PROFILE) block-level profiling into the same namespace.
        for aggregate in COLLECTOR.aggregate().values():
            summary.update(aggregate.as_metrics(prefix=prefix))
        return summary


class EarlyStoppingCallback(Callback):
    """Requests the Runner stop once the tracked metric changes by less than
    `tolerance` between consecutive epochs -- the epsilon-convergence
    criterion for e.g. the analytical gradient-descent solve (Phase 2A),
    layered on the same per-epoch metrics `MetricsHistoryCallback` already
    consumes rather than a bespoke recorder.
    """

    def __init__(self, tolerance: float, metric_key: str = "loss") -> None:
        """
        Args:
            tolerance: Stop once ``abs(current - previous) < tolerance`` for
                the tracked metric.
            metric_key: The `context.metrics` key to track (e.g. ``"loss"``).
        """
        self.tolerance = tolerance
        self.metric_key = metric_key
        self._previous: float | None = None

    def on_epoch_end(self, context: RunContext) -> None:
        """Set `context.stop_requested` if `metric_key` changed by less than
        `tolerance` since the previous epoch. Never fires on the first epoch
        (no previous value yet to compare against).

        Args:
            context: The current `RunContext`.

        Raises:
            ValueError: If `metric_key` is absent from `context.metrics` --
                a configured-key mismatch (e.g. watching ``"loss"`` while the
                strategy emits ``"cost"``), diagnosed by name instead of the
                bare `KeyError` it used to surface as (N8).
        """
        current = context.metrics.get(self.metric_key)
        if current is None:
            raise ValueError(
                f"EarlyStoppingCallback watches metric {self.metric_key!r}, "
                f"but this run's strategy emits {sorted(context.metrics)}. "
                "Configure metric_key to one of the emitted metrics."
            )
        if (
            self._previous is not None
            and abs(current - self._previous) < self.tolerance
        ):
            context.stop_requested = True
        self._previous = current


class ControlSequenceHistoryCallback(Callback):
    """Snapshots a controller's live decision variable ``U`` at the *start* of
    every epoch (i.e. ``U^(i)``, the iterate about to be refined this epoch),
    so each row aligns with the loss ``J^(i)`` `MetricsHistoryCallback` logs
    from that same iterate (`GradientDescentStrategy.step` computes the
    forward/loss before updating `U`). Saved as one stacked array artifact at
    train end -- native epoch-hook capture, no bespoke recorder.
    """

    def __init__(self, controller: Any, artifact_name: str = "control_history") -> None:
        """
        Args:
            controller: The controller whose live ``.U`` tensor is snapshotted
                (e.g. `models.open_loop.OpenLoopGDController`).
            artifact_name: The artifact name to save the stacked history under.
        """
        self.controller = controller
        self.artifact_name = artifact_name
        self._history: list[np.ndarray] = []

    def on_epoch_start(self, context: RunContext) -> None:
        """Append a detached copy of `self.controller.U` to the in-memory
        history.

        Args:
            context: The current `RunContext`.
        """
        self._history.append(self.controller.U.detach().cpu().numpy().copy())

    def on_train_end(self, context: RunContext) -> None:
        """Save the stacked per-epoch ``U`` history as one numpy artifact.

        Args:
            context: The current `RunContext`.
        """
        context.tracker.save_artifact(self.artifact_name, np.stack(self._history))


class ProblemSignatureCallback(Callback):
    """Logs the `Signable` tree describing this run's problem, controller, and
    sampling distributions, once, at train start -- the Phase 1D fix for the
    persistence layer completely missing the control-theory problem
    definition (system/cost matrices, noise stats).

    Composable with `ExperimentTrackingCallback`: this contributes
    ``signature``/``problem.*``/``controller.*``/``sampling.*`` keys;
    `ExperimentTrackingCallback` contributes the `TrainingConfig` fields --
    disjoint namespaces, no collision, both merge into the same persisted
    ``params`` dict via `ExperimentTracker.log_params`.
    """

    def __init__(
        self,
        problem: OptimalControlProblem,
        controller: Controller,
        distributions: Mapping[str, Signable] | None = None,
    ) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` this run solves.
            controller: The `Controller` this run evaluates/trains.
            distributions: Named `Signable` sampling distributions (e.g.
                ``{"initial_state": ..., "process_noise": ..., "measurement_noise": ...}``)
                -- the same objects actually used to draw the run's batches,
                never a parallel/shadow description of them.
        """
        self.problem = problem
        self.controller = controller
        self.distributions = distributions or {}

    def on_train_start(self, context: RunContext) -> None:
        """Compose the signature tree, compute its composite digest (via the
        shared `compute_run_signature` -- Phase 1F: the same helper a caller
        can use to compute a not-yet-run configuration's *expected* signature,
        e.g. to detect a stale cache, without duplicating this tree-assembly
        logic), flatten it into collision-proof dotted keys, and log both via
        `context.tracker`.

        Args:
            context: The current `RunContext`.
        """
        signature_tree = {
            "problem": self.problem.get_signature(),
            "controller": self.controller.get_signature(),
            "sampling": {
                name: dist.get_signature() for name, dist in self.distributions.items()
            },
        }
        context.tracker.log_params(
            {
                "signature": compute_run_signature(
                    self.problem, self.controller, self.distributions
                ),
                **flatten_signature(signature_tree),
            }
        )
