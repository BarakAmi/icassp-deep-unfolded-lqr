"""Universal training/inference Runner: manages the overarching loop and fires
Callback hooks, but delegates the actual mathematical step to an injected
TrainingStrategy per phase. Replaces the old, rigid EndToEndTrainer -- the loop
itself never assumes gradients, an optimizer, or even that the model is a
torch.nn.Module.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .callbacks import Callback
from .config import TrainingConfig
from .context import RunContext
from .strategy import BatchSampler, TrainingStrategy
from ..core.utils import ensure_positive_integer
from ..core.utils.functional import call_if_present, prefix_keys
from ..persistence.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingPhase:
    """One stretch of the run using a single strategy for `epochs` epochs, e.g.
    "warm-start for 20 epochs" followed by "end-to-end for 100 epochs" -- just
    two TrainingPhase entries with different strategies in the same `phases`
    sequence passed to Runner.

    Attributes:
        strategy: The `TrainingStrategy` (mathematical step) used for this phase.
        epochs: Number of epochs this phase runs for.
        name: Optional human-readable label (e.g. for logging).
    """

    strategy: TrainingStrategy
    epochs: int
    name: str = ""

    def __post_init__(self) -> None:
        """Fail-fast guard: `epochs` must be a positive integer.

        Raises:
            ValueError: If `epochs` is not a positive ``int``.
        """
        ensure_positive_integer(self.epochs, "epochs")


class Runner:
    """Implements the Engine protocol. Owns the loop and the callback firing;
    owns no math itself -- that lives entirely in each phase's TrainingStrategy.
    """

    def __init__(
        self,
        model: Any,
        tracker: ExperimentTracker,
        config: TrainingConfig,
        batch_sampler: BatchSampler,
        phases: Sequence[TrainingPhase],
        callbacks: Sequence[Callback] = (),
    ) -> None:
        """
        Args:
            model: The model being trained/evaluated; passed through to
                `RunContext.model` and toggled via ``.train()``/``.eval()`` if
                it supports them (see `_set_mode`).
            tracker: The `ExperimentTracker` this run logs to.
            config: This run's `TrainingConfig`.
            batch_sampler: Called with no arguments to produce each training batch.
            phases: The ordered `TrainingPhase`s composing this run; must be
                non-empty.
            callbacks: `Callback`s fired around each phase/epoch/batch.

        Raises:
            ValueError: If `phases` is empty.
        """
        if not phases:
            raise ValueError("Runner requires at least one TrainingPhase.")
        self.model = model
        self.tracker = tracker
        self.config = config
        self.batch_sampler = batch_sampler
        self.phases = list(phases)
        self.callbacks = list(callbacks)

    def _fire(self, hook_name: str, context: RunContext) -> None:
        """Invoke `hook_name` on every registered callback, in order.

        Args:
            hook_name: The `Callback` method name to invoke (e.g.
                ``"on_epoch_end"``).
            context: The `RunContext` passed to each callback.
        """
        for callback in self.callbacks:
            getattr(callback, hook_name)(context)

    def _set_mode(self, *, training: bool) -> None:
        """Toggle model.train()/model.eval() if the model supports it (e.g. an
        nn.Module); non-learnable models (plain analytical controllers) simply
        don't have these methods, and that's fine -- we don't require them.

        Args:
            training: If ``True``, call ``self.model.train()``; otherwise call
                ``self.model.eval()``.
        """
        call_if_present(self.model, "train" if training else "eval")

    def _build_context(self) -> RunContext:
        """Construct a fresh `RunContext` for this run, with `total_epochs`
        summed across all `phases` and `is_trainable` derived from whether
        any phase's strategy actually trains parameters.

        Returns:
            A new `RunContext` bound to this Runner's model/tracker/config.
        """
        total_epochs = sum(phase.epochs for phase in self.phases)
        # getattr(..., True): a strategy that predates this flag defaults to
        # "trainable" (today's over-inclusive behavior), so only strategies
        # that explicitly declare trains_parameters=False (e.g.
        # AnalyticalStrategy) opt out of logging optimizer metadata.
        is_trainable = any(
            getattr(phase.strategy, "trains_parameters", True) for phase in self.phases
        )
        return RunContext(
            model=self.model,
            tracker=self.tracker,
            config=self.config,
            total_epochs=total_epochs,
            is_trainable=is_trainable,
        )

    def train(self) -> Mapping[str, float]:
        """Run every `TrainingPhase` in sequence, firing `Callback` hooks
        around each phase/epoch/batch and delegating the mathematical step to
        each phase's `TrainingStrategy.step`.

        A callback may set ``context.stop_requested`` (e.g.
        `EarlyStoppingCallback`) during ``on_epoch_end`` to end the run early;
        the check happens right after that hook fires, so `on_train_end`
        still fires exactly once whether the run completes every phase's
        epochs or stops early.

        Returns:
            The final step's metrics, with keys prefixed ``"final_"``
            (e.g. ``{"final_loss": ...}``).
        """
        context = self._build_context()
        # INFO lifecycle contract (T1.5.e): run start/end and phase
        # transitions; per-epoch metrics stay at DEBUG.
        logger.info(
            "Training run started: %d phase(s), %d total epoch(s), trainable=%s",
            len(self.phases),
            context.total_epochs,
            context.is_trainable,
        )
        self._set_mode(training=True)
        self._fire("on_train_start", context)

        global_epoch = 0
        for phase in self.phases:
            # Before the phase's first hook fires, so a callback that records
            # per-epoch rows attributes every one of them to the phase that
            # produced it (the layer-wise families train one phase per
            # unfolding iteration).
            context.phase = phase.name
            logger.info(
                "Phase %r started: strategy=%s, epochs=%d",
                phase.name or "<unnamed>",
                type(phase.strategy).__name__,
                phase.epochs,
            )
            for _ in range(phase.epochs):
                context.epoch = global_epoch
                self._fire("on_epoch_start", context)

                batch = self.batch_sampler()
                context.batch = 0
                self._fire("on_batch_start", context)

                context.metrics = dict(phase.strategy.step(context, batch))

                logger.debug(
                    "Epoch %d/%d metrics: %s",
                    global_epoch + 1,
                    context.total_epochs,
                    context.metrics,
                )
                self._fire("on_batch_end", context)
                self._fire("on_epoch_end", context)
                global_epoch += 1
                if context.stop_requested:
                    logger.info("Early stop requested at epoch %d.", global_epoch)
                    break
            if context.stop_requested:
                break

        self._fire("on_train_end", context)
        logger.info("Training run finished: final metrics %s", context.metrics)
        return prefix_keys("final_", context.metrics)

    def evaluate(self) -> Mapping[str, float]:
        """Evaluate the model with the *last* phase's strategy, without
        updating any weights, and log the result via `self.tracker`.

        Returns:
            The evaluation metrics, with keys prefixed ``"eval_"``
            (e.g. ``{"eval_loss": ...}``).
        """
        self._set_mode(training=False)
        context = self._build_context()
        batch = self.batch_sampler()
        metrics = self.phases[-1].strategy.evaluate_step(context, batch)
        result = prefix_keys("eval_", metrics)
        self.tracker.log_metrics(result)
        logger.info("Evaluation finished: %s", dict(result))
        return result

    def run(self) -> Mapping[str, float]:
        """Convenience entrypoint: `train` then `evaluate`.

        Returns:
            The union of `train`'s and `evaluate`'s metrics.
        """
        train_metrics = self.train()
        eval_metrics = self.evaluate()
        return {**train_metrics, **eval_metrics}
