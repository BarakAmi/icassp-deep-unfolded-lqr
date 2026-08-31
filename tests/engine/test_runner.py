import json
from unittest.mock import MagicMock

import pytest
import torch

from mbl.engine.callbacks import (
    Callback,
    ExperimentTrackingCallback,
    ModelCheckpointCallback,
)
from mbl.engine.config import TrainingConfig
from mbl.engine.engine import Engine
from mbl.engine.runner import Runner, TrainingPhase
from mbl.engine.strategy import GradientDescentStrategy
from mbl.persistence.local_tracker import LocalExperimentTracker


class _RecordingCallback(Callback):
    """Records every hook invocation (name, epoch, metrics-snapshot) in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict]] = []

    def _record(self, hook: str, context) -> None:
        self.calls.append((hook, context.epoch, dict(context.metrics)))

    def on_train_start(self, context) -> None:
        self._record("on_train_start", context)

    def on_epoch_start(self, context) -> None:
        self._record("on_epoch_start", context)

    def on_batch_start(self, context) -> None:
        self._record("on_batch_start", context)

    def on_batch_end(self, context) -> None:
        self._record("on_batch_end", context)

    def on_epoch_end(self, context) -> None:
        self._record("on_epoch_end", context)

    def on_train_end(self, context) -> None:
        self._record("on_train_end", context)


class _FixedMetricStrategy:
    """A stub TrainingStrategy that always returns the same metrics dict and
    counts how many times step()/evaluate_step() were called.

    Deliberately has no `trains_parameters` attribute, exercising the
    getattr(..., True) default (a strategy predating that flag keeps today's
    "log everything" behavior).
    """

    def __init__(self, metrics: dict) -> None:
        self.metrics = metrics
        self.step_calls = 0
        self.evaluate_calls = 0

    def step(self, context, batch):
        self.step_calls += 1
        return dict(self.metrics)

    def evaluate_step(self, context, batch):
        self.evaluate_calls += 1
        return dict(self.metrics)


class _NonTrainableStrategy(_FixedMetricStrategy):
    """Explicitly declares trains_parameters = False, like AnalyticalStrategy."""

    trains_parameters = False


def _make_runner(phases, callbacks=(), model=None, tracker=None):
    config = TrainingConfig(batch_size=2, learning_rate=0.01)
    return Runner(
        model=model if model is not None else MagicMock(),
        tracker=tracker if tracker is not None else MagicMock(),
        config=config,
        batch_sampler=MagicMock(return_value=(1, 2, 3)),
        phases=phases,
        callbacks=callbacks,
    )


def test_runner_satisfies_engine_protocol() -> None:
    runner = _make_runner(
        [TrainingPhase(_FixedMetricStrategy({"loss": 1.0}), epochs=1)]
    )
    assert isinstance(runner, Engine)


def test_runner_requires_at_least_one_phase() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _make_runner([])


def test_runner_calls_strategy_step_once_per_epoch() -> None:
    strategy = _FixedMetricStrategy({"loss": 1.0})
    runner = _make_runner([TrainingPhase(strategy, epochs=4)])
    runner.train()
    assert strategy.step_calls == 4


def test_runner_never_calls_tracker_directly_during_train_without_callbacks() -> None:
    """Logging/checkpointing must come entirely from callbacks -- with none
    attached, the Runner itself must not touch the tracker at all during train()."""
    tracker = MagicMock()
    strategy = _FixedMetricStrategy({"loss": 1.0})
    runner = _make_runner([TrainingPhase(strategy, epochs=3)], tracker=tracker)
    runner.train()
    tracker.log_params.assert_not_called()
    tracker.log_metrics.assert_not_called()
    tracker.save_artifact.assert_not_called()


def test_runner_fires_hooks_in_order_once_per_epoch() -> None:
    strategy = _FixedMetricStrategy({"loss": 5.0})
    recorder = _RecordingCallback()
    runner = _make_runner([TrainingPhase(strategy, epochs=2)], callbacks=[recorder])
    runner.train()

    hook_sequence = [call[0] for call in recorder.calls]
    assert hook_sequence == [
        "on_train_start",
        "on_epoch_start",
        "on_batch_start",
        "on_batch_end",
        "on_epoch_end",
        "on_epoch_start",
        "on_batch_start",
        "on_batch_end",
        "on_epoch_end",
        "on_train_end",
    ]


def test_runner_passes_strategy_metrics_into_context() -> None:
    strategy = _FixedMetricStrategy({"loss": 3.5})
    recorder = _RecordingCallback()
    runner = _make_runner([TrainingPhase(strategy, epochs=1)], callbacks=[recorder])
    runner.train()

    epoch_end_metrics = next(
        m for hook, _, m in recorder.calls if hook == "on_epoch_end"
    )
    assert epoch_end_metrics == {"loss": 3.5}


def test_runner_train_returns_final_prefixed_metrics() -> None:
    strategy = _FixedMetricStrategy({"loss": 2.0, "extra": 9.0})
    runner = _make_runner([TrainingPhase(strategy, epochs=2)])
    result = runner.train()
    assert result == {"final_loss": 2.0, "final_extra": 9.0}


def test_runner_multi_phase_epoch_numbering_is_global_and_sequential() -> None:
    phase_a = _FixedMetricStrategy({"loss": 1.0})
    phase_b = _FixedMetricStrategy({"loss": 2.0})
    recorder = _RecordingCallback()
    runner = _make_runner(
        [
            TrainingPhase(phase_a, epochs=2, name="warm_start"),
            TrainingPhase(phase_b, epochs=3, name="end_to_end"),
        ],
        callbacks=[recorder],
    )
    runner.train()

    assert phase_a.step_calls == 2
    assert phase_b.step_calls == 3

    epochs_seen = [
        epoch for hook, epoch, _ in recorder.calls if hook == "on_epoch_start"
    ]
    assert epochs_seen == [0, 1, 2, 3, 4]  # global, continuous across phases


def test_runner_evaluate_uses_last_phases_strategy_and_logs_via_tracker() -> None:
    phase_a = _FixedMetricStrategy({"loss": 1.0})
    phase_b = _FixedMetricStrategy({"loss": 0.5})
    tracker = MagicMock()
    runner = _make_runner(
        [TrainingPhase(phase_a, epochs=1), TrainingPhase(phase_b, epochs=1)],
        tracker=tracker,
    )
    result = runner.evaluate()

    assert phase_a.evaluate_calls == 0
    assert phase_b.evaluate_calls == 1
    assert result == {"eval_loss": 0.5}
    tracker.log_metrics.assert_called_once_with({"eval_loss": 0.5})


def test_runner_run_trains_then_evaluates_and_merges() -> None:
    strategy = _FixedMetricStrategy({"loss": 1.0})
    runner = _make_runner([TrainingPhase(strategy, epochs=2)])
    result = runner.run()
    assert result == {"final_loss": 1.0, "eval_loss": 1.0}
    assert strategy.step_calls == 2
    assert strategy.evaluate_calls == 1


def test_runner_toggles_train_and_eval_mode_on_model_that_supports_it() -> None:
    model = MagicMock()
    strategy = _FixedMetricStrategy({"loss": 1.0})
    runner = _make_runner([TrainingPhase(strategy, epochs=1)], model=model)

    runner.train()
    model.train.assert_called_once()

    runner.evaluate()
    model.eval.assert_called_once()


def test_runner_supports_models_without_train_eval_methods() -> None:
    """A non-learnable model (e.g. a plain analytical controller) has no
    .train()/.eval(); the Runner must not require them."""

    class _PlainModel:
        pass

    model = _PlainModel()
    strategy = _FixedMetricStrategy({"cost": 1.0})
    runner = _make_runner([TrainingPhase(strategy, epochs=1)], model=model)

    runner.train()  # must not raise
    runner.evaluate()  # must not raise


def test_training_phase_rejects_non_positive_epochs() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingPhase(_FixedMetricStrategy({"loss": 1.0}), epochs=0)


# --- Real integration: genuine model + optimizer + tracker, not mocks ---


class _TinyQuadraticModel(torch.nn.Module):
    """Mimics UnfoldedController.forward's (X, Y, U, cost) signature with a
    trivial, analytically-checkable optimization problem: minimize x^2."""

    def __init__(self) -> None:
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([5.0]))

    def forward(self, initial_state, process_noise=None, measurement_noise=None):
        cost = self.x**2
        return initial_state, initial_state, initial_state, cost


def _fixed_batch_sampler():
    state = torch.zeros(1, 1)
    return state, state, state


def test_runner_single_phase_integration_reduces_loss_and_persists_real_artifacts(
    tmp_path,
) -> None:
    model = _TinyQuadraticModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = TrainingConfig(batch_size=1, learning_rate=0.1, log_every=10)
    tracker = LocalExperimentTracker(tmp_path, "runner_integration")

    strategy = GradientDescentStrategy(model, optimizer)
    runner = Runner(
        model=model,
        tracker=tracker,
        config=config,
        batch_sampler=_fixed_batch_sampler,
        phases=[TrainingPhase(strategy, epochs=50, name="end_to_end")],
        callbacks=[ExperimentTrackingCallback(), ModelCheckpointCallback()],
    )

    initial_loss = float(model.x.detach() ** 2)
    result = runner.train()

    assert result["final_loss"] < initial_loss / 1000
    assert abs(float(model.x.detach())) < 0.01

    metadata = json.loads((tracker.run_dir / "metadata.json").read_text())
    assert metadata["params"]["batch_size"] == 1
    assert "loss" in metadata["metrics"]

    model_path = tracker.artifacts_dir / "model.safetensors"
    assert model_path.exists()
    from safetensors.torch import load_file

    state_dict = load_file(model_path)
    assert state_dict.keys() == model.state_dict().keys()


def test_runner_composes_two_phases_with_different_optimizers(tmp_path) -> None:
    """Proves the architecture natively supports composing training recipes:
    a "warm-start" phase (high LR, few epochs) seamlessly followed by an
    "end-to-end" phase (lower LR, more epochs) on the *same* model."""
    model = _TinyQuadraticModel()
    warm_start_optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    end_to_end_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    config = TrainingConfig(batch_size=1, learning_rate=0.1, log_every=5)
    tracker = LocalExperimentTracker(tmp_path, "multi_phase_integration")

    warm_start_phase = TrainingPhase(
        GradientDescentStrategy(model, warm_start_optimizer),
        epochs=5,
        name="warm_start",
    )
    end_to_end_phase = TrainingPhase(
        GradientDescentStrategy(model, end_to_end_optimizer),
        epochs=20,
        name="end_to_end",
    )

    runner = Runner(
        model=model,
        tracker=tracker,
        config=config,
        batch_sampler=_fixed_batch_sampler,
        phases=[warm_start_phase, end_to_end_phase],
        callbacks=[ExperimentTrackingCallback(), ModelCheckpointCallback()],
    )

    initial_loss = float(model.x.detach() ** 2)
    result = runner.train()

    assert result["final_loss"] < initial_loss
    assert (tracker.artifacts_dir / "model.safetensors").exists()


def test_runner_context_is_trainable_defaults_true_for_untagged_strategy() -> None:
    """A TrainingStrategy with no trains_parameters attribute keeps today's
    over-inclusive default (log everything), not a silent opt-out."""
    recorder = _RecordingCallback()
    runner = _make_runner(
        [TrainingPhase(_FixedMetricStrategy({"loss": 1.0}), epochs=1)],
        callbacks=[recorder],
    )
    context = runner._build_context()
    assert context.is_trainable is True


def test_runner_context_is_trainable_false_when_every_phase_is_non_trainable() -> None:
    runner = _make_runner(
        [TrainingPhase(_NonTrainableStrategy({"cost": 1.0}), epochs=1)]
    )
    assert runner._build_context().is_trainable is False


def test_runner_context_is_trainable_true_if_any_phase_trains() -> None:
    """A mixed run (one non-trainable phase, one trainable phase) must still
    report is_trainable=True -- it describes 'was anything trained', not
    'is the last phase trainable'."""
    runner = _make_runner(
        [
            TrainingPhase(_NonTrainableStrategy({"cost": 1.0}), epochs=1),
            TrainingPhase(_FixedMetricStrategy({"loss": 1.0}), epochs=1),
        ]
    )
    assert runner._build_context().is_trainable is True


class _StopAtEpochCallback(Callback):
    """Sets context.stop_requested once context.epoch reaches stop_epoch --
    a stand-in for EarlyStoppingCallback exercising the Runner's break logic
    in isolation."""

    def __init__(self, stop_epoch: int) -> None:
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, context) -> None:
        if context.epoch == self.stop_epoch:
            context.stop_requested = True


def test_runner_stops_early_when_a_callback_requests_it() -> None:
    """A fake callback requesting a stop at epoch 2 of a 10-epoch phase must
    end the run after that epoch -- 3 epochs total (0, 1, 2), not 10."""
    strategy = _FixedMetricStrategy({"loss": 1.0})
    recorder = _RecordingCallback()
    runner = _make_runner(
        [TrainingPhase(strategy, epochs=10)],
        callbacks=[_StopAtEpochCallback(stop_epoch=2), recorder],
    )
    runner.train()

    assert strategy.step_calls == 3
    epochs_seen = [
        epoch for hook, epoch, _ in recorder.calls if hook == "on_epoch_start"
    ]
    assert epochs_seen == [0, 1, 2]


def test_runner_fires_on_train_end_exactly_once_on_early_stop() -> None:
    strategy = _FixedMetricStrategy({"loss": 1.0})
    recorder = _RecordingCallback()
    runner = _make_runner(
        [TrainingPhase(strategy, epochs=10)],
        callbacks=[_StopAtEpochCallback(stop_epoch=2), recorder],
    )
    runner.train()

    train_end_calls = [hook for hook, _, _ in recorder.calls if hook == "on_train_end"]
    assert len(train_end_calls) == 1


def test_runner_stops_across_a_phase_boundary() -> None:
    """A stop requested during the first of two phases must prevent the
    second phase from running at all (the break propagates past the phase
    loop, not just the inner epoch loop)."""
    phase_a = _FixedMetricStrategy({"loss": 1.0})
    phase_b = _FixedMetricStrategy({"loss": 2.0})
    runner = _make_runner(
        [
            TrainingPhase(phase_a, epochs=5, name="a"),
            TrainingPhase(phase_b, epochs=5, name="b"),
        ],
        callbacks=[_StopAtEpochCallback(stop_epoch=1)],
    )
    runner.train()

    assert phase_a.step_calls == 2  # epochs 0, 1
    assert phase_b.step_calls == 0


def test_runner_stop_requested_defaults_to_false_and_completes_all_epochs() -> None:
    """No stop-requesting callback attached -> every existing run is
    byte-for-byte unaffected: all epochs run, and the context's
    stop_requested stays False throughout."""
    strategy = _FixedMetricStrategy({"loss": 1.0})
    context_holder = {}

    class _CaptureContextCallback(Callback):
        def on_train_end(self, context) -> None:
            context_holder["context"] = context

    runner = _make_runner(
        [TrainingPhase(strategy, epochs=5)], callbacks=[_CaptureContextCallback()]
    )
    runner.train()

    assert strategy.step_calls == 5
    assert context_holder["context"].stop_requested is False
