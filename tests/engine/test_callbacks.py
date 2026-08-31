import logging
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.callbacks import (
    Callback,
    ComputeContextAnnouncementCallback,
    ControlSequenceHistoryCallback,
    EarlyStoppingCallback,
    ExperimentTrackingCallback,
    MetricsHistoryCallback,
    ModelCheckpointCallback,
    ParameterSnapshotCallback,
    ProblemSignatureCallback,
    ProfilingCallback,
    StructuredTrainingLogCallback,
    TrajectoryLoggingCallback,
    format_structured_log_row,
)
from mbl.engine.config import TrainingConfig
from mbl.engine.context import RunContext


def _make_context(
    *, epoch=0, total_epochs=5, metrics=None, is_trainable=True
) -> tuple[RunContext, MagicMock]:
    tracker = MagicMock()
    config = TrainingConfig(batch_size=4, learning_rate=0.01, log_every=2)
    model = MagicMock()
    context = RunContext(
        model=model,
        tracker=tracker,
        config=config,
        total_epochs=total_epochs,
        epoch=epoch,
        metrics=metrics or {},
        is_trainable=is_trainable,
    )
    return context, tracker


def test_base_callback_hooks_are_all_no_ops() -> None:
    context, tracker = _make_context()
    callback = Callback()
    for hook in (
        "on_train_start",
        "on_epoch_start",
        "on_batch_start",
        "on_batch_end",
        "on_epoch_end",
        "on_train_end",
    ):
        getattr(callback, hook)(context)
    tracker.assert_not_called()


def test_experiment_tracking_callback_logs_params_on_train_start() -> None:
    context, tracker = _make_context()
    ExperimentTrackingCallback().on_train_start(context)
    tracker.log_params.assert_called_once_with(context.config.get_config())


def test_experiment_tracking_callback_keeps_all_fields_when_trainable() -> None:
    """Regression: no change for trainable-model runs (e.g. neural/unfolded)
    -- every existing field stays logged, unfiltered."""
    context, tracker = _make_context(is_trainable=True)
    ExperimentTrackingCallback().on_train_start(context)

    logged = tracker.log_params.call_args[0][0]
    assert logged["learning_rate"] == 0.01
    assert logged["optimizer_name"] == "adam"
    assert "optimizer_kwargs" in logged
    assert logged["batch_size"] == 4
    assert logged["log_every"] == 2


def test_experiment_tracking_callback_drops_optimizer_fields_when_not_trainable() -> (
    None
):
    """The exact flagged bug (fake learning_rate=1.0/optimizer_name=adam for a
    RiccatiController run) fixed at the source: these keys must not even be
    passed to log_params when nothing is being optimized."""
    context, tracker = _make_context(is_trainable=False)
    ExperimentTrackingCallback().on_train_start(context)

    logged = tracker.log_params.call_args[0][0]
    assert "learning_rate" not in logged
    assert "optimizer_name" not in logged
    assert "optimizer_kwargs" not in logged


def test_experiment_tracking_callback_keeps_non_optimizer_fields_when_not_trainable() -> (
    None
):
    """batch_size describes the rollout itself, not the optimizer -- it must
    survive the filter regardless of trainability."""
    context, tracker = _make_context(is_trainable=False)
    ExperimentTrackingCallback().on_train_start(context)

    logged = tracker.log_params.call_args[0][0]
    assert logged["batch_size"] == 4


def test_experiment_tracking_callback_drops_log_every_and_seed_when_not_trainable() -> (
    None
):
    """log_every/seed are redundant for a non-trainable (e.g. analytical)
    run: the ground-truth seeds are already logged per-distribution under
    sampling.*.seed, and log_every only governs a training loop's cadence."""
    context, tracker = _make_context(is_trainable=False)
    ExperimentTrackingCallback().on_train_start(context)

    logged = tracker.log_params.call_args[0][0]
    assert "log_every" not in logged
    assert "seed" not in logged


def test_experiment_tracking_callback_logs_on_cadence_epochs() -> None:
    callback = ExperimentTrackingCallback()
    context, tracker = _make_context(epoch=0, total_epochs=5, metrics={"loss": 1.0})
    callback.on_epoch_end(context)  # epoch 0 % log_every(2) == 0 -> logs
    tracker.log_metrics.assert_called_once_with({"epoch": 0, "loss": 1.0})


def test_experiment_tracking_callback_omits_epoch_key_when_not_trainable() -> None:
    """An analytical (N=1 rollout) run has no epoch-level training curve, so
    logging an epoch index is redundant noise."""
    callback = ExperimentTrackingCallback()
    context, tracker = _make_context(
        epoch=0, total_epochs=1, metrics={"cost": 1.0}, is_trainable=False
    )
    callback.on_epoch_end(context)
    tracker.log_metrics.assert_called_once_with({"cost": 1.0})


def test_experiment_tracking_callback_skips_off_cadence_epochs() -> None:
    callback = ExperimentTrackingCallback()
    context, tracker = _make_context(epoch=1, total_epochs=5, metrics={"loss": 1.0})
    callback.on_epoch_end(context)  # epoch 1 % log_every(2) != 0, and not final
    tracker.log_metrics.assert_not_called()


def test_experiment_tracking_callback_always_logs_final_epoch() -> None:
    callback = ExperimentTrackingCallback()
    context, tracker = _make_context(epoch=4, total_epochs=5, metrics={"loss": 0.1})
    callback.on_epoch_end(context)  # epoch 4 is the final epoch (total_epochs - 1)
    tracker.log_metrics.assert_called_once_with({"epoch": 4, "loss": 0.1})


def test_compute_context_announcement_callback_logs_device_backend_precision(
    caplog,
) -> None:
    ctx = ComputeContext(
        backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT32
    )
    context, _ = _make_context()
    with caplog.at_level(logging.INFO):
        ComputeContextAnnouncementCallback(ctx).on_train_start(context)

    assert "[ComputeContext]" in caplog.text
    assert "DEVICE: cpu" in caplog.text
    assert "backend=torch" in caplog.text
    assert "precision=float32" in caplog.text


def test_compute_context_announcement_callback_only_fires_on_train_start() -> None:
    ctx = ComputeContext()
    context, tracker = _make_context()
    callback = ComputeContextAnnouncementCallback(ctx)
    for hook in ("on_epoch_start", "on_batch_start", "on_batch_end", "on_epoch_end"):
        getattr(callback, hook)(context)
    tracker.assert_not_called()


def test_structured_training_log_callback_logs_hyperparameters_on_train_start(
    caplog,
) -> None:
    context, _ = _make_context()
    with caplog.at_level(logging.INFO):
        StructuredTrainingLogCallback({}).on_train_start(context)
    assert "[Hyperparameters]" in caplog.text
    assert "batch_size" in caplog.text


def test_structured_training_log_callback_logs_within_the_warmup_block(caplog) -> None:
    """Directive 2's dynamic schedule: the first ``a`` epochs are narrated
    consecutively -- epoch 3 (< default a=10) logs even though it is off the
    coarse ``b`` cadence."""
    callback = StructuredTrainingLogCallback({"alpha": lambda: "[0.1, 0.2]"})
    context, _ = _make_context(epoch=3, total_epochs=50, metrics={"loss": 1.23456})
    with caplog.at_level(logging.INFO):
        callback.on_epoch_end(context)

    assert "Cost (Loss): 1.234560" in caplog.text
    # One learned parameter per line, no legacy "Learned Parameters:" prefix.
    assert "alpha = [0.1, 0.2]" in caplog.text
    assert "Learned Parameters" not in caplog.text


def test_structured_training_log_callback_skips_between_warmup_and_cadence(
    caplog,
) -> None:
    """With no warmup (a=0) and cadence b=20, an off-cadence non-final epoch
    is skipped."""
    callback = StructuredTrainingLogCallback(
        {}, log_first_epochs=0, log_every_epochs=20
    )
    context, _ = _make_context(epoch=5, total_epochs=50, metrics={"loss": 1.0})
    with caplog.at_level(logging.INFO):
        callback.on_epoch_end(context)
    assert caplog.text == ""


def test_structured_training_log_callback_logs_on_the_b_cadence(caplog) -> None:
    callback = StructuredTrainingLogCallback(
        {}, log_first_epochs=0, log_every_epochs=20
    )
    context, _ = _make_context(epoch=20, total_epochs=50, metrics={"loss": 0.5})
    with caplog.at_level(logging.INFO):
        callback.on_epoch_end(context)  # 20 % 20 == 0 -> logs
    assert "Cost (Loss): 0.500000" in caplog.text


def test_structured_training_log_callback_always_logs_final_epoch(caplog) -> None:
    """The final epoch is always narrated, even off both the warmup and the
    ``b`` cadence."""
    callback = StructuredTrainingLogCallback(
        {}, log_first_epochs=0, log_every_epochs=100
    )
    context, _ = _make_context(epoch=49, total_epochs=50, metrics={"loss": 0.1})
    with caplog.at_level(logging.INFO):
        callback.on_epoch_end(context)
    assert "Cost (Loss): 0.100000" in caplog.text


def test_structured_training_log_callback_renders_each_parameter_on_its_own_line() -> (
    None
):
    """Directive 2's formatting: each learned parameter on a distinct line;
    a multi-line value (e.g. a full matrix) is indented under its name."""
    callback = StructuredTrainingLogCallback(
        {"alpha": lambda: "0.1", "P": lambda: "[[1, 0],\n [0, 1]]"}
    )
    context, _ = _make_context(epoch=0, total_epochs=1, metrics={"loss": 1.0})
    callback.on_epoch_end(context)

    block = callback._history[-1]["learned_parameters"]
    assert "    alpha = 0.1" in block
    assert "    P =" in block
    assert "        [[1, 0]," in block  # matrix body indented under its name


def test_structured_training_log_callback_rejects_invalid_schedule() -> None:
    with pytest.raises(ValueError, match="log_first_epochs"):
        StructuredTrainingLogCallback({}, log_first_epochs=-1)
    with pytest.raises(ValueError, match="log_every_epochs"):
        StructuredTrainingLogCallback({}, log_every_epochs=0)


def test_structured_training_log_callback_evaluates_parameter_summaries_fresh_every_call() -> (
    None
):
    """The summary callables must be re-invoked at every logged epoch, never
    memoized from a value captured once at construction -- otherwise the
    narrated parameter block would silently go stale as training proceeds
    (the exact failure mode this callback exists to avoid)."""
    live_value = {"alpha": 0.0}
    callback = StructuredTrainingLogCallback(
        {"alpha": lambda: str(live_value["alpha"])}
    )

    context, _ = _make_context(epoch=0, total_epochs=1, metrics={"loss": 1.0})
    live_value["alpha"] = 0.5
    callback.on_epoch_end(context)

    assert callback._history[-1]["learned_parameters"] == "    alpha = 0.5"


def test_structured_training_log_callback_raises_on_unknown_metric_key() -> None:
    callback = StructuredTrainingLogCallback({}, metric_key="cost")
    context, _ = _make_context(epoch=0, total_epochs=1, metrics={"loss": 1.0})
    with pytest.raises(ValueError, match="cost"):
        callback.on_epoch_end(context)


def test_structured_training_log_callback_saves_history_artifact_on_train_end() -> None:
    callback = StructuredTrainingLogCallback({"alpha": lambda: "0.1"})
    context, tracker = _make_context(epoch=0, total_epochs=1, metrics={"loss": 2.5})
    callback.on_epoch_end(context)
    callback.on_train_end(context)

    tracker.save_artifact.assert_called_once()
    name, history = tracker.save_artifact.call_args[0]
    assert name == "training_log_history"
    assert isinstance(history, pd.DataFrame)
    assert list(history["epoch"]) == [0]
    assert list(history["cost"]) == [2.5]
    assert list(history["learned_parameters"]) == ["    alpha = 0.1"]


def test_structured_training_log_callback_uses_custom_artifact_name() -> None:
    callback = StructuredTrainingLogCallback({}, artifact_name="nb03_training_log")
    context, tracker = _make_context(epoch=0, total_epochs=1, metrics={"loss": 1.0})
    callback.on_epoch_end(context)
    callback.on_train_end(context)

    name, _ = tracker.save_artifact.call_args[0]
    assert name == "nb03_training_log"


def test_format_structured_log_row_displays_epochs_one_indexed() -> None:
    """The persisted epoch stays 0-indexed, but a human-facing row reads from
    1: the first epoch is ``Epoch: 1``."""
    row = format_structured_log_row(0, 1.5, "")

    assert row.startswith("Epoch:    1 ")
    assert "Cost (Loss): 1.500000" in row


def test_format_structured_log_row_shows_unfolding_depth_when_given() -> None:
    row = format_structured_log_row(2, 0.25, "", unfolding_depth=4)

    assert "Epoch:    3" in row  # 0-indexed 2 -> displayed 3
    assert "Unfolding Depth: 4" in row
    assert "Cost (Loss): 0.250000" in row
    # Order: Unfolding Depth | Epoch | Cost.
    assert row.index("Unfolding Depth") < row.index("Epoch") < row.index("Cost (Loss)")


def test_format_structured_log_row_omits_depth_when_absent() -> None:
    assert "Unfolding Depth" not in format_structured_log_row(0, 1.0, "")


def test_structured_training_log_callback_narrates_and_stores_unfolding_depth(
    caplog,
) -> None:
    callback = StructuredTrainingLogCallback({}, unfolding_depth=6)
    context, _ = _make_context(epoch=0, total_epochs=1, metrics={"loss": 1.0})
    with caplog.at_level(logging.INFO):
        callback.on_epoch_end(context)

    assert "Unfolding Depth: 6" in caplog.text
    assert callback._history[-1]["unfolding_depth"] == 6
    assert callback._history[-1]["epoch"] == 0  # persisted value stays 0-indexed


def test_structured_training_log_callback_omits_depth_column_when_unset() -> None:
    callback = StructuredTrainingLogCallback({})
    context, _ = _make_context(epoch=0, total_epochs=1, metrics={"loss": 1.0})
    callback.on_epoch_end(context)

    assert "unfolding_depth" not in callback._history[-1]


def test_model_checkpoint_callback_saves_module_models_on_train_end() -> None:
    context, tracker = _make_context()
    module = torch.nn.Linear(2, 1)
    context = replace(context, model=module)
    ModelCheckpointCallback().on_train_end(context)
    tracker.save_artifact.assert_called_once_with("model", module)


def test_model_checkpoint_callback_uses_custom_artifact_name() -> None:
    context, tracker = _make_context()
    module = torch.nn.Linear(2, 1)
    context = replace(context, model=module)
    ModelCheckpointCallback(artifact_name="final_weights").on_train_end(context)
    tracker.save_artifact.assert_called_once_with("final_weights", module)


def test_model_checkpoint_callback_skips_non_module_models() -> None:
    """The T3.i write-path law: closed-form controllers (not nn.Modules,
    no trained weights) are never pickled whole -- the checkpoint is skipped,
    their solved parameters ride ParameterSnapshotCallback instead."""
    context, tracker = _make_context()  # context.model is a bare MagicMock
    ModelCheckpointCallback().on_train_end(context)
    tracker.save_artifact.assert_not_called()


def test_model_checkpoint_callback_only_fires_on_train_end() -> None:
    context, tracker = _make_context()
    callback = ModelCheckpointCallback()
    for hook in (
        "on_train_start",
        "on_epoch_start",
        "on_batch_start",
        "on_batch_end",
        "on_epoch_end",
    ):
        getattr(callback, hook)(context)
    tracker.save_artifact.assert_not_called()


def test_metrics_history_callback_accumulates_every_epoch_not_just_the_last() -> None:
    """Regression case for the gap ExperimentTrackingCallback alone has:
    log_metrics overwrites by key, so only the final epoch survives without
    this callback -- prove the full per-epoch sequence is captured instead."""
    callback = MetricsHistoryCallback()
    tracker = MagicMock()
    config = TrainingConfig(batch_size=4, learning_rate=0.01)

    for epoch, loss in enumerate([3.0, 2.0, 1.0]):
        context = RunContext(
            model=MagicMock(),
            tracker=tracker,
            config=config,
            total_epochs=3,
            epoch=epoch,
            metrics={"loss": loss},
        )
        callback.on_epoch_end(context)

    callback.on_train_end(context)

    tracker.save_artifact.assert_called_once()
    name, history = tracker.save_artifact.call_args[0]
    assert name == "metrics_history"
    assert isinstance(history, pd.DataFrame)
    assert list(history["epoch"]) == [0, 1, 2]
    assert list(history["loss"]) == [3.0, 2.0, 1.0]


def test_metrics_history_callback_uses_custom_artifact_name() -> None:
    context, tracker = _make_context(metrics={"loss": 1.0})
    callback = MetricsHistoryCallback(artifact_name="loss_curve")
    callback.on_epoch_end(context)
    callback.on_train_end(context)

    name, _ = tracker.save_artifact.call_args[0]
    assert name == "loss_curve"


def test_metrics_history_callback_times_each_epoch_from_its_own_start() -> None:
    """The wall-clock column of `training.parquet` (Annex 02 §2.2)."""
    callback = MetricsHistoryCallback()
    tracker = MagicMock()

    for epoch in range(3):
        context, _ = _make_context(metrics={"loss": 1.0})
        context.epoch = epoch
        context.phase = "warmup" if epoch < 2 else "refinement"
        callback.on_epoch_start(context)
        callback.on_epoch_end(context)

    context.tracker = tracker
    callback.on_train_end(context)

    _, history = tracker.save_artifact.call_args[0]
    assert list(history["epoch"]) == [0, 1, 2]
    assert list(history["phase"]) == ["warmup", "warmup", "refinement"]
    assert all(value > 0 for value in history["wall_time_s"])


def test_an_epoch_that_never_announced_its_start_records_no_duration() -> None:
    """`None`, not the time since the previous epoch began.

    A stamp left in place would make this row report an elapsed time that is
    wrong and entirely plausible -- and three tests in this file drive the
    callback exactly this way.
    """
    callback = MetricsHistoryCallback()
    context, _ = _make_context(metrics={"loss": 1.0})

    callback.on_epoch_start(context)
    callback.on_epoch_end(context)
    callback.on_epoch_end(context)  # second epoch: no start of its own

    timed, untimed = callback._history
    assert timed["wall_time_s"] > 0
    assert untimed["wall_time_s"] is None


def test_metrics_history_callback_does_not_save_before_train_end() -> None:
    context, tracker = _make_context(metrics={"loss": 1.0})
    callback = MetricsHistoryCallback()
    callback.on_epoch_end(context)
    tracker.save_artifact.assert_not_called()


def test_trajectory_logging_callback_saves_states_observations_and_controls() -> None:
    context, tracker = _make_context()
    states = np.zeros((2, 4, 3))
    observations = np.ones((2, 3, 3))
    controls = np.full((2, 3, 2), 0.5)
    rollout_fn = MagicMock(return_value=(states, observations, controls))
    batch_sampler = MagicMock(return_value=("x0", "w", "v"))

    callback = TrajectoryLoggingCallback(rollout_fn, batch_sampler)
    callback.on_train_end(context)

    batch_sampler.assert_called_once_with()
    rollout_fn.assert_called_once_with(("x0", "w", "v"))
    assert tracker.save_artifact.call_count == 3
    saved = {
        name: value
        for name, value in (c[0] for c in tracker.save_artifact.call_args_list)
    }
    assert set(saved) == {
        "trajectory_states",
        "trajectory_observations",
        "trajectory_controls",
    }
    assert np.array_equal(saved["trajectory_states"], states)
    assert np.array_equal(saved["trajectory_controls"], controls)


def test_trajectory_logging_callback_converts_torch_tensors_to_numpy() -> None:
    context, tracker = _make_context()
    states = torch.zeros(2, 4, 3, requires_grad=True)
    observations = torch.ones(2, 3, 3)
    controls = torch.full((2, 3, 2), 0.5)
    rollout_fn = MagicMock(return_value=(states, observations, controls))
    batch_sampler = MagicMock(return_value=("x0", "w", "v"))

    callback = TrajectoryLoggingCallback(rollout_fn, batch_sampler)
    callback.on_train_end(context)  # must not raise despite requires_grad=True

    saved = {
        name: value
        for name, value in (c[0] for c in tracker.save_artifact.call_args_list)
    }
    assert isinstance(saved["trajectory_states"], np.ndarray)
    assert np.array_equal(saved["trajectory_states"], np.zeros((2, 4, 3)))


def test_trajectory_logging_callback_uses_custom_artifact_prefix() -> None:
    context, tracker = _make_context()
    rollout_fn = MagicMock(return_value=(np.zeros(1), np.zeros(1), np.zeros(1)))
    batch_sampler = MagicMock(return_value=None)

    callback = TrajectoryLoggingCallback(
        rollout_fn, batch_sampler, artifact_prefix="final"
    )
    callback.on_train_end(context)

    names = {c[0][0] for c in tracker.save_artifact.call_args_list}
    assert names == {"final_states", "final_observations", "final_controls"}


def test_trajectory_logging_callback_only_fires_on_train_end() -> None:
    context, tracker = _make_context()
    rollout_fn = MagicMock()
    batch_sampler = MagicMock()
    callback = TrajectoryLoggingCallback(rollout_fn, batch_sampler)

    for hook in (
        "on_train_start",
        "on_epoch_start",
        "on_batch_start",
        "on_batch_end",
        "on_epoch_end",
    ):
        getattr(callback, hook)(context)

    rollout_fn.assert_not_called()
    batch_sampler.assert_not_called()
    tracker.save_artifact.assert_not_called()


def test_parameter_snapshot_callback_saves_each_named_parameter() -> None:
    context, tracker = _make_context()
    step_size = torch.full((3, 2), 0.1)
    riccati_matrix = np.eye(2)

    callback = ParameterSnapshotCallback(
        {"step_size": step_size, "riccati_matrix": riccati_matrix}
    )
    callback.on_train_end(context)

    assert tracker.save_artifact.call_count == 2
    saved = dict(c[0] for c in tracker.save_artifact.call_args_list)
    assert set(saved) == {"parameter_step_size", "parameter_riccati_matrix"}
    assert np.allclose(saved["parameter_step_size"], np.full((3, 2), 0.1), atol=1e-6)
    assert np.array_equal(saved["parameter_riccati_matrix"], np.eye(2))


def test_parameter_snapshot_callback_reads_the_trained_value_not_a_stale_copy() -> None:
    """Parameters are mutated in place by an optimizer -- the callback must
    read the live tensor at on_train_end time, not a value captured at
    construction."""
    context, tracker = _make_context()
    step_size = torch.nn.Parameter(torch.zeros(2))
    callback = ParameterSnapshotCallback({"step_size": step_size})

    with torch.no_grad():
        step_size.copy_(torch.ones(2) * 5.0)  # simulate an optimizer update
    callback.on_train_end(context)

    saved = dict(c[0] for c in tracker.save_artifact.call_args_list)
    assert np.array_equal(saved["parameter_step_size"], np.full(2, 5.0))


def test_parameter_snapshot_callback_evaluates_callable_values_fresh_at_train_end() -> (
    None
):
    """A reparameterized parameter (e.g. `StepSizeParameter.get`) computes a
    FRESH tensor from the raw nn.Parameter on every call -- it is not itself
    the live, in-place-mutated object, so a plain value (as in the test
    above) would freeze whatever it evaluated to at construction time,
    before training even starts. Passing the zero-argument getter itself
    (never invoking it early) is what lets on_train_end read the trained
    value."""
    context, tracker = _make_context()
    rho = torch.nn.Parameter(torch.zeros(2))

    def get_reparameterized() -> torch.Tensor:
        return torch.sigmoid(rho) * 5.0

    callback = ParameterSnapshotCallback({"step_size": get_reparameterized})
    with torch.no_grad():
        rho.copy_(torch.ones(2) * 100.0)  # simulate an optimizer update
    callback.on_train_end(context)

    saved = dict(c[0] for c in tracker.save_artifact.call_args_list)
    # sigmoid(100) * 5 ~= 5.0, not sigmoid(0) * 5 == 2.5 (the stale value a
    # plain `get_reparameterized()` call at construction time would freeze).
    assert np.allclose(saved["parameter_step_size"], np.full(2, 5.0), atol=1e-3)


def test_parameter_snapshot_callback_converts_torch_tensors_to_numpy() -> None:
    context, tracker = _make_context()
    callback = ParameterSnapshotCallback({"p": torch.ones(2, 2, requires_grad=True)})
    callback.on_train_end(context)  # must not raise despite requires_grad=True

    saved = dict(c[0] for c in tracker.save_artifact.call_args_list)
    assert isinstance(saved["parameter_p"], np.ndarray)


def test_parameter_snapshot_callback_uses_custom_artifact_prefix() -> None:
    context, tracker = _make_context()
    callback = ParameterSnapshotCallback({"p": np.zeros(1)}, artifact_prefix="learned")
    callback.on_train_end(context)

    name, _ = tracker.save_artifact.call_args[0]
    assert name == "learned_p"


def test_parameter_snapshot_callback_only_fires_on_train_end() -> None:
    context, tracker = _make_context()
    callback = ParameterSnapshotCallback({"p": np.zeros(1)})

    for hook in (
        "on_train_start",
        "on_epoch_start",
        "on_batch_start",
        "on_batch_end",
        "on_epoch_end",
    ):
        getattr(callback, hook)(context)

    tracker.save_artifact.assert_not_called()


def _drive_profiling_epochs(callback: ProfilingCallback, context, n_epochs: int):
    callback.on_train_start(context)
    for epoch in range(n_epochs):
        context.epoch = epoch
        callback.on_epoch_start(context)
        sum(range(1000))  # do a little work so timings are non-trivial
        callback.on_epoch_end(context)
    callback.on_train_end(context)


def test_profiling_callback_logs_latency_summary_into_metadata() -> None:
    context, tracker = _make_context()
    _drive_profiling_epochs(ProfilingCallback(), context, n_epochs=3)

    tracker.log_metrics.assert_called_once()
    (summary,) = tracker.log_metrics.call_args[0]
    # Latency scalars are always captured (RAM/VRAM depend on platform).
    assert summary["profile/num_epochs_profiled"] == 3.0
    assert summary["profile/train_wall_time_total_s"] >= 0.0
    assert "profile/epoch_wall_time_mean_s" in summary
    assert "profile/epoch_wall_time_p95_s" in summary
    assert summary["profile/epoch_cpu_time_total_s"] >= 0.0


def test_profiling_callback_saves_per_epoch_history_artifact() -> None:
    context, tracker = _make_context()
    _drive_profiling_epochs(ProfilingCallback(), context, n_epochs=2)

    tracker.save_artifact.assert_called_once()
    name, history = tracker.save_artifact.call_args[0]
    assert name == "profiling_history"
    assert isinstance(history, pd.DataFrame)
    assert list(history["epoch"]) == [0, 1]
    assert (history["wall_time_s"] >= 0.0).all()


def test_profiling_callback_drops_epoch_level_stats_when_not_trainable() -> None:
    """A single analytical rollout has no meaningful per-epoch statistics --
    mean/p50/p95 over one sample is redundant noise, not a statistic -- so
    only the aggregate scalars (whole-run wall time, peak RAM/VRAM) survive."""
    context, tracker = _make_context(is_trainable=False)
    _drive_profiling_epochs(ProfilingCallback(), context, n_epochs=1)

    (summary,) = tracker.log_metrics.call_args[0]
    assert "profile/num_epochs_profiled" not in summary
    assert "profile/epoch_wall_time_mean_s" not in summary
    assert "profile/epoch_wall_time_p50_s" not in summary
    assert "profile/epoch_wall_time_p95_s" not in summary
    assert "profile/epoch_cpu_time_total_s" not in summary
    assert summary["profile/train_wall_time_total_s"] >= 0.0


def test_profiling_callback_does_nothing_without_epochs() -> None:
    context, tracker = _make_context()
    callback = ProfilingCallback()
    callback.on_train_start(context)
    callback.on_train_end(context)  # no epochs profiled

    tracker.log_metrics.assert_not_called()
    tracker.save_artifact.assert_not_called()


def test_early_stopping_callback_never_fires_on_the_first_epoch() -> None:
    """No previous value exists yet at epoch 0, so stop_requested must stay
    False no matter how small the metric already is."""
    callback = EarlyStoppingCallback(tolerance=1.0)
    context, _ = _make_context(epoch=0, metrics={"loss": 0.0})
    callback.on_epoch_end(context)
    assert context.stop_requested is False


def test_early_stopping_callback_requests_stop_within_tolerance() -> None:
    callback = EarlyStoppingCallback(tolerance=0.1)
    context, _ = _make_context(epoch=0, metrics={"loss": 1.0})
    callback.on_epoch_end(context)
    assert context.stop_requested is False

    context.epoch = 1
    context.metrics = {"loss": 1.05}  # |1.05 - 1.0| = 0.05 < tolerance
    callback.on_epoch_end(context)
    assert context.stop_requested is True


def test_early_stopping_callback_does_not_stop_outside_tolerance() -> None:
    callback = EarlyStoppingCallback(tolerance=0.1)
    context, _ = _make_context(epoch=0, metrics={"loss": 1.0})
    callback.on_epoch_end(context)

    context.epoch = 1
    context.metrics = {"loss": 0.5}  # |0.5 - 1.0| = 0.5 >= tolerance
    callback.on_epoch_end(context)
    assert context.stop_requested is False


def test_early_stopping_callback_uses_custom_metric_key() -> None:
    callback = EarlyStoppingCallback(tolerance=0.1, metric_key="cost")
    context, _ = _make_context(epoch=0, metrics={"cost": 2.0})
    callback.on_epoch_end(context)

    context.epoch = 1
    context.metrics = {"cost": 2.02}
    callback.on_epoch_end(context)
    assert context.stop_requested is True


def test_control_sequence_history_callback_captures_one_snapshot_per_epoch() -> None:
    controller = MagicMock()
    controller.U = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    callback = ControlSequenceHistoryCallback(controller)
    context, tracker = _make_context()

    for value in (1.0, 2.0, 3.0):
        controller.U = torch.full((2, 2), value)
        callback.on_epoch_start(context)
    callback.on_train_end(context)

    tracker.save_artifact.assert_called_once()
    name, history = tracker.save_artifact.call_args[0]
    assert name == "control_history"
    assert history.shape == (3, 2, 2)
    assert np.allclose(history[1], 2.0)


def test_control_sequence_history_callback_snapshots_before_the_update_within_an_epoch() -> (
    None
):
    """on_epoch_start must see U^(i) (pre-update), not U^(i+1) -- so a
    snapshot taken before an in-place mutation reflects the old value."""
    controller = MagicMock()
    controller.U = torch.zeros(1, 3)
    callback = ControlSequenceHistoryCallback(controller)
    context, _ = _make_context()

    callback.on_epoch_start(context)
    with torch.no_grad():
        controller.U.add_(1.0)  # simulate the optimizer's in-place update

    assert np.allclose(callback._history[0], 0.0)


def test_control_sequence_history_callback_aligns_with_metrics_history_callback() -> (
    None
):
    """The (U^(i), J^(i)) pairing: both callbacks driven through the same
    epoch sequence must end up with equal-length, index-aligned histories."""
    controller = MagicMock()
    controller.U = torch.zeros(1, 2)
    control_cb = ControlSequenceHistoryCallback(controller)
    metrics_cb = MetricsHistoryCallback()

    for epoch, loss in enumerate([3.0, 2.0, 1.0]):
        context, _ = _make_context(epoch=epoch, total_epochs=3, metrics={"loss": loss})
        controller.U = torch.full((1, 2), float(epoch))
        control_cb.on_epoch_start(context)
        metrics_cb.on_epoch_end(context)

    assert len(control_cb._history) == len(metrics_cb._history) == 3
    for epoch in range(3):
        assert np.allclose(control_cb._history[epoch], float(epoch))
        assert metrics_cb._history[epoch]["loss"] == [3.0, 2.0, 1.0][epoch]


def test_control_sequence_history_callback_uses_custom_artifact_name() -> None:
    controller = MagicMock()
    controller.U = torch.zeros(1, 2)
    callback = ControlSequenceHistoryCallback(controller, artifact_name="u_history")
    context, tracker = _make_context()

    callback.on_epoch_start(context)
    callback.on_train_end(context)

    name, _ = tracker.save_artifact.call_args[0]
    assert name == "u_history"


def _build_signature_test_problem(horizon: int = 1):
    import numpy as np

    from mbl.core.cost.quadratic_cost import QuadraticCost
    from mbl.core.optimal_control_problem import OptimalControlProblem
    from mbl.core.system.linear_system import LinearSystem

    system = LinearSystem.fully_observable(np.eye(2), np.eye(2))
    # RiccatiController's finite_horizon_riccati requires genuinely
    # time-stacked Q/R (it indexes Q[horizon]/R[k]).
    cost = QuadraticCost(
        Q=np.repeat(np.eye(2)[None], horizon + 1, axis=0),
        R=np.repeat(np.eye(2)[None], horizon, axis=0),
    )
    return OptimalControlProblem(system=system, cost=cost)


def test_problem_signature_callback_logs_signature_problem_controller_and_sampling() -> (
    None
):
    from mbl.models.analytic.riccati import RiccatiController
    from mbl.models.samplers import NumpyGaussianDistribution, ZeroDistribution

    problem = _build_signature_test_problem()
    controller = RiccatiController(problem, horizon=1)
    distributions = {
        "initial_state": NumpyGaussianDistribution(std=1.0, seed=0),
        "process_noise": NumpyGaussianDistribution(std=0.5, seed=0),
        "measurement_noise": ZeroDistribution(),
    }

    context, tracker = _make_context()
    ProblemSignatureCallback(problem, controller, distributions).on_train_start(context)

    logged = tracker.log_params.call_args[0][0]
    assert "signature" in logged
    assert logged["problem.system.type"] == "LinearSystem"
    assert logged["problem.cost.type"] == "QuadraticCost"
    assert logged["controller.type"] == "RiccatiController"
    assert logged["controller.horizon"] == 1
    assert logged["sampling.initial_state.std"] == 1.0
    assert logged["sampling.process_noise.std"] == 0.5
    assert logged["sampling.measurement_noise.type"] == "Zero"


def test_problem_signature_callback_digest_is_stable_for_identical_setup() -> None:
    from mbl.models.analytic.riccati import RiccatiController

    problem_1 = _build_signature_test_problem()
    problem_2 = _build_signature_test_problem()
    controller_1 = RiccatiController(problem_1, horizon=1)
    controller_2 = RiccatiController(problem_2, horizon=1)

    context_1, tracker_1 = _make_context()
    context_2, tracker_2 = _make_context()
    ProblemSignatureCallback(problem_1, controller_1).on_train_start(context_1)
    ProblemSignatureCallback(problem_2, controller_2).on_train_start(context_2)

    assert (
        tracker_1.log_params.call_args[0][0]["signature"]
        == tracker_2.log_params.call_args[0][0]["signature"]
    )


def test_problem_signature_callback_digest_differs_for_different_horizon() -> None:
    from mbl.models.analytic.riccati import RiccatiController

    problem_1 = _build_signature_test_problem(horizon=1)
    problem_2 = _build_signature_test_problem(horizon=2)
    context_1, tracker_1 = _make_context()
    context_2, tracker_2 = _make_context()
    ProblemSignatureCallback(
        problem_1, RiccatiController(problem_1, horizon=1)
    ).on_train_start(context_1)
    ProblemSignatureCallback(
        problem_2, RiccatiController(problem_2, horizon=2)
    ).on_train_start(context_2)

    assert (
        tracker_1.log_params.call_args[0][0]["signature"]
        != tracker_2.log_params.call_args[0][0]["signature"]
    )


def test_problem_signature_callback_defaults_to_no_sampling_distributions() -> None:
    """No distributions given -> no sampling.* keys at all (an empty dict, unlike
    an empty constraints list, has no leaves to preserve -- this is the honest
    signal that no sampling distributions were recorded for this run)."""
    from mbl.models.analytic.riccati import RiccatiController

    problem = _build_signature_test_problem()
    controller = RiccatiController(problem, horizon=1)
    context, tracker = _make_context()

    ProblemSignatureCallback(problem, controller).on_train_start(context)

    logged = tracker.log_params.call_args[0][0]
    assert not any(key.startswith("sampling.") for key in logged)
    assert "signature" in logged
    assert "controller.type" in logged


def test_problem_signature_callback_composes_with_experiment_tracking_callback_no_collision() -> (
    None
):
    """Both callbacks call log_params independently; their key namespaces
    must never collide (ExperimentTrackingCallback: TrainingConfig fields;
    ProblemSignatureCallback: signature/problem.*/controller.*/sampling.*)."""
    from mbl.models.analytic.riccati import RiccatiController

    problem = _build_signature_test_problem()
    controller = RiccatiController(problem, horizon=1)
    context, tracker = _make_context(is_trainable=False)

    ExperimentTrackingCallback().on_train_start(context)
    ProblemSignatureCallback(problem, controller).on_train_start(context)

    all_calls = [call.args[0] for call in tracker.log_params.call_args_list]
    merged: dict = {}
    for call_params in all_calls:
        assert not (set(merged) & set(call_params)), "key collision between callbacks"
        merged.update(call_params)

    assert "batch_size" in merged
    assert "learning_rate" not in merged  # filtered (is_trainable=False)
    assert "signature" in merged
    assert "controller.type" in merged
