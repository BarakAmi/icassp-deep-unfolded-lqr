"""What a record says about how it came to be — Annex 02 §2.2.

Written before the implementation. These run the **real** producer over a real
study into `tmp_path`; nothing here is asserted against a double, because the
gap being closed is precisely that a frame was built by the real machinery and
then dropped by the real machinery.

The study fixture is `test_producer`'s, deliberately: one trainable contender
that carries a history and one closed-form contender that must not, in a shape
this suite already trusts. The layer-wise study below is this file's own,
because no other test needs a family whose training changes phase.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import pandas as pd
import pytest

from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.runner.producer import run_study
from mbl.spec.loader import load_study
from mbl.store.content_store import MeasurementStore, ModelStore
from mbl.store.layout import SYNTHESIS_LOG, TRAINING_HISTORY

from .test_producer import DEFAULTS, STUDY, _problem_file, _study

#: The evaluation batch count the fixture study declares.
EVAL_BATCHES = 2

#: The epoch count the fixture study declares for its trainable contender.
#: Cross-checked against the document itself below, so raising it there cannot
#: silently weaken the row-count assertion into "some rows exist".
EPOCHS = 8

#: One learned contender whose training runs in three phases: two layer warm-ups
#: and a refinement. The phase column is unreadable without a family like this,
#: and `unfolded_warmstart` is the family the flagship contender uses.
LAYERWISE_STUDY = """
id = "probe/layerwise"

[problem]
path = "problem.npz"

[compute]
backend = "torch"
device = "cpu"
precision = "float64"

[training]
seeds = [0]

[training.data]
kind = "gaussian"
seed = 11
process_noise_std = 0.35
initial_state_std = 0.8

[training.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 4

[training.batch]
effective_size = 16

[evaluation]
metrics = ["expected_cost"]

[evaluation.protocol]
_build = "protocol"
n_batches = 2

[evaluation.protocol.batch_spec]
_build = "gaussian"
batch_size = 8
seed = 7
process_noise_std = 0.9
initial_state_std = 1.0

[[contenders]]
label = "warmstart"
family = "unfolded_warmstart"

[contenders.config]
kind = "learned_step_size_and_matrix"
num_iterations = 2
step_size_init = 0.1
step_size_max = 1.0
horizon = 6

[contenders.config.schedule]
_build = "layerwise"
optimizer = "adam"
learning_rate = 0.05
warmup_epochs_per_layer = 2
refinement_epochs = 3
train_matrix_from = "refinement"
activation = "single"
"""

#: `warmup_epochs_per_layer * num_iterations + refinement_epochs`.
LAYERWISE_EPOCHS = 2 * 2 + 3


@pytest.fixture(scope="module")
def executed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ModelStore]:
    """The two-contender study, run once for every assertion in this file."""
    directory = tmp_path_factory.mktemp("payloads")
    study = _study(directory, depths=[2])
    store = directory / "store"
    run_study(study, store=store, bindings=DEFAULT_SPEC_BINDINGS)
    return store, ModelStore(store)


def _records(store: ModelStore, family: str) -> list:
    return [
        record
        for record in (store.get(model_id) for model_id in store.list_ids())
        if record.spec["family"] == family
    ]


# --------------------------------------------------------------------------
# The offline payload
# --------------------------------------------------------------------------


def test_a_trained_model_carries_one_row_per_epoch(
    executed: tuple[Path, ModelStore],
) -> None:
    """The gap this closes: 162 stored records, zero `training.parquet`."""
    _, models = executed
    trained = _records(models, "unfolded")
    assert trained, "the fixture must produce at least one trained model"

    for record in trained:
        history = record.history
        assert history is not None
        assert len(history) == EPOCHS
        assert list(history["epoch"]) == list(range(EPOCHS))
        assert {"epoch", "phase", "wall_time_s"} <= set(history.columns)
        # The strategy's own metric, whatever it is called -- named by
        # subtraction so a renamed loss key is not silently accepted as absent.
        assert set(history.columns) - {"epoch", "phase", "wall_time_s"}
        assert (history["wall_time_s"] > 0).all()


def test_an_analytic_model_carries_no_history_and_no_member(
    executed: tuple[Path, ModelStore],
) -> None:
    """A family with no training phase has no training curve, and the record
    must show that by *absence* rather than by an empty frame."""
    store_root, models = executed
    analytic = _records(models, "truncated_riccati")
    assert analytic, "the fixture must produce at least one analytic model"

    for record in analytic:
        assert record.history is None
        assert not (models.path(record.model_id) / TRAINING_HISTORY).exists()


def test_the_history_survives_a_round_trip_through_the_store(
    executed: tuple[Path, ModelStore],
) -> None:
    """Read back off disk, not out of the run: the parquet is the artifact."""
    _, models = executed
    record = _records(models, "unfolded")[0]
    on_disk = pd.read_parquet(models.path(record.model_id) / TRAINING_HISTORY)

    pd.testing.assert_frame_equal(on_disk, record.history)


def test_the_phase_column_names_the_phase_that_produced_each_row(
    tmp_path: Path,
) -> None:
    """A layer-wise family trains one phase per unfolding iteration, each with a
    fresh optimizer. Without this column the curve cannot distinguish "this
    layer never converged" from "the refinement undid it"."""
    _problem_file(tmp_path, "problem.npz", seed=0)
    document = tmp_path / "layerwise.toml"
    document.write_text(LAYERWISE_STUDY)
    study = load_study(document, bindings=DEFAULT_SPEC_BINDINGS).study

    run_study(study, store=tmp_path / "store", bindings=DEFAULT_SPEC_BINDINGS)

    models = ModelStore(tmp_path / "store")
    (record,) = [models.get(model_id) for model_id in models.list_ids()]
    history = record.history
    assert history is not None
    assert len(history) == LAYERWISE_EPOCHS
    # Three phases, in order, each contributing its declared number of epochs.
    runs = [name for name, _ in itertools.groupby(history["phase"])]
    assert len(runs) == 3, f"expected three contiguous phases, got {runs}"
    assert list(history["phase"]).count(runs[-1]) == 3  # refinement_epochs


def test_the_study_document_and_the_fixture_agree_on_the_epoch_count() -> None:
    """The two numbers this file asserts against are one number in the study.

    Without this, raising the fixture's epochs would silently weaken
    `test_a_trained_model_carries_one_row_per_epoch` into "some rows exist".
    """
    document = STUDY.format(**DEFAULTS)
    assert f"epochs = {EPOCHS}" in document
    assert f"n_batches = {EVAL_BATCHES}" in document


# --------------------------------------------------------------------------
# The online payload
# --------------------------------------------------------------------------


def test_a_measurement_carries_one_row_per_evaluation_batch(
    executed: tuple[Path, ModelStore],
) -> None:
    """`eval_batch_costs` is computed on every evaluation and was reduced to one
    scalar and discarded, so a measurement reported an expected cost and nothing
    about the spread it came from."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    identifiers = list(measurements.list_ids())
    assert identifiers

    for measurement_id in identifiers:
        record = measurements.get(measurement_id)
        trace = record.trace
        assert trace is not None
        assert list(trace.columns) == ["batch_index", "batch_cost", "wall_time_s"]
        assert list(trace["batch_index"]) == list(range(EVAL_BATCHES))
        assert (trace["wall_time_s"] > 0).all()


def test_the_trace_holds_the_numbers_the_metric_was_reduced_from(
    executed: tuple[Path, ModelStore],
) -> None:
    """Not "a plausible cost per batch": the same array, reduced the same way.

    `eval_expected_cost` is `costs.mean()` over exactly this column, so the two
    agree bit for bit -- and a trace built from the wrong array (the
    per-trajectory one, say) would agree only to a few decimals.
    """
    store_root, _ = executed
    measurements = MeasurementStore(store_root)

    for measurement_id in measurements.list_ids():
        record = measurements.get(measurement_id)
        assert record.trace is not None
        column = np.asarray(record.trace["batch_cost"], dtype=np.float64)
        assert column.mean() == record.metrics["eval_expected_cost"]
        # ... and it is a spread, not one number repeated: two independent
        # evaluation batches of a stochastic rollout never agree exactly.
        assert len(set(column.tolist())) == EVAL_BATCHES


def test_the_trace_and_the_samples_describe_the_same_batches(
    executed: tuple[Path, ModelStore],
) -> None:
    """The two payloads pair on `batch_index` (Annex 03 §A.4) and are two named
    reductions of one rollout, so they agree to floating-point association order
    and not beyond it."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    record = measurements.get(next(iter(measurements.list_ids())))
    assert record.trace is not None and record.samples is not None

    per_batch = record.samples.groupby("batch_index")["trajectory_cost"].mean()
    np.testing.assert_allclose(
        per_batch.to_numpy(), record.trace["batch_cost"].to_numpy(), rtol=1e-12
    )


# --------------------------------------------------------------------------
# The constraint-activity columns (Annex 03 §A.3.1a)
# --------------------------------------------------------------------------


def test_the_samples_carry_the_constraint_activity_columns(
    executed: tuple[Path, ModelStore],
) -> None:
    """§A.3.1a. Columns on the cost's own pairing key rather than a second
    frame: they describe the same trajectory, drawn in the same rollout, and a
    reader asking which trajectories were pinned to the bound and what they
    cost should not have to join two payloads to find out."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    record = measurements.get(next(iter(measurements.list_ids())))
    assert record.samples is not None
    assert {"max_abs_control", "control_saturation"} <= set(record.samples.columns)
    assert len(record.samples["max_abs_control"]) == len(record.samples)


def test_the_columns_survive_the_round_trip_through_the_store(
    executed: tuple[Path, ModelStore],
) -> None:
    """A parquet round trip is where a float column quietly becomes something
    else. The study's box is 0.5, so every value is bounded by it and every
    fraction lies in the unit interval — a column that came back as an index,
    a string or an object would fail one of the three."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    for measurement_id in measurements.list_ids():
        samples = measurements.get(measurement_id).samples
        assert samples is not None
        maxima = samples["max_abs_control"].to_numpy()
        fractions = samples["control_saturation"].to_numpy()
        assert maxima.dtype == fractions.dtype == np.float64
        assert (maxima <= 0.5 + 1e-12).all()
        assert ((fractions >= 0.0) & (fractions <= 1.0)).all()


def test_the_metric_is_the_reduction_of_the_column_it_summarises(
    executed: tuple[Path, ModelStore],
) -> None:
    """The scalar a gate reads and the payload an analysis re-reduces must be
    the same measurement. Two reductions, and deliberately not the same one:
    feasibility is a claim about the worst entry, activity about the average."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    for measurement_id in measurements.list_ids():
        record = measurements.get(measurement_id)
        assert record.samples is not None
        assert record.metrics["eval_max_abs_control"] == pytest.approx(
            record.samples["max_abs_control"].max()
        )
        assert record.metrics["eval_saturation_fraction"] == pytest.approx(
            record.samples["control_saturation"].mean()
        )


def test_every_stored_contender_is_feasible_and_meets_the_box(
    executed: tuple[Path, ModelStore],
) -> None:
    """What the columns are FOR, asserted on real records rather than on a
    hand-built frame: every contender the producer stored respects the declared
    box, and the box is active often enough for the study to be about it."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    for measurement_id in measurements.list_ids():
        metrics = measurements.get(measurement_id).metrics
        assert metrics["eval_max_abs_control"] <= 0.5 + 1e-12
        assert metrics["eval_saturation_fraction"] > 0.0


def test_the_metrics_reader_agrees_with_the_full_record(
    executed: tuple[Path, ModelStore],
) -> None:
    """`MeasurementStore.metrics` is the light path the gate reads through, so
    it must not become a second source that can disagree with the record."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)
    for measurement_id in measurements.list_ids():
        assert measurements.metrics(measurement_id) == (
            measurements.get(measurement_id).metrics
        )


def test_the_online_pass_is_timed_into_the_measurement_provenance(
    executed: tuple[Path, ModelStore],
) -> None:
    """Annex 04 §7's online compute row had no source at all before this."""
    store_root, _ = executed
    measurements = MeasurementStore(store_root)

    for measurement_id in measurements.list_ids():
        record = measurements.get(measurement_id)
        assert record.trace is not None
        recorded = float(record.spec["provenance"]["wall_time_s"])
        batches = float(record.trace["wall_time_s"].sum())
        # The pass includes building the batches, which is real time this
        # measurement paid for and which no per-batch row can hold.
        assert recorded >= batches > 0


# --------------------------------------------------------------------------
# The captured logs (Annex 02 §8, whose member §2 never named)
# --------------------------------------------------------------------------


def test_a_model_log_names_the_point_it_belongs_to(
    executed: tuple[Path, ModelStore],
) -> None:
    """A log that began mid-training is one a reader has to correlate by hand,
    so the capture opens before the line that names the model."""
    _, models = executed

    for model_id in models.list_ids():
        record = models.get(model_id)
        assert record.log is not None
        assert model_id in record.log
        assert "synthesizing on" in record.log


def test_a_measurement_log_records_the_online_pass(
    executed: tuple[Path, ModelStore],
) -> None:
    store_root, _ = executed
    measurements = MeasurementStore(store_root)

    for measurement_id in measurements.list_ids():
        record = measurements.get(measurement_id)
        assert record.log is not None
        assert "Online evaluation finished" in record.log


def test_the_log_is_stored_as_text_and_read_back_unchanged(
    executed: tuple[Path, ModelStore],
) -> None:
    _, models = executed
    model_id = next(iter(models.list_ids()))
    record = models.get(model_id)

    on_disk = (models.path(model_id) / SYNTHESIS_LOG).read_text(encoding="utf-8")
    assert on_disk == record.log


def test_the_logs_do_not_leak_between_points(
    executed: tuple[Path, ModelStore],
) -> None:
    """Each capture covers one phase. A handler left attached would make every
    later point's log contain every earlier point's -- which grows as the square
    of the study and would be read as "this model trained for hours"."""
    _, models = executed
    identifiers = sorted(models.list_ids())
    assert len(identifiers) > 1

    for model_id in identifiers:
        log = models.get(model_id).log or ""
        others = [other for other in identifiers if other != model_id]
        assert not [other for other in others if other in log]
