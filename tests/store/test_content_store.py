"""Acceptance tests for the content-addressed stores (Annex 02 §2, §7).

Written before the implementation. The load-bearing property is atomicity: a
torn write must leave nothing that a later run could mistake for a finished
record. That is not an operational nicety -- a half-written directory that
`exists()` accepts causes the runner to skip the node and the study to proceed
on a truncated checkpoint, which is a correctness defect wearing an operational
costume.
"""

from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
import torch

from mbl.store.content_store import (
    ContentConflictError,
    MeasurementRecord,
    MeasurementStore,
    ModelRecord,
    ModelStore,
    RecordNotFoundError,
)
from mbl.store.ids import MeasurementID, ModelID

MODEL_A = ModelID("0123456789abcdef")
MODEL_B = ModelID("fedcba9876543210")
MEASUREMENT_A = MeasurementID("aaaabbbbccccdddd")


def _weights() -> dict[str, torch.Tensor]:
    return {
        "alpha": torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
        "L": torch.eye(3, dtype=torch.float32),
        "depth": torch.tensor(10, dtype=torch.int64),
    }


def _model_record(model_id: ModelID = MODEL_A, *, epochs: int = 200) -> ModelRecord:
    return ModelRecord(
        model_id=model_id,
        spec={
            "contender": {"family": "unfolded"},
            "training": {"plan": {"epochs": epochs}},
        },
        weights=_weights(),
        history=pd.DataFrame({"epoch": [0, 1, 2], "loss": [1.0, 0.5, 0.25]}),
        log="Synthesis (training) started for 'unfolded_alpha_p'\n",
        provenance={"stamp": "mbl-0.1.0/schema-1", "wall_time_s": 12.5},
    )


def _measurement_record() -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=MEASUREMENT_A,
        spec={"model": str(MODEL_A), "eval_problem": "0123456789abcdef"},
        metrics={"expected_cost": 3.25, "constraint_activity": 0.11},
        samples=pd.DataFrame({"trajectory": [0, 1], "cost": [3.1, 3.4]}),
        trace=pd.DataFrame({"batch_index": [0, 1], "wall_time_s": [0.5, 0.4]}),
        log="Online evaluation finished\n",
    )


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    return ModelStore(tmp_path / "store")


# --------------------------------------------------------------------------
# Round-trip fidelity -- the Stage 1 gate
# --------------------------------------------------------------------------


def test_a_checkpoint_round_trips_exactly(store: ModelStore) -> None:
    store.put(_model_record())
    loaded = store.get(MODEL_A)

    assert set(loaded.weights) == set(_weights())
    for key, original in _weights().items():
        assert loaded.weights[key].dtype == original.dtype, key
        assert loaded.weights[key].shape == original.shape, key
        assert torch.equal(loaded.weights[key], original), key


def test_spec_and_provenance_round_trip_exactly(store: ModelStore) -> None:
    record = _model_record()
    store.put(record)
    loaded = store.get(MODEL_A)

    assert loaded.spec == record.spec
    assert loaded.provenance == record.provenance


def test_training_history_round_trips(store: ModelStore) -> None:
    store.put(_model_record())
    loaded = store.get(MODEL_A)
    assert loaded.history is not None
    pd.testing.assert_frame_equal(loaded.history, _model_record().history)


def test_a_model_without_history_round_trips_as_none(store: ModelStore) -> None:
    record = ModelRecord(
        model_id=MODEL_B,
        spec={"contender": {"family": "riccati"}},
        weights={},
        history=None,
        log=None,
        provenance={"stamp": "mbl-0.1.0/schema-1"},
    )
    store.put(record)
    assert store.get(MODEL_B).history is None


def test_an_empty_model_log_is_written_as_no_member_at_all(store: ModelStore) -> None:
    """A zero-byte `synthesis.log` is indistinguishable from a phase whose log
    was lost, and reads as "captured, and it had nothing to say"."""
    store.put(dataclasses.replace(_model_record(), log=""))

    assert not (store.path(MODEL_A) / "synthesis.log").exists()
    assert store.get(MODEL_A).log is None


def test_an_empty_measurement_log_is_written_as_no_member_at_all(
    tmp_path: Path,
) -> None:
    """The same rule on the other store. Stated twice because it is implemented
    twice; a rule that holds on one of two records is a rule with a hole."""
    measurements = MeasurementStore(tmp_path / "store")
    measurements.put(dataclasses.replace(_measurement_record(), log=""))

    assert not (measurements.path(MEASUREMENT_A) / "evaluation.log").exists()
    assert measurements.get(MEASUREMENT_A).log is None


def test_the_written_bytes_are_deterministic(tmp_path: Path) -> None:
    a, b = ModelStore(tmp_path / "a"), ModelStore(tmp_path / "b")
    a.put(_model_record())
    b.put(_model_record())

    for name in ("spec.json", "weights.safetensors", "provenance.json"):
        assert (a.path(MODEL_A) / name).read_bytes() == (
            b.path(MODEL_A) / name
        ).read_bytes()


def test_key_insertion_order_does_not_change_the_written_bytes(tmp_path: Path) -> None:
    """Serialisation must be canonical, not merely repeatable.

    Two processes can build a logically identical spec by different routes and
    end up with different key insertion order. Without sorted keys the JSON
    bytes differ, so the manifest digests differ, so a perfectly correct `put`
    raises ContentConflictError -- a false positive that looks exactly like an
    identity-derivation bug.

    Established by mutation: dropping `sort_keys` survived the whole file until
    this test existed, because every other case builds the same dict literal
    twice and therefore never varies the order.
    """
    forward = {"alpha": 1, "beta": {"x": 1, "y": 2}, "gamma": 3}
    reversed_order = {"gamma": 3, "beta": {"y": 2, "x": 1}, "alpha": 1}
    assert list(forward) != list(reversed_order)

    a, b = ModelStore(tmp_path / "a"), ModelStore(tmp_path / "b")
    a.put(ModelRecord(MODEL_A, forward, {}, None, None, forward))
    b.put(ModelRecord(MODEL_A, reversed_order, {}, None, None, reversed_order))

    for name in ("spec.json", "provenance.json", "manifest.json"):
        assert (a.path(MODEL_A) / name).read_bytes() == (
            b.path(MODEL_A) / name
        ).read_bytes()


def test_reordered_content_is_not_treated_as_a_conflict(tmp_path: Path) -> None:
    """The consequence of the above, stated as the behaviour that matters."""
    store = ModelStore(tmp_path / "store")
    store.put(ModelRecord(MODEL_A, {"a": 1, "b": 2}, {}, None, None, {"p": 1, "q": 2}))
    store.put(ModelRecord(MODEL_A, {"b": 2, "a": 1}, {}, None, None, {"q": 2, "p": 1}))
    assert store.exists(MODEL_A)


# --------------------------------------------------------------------------
# Atomicity -- the property that matters
# --------------------------------------------------------------------------


def test_a_failure_mid_write_leaves_no_record_at_all(
    store: ModelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish step is the only moment a record becomes visible. Failing
    before it must leave `exists()` False -- not a directory a later run would
    skip as already done."""
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        store.put(_model_record())

    assert not store.exists(MODEL_A)
    assert list(store.list_ids()) == []


def test_a_failure_mid_write_leaves_nothing_readable(
    store: ModelStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(RuntimeError):
        store.put(_model_record())

    with pytest.raises(RecordNotFoundError):
        store.get(MODEL_A)


def test_staging_leftovers_are_never_mistaken_for_records(store: ModelStore) -> None:
    """A crashed write can leave a staging directory behind. It must not be
    listed, must not satisfy `exists`, and must not be loadable."""
    store.put(_model_record())
    leftover = store.staging_root / "some-abandoned-write"
    leftover.mkdir(parents=True)
    (leftover / "spec.json").write_text("{}")

    assert list(store.list_ids()) == [MODEL_A]


def test_a_directory_holding_only_a_partial_is_not_a_record(store: ModelStore) -> None:
    """`.partial/` is a resume artifact for an in-flight training (Annex 02
    §7.2), never a result. A model whose directory holds only a partial must
    read as absent, or the runner would skip retraining it."""
    partial = store.path(MODEL_B) / ".partial"
    partial.mkdir(parents=True)
    (partial / "state.safetensors").write_bytes(b"in-flight")

    assert not store.exists(MODEL_B)
    assert MODEL_B not in list(store.list_ids())


def test_an_incomplete_record_directory_reads_as_absent(store: ModelStore) -> None:
    """Belt-and-braces for a torn write that somehow lands in place: a record
    missing any required member is not a record."""
    store.put(_model_record())
    (store.path(MODEL_A) / "weights.safetensors").unlink()

    assert not store.exists(MODEL_A)


# --------------------------------------------------------------------------
# Content addressing: same id must mean same content
# --------------------------------------------------------------------------


def test_putting_identical_content_twice_is_a_no_op(store: ModelStore) -> None:
    store.put(_model_record())
    first = (store.path(MODEL_A) / "weights.safetensors").stat().st_mtime_ns
    store.put(_model_record())

    assert store.exists(MODEL_A)
    assert (store.path(MODEL_A) / "weights.safetensors").stat().st_mtime_ns == first


def test_putting_different_content_under_one_id_is_refused(store: ModelStore) -> None:
    """Two different contents under one id means identity derivation is broken.
    Overwriting would hide that; raising surfaces it."""
    store.put(_model_record(epochs=200))

    with pytest.raises(ContentConflictError, match="differs from the stored"):
        store.put(_model_record(epochs=400))


def test_the_original_survives_a_refused_conflicting_put(store: ModelStore) -> None:
    store.put(_model_record(epochs=200))
    with pytest.raises(ContentConflictError):
        store.put(_model_record(epochs=400))

    assert store.get(MODEL_A).spec["training"]["plan"]["epochs"] == 200


# --------------------------------------------------------------------------
# The no-pickle law (Annex 02 §7, retained from the current system)
# --------------------------------------------------------------------------


def test_no_pickle_bearing_file_is_ever_written(store: ModelStore) -> None:
    store.put(_model_record())
    written = {p.suffix for p in store.path(MODEL_A).rglob("*") if p.is_file()}
    # `.log` joined the permitted set with Annex 02 §2.2: the law is about
    # payloads that execute on load, and a captured log is inert text.
    assert written <= {".json", ".safetensors", ".parquet", ".log"}


def test_an_unserialisable_spec_is_refused_rather_than_stringified(
    store: ModelStore,
) -> None:
    record = ModelRecord(
        model_id=MODEL_B,
        spec={"opaque": object()},
        weights={},
        history=None,
        log=None,
        provenance={},
    )
    with pytest.raises(TypeError):
        store.put(record)
    assert not store.exists(MODEL_B)


# --------------------------------------------------------------------------
# Integrity and enumeration
# --------------------------------------------------------------------------


def test_verify_accepts_an_untouched_record(store: ModelStore) -> None:
    store.put(_model_record())
    assert store.verify(MODEL_A) == ()


def test_verify_detects_a_corrupted_checkpoint(store: ModelStore) -> None:
    """Negative control: the integrity check must be able to fail."""
    store.put(_model_record())
    target = store.path(MODEL_A) / "weights.safetensors"
    target.write_bytes(target.read_bytes() + b"corruption")

    problems = store.verify(MODEL_A)
    assert problems and "weights.safetensors" in problems[0]


def test_list_ids_returns_exactly_what_was_put(store: ModelStore) -> None:
    store.put(_model_record(MODEL_A))
    store.put(
        ModelRecord(
            model_id=MODEL_B,
            spec={},
            weights={},
            history=None,
            log=None,
            provenance={},
        )
    )
    assert sorted(store.list_ids()) == sorted([MODEL_A, MODEL_B])


def test_list_ids_ignores_directories_that_are_not_ids(store: ModelStore) -> None:
    store.put(_model_record())
    (store.records_root / "not-an-id").mkdir()
    assert list(store.list_ids()) == [MODEL_A]


def test_getting_an_absent_record_raises(store: ModelStore) -> None:
    with pytest.raises(RecordNotFoundError, match=MODEL_A):
        store.get(MODEL_A)


def test_delete_removes_a_record(store: ModelStore) -> None:
    store.put(_model_record())
    store.delete(MODEL_A)
    assert not store.exists(MODEL_A)
    assert list(store.list_ids()) == []


# --------------------------------------------------------------------------
# Measurements use the same machinery
# --------------------------------------------------------------------------


def test_a_measurement_round_trips(tmp_path: Path) -> None:
    measurements = MeasurementStore(tmp_path / "store")
    record = _measurement_record()
    measurements.put(record)
    loaded = measurements.get(MEASUREMENT_A)

    assert loaded.metrics == record.metrics
    assert loaded.spec == record.spec
    assert loaded.samples is not None
    pd.testing.assert_frame_equal(loaded.samples, record.samples)


def test_measurement_metrics_survive_as_numbers_not_strings(tmp_path: Path) -> None:
    measurements = MeasurementStore(tmp_path / "store")
    measurements.put(_measurement_record())
    raw = json.loads((measurements.path(MEASUREMENT_A) / "metrics.json").read_text())
    assert isinstance(raw["expected_cost"], float)


def test_models_and_measurements_do_not_share_a_namespace(tmp_path: Path) -> None:
    """Both are 16-hex ids; they must not be able to collide on disk."""
    root = tmp_path / "store"
    models, measurements = ModelStore(root), MeasurementStore(root)
    assert models.records_root != measurements.records_root


def _boom(*args: object, **kwargs: object) -> None:
    raise RuntimeError("simulated crash before publish")


# --------------------------------------------------------------------------
# Concurrency (Annex 02 §7.3): several workers write the store at once
# --------------------------------------------------------------------------


def _put_in_subprocess(root: str, epochs: int) -> str:
    """Run in a fresh process: put a record, report what happened."""
    store = ModelStore(Path(root))
    try:
        store.put(_model_record(epochs=epochs))
    except ContentConflictError:
        return "conflict"
    except Exception as error:  # noqa: BLE001 -- the point is to see ANY failure
        return f"{type(error).__name__}: {error}"
    return "ok"


def test_concurrent_writers_of_the_same_record_all_succeed(tmp_path: Path) -> None:
    """Identical content from several processes must reconcile, not collide.

    The window is between `exists()` returning False and `os.replace` landing:
    every process can see "absent" and then try to publish. Only one rename can
    win; the losers must reconcile against the published content rather than
    surfacing a filesystem error.
    """
    root = str(tmp_path / "store")
    with ProcessPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_put_in_subprocess, [root] * 8, [200] * 8))

    assert set(outcomes) == {"ok"}, outcomes
    assert ModelStore(Path(root)).verify(MODEL_A) == ()


def test_concurrent_writers_of_conflicting_content_do_not_corrupt(
    tmp_path: Path,
) -> None:
    """Differing content under one id is a defect either way; what must not
    happen is a half-published record."""
    root = str(tmp_path / "store")
    with ProcessPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(_put_in_subprocess, [root] * 4, [200, 400, 800, 1600]))

    assert set(outcomes) <= {"ok", "conflict"}, outcomes
    store = ModelStore(Path(root))
    assert store.exists(MODEL_A)
    assert store.verify(MODEL_A) == ()
