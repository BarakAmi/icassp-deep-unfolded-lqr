"""The anti-orphan gate: a record written before a payload existed is complete.

Annex 02 §2.2. `exists()` is what makes a torn write read as absent, and absent
means *recompute*. The live store holds 162 model records and 162 measurements
written before `training.parquet`, `evaluation.parquet` or either `.log` member
existed. If any payload ever joins a store's `required` tuple, every one of them
reads as absent and the next run retrains them: 3 h 06 m of training, discarded
for a diagnostic file.

**The member names here are spelled as literals, deliberately.** These tests
assert what is *on disk* in records this code did not write. A constant renamed
in `layout.py` must not be able to follow itself into the assertion and make it
agree with whatever the writer now does -- the same reason a golden array is not
recomputed by the code it checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mbl.store.content_store import (
    MeasurementID,
    MeasurementStore,
    ModelID,
    ModelStore,
)

#: Exactly the members a model record held before this change, and holds today
#: in all 162 stored records.
LEGACY_MODEL_MEMBERS = ("spec.json", "weights.safetensors", "provenance.json")

#: The same, for a measurement. `samples.parquet` is deliberately absent: it has
#: always been optional, and a record without it is the older shape still.
LEGACY_MEASUREMENT_MEMBERS = ("spec.json", "metrics.json")

#: Every member added by Annex 02 §2.2. None of them may ever be required.
PAYLOAD_MEMBERS = (
    "training.parquet",
    "synthesis.log",
    "evaluation.parquet",
    "evaluation.log",
)

MODEL = ModelID("0123456789abcdef")
MEASUREMENT = MeasurementID("aaaabbbbccccdddd")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_legacy(directory: Path, members: dict[str, bytes]) -> None:
    """Write a record directory the way the previous code wrote it.

    Not through `put`: `put` writes whatever *today's* code writes, which is the
    one thing this file must not depend on.
    """
    directory.mkdir(parents=True)
    for name, payload in members.items():
        (directory / name).write_bytes(payload)
    manifest = {name: _sha256(directory / name) for name in sorted(members)}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def legacy_model(tmp_path: Path) -> ModelStore:
    store = ModelStore(tmp_path / "store")
    weights = tmp_path / "weights.safetensors"
    save_file({"alpha": torch.tensor([0.25, 0.5], dtype=torch.float64)}, str(weights))
    _publish_legacy(
        store.path(MODEL),
        {
            "spec.json": b'{\n  "contender": {\n    "family": "unfolded"\n  }\n}\n',
            "weights.safetensors": weights.read_bytes(),
            "provenance.json": b'{\n  "stamp": "mbl-0.1.0/schema-1"\n}\n',
        },
    )
    return store


@pytest.fixture
def legacy_measurement(tmp_path: Path) -> MeasurementStore:
    store = MeasurementStore(tmp_path / "store")
    _publish_legacy(
        store.path(MEASUREMENT),
        {
            "spec.json": b'{\n  "model": "0123456789abcdef"\n}\n',
            "metrics.json": b'{\n  "eval_expected_cost": 2.106243\n}\n',
        },
    )
    return store


# --------------------------------------------------------------------------
# A record written yesterday is a record today
# --------------------------------------------------------------------------


def test_a_model_written_before_payloads_existed_is_complete(
    legacy_model: ModelStore,
) -> None:
    """The gate. `exists()` false here means 162 records read as absent, and the
    next run retrains every one of them."""
    assert legacy_model.exists(MODEL)
    assert list(legacy_model.list_ids()) == [MODEL]
    assert legacy_model.verify(MODEL) == ()


def test_a_measurement_written_before_payloads_existed_is_complete(
    legacy_measurement: MeasurementStore,
) -> None:
    assert legacy_measurement.exists(MEASUREMENT)
    assert list(legacy_measurement.list_ids()) == [MEASUREMENT]
    assert legacy_measurement.verify(MEASUREMENT) == ()


def test_no_payload_member_is_required(
    legacy_model: ModelStore, legacy_measurement: MeasurementStore
) -> None:
    """Stated on the tuples as well as on the behaviour above.

    The behavioural tests fail if a payload becomes required; this one says
    *which* line did it, in the one place where "make the test pass" is the
    wrong repair. Adding a member here is not a rename -- it is the store
    declaring that every record it already holds is incomplete.
    """
    for member in PAYLOAD_MEMBERS:
        assert member not in ModelStore.required, (
            f"{member} joined ModelStore.required: every record published "
            "before it existed now reads as absent, and absent means retrain"
        )
        assert member not in MeasurementStore.required
    assert set(ModelStore.required) == {*LEGACY_MODEL_MEMBERS, "manifest.json"}
    assert set(MeasurementStore.required) == {
        *LEGACY_MEASUREMENT_MEMBERS,
        "manifest.json",
    }


# --------------------------------------------------------------------------
# ... and it loads, with its payloads absent rather than empty
# --------------------------------------------------------------------------


def test_a_legacy_model_loads_with_every_payload_absent(
    legacy_model: ModelStore,
) -> None:
    record = legacy_model.get(MODEL)
    assert record.history is None
    assert record.log is None
    assert record.spec == {"contender": {"family": "unfolded"}}
    assert record.provenance == {"stamp": "mbl-0.1.0/schema-1"}


def test_a_legacy_measurement_loads_with_every_payload_absent(
    legacy_measurement: MeasurementStore,
) -> None:
    record = legacy_measurement.get(MEASUREMENT)
    assert record.samples is None
    assert record.trace is None
    assert record.log is None
    assert record.metrics == {"eval_expected_cost": 2.106243}
