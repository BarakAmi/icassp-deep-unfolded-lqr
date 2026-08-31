"""A record that is not finite is not a result; it is the absence of one.

Training can diverge for reasons that are *correct*: an unstable plant under a
box too tight to stabilise it will blow up, and control theory says so. What
must not happen is that the blow-up is filed as a result. Measured before this
guard existed, on a deliberately unstable plant (spectral radius 1.6, u_max
0.05): `mbl run` completed, published 7 models and 7 measurements, and stored
expected costs of **2.07e+19** -- and with a longer horizon or float32 those
same runs reach `nan`, which every later stage silently drops rather than
reports (`present/axis_scaling.py` filters non-finite points out of a figure).

`ContentStore.put` is the one seam both record kinds pass through and it already
refuses content the no-pickle law cannot serialise. Not-finite joins it there,
for the same reason: **one place, before anything is observable**, so no future
producer can forget it.

What this deliberately does NOT do is judge magnitude. A cost of 2.07e+19 is
finite, and whether it is *plausible* is a scientific question whose answer
differs per study -- which is what the declared `[[gates]]` are for, registered
before the numbers exist. Storage refuses what cannot be a number; a gate
refuses what should not be believed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from mbl.store.content_store import (
    MeasurementRecord,
    MeasurementStore,
    ModelRecord,
    ModelStore,
    NonFiniteRecordError,
)
from mbl.store.ids import MeasurementID, ModelID

MODEL = ModelID("0123456789abcdef")
MEASUREMENT = MeasurementID("aaaabbbbccccdddd")


def _model(**overrides) -> ModelRecord:
    fields: dict = {
        "model_id": MODEL,
        "spec": {"contender": {"family": "unfolded"}},
        "weights": {"alpha": torch.tensor([0.25, 0.5], dtype=torch.float64)},
        "history": None,
        "log": None,
        "provenance": {"stamp": "mbl-0.1.0/schema-1"},
    }
    return ModelRecord(**{**fields, **overrides})


def _measurement(**overrides) -> MeasurementRecord:
    fields: dict = {
        "measurement_id": MEASUREMENT,
        "spec": {"model": str(MODEL)},
        "metrics": {"expected_cost": 3.25},
        "samples": None,
        "trace": None,
        "log": None,
    }
    return MeasurementRecord(**{**fields, **overrides})


@pytest.fixture
def models(tmp_path: Path) -> ModelStore:
    return ModelStore(tmp_path / "store")


@pytest.fixture
def measurements(tmp_path: Path) -> MeasurementStore:
    return MeasurementStore(tmp_path / "store")


# -- the refusal -------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_metric_is_refused(
    measurements: MeasurementStore, bad: float
) -> None:
    with pytest.raises(NonFiniteRecordError) as error:
        measurements.put(_measurement(metrics={"expected_cost": bad}))

    message = str(error.value)
    assert "expected_cost" in message, "the refusal must name the offending key"
    assert str(MEASUREMENT) in message, "and the record it would have been filed as"
    assert not measurements.exists(str(MEASUREMENT)), "it was published anyway"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_weight_is_refused(models: ModelStore, bad: float) -> None:
    with pytest.raises(NonFiniteRecordError) as error:
        models.put(
            _model(weights={"alpha": torch.tensor([0.25, bad], dtype=torch.float64)})
        )

    assert "alpha" in str(error.value)
    assert not models.exists(str(MODEL)), "a diverged model reached the store"


def test_a_non_finite_sample_is_refused(measurements: MeasurementStore) -> None:
    """Per-trajectory costs are what an analysis aggregates, so a `nan` here
    reaches a published table by way of a mean."""
    with pytest.raises(NonFiniteRecordError, match="cost"):
        measurements.put(
            _measurement(
                samples=pd.DataFrame({"trajectory": [0, 1], "cost": [3.1, math.nan]})
            )
        )


def test_a_non_finite_training_history_is_refused(models: ModelStore) -> None:
    """The diagnostic payload is the one that would *show* the divergence, so
    admitting it while refusing the weights would store the evidence and drop
    the finding."""
    with pytest.raises(NonFiniteRecordError, match="loss"):
        models.put(
            _model(history=pd.DataFrame({"epoch": [0, 1], "loss": [1.0, math.nan]}))
        )


# -- what it must NOT refuse -------------------------------------------------


def test_a_finite_but_enormous_cost_is_stored(
    measurements: MeasurementStore,
) -> None:
    """Anti-vacuity, and the boundary this guard deliberately declines to draw.

    2.07e+19 is what the unstable plant actually produced. It is finite, so it
    is a number, and whether it is *plausible* is a study's declared gate to
    decide -- not the store's. A guard that rejected "large" would be choosing
    a threshold nobody registered, which is exactly what this project refuses
    to do with `min_relative_gap`.
    """
    measurements.put(_measurement(metrics={"expected_cost": 2.07e19}))

    assert measurements.exists(str(MEASUREMENT))
    assert measurements.get(str(MEASUREMENT)).metrics["expected_cost"] == 2.07e19


def test_a_record_with_no_payloads_is_unaffected(models: ModelStore) -> None:
    """An analytic contender carries no weights and no history: the guard must
    have nothing to say about it."""
    models.put(_model(weights={}))

    assert models.exists(str(MODEL))


def test_an_integer_weight_is_not_mistaken_for_non_finite(
    models: ModelStore,
) -> None:
    """`torch.isfinite` raises on some integer dtypes rather than returning
    `True`, so the guard must narrow before it tests."""
    models.put(_model(weights={"depth": torch.tensor(10, dtype=torch.int64)}))

    assert models.exists(str(MODEL))


def test_a_non_numeric_column_is_not_mistaken_for_non_finite(
    measurements: MeasurementStore,
) -> None:
    """Samples carry string columns (a contender label, a metric name); asking
    `isfinite` of them would raise and turn a good record into a refusal."""
    measurements.put(
        _measurement(
            samples=pd.DataFrame({"contender": ["cocp", "riccati"], "cost": [3.1, 3.4]})
        )
    )

    assert measurements.exists(str(MEASUREMENT))


def test_a_sparse_specification_buffer_is_not_a_crash(models: ModelStore) -> None:
    """Found by the full suite, not by this file: `torch.isfinite` raises
    ``unsupported tensor layout: SparseCsr`` rather than answering, and every
    unfolded family carries sparse specification buffers (`_layer.P`,
    `_layer.q`, `_layer.A`) that `dense_state_dict` already skips on write.

    A guard that crashed on them would have made the store unusable for the
    project's flagship contender.
    """
    # COO, not CSR. Constructing a CSR tensor emits torch's "beta state"
    # `UserWarning` ONCE PER PROCESS, so a `pytest.warns` around it passes
    # alone and fails whenever an earlier test in the run already built one --
    # which is how this test first failed in the full suite and nowhere else.
    # Both layouts are non-strided, which is the only property under test.
    sparse = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64).to_sparse()

    models.put(_model(weights={"alpha": torch.zeros(2), "_layer.P": sparse}))

    # Asserted by its EFFECT rather than by the log line: whether a record is
    # captured depends on which handlers the rest of the suite has attached,
    # and a test that passes alone and fails in company is worse than no test.
    stored = models.get(str(MODEL)).weights
    assert "alpha" in stored
    assert "_layer.P" not in stored, "the sparse buffer was persisted after all"
