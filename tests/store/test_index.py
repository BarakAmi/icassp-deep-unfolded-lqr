"""Acceptance tests for the store index (Annex 02 §4).

Written before the implementation. Three properties carry weight:

* **Lookup is O(1).** The Stage 1 gate requires sub-millisecond resolution at
  10,000 entries, against today's linear scan of 3,225 directories with a JSON
  parse each.
* **`is_shifted` cannot be silently wrong.** It is materialised from the model's
  training problem, so a measurement whose model is unknown must be *refused*,
  never stored as nominal -- reporting a shifted measurement as in-distribution
  is exactly the error that invalidates an out-of-distribution claim.
* **The index is derived, except the queue.** It must rebuild from the trees at
  any time, and must not destroy the one table that is genuine state.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from mbl.store.content_store import (
    MeasurementRecord,
    MeasurementStore,
    ModelRecord,
    ModelStore,
)
from mbl.store.ids import MeasurementID, ModelID, ProblemID
from mbl.store.index import (
    AmbiguousPrefixError,
    MeasurementRow,
    ModelRow,
    StoreIndex,
    UnknownModelError,
    UnknownPrefixError,
)

PROBLEM = ProblemID("1111111111111111")
SHIFTED_PROBLEM = ProblemID("2222222222222222")
MODEL = ModelID("aaaa0000aaaa0000")
OTHER_MODEL = ModelID("bbbb1111bbbb1111")


def _model_row(
    model_id: ModelID = MODEL, *, family: str = "unfolded", seed: int = 0
) -> ModelRow:
    return ModelRow(
        model_id=model_id,
        semantic_name=f"boxlqr-n7m3/{family}/seed{seed}",
        problem_id=PROBLEM,
        family=family,
        contender_id="unfolded_alpha_p",
        seed=seed,
        state_dim=7,
        control_dim=3,
        horizon=100,
        created_utc="2026-07-29T12:00:00Z",
        stamp="mbl-0.1.0/schema-1",
        wall_time_s=12.5,
        peak_vram_mb=1024.0,
        spec_json='{"contender": {"family": "unfolded"}}',
    )


def _measurement_row(
    measurement_id: MeasurementID,
    *,
    model_id: ModelID = MODEL,
    eval_problem: ProblemID = PROBLEM,
) -> MeasurementRow:
    return MeasurementRow(
        measurement_id=measurement_id,
        model_id=model_id,
        eval_problem_id=eval_problem,
        created_utc="2026-07-29T12:05:00Z",
        spec_json="{}",
        metrics={"expected_cost": 3.25},
    )


@pytest.fixture
def index(tmp_path: Path) -> StoreIndex:
    return StoreIndex(tmp_path / "store")


# --------------------------------------------------------------------------
# Durability configuration (Annex 02 §7.3)
# --------------------------------------------------------------------------


def test_the_database_runs_in_wal_mode(index: StoreIndex) -> None:
    """WAL is what lets readers proceed while a writer holds the database, and
    is a precondition for the parallel runner."""
    with index.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_a_busy_timeout_is_configured(index: StoreIndex) -> None:
    with index.connect() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0


# --------------------------------------------------------------------------
# Round trip and idempotence
# --------------------------------------------------------------------------


def test_a_model_round_trips(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    loaded = index.model(MODEL)
    assert loaded == _model_row()


def test_upserting_the_same_model_twice_leaves_one_row(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    index.upsert_model(_model_row())
    assert len(list(index.models())) == 1


def test_upsert_updates_mutable_provenance_without_duplicating(
    index: StoreIndex,
) -> None:
    index.upsert_model(_model_row())
    revised = ModelRow(**{**_model_row().__dict__, "wall_time_s": 99.0})
    index.upsert_model(revised)
    assert index.model(MODEL).wall_time_s == 99.0
    assert len(list(index.models())) == 1


def test_a_measurement_and_its_metrics_round_trip(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    index.upsert_measurement(_measurement_row(MeasurementID("cccc2222cccc2222")))
    rows = list(index.measurements(model_id=MODEL))
    assert len(rows) == 1
    assert rows[0].metrics == {"expected_cost": 3.25}


# --------------------------------------------------------------------------
# is_shifted must never be silently wrong
# --------------------------------------------------------------------------


def test_a_nominal_measurement_is_not_marked_shifted(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    mid = MeasurementID("cccc2222cccc2222")
    index.upsert_measurement(_measurement_row(mid, eval_problem=PROBLEM))
    assert index.measurements(model_id=MODEL)[0].is_shifted is False


def test_evaluating_on_another_problem_is_marked_shifted(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    mid = MeasurementID("dddd3333dddd3333")
    index.upsert_measurement(_measurement_row(mid, eval_problem=SHIFTED_PROBLEM))
    assert index.measurements(model_id=MODEL)[0].is_shifted is True


def test_a_measurement_whose_model_is_unknown_is_refused(index: StoreIndex) -> None:
    """The gap this closes: `is_shifted` is derived from the model's training
    problem, so with no model row it cannot be computed. Defaulting it to 0
    would report a shifted measurement as in-distribution -- the exact error
    that invalidates an out-of-distribution claim. Refuse instead."""
    with pytest.raises(UnknownModelError, match=MODEL):
        index.upsert_measurement(_measurement_row(MeasurementID("eeee4444eeee4444")))


def test_shifted_filter_selects_only_shifted_measurements(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    index.upsert_measurement(
        _measurement_row(MeasurementID("cccc2222cccc2222"), eval_problem=PROBLEM)
    )
    index.upsert_measurement(
        _measurement_row(
            MeasurementID("dddd3333dddd3333"), eval_problem=SHIFTED_PROBLEM
        )
    )
    shifted = index.measurements(model_id=MODEL, shifted=True)
    assert [r.measurement_id for r in shifted] == ["dddd3333dddd3333"]


# --------------------------------------------------------------------------
# Filters and prefix resolution (Annex 02 §3)
# --------------------------------------------------------------------------


def test_models_can_be_filtered(index: StoreIndex) -> None:
    index.upsert_model(_model_row(MODEL, family="unfolded", seed=0))
    index.upsert_model(_model_row(OTHER_MODEL, family="cocp", seed=1))

    assert [m.model_id for m in index.models(family="unfolded")] == [MODEL]
    assert [m.model_id for m in index.models(seed=1)] == [OTHER_MODEL]
    assert len(list(index.models(problem_id=PROBLEM))) == 2


def test_a_unique_prefix_resolves_to_its_model(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    assert index.resolve_model_prefix("aaaa00") == MODEL


def test_the_full_id_resolves(index: StoreIndex) -> None:
    index.upsert_model(_model_row())
    assert index.resolve_model_prefix(MODEL) == MODEL


def test_an_ambiguous_prefix_is_refused_rather_than_guessed(index: StoreIndex) -> None:
    """Silently picking one of several matches would attach a command to the
    wrong model."""
    index.upsert_model(_model_row(ModelID("aaaa0000aaaa0000")))
    index.upsert_model(_model_row(ModelID("aaaa0000ffffffff")))
    with pytest.raises(AmbiguousPrefixError, match="2 models"):
        index.resolve_model_prefix("aaaa0000")


def test_an_unknown_prefix_is_refused(index: StoreIndex) -> None:
    with pytest.raises(UnknownPrefixError):
        index.resolve_model_prefix("deadbe")


# --------------------------------------------------------------------------
# The Stage 1 performance gate
# --------------------------------------------------------------------------


def test_lookup_is_sub_millisecond_at_ten_thousand_entries(index: StoreIndex) -> None:
    """The Stage 1 gate. Compared against today's linear scan of 3,225
    directories, each with a JSON parse."""
    ids = [ModelID(f"{i:016x}") for i in range(10_000)]
    index.upsert_models(_model_row(i) for i in ids)
    assert len(list(index.models())) == 10_000

    probes = ids[::997]
    timings = []
    for model_id in probes:
        start = time.perf_counter()
        index.model(model_id)
        timings.append(time.perf_counter() - start)

    median_ms = statistics.median(timings) * 1000
    assert median_ms < 1.0, f"median lookup {median_ms:.3f} ms at 10,000 entries"


# --------------------------------------------------------------------------
# Derived, except the queue
# --------------------------------------------------------------------------


def test_reindex_rebuilds_every_row_from_the_trees(tmp_path: Path) -> None:
    root = tmp_path / "store"
    models, measurements = ModelStore(root), MeasurementStore(root)
    models.put(
        ModelRecord(
            model_id=MODEL,
            spec={
                "problem_id": PROBLEM,
                "family": "unfolded",
                "contender_id": "u",
                "seed": 0,
            },
            weights={},
            history=pd.DataFrame({"epoch": [0], "loss": [1.0]}),
            log=None,
            provenance={
                "created_utc": "2026-07-29T12:00:00Z",
                "stamp": "s",
                "wall_time_s": 1.0,
            },
        )
    )
    measurements.put(
        MeasurementRecord(
            measurement_id=MeasurementID("cccc2222cccc2222"),
            spec={"model": MODEL, "eval_problem": SHIFTED_PROBLEM},
            metrics={"expected_cost": 2.0},
            samples=None,
            trace=None,
            log=None,
        )
    )

    index = StoreIndex(root)
    index.reindex(models, measurements)

    assert [m.model_id for m in index.models()] == [MODEL]
    rebuilt = index.measurements(model_id=MODEL)
    assert len(rebuilt) == 1
    assert rebuilt[0].is_shifted is True
    assert rebuilt[0].metrics == {"expected_cost": 2.0}


def test_reindex_orders_models_before_measurements(tmp_path: Path) -> None:
    """Without that ordering, `is_shifted` cannot be derived and every
    measurement would be refused or wrong."""
    root = tmp_path / "store"
    models, measurements = ModelStore(root), MeasurementStore(root)
    measurements.put(
        MeasurementRecord(
            measurement_id=MeasurementID("cccc2222cccc2222"),
            spec={"model": MODEL, "eval_problem": PROBLEM},
            metrics={},
            samples=None,
            trace=None,
            log=None,
        )
    )
    models.put(
        ModelRecord(
            model_id=MODEL,
            spec={"problem_id": PROBLEM, "family": "u", "contender_id": "u", "seed": 0},
            weights={},
            history=None,
            log=None,
            provenance={"created_utc": "t", "stamp": "s"},
        )
    )
    index = StoreIndex(root)
    index.reindex(models, measurements)  # must not raise
    assert index.measurements(model_id=MODEL)[0].is_shifted is False


def test_reindex_preserves_the_queue(index: StoreIndex, tmp_path: Path) -> None:
    """The queue is the one table that is genuine state, not a projection of the
    trees (Annex 02 §4). Rebuilding must not discard a two-day batch."""
    index.enqueue(entry_id="e1", study="box_lqr/depth", tier="publication")
    index.upsert_model(_model_row())

    index.reindex(ModelStore(tmp_path / "store"), MeasurementStore(tmp_path / "store"))

    assert [e["study"] for e in index.queue_entries()] == ["box_lqr/depth"]
    assert (
        list(index.models()) == []
    )  # the derived rows did rebuild (from an empty tree)


def test_reindex_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "store"
    models, measurements = ModelStore(root), MeasurementStore(root)
    models.put(
        ModelRecord(
            model_id=MODEL,
            spec={"problem_id": PROBLEM, "family": "u", "contender_id": "u", "seed": 0},
            weights={},
            history=None,
            log=None,
            provenance={"created_utc": "t", "stamp": "s"},
        )
    )
    index = StoreIndex(root)
    index.reindex(models, measurements)
    index.reindex(models, measurements)
    assert len(list(index.models())) == 1


# --------------------------------------------------------------------------
# Concurrency (the lesson from the content-store slice: test it, do not assume)
# --------------------------------------------------------------------------


def _upsert_in_subprocess(root: str, ordinal: int) -> str:
    index = StoreIndex(Path(root))
    try:
        for i in range(25):
            index.upsert_model(_model_row(ModelID(f"{ordinal:08x}{i:08x}")))
    except sqlite3.Error as error:
        return f"sqlite3.{type(error).__name__}: {error}"
    return "ok"


def test_eight_concurrent_writers_do_not_hit_database_is_locked(tmp_path: Path) -> None:
    """Without WAL and a busy timeout this is where a parallel run dies at 2am."""
    root = tmp_path / "store"
    StoreIndex(root)  # create the schema once, before the fan-out
    with ProcessPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_upsert_in_subprocess, [str(root)] * 8, range(8)))

    assert set(outcomes) == {"ok"}, outcomes
    assert len(list(StoreIndex(root).models())) == 200
