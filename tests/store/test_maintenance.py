"""Acceptance tests for store maintenance (Annex 02 §5, `mbl store …`).

Written before the implementation. Two properties carry the weight:

* **`verify` must report what it could not check**, rather than counting an
  unverifiable record as a passing one. A maintenance command whose clean bill
  of health silently excludes most of the store is worse than no command.
* **`gc` must refuse to run against an empty study table.** "No study
  references anything" and "the study table has not been populated yet" are the
  same query result and opposite intentions, and today -- with studies still
  unbuilt -- it is always the second. Proceeding would delete the whole store.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

from mbl.store.content_store import (
    MANIFEST,
    MeasurementRecord,
    MeasurementStore,
    ModelRecord,
    ModelStore,
)
from mbl.store.ids import MeasurementID, ModelID, measurement_id
from mbl.store.index import MeasurementRow, ModelRow, StoreIndex
from mbl.store.maintenance import (
    GarbagePolicy,
    NoLiveStudiesError,
    apply_gc,
    directory_size,
    plan_gc,
    problem_sizes,
    store_stat,
    verify_store,
)

PROBLEM_A = "0123456789abcdef"
PROBLEM_B = "fedcba9876543210"
PROTOCOL = {"batches": 8, "metrics": ["expected_cost"]}


def _model_id(n: int) -> ModelID:
    return ModelID(f"{n:016x}")


def _measurement_id(model: ModelID, problem: str) -> MeasurementID:
    return measurement_id(model, problem, PROTOCOL)


class _Store:
    """A store on disk plus its index, kept in step."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.models = ModelStore(root)
        self.measurements = MeasurementStore(root)
        self.index = StoreIndex(root)

    def add_model(
        self, model: ModelID, problem: str = PROBLEM_A, seed: int = 0
    ) -> ModelID:
        self.models.put(
            ModelRecord(
                model_id=model,
                spec={
                    "problem_id": problem,
                    "contender_id": "unfolded-aP",
                    "seed": seed,
                },
                weights={"alpha": torch.tensor([0.5, 0.25], dtype=torch.float64)},
                history=pd.DataFrame({"epoch": [0, 1], "loss": [1.0, 0.5]}),
                log=None,
                provenance={"created_utc": "2026-05-14T08:00:00", "stamp": "mbl-0.1.0"},
            )
        )
        self.index.upsert_model(
            ModelRow(
                model_id=model,
                semantic_name=f"boxlqr/unfolded-aP-J10/adam-seed{seed}#{model[:6]}",
                problem_id=problem,
                family="unfolded",
                contender_id="unfolded-aP",
                seed=seed,
                created_utc="2026-05-14T08:00:00",
            )
        )
        return model

    def add_measurement(
        self,
        model: ModelID,
        problem: str = PROBLEM_A,
        *,
        record_id: MeasurementID | None = None,
        protocol: dict[str, object] | None = None,
    ) -> MeasurementID:
        spec: dict[str, object] = {"model": str(model), "eval_problem": problem}
        if protocol is not None:
            spec["eval_protocol"] = protocol
        identifier = record_id or _measurement_id(model, problem)
        self.measurements.put(
            MeasurementRecord(
                measurement_id=identifier,
                spec=spec,
                metrics={"expected_cost": 3.25},
                samples=None,
                trace=None,
                log=None,
            )
        )
        self.index.upsert_measurement(
            MeasurementRow(
                measurement_id=identifier,
                model_id=model,
                eval_problem_id=problem,
                created_utc="2026-06-02T08:00:00",
                metrics={"expected_cost": 3.25},
            )
        )
        return identifier

    def enrol(self, study: str, kind: str, member: str) -> None:
        with self.index.connect() as conn:
            conn.execute(
                "INSERT INTO study_members VALUES (?,?,?)", (study, kind, member)
            )


@pytest.fixture
def store(tmp_path: Path) -> _Store:
    return _Store(tmp_path / "store")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def test_a_clean_store_verifies(store: _Store) -> None:
    model = store.add_model(_model_id(1))
    store.add_measurement(model, protocol=PROTOCOL)
    report = verify_store(store.root, store.index)
    assert report.ok, report.lines()


def test_a_corrupted_member_is_detected(store: _Store) -> None:
    """The manifest exists to make silent bit rot loud."""
    model = store.add_model(_model_id(1))
    weights = store.models.path(model) / "weights.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 0xFF
    weights.write_bytes(bytes(payload))

    report = verify_store(store.root, store.index)
    assert not report.ok
    assert any("weights.safetensors" in line for line in report.corrupt)


def test_a_record_missing_from_the_index_is_reported(store: _Store) -> None:
    """A model on disk that no index row mentions is invisible to every query,
    so the store holds work nothing can find."""
    store.add_model(_model_id(1))
    store.models.put(
        ModelRecord(
            model_id=_model_id(2),
            spec={"problem_id": PROBLEM_A},
            weights={},
            history=None,
            log=None,
            provenance={},
        )
    )
    report = verify_store(store.root, store.index)
    assert not report.ok
    assert any(_model_id(2) in line for line in report.unindexed)


def test_an_index_row_with_no_record_is_reported(store: _Store) -> None:
    """The opposite orphan: a query returns a model whose checkpoint is gone."""
    store.add_model(_model_id(1))
    store.index.upsert_model(
        ModelRow(_model_id(9), "n#a", PROBLEM_A, "unfolded", "unfolded-aP", 0)
    )
    report = verify_store(store.root, store.index)
    assert not report.ok
    assert any(_model_id(9) in line for line in report.missing)


def test_a_measurement_filed_under_the_wrong_identifier_is_detected(
    store: _Store,
) -> None:
    """Content addressing means the directory name is a claim about the content.
    Recomputing the identifier from the specification is the only check that can
    falsify that claim."""
    model = store.add_model(_model_id(1))
    store.add_measurement(model, record_id=MeasurementID("dead" * 4), protocol=PROTOCOL)
    report = verify_store(store.root, store.index)
    assert not report.ok
    assert any("deaddeaddeaddead" in line for line in report.inconsistent)


def test_a_specification_that_cannot_pin_an_identifier_is_counted_not_passed(
    store: _Store,
) -> None:
    """A measurement whose spec omits its evaluation protocol cannot have its
    identifier recomputed. Counting it as verified would let `verify` report a
    clean store while checking almost none of it."""
    model = store.add_model(_model_id(1))
    store.add_measurement(model)  # no eval_protocol
    report = verify_store(store.root, store.index)
    assert report.unverifiable >= 1
    assert not report.inconsistent
    assert "unverifiable" in "\n".join(report.lines())


def test_every_model_is_unverifiable_until_the_grammar_defines_its_spec(
    store: _Store,
) -> None:
    """Model identifiers derive from five specification sub-trees the grammar
    does not yet emit. That is a known gap, and it is *reported* rather than
    quietly treated as success."""
    store.add_model(_model_id(1))
    report = verify_store(store.root, store.index)
    assert report.ok
    assert report.unverifiable == 1


# --------------------------------------------------------------------------
# gc -- the empty-study-table guard
# --------------------------------------------------------------------------


def test_gc_refuses_when_no_study_references_anything(store: _Store) -> None:
    """The guard that stops `mbl store gc` deleting the whole store today.

    Studies arrive with Tier 5, so `study_members` is empty on every store that
    currently exists; without this, the set of referenced measurements is empty
    and every measurement is collectable.
    """
    model = store.add_model(_model_id(1))
    store.add_measurement(model)
    with pytest.raises(NoLiveStudiesError):
        plan_gc(store.root, store.index)


def test_gc_refuses_when_the_named_studies_do_not_exist(store: _Store) -> None:
    """A typo in `--keep-studies` must not read as 'nothing is live'."""
    model = store.add_model(_model_id(1))
    store.add_measurement(model)
    store.enrol("study-a", "measurement", _measurement_id(model, PROBLEM_A))
    with pytest.raises(NoLiveStudiesError):
        plan_gc(store.root, store.index, GarbagePolicy(keep_studies=("study-typo",)))


def test_partial_collection_needs_no_studies(store: _Store) -> None:
    """Resume artifacts belong to no study, so collecting them is always safe
    -- and must stay available on a store that has no studies at all."""
    model = store.add_model(_model_id(1))
    (store.models.path(model) / ".partial").mkdir()
    plan = plan_gc(
        store.root, store.index, GarbagePolicy(partials=True, measurements=False)
    )
    assert [p.name for p in plan.partials] == [".partial"]


# --------------------------------------------------------------------------
# gc -- what it collects
# --------------------------------------------------------------------------


def test_an_unreferenced_measurement_is_collected_and_a_referenced_one_is_not(
    store: _Store,
) -> None:
    model = store.add_model(_model_id(1))
    kept = store.add_measurement(model, PROBLEM_A)
    dropped = store.add_measurement(model, PROBLEM_B)
    store.enrol("study-a", "measurement", kept)

    plan = plan_gc(store.root, store.index)
    assert plan.measurements == (dropped,)


def test_models_are_never_collected_without_the_explicit_flag(store: _Store) -> None:
    """The asymmetry the annex insists on: a measurement is cheap to recompute,
    a model is the one expensive object."""
    store.add_model(_model_id(1))
    store.enrol("study-a", "measurement", "aaaabbbbccccdddd")
    assert plan_gc(store.root, store.index).models == ()
    assert plan_gc(
        store.root, store.index, GarbagePolicy(include_models=True)
    ).models == (_model_id(1),)


def test_the_model_behind_a_kept_measurement_survives_model_collection(
    store: _Store,
) -> None:
    """A study may reference a measurement without naming its model. Collecting
    that model would leave a measurement of nothing."""
    referenced = store.add_model(_model_id(1))
    orphan = store.add_model(_model_id(2))
    kept = store.add_measurement(referenced, PROBLEM_A)
    store.enrol("study-a", "measurement", kept)

    plan = plan_gc(store.root, store.index, GarbagePolicy(include_models=True))
    assert plan.models == (orphan,)


def test_keep_studies_narrows_which_references_count(store: _Store) -> None:
    model = store.add_model(_model_id(1))
    for_a = store.add_measurement(model, PROBLEM_A)
    for_b = store.add_measurement(model, PROBLEM_B)
    store.enrol("study-a", "measurement", for_a)
    store.enrol("study-b", "measurement", for_b)

    assert plan_gc(store.root, store.index).measurements == ()
    assert plan_gc(
        store.root, store.index, GarbagePolicy(keep_studies=("study-a",))
    ).measurements == (for_b,)


def test_only_partials_older_than_the_cutoff_are_collected(store: _Store) -> None:
    fresh = store.add_model(_model_id(1))
    stale = store.add_model(_model_id(2))
    for model in (fresh, stale):
        (store.models.path(model) / ".partial").mkdir()
    old = time.time() - 30 * 86_400
    os.utime(store.models.path(stale) / ".partial", (old, old))

    plan = plan_gc(
        store.root,
        store.index,
        GarbagePolicy(partials=True, measurements=False, older_than_seconds=7 * 86_400),
    )
    assert [p.parent.name for p in plan.partials] == [stale]


def test_a_plan_reports_the_bytes_it_would_free(store: _Store) -> None:
    model = store.add_model(_model_id(1))
    dropped = store.add_measurement(model, PROBLEM_B)
    store.enrol("study-a", "measurement", "aaaabbbbccccdddd")
    plan = plan_gc(store.root, store.index)
    assert plan.bytes_freed == directory_size(store.measurements.path(dropped))
    assert plan.bytes_freed > 0


# --------------------------------------------------------------------------
# gc -- applying it
# --------------------------------------------------------------------------


def test_applying_a_plan_removes_the_record_from_disk_and_from_the_index(
    store: _Store,
) -> None:
    """Deleting one but not the other is the failure that produces exactly the
    orphans `verify` hunts for."""
    model = store.add_model(_model_id(1))
    dropped = store.add_measurement(model, PROBLEM_B)
    store.enrol("study-a", "measurement", "aaaabbbbccccdddd")

    apply_gc(store.root, store.index, plan_gc(store.root, store.index))

    assert not store.measurements.exists(dropped)
    assert store.index.measurements() == []
    with store.index.connect() as conn:
        assert conn.execute("SELECT count(*) FROM metrics").fetchone()[0] == 0


def test_collecting_a_model_first_collects_its_measurements(store: _Store) -> None:
    """Ordering is load-bearing: the index's foreign key forbids a measurement
    row pointing at a deleted model, so the reverse order fails outright."""
    model = store.add_model(_model_id(1))
    store.add_measurement(model, PROBLEM_A)
    store.enrol("study-a", "model", "aaaabbbbccccdddd")

    apply_gc(
        store.root,
        store.index,
        plan_gc(store.root, store.index, GarbagePolicy(include_models=True)),
    )
    assert store.index.models() == []
    assert store.index.measurements() == []
    assert not store.models.exists(model)


def test_applying_a_plan_leaves_a_verifiable_store(store: _Store) -> None:
    """End-to-end closure: whatever gc does, `verify` must still be clean
    afterwards -- no orphan in either direction."""
    keep = store.add_model(_model_id(1))
    drop = store.add_model(_model_id(2))
    kept = store.add_measurement(keep, PROBLEM_A, protocol=PROTOCOL)
    store.add_measurement(drop, PROBLEM_B, protocol=PROTOCOL)
    store.enrol("study-a", "measurement", kept)

    apply_gc(
        store.root,
        store.index,
        plan_gc(store.root, store.index, GarbagePolicy(include_models=True)),
    )
    report = verify_store(store.root, store.index)
    assert report.ok, report.lines()


def test_applying_a_partial_plan_removes_only_the_resume_artifact(
    store: _Store,
) -> None:
    model = store.add_model(_model_id(1))
    (store.models.path(model) / ".partial").mkdir()
    apply_gc(
        store.root,
        store.index,
        plan_gc(
            store.root, store.index, GarbagePolicy(partials=True, measurements=False)
        ),
    )
    assert not (store.models.path(model) / ".partial").exists()
    assert store.models.exists(model)


# --------------------------------------------------------------------------
# stat
# --------------------------------------------------------------------------


def test_stat_counts_and_sizes_each_class(store: _Store) -> None:
    model = store.add_model(_model_id(1))
    store.add_measurement(model, PROBLEM_A)
    store.add_measurement(model, PROBLEM_B)

    stat = store_stat(store.root, store.index)
    by_kind = {c.kind: c for c in stat.classes}
    assert by_kind["models"].count == 1
    assert by_kind["measurements"].count == 2
    assert by_kind["models"].size_bytes == directory_size(store.models.records_root)
    assert stat.total_bytes >= sum(c.size_bytes for c in stat.classes if c.count)


def test_stat_buckets_growth_by_month(store: _Store) -> None:
    model = store.add_model(_model_id(1))
    store.add_measurement(model, PROBLEM_A)
    stat = store_stat(store.root, store.index)
    assert [(b.month, b.models, b.measurements) for b in stat.growth] == [
        ("2026-05", 1, 0),
        ("2026-06", 0, 1),
    ]


def test_the_index_is_reported_as_its_own_class(store: _Store) -> None:
    """`index.sqlite` is derived, so knowing its share of the store is what says
    whether a `reindex` would reclaim anything."""
    store.add_model(_model_id(1))
    kinds = [c.kind for c in store_stat(store.root, store.index).classes]
    assert "index" in kinds


# --------------------------------------------------------------------------
# per-problem sizes, for `mbl models tree`
# --------------------------------------------------------------------------


def test_a_problems_size_includes_the_measurements_of_its_models(
    store: _Store,
) -> None:
    """A measurement is filed under no problem of its own; it costs whatever
    problem trained the model it evaluates. Omitting them would report the
    store as tens of times smaller than it is."""
    model = store.add_model(_model_id(1), PROBLEM_A)
    measurement = store.add_measurement(model, PROBLEM_B)

    sizes = problem_sizes(store.root, store.index)
    assert sizes[PROBLEM_A] == directory_size(
        store.models.path(model)
    ) + directory_size(store.measurements.path(measurement))
    assert PROBLEM_B not in sizes


def test_two_models_of_one_problem_have_their_sizes_summed(store: _Store) -> None:
    first = store.add_model(_model_id(1), PROBLEM_A, seed=0)
    second = store.add_model(_model_id(2), PROBLEM_A, seed=1)
    sizes = problem_sizes(store.root, store.index)
    assert sizes[PROBLEM_A] == directory_size(
        store.models.path(first)
    ) + directory_size(store.models.path(second))


def test_directory_size_counts_nested_members(store: _Store) -> None:
    """A record can hold a `.partial/` resume artifact, which is where the
    bytes actually pile up -- an interrupted training leaves optimiser state
    behind. Counting only the top level would report the store as smaller than
    it is, and would tell `gc --partials` it has nothing to reclaim."""
    model = store.add_model(_model_id(1))
    directory = store.models.path(model)
    flat = sum(m.stat().st_size for m in directory.iterdir() if m.is_file())
    assert MANIFEST in {m.name for m in directory.iterdir()}
    assert directory_size(directory) == flat

    partial = directory / ".partial"
    partial.mkdir()
    (partial / "optimizer.safetensors").write_bytes(b"x" * 4096)
    assert directory_size(directory) == flat + 4096


def test_a_plan_counts_the_contents_of_a_partial_it_would_free(
    store: _Store,
) -> None:
    """`bytes_freed` is what the confirmation prompt shows, so it has to include
    what is inside the artifact, not just the directory entry."""
    model = store.add_model(_model_id(1))
    partial = store.models.path(model) / ".partial"
    partial.mkdir()
    (partial / "optimizer.safetensors").write_bytes(b"x" * 8192)

    plan = plan_gc(
        store.root, store.index, GarbagePolicy(partials=True, measurements=False)
    )
    assert plan.bytes_freed == 8192
