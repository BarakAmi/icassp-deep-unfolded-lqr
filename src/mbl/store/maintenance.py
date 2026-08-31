"""Store maintenance: `stat`, `verify` and `gc` (Annex 02 §5).

These are store operations, not command-line ones. Keeping them here rather than
in `mbl.cli` means they can be tested against a real store without going through
argument parsing, and reused by the runner when it arrives in Tier 5.

Two decisions in here are safety decisions rather than convenience ones.

**`verify` reports what it could not check.** Recomputing an identifier from a
specification is the only test that can falsify the claim a content-addressed
directory name makes about its own content -- but a model's identifier derives
from five specification sub-trees the grammar (Tier 3) does not yet emit. Those
records are counted as *unverifiable* and printed as such. Treating them as
passes would let `verify` report a clean store while checking almost none of it.

**`gc` refuses to run against an empty study table.** Collection keeps whatever
a live study references, so with no studies the referenced set is empty and
every measurement is collectable. Studies arrive in Tier 5, which means that on
every store existing today the honest answer is not "nothing is referenced" but
"nothing can say yet what is referenced". The distinction is a whole store, so
it is refused rather than confirmed away.
"""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import measurement_id, model_id
from .index import StoreIndex
from .layout import MEASUREMENTS_DIR, MODELS_DIR, PARTIAL

#: Specification keys `model_id` needs before an identifier can be recomputed.
MODEL_IDENTITY_KEYS = ("problem", "contender", "training", "ctx", "seed")

#: The same, for `measurement_id`.
MEASUREMENT_IDENTITY_KEYS = ("model", "eval_problem", "eval_protocol")


class NoLiveStudiesError(Exception):
    """Garbage collection was asked to keep what live studies reference, and
    nothing is live.

    Refused rather than treated as "keep nothing", because the two states are
    the same query result with opposite meanings and the destructive reading is
    always the wrong guess.
    """


def records_root(root: Path, kind: str) -> Path:
    """The directory one class of record lives in."""
    return Path(root) / kind


def record_path(root: Path, kind: str, record_id: str) -> Path:
    """Where one record lives."""
    return records_root(root, kind) / record_id


def directory_size(path: Path) -> int:
    """Total bytes of every file at or below `path`. Absent paths are zero."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


# --------------------------------------------------------------------------
# stat
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassStat:
    """One row of `mbl store stat`."""

    kind: str
    count: int
    size_bytes: int


@dataclass(frozen=True)
class GrowthBucket:
    """Records created in one calendar month."""

    month: str
    models: int
    measurements: int


@dataclass(frozen=True)
class StoreStat:
    """What `mbl store stat` reports."""

    classes: tuple[ClassStat, ...]
    growth: tuple[GrowthBucket, ...]
    total_bytes: int


def store_stat(root: Path, index: StoreIndex) -> StoreStat:
    """Size and count by class, plus growth by month.

    `total_bytes` is the whole tree rather than the sum of the classes, so
    staging leftovers and write-ahead logs are visible instead of unaccounted.
    """
    model_rows, measurement_rows = index.models(), index.measurements()

    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in model_rows:
        if row.created_utc:
            buckets[row.created_utc[:7]][0] += 1
    for measurement in measurement_rows:
        if measurement.created_utc:
            buckets[measurement.created_utc[:7]][1] += 1

    return StoreStat(
        classes=(
            ClassStat(
                "models",
                len(model_rows),
                directory_size(records_root(root, MODELS_DIR)),
            ),
            ClassStat(
                "measurements",
                len(measurement_rows),
                directory_size(records_root(root, MEASUREMENTS_DIR)),
            ),
            ClassStat("index", 1, directory_size(index.path)),
        ),
        growth=tuple(
            GrowthBucket(month, counts[0], counts[1])
            for month, counts in sorted(buckets.items())
        ),
        total_bytes=directory_size(Path(root)),
    )


def problem_sizes(root: Path, index: StoreIndex) -> dict[str, int]:
    """Bytes each problem occupies, for `mbl models tree`'s header.

    A measurement is filed under no problem of its own, so it is charged to the
    problem that trained the model it evaluates -- including when it evaluates
    that model somewhere else, which is what a shifted measurement is. Charging
    it to its *evaluation* problem instead would attribute the cost of a
    distribution-shift sweep to a problem nothing was ever trained on.
    """
    problem_of = {row.model_id: row.problem_id for row in index.models()}

    sizes: dict[str, int] = defaultdict(int)
    for model, problem in problem_of.items():
        sizes[problem] += directory_size(record_path(root, MODELS_DIR, model))
    for measurement in index.measurements():
        trained_on = problem_of.get(measurement.model_id)
        if trained_on is not None:
            sizes[trained_on] += directory_size(
                record_path(root, MEASUREMENTS_DIR, measurement.measurement_id)
            )
    return dict(sizes)


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """What `mbl store verify` found.

    Attributes:
        corrupt: Members failing their recorded checksum, or absent.
        inconsistent: Records whose identifier does not follow from their own
            specification, or which reference something that is not there.
        missing: Indexed records with no directory.
        unindexed: Directories no index row mentions.
        unverifiable: Records whose specification cannot pin an identifier yet.
        checked: Records whose identifier was recomputed and matched.
    """

    corrupt: tuple[str, ...] = ()
    inconsistent: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unindexed: tuple[str, ...] = ()
    unverifiable: int = 0
    checked: int = 0

    @property
    def ok(self) -> bool:
        """Whether anything actionable was found.

        `unverifiable` is deliberately not a failure: it is a known gap in the
        grammar, not a defect in the store. It is reported on every run so that
        it cannot be mistaken for coverage.
        """
        return not (self.corrupt or self.inconsistent or self.missing or self.unindexed)

    def lines(self) -> tuple[str, ...]:
        """The report as printed, most serious first."""
        out: list[str] = []
        for label, entries in (
            ("corrupt", self.corrupt),
            ("inconsistent", self.inconsistent),
            ("missing from disk", self.missing),
            ("missing from the index", self.unindexed),
        ):
            out.extend(f"{label}: {entry}" for entry in entries)
        out.append(
            f"{self.checked} identifiers recomputed and matched; "
            f"{self.unverifiable} unverifiable (specification cannot pin an "
            "identifier until the grammar defines it)"
        )
        return tuple(out)


def _recomputed(spec: Mapping[str, Any], keys: Sequence[str], kind: str) -> str | None:
    """The identifier `spec` implies, or `None` when it cannot say."""
    if not all(key in spec for key in keys):
        return None
    if kind == "model":
        return model_id(
            problem=str(spec["problem"]),
            contender=spec["contender"],
            training=spec["training"],
            ctx=spec["ctx"],
            seed=int(spec["seed"]),
        )
    return measurement_id(
        str(spec["model"]), str(spec["eval_problem"]), spec["eval_protocol"]
    )


def verify_store(root: Path, index: StoreIndex) -> VerifyReport:
    """Check integrity, identifier consistency and orphans in both directions.

    The only maintenance operation that opens records, so it is also the only
    one that imports the tensor libraries -- deferred here rather than at module
    scope so that `stat` and `gc` stay instant.
    """
    from .content_store import MeasurementStore, ModelStore

    models, measurements = ModelStore(root), MeasurementStore(root)
    corrupt: list[str] = []
    inconsistent: list[str] = []
    unverifiable = 0
    checked = 0

    on_disk = {
        "model": sorted(models.list_ids()),
        "measurement": sorted(measurements.list_ids()),
    }
    for kind, store, keys in (
        ("model", models, MODEL_IDENTITY_KEYS),
        ("measurement", measurements, MEASUREMENT_IDENTITY_KEYS),
    ):
        for record_id in on_disk[kind]:
            corrupt.extend(store.verify(record_id))
            derived = _recomputed(store.spec(record_id), keys, kind)
            if derived is None:
                unverifiable += 1
            elif derived != record_id:
                inconsistent.append(
                    f"{record_id}: its specification derives {derived}, so the "
                    "record is filed under an identifier its own content does "
                    "not produce"
                )
            else:
                checked += 1

    indexed_models = {row.model_id for row in index.models()}
    indexed_measurements = {row.measurement_id for row in index.measurements()}
    for row in index.measurements():
        if row.model_id not in set(on_disk["model"]):
            inconsistent.append(
                f"{row.measurement_id}: evaluates model {row.model_id}, which is "
                "not in the store"
            )

    missing = sorted(
        (indexed_models - set(on_disk["model"]))
        | (indexed_measurements - set(on_disk["measurement"]))
    )
    unindexed = sorted(
        (set(on_disk["model"]) - indexed_models)
        | (set(on_disk["measurement"]) - indexed_measurements)
    )
    return VerifyReport(
        corrupt=tuple(corrupt),
        inconsistent=tuple(inconsistent),
        missing=tuple(missing),
        unindexed=tuple(unindexed),
        unverifiable=unverifiable,
        checked=checked,
    )


# --------------------------------------------------------------------------
# gc
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GarbagePolicy:
    """What a collection is allowed to remove.

    Bundled rather than passed as loose flags so that the destructive options
    travel together and are read together at the call site -- `include_models`
    means something very different depending on `keep_studies`.

    Attributes:
        keep_studies: Restrict liveness to these studies. `None` means every
            study in the index is live.
        measurements: Collect measurements no live study references.
        include_models: Also collect models nothing live references. Off by
            default, because a measurement is cheap to recompute from a model
            and a model is the one expensive object in the store.
        partials: Collect abandoned resume artifacts. Independent of studies.
        older_than_seconds: Only collect partials at least this old.
    """

    keep_studies: Sequence[str] | None = None
    measurements: bool = True
    include_models: bool = False
    partials: bool = False
    older_than_seconds: float | None = None


@dataclass(frozen=True)
class GarbagePlan:
    """What a collection would remove. Computed before anything is deleted, so
    it can be printed for confirmation."""

    measurements: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    partials: tuple[Path, ...] = ()
    bytes_freed: int = 0

    @property
    def empty(self) -> bool:
        return not (self.measurements or self.models or self.partials)


def _live_studies(index: StoreIndex, keep: Sequence[str] | None) -> set[str]:
    with index.connect() as conn:
        known = {
            row["study_id"]
            for row in conn.execute("SELECT DISTINCT study_id FROM study_members")
        }
    return known if keep is None else known & set(keep)


def _members(index: StoreIndex, studies: Iterable[str], kind: str) -> set[str]:
    studies = list(studies)
    if not studies:
        return set()
    placeholders = ",".join("?" * len(studies))
    with index.connect() as conn:
        return {
            row["member_id"]
            for row in conn.execute(
                f"SELECT member_id FROM study_members WHERE kind = ? "
                f"AND study_id IN ({placeholders})",
                [kind, *studies],
            )
        }


def _stale_partials(
    root: Path, older_than_seconds: float | None, now: float | None
) -> tuple[Path, ...]:
    cutoff = (
        None
        if older_than_seconds is None
        else (now if now is not None else time.time()) - older_than_seconds
    )
    found: list[Path] = []
    for kind in (MODELS_DIR, MEASUREMENTS_DIR):
        directory = records_root(root, kind)
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            partial = child / PARTIAL
            if partial.is_dir() and (
                cutoff is None or partial.stat().st_mtime <= cutoff
            ):
                found.append(partial)
    return tuple(found)


def plan_gc(
    root: Path,
    index: StoreIndex,
    policy: GarbagePolicy | None = None,
    *,
    now: float | None = None,
) -> GarbagePlan:
    """Decide what to collect, without removing anything.

    Args:
        root: The store root.
        index: The store's index.
        policy: What may be removed. Defaults to measurements only.
        now: Clock override, for tests.

    Raises:
        NoLiveStudiesError: If records were in scope and no study is live.
    """
    root = Path(root)
    policy = policy or GarbagePolicy()
    keep_studies = policy.keep_studies
    measurements = policy.measurements
    include_models = policy.include_models
    collect_measurements: set[str] = set()
    collect_models: set[str] = set()

    if measurements or include_models:
        live = _live_studies(index, keep_studies)
        if not live:
            raise NoLiveStudiesError(
                "no live study references anything, so every measurement in the "
                "store would be collectable. This is refused: studies arrive with "
                "Tier 5, so an empty study table means 'nothing can say yet what "
                "is referenced', not 'nothing is referenced'. Use --partials to "
                "collect abandoned resume artifacts, which belong to no study."
                + (
                    f" (--keep-studies named {', '.join(keep_studies)}, none of "
                    "which is in the index)"
                    if keep_studies
                    else ""
                )
            )
        all_measurements = index.measurements()
        referenced_measurements = _members(index, live, "measurement")
        collect_measurements = {
            row.measurement_id
            for row in all_measurements
            if row.measurement_id not in referenced_measurements
        }
        if include_models:
            protected = _members(index, live, "model") | {
                row.model_id
                for row in all_measurements
                if row.measurement_id in referenced_measurements
            }
            collect_models = {
                row.model_id for row in index.models() if row.model_id not in protected
            }
            # A measurement of a collected model goes with it, whatever the
            # measurement flag says: leaving it would leave an evaluation of a
            # model that is no longer there.
            collect_measurements |= {
                row.measurement_id
                for row in all_measurements
                if row.model_id in collect_models
            }
        if not measurements:
            collect_measurements &= {
                row.measurement_id
                for row in all_measurements
                if row.model_id in collect_models
            }

    stale = (
        _stale_partials(root, policy.older_than_seconds, now) if policy.partials else ()
    )
    freed = (
        sum(
            directory_size(record_path(root, MEASUREMENTS_DIR, m))
            for m in collect_measurements
        )
        + sum(directory_size(record_path(root, MODELS_DIR, m)) for m in collect_models)
        + sum(directory_size(p) for p in stale)
    )
    return GarbagePlan(
        measurements=tuple(sorted(collect_measurements)),
        models=tuple(sorted(collect_models)),
        partials=stale,
        bytes_freed=freed,
    )


def apply_gc(root: Path, index: StoreIndex, plan: GarbagePlan) -> None:
    """Carry out `plan`, on disk and in the index.

    Measurements are removed from the index before models, because the foreign
    key forbids the reverse order -- which is also the property that stops a
    half-applied collection leaving a measurement of a model that is gone.
    """
    root = Path(root)
    for measurement in plan.measurements:
        shutil.rmtree(
            record_path(root, MEASUREMENTS_DIR, measurement), ignore_errors=True
        )
    for model in plan.models:
        shutil.rmtree(record_path(root, MODELS_DIR, model), ignore_errors=True)
    for partial in plan.partials:
        shutil.rmtree(partial, ignore_errors=True)
    index.delete_measurements(plan.measurements)
    index.delete_models(plan.models)
