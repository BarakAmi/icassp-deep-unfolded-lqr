"""The store's queryable index (Annex 02 §4).

A single SQLite database answering "what is in the store, right now?" as an
indexed primary-key hit rather than a linear scan of thousands of directories,
each parsed from JSON.

The index is **derived**: every row here is a projection of a `spec.json` or
`provenance.json` under the content stores, so it can be dropped and rebuilt at
any time. The one exception is `queue`, which is genuine execution state, and
`reindex` therefore preserves it while rebuilding everything else -- otherwise
rebuilding the index would discard a two-day batch.

Rows are passed in explicitly rather than parsed from specs here. Extracting
`state_dim` or `family` from a specification is the grammar's job (Tier 3), and
hard-coding those paths in the index would couple it to a spec shape that does
not exist yet.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ids import MeasurementID, ModelID, ProblemID

if TYPE_CHECKING:
    # Only `reindex` touches the content stores, and only through their public
    # methods. Importing them at module scope would make every read of the
    # index -- the `mbl` command line's whole job -- load PyTorch and pandas
    # first, for a query that never opens a checkpoint.
    from .content_store import MeasurementStore, ModelStore

#: How long a writer waits for a competing writer before reporting failure.
#: WAL keeps readers unblocked, but two writers still serialise, and a parallel
#: runner must wait rather than abort (Annex 02 §7.3).
BUSY_TIMEOUT_MS = 30_000

#: Tables rebuilt by `reindex`. `queue` is deliberately absent: it is state, not
#: a projection of the trees.
DERIVED_TABLES = ("metrics", "measurements", "models", "study_members")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY, semantic_name TEXT, problem_id TEXT,
    family TEXT, contender_id TEXT, seed INTEGER,
    state_dim INTEGER, control_dim INTEGER, horizon INTEGER,
    created_utc TEXT, stamp TEXT, wall_time_s REAL, peak_vram_mb REAL,
    spec_json TEXT
);
CREATE INDEX IF NOT EXISTS models_by_problem ON models(problem_id);
CREATE INDEX IF NOT EXISTS models_by_family  ON models(family);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    eval_problem_id TEXT NOT NULL,
    is_shifted INTEGER NOT NULL,
    created_utc TEXT, spec_json TEXT
);
CREATE INDEX IF NOT EXISTS measurements_by_model ON measurements(model_id, is_shifted);

CREATE TABLE IF NOT EXISTS metrics (
    measurement_id TEXT NOT NULL REFERENCES measurements(measurement_id),
    key TEXT NOT NULL, value REAL,
    PRIMARY KEY (measurement_id, key)
);

CREATE TABLE IF NOT EXISTS study_members (
    study_id TEXT NOT NULL, kind TEXT NOT NULL, member_id TEXT NOT NULL,
    PRIMARY KEY (study_id, kind, member_id)
);

CREATE TABLE IF NOT EXISTS aliases (alias TEXT PRIMARY KEY, target_id TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS queue (
    entry_id TEXT PRIMARY KEY, study TEXT NOT NULL, tier TEXT, options_json TEXT,
    state TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
    added_utc TEXT, started_utc TEXT, finished_utc TEXT, heartbeat_utc TEXT,
    failure_reason TEXT
);
CREATE INDEX IF NOT EXISTS queue_by_state ON queue(state, priority DESC, added_utc);
"""


class IndexError_(Exception):
    """Base class for index failures."""


class UnknownModelError(IndexError_, KeyError):
    """A measurement referenced a model the index has never seen."""


class UnknownPrefixError(IndexError_, KeyError):
    """No record matches the given identifier prefix."""


class AmbiguousPrefixError(IndexError_):
    """Several records match the given prefix.

    Refused rather than resolved to the first match, because silently picking one
    would attach a command to the wrong model.
    """


@dataclass(frozen=True)
class ModelRow:
    """One row of the `models` table -- a projection of a stored model."""

    model_id: ModelID
    semantic_name: str
    problem_id: ProblemID
    family: str
    contender_id: str
    seed: int
    state_dim: int | None = None
    control_dim: int | None = None
    horizon: int | None = None
    created_utc: str | None = None
    stamp: str | None = None
    wall_time_s: float | None = None
    peak_vram_mb: float | None = None
    spec_json: str = "{}"


@dataclass(frozen=True)
class MeasurementRow:
    """One row of the `measurements` table, plus its metrics.

    `is_shifted` is not settable: it is derived from the model's training problem
    when the row is written, so it cannot drift from the truth.
    """

    measurement_id: MeasurementID
    model_id: ModelID
    eval_problem_id: ProblemID
    created_utc: str | None = None
    spec_json: str = "{}"
    metrics: Mapping[str, float] = field(default_factory=dict)
    is_shifted: bool | None = None


class StoreIndex:
    """The store's index. Cheap to construct; the schema is created on demand."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "index.sqlite"
        with self.connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A configured connection, committed on success and closed always.

        WAL and the busy timeout are set per connection because they are
        connection-scoped in SQLite; foreign keys likewise, and they are on so
        that a measurement can never reference a model that is not there.
        """
        conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- models -----------------------------------------------------------

    def upsert_model(self, row: ModelRow) -> None:
        """Insert or update one model row."""
        self.upsert_models([row])

    def upsert_models(self, rows: Iterable[ModelRow]) -> None:
        """Insert or update many model rows in one transaction."""
        payload = [
            (
                r.model_id,
                r.semantic_name,
                r.problem_id,
                r.family,
                r.contender_id,
                r.seed,
                r.state_dim,
                r.control_dim,
                r.horizon,
                r.created_utc,
                r.stamp,
                r.wall_time_s,
                r.peak_vram_mb,
                r.spec_json,
            )
            for r in rows
        ]
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(model_id) DO UPDATE SET
                     semantic_name=excluded.semantic_name, problem_id=excluded.problem_id,
                     family=excluded.family, contender_id=excluded.contender_id,
                     seed=excluded.seed, state_dim=excluded.state_dim,
                     control_dim=excluded.control_dim, horizon=excluded.horizon,
                     created_utc=excluded.created_utc, stamp=excluded.stamp,
                     wall_time_s=excluded.wall_time_s, peak_vram_mb=excluded.peak_vram_mb,
                     spec_json=excluded.spec_json""",
                payload,
            )

    def model(self, model_id: str) -> ModelRow:
        """One model by identifier.

        Raises:
            UnknownModelError: If the index has no such model.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        if row is None:
            raise UnknownModelError(f"no model {model_id} in {self.path}")
        return _model_row(row)

    def models(
        self,
        *,
        problem_id: str | None = None,
        family: str | None = None,
        contender_id: str | None = None,
        seed: int | None = None,
    ) -> list[ModelRow]:
        """Models matching every supplied filter, ordered by identifier."""
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("problem_id", problem_id),
            ("family", family),
            ("contender_id", contender_id),
            ("seed", seed),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM models{where} ORDER BY model_id", params
            ).fetchall()
        return [_model_row(r) for r in rows]

    def resolve_model_prefix(self, prefix: str) -> ModelID:
        """The single model whose identifier starts with `prefix`.

        Raises:
            UnknownPrefixError: If nothing matches.
            AmbiguousPrefixError: If more than one does.
        """
        return self._single(
            "identifier prefix", prefix, self._by_prefix("model_id", prefix)
        )

    def resolve_model_reference(self, reference: str) -> ModelID:
        """The single model a user-typed reference denotes (Annex 02 §3).

        Tried in order: the whole identifier, an alias, an identifier prefix,
        then a semantic-name prefix -- so an exact identifier can never be
        shadowed by something that merely starts with the same characters.

        Raises:
            UnknownPrefixError: If nothing matches.
            AmbiguousPrefixError: If more than one does at the first level that
                matches at all. Refused rather than resolved to the first hit,
                because silently picking one attaches the command to the wrong
                model.
        """
        with self.connect() as conn:
            exact = conn.execute(
                "SELECT model_id FROM models WHERE model_id = ?", (reference,)
            ).fetchone()
            if exact is not None:
                return ModelID(exact["model_id"])
            alias = conn.execute(
                "SELECT target_id FROM aliases WHERE alias = ?", (reference,)
            ).fetchone()
        if alias is not None:
            return ModelID(alias["target_id"])
        for column in ("model_id", "semantic_name"):
            matches = self._by_prefix(column, reference)
            if matches:
                return self._single("reference", reference, matches)
        raise UnknownPrefixError(f"no model matches {reference!r}")

    def _by_prefix(self, column: str, prefix: str) -> list[str]:
        """Identifiers whose `column` starts with `prefix`.

        The pattern is escaped: SQL `LIKE` treats `_` as a single-character
        wildcard, so an unescaped `a_c` would also match `abc` -- turning a
        typo into a confident answer about the wrong model.
        """
        with self.connect() as conn:
            return [
                row["model_id"]
                for row in conn.execute(
                    f"SELECT model_id FROM models WHERE {column} LIKE ? ESCAPE '\\' "
                    "ORDER BY model_id",
                    (f"{_escape_like(prefix)}%",),
                ).fetchall()
            ]

    @staticmethod
    def _single(kind: str, needle: str, matches: Sequence[str]) -> ModelID:
        if not matches:
            raise UnknownPrefixError(f"no model matches {kind} {needle!r}")
        if len(matches) > 1:
            raise AmbiguousPrefixError(
                f"{kind} {needle!r} matches {len(matches)} models: "
                f"{', '.join(matches[:5])}{'...' if len(matches) > 5 else ''}"
            )
        return ModelID(matches[0])

    # -- measurements -----------------------------------------------------

    def upsert_measurement(self, row: MeasurementRow) -> None:
        """Insert or update one measurement and its metrics.

        `is_shifted` is computed here, from the model's own training problem.

        Raises:
            UnknownModelError: If the referenced model is not indexed. It is
                refused rather than defaulted, because with no training problem
                to compare against, `is_shifted` would have to be guessed -- and
                guessing "not shifted" would report an out-of-distribution
                measurement as in-distribution.
        """
        with self.connect() as conn:
            model = conn.execute(
                "SELECT problem_id FROM models WHERE model_id = ?", (row.model_id,)
            ).fetchone()
            if model is None:
                raise UnknownModelError(
                    f"cannot index measurement {row.measurement_id}: its model "
                    f"{row.model_id} is not in the index, so `is_shifted` cannot be "
                    "derived. Index the model first."
                )
            is_shifted = int(row.eval_problem_id != model["problem_id"])
            conn.execute(
                """INSERT INTO measurements VALUES (?,?,?,?,?,?)
                   ON CONFLICT(measurement_id) DO UPDATE SET
                     model_id=excluded.model_id,
                     eval_problem_id=excluded.eval_problem_id,
                     is_shifted=excluded.is_shifted,
                     created_utc=excluded.created_utc, spec_json=excluded.spec_json""",
                (
                    row.measurement_id,
                    row.model_id,
                    row.eval_problem_id,
                    is_shifted,
                    row.created_utc,
                    row.spec_json,
                ),
            )
            conn.execute(
                "DELETE FROM metrics WHERE measurement_id = ?", (row.measurement_id,)
            )
            conn.executemany(
                "INSERT INTO metrics VALUES (?,?,?)",
                [(row.measurement_id, k, float(v)) for k, v in row.metrics.items()],
            )

    def measurements(
        self, *, model_id: str | None = None, shifted: bool | None = None
    ) -> list[MeasurementRow]:
        """Measurements matching the supplied filters, with their metrics."""
        clauses: list[str] = []
        params: list[object] = []
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if shifted is not None:
            clauses.append("is_shifted = ?")
            params.append(int(shifted))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM measurements{where} ORDER BY measurement_id", params
            ).fetchall()
            metrics: dict[str, dict[str, float]] = {}
            for m in conn.execute("SELECT * FROM metrics").fetchall():
                metrics.setdefault(m["measurement_id"], {})[m["key"]] = m["value"]
        return [
            MeasurementRow(
                measurement_id=MeasurementID(r["measurement_id"]),
                model_id=ModelID(r["model_id"]),
                eval_problem_id=ProblemID(r["eval_problem_id"]),
                created_utc=r["created_utc"],
                spec_json=r["spec_json"],
                metrics=metrics.get(r["measurement_id"], {}),
                is_shifted=bool(r["is_shifted"]),
            )
            for r in rows
        ]

    # -- deletion ---------------------------------------------------------

    def delete_measurements(self, measurement_ids: Iterable[str]) -> None:
        """Drop measurement rows and their metrics."""
        payload = [(identifier,) for identifier in measurement_ids]
        if not payload:
            return
        with self.connect() as conn:
            conn.executemany("DELETE FROM metrics WHERE measurement_id = ?", payload)
            conn.executemany(
                "DELETE FROM measurements WHERE measurement_id = ?", payload
            )

    def delete_models(self, model_ids: Iterable[str]) -> None:
        """Drop model rows.

        Their measurements must already be gone: the foreign key refuses a
        measurement pointing at an absent model, which is what stops a deletion
        from leaving a measurement of nothing.
        """
        payload = [(identifier,) for identifier in model_ids]
        if not payload:
            return
        with self.connect() as conn:
            conn.executemany("DELETE FROM models WHERE model_id = ?", payload)

    # -- queue (state, not a projection) ----------------------------------

    def enqueue(
        self, *, entry_id: str, study: str, tier: str | None = None, priority: int = 0
    ) -> None:
        """Add a study to the execution queue (Annex 06 §5)."""
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO queue (entry_id, study, tier, priority, "
                "state, added_utc) VALUES (?,?,?,?,'pending',datetime('now'))",
                (entry_id, study, tier, priority),
            )

    def queue_entries(self, *, state: str | None = None) -> list[dict[str, Any]]:
        """Queue rows, newest-priority first."""
        where = "" if state is None else " WHERE state = ?"
        params: list[object] = [] if state is None else [state]
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM queue{where} ORDER BY priority DESC, added_utc", params
            ).fetchall()
        return [dict(r) for r in rows]

    # -- rebuild ----------------------------------------------------------

    def reindex(self, models: ModelStore, measurements: MeasurementStore) -> None:
        """Rebuild every derived table from the content stores.

        Models are indexed **before** measurements, and that ordering is
        load-bearing rather than incidental: `is_shifted` is derived from a
        model's training problem, so a measurement encountered first could not be
        indexed at all.

        `queue` is untouched.
        """
        with self.connect() as conn:
            for table in DERIVED_TABLES:
                conn.execute(f"DELETE FROM {table}")

        self.upsert_models(
            _model_row_from_record(models, model_id) for model_id in models.list_ids()
        )
        for measurement_id in measurements.list_ids():
            record = measurements.get(measurement_id)
            self.upsert_measurement(
                MeasurementRow(
                    measurement_id=MeasurementID(measurement_id),
                    model_id=ModelID(str(record.spec["model"])),
                    eval_problem_id=ProblemID(str(record.spec["eval_problem"])),
                    created_utc=None,
                    spec_json=json.dumps(record.spec, sort_keys=True),
                    metrics=record.metrics,
                )
            )


def _escape_like(text: str) -> str:
    """Neutralise SQL `LIKE`'s wildcards in user-supplied text."""
    for character in ("\\", "%", "_"):
        text = text.replace(character, f"\\{character}")
    return text


def _model_row(row: sqlite3.Row) -> ModelRow:
    return ModelRow(
        model_id=ModelID(row["model_id"]),
        semantic_name=row["semantic_name"],
        problem_id=ProblemID(row["problem_id"]),
        family=row["family"],
        contender_id=row["contender_id"],
        seed=row["seed"],
        state_dim=row["state_dim"],
        control_dim=row["control_dim"],
        horizon=row["horizon"],
        created_utc=row["created_utc"],
        stamp=row["stamp"],
        wall_time_s=row["wall_time_s"],
        peak_vram_mb=row["peak_vram_mb"],
        spec_json=row["spec_json"],
    )


def _model_row_from_record(models: ModelStore, model_id: str) -> ModelRow:
    """Project a stored model into an index row.

    Reads only fields the store itself guarantees. Richer projection -- problem
    dimensions, the semantic name -- belongs with the grammar that defines those
    shapes, and is filled in by the writer rather than guessed here.
    """
    record = models.get(model_id)
    spec, prov = record.spec, record.provenance
    return ModelRow(
        model_id=ModelID(model_id),
        semantic_name=str(spec.get("semantic_name", "")),
        problem_id=ProblemID(str(spec["problem_id"])),
        family=str(spec.get("family", "")),
        contender_id=str(spec.get("contender_id", "")),
        seed=int(spec.get("seed", 0)),
        state_dim=spec.get("state_dim"),
        control_dim=spec.get("control_dim"),
        horizon=spec.get("horizon"),
        created_utc=prov.get("created_utc"),
        stamp=prov.get("stamp"),
        wall_time_s=prov.get("wall_time_s"),
        peak_vram_mb=prov.get("peak_vram_mb"),
        spec_json=json.dumps(spec, sort_keys=True),
    )
