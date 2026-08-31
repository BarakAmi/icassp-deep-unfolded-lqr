"""The Experiment History Registry — the laboratory's logbook (REFACTOR_PLAN
v3, T3.j, closes G4): one centralized, lightweight, append-only catalog at the
artifacts root, recording one standardized entry per `run_experiment`
execution — **whether freshly computed or served from cache** — so reuse is
auditable instead of invisible.

Format: CSV, one record per line, atomic single-record appends,
human-scannable and one ``pandas.read_csv`` away from meta-analysis.

Non-authoritative index law: the run directories remain the sole source of
truth. `rebuild` regenerates the registry by scanning run metadata, and any
divergence resolves in favor of the run directories — losing or corrupting
the registry loses nothing. (Cache-hit executions created no run directory,
so a rebuild recovers fresh computations only; that is the accepted
consequence of the law, not a defect.)
"""

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .cache import CACHE_KEY_PARAM, STAMP_PARAM
from ..persistence import discover_runs, load_run_metadata

logger = logging.getLogger(__name__)

#: The registry file's name at the artifacts root.
REGISTRY_FILENAME = "experiment_registry.csv"

#: The record schema (data contract) — one column order, everywhere.
FIELDNAMES = (
    "experiment_signature",
    "timestamp_utc",
    "experiment_name",
    "contenders",
    "horizon",
    "state_dim",
    "control_dim",
    "final_costs",
    "dispositions",
    "run_dirs",
    "provenance_stamp",
)

#: Separator packing the per-contender sub-fields into one CSV cell.
PACK_SEPARATOR = ";"


@dataclass(frozen=True)
class HistoryRecord:
    """One standardized registry entry (see `FIELDNAMES` for the contract).

    Attributes:
        experiment_signature: The experiment's composite signature digest.
        timestamp_utc: ISO-8601 UTC timestamp of the execution.
        experiment_name: The experiment's declared name.
        contenders: ``label(family)`` per contender, packed.
        horizon: The problem horizon ``T``.
        state_dim: The state dimension ``n``.
        control_dim: The control dimension ``m``.
        final_costs: ``label=cost`` per contender, packed.
        dispositions: ``label=fresh|hit|recompute`` per contender, packed.
        run_dirs: The producing run-directory names, packed.
        provenance_stamp: The code-provenance stamp the execution ran under.
    """

    experiment_signature: str
    timestamp_utc: str
    experiment_name: str
    contenders: str
    horizon: int
    state_dim: int
    control_dim: int
    final_costs: str
    dispositions: str
    run_dirs: str
    provenance_stamp: str


def pack(entries: dict[str, str]) -> str:
    """Pack per-contender key=value pairs into one registry cell.

    Args:
        entries: label -> value.

    Returns:
        ``"label1=v1;label2=v2"`` (insertion order preserved).
    """
    return PACK_SEPARATOR.join(f"{k}={v}" for k, v in entries.items())


class ExperimentHistoryRegistry:
    """The append-only master catalog at one artifacts root."""

    def __init__(self, root: Path | str) -> None:
        """
        Args:
            root: The experiments/artifacts root the registry file lives in.
        """
        self.root = Path(root)
        self.path = self.root / REGISTRY_FILENAME

    def append(self, record: HistoryRecord) -> None:
        """Append one record — a single atomic write, header included on
        first use.

        Args:
            record: The standardized entry.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if is_new:
                writer.writeheader()
            writer.writerow(record.__dict__)
        logger.info(
            "History registry: appended %r (%s) to %s.",
            record.experiment_name,
            record.dispositions,
            self.path,
        )

    def records(self) -> list[dict[str, str]]:
        """Every registry record, oldest first.

        Returns:
            The parsed rows (empty when the registry doesn't exist yet).
        """
        if not self.path.exists():
            return []
        with self.path.open(newline="") as f:
            return list(csv.DictReader(f))

    def rebuild(self) -> int:
        """Regenerate the registry in full from the run directories (the
        non-authoritative index law): one record per cache-indexed run,
        oldest first, replacing the current file.

        Returns:
            The number of records rebuilt.
        """
        records: list[HistoryRecord] = []
        for run in sorted(discover_runs(self.root), key=lambda r: r.run_id):
            metadata = load_run_metadata(run.run_dir)
            params = metadata.get("params", {})
            if CACHE_KEY_PARAM not in params:
                continue  # not an experiment-layer run (e.g. a bare engine run)
            label = params.get("contender_label", run.run_id)
            cost = metadata.get("metrics", {}).get("eval_expected_cost", "")
            records.append(
                HistoryRecord(
                    experiment_signature=params.get("experiment_signature", ""),
                    timestamp_utc=metadata.get("created_at", ""),
                    experiment_name=params.get("experiment_name", ""),
                    contenders=pack({label: params.get("contender_family", "")}),
                    horizon=params.get("horizon", 0),
                    state_dim=params.get("state_dim", 0),
                    control_dim=params.get("control_dim", 0),
                    final_costs=pack({label: str(cost)}),
                    dispositions=pack({label: "fresh"}),
                    run_dirs=run.run_id,
                    provenance_stamp=params.get(STAMP_PARAM, ""),
                )
            )
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for record in records:
                writer.writerow(record.__dict__)
        logger.info(
            "History registry: rebuilt %d record(s) from %s.", len(records), self.root
        )
        return len(records)


def utc_timestamp() -> str:
    """The registry's canonical UTC timestamp (second resolution)."""
    return datetime.now(UTC).isoformat(timespec="seconds")
