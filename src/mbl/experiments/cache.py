"""`ExperimentCache` — signature-keyed execution interception, code-aware
(REFACTOR_PLAN v3, T3.e): a two-method get/put protocol keyed by content
signature ⊕ code-provenance stamp, stored *per contender* (two-level
granularity — adding a sixth contender to a five-contender experiment
recomputes one synthesis, not six).

Storage is the existing persistence layer: the default implementation is an
*index over* the tracker's run layout (``run_YYYYMMDD_HHMMSS_<name>/
metadata.json`` + ``artifacts/``), never a parallel store, and every artifact
byte obeys the T3.i serialization policy (the cache reads exclusively through
the modern, no-code-execution loader registry — the quarantined legacy
loaders are structurally unreachable from here).

Behavior on a stamp mismatch: a loud, WARNING-logged miss. Orphaned artifacts
are never deleted — old runs remain readable for analysis via the tracker;
they simply stop being served as current results.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ..core.utils.signing import compute_signature_digest
from ..persistence import (
    LocalExperimentTracker,
    default_artifact_loader_registry,
    discover_runs,
    generate_run_id,
    list_run_artifacts,
    load_run_metadata,
)

logger = logging.getLogger(__name__)

#: The metadata param keys the cache owns inside a run's ``metadata.json``.
CACHE_KEY_PARAM = "experiment_cache_key"
CONTENT_DIGEST_PARAM = "experiment_content_digest"
STAMP_PARAM = "provenance_stamp"


@dataclass(frozen=True)
class CachePolicy:
    """Per-invocation cache control (T3.e):

    * ``"reuse"`` — hit → load; miss → compute and store (the default).
    * ``"recompute"`` — always compute and store (force-rerun for
      debugging); the fresh result overwrites nothing (a new run directory
      is minted; the newest matching run wins subsequent lookups).
    * ``"readonly"`` — hit → load; miss → typed error, never compute (for
      analysis sessions that must not write).
    """

    mode: Literal["reuse", "recompute", "readonly"] = "reuse"


class CacheReadOnlyMissError(RuntimeError):
    """A readonly-policy execution missed the cache — nothing may be computed."""


def compose_cache_key(content_digest: str, stamp: str) -> str:
    """The stamped cache key: content signature ⊕ code-provenance stamp.

    Args:
        content_digest: The per-contender content digest
            (`Experiment.contender_content_digest`).
        stamp: The code-provenance stamp (`provenance.code_provenance_stamp`).

    Returns:
        The composite key digest.
    """
    return compute_signature_digest(
        {"content": content_digest, "provenance_stamp": stamp}
    )


@dataclass(frozen=True)
class CachedContenderRun:
    """One cache hit, loaded: the producing run's metadata and artifacts.

    Attributes:
        run_dir: The producing run directory.
        params: The run's persisted params (``metadata.json``).
        metrics: The run's persisted scalar metrics.
        arrays: The run's artifacts, loaded through the modern registry and
            keyed by bare artifact name.
    """

    run_dir: Path
    params: Mapping[str, Any]
    metrics: Mapping[str, float]
    arrays: Mapping[str, Any]


@dataclass(frozen=True)
class ContenderPayload:
    """Everything one fresh contender execution asks the cache to persist.

    Attributes:
        content_digest: The stamp-free content digest.
        stamp: The code-provenance stamp of this execution.
        params: Additional identity params (experiment name, label, family,
            dimensions, signature digests).
        metrics: The result's scalar metrics.
        arrays: The result's dense payloads (persisted per T3.i).
    """

    content_digest: str
    stamp: str
    params: Mapping[str, Any]
    metrics: Mapping[str, float]
    arrays: Mapping[str, np.ndarray]


@runtime_checkable
class ExperimentCache(Protocol):
    """The two-method interception seam `run_experiment` calls through."""

    def get(
        self, key: str, *, content_digest: str, stamp: str
    ) -> CachedContenderRun | None:
        """Load the newest run stored under `key`, or ``None`` on a miss."""
        ...

    def open_run(self, run_name: str) -> LocalExperimentTracker:
        """Mint the tracker a fresh (miss-path) computation persists into."""
        ...

    def put(
        self,
        key: str,
        tracker: LocalExperimentTracker,
        payload: ContenderPayload,
    ) -> None:
        """Index `tracker`'s run under `key` and persist `payload`."""
        ...


class TrackerBackedExperimentCache:
    """The default `ExperimentCache`: an index over the tracker's run layout
    under one experiments root. Runs ARE the cache entries; `get` scans run
    metadata for the stamped key, `put` stamps a live tracker's metadata and
    persists the payload through the serializer registry (T3.i formats).
    """

    def __init__(self, root: Path | str) -> None:
        """
        Args:
            root: The experiments/artifacts root directory (created lazily
                by the first `open_run`).
        """
        self.root = Path(root)

    def get(
        self, key: str, *, content_digest: str, stamp: str
    ) -> CachedContenderRun | None:
        """Newest run whose metadata carries `key`, loaded eagerly.

        A run matching the content digest under a DIFFERENT stamp is a
        loud, WARNING-logged miss (never deleted, never served); scanning
        continues in case a current-stamp run also exists.

        Args:
            key: The stamped cache key.
            content_digest: The stamp-free content digest (stale detection).
            stamp: The current code-provenance stamp (diagnostics only —
                the key already embeds it).

        Returns:
            The loaded entry, or ``None``.
        """
        for run in discover_runs(self.root):
            metadata = load_run_metadata(run.run_dir)
            params = metadata.get("params", {})
            if params.get(CACHE_KEY_PARAM) == key:
                arrays = self._load_artifacts(run.run_dir)
                if arrays is None:
                    # metadata.json survived (e.g. committed) but the heavy
                    # artifacts/ payload didn't -- nothing to serve.
                    continue
                return CachedContenderRun(
                    run_dir=run.run_dir,
                    params=params,
                    metrics=metadata.get("metrics", {}),
                    arrays=arrays,
                )
            if (
                params.get(CONTENT_DIGEST_PARAM) == content_digest
                and params.get(STAMP_PARAM) != stamp
            ):
                logger.warning(
                    "Cache entry %s matches this configuration's content "
                    "signature but was produced under provenance stamp %r "
                    "(current: %r) -- released code semantics have changed "
                    "since it was computed. Treating as a MISS; the stale "
                    "run remains on disk for analysis but will never be "
                    "served as a current result.",
                    run.run_id,
                    params.get(STAMP_PARAM),
                    stamp,
                )
        return None

    def open_run(self, run_name: str) -> LocalExperimentTracker:
        """See `ExperimentCache.open_run`.

        Run-id collision guard: the tracker's naming is second-resolution
        timestamps, so a same-named run minted within the same second (e.g.
        a `recompute` immediately after a fresh execution) would silently
        MERGE into the existing directory. Distinct executions must be
        distinct cache entries, so the name is disambiguated first.

        Args:
            run_name: The run's bare name (timestamp prefix added by the
                tracker's own naming).

        Returns:
            A fresh `LocalExperimentTracker` rooted at this cache's root.
        """
        candidate = run_name
        attempt = 1
        while (
            self.root / generate_run_id(candidate, timestamp=datetime.now())
        ).exists():
            attempt += 1
            candidate = f"{run_name}~{attempt}"
        return LocalExperimentTracker(self.root, candidate)

    def put(
        self,
        key: str,
        tracker: LocalExperimentTracker,
        payload: ContenderPayload,
    ) -> None:
        """See `ExperimentCache.put`.

        Args:
            key: The stamped cache key to index under.
            tracker: The live tracker the computation ran through (already
                holding whatever the engine persisted during synthesis).
            payload: The execution's identity/metrics/arrays bundle (see
                `ContenderPayload`).
        """
        tracker.log_params(
            {
                CACHE_KEY_PARAM: key,
                CONTENT_DIGEST_PARAM: payload.content_digest,
                STAMP_PARAM: payload.stamp,
                **payload.params,
            }
        )
        tracker.log_metrics(payload.metrics)
        for name, array in payload.arrays.items():
            tracker.save_artifact(name, np.asarray(array))
        logger.info(
            "Cache PUT: key %s -> %s (%d artifact payload(s)).",
            key,
            tracker.run_dir.name,
            len(payload.arrays),
        )

    def _load_artifacts(self, run_dir: Path) -> dict[str, Any] | None:
        """Every artifact of `run_dir` through the MODERN loader registry
        only, keyed by bare name; ``None`` when the artifacts are gone."""
        registry = default_artifact_loader_registry()
        paths = list_run_artifacts(run_dir)
        if not paths:
            return None
        return {path.stem: registry.load(path) for path in paths}
