"""Content-addressed stores for models and measurements (Annex 02 §2, §7).

A record is a directory named by its identifier, holding only `json`,
`safetensors`, `parquet` and plain-text members -- the no-pickle law, so nothing
executes on load. (Text became a member with Annex 02 §2.2's captured logs; the
law is about payloads that *run*, and a log is inert.)

**Publication is atomic, and that is the point.** Every member is written into a
staging directory first and the record becomes visible in exactly one
non-interruptible step, `os.replace`. Without that, a process killed mid-write
leaves a directory that *looks* satisfied: the runner skips the node, and the
study proceeds on a truncated checkpoint. That is a correctness defect in
operational clothing, which is why it is built in from the start rather than
retrofitted once the store holds real content.

Because a record is addressed by its content's identity, the same id must always
mean the same bytes. A second `put` of identical content is a no-op; a `put` of
*different* content under an existing id is refused, because it can only mean
identity derivation is broken and overwriting would hide that.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from uuid import uuid4
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..persistence.artifact_serializers import dense_state_dict
from .ids import ID_WIDTH, MeasurementID, ModelID
from .layout import (
    EVALUATION_LOG,
    EVALUATION_TRACE,
    MANIFEST,
    MEASUREMENTS_DIR,
    MODELS_DIR,
    PARTIAL,
    SAMPLES,
    STAGING_DIR,
    SYNTHESIS_LOG,
    TRAINING_HISTORY,
)

__all__ = [
    "MANIFEST",
    "PARTIAL",
    "ContentConflictError",
    "MeasurementRecord",
    "MeasurementStore",
    "ModelRecord",
    "ModelStore",
    "RecordNotFoundError",
    "StoreError",
]

_HEX = frozenset("0123456789abcdef")

#: Publication retries. Each retry means another worker won the rename; the
#: next pass reconciles against what they published. Three is ample -- the
#: window is a single rename, and every retry strictly reduces the work left.
_PUBLISH_ATTEMPTS = 3


class StoreError(Exception):
    """Base class for store failures."""


class RecordNotFoundError(StoreError, KeyError):
    """No complete record exists under the requested identifier."""


class NonFiniteRecordError(StoreError, ValueError):
    """A record carried a value that is not finite, so it was not published.

    Training diverges for reasons that are often *correct* -- an unstable plant
    under a box too tight to stabilise it will blow up, and control theory says
    so. What must not happen is that the blow-up is filed as a result: a `nan`
    cost is dropped silently by every later stage (`present/axis_scaling.py`
    filters non-finite points out of a figure), so a contender can vanish from
    a chapter without anything saying why.

    Refused here, at the one seam both record kinds pass through, and before
    anything is observable -- the same place and the same contract as the
    no-pickle refusal.

    **Magnitude is not judged.** A finite cost of 2.07e+19 is a number, and
    whether it is plausible is a study's declared gate to decide. Storage
    refuses what cannot be a number; a gate refuses what should not be
    believed.
    """


class ContentConflictError(StoreError):
    """A `put` presented content differing from what is already stored.

    Content addressing means one identifier denotes one content, so this can
    only mean identity derivation is broken. Raised rather than overwriting,
    because overwriting would destroy the evidence.
    """


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Canonical JSON: sorted keys and a fixed separator, so identical content
    yields identical bytes and the manifest digests are comparable."""
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_finite(record_id: str, payloads: Mapping[str, Any]) -> None:
    """Refuse a record carrying a non-finite number, naming where it is.

    Only numeric leaves are examined, and narrowly: `torch.isfinite` raises on
    some integer dtypes rather than answering `True`, and a samples frame
    legitimately carries string columns, so both are skipped rather than
    tested. A guard that turned a good record into a refusal would be worse
    than the gap it closes.

    Args:
        record_id: The identifier the record would have been filed under.
        payloads: Named payloads to examine -- scalar mappings, frames and
            tensor mappings, in any mixture.

    Raises:
        NonFiniteRecordError: Naming the payload, the key and the value.
    """
    for payload_name, payload in payloads.items():
        if payload is None:
            continue
        offenders: list[str] = []
        if isinstance(payload, pd.DataFrame):
            for column in payload.columns:
                series = payload[column]
                if not is_numeric_dtype(series):
                    continue
                # Pure pandas, deliberately: `.to_numpy()` is a conversion
                # primitive and `tests/architecture/test_boundaries.py` confines
                # those to an audited allowlist that this module is not on. The
                # guard has no business being the reason that list grows.
                if bool((series.isna() | series.isin([np.inf, -np.inf])).any()):
                    offenders.append(str(column))
        elif isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(value, torch.Tensor):
                    # `torch.isfinite` raises "unsupported tensor layout" on a
                    # sparse tensor, and the unfolded families carry sparse
                    # specification buffers (`_layer.P`, `_layer.q`, `_layer.A`)
                    # -- which `dense_state_dict` skips on write for the same
                    # reason. Asking the question of them would turn every
                    # unfolded model into a crash.
                    if (
                        value.layout is torch.strided
                        and value.is_floating_point()
                        and not torch.isfinite(value).all()
                    ):
                        offenders.append(str(key))
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        offenders.append(f"{key}={value}")
        if offenders:
            raise NonFiniteRecordError(
                f"{record_id}: {payload_name} carries a non-finite value in "
                f"{sorted(offenders)}, so the record is not published. A "
                "diverged run is a finding, not a result -- every later stage "
                "drops a non-finite value silently, so a contender would "
                "disappear from a figure with nothing saying why. Check the "
                "problem's stability against its control bound, the training "
                "rate, and the precision."
            )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def _write_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    save_file(dense_state_dict(dict(tensors)), path)


def _write_text(path: Path, text: str) -> None:
    """A text member. Callers skip an **empty** one rather than writing it: a
    zero-byte `synthesis.log` is indistinguishable on disk from a phase whose
    log was lost, and reads as "captured, and it had nothing to say"."""
    path.write_text(text, encoding="utf-8")


def _read_frame(path: Path) -> pd.DataFrame | None:
    """A payload frame, or `None` where the record carries none.

    Absence is the answer for every record published before the member existed
    (Annex 02 §2.2) -- 162 of them today -- so it is a normal state and never an
    error.
    """
    return pd.read_parquet(path) if path.is_file() else None


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


@dataclass(frozen=True)
class ModelRecord:
    """One trained or solved model, as it is persisted.

    Attributes:
        model_id: The identifier this record is filed under.
        spec: The full specification that produced the model.
        weights: The checkpoint's tensors; empty for an analytic contender that
            learns nothing.
        history: Per-epoch training history, or `None` for a family with no
            training phase.
        log: The offline phase's captured log, or `None` where none was
            captured.
        provenance: Package version, stamp, hardware, wall-clock, peak memory.
    """

    model_id: ModelID
    spec: Mapping[str, Any]
    weights: Mapping[str, torch.Tensor]
    history: pd.DataFrame | None
    log: str | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class MeasurementRecord:
    """One evaluation of one model.

    Attributes:
        measurement_id: The identifier this record is filed under.
        spec: Model reference plus evaluation problem and protocol.
        metrics: Scalar results.
        samples: Per-trajectory costs and derived per-step series, or `None`.
        trace: Per-batch cost and wall-clock of the online pass, or `None`.
        log: The online phase's captured log, or `None` where none was
            captured.
    """

    measurement_id: MeasurementID
    spec: Mapping[str, Any]
    metrics: Mapping[str, float]
    samples: pd.DataFrame | None
    trace: pd.DataFrame | None
    log: str | None


RecordT = TypeVar("RecordT")


class _ContentStore(Generic[RecordT]):
    """Shared machinery: atomic publication, integrity, enumeration.

    Subclasses declare their subdirectory, their required members, and how a
    record is written and read.
    """

    #: Subdirectory of the store root holding this kind of record.
    kind: str
    #: Members that must all be present for a directory to count as a record.
    required: tuple[str, ...]

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.records_root = self.root / self.kind
        self.staging_root = self.root / STAGING_DIR / self.kind

    # -- paths ------------------------------------------------------------

    def path(self, record_id: str) -> Path:
        """The directory a record is (or would be) stored at."""
        return self.records_root / record_id

    # -- queries ----------------------------------------------------------

    def exists(self, record_id: str) -> bool:
        """Whether a *complete* record exists.

        Every required member must be present. A directory holding only a
        `.partial/` resume artifact, or one missing a member because a write was
        torn, reads as absent -- otherwise the runner would skip work that was
        never finished.
        """
        directory = self.path(record_id)
        return all((directory / member).is_file() for member in self.required)

    def list_ids(self) -> Iterator[str]:
        """Every complete record's identifier.

        Directories whose names are not well-formed identifiers are ignored, so
        staging leftovers and stray files never masquerade as records.
        """
        if not self.records_root.is_dir():
            return
        for child in sorted(self.records_root.iterdir()):
            if not child.is_dir() or len(child.name) != ID_WIDTH:
                continue
            if not _HEX.issuperset(child.name):
                continue
            if self.exists(child.name):
                yield child.name

    def verify(self, record_id: str) -> tuple[str, ...]:
        """Re-hash every member and compare against the stored manifest.

        Returns:
            One human-readable problem per mismatching or missing member; empty
            when the record is intact.
        """
        directory = self.path(record_id)
        manifest_path = directory / MANIFEST
        if not manifest_path.is_file():
            return (f"{record_id}: {MANIFEST} is missing",)
        expected: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = []
        for member, digest in sorted(expected.items()):
            member_path = directory / member
            if not member_path.is_file():
                problems.append(f"{record_id}: {member} is missing")
            elif _sha256(member_path) != digest:
                problems.append(f"{record_id}: {member} fails its checksum")
        return tuple(problems)

    # -- mutation ---------------------------------------------------------

    def put(self, record: RecordT) -> None:
        """Write `record`, atomically.

        Members are built in a staging directory on the same filesystem, so the
        final `os.replace` is a rename rather than a copy and therefore atomic.
        Nothing is observable until it succeeds.

        Raises:
            ContentConflictError: If a record already exists under this
                identifier with different content.
            NonFiniteRecordError: If any numeric payload is not finite. Nothing
                is published in that case either.
            TypeError: If any part of the record is not serialisable under the
                no-pickle law. Nothing is published in that case.
        """
        record_id = self._id_of(record)
        _require_finite(record_id, self._numeric_payloads(record))
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.staging_root, prefix=f"{record_id}."))
        try:
            self._write(staging, record)
            _write_json(staging / MANIFEST, self._manifest(staging))
            self._publish(staging, record_id)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def delete(self, record_id: str) -> None:
        """Remove a record entirely. Absent records are not an error."""
        shutil.rmtree(self.path(record_id), ignore_errors=True)

    def get(self, record_id: str) -> RecordT:
        """Load a complete record.

        Raises:
            RecordNotFoundError: If no complete record exists.
        """
        if not self.exists(record_id):
            raise RecordNotFoundError(
                f"no complete record for {record_id} under {self.records_root}"
            )
        return self._read(self.path(record_id), record_id)

    def spec(self, record_id: str) -> dict[str, Any]:
        """The record's specification alone, without its heavy members.

        `get` deserialises the checkpoint and the history frame. A sweep that
        only needs to re-derive identifiers -- `mbl store verify` over the whole
        store -- would otherwise load every tensor it is about to discard.

        Raises:
            RecordNotFoundError: If no complete record exists.
        """
        if not self.exists(record_id):
            raise RecordNotFoundError(
                f"no complete record for {record_id} under {self.records_root}"
            )
        loaded: dict[str, Any] = json.loads(
            (self.path(record_id) / "spec.json").read_text(encoding="utf-8")
        )
        return loaded

    # -- internals --------------------------------------------------------

    def _manifest(self, staging: Path) -> dict[str, str]:
        return {
            member.name: _sha256(member)
            for member in sorted(staging.iterdir())
            if member.is_file() and member.name != MANIFEST
        }

    def _publish(self, staging: Path, record_id: str) -> None:
        """Make `staging` visible as `record_id`, or reconcile with what is there.

        `os.replace` on a directory requires the destination to be absent or an
        empty directory, so an existing record is reconciled by comparing
        manifests rather than replaced: identical content is a no-op, differing
        content is a conflict.
        """
        target = self.path(record_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        for remaining in reversed(range(_PUBLISH_ATTEMPTS)):
            if self.exists(record_id):
                self._require_same_content(staging, target, record_id)
                return
            try:
                os.replace(staging, target)
                return
            except OSError:
                # `exists()` and the rename are not one atomic step, so a
                # concurrent worker may publish in between -- and on POSIX,
                # renaming onto a non-empty directory fails with ENOTEMPTY
                # rather than overwriting. The next pass sees their record and
                # reconciles against it, which is the correct outcome for
                # content-addressed data.
                if not remaining:
                    raise
                self._clear_incomplete(target, record_id)

    def _clear_incomplete(self, target: Path, record_id: str) -> None:
        """Move an incomplete leftover out of the way, atomically.

        Deliberately **not** `rmtree(target)`: between observing "incomplete"
        and deleting, a concurrent worker can publish a complete record there,
        and a recursive delete would destroy it. `os.replace` onto a private
        name is a single atomic step, so losing the race is harmless -- the
        rename either moves the leftover we saw or fails, and either way the
        next pass re-evaluates from scratch.

        Any `.partial/` inside is discarded rather than rescued: it is a resume
        artifact for a training that has now finished, so by the time a complete
        record is being published it is stale by definition (Annex 02 §7.2).
        """
        if not target.exists() or self.exists(record_id):
            return
        aside = target.parent / f".superseded.{record_id}.{os.getpid()}.{uuid4().hex}"
        try:
            os.replace(target, aside)
        except OSError:
            return  # someone else cleared or replaced it; re-evaluate next pass
        shutil.rmtree(aside, ignore_errors=True)

    def _require_same_content(
        self, staging: Path, target: Path, record_id: str
    ) -> None:
        new = self._manifest(staging)
        stored = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
        if new != stored:
            differing = sorted(
                k for k in set(new) | set(stored) if new.get(k) != stored.get(k)
            )
            raise ContentConflictError(
                f"content for {record_id} differs from the stored record "
                f"in {differing}; one identifier must denote one content, so this "
                "indicates a defect in identity derivation rather than a race"
            )

    def _id_of(self, record: RecordT) -> str:
        raise NotImplementedError

    def _numeric_payloads(self, record: RecordT) -> Mapping[str, Any]:
        """The record's numeric payloads, by name, for `_require_finite`.

        Declared here so a record kind added later cannot be published without
        a subclass deciding what of it is a number -- the omission would
        otherwise be silent, which is the shape of defect this guard exists to
        stop.
        """
        raise NotImplementedError

    def _write(self, staging: Path, record: RecordT) -> None:
        raise NotImplementedError

    def _read(self, directory: Path, record_id: str) -> RecordT:
        raise NotImplementedError


class ModelStore(_ContentStore[ModelRecord]):
    """Models: `spec.json`, `weights.safetensors`, `provenance.json`, plus the
    payloads of Annex 02 §2.2 -- `training.parquet` where a training phase
    exists and `synthesis.log` where one was captured.

    **The payloads are not in `required`, and nothing may put them there.** A
    record published before a payload existed would then read as absent, and
    absent means recompute (`tests/store/test_payloads_are_optional.py`)."""

    kind = MODELS_DIR
    required = ("spec.json", "weights.safetensors", "provenance.json", MANIFEST)

    def _numeric_payloads(self, record: ModelRecord) -> Mapping[str, Any]:
        return {"weights": record.weights, "history": record.history}

    def _id_of(self, record: ModelRecord) -> str:
        return record.model_id

    def _write(self, staging: Path, record: ModelRecord) -> None:
        _write_json(staging / "spec.json", record.spec)
        _write_json(staging / "provenance.json", record.provenance)
        _write_tensors(staging / "weights.safetensors", record.weights)
        if record.history is not None:
            _write_frame(staging / TRAINING_HISTORY, record.history)
        if record.log:
            _write_text(staging / SYNTHESIS_LOG, record.log)

    def _read(self, directory: Path, record_id: str) -> ModelRecord:
        return ModelRecord(
            model_id=ModelID(record_id),
            spec=json.loads((directory / "spec.json").read_text(encoding="utf-8")),
            weights=load_file(directory / "weights.safetensors"),
            history=_read_frame(directory / TRAINING_HISTORY),
            log=_read_text(directory / SYNTHESIS_LOG),
            provenance=json.loads(
                (directory / "provenance.json").read_text(encoding="utf-8")
            ),
        )


class MeasurementStore(_ContentStore[MeasurementRecord]):
    """Measurements: `spec.json`, `metrics.json`, `samples.parquet` where
    per-trajectory detail was retained, plus the payloads of Annex 02 §2.2 --
    `evaluation.parquet` and `evaluation.log`.

    The same rule as `ModelStore`: no payload is ever required."""

    kind = MEASUREMENTS_DIR
    required = ("spec.json", "metrics.json", MANIFEST)

    def metrics(self, record_id: str) -> dict[str, float]:
        """The record's scalar metrics alone, without its heavy members.

        The same bargain `spec` strikes, for the other required member: a
        gate reading one saturated fraction per measurement would otherwise
        deserialise a per-trajectory frame of tens of thousands of rows, per
        measurement, to reach three floats -- 225 of them for one study.

        Raises:
            RecordNotFoundError: If no complete record exists.
        """
        if not self.exists(record_id):
            raise RecordNotFoundError(
                f"no complete record for {record_id} under {self.records_root}"
            )
        loaded: dict[str, float] = json.loads(
            (self.path(record_id) / "metrics.json").read_text(encoding="utf-8")
        )
        return loaded

    def _numeric_payloads(self, record: MeasurementRecord) -> Mapping[str, Any]:
        return {
            "metrics": record.metrics,
            "samples": record.samples,
            "trace": record.trace,
        }

    def _id_of(self, record: MeasurementRecord) -> str:
        return record.measurement_id

    def _write(self, staging: Path, record: MeasurementRecord) -> None:
        _write_json(staging / "spec.json", record.spec)
        _write_json(staging / "metrics.json", record.metrics)
        if record.samples is not None:
            _write_frame(staging / SAMPLES, record.samples)
        if record.trace is not None:
            _write_frame(staging / EVALUATION_TRACE, record.trace)
        if record.log:
            _write_text(staging / EVALUATION_LOG, record.log)

    def _read(self, directory: Path, record_id: str) -> MeasurementRecord:
        return MeasurementRecord(
            measurement_id=MeasurementID(record_id),
            spec=json.loads((directory / "spec.json").read_text(encoding="utf-8")),
            metrics=json.loads(
                (directory / "metrics.json").read_text(encoding="utf-8")
            ),
            samples=_read_frame(directory / SAMPLES),
            trace=_read_frame(directory / EVALUATION_TRACE),
            log=_read_text(directory / EVALUATION_LOG),
        )
