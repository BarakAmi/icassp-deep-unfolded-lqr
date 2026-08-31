"""Tier 4 — the content-addressed result store.

Identity first (`ids`); the stores, index, semantic naming and CLI build on it.

**Exports are resolved lazily** (PEP 562). `content_store` needs PyTorch, pandas
and safetensors to read a checkpoint, which costs seconds of import; the index,
the identifier types and semantic naming need none of them. Eager re-exports
here would make `from mbl.store import StoreIndex` pay that cost, and the `mbl`
command line -- which reads the index and nothing else for most subcommands --
would be several seconds slower on every invocation. Names below resolve on
first access and are cached in the module namespace thereafter, so the public
surface is unchanged.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .content_store import (
        ContentConflictError,
        NonFiniteRecordError,
        MeasurementRecord,
        MeasurementStore,
        ModelRecord,
        ModelStore,
        RecordNotFoundError,
        StoreError,
    )
    from .ids import (
        ID_WIDTH,
        MeasurementID,
        ModelID,
        ProblemID,
        StudyID,
        measurement_id,
        model_id,
        problem_id,
        study_id,
    )
    from .index import (
        AmbiguousPrefixError,
        MeasurementRow,
        ModelRow,
        StoreIndex,
        UnknownModelError,
        UnknownPrefixError,
    )
    from .naming import (
        ContenderName,
        ProblemName,
        TrainingName,
        semantic_name,
        split_semantic_name,
    )

#: Which submodule each exported name lives in. Kept explicit rather than
#: searched, so a typo is an immediate `AttributeError` naming the symbol
#: instead of a silent import of the wrong module.
_ORIGIN: dict[str, str] = {
    "ContentConflictError": "content_store",
    "MeasurementRecord": "content_store",
    "MeasurementStore": "content_store",
    "ModelRecord": "content_store",
    "ModelStore": "content_store",
    "RecordNotFoundError": "content_store",
    "StoreError": "content_store",
    "AmbiguousPrefixError": "index",
    "MeasurementRow": "index",
    "ModelRow": "index",
    "StoreIndex": "index",
    "UnknownModelError": "index",
    "UnknownPrefixError": "index",
    "ContenderName": "naming",
    "ProblemName": "naming",
    "TrainingName": "naming",
    "semantic_name": "naming",
    "split_semantic_name": "naming",
    "ID_WIDTH": "ids",
    "MeasurementID": "ids",
    "ModelID": "ids",
    "ProblemID": "ids",
    "StudyID": "ids",
    "measurement_id": "ids",
    "model_id": "ids",
    "problem_id": "ids",
    "study_id": "ids",
}

# Spelled out rather than derived from `_ORIGIN`, so that a static reader --
# ruff, an IDE, a person -- can see the public surface without executing the
# module.
__all__ = [
    "ID_WIDTH",
    "AmbiguousPrefixError",
    "ContenderName",
    "ContentConflictError",
    "NonFiniteRecordError",
    "MeasurementID",
    "MeasurementRecord",
    "MeasurementRow",
    "MeasurementStore",
    "ModelID",
    "ModelRecord",
    "ModelRow",
    "ModelStore",
    "ProblemID",
    "ProblemName",
    "RecordNotFoundError",
    "StoreError",
    "StoreIndex",
    "StudyID",
    "TrainingName",
    "UnknownModelError",
    "UnknownPrefixError",
    "measurement_id",
    "model_id",
    "problem_id",
    "semantic_name",
    "split_semantic_name",
    "study_id",
]


def __getattr__(name: str) -> Any:
    origin = _ORIGIN.get(name)
    if origin is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{origin}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
