"""Tier 6's registry: a kind name to the function that computes its table.

Annex 03 §A.6. An analysis is a **pure function from a set of measurements to
a tidy table**, registered by name and referenced from the study spec, and the
two rules in that sentence are the ones this module exists to make structural:

* **It never plots.** Rendering is Tier 7 and takes this table as its input.
* **It never reads a model.** An analysis that reached for weights would be
  one that could not run without the checkpoints still on disk — and the whole
  reason the store separates a `ModelID` from a `MeasurementID` is so that a
  figure survives the models being deleted.

Neither is enforceable by a type, so both are properties of the context this
module hands an analysis: it carries the study, the measurement store, and
nothing else. A `ModelStore` is simply not reachable from here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..spec.analysis import AnalysisSpec
from ..spec.errors import SpecificationError
from ..spec.study import StudySpec
from ..store.content_store import MeasurementStore


@dataclass(frozen=True)
class AnalysisContext:
    """Everything an analysis may see.

    Attributes:
        study: The resolved study, which is what says *what varies* — the
            sweep axes, the contender labels and their roles. The store holds
            numbers; only the study knows what they are numbers *of*.
        study_id: The `StudyID` the measurements are filed under.
        spec: The declaration being executed.
        measurements: The measurement store. **The only store handed over**:
            see the module docstring.
    """

    study: StudySpec
    study_id: str
    spec: AnalysisSpec
    measurements: MeasurementStore


@dataclass(frozen=True)
class AnalysisOutput:
    """One analysis, computed.

    Attributes:
        table: The tidy table — simultaneously the analysis result and the
            `.data.parquet` of any figure rendering it (Annex 03 §A.6).
        sidecar: The statistical declarations §A.4 requires every table to
            state: the aggregate, the interval kind and level, the
            aggregation order, the seed count and the trajectory count.
    """

    table: pd.DataFrame
    sidecar: Mapping[str, Any] = field(default_factory=dict)


#: What every analysis kind is.
AnalysisKind = Callable[[AnalysisContext], AnalysisOutput]

_REGISTRY: dict[str, AnalysisKind] = {}


def register_analysis(kind: str) -> Callable[[AnalysisKind], AnalysisKind]:
    """Register `kind` under its Annex 03 §A.6 name.

    Args:
        kind: The registry key a study's `AnalysisSpec.kind` names.

    Returns:
        The decorator.

    Raises:
        SpecificationError: If `kind` is already registered. Two analyses
            answering to one name is not an override, it is whichever module
            imported last -- and this project's kinds are declared in study
            documents that would then mean different things per import order.
    """

    def decorate(function: AnalysisKind) -> AnalysisKind:
        if kind in _REGISTRY:
            raise SpecificationError(
                f"analysis kind {kind!r} is already registered by "
                f"{_REGISTRY[kind].__module__}; a kind names one computation"
            )
        _REGISTRY[kind] = function
        return function

    return decorate


def resolve_analysis(kind: str) -> AnalysisKind:
    """The function `kind` names.

    Raises:
        SpecificationError: If nothing is registered under it, naming what is
            -- this is the surface an author is editing.
    """
    function = _REGISTRY.get(kind)
    if function is None:
        raise SpecificationError(
            f"no analysis is registered under kind {kind!r}; available: "
            f"{', '.join(sorted(_REGISTRY)) or '(none)'}"
        )
    return function


def registered_analyses() -> tuple[str, ...]:
    """Every registered kind, sorted."""
    return tuple(sorted(_REGISTRY))
