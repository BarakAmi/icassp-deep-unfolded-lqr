"""Tier 7's registry: a figure kind to the function that renders it.

Annex 03 §B.6. A figure is a **pure function from an analysis table plus a
spec to rendered artifacts**. It never computes and it never touches the
store's models, and both are properties of the context this module hands a
renderer: a tidy table, a specification, and a style profile. There is no
store of any kind in reach, which is what makes `mbl figure rebuild <id>
--style ieee-2col` a re-render rather than a re-execution — and the reason D6
rejected the matplotlib pickle in favour of a data-plus-spec pair.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from ..spec.contender import Role
from ..spec.errors import SpecificationError
from .profiles import StyleProfile


@dataclass(frozen=True)
class FigureContext:
    """Everything a renderer may see.

    Attributes:
        figure_id: What the four artifacts are filed under.
        table: The analysis table — simultaneously the analysis result and
            this figure's `.data.parquet` (§A.6, §B.1.1).
        config: The `FigureSpec.config`: annotations, series selection,
            reference-line choices. **Never a width or a font size** (§B.2.1):
            those are the profile's, or "no figure is scaled after export"
            stops being enforceable one figure at a time.
        profile: The style profile to render at.
        series_order: Every contender the study declares, in declaration
            order. What §B.3.1's "a figure that drops a contender must not
            repaint the survivors" is computed from.
        roles: Each contender's `Role`, for §B.3.1's reserved encodings.
        display_names: What a reader is shown for each contender, keyed by the
            same label everything else joins on (Annex 01 §2.2.1). Absent keys
            fall back to the label, so a figure rendered from a study that
            declares none is unchanged.
    """

    figure_id: str
    table: pd.DataFrame
    config: Mapping[str, Any]
    profile: StyleProfile
    series_order: Sequence[str] = ()
    roles: Mapping[str, Role] = field(default_factory=dict)
    display_names: Mapping[str, str] = field(default_factory=dict)


#: What every figure kind is: a context in, a rendered `Figure` out. Writing
#: the artifacts is the runner's job, so a renderer cannot half-emit a bundle.
FigureKind = Callable[[FigureContext], Figure]

_REGISTRY: dict[str, FigureKind] = {}


def register_figure(kind: str) -> Callable[[FigureKind], FigureKind]:
    """Register `kind` under its Annex 03 §B.6 name.

    Raises:
        SpecificationError: If `kind` is already registered. Two renderers
            answering to one name is not an override, it is whichever module
            imported last — and kinds are named in study documents that would
            then mean different things per import order.
    """

    def decorate(function: FigureKind) -> FigureKind:
        if kind in _REGISTRY:
            raise SpecificationError(
                f"figure kind {kind!r} is already registered by "
                f"{_REGISTRY[kind].__module__}; a kind names one renderer"
            )
        _REGISTRY[kind] = function
        return function

    return decorate


def resolve_figure(kind: str) -> FigureKind:
    """The renderer `kind` names.

    Raises:
        SpecificationError: If nothing is registered under it, naming what is.
    """
    function = _REGISTRY.get(kind)
    if function is None:
        raise SpecificationError(
            f"no figure is registered under kind {kind!r}; available: "
            f"{', '.join(sorted(_REGISTRY)) or '(none)'}"
        )
    return function


def registered_figures() -> tuple[str, ...]:
    """Every registered kind, sorted."""
    return tuple(sorted(_REGISTRY))
