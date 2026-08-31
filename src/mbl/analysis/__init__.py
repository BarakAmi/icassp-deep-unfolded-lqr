"""Tier 6 — analyses: measurements to tidy tables (Annex 03 Part A).

An analysis is a **pure function from a set of `MeasurementRecord`s to a tidy
table**. It never plots — that is Tier 7 — and it never reads a model, which is
the property that lets a figure be rebuilt after the checkpoints are gone.

**Its input is the store, never a run.** Nothing here trains, evaluates, or
constructs a controller; an analysis does not care whether the measurements it
reads were produced a second ago or a month ago, and that is what makes a table
re-runnable without re-execution. It is also what makes the whole store worth
having.

This tier may import `core`, `store` and `spec` — the identifiers, the records
and the grammar. It may not be imported *by* any of them; `spec` is forbidden
from reaching it, statically and transitively, by
`tests/architecture/test_boundaries.py`.
"""

from __future__ import annotations

from .cost_by_category import cost_by_category
from .cost_vs_axis import cost_vs_axis
from .registry import (
    AnalysisContext,
    AnalysisKind,
    AnalysisOutput,
    register_analysis,
    registered_analyses,
    resolve_analysis,
)
from .runner import run_analyses

__all__ = [
    "AnalysisContext",
    "AnalysisKind",
    "AnalysisOutput",
    "cost_by_category",
    "cost_vs_axis",
    "register_analysis",
    "registered_analyses",
    "resolve_analysis",
    "run_analyses",
]
