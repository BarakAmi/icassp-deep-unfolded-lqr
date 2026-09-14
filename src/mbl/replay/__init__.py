"""Tier 8 — the replay seam, and the only `mbl` module a notebook imports.

D1: *replay-first notebooks*. All training and evaluation happens in the
headless runner; a notebook declares the study, verifies the store satisfies
it, and loads finished results. This package is the surface that makes that
true, and it is deliberately the whole of it — a notebook importing anything
else from this project is a notebook that can drift.

    from mbl.replay import load_study, resolve, render_protocol, render_provenance

    study = load_study("studies/box_lqr/depth_scaling.toml", tier="publication")
    display(Markdown(study.render_protocol()))        # Annex 04 §3, generated

    results = resolve(study, store="store")           # raises if incomplete
    display(Markdown(render_provenance(results)))     # Annex 04 §7, generated

**Nothing here trains.** Not "trains only when the store is empty" — the
execution layer is not reachable from these functions at all, which is asserted
by making `run_study` and contender resolution *detonate* and running the suite
anyway. `execute_missing` (D1's escape hatch, off by default) is deliberately
absent from this phase: it needs a home that is not a second copy of
`cli.run.execute_run`, and choosing that home is a layering decision better made
once a notebook actually exercises it (Stage 6 Phase B).

**Nothing below this tier may import it**, which
`tests/architecture/test_boundaries.py` enforces in both directions.
"""

from __future__ import annotations

from .errors import GateFailedError, StoreIncompleteError, UnknownArtifactError
from .loading import DEFAULT_TIER, LAUNCHER, LoadedStudy, load_study
from .rendering import (
    render_composed_figure,
    render_figure,
    render_gates,
    render_problem,
    render_protocol,
    render_provenance,
)
from .resolution import Completeness, Resolution, Stage, resolve, survey

__all__ = [
    "DEFAULT_TIER",
    "LAUNCHER",
    "Completeness",
    "GateFailedError",
    "LoadedStudy",
    "Resolution",
    "Stage",
    "StoreIncompleteError",
    "UnknownArtifactError",
    "load_study",
    "render_composed_figure",
    "render_figure",
    "render_gates",
    "render_problem",
    "render_protocol",
    "render_provenance",
    "resolve",
    "survey",
]
