"""The four-artifact output contract — Annex 03 §B.1.

    <id>.pdf            vector, for LaTeX inclusion
    <id>.png            raster at 600 dpi
    <id>.data.parquet   the exact values plotted
    <id>.spec.json      the declarative specification

**`<id>.data.parquet` is a copy of the analysis table** (§B.1.1), not a
reference to it. §A.6 and §B.1 disagreed about where the bytes live and the
slice resolved it: an analysis is re-runnable and **replaces in place**, so a
figure that referenced `analyses/<source>.parquet` would silently change what
it claims to have plotted the next time the analysis ran with a different
aggregate — and nothing in the four artifacts could detect it. The copy freezes
the data actually plotted beside the spec that plotted it, which is the
guarantee D6 rejected the pickle in order to keep.

**All four are written atomically, or none are.** A reader must never find a
PDF from one render beside a spec from another; the pair is a single statement
about a single table, exactly as the analysis's own pair is.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from .profiles import StyleProfile, profile_context
from .tables import render_tables

#: §B.1's raster resolution. Named, because `viz.style.io`'s own default is
#: 150 and a figure written at that would be a preview rather than an artifact.
RASTER_DPI = 600

#: The four suffixes, in the order §B.1 lists them. A tuple rather than four
#: literals so that "did every member land?" is one comparison.
FIGURE_SUFFIXES = (".pdf", ".png", ".data.parquet", ".spec.json")


@dataclass(frozen=True)
class FigureArtifacts:
    """Where one figure's four files went.

    Attributes:
        figure_id: The stem all four share.
        paths: The four paths, in `FIGURE_SUFFIXES` order.
    """

    figure_id: str
    paths: tuple[Path, ...]

    @property
    def pdf(self) -> Path:
        """The publication artifact."""
        return self.paths[0]

    @property
    def data(self) -> Path:
        """The exact values plotted."""
        return self.paths[2]


def write_figure_artifacts(
    figure: Figure,
    *,
    directory: Path,
    figure_id: str,
    table: pd.DataFrame,
    spec: Mapping[str, Any],
    profile: StyleProfile,
) -> FigureArtifacts:
    """Emit §B.1's four artifacts for one rendered figure.

    **The profile is required, and the write happens inside it.** matplotlib
    reads `pdf.fonttype` when `savefig` runs, not when the figure is built, so
    a caller that rendered inside `profile_context` and wrote outside it
    produced a Type 3 PDF while every rcParam assertion passed. That is
    §B.2.1's own warning arriving in the code, and it was found by rendering
    the real NB04 figure rather than by any test — every test until then both
    rendered and wrote inside one context. Taking the profile here makes the
    guarantee structural: there is no way to write artifacts without it.

    Args:
        figure: The rendered figure. Not closed here — the caller owns it,
            because a caller that wanted to inspect it (the greyscale gate
            does) must be able to.
        directory: Where the four files go, created on demand.
        figure_id: Their shared stem.
        table: The analysis table, copied to `<id>.data.parquet` (§B.1.1).
        spec: The declarative specification, written to `<id>.spec.json`. Its
            `table` key, when present, also *produces* `<id>.table.<format>`
            (§B.1.2).
        profile: The style profile the figure was rendered at.

    Returns:
        The four paths. A declared table is written beside them and is
        deliberately absent from `paths`: the completeness check a notebook
        runs counts four, and a fifth required member "would mark every
        already-rendered figure incomplete and stop replay resolving until the
        whole store was re-rendered".

    Raises:
        ValueError: If `figure_id` is not a single path segment. It comes from
            an author-written document and becomes four filenames.
        TypeError: If `spec` is not JSON-serialisable — raised before anything
            is moved into place, so a previous render stands.
    """
    name = _segment(figure_id)
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=directory))
    try:
        # Everything is serialised into staging FIRST, so a spec that cannot
        # be written leaves the previous four files intact rather than three
        # of them replaced.
        with profile_context(profile):
            # NO CREATION DATE. Two renders of one figure differ in exactly
            # three bytes otherwise -- the clock inside `/CreationDate` -- and
            # nothing else, so the rendering is already deterministic and only
            # the stamp says otherwise. It matters because a figure that ships
            # in a paper's artifact repository is redrawn by the reviewer: with
            # the stamp, an identical redraw reports a modified file and the
            # reader cannot tell "the same" from "not the same" without opening
            # both. `None` omits the key rather than writing an empty one.
            figure.savefig(
                staging / f"{name}.pdf",
                bbox_inches="tight",
                metadata={"CreationDate": None},
            )
            figure.savefig(
                staging / f"{name}.png",
                dpi=RASTER_DPI,
                bbox_inches="tight",
                metadata={"Software": None},
            )
        table.to_parquet(staging / f"{name}.data.parquet", index=False)
        (staging / f"{name}.spec.json").write_text(
            json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Emitted HERE rather than passed in, and that is what makes the two
        # inseparable: the table is rendered from the very frame and the very
        # spec being written, in the same staging-then-replace operation, so
        # there is no call path that can write a spec declaring a table without
        # the table, or a table a version behind the figure beside it.
        tables = _presented_tables(table, spec)
        table_suffixes = tuple(f".table.{key}" for key in sorted(tables or {}))
        for suffix in table_suffixes:
            (staging / f"{name}{suffix}").write_text(
                (tables or {})[suffix.removeprefix(".table.")], encoding="utf-8"
            )
        paths = tuple(directory / f"{name}{suffix}" for suffix in FIGURE_SUFFIXES)
        for suffix, target in zip(FIGURE_SUFFIXES, paths):
            os.replace(staging / f"{name}{suffix}", target)
        for suffix in table_suffixes:
            os.replace(staging / f"{name}{suffix}", directory / f"{name}{suffix}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return FigureArtifacts(figure_id=name, paths=paths)


def _presented_tables(
    frame: pd.DataFrame, spec: Mapping[str, Any]
) -> dict[str, str] | None:
    """§B.1.2's presented table, or `None` when the figure declares none.

    Both inputs are the artifacts themselves — the frame about to become
    `<id>.data.parquet` and the spec about to become `<id>.spec.json` — which
    is clause 1 made structural rather than remembered. A rebuild reads those
    two files back and reaches this function with identical arguments, so the
    rebuilt table is the same bytes.
    """
    declaration = spec.get("table")
    if declaration is None:
        return None
    return render_tables(
        frame,
        declaration=dict(declaration),
        display_names=dict(spec.get("display_names", {})),
        declarations=dict(spec.get("declarations", {})),
    )


def read_figure_spec(directory: Path, figure_id: str) -> dict[str, Any]:
    """One figure's stored specification.

    The half of §B.1's promise that `mbl figure rebuild` stands on: the spec
    and the data are enough, with no store, no analysis and no study document
    in reach.

    Raises:
        FileNotFoundError: If the figure has not been rendered here.
    """
    path = directory / f"{_segment(figure_id)}.spec.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no figure {figure_id!r} under {directory}; render it before rebuilding it"
        )
    return dict(json.loads(path.read_text(encoding="utf-8")))


def read_figure_data(directory: Path, figure_id: str) -> pd.DataFrame:
    """One figure's stored table — the other half of the rebuild pair."""
    path = directory / f"{_segment(figure_id)}.data.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"figure {figure_id!r} under {directory} has no data artifact; "
            "the four are written together, so this one is torn"
        )
    return pd.read_parquet(path)


def _segment(figure_id: str) -> str:
    """`figure_id`, if it is a single safe path segment."""
    if (
        not figure_id
        or figure_id in (".", "..")
        or "/" in figure_id
        or "\\" in figure_id
    ):
        raise ValueError(
            f"figure id {figure_id!r} is not a single path segment; it becomes "
            "four filenames, and a document is author-written text rather than "
            "a trusted path"
        )
    return figure_id
