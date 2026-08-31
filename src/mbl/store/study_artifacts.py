"""The per-study output tree (Annex 02 §2): analyses, and later figures.

    store/studies/<StudyID>/
        analyses/<id>.parquet   the tidy table
        analyses/<id>.json      the statistical declarations of Annex 03 §A.6
        figures/                the four-artifact figure outputs (Annex 03 §B.1)

**This is deliberately not a content store.** `models/` and `measurements/`
are keyed by a digest of what they are, so publishing the same content twice
is a no-op and publishing *different* content under one identifier is a
refusal — the property that makes a stored model trustworthy. An analysis is
keyed by a name its author chose, and is *meant* to be recomputed: a better
interval, a corrected aggregate, a heavier bootstrap, over measurements that
did not change at all. Refusing the second write would make an analysis
un-rerunnable, which is the single property Annex 03 §A.6 exists to give it.

So a re-analysis **replaces** — and the thing that must be true is that it
replaces *atomically*. The table and its sidecar are one statement about one
set of measurements; a reader that saw a parquet from one run beside a sidecar
from another would read a plausible and wrong pair, with nothing to detect it.
Both members are therefore assembled in a staging directory and moved into
place only once both exist.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .content_store import StoreError
from .ids import StudyID
from .layout import STUDIES_DIR

#: Subdirectory of a study holding its tidy tables.
ANALYSES_DIR = "analyses"

#: Subdirectory of a study holding its rendered figures.
FIGURES_DIR = "figures"


@dataclass(frozen=True)
class StoredAnalysis:
    """One analysis, read back.

    Attributes:
        analysis_id: The name it is filed under.
        table: The tidy table — simultaneously the analysis result and any
            figure's `.data.parquet` (Annex 03 §A.6). The two are one object.
        sidecar: The statistical declarations: aggregate, interval kind and
            level, aggregation order, seed and trajectory counts.
    """

    analysis_id: str
    table: pd.DataFrame
    sidecar: Mapping[str, Any]


class StudyArtifactStore:
    """Reads and writes `store/studies/<StudyID>/`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- paths ------------------------------------------------------------

    def study_root(self, study_id: str) -> Path:
        """The directory one study's outputs live under."""
        return self.root / STUDIES_DIR / _segment(study_id, "study id")

    def analyses_root(self, study_id: str) -> Path:
        """Where this study's tidy tables live."""
        return self.study_root(study_id) / ANALYSES_DIR

    def figures_root(self, study_id: str) -> Path:
        """Where this study's rendered figures live.

        Fixed here rather than in Tier 7, so the two phases cannot disagree
        about where a study's outputs go.
        """
        return self.study_root(study_id) / FIGURES_DIR

    # -- analyses ---------------------------------------------------------

    def list_analyses(self, study_id: str) -> Iterator[str]:
        """Every analysis id this study has a *complete* pair for.

        A directory holding only one of the two members is not an analysis:
        it is a torn write, and reporting it would offer a table whose
        declarations are missing, or declarations describing nothing.
        """
        directory = self.analyses_root(study_id)
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.parquet")):
            if path.with_suffix(".json").is_file():
                yield path.stem

    def put_analysis(
        self,
        study_id: str,
        analysis_id: str,
        table: pd.DataFrame,
        sidecar: Mapping[str, Any],
    ) -> None:
        """Write (or replace) one analysis, atomically in both members.

        Args:
            study_id: The `StudyID` this analysis is derived from.
            analysis_id: The `AnalysisSpec.id`. Becomes a filename, so it is
                validated as one path segment rather than trusted — it comes
                from an author-written document, and a `..` would write
                outside the store.
            table: The tidy table.
            sidecar: The statistical declarations.

        Raises:
            StoreError: If either identifier is not a single path segment.
            TypeError: If the sidecar is not JSON-serialisable. Raised before
                anything is moved into place, so the previous version stands.
        """
        directory = self.analyses_root(study_id)
        name = _segment(analysis_id, "analysis id")
        directory.mkdir(parents=True, exist_ok=True)

        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=directory))
        try:
            # Serialised into the staging directory FIRST, both of them, so a
            # sidecar that cannot be written leaves the previous pair intact
            # rather than half-replaced.
            table.to_parquet(staging / f"{name}.parquet", index=False)
            (staging / f"{name}.json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
            )
            for member in (f"{name}.parquet", f"{name}.json"):
                os.replace(staging / member, directory / member)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def get_analysis(self, study_id: str, analysis_id: str) -> StoredAnalysis:
        """One analysis, read back.

        Raises:
            StoreError: If either member is absent.
        """
        directory = self.analyses_root(study_id)
        name = _segment(analysis_id, "analysis id")
        parquet, sidecar = directory / f"{name}.parquet", directory / f"{name}.json"
        for path in (parquet, sidecar):
            if not path.is_file():
                raise StoreError(
                    f"study {study_id!r} has no analysis {analysis_id!r} "
                    f"({path.name} is absent); run the analysis before "
                    "reading it"
                )
        return StoredAnalysis(
            analysis_id=name,
            table=pd.read_parquet(parquet),
            sidecar=json.loads(sidecar.read_text(encoding="utf-8")),
        )


def _segment(value: str, what: str) -> str:
    """`value`, if it is a single safe path segment.

    A study id is a derived identifier and is checked as one; an analysis id
    is author-written text that becomes a filename, and is checked for the
    traversal it could otherwise perform.
    """
    if what == "study id":
        try:
            # The identifier type is the validator: reimplementing "sixteen
            # lowercase hex characters" here would be a second definition of
            # what a StudyID is, free to drift from the first.
            return str(StudyID(value))
        except ValueError as error:
            raise StoreError(
                f"{value!r} is not a store identifier ({error}); a study's "
                "outputs are filed under its StudyID, never under a name"
            ) from error
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise StoreError(
            f"{what} {value!r} is not a single path segment; it becomes a "
            "filename under the store root, and a document is author-written "
            "text rather than a trusted path"
        )
    return value
