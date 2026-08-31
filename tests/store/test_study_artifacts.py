"""Acceptance tests for the study-artifact tree — slice Phase B, step B2a.

Written before the implementation. Annex 02 §2 already fixes the layout:

    store/studies/<StudyID>/
        analyses/*.parquet    tidy analysis tables
        figures/*             the four-artifact figure outputs (Annex 03)

**This tree is deliberately NOT a content store.** `models/` and
`measurements/` are keyed by a digest of what they are, so publishing the same
content twice is a no-op and publishing different content under one identifier
is a refusal. An analysis is keyed by a *name the author chose* and is expected
to be recomputed — a better interval, a corrected aggregate — over measurements
that did not change. Refusing the second write would make an analysis
un-rerunnable, which is the one property Annex 03 §A.6 exists to give it. So a
re-analysis **replaces**, and the test that matters is that it replaces
*atomically*: a reader must never see a parquet from one run beside a sidecar
from another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mbl.store.content_store import StoreError
from mbl.store.study_artifacts import StudyArtifactStore

STUDY = "0123456789abcdef"
OTHER = "fedcba9876543210"


def _table(value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"contender": ["a", "b"], "aggregate": [value, value + 1.0]})


def _sidecar(aggregate: str = "mean") -> dict[str, object]:
    return {"aggregate": aggregate, "interval": {"kind": None}}


class TestTheLayoutIsTheAnnexes:
    def test_an_analysis_lands_where_annex_02_says(self, tmp_path: Path) -> None:
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "cost_by_depth", _table(), _sidecar())

        assert (
            tmp_path / "studies" / STUDY / "analyses" / "cost_by_depth.parquet"
        ).is_file()
        assert (
            tmp_path / "studies" / STUDY / "analyses" / "cost_by_depth.json"
        ).is_file()

    def test_two_studies_do_not_share_a_directory(self, tmp_path: Path) -> None:
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "a", _table(1.0), _sidecar())
        store.put_analysis(OTHER, "a", _table(9.0), _sidecar())

        assert store.get_analysis(STUDY, "a").table["aggregate"][0] == 1.0
        assert store.get_analysis(OTHER, "a").table["aggregate"][0] == 9.0

    def test_a_figure_directory_is_reserved_beside_it(self, tmp_path: Path) -> None:
        # Phase C writes four artifacts per figure here. The path is fixed now
        # so that the two phases cannot disagree about where a study's outputs
        # live, which is the class of drift Annex 02 exists to prevent.
        store = StudyArtifactStore(tmp_path)
        assert store.figures_root(STUDY) == tmp_path / "studies" / STUDY / "figures"


class TestARoundTrip:
    def test_the_table_and_the_sidecar_come_back(self, tmp_path: Path) -> None:
        store = StudyArtifactStore(tmp_path)
        table, sidecar = _table(2.5), _sidecar("median")
        store.put_analysis(STUDY, "cost_by_depth", table, sidecar)

        loaded = store.get_analysis(STUDY, "cost_by_depth")

        pd.testing.assert_frame_equal(loaded.table, table)
        assert loaded.sidecar == sidecar

    def test_listing_names_every_analysis(self, tmp_path: Path) -> None:
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "b", _table(), _sidecar())
        store.put_analysis(STUDY, "a", _table(), _sidecar())

        assert sorted(store.list_analyses(STUDY)) == ["a", "b"]

    def test_listing_an_unknown_study_is_empty_rather_than_an_error(
        self, tmp_path: Path
    ) -> None:
        assert list(StudyArtifactStore(tmp_path).list_analyses(STUDY)) == []

    def test_reading_an_absent_analysis_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(StoreError, match="cost_by_depth"):
            StudyArtifactStore(tmp_path).get_analysis(STUDY, "cost_by_depth")


class TestReanalysisReplaces:
    def test_a_second_write_replaces_the_first(self, tmp_path: Path) -> None:
        """The deliberate difference from `models/` and `measurements/`.

        Those refuse a conflicting write, because their identifier IS their
        content. An analysis is named by its author and is *meant* to be
        recomputed over measurements that did not change; refusing would make
        it un-rerunnable, which is the property Annex 03 §A.6 gives it.
        """
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "a", _table(1.0), _sidecar("mean"))
        store.put_analysis(STUDY, "a", _table(7.0), _sidecar("median"))

        loaded = store.get_analysis(STUDY, "a")
        assert loaded.table["aggregate"][0] == 7.0
        assert loaded.sidecar["aggregate"] == "median"

    def test_a_replacement_leaves_no_stale_member(self, tmp_path: Path) -> None:
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "a", _table(1.0), _sidecar("mean"))
        store.put_analysis(STUDY, "a", _table(7.0), _sidecar("median"))

        directory = tmp_path / "studies" / STUDY / "analyses"
        assert sorted(p.name for p in directory.iterdir()) == ["a.json", "a.parquet"]

    def test_a_failed_write_leaves_the_previous_version_intact(
        self, tmp_path: Path
    ) -> None:
        """Atomicity, tested in the direction that matters.

        A reader must never see one run's parquet beside another run's
        sidecar. The sidecar is written second, so a sidecar that cannot be
        serialised is exactly the interleaving to force.
        """
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "a", _table(1.0), _sidecar("mean"))

        with pytest.raises((TypeError, ValueError, StoreError)):
            store.put_analysis(STUDY, "a", _table(7.0), {"bad": object()})

        loaded = store.get_analysis(STUDY, "a")
        assert loaded.table["aggregate"][0] == 1.0
        assert loaded.sidecar["aggregate"] == "mean"

    def test_no_staging_directory_survives_a_write(self, tmp_path: Path) -> None:
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "a", _table(), _sidecar())

        directory = tmp_path / "studies" / STUDY / "analyses"
        assert not [p for p in directory.iterdir() if p.name.startswith(".")]


class TestTheNameIsAPathSegment:
    @pytest.mark.parametrize("name", ["../escape", "a/b", "", "."])
    def test_a_name_that_is_not_one_segment_is_refused(
        self, tmp_path: Path, name: str
    ) -> None:
        # An analysis id comes from a study document, which is author-written
        # text. It becomes a filename, so it is validated as one rather than
        # trusted -- a `..` here would write outside the store.
        with pytest.raises(StoreError):
            StudyArtifactStore(tmp_path).put_analysis(STUDY, name, _table(), _sidecar())

    def test_a_study_id_that_is_not_an_identifier_is_refused(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(StoreError):
            StudyArtifactStore(tmp_path).put_analysis(
                "../escape", "a", _table(), _sidecar()
            )


class TestTheSidecarIsReadableWithoutPandas:
    def test_the_sidecar_is_plain_json(self, tmp_path: Path) -> None:
        # `tests/store/test_import_cost.py` forbids the CLI read path from
        # paying for a tensor library; the statistical declarations of §A.6
        # are what a reader most often wants, and they must be readable
        # without opening a parquet.
        store = StudyArtifactStore(tmp_path)
        store.put_analysis(STUDY, "a", _table(), _sidecar("median"))

        path = tmp_path / "studies" / STUDY / "analyses" / "a.json"
        assert json.loads(path.read_text(encoding="utf-8"))["aggregate"] == "median"
