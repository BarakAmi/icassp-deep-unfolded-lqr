"""Reading a PDF's fonts from the file (Annex 07 §7 gate (d), Phase A3).

Four mechanisms already forbid Type 3: §B.2 states it, `StyleProfile` sets
`pdf.fonttype: 42`, `write_figure_artifacts` opens the profile around `savefig`,
and `tests/present/test_profiles.py` asserts the result. All four are correct,
and three PDFs in the repository root carry Type 3 anyway -- because every one
of those assertions is made about a file the test has *just rendered*. They
verify the producer. Nothing verified the product, and a figure hand-exported
through a bare `plt.savefig` meets none of them.

The suite below is therefore built the other way round: it is handed files and
asked what they contain. Its own negative control is a PDF deliberately written
at `pdf.fonttype: 3`, so the reader has to distinguish two files that differ
only in the setting under test.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rc_context  # noqa: E402

from mbl.present.fonts import (  # noqa: E402
    TYPE_3,
    Type3FontError,
    pdf_font_subtypes,
    require_embedded_outlines,
    type3_offenders,
)
from mbl.present.profiles import profile_context, resolve_profile  # noqa: E402


def _figure(path: Path, fonttype: int | None) -> Path:
    """A figure with real text on it, written at a chosen embedding."""
    settings = {} if fonttype is None else {"pdf.fonttype": fonttype}
    with rc_context(settings):
        figure, axes = plt.subplots()
        axes.plot([1, 2, 3], [1, 4, 9], label="curve")
        axes.set_xlabel("depth")
        axes.legend()
        figure.savefig(path)
        plt.close(figure)
    return path


class TestItReadsWhatTheFileHolds:
    def test_a_profile_rendered_pdf_embeds_outlines(self, tmp_path: Path) -> None:
        with profile_context(resolve_profile("ieee-2col")):
            path = _figure(tmp_path / "good.pdf", None)
        subtypes = pdf_font_subtypes(path)
        assert TYPE_3 not in subtypes
        assert "CIDFontType2" in subtypes
        require_embedded_outlines(path)

    def test_a_type_3_pdf_is_seen_and_refused(self, tmp_path: Path) -> None:
        """The negative control, and the reason the reader exists.

        Same figure, same code path, one rcParam different -- which is exactly
        the difference between the store's PDFs and the repository root's.
        """
        path = _figure(tmp_path / "bad.pdf", 3)
        assert TYPE_3 in pdf_font_subtypes(path)
        with pytest.raises(Type3FontError, match="Type 3"):
            require_embedded_outlines(path)

    def test_a_subtype_hidden_in_a_compressed_stream_is_still_found(
        self, tmp_path: Path
    ) -> None:
        """The scope claim in the module docstring, asserted rather than said.

        A PDF may hold its objects in a deflate stream, and a reader that only
        scanned the raw bytes would report 'no Type 3' about a file full of
        them. This constructs the adversarial case directly: the marker exists
        ONLY inside the compressed stream.
        """
        payload = zlib.compress(b"<< /Type /Font /Subtype /Type3 >>")
        path = tmp_path / "compressed.pdf"
        path.write_bytes(b"%PDF-1.4\nstream\n" + payload + b"\nendstream\n")
        assert b"/Subtype" not in path.read_bytes().replace(payload, b"")
        assert TYPE_3 in pdf_font_subtypes(path)

    def test_a_file_with_no_readable_font_is_refused_not_passed(
        self, tmp_path: Path
    ) -> None:
        """'Found no Type 3' and 'found nothing' are different facts.

        A reader that cannot see inside a file has not established that the
        file is clean, and a gate that treats silence as a pass is the shape of
        defect this whole module exists to correct.
        """
        path = tmp_path / "opaque.pdf"
        path.write_bytes(b"%PDF-1.4\n% no font objects here\n")
        assert pdf_font_subtypes(path) == frozenset()
        with pytest.raises(Type3FontError, match="no font subtype"):
            require_embedded_outlines(path)

    def test_a_missing_file_is_not_silently_clean(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            pdf_font_subtypes(tmp_path / "absent.pdf")


class TestTheSubmissionProfileProducesASubmittablePdf:
    """A3 and A7 joined: render through the venue, then inspect the FILE.

    This is the assertion that did not exist. Every other font test in the
    repository renders and asserts on what it rendered; this one renders
    through `ieee-paper` -- the profile the ICASSP submission is typeset at --
    writes a PDF, and then reads that PDF back the way a submission checker
    would. It is the only test here that would have caught the three Type 3
    files in the repository root, because it is the only one whose subject is a
    file rather than a figure object.
    """

    def test_the_paper_profile_writes_embedded_outlines(self, tmp_path: Path) -> None:
        profile = resolve_profile("ieee-paper")
        with profile_context(profile):
            path = _figure(tmp_path / "submission.pdf", None)
        require_embedded_outlines(path)
        assert TYPE_3 not in pdf_font_subtypes(path)

    def test_a_bare_savefig_outside_the_profile_would_have_failed_it(
        self, tmp_path: Path
    ) -> None:
        """The control, and the exact mistake that made the paper's PDFs.

        The export notebook called `figure_from_artifacts` and then a bare
        `plt.savefig`. `pdf.fonttype` is read when `savefig` runs, so the save
        happening outside the profile is sufficient on its own to produce Type
        3 -- even from a figure that was drawn correctly.
        """
        profile = resolve_profile("ieee-paper")
        with profile_context(profile):
            figure, axes = plt.subplots()
            axes.plot([1, 2, 3], [1, 4, 9], label="curve")
            axes.legend()
        path = tmp_path / "bare.pdf"
        figure.savefig(path)  # outside the context, as the notebook did
        plt.close(figure)
        assert TYPE_3 in pdf_font_subtypes(path)

    def test_the_page_is_the_geometry_the_submission_is_typeset_at(self) -> None:
        profile = resolve_profile("ieee-paper")
        assert profile.figsize == (9.0, 5.0)
        assert profile.base_font_pt == 16.0
        assert profile.font_family == "serif"


class TestTheSweep:
    def test_it_reports_every_offender_and_not_merely_the_first(
        self, tmp_path: Path
    ) -> None:
        """Three files is one repair; three failures one at a time is three."""
        bad_one = _figure(tmp_path / "bad1.pdf", 3)
        with profile_context(resolve_profile("ieee-2col")):
            good = _figure(tmp_path / "good.pdf", None)
        bad_two = _figure(tmp_path / "bad2.pdf", 3)
        assert type3_offenders([bad_one, good, bad_two]) == (bad_one, bad_two)

    def test_a_clean_tree_reports_nothing(self, tmp_path: Path) -> None:
        with profile_context(resolve_profile("thesis")):
            paths = [_figure(tmp_path / f"f{index}.pdf", None) for index in range(3)]
        assert type3_offenders(paths) == ()
