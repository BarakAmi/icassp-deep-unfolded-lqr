"""What a PDF actually embeds, read from the file rather than from the writer.

**The gap this closes.** Annex 03 §B.2 forbids Type 3 fonts, IEEE and ACM
submission checkers reject them, `StyleProfile.rc_params` sets
`pdf.fonttype: 42`, and `write_figure_artifacts` opens the profile around
`savefig` so the setting is live at the moment that decides it. Four
mechanisms, all correct, and three PDFs in the repository root still carry Type
3 -- because every test guarding the rule asserts on a file it has *just
rendered*. They prove the writer is right. They say nothing about the files the
repository contains, and a figure hand-exported through a bare `plt.savefig`
never meets any of them.

So this reads the bytes of a file someone hands it. It is the difference
between verifying a producer and verifying a product, and it is the form a
publication gate has to take (Annex 07 §7 gate (d)): the thing shipped is the
thing checked.

**Scope, stated because the limit matters.** A PDF may hold its objects in a
compressed object stream, and the font subtypes are then unreadable without
inflating it. `pdf_font_subtypes` inflates every `FlateDecode` stream it finds
before looking, so a matplotlib PDF -- compressed or not -- reads correctly. A
file whose fonts are hidden some other way would read as carrying none, which
is why `require_embedded_outlines` refuses a PDF that appears to embed *no*
font at all rather than passing it: "no Type 3 found" and "nothing found" are
different facts, and only the first is good news.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "TYPE_3",
    "Type3FontError",
    "pdf_font_subtypes",
    "require_embedded_outlines",
    "type3_offenders",
]

#: The subtype §B.2 forbids. A Type 3 font is a bitmapped glyph procedure
#: rather than an embedded outline, so it neither scales nor extracts as text.
TYPE_3 = "Type3"

#: Font subtypes in a PDF's object stream. matplotlib at `pdf.fonttype: 42`
#: writes `/Type0` with a `/CIDFontType2` descendant -- the composite TrueType
#: form -- and at `3` writes `/Type3`.
_SUBTYPE = re.compile(rb"/Subtype\s*/(\w+)")

#: A deflate-compressed object stream, whose bytes hide the pattern above.
_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)


class Type3FontError(RuntimeError):
    """A PDF embeds a font form that a submission checker rejects."""


def pdf_font_subtypes(path: Path | str) -> frozenset[str]:
    """Every font subtype embedded in `path`.

    Args:
        path: A PDF file.

    Returns:
        The subtype names, e.g. `{"Type0", "CIDFontType2"}` for an embedded
        TrueType outline or `{"Type3"}` for the bitmapped form.

    Raises:
        FileNotFoundError: If `path` is not a file.
    """
    raw = Path(path).read_bytes()
    found = set(_SUBTYPE.findall(raw))
    for compressed in _STREAM.findall(raw):
        try:
            found |= set(_SUBTYPE.findall(zlib.decompress(compressed)))
        except zlib.error:
            # Not a deflate stream, or not one on its own -- image data and
            # content streams both land here. Nothing to read, nothing wrong.
            continue
    return frozenset(name.decode("ascii", "replace") for name in found)


def require_embedded_outlines(path: Path | str) -> None:
    """Refuse a PDF that carries Type 3, or that carries no font at all.

    Args:
        path: A PDF file.

    Raises:
        Type3FontError: If the file embeds a Type 3 font, or embeds no font
            subtype this can read -- the second because a reader that finds
            nothing has not established that there is nothing to find.
    """
    path = Path(path)
    subtypes = pdf_font_subtypes(path)
    if TYPE_3 in subtypes:
        raise Type3FontError(
            f"{path} embeds a Type 3 font, which Annex 03 §B.2 forbids and "
            "IEEE and ACM submission checkers reject. It was written by a "
            "`savefig` that ran outside `profile_context`, so matplotlib used "
            "its ambient `pdf.fonttype: 3`; render through "
            "`write_figure_artifacts`, or open the profile around the save."
        )
    if not subtypes:
        raise Type3FontError(
            f"{path} embeds no font subtype this reader can see. That is not "
            "the same as embedding no Type 3 font -- the objects may be "
            "compressed in a form it does not inflate -- so it is refused "
            "rather than passed."
        )


def type3_offenders(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    """Those of `paths` that embed a Type 3 font, in the order given.

    The sweep form, for a publication gate over a whole tree. It reports every
    offender rather than stopping at the first, because a list of three files
    is one repair and three failures one at a time is three.
    """
    return tuple(
        Path(path) for path in paths if TYPE_3 in pdf_font_subtypes(Path(path))
    )
