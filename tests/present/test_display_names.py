"""A reader sees a display name; the machinery joins on a label.

Annex 01 §2.2.1 and Annex 04 §1.3. The defect these tests close is visible on
one screen of the tracked chapter: its authored table says **Truncated-Riccati**
and the figure two cells below it legends `truncated_riccati`.

The load-bearing property is not "the legend is prettier". It is that **only the
drawn text changes**: the label still selects the series, still keys the colour
registry, still groups the gates, still answers `--contender`. Every test below
that asserts a name has a sibling asserting an encoding did not move.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mbl.present.axis_scaling import axis_scaling
from mbl.present.encodings import encode_series, legend_label
from mbl.present.profiles import resolve_profile
from mbl.present.registry import FigureContext
from mbl.spec.contender import ContenderSpec, Role
from mbl.spec.errors import SpecificationError

#: The six names the tracked chapter's own table uses, verbatim.
NB04_DISPLAY = {
    "truncated_riccati": "Truncated-Riccati",
    "standard_pgd": "Standard-PGD",
    "unfolded_alpha": r"Unfolded-$\alpha$",
    "unfolded_alpha_p": r"Unfolded-$\alpha$+P",
    "cocp": "COCP",
    "cocp_lower_bound": "COCP-LB",
}

ORDER = tuple(NB04_DISPLAY)
ROLES = {label: Role.CONTENDER for label in ORDER}
ROLES["truncated_riccati"] = Role.BASELINE
ROLES["standard_pgd"] = Role.BASELINE
ROLES["cocp_lower_bound"] = Role.BOUND

#: The shape a raw identifier has, and the regex `tests/replay/test_notebook.py`
#: already refuses in a markdown cell. Reused rather than re-invented so the
#: rule cannot drift between prose and figures.
IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")


def _table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in ORDER:
        swept = label in {"standard_pgd", "unfolded_alpha", "unfolded_alpha_p"}
        # A depth-invariant contender carries a NaN axis value, and that is what
        # `axis_scaling` dispatches on — a flat reference is drawn by
        # `_draw_flat`, which is the only path that annotates a bound with its
        # value. Giving it a number instead produces a one-point curve and a
        # legend entry missing its annotation, which is how this fixture was
        # wrong on its first writing.
        for depth in (1.0, 8.0) if swept else (float("nan"),):
            rows.append(
                {
                    "contender": label,
                    "axis_value": float(depth),
                    "aggregate": 2.1 + 0.01 * len(rows),
                    "interval_low": float("nan"),
                    "interval_high": float("nan"),
                    "role": ROLES[label].value,
                    "n": 1,
                }
            )
    return pd.DataFrame(rows)


def _context(**overrides: Any) -> FigureContext:
    return FigureContext(
        figure_id="fig_cost_vs_depth",
        table=_table(),
        config={},
        profile=resolve_profile("thesis"),
        series_order=ORDER,
        roles=ROLES,
        **overrides,
    )


def _legend_texts(figure: Any) -> set[str]:
    axes = figure.axes[0]
    return {str(artist.get_label()) for artist in axes.get_lines()}


# --------------------------------------------------------------------------
# A2 / A5 — what the reader is shown
# --------------------------------------------------------------------------


def test_the_legend_says_the_declared_display_names() -> None:
    """Set equality, not `in`: a renderer that got one name right and left the
    other five raw would pass a membership check."""
    figure = axis_scaling(_context(display_names=NB04_DISPLAY))
    try:
        drawn = {text.split(" (")[0] for text in _legend_texts(figure)}
        assert drawn == set(NB04_DISPLAY.values())
    finally:
        plt.close(figure)


def test_without_display_names_the_legend_falls_back_to_the_label() -> None:
    """The fallback is deliberate (a study with no chapter may declare none),
    and this test is what makes it visible rather than assumed."""
    figure = axis_scaling(_context())
    try:
        drawn = {text.split(" (")[0] for text in _legend_texts(figure)}
        assert drawn == set(ORDER)
    finally:
        plt.close(figure)


def test_no_drawn_text_has_the_shape_of_an_identifier() -> None:
    """A5, with its hole stated rather than hidden.

    This is the same regex that refuses an implementation identifier in a
    markdown cell, so the rule cannot drift between prose and figures. **It
    cannot see a single-token label**: `cocp` carries no underscore and would
    pass here unchanged. That is why the test above compares the whole set, and
    why this check must never be presented as sufficient on its own.
    """
    figure = axis_scaling(_context(display_names=NB04_DISPLAY))
    try:
        offenders = [text for text in _legend_texts(figure) if IDENTIFIER.search(text)]
        assert not offenders, f"raw identifiers reached the legend: {offenders}"
    finally:
        plt.close(figure)


def test_the_bound_still_carries_its_value_after_the_name_changes() -> None:
    """§B.3.1's annotation survives the substitution — the value is appended to
    whatever name the series is drawn under."""
    figure = axis_scaling(_context(display_names=NB04_DISPLAY))
    try:
        annotated = [t for t in _legend_texts(figure) if t.startswith("COCP-LB (")]
        assert len(annotated) == 1
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------
# A8 — naming moves no encoding
# --------------------------------------------------------------------------


def test_the_encodings_are_identical_with_and_without_display_names() -> None:
    """The whole design in one assertion: colour, linestyle and marker are
    functions of the *label*, so renaming what a reader sees repaints nothing."""
    plain = encode_series(ORDER, ROLES)
    assert plain == encode_series(ORDER, ROLES)  # the comparison is meaningful

    bare = axis_scaling(_context())
    named = axis_scaling(_context(display_names=NB04_DISPLAY))
    try:

        def drawn(figure: Any) -> set[tuple[Any, Any, Any]]:
            return {
                (line.get_color(), line.get_linestyle(), line.get_marker())
                for line in figure.axes[0].get_lines()
            }

        assert drawn(bare) == drawn(named)
    finally:
        plt.close(bare)
        plt.close(named)


def test_legend_label_falls_back_per_series_not_per_figure() -> None:
    """A partial map names what it knows and leaves the rest joined — the state
    a study in the middle of being annotated is actually in."""
    partial = {"cocp": "COCP"}
    assert legend_label("cocp", None, Role.CONTENDER, partial) == "COCP"
    assert legend_label("unfolded_alpha", None, Role.CONTENDER, partial) == (
        "unfolded_alpha"
    )


# --------------------------------------------------------------------------
# A9 — the declaration refusals
# --------------------------------------------------------------------------


def test_a_display_name_beginning_with_an_underscore_is_refused() -> None:
    """Matplotlib hides such an artist from the legend AND §B.4's greyscale gate
    skips it as chrome, so the series would vanish from both with nothing
    raising."""
    with pytest.raises(SpecificationError, match="begins with '_'"):
        ContenderSpec(family="riccati", label="riccati", display="_hidden")


def test_a_declared_display_name_survives_to_resolved_display() -> None:
    spec = ContenderSpec(family="riccati", label="riccati", display="Riccati")
    assert spec.resolved_display == "Riccati"


def test_resolved_display_falls_back_to_the_label() -> None:
    spec = ContenderSpec(family="riccati", label="riccati")
    assert spec.resolved_display == "riccati"


# --------------------------------------------------------------------------
# A3 — the chapter and the figure cannot disagree
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# A4 — the rebuild reads the name from the artifact, never from the document
# --------------------------------------------------------------------------


class TestRebuiltFromArtifactsAlone:
    """`mbl figure rebuild` reads no study document by design, so a display name
    resolved only at render time would silently revert every rebuilt legend to
    raw labels — the same silent-degradation shape `spec.get("roles", {})` has.
    """

    @staticmethod
    def _prepared(tmp_path: Path) -> tuple[Path, Path]:
        from ..cli.test_figure import _prepare

        return _prepare(tmp_path)

    def test_the_stored_spec_carries_the_display_names(self, tmp_path: Path) -> None:
        import json

        from mbl.cli.app import main

        document, store = self._prepared(tmp_path)
        document.write_text(
            document.read_text().replace(
                'label = "unfolded_a"', "label = 'unfolded_a'\ndisplay = 'Unfolded-A'"
            )
        )
        assert main(["--store", str(store), "figure", "render", str(document)]) == 0

        figures = next((store / "studies").glob("*/figures"))
        spec = json.loads((figures / "fig_cost_vs_depth.spec.json").read_text())
        assert spec["display_names"]["unfolded_a"] == "Unfolded-A"

    def test_a_rebuild_uses_them_with_the_document_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The claim, tested where it can fail: the map reaching the renderer on
        a rebuild came out of `spec.json`, with the document gone from disk."""
        import shutil

        from mbl.cli.app import main
        import mbl.present.runner as runner

        document, store = self._prepared(tmp_path)
        document.write_text(
            document.read_text().replace(
                'label = "unfolded_a"', "label = 'unfolded_a'\ndisplay = 'Unfolded-A'"
            )
        )
        assert main(["--store", str(store), "figure", "render", str(document)]) == 0

        # Everything a render would have read, removed: the document, the
        # models and the analyses.
        shutil.rmtree(document.parent)
        shutil.rmtree(store / "models")
        for path in next((store / "studies").glob("*/analyses")).iterdir():
            path.unlink()

        seen: dict[str, Mapping[str, str]] = {}
        real = runner.resolve_figure

        def capturing(kind: str) -> Any:
            renderer = real(kind)

            def wrapped(context: FigureContext) -> Any:
                seen["display_names"] = dict(context.display_names)
                return renderer(context)

            return wrapped

        monkeypatch.setattr(runner, "resolve_figure", capturing)
        assert (
            main(
                [
                    "--store",
                    str(store),
                    "figure",
                    "rebuild",
                    "fig_cost_vs_depth",
                    "--style",
                    "ieee-2col",
                ]
            )
            == 0
        )

        assert seen["display_names"] == {
            "unfolded_a": "Unfolded-A",
            "baseline": "baseline",
        }
