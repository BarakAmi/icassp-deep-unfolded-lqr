"""`mbl figure` — a study's declared figures, rendered from its tables.

The last step of the slice: a document, through the store, to four artifacts a
manuscript can include. The same shape as `mbl run` and `mbl analyse` — the
decision is made here and returned as `(ok, message)`, because `cli/app.py` is
the only module on this surface allowed to `print`, and because `mbl.present`
pulls matplotlib and pandas and every *reading* command must stay able to
answer in milliseconds.

Two verbs, and the second is the one D6 exists to make possible:

**`render`** reads the study document, resolves it at a tier, and draws every
figure it declares from the analysis tables already in the store. It trains
nothing, evaluates nothing and analyses nothing.

**`rebuild`** re-renders one stored figure at another style profile from
`<id>.spec.json` and `<id>.data.parquet` **alone** — no study document, no
analysis, no model, and no `StudyID` from the caller. That is the capability a
pickled `Figure` never had, and the reason Annex 03 §B.1 emits a pair rather
than a pickle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec.errors import SpecificationError

#: What a study is rendered at when nothing says otherwise. The same default
#: `mbl run` and `mbl analyse` use, so the three agree without restating it.
DEFAULT_TIER = "standard"


@dataclass(frozen=True)
class FigureReport:
    """What rendering produced.

    Attributes:
        study: The study's declared id, or the figure id for a rebuild.
        profile: The style profile everything was rendered at.
        rows: `(figure_id, kind, pdf path)` per figure, in declaration order.
    """

    study: str
    profile: str
    rows: tuple[tuple[str, str, str], ...]

    def render(self) -> str:
        """One line per figure, plus a header naming the study and profile."""
        if not self.rows:
            return (
                f"{self.study}: declares no figures, so nothing was rendered. "
                "Add a [[figures]] table to the document (Annex 01 §2.6)"
            )
        lines = [f"{self.study} at style {self.profile}:"]
        lines += [
            f"  {figure_id} ({kind}) -> {path}" for figure_id, kind, path in self.rows
        ]
        return "\n".join(lines)


def render(  # noqa: PLR0913 -- one command's full identity, and each argument
    # is an independently-optional axis rather than a group that collapses:
    # WHICH document (study), WHERE its results are (store), at what EFFORT
    # (tier, catalogue, overrides -- the three editing levels of Annex 01
    # §2.3.1, which `mbl run` and `mbl analyse` also take separately), WHICH
    # figures (only) and at what VENUE (style). Bundling any of them would
    # make this command's surface differ from the two it must agree with.
    study: Path | str,
    *,
    store: Path,
    tier: str,
    catalogue: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    only: Sequence[str] | None = None,
    style: str | None = None,
) -> tuple[bool, str]:
    """`mbl figure render`, as a value.

    Returns:
        `(ok, message)`. `ok` is `False` for any specification failure or for
        a figure that fails the greyscale law — §B.4's "a figure that fails
        does not merge", reported where a person can act on it.
    """
    from ..present import render_figures
    from ..present.greyscale import GreyscaleError
    from .run import plan_run

    try:
        resolved, _ = plan_run(
            study, store=store, tier=tier, catalogue=catalogue, overrides=overrides
        )
        outcomes = render_figures(resolved.study, store=store, style=style, only=only)
    except (SpecificationError, GreyscaleError, KeyError) as error:
        return False, str(error).strip("'")
    return True, FigureReport(
        study=resolved.study.id,
        profile=outcomes[0].profile if outcomes else (style or "thesis"),
        rows=tuple(
            (outcome.figure_id, outcome.kind, str(outcome.artifacts.pdf))
            for outcome in outcomes
        ),
    ).render()


def rebuild(figure_id: str, *, store: Path, style: str | None) -> tuple[bool, str]:
    """`mbl figure rebuild`, as a value.

    Reads nothing but the figure's own two artifacts. A study document is not
    a parameter here, deliberately: the point of the pair is that it survives
    the document, the analysis and the models all being gone.
    """
    from ..present.greyscale import GreyscaleError
    from ..present.runner import rebuild_figure

    try:
        outcome = rebuild_figure(store=store, figure_id=figure_id, style=style)
    except (SpecificationError, GreyscaleError, KeyError, FileNotFoundError) as error:
        return False, str(error).strip("'")
    return True, FigureReport(
        study=figure_id,
        profile=outcome.profile,
        rows=((outcome.figure_id, outcome.kind, str(outcome.artifacts.pdf)),),
    ).render()
