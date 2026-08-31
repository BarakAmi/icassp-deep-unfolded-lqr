"""Rendering a study's declared figures into its output tree.

Thin, exactly as `analysis.runner` is: resolve the kind, hand it the analysis
table its `source` names, run the greyscale gate on what came back, and write
the four artifacts. Every decision that could be wrong lives in the renderer,
the gate or the writer, each tested on its own.

**Registration lives in the package `__init__`, not here.** Importing any
submodule executes `mbl/present/__init__.py`, which imports the kinds, so a
second import in this module would be decoration -- mutation testing removed
one and nothing failed, which is how it was found. The property itself (a
caller who imported only the runner can still resolve `axis_scaling`) is
pinned by a subprocess test, because within one interpreter the module is
always already imported.

**The order is the point.** The gate runs on the rendered figure *before*
anything is written, so a figure that would lose a series in print never
reaches the store — §B.4's "a figure that fails does not merge", one step
earlier than merging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..spec.contender import Role
from ..spec.errors import SpecificationError
from ..spec.figure import FigureSpec
from ..spec.study import StudySpec
from ..store.layout import STUDIES_DIR
from ..store.study_artifacts import FIGURES_DIR, StudyArtifactStore
from .artifacts import (
    FigureArtifacts,
    read_figure_data,
    read_figure_spec,
    write_figure_artifacts,
)
from .greyscale import require_greyscale_separable
from .profiles import StyleProfile, profile_context, resolve_profile
from .registry import FigureContext, resolve_figure
from .selection import select_series


@dataclass(frozen=True)
class FigureOutcome:
    """What rendering one figure produced.

    Attributes:
        figure_id: The declaration's id, which is also the four stems.
        kind: The registry kind that rendered it.
        profile: The style profile it was rendered at.
        artifacts: Where the four files went.
    """

    figure_id: str
    kind: str
    profile: str
    artifacts: FigureArtifacts


def render_figures(
    study: StudySpec,
    *,
    store: Path,
    style: str | None = None,
    only: Sequence[str] | None = None,
) -> tuple[FigureOutcome, ...]:
    """Render every figure `study` declares, into its own output tree.

    Args:
        study: The resolved study. It supplies the contender declaration
            order and the roles, which is what §B.3.1's "dropping a contender
            must not repaint the survivors" is computed from.
        store: The store root.
        style: A style profile name, or `None` for the default.
        only: Figure ids to render, or `None` for all.

    Returns:
        One outcome per figure, in declaration order.

    Raises:
        SpecificationError: If `only` names an undeclared figure, if a kind is
            unregistered, or if a figure's source analysis has not been run.
        GreyscaleError: If a rendered figure would lose a series in print.
    """
    profile = resolve_profile(style)
    selected = _selected(study, only)
    artifacts = StudyArtifactStore(store)
    study_id = str(study.study_id)
    roles = {spec.resolved_label: spec.role for spec in study.contenders}
    order = tuple(spec.resolved_label for spec in study.contenders)
    display_names = {
        spec.resolved_label: spec.resolved_display for spec in study.contenders
    }

    outcomes: list[FigureOutcome] = []
    for spec in selected:
        source = artifacts_analysis(artifacts, study_id, spec)
        # §B.1.2 clause 2: the subset is chosen BEFORE the frame is frozen, so
        # that "the exact values plotted" is true of the bytes and not only of
        # the intention. It was not — the renderer selected internally while
        # this froze the table it had been handed.
        table = select_series(source.table, spec.config)
        figure = resolve_figure(spec.kind)(
            FigureContext(
                figure_id=spec.id,
                table=table,
                config=spec.config,
                profile=profile,
                series_order=order,
                roles=roles,
                display_names=display_names,
            )
        )
        try:
            # Before anything is written: a figure that fails the print law
            # must not reach the store at all.
            require_greyscale_separable(figure, spec.id)
            stored = _stored_spec(spec, profile, study, source.sidecar)
            written = write_figure_artifacts(
                figure,
                directory=artifacts.figures_root(study_id),
                figure_id=spec.id,
                table=table,
                spec=stored,
                profile=profile,
            )
        finally:
            plt.close(figure)
        outcomes.append(
            FigureOutcome(
                figure_id=spec.id,
                kind=spec.kind,
                profile=profile.name,
                artifacts=written,
            )
        )
    return tuple(outcomes)


def _selected(study: StudySpec, only: Sequence[str] | None) -> tuple[FigureSpec, ...]:
    if only is None:
        return study.figures
    declared = {spec.id: spec for spec in study.figures}
    unknown = sorted(set(only) - set(declared))
    if unknown:
        raise SpecificationError(
            f"study {study.id!r} declares no figures {', '.join(unknown)}; "
            f"declared: {', '.join(sorted(declared)) or '(none)'}"
        )
    return tuple(declared[name] for name in only)


def artifacts_analysis(
    artifacts: StudyArtifactStore, study_id: str, spec: FigureSpec
) -> Any:
    """The stored analysis `spec.source` names — its table AND its sidecar.

    The sidecar is fetched here rather than the table alone because §B.1.2
    requires §A.4's declarations to travel with the table, and §B.1.1 requires
    a figure to rebuild with no analysis in reach — so they are frozen into
    `<id>.spec.json` at render time, which is the only moment both are
    available.

    Refused by name when it is absent: a figure renders an analysis and never
    computes one, so "run the analysis first" is the whole message.
    """
    try:
        return artifacts.get_analysis(study_id, spec.source)
    except Exception as error:  # noqa: BLE001 -- re-raised in the grammar's own vocabulary
        raise SpecificationError(
            f"figure {spec.id!r} renders analysis {spec.source!r}, which this "
            f"study has not produced ({error}); run `mbl analyse` first"
        ) from error


#: What §A.4 requires every table to state, taken from the analysis's sidecar.
#: Named rather than copied wholesale: the sidecar also carries the study id,
#: the row count and the config, which are provenance rather than declarations.
DECLARATION_KEYS = (
    "aggregate",
    "aggregation_order",
    "interval",
    "within_seed_spread_kind",
    "across_seed_spread_kind",
    "across_seed_spread_over",
    "n_training_seeds",
    "n_evaluation_trajectories",
)


def _stored_spec(
    spec: FigureSpec,
    profile: StyleProfile,
    study: StudySpec,
    sidecar: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """`<id>.spec.json` — enough to re-render and to re-style.

    The profile is recorded as what this render USED, never as what a rebuild
    must use: §B.2 makes the specification profile-independent, and a spec
    that pinned its own width would make `--style ieee-2col` a contradiction.
    """
    return {
        **spec.get_signature(),
        "study": study.id,
        "study_id": str(study.study_id),
        "rendered_at_profile": profile.name,
        "series_order": [contender.resolved_label for contender in study.contenders],
        "roles": {
            contender.resolved_label: contender.role.value
            for contender in study.contenders
        },
        # Written here for the same reason `roles` is: `rebuild_figure` reads no
        # study document by design, so a display name resolved only at render
        # time would silently revert every rebuilt legend to raw labels.
        "display_names": {
            contender.resolved_label: contender.resolved_display
            for contender in study.contenders
        },
        # §B.1.2: §A.4's declarations "live in the analysis's sidecar, while a
        # figure is required to be rebuildable with no analysis in reach — they
        # are therefore frozen into `<id>.spec.json` when the figure is
        # rendered". A table that had to re-read `analyses/<source>.json` to
        # state its own n would be the reference §B.1.1 refused, reintroduced
        # through the caption.
        "declarations": {
            key: (sidecar or {})[key]
            for key in DECLARATION_KEYS
            if key in (sidecar or {})
        },
    }


def figure_from_artifacts(spec_path: Path | str, *, style: str | None = None) -> Figure:
    """Re-render a stored figure from its two raw files, and hand it back.

    **The interactive half of `rebuild_figure`.** That function finds a figure
    by id, re-renders it and writes the four artifacts back in place, which is
    what a command line wants. This one is given the `<id>.spec.json` PATH,
    reads `<id>.data.parquet` beside it, and returns the live `Figure` without
    writing anything — which is what a notebook or a person poking at a
    rendering wants. Both go through the same loading and the same renderer, so
    a figure examined here is the figure that gets written.

    Taking the path rather than an id is the point: there is no store to
    search, no `StudyID` to know, and no ambiguity to refuse. Two studies may
    hold a figure of the same name, and `rebuild_figure` correctly declines to
    guess between them; here the caller has already said which by naming a
    file.

    Args:
        spec_path: Path to a figure's `<id>.spec.json`.
        style: A profile name, or `None` to re-render at the profile the spec
            records — so the default reproduces the stored artifact rather
            than silently restyling it.

    Returns:
        The rendered figure. **The caller owns it** and should `plt.close` it;
        matplotlib warns past twenty open figures.

    Raises:
        SpecificationError: If the spec or its data file is missing, or names
            a renderer that is not registered.
        GreyscaleError: If the figure would lose a series in print.
    """
    path = Path(spec_path)
    if not path.is_file():
        raise SpecificationError(
            f"no figure specification at {path}; a figure is re-rendered from "
            "its `<id>.spec.json` and the `<id>.data.parquet` beside it"
        )
    figure_id = path.name[: -len(".spec.json")]
    spec = read_figure_spec(path.parent, figure_id)
    table = read_figure_data(path.parent, figure_id)
    profile = resolve_profile(
        style or str(spec.get("rendered_at_profile") or "") or None
    )
    return _draw(figure_id, spec, table, profile)


def _draw(
    figure_id: str, spec: Mapping[str, Any], table: pd.DataFrame, profile: StyleProfile
) -> Figure:
    """One reading of a stored spec into a rendered figure.

    Shared by `rebuild_figure` and `figure_from_artifacts` so that the two
    cannot drift: a figure a person inspected and a figure the store holds
    must be the same figure.

    **The profile is opened here, not only handed on.** Passing it into the
    `FigureContext` gives a renderer the venue's colours and widths; it does
    nothing about the *page* -- the face, the type scale and the PDF font
    embedding live in matplotlib's rcParams, and a text object takes its family
    at the moment it is created. `axis_scaling` and `grouped_bars` each open
    `profile_context` themselves, so the omission was invisible in every figure
    but one: `severity_panels` did not, and Figure 2 rebuilt in matplotlib's
    ambient sans-serif at 10 pt instead of the venue's serif at 8 pt, exit code
    0 and no warning. Opening it here means the next renderer cannot repeat it.
    Nesting is harmless -- `rc_context` restores what it found.
    """
    with profile_context(profile):
        return resolve_figure(str(spec["kind"]))(
            FigureContext(
                figure_id=figure_id,
                table=table,
                config=dict(spec.get("config", {})),
                profile=profile,
                series_order=tuple(spec.get("series_order", ())),
                roles={
                    label: Role(value)
                    for label, value in dict(spec.get("roles", {})).items()
                },
                display_names=dict(spec.get("display_names", {})),
            )
        )


def rebuild_figure(
    *, store: Path, figure_id: str, style: str | None = None
) -> FigureOutcome:
    """Re-render one stored figure at another style profile.

    **This is D6's whole argument, executed.** A pickled `Figure` silently
    fails to load across matplotlib versions; a data-plus-spec pair does not,
    and it can also do something a pickle never could — *re-style*. So this
    reads `<id>.spec.json` and `<id>.data.parquet` and nothing else. No study
    document, no analysis, no model, and no `StudyID` supplied by the caller:
    the figure is found by looking for its own spec under the store.

    Args:
        store: The store root.
        figure_id: The figure's id — the stem of its four artifacts.
        style: The profile to re-render at, or `None` for the default.

    Returns:
        The outcome, with the four artifacts rewritten in place.

    Raises:
        SpecificationError: If no study in the store holds that figure, or if
            more than one does — an ambiguous id is a question for the author,
            not a coin toss.
        GreyscaleError: If the re-styled figure would lose a series in print.
    """
    profile = resolve_profile(style)
    directory = _locate(Path(store), figure_id)
    spec = read_figure_spec(directory, figure_id)
    table = read_figure_data(directory, figure_id)

    figure = _draw(figure_id, spec, table, profile)
    try:
        require_greyscale_separable(figure, figure_id)
        stored = {**spec, "rendered_at_profile": profile.name}
        written = write_figure_artifacts(
            figure,
            directory=directory,
            figure_id=figure_id,
            table=table,
            spec=stored,
            # §B.1.2 clause 1, executed: `write_figure_artifacts` re-emits the
            # table from `<id>.data.parquet` and `<id>.spec.json` and from
            # nothing else. No store, no analysis, no study document — which
            # is why the declarations had to be frozen into the spec.
            profile=profile,
        )
    finally:
        plt.close(figure)
    return FigureOutcome(
        figure_id=figure_id,
        kind=str(spec["kind"]),
        profile=profile.name,
        artifacts=written,
    )


def _locate(store: Path, figure_id: str) -> Path:
    """The `figures/` directory holding `figure_id`'s spec.

    Searched rather than derived, because a rebuild has no study document and
    therefore no `StudyID`. Two studies holding one figure id is refused: the
    author has to say which, and picking the first would make the answer
    depend on directory iteration order.
    """
    found = sorted(
        path.parent
        for path in (store / STUDIES_DIR).glob(f"*/{FIGURES_DIR}/{figure_id}.spec.json")
    )
    if not found:
        raise SpecificationError(
            f"no figure {figure_id!r} under {store}; render it before "
            "rebuilding it, and check the store root"
        )
    if len(found) > 1:
        raise SpecificationError(
            f"figure {figure_id!r} exists in {len(found)} studies "
            f"({', '.join(path.parent.name for path in found)}); an ambiguous "
            "id is a question for the author rather than a coin toss"
        )
    return found[0]
