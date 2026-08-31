"""Does the store satisfy the study? — Annex 04 §1.4's second code cell.

The pipeline has three stages and so does this check. A study is complete when
every point it materialises has a **measurement**, every analysis it declares
has a **table**, and every figure it declares has its **four artifacts**. The
stages are checked in that order and the refusal names the *first* one that is
unsatisfied, with the command for that stage and no other — because "run the
study" is the wrong advice when the models are already there and only the
figure is missing, and a reader who follows it pays hours to learn that.

**A subsetted store is refused before incompleteness is even considered.** A
`smoke` run truncates the sweep axis, so its store is *complete* for the study
it ran; nothing about counting can catch it. Only the stamp can, and this is
the last place it could still reach a reader. The guard is
`require_analysable_measurement` — the analysis tier's own, reached rather than
reimplemented, so the two surfaces cannot drift into disagreeing about what is
analysable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..analysis.gates import GateOutcome, evaluate_gates
from ..present.artifacts import FIGURE_SUFFIXES
from ..spec.study import StudyPoint
from ..spec.tiers import require_analysable_measurement
from ..store.content_store import MeasurementStore, ModelStore
from ..store.location import default_store
from ..store.study_artifacts import StudyArtifactStore
from .errors import GateFailedError, StoreIncompleteError
from .loading import LoadedStudy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..store.study_artifacts import StoredAnalysis

#: How many missing items a refusal lists before it summarises. A reader
#: fixing a study needs to recognise *what kind* of thing is absent, not to
#: read 135 identifiers.
LISTED = 5

#: The analysis kind whose tidy table the post-hoc gates are evaluated over.
COST_ANALYSIS_KIND = "cost_vs_axis"

#: The metric `constraint_binds` reduces (Annex 03 §A.3.1a). Absent from every
#: measurement written before it was retained, and from every measurement of a
#: problem that declares no box -- both of which the gate reports as not
#: evaluable rather than as passed.
SATURATION_METRIC = "eval_saturation_fraction"


class Stage(StrEnum):
    """The pipeline stage a refusal points at, and its command verb."""

    RUN = "run"
    ANALYSE = "analyse"
    FIGURE = "figure render"


@dataclass(frozen=True)
class Completeness:
    """What the store holds, against what the study declares.

    Attributes:
        study: The study's declared id.
        tier: The tier it was resolved at.
        points: How many points it materialises.
        models_held: How many of those points' models the store has.
        measurements_held: How many of those points' measurements it has.
        analyses: Declared analysis ids, and whether each is present.
        figures: Declared figure ids, and whether each is present.
        missing: One human-readable line per absent item, in declaration order.
        stage: The first unsatisfied stage, or `None` when complete.
    """

    study: str
    tier: str
    points: int
    models_held: int
    measurements_held: int
    analyses: Mapping[str, bool]
    figures: Mapping[str, bool]
    missing: tuple[str, ...]
    stage: Stage | None

    @property
    def complete(self) -> bool:
        """`True` when nothing the study declares is absent."""
        return self.stage is None


@dataclass(frozen=True)
class Resolution:
    """A verified study, and everything it produced.

    Annex 04 §1.4 writes `results = resolve(study)`, so resolution *is* the
    load: a notebook that verified and then loaded separately would have a cell
    between the two in which the check has passed and the data is not yet in
    hand, which is exactly where a reader inserts something that trains.

    Attributes:
        loaded: The study, its tier and its overrides.
        store: The store root this was resolved against. Carried rather than
            re-derived from an artifact path: §7's provenance section needs the
            producer's stamp, which lives in a stored record, and recovering
            the root by walking `parents[3]` off a figure would be a hidden
            dependency on the store's directory depth.
        completeness: What was found. Always complete, or this does not exist.
        gates: One outcome per declared gate, in declaration order. Always
            non-blocking, or this does not exist -- Annex 04 §4 requires a
            failed gate to stop the notebook, so `resolve` raises rather than
            handing back a verdict a cell could forget to read.
        analyses: Every declared analysis, by id.
        figures: Every declared figure's artifact paths, by id then suffix.
    """

    loaded: LoadedStudy
    store: Path
    completeness: Completeness
    gates: tuple[GateOutcome, ...]
    analyses: Mapping[str, StoredAnalysis]
    figures: Mapping[str, Mapping[str, Path]]


def _describe(point: StudyPoint) -> str:
    """One missing point, named the way a reader recognises it."""
    depth = ", ".join(
        f"{k.rsplit('.', 1)[-1]}={v}" for k, v in point.axis_values.items()
    )
    where = f" ({depth})" if depth else ""
    return (
        f"{point.contender.resolved_label}{where} seed {point.seed}"
        f" -> {point.measurement_id}"
    )


def _figure_artifacts(root: Path, figure_id: str) -> dict[str, Path]:
    """The artifacts present for one figure, keyed by suffix without its dot.

    `FIGURE_SUFFIXES` carries the leading dot (`".data.parquet"`), so the path
    is `stem + suffix` and never `stem + "." + suffix` — the latter produced
    `fig..pdf`, found by the first run of this phase's own suite rather than by
    reading, which is why the constant is used instead of four literals.
    """
    return {
        suffix.lstrip("."): root / f"{figure_id}{suffix}"
        for suffix in FIGURE_SUFFIXES
        if (root / f"{figure_id}{suffix}").is_file()
    }


#: The keys `_figure_artifacts` yields for a complete figure.
FIGURE_KEYS = frozenset(suffix.lstrip(".") for suffix in FIGURE_SUFFIXES)


def _refuse_subsetted(
    measurements: MeasurementStore, points: Sequence[StudyPoint]
) -> None:
    """Apply the analysis tier's own guard to every stored measurement.

    Checked before completeness, because a subsetted store is not incomplete —
    it is complete for a study that truncated its own axis, so counting can
    never see it.
    """
    for point in points:
        if not measurements.exists(point.measurement_id):
            continue
        spec: Mapping[str, Any] = measurements.get(point.measurement_id).spec
        provenance = spec.get("provenance", {})
        require_analysable_measurement(
            provenance if isinstance(provenance, Mapping) else {}
        )


def survey(loaded: LoadedStudy, *, store: Path | str | None = None) -> Completeness:
    """Report what the store holds, without raising for what it does not.

    The inspection half, kept public so a notebook can *show* a reader how far
    along a study is rather than only failing. `resolve` is what refuses.

    Args:
        loaded: The resolved study.
        store: The store root, or `None` for the command line's own resolution
            (`$MBL_STORE`, else `./store`).

    Returns:
        The completeness report.
    """
    root = default_store() if store is None else Path(store)
    study = loaded.study
    points = study.materialise()
    models, measurements = ModelStore(root), MeasurementStore(root)
    artifacts = StudyArtifactStore(root)
    study_id = str(study.study_id)

    absent = [p for p in points if not measurements.exists(p.measurement_id)]
    models_held = sum(1 for p in points if models.exists(p.model_id))

    analyses_root = artifacts.analyses_root(study_id)
    analyses = {
        spec.id: (analyses_root / f"{spec.id}.parquet").is_file()
        for spec in study.analyses
    }
    figures_root = artifacts.figures_root(study_id)
    figures = {
        spec.id: set(_figure_artifacts(figures_root, spec.id)) == FIGURE_KEYS
        for spec in study.figures
    }

    missing: list[str] = []
    stage: Stage | None = None
    if absent:
        stage = Stage.RUN
        missing += [_describe(p) for p in absent]
    elif not all(analyses.values()):
        stage = Stage.ANALYSE
        missing += [f"analysis {name!r}" for name, ok in analyses.items() if not ok]
    elif not all(figures.values()):
        stage = Stage.FIGURE
        missing += [f"figure {name!r}" for name, ok in figures.items() if not ok]

    return Completeness(
        study=study.id,
        tier=loaded.tier,
        points=len(points),
        models_held=models_held,
        measurements_held=len(points) - len(absent),
        analyses=analyses,
        figures=figures,
        missing=tuple(missing),
        stage=stage,
    )


def _gate_table(
    loaded: LoadedStudy, analyses: Mapping[str, StoredAnalysis]
) -> pd.DataFrame:
    """The tidy table the post-hoc gates read.

    The study's first `cost_vs_axis` analysis, chosen by **kind** rather than by
    position: a study declaring a second analysis first would otherwise hand the
    gate a table with no `aggregate` column and get a crash where it wanted a
    verdict. An empty frame when the study declares none, which the evaluator
    reports as not evaluable rather than as passed.
    """
    for spec in loaded.study.analyses:
        if spec.kind == COST_ANALYSIS_KIND and spec.id in analyses:
            return analyses[spec.id].table
    return pd.DataFrame(columns=["contender", "role", "axis_value", "aggregate"])


def _activity_table(
    loaded: LoadedStudy, measurements: MeasurementStore
) -> pd.DataFrame:
    """The constraint-activity table `constraint_binds` reads.

    Built from the **measurements** rather than from a declared analysis, and
    that is the point: a gate that only fires when a document remembers to
    declare a second analysis is the inert declaration this whole area exists
    to prevent. Every study that declares the gate gets it evaluated, on the
    measurements it already has.

    One row per `(contender, axis position)`, reduced the way §A.3 fixes:
    `eval_saturation_fraction` is already the mean over a seed's evaluation
    trajectories, so the second step is the mean over training seeds.

    An empty frame where the store carries no such statistic — which the
    evaluator reports as **not evaluable**, naming what is absent, rather than
    as passed.
    """
    rows: list[dict[str, Any]] = []
    for point in loaded.study.materialise():
        measurement_id = str(point.measurement_id)
        if not measurements.exists(measurement_id):
            continue
        metrics = measurements.metrics(measurement_id)
        if SATURATION_METRIC not in metrics:
            continue
        rows.append(
            {
                "contender": point.contender.resolved_label,
                "role": point.contender.role.value,
                # The axis POSITION, as a key rather than as a number: an axis
                # may be valued in whole specs, which have no float to be, and
                # the gate only ever groups by it.
                "axis_value": repr(sorted(point.axis_values.items(), key=str)),
                "seed": point.seed,
                "aggregate": float(metrics[SATURATION_METRIC]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["contender", "role", "axis_value", "aggregate"])
    frame = pd.DataFrame(rows)
    reduced: pd.DataFrame = frame.groupby(
        ["contender", "role", "axis_value"], as_index=False
    )["aggregate"].mean()
    return reduced.reset_index(drop=True)


def _refusal(loaded: LoadedStudy, report: Completeness, root: Path) -> str:
    """The message, which is this module's actual deliverable.

    **It names the store it consulted**, and that line was added because its
    absence cost the author an evening. They were told a study was 0 of 135
    measured and handed a command; they ran the command and were told
    everything already existed. Both messages were true, neither was
    actionable, and the reason was that the two were talking about different
    stores -- the notebook's working directory being the notebook's own. A
    refusal that says what is missing but not *where it looked* is unactionable
    exactly when the error is where it looked.
    """
    assert report.stage is not None  # only called when incomplete
    listed = list(report.missing[:LISTED])
    if len(report.missing) > LISTED:
        listed.append(f"... and {len(report.missing) - LISTED} more")
    held, total = report.measurements_held, report.points
    return "\n".join(
        [
            f"study {report.study!r} at tier {report.tier!r} is "
            f"{held}/{total} measured and its {report.stage.value} stage is "
            "not satisfied.",
            f"Store: {root.resolve()}",
            "Missing:",
            *(f"  {line}" for line in listed),
            "Run:",
            f"  {loaded.command(report.stage.value)}",
        ]
    )


def _run_stage(loaded: LoadedStudy, root: Path, stage: Stage) -> None:
    """One pipeline stage, reached rather than reimplemented.

    All three entry points sit below Tier 8, so this composes them and copies
    nothing. `cli.run.execute_run` is not reached and is not a layer beneath
    this one: what it adds over the producer is a parsed command line and a
    line of output, and a notebook has neither.

    Imported inside the function on purpose. Phase A's claim is that the
    execution layer is *unreachable* from this seam, and that claim survives
    exactly as far as an explicitly-passed flag that defaults to off — a
    module-level import would put the trainer, and torch's import cost, behind
    every notebook that never asked for either.
    """
    from ..analysis.runner import run_analyses
    from ..experiments.bindings import DEFAULT_SPEC_BINDINGS
    from ..present.runner import render_figures
    from ..runner import producer

    if stage is Stage.RUN:
        producer.run_study(
            loaded.study,
            store=root,
            bindings=DEFAULT_SPEC_BINDINGS,
            provenance=loaded.provenance,
        )
    elif stage is Stage.ANALYSE:
        run_analyses(loaded.study, store=root)
    else:
        render_figures(loaded.study, store=root)


def _produce(loaded: LoadedStudy, root: Path) -> None:
    """D1's escape hatch: produce what is absent, one stage at a time.

    **From the first unsatisfied stage and no earlier**, which is half the
    design. The ordinary reason a complete store stops satisfying a study is
    that a new analysis or figure was declared in the document; re-running the
    study then retrains nothing — the producer reuses by identifier — but it
    does pay the *evaluation* pass for every point, which for NB04 at
    `publication` is hours for numbers the store already held.

    The other half is that a stage which does **not** advance stops the
    pipeline, so the reader gets the refusal that names a command rather than a
    later stage's own error from mid-pipeline — precise, but naming nothing to
    run, and a notebook has no argv to experiment with. That falls out of
    re-surveying: a stage that did not advance leaves `pending` where it was,
    every later stage sees a mismatch and does nothing, and the refusal below
    is reached unchanged.

    An explicit stop was written here first, and mutation testing showed it
    survived its own removal — not a weak test but a **duplicate**, and the
    third time this project has answered a survivor by deleting code rather
    than by testing it.
    """
    for stage in (Stage.RUN, Stage.ANALYSE, Stage.FIGURE):
        pending = survey(loaded, store=root).stage
        if pending is None:
            return
        if pending is stage:
            _run_stage(loaded, root, stage)


def resolve(
    loaded: LoadedStudy,
    *,
    store: Path | str | None = None,
    execute_missing: bool = False,
) -> Resolution:
    """Verify the store satisfies the study, and load what it produced.

    Args:
        loaded: The resolved study.
        store: The store root, or `None` for the command line's own resolution
            (`$MBL_STORE`, else `./store`). Defaulted rather than required so
            that a notebook need not carry a directory name in a cell — a cell
            containing a path is a cell that goes stale, and it goes stale
            *silently*, because a store at the wrong path is empty rather than
            malformed.
        execute_missing: Annex 04 §1.4's escape hatch (D1), **off by default**.
            When `True`, whatever is absent is produced first, from the first
            unsatisfied stage onwards. It is not an override: if producing does
            not in fact satisfy the study, the refusal below is unchanged.

    Returns:
        The resolution, carrying every declared analysis and figure.

    Raises:
        SpecificationError: If any stored measurement was produced under a
            truncated sweep axis. Raised *before* completeness, because such a
            store is complete for the study that truncated itself.
        StoreIncompleteError: If any measurement, analysis or figure the study
            declares is absent. The message names the first unsatisfied stage
            and the command for it.
    """
    root = default_store() if store is None else Path(store)
    if execute_missing:
        _produce(loaded, root)
    _refuse_subsetted(MeasurementStore(root), loaded.study.materialise())

    report = survey(loaded, store=root)
    if not report.complete:
        raise StoreIncompleteError(_refusal(loaded, report, root))

    artifacts = StudyArtifactStore(root)
    study_id = str(loaded.study.study_id)
    figures_root = artifacts.figures_root(study_id)
    analyses = {
        spec.id: artifacts.get_analysis(study_id, spec.id)
        for spec in loaded.study.analyses
    }
    gates = evaluate_gates(
        loaded.study.gates,
        table=_gate_table(loaded, analyses),
        activity=_activity_table(loaded, MeasurementStore(root)),
    )
    blocking = [outcome for outcome in gates if outcome.blocking]
    if blocking:
        raise GateFailedError(
            "\n".join(
                [
                    f"study {loaded.study.id!r} declares {len(blocking)} gate(s) "
                    "that did not hold, so its results must not be read:",
                    *(f"  {o.kind.value}: {o.detail}" for o in blocking),
                ]
            )
        )
    return Resolution(
        loaded=loaded,
        store=root,
        completeness=report,
        gates=gates,
        analyses=analyses,
        figures={
            spec.id: _figure_artifacts(figures_root, spec.id)
            for spec in loaded.study.figures
        },
    )
