"""Running a study's declared analyses into its output tree.

Thin by construction, exactly as `mbl run` is thin over the producer: resolve
each declared kind, hand it a context holding the measurement store and
nothing else, and write what it returns. Every decision that could be wrong
lives in the analysis or in the store, both of which are tested on their own.

**Nothing here trains, evaluates or builds a controller.** That is not a
convention -- there is no `ModelStore` and no recipe registry in scope, so the
alternative is unreachable from these imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..spec.analysis import AnalysisSpec
from ..spec.errors import SpecificationError
from ..spec.study import StudySpec
from ..store.content_store import MeasurementStore
from ..store.study_artifacts import StudyArtifactStore
from .registry import AnalysisContext, resolve_analysis


@dataclass(frozen=True)
class AnalysisOutcome:
    """What running one analysis produced.

    Attributes:
        analysis_id: The declaration's id, which is also the filename.
        kind: The registry kind that computed it.
        rows: How many rows the tidy table holds.
        path: Where the table was written.
    """

    analysis_id: str
    kind: str
    rows: int
    path: Path


def run_analyses(
    study: StudySpec,
    *,
    store: Path,
    only: Sequence[str] | None = None,
) -> tuple[AnalysisOutcome, ...]:
    """Compute every analysis `study` declares and write it into the store.

    Args:
        study: The resolved study. It is what says *what varies*; the store
            holds numbers, and only the study knows what they are numbers of.
        store: The store root.
        only: Analysis ids to run, or `None` for all of them.

    Returns:
        One outcome per analysis, in declaration order.

    Raises:
        SpecificationError: If `only` names an analysis the study does not
            declare, if a kind is unregistered, or if any analysis refuses --
            an absent measurement and a subsetted one both surface here.
    """
    selected = _selected(study, only)
    measurements = MeasurementStore(store)
    artifacts = StudyArtifactStore(store)
    study_id = str(study.study_id)

    outcomes: list[AnalysisOutcome] = []
    for spec in selected:
        output = resolve_analysis(spec.kind)(
            AnalysisContext(
                study=study,
                study_id=study_id,
                spec=spec,
                measurements=measurements,
            )
        )
        artifacts.put_analysis(study_id, spec.id, output.table, output.sidecar)
        outcomes.append(
            AnalysisOutcome(
                analysis_id=spec.id,
                kind=spec.kind,
                rows=int(len(output.table)),
                path=artifacts.analyses_root(study_id) / f"{spec.id}.parquet",
            )
        )
    return tuple(outcomes)


def _selected(study: StudySpec, only: Sequence[str] | None) -> tuple[AnalysisSpec, ...]:
    """The analyses to run, refusing a name the study does not declare.

    Refused rather than skipped: a caller naming an analysis that is not
    there has made a typo, and silently running nothing would report success
    for work that never happened.
    """
    if only is None:
        return study.analyses
    declared = {spec.id: spec for spec in study.analyses}
    unknown = sorted(set(only) - set(declared))
    if unknown:
        raise SpecificationError(
            f"study {study.id!r} declares no analyses {', '.join(unknown)}; "
            f"declared: {', '.join(sorted(declared)) or '(none)'}"
        )
    return tuple(declared[name] for name in only)
