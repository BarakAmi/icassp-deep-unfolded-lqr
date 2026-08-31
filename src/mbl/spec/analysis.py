"""What a study asks to be computed from its measurements (Annex 01 §2.6).

An analysis is a pure function from a set of `MeasurementRecord`s to a tidy
table, registered by name and referenced from the study — Annex 03 §A.6. It
never plots and it never reads a model: an analysis that reached for weights
would be one that could not run without the checkpoints still on disk, which
is precisely the property the store exists to remove.

**An analysis takes no part in any identifier**, and that is the whole reason
this type is separate from everything above it. A study's identity is what it
*computes*; an analysis is a derivation *from* what it computed. If declaring
one moved the `StudyID`, `store/studies/<StudyID>/` would become a new
directory, the manifest would be orphaned, and adding a plot would cost a
retraining. `store.ids.STUDY_KEYS_EXCLUDED_FROM_STUDY_ID` names the key, so
the exclusion lives in the same place as every other one rather than in this
module's silence.

The `kind` is a registry key, exactly as a contender's `family` is, and for
the same reason: adding an analysis kind must never touch this tier. What it
names is resolved by Tier 6, which is above this module and therefore cannot
be imported by it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import SpecificationError


@dataclass(frozen=True)
class AnalysisSpec:
    """One tidy table a study asks for.

    Attributes:
        id: What the table is filed under, within this study. It is the name
            a `FigureSpec.source` refers to and the stem of the emitted
            `analyses/<id>.parquet`.
        kind: The registry name of the analysis (``"cost_vs_axis"``,
            ``"paired_comparison"``, ...). Resolved by Tier 6.
        config: The kind's own parameters — which axis, which aggregate,
            which reference. Deliberately open, because the closed thing here
            is the *kind*, and its schema belongs with the analysis.
    """

    id: str
    kind: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise SpecificationError(
                "an analysis needs an id; it is what a figure's `source` "
                "names and what the emitted table is filed under, and an "
                "unnamed table can be neither referenced nor found"
            )
        if not self.kind:
            raise SpecificationError(
                f"analysis {self.id!r} declares no kind; the kind is the "
                "registry key the table is computed by, and an unnamed one "
                "computes nothing"
            )
        require_serialisable_config(self.config, f"analysis {self.id!r}")

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the whole declaration.

        Parameters are sorted for the reason `metrics`, `gates` and
        `DataSpec.params` are: two authors writing the same declaration in a
        different order must collide rather than diverge.
        """
        return {
            "type": type(self).__name__,
            "id": self.id,
            "kind": self.kind,
            "config": {key: self.config[key] for key in sorted(self.config)},
        }


def require_serialisable_config(config: Mapping[str, Any], where: str) -> None:
    """Refuse a parameter that cannot survive being written down.

    An analysis and a figure both emit their declaration to disk — the JSON
    sidecar of Annex 03 §A.6 and the `.spec.json` of §B.1 — and a
    specification that cannot be written is one whose figure cannot be
    rebuilt, which is the single capability D6 chose this format for.

    Refused here rather than at the first write, which is several frames and
    one whole phase away from the declaration that caused it.

    Args:
        config: The declaration's open parameter mapping.
        where: What to name in the message.

    Raises:
        SpecificationError: If any value is not JSON-serialisable.
    """
    for name, value in config.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError) as error:
            raise SpecificationError(
                f"{where} parameter {name!r} is not serialisable ({error}); a "
                "declaration that is written to a sidecar has to survive "
                "being written down"
            ) from error


def require_unique_analysis_ids(
    analyses: tuple[AnalysisSpec, ...], study_id: str
) -> None:
    """Refuse a study declaring one analysis id twice.

    The id is a filename and a reference target at once, so two analyses
    sharing one would have the second silently overwrite the first's table
    while every figure naming it rendered whichever ran last.

    Args:
        analyses: The declared analyses.
        study_id: The study's name, for the message.

    Raises:
        SpecificationError: If any id appears more than once.
    """
    ids = [spec.id for spec in analyses]
    repeated = sorted({name for name in ids if ids.count(name) > 1})
    if repeated:
        raise SpecificationError(
            f"study {study_id!r} declares the analyses {', '.join(repeated)} "
            "more than once; an analysis id is both a filename and what a "
            "figure's source names, so two of them make the output ambiguous"
        )
