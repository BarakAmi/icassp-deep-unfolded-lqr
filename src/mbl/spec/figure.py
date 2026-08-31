"""What a study asks to be drawn (Annex 01 §2.6, Annex 03 Part B).

A figure is a pure function from an analysis table plus a specification to
rendered artifacts. It **never computes** and it never touches the store's
models, which is the property that makes `mbl figure rebuild <id> --style
ieee-2col` possible with no re-execution — and the reason D6 rejected the
matplotlib pickle in favour of a data-plus-spec pair.

`source` is what makes that work: it names the `AnalysisSpec` whose parquet is
simultaneously the analysis result and this figure's `.data.parquet`
(Annex 03 §A.6). The two are one object, so a figure whose source names
nothing is not a figure with a missing option — it is a figure with no data,
and it is refused where it is declared rather than where it would fail.

**A figure takes no part in any identifier**, for the reason recorded in
`analysis`: adding a plot must not cost a retraining.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from .analysis import require_serialisable_config
from .errors import SpecificationError


#: The tidy-table column §A.3.2 makes the figure's dispersion. Named here as a
#: literal rather than imported: `spec` is Tier 3 and the analysis tier that
#: emits the column is above it, so an import would be the boundary violation
#: the standing test refuses. A test asserts the two spellings agree.
SPREAD_COLUMN = "across_seed_spread"


@dataclass(frozen=True)
class FigureSpec:
    """One figure a study asks for.

    Attributes:
        id: What the figure is filed under. It is the `<id>` of the
            four-artifact contract — `<id>.pdf`, `<id>.png`,
            `<id>.data.parquet`, `<id>.spec.json` (Annex 03 §B.1).
        kind: The registry name of the renderer (``"axis_scaling"``,
            ``"learning_curves"``, ...). Resolved by Tier 7.
        source: The `AnalysisSpec.id` this figure renders. Validated against
            the study's declared analyses by `require_resolvable_sources`.
        config: Style overrides, reference-line selection, annotations. The
            *profile* is deliberately not among them: Annex 03 §B.2 makes the
            spec profile-independent so one declaration renders at every
            venue width.
        table: The presented table this figure carries (§B.1.2) — `columns`,
            `precision`, `caption` — or `None` for no table.

            **Optional at the artifact level and expected at the authoring
            level.** A figure declaring none emits exactly the four artifacts
            of §B.1, because a fifth *required* member "would mark every
            already-rendered figure incomplete and stop replay resolving until
            the whole store was re-rendered". What is not optional is the
            standard: a figure in a findings document without its table is
            incomplete work, and this makes declaring one a single block.
    """

    id: str
    kind: str
    source: str
    config: Mapping[str, Any] = field(default_factory=dict)
    table: Mapping[str, Any] | None = None

    def _require_dispersion_is_reported(self) -> None:
        """Annex 03 §A.3.2 rule 4: `dispersion = "none"` needs the table.

        Suppressing the MARK is a presentation decision, and one this standard
        now permits: on ICASSP Figure 1 four of nine contenders have exactly
        zero spread, four more draw a bar of half a pixel to three pixels, and
        the ninth draws one taller than the panel holding it -- a composition
        that tells a reader the other eight have no spread, which is false for
        four of them.

        Suppressing the QUANTITY is a different act, and rule 3 forbids it: the
        dispersion has to remain somewhere a reader can find it. So a figure
        that draws no bar must carry the spread in its own table, and the check
        is here rather than in the renderer because the renderer is handed the
        config without the table -- it could not see both halves of the
        condition even if it wanted to.

        Raises:
            SpecificationError: If the figure declares no dispersion and its
                table does not report the spread.
        """
        if str(self.config.get("dispersion", "")) != "none":
            return
        columns = tuple((self.table or {}).get("columns", ()))
        if SPREAD_COLUMN in columns:
            return
        raise SpecificationError(
            f'figure {self.id!r} declares dispersion = "none" but its table '
            f"does not carry {SPREAD_COLUMN!r}"
            + (" (it declares no table at all)" if self.table is None else "")
            + ". Annex 03 §A.3.2 permits a figure to draw no dispersion mark "
            "and does not permit it to drop the quantity: rule 3 keeps the "
            "spread reported wherever it is not drawn. Add the column to "
            "`[figures.table]`, or let the figure draw its bars"
        )

    def __post_init__(self) -> None:
        self._require_dispersion_is_reported()
        if not self.id:
            raise SpecificationError(
                "a figure needs an id; it is the `<id>` of the four artifacts "
                "of Annex 03 §B.1, and an unnamed figure has nowhere to be "
                "written and nothing for `mbl figure rebuild` to name"
            )
        if not self.kind:
            raise SpecificationError(
                f"figure {self.id!r} declares no kind; the kind is the "
                "registry key the renderer is resolved by, and an unnamed one "
                "renders nothing"
            )
        if not self.source:
            raise SpecificationError(
                f"figure {self.id!r} declares no source; a figure renders an "
                "analysis table and never computes one, so a figure without a "
                "source has no data rather than a missing option"
            )
        require_serialisable_config(self.config, f"figure {self.id!r}")
        if self.table is not None:
            # Refused where it is written rather than where it would fail: the
            # table is serialised into `<id>.spec.json`, and a value that
            # cannot be is a document defect, not a render-time surprise.
            require_serialisable_config(self.table, f"figure {self.id!r} table")

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the whole declaration, `source` included.

        `source` is signed because it is what the figure renders. Two figures
        over different tables must not record identical specifications, or the
        `.spec.json` stops being enough to rebuild either of them.
        """
        return {
            "type": type(self).__name__,
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "config": {key: self.config[key] for key in sorted(self.config)},
            # Signed for the same reason `config` is — `<id>.spec.json` has to
            # be enough to reproduce every artifact the render wrote, and the
            # table is one of them. It reaches no identifier: `store/ids.py`
            # strips `figures` by name, which is what makes adding a table to a
            # study cost no retraining.
            "table": (
                None
                if self.table is None
                else {key: self.table[key] for key in sorted(self.table)}
            ),
        }


def require_unique_figure_ids(figures: tuple[FigureSpec, ...], study_id: str) -> None:
    """Refuse a study declaring one figure id twice.

    The id is the stem of four artifacts. Two figures sharing one would
    overwrite each other's `.pdf`, `.png`, `.data.parquet` and `.spec.json`,
    leaving a set of four files that came from two different declarations.

    Args:
        figures: The declared figures.
        study_id: The study's name, for the message.

    Raises:
        SpecificationError: If any id appears more than once.
    """
    ids = [spec.id for spec in figures]
    repeated = sorted({name for name in ids if ids.count(name) > 1})
    if repeated:
        raise SpecificationError(
            f"study {study_id!r} declares the figures {', '.join(repeated)} "
            "more than once; a figure id is the stem of four artifacts, so two "
            "of them would overwrite each other file by file"
        )


def require_resolvable_sources(
    figures: tuple[FigureSpec, ...], analysis_ids: Collection[str], study_id: str
) -> None:
    """Refuse a figure naming an analysis the study does not declare.

    **A validator lives where it has the data**, and this one is handed the
    analysis *ids* rather than the specs: `figure` then needs nothing from
    `analysis` beyond the string, and the study — the only object that holds
    both collections — is where it is called from.

    Args:
        figures: The declared figures.
        analysis_ids: Every declared `AnalysisSpec.id`.
        study_id: The study's name, for the message.

    Raises:
        SpecificationError: If any `source` names no declared analysis.
    """
    available = sorted(analysis_ids)
    for spec in figures:
        if spec.source not in analysis_ids:
            raise SpecificationError(
                f"study {study_id!r} figure {spec.id!r} renders "
                f"{spec.source!r}, which it does not declare; the analysis "
                "table IS the figure's data artifact, so this figure has "
                f"nothing to draw. Declared analyses: "
                f"{', '.join(available) if available else '(none)'}"
            )
