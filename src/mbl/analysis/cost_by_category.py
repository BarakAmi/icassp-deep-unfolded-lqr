"""`cost_by_category` — the same reduction against an axis of names (§A.6.1).

A swept axis whose values are *names* — an evaluation condition, a mismatch
experiment, a controller variant — is already expressible in this grammar and
already reaches identity: measured on `contenders.*.config.kind` with two
string values, a study materialises six points with six distinct `ModelID`s
and carries the value through as a Python string. What it did not survive is
the reduction, and it failed as a **crash**: `float("learned_step_size")` out
of the middle of `cost_vs_axis`.

**This is a second kind because the FIGURE differs, not the statistic.** The
reduction is shared with `cost_vs_axis` down to the last line — the same
grouping, the same replicate guard, the same per-seed aggregate, the same
three dispersions, the same BCa interval. What changes is what a reader may
conclude from the gap between two x positions: a category has no position, so
a line through two of them asserts a rate of change that does not exist.

**Any axis may be declared categorical, including a numeric one.** A bar chart
over integer depths needs no new kind and no new rule, and restricting this to
string-valued axes would make the renderer's reach depend on what the
campaign's axes happen to hold today — the "never design around a free
parameter" failure this project has a standing rule against.

**Order is the study's declared order.** Re-sorting by the measured aggregate
would make the figure's shape a function of its own data, so a re-run that
moved one number would reorder the axis and the two figures could no longer be
compared. `axis_value` therefore carries the category's *position* in the
declaration and `axis_label` carries its name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..spec.errors import SpecificationError
from .reduction import (
    Declared,
    across_seed_spread,
    aggregate_of,
    axis_key,
    axis_value_of,
    declared,
    grouped,
    interval_of,
    reduce_group,
    require_replicates,
    sidecar,
)
from .registry import AnalysisContext, AnalysisOutput, register_analysis

#: The column §A.6.1 adds so the analysis/figure pairing can be **checked**
#: rather than trusted. A categorical table rendered by `axis_scaling` would
#: draw a line through positions 0, 1, 2 and read as a trend — plausible, and
#: wrong.
AXIS_KIND_COLUMN = "axis_kind"

#: Its two values. **Absent means `NUMERIC`**, so every artifact written
#: before §A.6.1 existed still rebuilds under §B.1.1.
CATEGORICAL = "categorical"
NUMERIC = "numeric"


@register_analysis("cost_by_category")
def cost_by_category(context: AnalysisContext) -> AnalysisOutput:
    """Aggregate cost per contender, per category, with intervals.

    Args:
        context: The study, the measurement store and the declaration.

    Returns:
        The tidy table and its statistical sidecar. The schema is
        `cost_vs_axis`'s plus `axis_kind`, so the table emitter of §B.1.2 and
        every column §A.3.2 requires are unchanged.

    Raises:
        SpecificationError: If the declaration names no usable axis, if a
            point's measurement is absent from the store, if any measurement
            was produced under a subsetted run, or if a materialised point
            carries a category the study does not declare.
    """
    spec = declared(context.study, context.spec.config)
    # Annex 03 §A.6.1 (as amended): the projection and the per-position
    # reference normalisation belong to the numeric kind. This kind orders by
    # declared labels and normalises by an axis CATEGORY (`relative_to`), so a
    # declaration of either key here would read nothing — the
    # inert-declaration defect, refused by name.
    if spec.axis_property is not None:
        raise SpecificationError(
            f"axis_property is declared on a {CATEGORICAL!r} analysis, which "
            "orders by declared labels and reads no projection; the key "
            "belongs to cost_vs_axis (Annex 03 §A.6.1)"
        )
    if spec.relative_to_contender is not None:
        raise SpecificationError(
            f"relative_to_contender is declared on a {CATEGORICAL!r} "
            "analysis; this kind normalises by an axis category via "
            "relative_to, and the per-position contender normalisation "
            "belongs to cost_vs_axis (Annex 03 §A.6.1)"
        )
    # The whole ZIPPED GROUP is the category axis (Annex 01 §2.5): a plant
    # scored blind and aware is two categories, and keying on the plant alone
    # was measured to fold them into one group and refuse them as repeated
    # seeds. A product-composed second axis still lands in `others` inside
    # `require_replicates`, so the pooling refusal stays armed where pooling
    # is real.
    group_paths = context.study.zip_group_paths(spec.axis_path)
    groups = grouped(context.study, spec, group_paths=group_paths)
    require_replicates(groups, context.study, spec, group_paths=group_paths)
    names = _category_names(context, spec.axis_path, group_paths)
    reference = _reference(context.spec.config, names, spec.axis_path)
    rows: list[dict[str, Any]] = []

    for (label, key), points in groups.items():
        axis_value = axis_value_of(points, spec)
        reduced = reduce_group(context, points, spec)
        contender = points[0].contender
        rows.append(
            {
                "contender": label,
                "role": contender.role.value,
                "axis_path": spec.axis_path if axis_value is not None else "",
                # The POSITION, never the value: `cost_vs_axis` writes the value
                # itself here and that is exactly the difference between the two
                # kinds. A null one still means "this contender has no category",
                # which is what makes it a level rather than a bar.
                "axis_value": np.nan
                if axis_value is None
                else float(names.position(key)),
                "axis_label": "" if axis_value is None else names.of(key),
                AXIS_KIND_COLUMN: CATEGORICAL,
                "n_seeds": len(reduced.per_seed),
                "n_trajectories": reduced.trajectories,
                "aggregate": reduced.aggregate,
                "interval_low": reduced.interval[0],
                "interval_high": reduced.interval[1],
                "within_seed_spread": reduced.within_seed_spread,
                "across_seed_spread": reduced.across_seed_spread,
                "seeds": list(reduced.seeds),
                "per_seed_aggregate": [float(value) for value in reduced.per_seed],
            }
        )

    extra: dict[str, Any] = {}
    if reference is not None:
        rows = _as_degradation(rows, reference, spec)
        extra = {
            "relative_to": reference,
            "quantity_unit": "percent",
            "pairing": "training_seeds",
            "within_seed_spread_reason": (
                "a degradation is a ratio of two per-seed means, so it has no "
                "per-trajectory value to disperse; the absolute table carries "
                "the within-seed spread of each condition"
            ),
        }

    table = pd.DataFrame(rows).sort_values(
        ["contender", "axis_value"], na_position="first", ignore_index=True
    )
    return AnalysisOutput(
        table=table,
        sidecar=sidecar(
            context,
            spec,
            table,
            axis_kind=CATEGORICAL,
            # So a reader of the sidecar alone can tell what the positions mean
            # and in what order they were declared -- the one fact the table's
            # integers do not carry on their own.
            categories=list(names.ordered),
            **extra,
        ),
    )


# -- percentage degradation (the author's decision, 2026-08-07) -------------


def _reference(
    config: Mapping[str, Any], names: _Categories, axis_path: str
) -> str | None:
    """The category every other one is expressed against, or `None`.

    Raises:
        SpecificationError: If it names a category this axis does not declare.
            Named rather than defaulted, because a reference nobody declared
            would silently make the whole table absolute again.
    """
    declared_reference = config.get("relative_to")
    if declared_reference is None:
        return None
    wanted = str(declared_reference)
    if wanted not in names.ordered:
        raise SpecificationError(
            f"relative_to names {wanted!r}, which axis {axis_path!r} does not "
            f"declare; its categories are {', '.join(names.ordered)}"
        )
    return wanted


def _as_degradation(
    rows: list[dict[str, Any]], reference: str, spec: Declared
) -> list[dict[str, Any]]:
    """Every category as a percentage change from `reference`, paired per seed.

    `D_s = 100 · (c_s(this) − c_s(reference)) / c_s(reference)`, then reduced
    over seeds by the same statistics the absolute table uses.

    **The pairing is per seed and it is the point.** Dividing the two *column*
    means is the number a reader would get from the absolute table with a
    calculator, and it is a different one: an analytic contender scored under
    §A.2's common-random-numbers law has identical per-seed values, so its
    paired degradation carries a spread of exactly 0.0, where a spread combined
    from two independently reduced columns would not.

    **It is the relative change OF THE MEAN, not the mean of per-trajectory
    relative changes.** The second is dominated by trajectories whose reference
    cost is near zero, and is a different quantity — one that would put a
    controller's worst trajectory in charge of the figure.

    The reference rows are the denominator and are dropped: a bar at exactly
    0 % has no length, and drawing it would spend a fill slot and a legend row
    on ink that says nothing (Annex 03 §B.5.2).

    Raises:
        SpecificationError: If a contender has no reference row, if the two
            conditions were not scored on the same seeds, or if a reference
            value is zero.
    """
    by_contender: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_contender.setdefault(str(row["contender"]), {})[str(row["axis_label"])] = row

    degraded: list[dict[str, Any]] = []
    for contender, categories in by_contender.items():
        base = categories.get(reference)
        if base is None:
            raise SpecificationError(
                f"contender {contender!r} carries no {reference!r} row, so its "
                "degradation relative to it is undefined. A contender an axis's "
                "`applies_to` excludes has no category at all -- widen it, or "
                "drop the contender from this analysis with `series`. It is "
                "refused rather than dropped because a figure quietly missing "
                "the contender its caption is about is the plausible and wrong "
                "output this architecture exists to prevent"
            )
        anchor = dict(zip(base["seeds"], base["per_seed_aggregate"], strict=True))
        for category, row in categories.items():
            if category == reference:
                continue
            degraded.append(_one_degradation(row, anchor, reference, spec))
    return degraded


def _one_degradation(
    row: Mapping[str, Any],
    anchor: Mapping[int, float],
    reference: str,
    spec: Declared,
) -> dict[str, Any]:
    """One (contender, category) row, expressed against `anchor`."""
    per_seed: list[float] = []
    for seed, value in zip(row["seeds"], row["per_seed_aggregate"], strict=True):
        if seed not in anchor:
            raise SpecificationError(
                f"contender {row['contender']!r} was scored at seed {seed} under "
                f"{row['axis_label']!r} and not under {reference!r}, so the two "
                "cannot be paired. §A.3 aggregates over training seeds and a "
                "degradation pairs them one to one; an unpaired difference "
                "would compare one seed's shift with another seed's nominal"
            )
        before = float(anchor[seed])
        if before == 0.0:
            raise SpecificationError(
                f"contender {row['contender']!r} has a {reference!r} cost of "
                f"exactly zero at seed {seed}, so a percentage relative to it "
                "is undefined; report the absolute difference instead"
            )
        per_seed.append(100.0 * (float(value) - before) / before)

    return {
        **row,
        "aggregate": aggregate_of(np.asarray(per_seed), spec.aggregate),
        "interval_low": interval_of(per_seed, spec)[0],
        "interval_high": interval_of(per_seed, spec)[1],
        # A ratio of two per-seed means has no per-trajectory value to disperse.
        # NaN rather than a carried-over number: §A.3.2 is explicit that the
        # three dispersions must not share a name, and reusing this condition's
        # own within-seed spread here would label a cost dispersion as a
        # percentage one.
        "within_seed_spread": float("nan"),
        "across_seed_spread": across_seed_spread(per_seed, spec),
        "per_seed_aggregate": per_seed,
    }


@dataclass(frozen=True)
class _Categories:
    """The axis's categories: what they are called and what order they came in.

    Attributes:
        ordered: The names, in the study's declaration order.
        _positions: `key -> index`, where the key identifies a materialised
            axis value.
        _names: `key -> the name a reader is shown`.
    """

    ordered: tuple[str, ...]
    _positions: Mapping[tuple[Any, ...], int]
    _names: Mapping[tuple[Any, ...], str]

    def _resolve(self, key: tuple[Any, ...]) -> tuple[Any, ...]:
        """The declared position a materialised key belongs to.

        An axis a contender does not carry (`applies_to`, Annex 01 §2.5)
        writes nothing into its points, so the contender's key holds `None`
        where the position declares a value — found on the merged Figure 2,
        where the analytic families ride the world axis vacuously and their
        `(plant, "blind", None)` matched no declared tuple. A `None`
        component is a hole, and a hole must fill **unambiguously**: the key
        matches the one declared position agreeing on every component it
        does carry, and anything else — zero matches, or two positions that
        differ only where this contender is blind — is refused by name
        rather than resolved by declaration order.
        """
        if key in self._positions:
            return key
        matches = [
            declared
            for declared in self._positions
            if len(declared) == len(key)
            and all(
                mine is None or mine == theirs
                for mine, theirs in zip(key, declared, strict=True)
            )
        ]
        if len(matches) == 1:
            return matches[0]
        raise SpecificationError(
            f"category key {key!r} matches {len(matches)} declared "
            f"positions of the zipped group; an axis a contender does not "
            "carry leaves a hole in its key, and the hole must fill "
            "unambiguously — separate the positions on an axis every "
            "contender carries, or extend applies_to"
        )

    def position(self, key: tuple[Any, ...]) -> int:
        return self._positions[self._resolve(key)]

    def of(self, key: tuple[Any, ...]) -> str:
        return self._names[self._resolve(key)]


def _category_names(
    context: AnalysisContext, axis_path: str, group_paths: tuple[str, ...]
) -> _Categories:
    """The declared positions, named — from `labels` where the axis has them.

    **Keyed on the whole zipped position**, one `axis_key` per group member: a
    `ProblemSpec` keys as its `ProblemID` (two loads of one frozen file are one
    plant) and scalars as themselves, and the tuple over the group is what
    makes one plant under two rehost modes two categories. A single-member
    group reduces to the old scalar behaviour in a 1-tuple. Built
    `dict`-injectively and **refused on a duplicate position**: the old
    value-keyed map was last-wins on repeats, which would silently relabel a
    category rather than fail.

    Raises:
        SpecificationError: If a spec-valued axis declares no `labels` (this is
            where Annex 01 §2.5.1's requirement lives, corrected on 2026-08-07:
            the grammar cannot demand it, because a producer-only sweep never
            shows a value to anybody), or if two positions of the zipped group
            are identical — one category declared twice is a declaration error,
            not a wider bar.
    """
    by_path = {candidate.path: candidate for candidate in context.study.sweep}
    axis = by_path.get(axis_path)
    if axis is None:  # pragma: no cover -- `axis_path` already resolved it
        raise SpecificationError(
            f"axis {axis_path!r} is not swept by study {context.study.id!r}"
        )
    if not axis.labels and any(
        hasattr(value, "get_signature") for value in axis.values
    ):
        raise SpecificationError(
            f"axis {axis_path!r} sweeps whole specifications and declares no "
            "`labels`, so its categories have no readable names — a table, a "
            "legend or a bar tick would print a repr, and a filename would name "
            "a category `icassp_n4m2_N100_u0p1_s0_rotA30` in a paper. Add "
            "`labels = [...]` to the axis, one per value, in the same order "
            "(Annex 01 §2.5.1). They take no part in any identifier, so adding "
            "them retrains nothing"
        )
    # `str(value)`, not `axis_key(value)`: the key is for GROUPING and may be
    # any hashable, while a label is text a reader meets. They coincided for
    # every scalar axis and came apart the moment the key stopped being the
    # value -- an integer axis then labelled its categories `2` and `4` as ints
    # and a test comparing them with `"2"` caught it.
    names = tuple(
        axis.labels[index] if axis.labels else str(value)
        for index, value in enumerate(axis.values)
    )
    group_axes = [by_path[path] for path in group_paths]
    keys = [
        tuple(axis_key(value) for value in position)
        for position in zip(*(member.values for member in group_axes), strict=True)
    ]
    if len(set(keys)) != len(keys):
        raise SpecificationError(
            f"axis {axis_path!r} declares the same category position more than "
            "once across its zipped group; each position is one category, so a "
            "repeat is a declaration error rather than a wider bar"
        )
    return _Categories(
        ordered=names,
        _positions={key: index for index, key in enumerate(keys)},
        _names=dict(zip(keys, names, strict=True)),
    )
