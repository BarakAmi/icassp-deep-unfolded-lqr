"""`cost_vs_axis` — aggregate cost with intervals, per contender, per axis point.

Annex 03 §A.6's first kind, and the one the Stage 3–5 vertical slice is built
around. The reduction itself lives in `reduction`, shared with
`cost_by_category`: §A.3 is a *declaration* about how numbers are combined and
two copies of it are two answers to "what did the error bar reduce over". What
is this module's own is the axis — a **numeric** one, whose gaps carry
distance and may therefore be joined by a line.

Three properties of §A.3 are load-bearing and each is stated in the sidecar
rather than assumed by the reader:

**The aggregation order is fixed and reported.** First over evaluation
trajectories within a seed, then over training seeds. The point estimate is
the same either way for equal batch sizes; the *interval* is not, and reporting
an across-trajectory interval as though it answered "would this hold if I
retrained?" is the failure that is invisible in a rendered figure.

*A measured caveat on that order, found by mutation testing.* A mutant that
pooled every trajectory of every seed into one mean **cannot currently be
killed by any study this grammar can express**, and that is not a gap in the
tests: `evaluation` is declared once per study and §A.2's common-random-numbers
law scores every contender and every seed on the identical realisations, so
every seed contributes the same trajectory count and a mean of means *is* the
pooled mean. The order is performed and declared anyway, because the equality
is a property of today's protocol rather than of the statistic — a study whose
seeds were scored on different batch counts would silently weight them by size.

**One training seed is the normal case, and it has no across-seed interval.**
`training.seeds` is 1 at `smoke` and at `standard`; only `publication` raises
it to 5. So for most runs the interval §A.3 says to report does not exist, and
this analysis emits a null one with a stated reason. Substituting the
across-trajectory interval — always available, always narrower — would be
undetectable downstream, which is exactly why it is refused rather than
defaulted. The within-seed dispersion travels in a **differently named**
column so that no figure can render it as an across-seed band by accident.

**A depth-invariant contender has no axis value.** `truncated_riccati`, `cocp`
and `cocp_lower_bound` carry no unrolling depth, so their rows carry a null
`axis_value` and `Role` decides how they are drawn (Annex 03 §B.3.1: a bound
is grey, dashed, annotated with its value). Placing them at some arbitrary
depth would turn a reference line into a curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..spec.errors import SpecificationError
from .bounds import bound_provenance, bound_rows
from .reduction import (
    AGGREGATES,
    axis_value_of,
    DEFAULT_INTERVAL_LEVEL,
    DEFAULT_QUANTITY,
    DEFAULT_RESAMPLES,
    MIN_SEEDS_FOR_INTERVAL,
    Declared,
    across_seed_spread,
    aggregate_of,
    declared,
    grouped,
    interval_of,
    reduce_group,
    require_replicates,
    sidecar,
)
from .registry import AnalysisContext, AnalysisOutput, register_analysis

__all__ = [
    "AGGREGATES",
    "DEFAULT_INTERVAL_LEVEL",
    "DEFAULT_QUANTITY",
    "DEFAULT_RESAMPLES",
    "MIN_SEEDS_FOR_INTERVAL",
    "cost_vs_axis",
]

#: What §A.6.1 says this kind's axis must be, stated in the message an author
#: reads. Named beside the guard rather than inlined so the two spellings of
#: the sibling kind cannot drift apart.
CATEGORICAL_KIND = "cost_by_category"

#: The closed set of intrinsic numeric properties a problem-valued axis may be
#: projected onto (Annex 03 §A.6.1 as amended 2026-08-08). Grows only with the
#: annex; the first consumer is Figure 3's state-dimension sweep.
PROPERTIES: dict[str, Any] = {
    "state_dim": lambda problem: float(problem.state_dim),
}


@register_analysis("cost_vs_axis")
def cost_vs_axis(context: AnalysisContext) -> AnalysisOutput:
    """Aggregate cost per contender, per axis point, with intervals.

    Args:
        context: The study, the measurement store and the declaration.

    Returns:
        The tidy table and its statistical sidecar.

    Raises:
        SpecificationError: If the declaration names no usable axis, if the
            axis is not numeric (§A.6.1), if a point's measurement is absent
            from the store, or if any measurement was produced under a
            subsetted run (§A.3's `require_analysable_measurement`).
    """
    spec = declared(context.study, context.spec.config)
    groups = grouped(context.study, spec)
    require_replicates(groups, context.study, spec)
    projection = _resolve_projection(groups, spec)
    _require_numeric_axis(groups, spec, projection)
    rows: list[dict[str, Any]] = []

    for (label, _), points in groups.items():
        axis_value = _projected_value(axis_value_of(points, spec), projection)
        reduced = reduce_group(context, points, spec)
        contender = points[0].contender
        rows.append(
            {
                "contender": label,
                "role": contender.role.value,
                "axis_path": spec.axis_path if axis_value is not None else "",
                "axis_value": np.nan if axis_value is None else float(axis_value),
                "axis_label": "" if axis_value is None else str(axis_value),
                "n_seeds": len(reduced.per_seed),
                "n_trajectories": reduced.trajectories,
                "aggregate": reduced.aggregate,
                "interval_low": reduced.interval[0],
                "interval_high": reduced.interval[1],
                "within_seed_spread": reduced.within_seed_spread,
                "across_seed_spread": reduced.across_seed_spread,
                # §A.3.2: the reduction behind the bar is auditable from the
                # same parquet the figure was drawn from. Obtaining these in
                # the table emitter would reach past `<id>.data.parquet` and
                # reopen the drift §B.1.2 exists to close.
                "seeds": list(reduced.seeds),
                "per_seed_aggregate": [float(value) for value in reduced.per_seed],
            }
        )

    if spec.relative_to_contender is not None:
        rows = _as_reference_ratio(rows, spec)
    # Bounds are appended AFTER any ratio transform: a floor expressed relative
    # to a contender is a different quantity, and silently rescaling it would
    # be the convention error one tier along.
    rows = rows + bound_rows(context.study, context.spec.config)
    table = pd.DataFrame(rows).sort_values(
        ["contender", "axis_value"], na_position="first", ignore_index=True
    )
    extra: dict[str, Any] = {}
    provenance = bound_provenance(context.study, context.spec.config)
    if provenance:
        # A bound quoted without its residuals is a number; this project has one
        # on record at 56.043029, placed ABOVE the optimum it claimed to bound.
        extra["bounds"] = provenance
    if projection is not None:
        extra["axis_property"] = spec.axis_property
    if spec.relative_to_contender is not None:
        extra |= {
            "relative_to_contender": spec.relative_to_contender,
            "quantity_unit": "ratio",
            "pairing": "training_seeds",
            "within_seed_spread_reason": (
                "a ratio of two per-seed means has no per-trajectory value to "
                "disperse; the figure's table carries the absolute costs"
            ),
        }
    return AnalysisOutput(table=table, sidecar=sidecar(context, spec, table, **extra))


def _resolve_projection(
    groups: dict[tuple[str, Any], list[Any]], spec: Declared
) -> dict[str, float] | None:
    """The problem-valued axis's numeric projection, or `None` for an axis
    that is numeric by itself (Annex 03 §A.6.1 as amended).

    Refused when inert, in either direction: a problem-valued axis without a
    declared `axis_property` has no float of its own, and a declared property
    over an axis whose values are not problems reads nothing. The projection
    must be injective over the axis — two distinct plants landing on one x
    would let a line assert a rate of change that does not exist.

    Returns:
        ``{problem_id: projected value}``, or `None`.
    """
    problems: dict[str, Any] = {}
    for points in groups.values():
        value = axis_value_of(points, spec)
        if value is not None and hasattr(value, "problem_id"):
            problems[str(value.problem_id)] = value
    if spec.axis_property is None:
        if problems:
            raise SpecificationError(
                f"axis {spec.axis_path!r} takes whole-problem values, which "
                "have no float of their own; declare which intrinsic property "
                'is the x axis, e.g. axis_property = "state_dim" (Annex 03 '
                "§A.6.1)"
            )
        return None
    if spec.axis_property not in PROPERTIES:
        raise SpecificationError(
            f"axis_property {spec.axis_property!r} is not one of "
            f"{sorted(PROPERTIES)}; the set is closed and grows only with "
            "Annex 03 §A.6.1"
        )
    if not problems:
        raise SpecificationError(
            f"axis_property {spec.axis_property!r} is declared over axis "
            f"{spec.axis_path!r}, whose values are not problems; the "
            "projection would read nothing, and a declaration that takes no "
            "effect is the inert-declaration defect"
        )
    project = PROPERTIES[spec.axis_property]
    projection = {key: float(project(value)) for key, value in problems.items()}
    seen: dict[float, str] = {}
    for key, projected in projection.items():
        if projected in seen:
            raise SpecificationError(
                f"plants {seen[projected]} and {key} both project onto "
                f"{spec.axis_property} = {projected!r}; two rows at one x "
                "would let a line assert a rate of change that does not "
                f"exist, so use kind = {CATEGORICAL_KIND!r} for this axis"
            )
        seen[projected] = key
    return projection


def _projected_value(value: Any, projection: dict[str, float] | None) -> Any:
    """A group's axis value, through the projection when one is declared."""
    if value is None or projection is None:
        return value
    if hasattr(value, "problem_id"):
        return projection[str(value.problem_id)]
    return value


def _as_reference_ratio(
    rows: list[dict[str, Any]], spec: Declared
) -> list[dict[str, Any]]:
    """Every row as the per-seed ratio to the reference contender at the SAME
    axis position — the transpose of `cost_by_category`'s `relative_to`,
    which normalises by an axis category within a contender.

    The pairing is per training seed, exactly as §A.4 pairs, and only then
    reduced. The reference's own rows come out at exactly 1.0 and are KEPT:
    on a line figure the floor at 1.0 is the visual reference the
    normalisation is for (Annex 03 §A.6.1 as amended).

    Raises:
        SpecificationError: If a row carries no axis position, if the
            reference was not scored at a row's position, if the two were not
            scored on the same seeds, or if a reference value is exactly zero.
    """
    reference = spec.relative_to_contender
    anchors: dict[float, dict[Any, float]] = {}
    for row in rows:
        if row["contender"] == reference and not np.isnan(row["axis_value"]):
            anchors[float(row["axis_value"])] = dict(
                zip(row["seeds"], row["per_seed_aggregate"], strict=True)
            )
    out: list[dict[str, Any]] = []
    for row in rows:
        if np.isnan(row["axis_value"]):
            raise SpecificationError(
                f"contender {row['contender']!r} carries no axis position, so "
                f"its ratio against {reference!r} pairs with no denominator; "
                "widen the axis's applies_to, or drop the contender from this "
                "analysis with `series`"
            )
        anchor = anchors.get(float(row["axis_value"]))
        if anchor is None:
            raise SpecificationError(
                f"{reference!r} was not scored at axis value "
                f"{row['axis_value']!r}, so {row['contender']!r} has no "
                "denominator there; the reference must ride every position "
                "the normalised contenders ride"
            )
        per_seed: list[float] = []
        for seed, value in zip(row["seeds"], row["per_seed_aggregate"], strict=True):
            if seed not in anchor:
                raise SpecificationError(
                    f"contender {row['contender']!r} was scored at seed "
                    f"{seed} where {reference!r} was not, so the two cannot "
                    "be paired; §A.3 aggregates over training seeds and a "
                    "ratio pairs them one to one"
                )
            before = float(anchor[seed])
            if before == 0.0:
                raise SpecificationError(
                    f"{reference!r} has a cost of exactly zero at seed "
                    f"{seed}, axis value {row['axis_value']!r}, so a ratio "
                    "to it is undefined; report the absolute difference "
                    "instead"
                )
            per_seed.append(float(value) / before)
        out.append(
            {
                **row,
                "aggregate": aggregate_of(np.asarray(per_seed), spec.aggregate),
                "interval_low": interval_of(per_seed, spec)[0],
                "interval_high": interval_of(per_seed, spec)[1],
                # A ratio of two per-seed means has no per-trajectory value to
                # disperse (§A.3.2: the three dispersions must not share a
                # name).
                "within_seed_spread": float("nan"),
                "across_seed_spread": across_seed_spread(per_seed, spec),
                "per_seed_aggregate": per_seed,
            }
        )
    return out


def _require_numeric_axis(
    groups: dict[tuple[str, Any], list[Any]],
    spec: Declared,
    projection: dict[str, float] | None = None,
) -> None:
    """Refuse an axis whose values are names, naming the kind that takes them.

    A projected problem-valued axis (`projection` not `None`) is numeric by
    construction — `_resolve_projection` already validated every position —
    so the check below concerns only the raw-value axes.

    **The defect this replaces was a crash, not a refusal.** A swept axis whose
    values are strings is already expressible, already materialises and already
    reaches identity — measured on `contenders.*.config.kind`, six points with
    six distinct `ModelID`s — and it reached this function's row construction,
    where `float("learned_step_size")` raised a bare `ValueError` out of the
    middle of a statistic. An author meeting that has no way to tell whether
    the grammar or the arithmetic was at fault.

    Checked here rather than at the declaration because the declaration knows
    only a *path*: whether the values along it are numeric is a fact about the
    axis, and the axis is only resolved once the points are materialised.

    Raises:
        SpecificationError: Naming the offending value and `cost_by_category`.
    """
    if projection is not None:
        return
    for points in groups.values():
        axis_value = axis_value_of(points, spec)
        if axis_value is None or isinstance(axis_value, bool):
            continue
        try:
            float(axis_value)
        except (TypeError, ValueError):
            raise SpecificationError(
                f"axis {spec.axis_path!r} takes the value {axis_value!r}, which "
                "is a name rather than a position, and `cost_vs_axis` places "
                "its rows on a numeric axis. Annex 03 §A.6.1: a category has "
                "no position, cannot be interpolated between and has no "
                f"distance to its neighbour, so use kind = {CATEGORICAL_KIND!r}, "
                "which reduces identically and orders the categories as the "
                "study declares them"
            ) from None
