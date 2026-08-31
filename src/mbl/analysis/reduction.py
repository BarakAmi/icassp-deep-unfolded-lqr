"""The reduction §A.3 fixes, shared by every kind that performs it.

`cost_vs_axis` and `cost_by_category` differ in exactly one thing — what a gap
between two x positions is allowed to mean — and in nothing else. The
grouping, the replicate guard, the per-seed aggregate, the three dispersions of
§A.3.2 and the BCa interval of §A.4 are one computation, so they are one
implementation.

**This is not a tidiness refactor.** §A.3 is a *declaration* about how numbers
are combined, stated in a sidecar that a reader is entitled to trust; two
copies of it are two answers to "what did the error bar reduce over", and the
second one drifts silently because both figures keep rendering. The one place
this project has already been bitten by a second reading of one rule is
`_flat_levels`, whose own docstring records three callers that agreed "only by
accident".

Nothing here knows what an axis value *is*. A kind supplies its own row
construction and its own ordering, which is the only place the two differ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spec.errors import SpecificationError
from ..spec.study import StudyPoint, StudySpec
from ..spec.tiers import require_analysable_measurement
from .registry import AnalysisContext

#: The samples column §A.3.1 makes required content, and every kind's default
#: quantity.
DEFAULT_QUANTITY = "trajectory_cost"

#: Aggregates §A.3 admits, and the dispersion each one is paired with. A mean
#: reports a standard deviation; a median reports an interquartile range,
#: because a standard deviation about a median is a statistic of nothing.
AGGREGATES: dict[str, str] = {"mean": "std", "median": "iqr"}

#: Interval defaults. §A.3's table: bootstrap 95 % BCa, 10,000 resamples.
DEFAULT_INTERVAL_LEVEL = 0.95
DEFAULT_RESAMPLES = 10_000

#: Fewer training seeds than this and there is no across-seed interval to
#: report -- not a narrow one, none.
MIN_SEEDS_FOR_INTERVAL = 2

#: Entropy for the BCa resampling. Without it `scipy.stats.bootstrap` builds a
#: generator from OS entropy, and two analyses of one unchanged store returned
#: different intervals -- measured, up to 1.1e-3 apart run to run and up to
#: 2.8e-2 against the tables the ICASSP campaign stored, while every other
#: column re-derived at exactly 0.0.
#:
#: Nothing published moved, because the paper's figures declare
#: `dispersion: none` and draw no band. It is fixed anyway: a reviewer told
#: they can recompute every number in the paper will diff two tables and find a
#: mismatch in a *correct* bundle, and an acceptance test comparing recomputed
#: against stored would need a permanent exemption for these two columns.
#:
#: A generator is built PER CALL from this constant, never threaded through the
#: reduction. An interval must depend on the values it summarises and on
#: nothing else -- one shared stream would make row five's interval a function
#: of rows one to four, so re-ordering a table, which changes no measurement,
#: would change a published number.
INTERVAL_SEED = 0


@dataclass(frozen=True)
class Declared:
    """An analysis's configuration, validated once.

    Attributes:
        axis_property: The intrinsic numeric property a problem-valued axis is
            projected onto (Annex 03 §A.6.1 as amended); `None` for an axis
            that is numeric by itself. Read by `cost_vs_axis` only; the
            categorical kind refuses it.
        relative_to_contender: The contender whose per-seed values are the
            per-position denominator (the ratio normalisation); `None` for
            absolute quantities. Read by `cost_vs_axis` only.
    """

    axis_path: str
    quantity: str
    aggregate: str
    dispersion: str
    level: float
    resamples: int
    axis_property: str | None = None
    relative_to_contender: str | None = None


@dataclass(frozen=True)
class GroupReduction:
    """One `(contender, axis value)` group, reduced.

    Attributes:
        aggregate: The point estimate — the aggregate of the per-seed
            aggregates, which is §A.3's second step.
        per_seed: The aggregate within each training seed, in point order.
        seeds: The training seeds, in the same order — so the reduction behind
            the bar is auditable from the table (§A.3.2).
        trajectories: How many evaluation trajectories went in, over all seeds.
        within_seed_spread: The dispersion over evaluation trajectories,
            averaged over seeds.
        across_seed_spread: The dispersion of `per_seed` — **what the figure
            draws** — or `NaN` below two seeds.
        interval: The across-seed BCa interval, or `(nan, nan)`.
    """

    aggregate: float
    per_seed: tuple[float, ...]
    seeds: tuple[int, ...]
    trajectories: int
    within_seed_spread: float
    across_seed_spread: float
    interval: tuple[float, float]


def declared(study: StudySpec, config: Mapping[str, Any]) -> Declared:
    """The configuration, validated against the study it will be run on.

    Raises:
        SpecificationError: On an aggregate §A.3 does not admit, or when the
            axis cannot be resolved.
    """
    aggregate = str(config.get("aggregate", "mean"))
    if aggregate not in AGGREGATES:
        raise SpecificationError(
            f"aggregate {aggregate!r} is not one of {sorted(AGGREGATES)}; §A.3 "
            "admits a mean for light-tailed cost and a median for heavy-tailed"
        )
    reference = config.get("relative_to_contender")
    if reference is not None:
        labels = sorted(spec.resolved_label for spec in study.contenders)
        if str(reference) not in labels:
            raise SpecificationError(
                f"relative_to_contender names {str(reference)!r}, which no "
                f"contender declaration carries; declared labels: {labels}"
            )
    return Declared(
        axis_path=axis_path(study, config),
        quantity=str(config.get("quantity", DEFAULT_QUANTITY)),
        aggregate=aggregate,
        dispersion=AGGREGATES[aggregate],
        level=float(config.get("interval_level", DEFAULT_INTERVAL_LEVEL)),
        resamples=int(config.get("resamples", DEFAULT_RESAMPLES)),
        axis_property=(
            str(config["axis_property"]) if "axis_property" in config else None
        ),
        relative_to_contender=None if reference is None else str(reference),
    )


def axis_path(study: StudySpec, config: Mapping[str, Any]) -> str:
    """Which swept axis is the x axis.

    Defaulted only when the study declares exactly one, which is a fact about
    *this* study rather than about studies in general: a two-axis study must
    say which one it means, because picking the first would be picking by
    declaration order.
    """
    wanted = config.get("axis")
    available = [axis.path for axis in study.sweep]
    if wanted is None:
        if len(available) == 1:
            return available[0]
        raise SpecificationError(
            f"study {study.id!r} declares {len(available)} swept axes "
            f"({', '.join(available) or 'none'}), so `axis` cannot be "
            "defaulted; name the one this table is against"
        )
    if wanted not in available:
        raise SpecificationError(
            f"axis {wanted!r} is not swept by study {study.id!r}; declared "
            f"axes: {', '.join(available) or '(none)'}"
        )
    return str(wanted)


def axis_key(value: Any) -> Any:
    """One identity for an axis value, and the answer to "same category?".

    **A group key has to be hashable and an axis value need not be.** Scalars
    are their own key, which is every axis this grammar had until
    `evaluation.problem` became sweepable; a `ProblemSpec` holds NumPy arrays
    in dicts and raises `TypeError: unhashable type: 'dict'` — found by
    grouping one, not by reading.

    Its `ProblemID` is the right key rather than a fallback: it is derived from
    the matrices alone, so two loads of one frozen file are one category, which
    is the equality the store itself already uses. `str` would key on a repr
    and make formatting decide.
    """
    if value is None:
        return None
    identifier = getattr(value, "problem_id", None)
    if identifier is not None:
        return str(identifier)
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def category_key_of(point: StudyPoint, group_paths: Sequence[str]) -> tuple[Any, ...]:
    """One point's category identity over a zipped group of axes.

    A tuple of `axis_key` per group member, in declaration order. For a
    single-axis group this is a 1-tuple of the old scalar key, so nothing
    about the ordinary case changes shape twice.
    """
    return tuple(axis_key(point.axis_values.get(path)) for path in group_paths)


def grouped(
    study: StudySpec,
    spec: Declared,
    *,
    group_paths: Sequence[str] | None = None,
) -> dict[tuple[str, Any], list[StudyPoint]]:
    """Points by (contender label, axis KEY), preserving declaration order.

    A contender the axis does not apply to contributes `None` as its axis
    value and therefore one group, which is what makes it a flat reference
    rather than a curve or a bar.

    **The key is `axis_key(value)` and not the value**, so a caller wanting the
    value itself reads it back off the group's own points — which is where it
    has always lived, and is the one form that cannot disagree with what was
    materialised.

    `group_paths` widens the key to a whole **zipped group** (Annex 01 §2.5):
    two positions sharing the declared axis's value — one plant scored blind
    and aware — are two categories, and keying on the single value would fold
    them into one. Defaulted to the declared axis alone, which is every
    caller's behaviour before zipped companions existed; a caller that wants
    the composite passes `study.zip_group_paths(spec.axis_path)`.
    """
    paths = tuple(group_paths) if group_paths is not None else (spec.axis_path,)
    groups: dict[tuple[str, Any], list[StudyPoint]] = {}
    for point in study.materialise():
        key = (
            point.contender.resolved_label,
            category_key_of(point, paths),
        )
        groups.setdefault(key, []).append(point)
    return groups


def axis_value_of(points: Sequence[StudyPoint], spec: Declared) -> Any:
    """The axis value a group was bound at, or `None` for a group with none."""
    return points[0].axis_values.get(spec.axis_path) if points else None


def require_replicates(
    groups: dict[tuple[str, Any], list[StudyPoint]],
    study: StudySpec,
    spec: Declared,
    *,
    group_paths: Sequence[str] | None = None,
) -> None:
    """Refuse a group that holds anything other than training replicates.

    `grouped` keys on `(contender, axis value)` for the **one** declared axis,
    so on a two-axis study every group silently collects the other axis's
    points as well. That is not a blurred mean — it **fabricates the error
    bar**. Measured on a one-seed study with a second swept axis: the groups
    came back with `seeds = [0, 0]`, and the analysis reported `n_seeds = 2`
    with an `across_seed_spread` computed across two points that are not two
    seeds, beneath a sidecar declaring `across_seed_spread_over =
    "training_seeds"`.

    §A.3 names training seeds as the across-seed unit and §A.3.2 makes that
    spread the quantity a figure **draws**, so a number manufactured from a
    second axis is drawn as retraining variability — a claim nobody made, and
    one nothing downstream could detect, because the frame, the figure and the
    table would agree with each other and all be wrong.

    The detector is the one property a replicate group cannot violate: its
    seeds are distinct. That is checked rather than "is there more than one
    swept axis", because a second axis is only a problem when it actually
    lands in a group — an axis whose `applies_to` excludes this contender is
    harmless, and an enumeration of axis shapes would have to keep guessing
    which.

    Raises:
        SpecificationError: If any group repeats a training seed.
    """
    grouped_by = set(group_paths) if group_paths is not None else {spec.axis_path}
    others = [axis.path for axis in study.sweep if axis.path not in grouped_by]
    for (label, key), points in groups.items():
        seeds = [point.seed for point in points]
        if len(set(seeds)) == len(seeds):
            continue
        parts = key if isinstance(key, tuple) else (key,)
        where = (
            "no axis value"
            if all(part is None for part in parts)
            else f"{spec.axis_path} = {parts[0] if len(parts) == 1 else parts}"
        )
        blame = (
            f"; the study also sweeps {', '.join(others)}, which this analysis "
            "does not group by, so those points land here as though they were "
            "replicates"
            if others
            else ""
        )
        raise SpecificationError(
            f"contender {label!r} at {where} has {len(points)} points over "
            f"{len(set(seeds))} distinct training seed(s) — seeds {seeds}{blame}. "
            "§A.3 aggregates over training seeds and §A.3.2 draws that spread, "
            "so pooling anything else here would report a manufactured number "
            "as retraining variability. Split this into one document per swept "
            "axis: there is no way to group by two, and this analysis reduces "
            "over whatever a group contains"
        )


def costs(context: AnalysisContext, point: StudyPoint, quantity: str) -> np.ndarray:
    """One point's per-trajectory costs, read from the store.

    **The call site `require_analysable_measurement` was built for.** A
    measurement stamped `axis_subset` came from a run that truncated a swept
    axis to check plumbing, and a figure drawn from it would be plausible and
    wrong.
    """
    measurement_id = str(point.measurement_id)
    if not context.measurements.exists(measurement_id):
        raise SpecificationError(
            f"study {context.study.id!r} contender "
            f"{point.contender.resolved_label!r} at seed {point.seed} has no "
            f"measurement {measurement_id} in the store; an analysis reads what "
            "was produced and never produces it -- run the study first"
        )
    record = context.measurements.get(measurement_id)
    require_analysable_measurement(record.spec.get("provenance", {}))
    if record.samples is None or quantity not in record.samples.columns:
        raise SpecificationError(
            f"measurement {measurement_id} carries no {quantity!r} column; "
            "Annex 03 §A.3.1 makes the per-trajectory cost required content, "
            "and a measurement written before it cannot be aggregated"
        )
    return np.asarray(record.samples[quantity], dtype=np.float64)


def reduce_group(
    context: AnalysisContext, points: Sequence[StudyPoint], spec: Declared
) -> GroupReduction:
    """§A.3's two-step reduction over one group, in the order it declares.

    First over evaluation trajectories within a seed, then over training
    seeds. The point estimate is the same either way for equal batch sizes;
    the *interval* is not, and reporting an across-trajectory interval as
    though it answered "would this hold if I retrained?" is the failure that
    is invisible in a rendered figure.
    """
    per_seed: list[float] = []
    per_seed_spread: list[float] = []
    seeds: list[int] = []
    trajectories = 0
    for point in points:
        values = costs(context, point, spec.quantity)
        per_seed.append(aggregate_of(values, spec.aggregate))
        per_seed_spread.append(spread_of(values, spec.dispersion))
        seeds.append(int(point.seed))
        trajectories += values.size
    return GroupReduction(
        aggregate=aggregate_of(np.asarray(per_seed), spec.aggregate),
        per_seed=tuple(per_seed),
        seeds=tuple(seeds),
        trajectories=trajectories,
        within_seed_spread=float(np.mean(per_seed_spread)),
        across_seed_spread=across_seed_spread(per_seed, spec),
        interval=interval_of(per_seed, spec),
    )


def aggregate_of(values: np.ndarray | Sequence[float], aggregate: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array) if aggregate == "mean" else np.median(array))


def spread_of(values: np.ndarray, dispersion: str) -> float:
    """The within-seed dispersion paired with the declared aggregate."""
    if dispersion == "std":
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    quartiles = np.percentile(values, [25.0, 75.0])
    return float(quartiles[1] - quartiles[0])


def across_seed_spread(per_seed: Sequence[float], spec: Declared) -> float:
    """The dispersion of the per-seed aggregates — what the figure draws.

    A third dispersion, and §A.3.2 is explicit that the three must not share a
    name. `within_seed_spread` is over evaluation trajectories and exists for
    one seed; `interval_low`/`interval_high` are the BCa interval §A.4
    reports; this is the standard deviation (or interquartile range) of the
    per-seed aggregates, and it is the only one drawn.

    **`NaN` below two seeds, and deliberately not `0.0`.** `spread_of` answers
    `0.0` for a single value, which is the right answer to "how spread are
    these trajectories" and the wrong answer to "how spread are these seeds":
    one seed has no across-seed quantity at all. The two facts print as `—`
    and `0.000` and a zero here would silently claim the second.
    """
    if len(per_seed) < MIN_SEEDS_FOR_INTERVAL:
        return float("nan")
    return spread_of(np.asarray(per_seed, dtype=np.float64), spec.dispersion)


def interval_of(per_seed: Sequence[float], spec: Declared) -> tuple[float, float]:
    """The across-seed interval, or `(nan, nan)` when there is none.

    §A.3 reports the ACROSS-SEED interval, because that is the one answering
    "would this hold if I retrained?". With one seed there is no such
    interval, and this returns NaN rather than the across-trajectory interval
    -- which is always available, always narrower, and undetectable once it
    has been drawn as a band.
    """
    if len(per_seed) < MIN_SEEDS_FOR_INTERVAL:
        return (float("nan"), float("nan"))
    values = np.asarray(per_seed, dtype=np.float64)
    if bool(np.all(values == values[0])):
        # Every seed produced the identical number, which for an analytic
        # contender is not a degenerate case but the *answer*: a closed-form
        # solve has no training variance, so its across-seed interval has zero
        # width. Returning it directly rather than bootstrapping is not a
        # shortcut -- BCa's acceleration divides by the jackknife spread, so
        # the bootstrap is undefined here and scipy warns rather than says so.
        # Found by running the analysis, not by reading it.
        return (float(values[0]), float(values[0]))
    from scipy.stats import bootstrap

    statistic = np.mean if spec.aggregate == "mean" else np.median
    result = bootstrap(
        (values,),
        statistic,
        confidence_level=spec.level,
        n_resamples=spec.resamples,
        method="BCa",
        random_state=np.random.default_rng(INTERVAL_SEED),
    )
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def sidecar(
    context: AnalysisContext,
    spec: Declared,
    table: Any,
    **extra: Any,
) -> dict[str, Any]:
    """What §A.4 requires every table to state, emitted from the declaration.

    Stated rather than left to the caller precisely because §A.4 says these
    "are emitted automatically from the specification, so they cannot be
    omitted or misstated".

    Args:
        context: The analysis's context.
        spec: The validated declaration.
        table: The tidy table, for the counts it reports.
        extra: Kind-specific declarations appended verbatim — §A.6.1's
            `axis_kind` and its declared category order.
    """
    seeds = sorted({int(count) for count in table["n_seeds"]})
    has_interval = bool(table["interval_low"].notna().any())
    return {
        "analysis_id": context.spec.id,
        "kind": context.spec.kind,
        "study": context.study.id,
        "study_id": context.study_id,
        "config": {
            key: context.spec.config[key] for key in sorted(context.spec.config)
        },
        "quantity": spec.quantity,
        "aggregate": spec.aggregate,
        # The order is REPORTED, not merely performed (§A.3).
        "aggregation_order": ["evaluation_trajectories", "training_seeds"],
        "interval": {
            "kind": "bootstrap_bca" if has_interval else None,
            "level": spec.level if has_interval else None,
            "resamples": spec.resamples if has_interval else None,
            "over": "training_seeds",
            "reason": None
            if has_interval
            else (
                "fewer than two training seeds, so the across-seed interval "
                "§A.3 reports does not exist; the within-seed dispersion is "
                "reported separately as `within_seed_spread` and is not a "
                "confidence interval"
            ),
        },
        "within_seed_spread_kind": spec.dispersion,
        # §A.3.2 requires the sidecar to state these beside the interval's own
        # declarations, so a reader is never left inferring which of the three
        # dispersions the figure's bars are.
        "across_seed_spread_kind": spec.dispersion,
        "across_seed_spread_over": "training_seeds",
        "axis": spec.axis_path,
        "n_training_seeds": seeds,
        "n_evaluation_trajectories": int(table["n_trajectories"].sum()),
        "n_rows": int(len(table)),
        **extra,
    }
