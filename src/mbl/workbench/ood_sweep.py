"""The zero-shot OOD sweep engine (docs/planning/03_studies/nb06_ood_generalization/zero_shot_ood_generalization_benchmark.md
Sec 3.3/3.6-3.8): drives every (contender, level, seed) point of one
perturbation axis through the per-contender fairness/floor/rehost machinery
and returns one frozen result -- the notebook cell is a handful of lines,
with zero perturbation loops of its own (the established
`run_depth_sweep_study` workbench contract: runs things, returns data,
displays nothing).

One generic core (`_run_ood_sweep_core`), six thin axis-specific wrappers
(`run_horizon_ood_sweep`/`run_scale_ood_sweep` (also covers the Axis
S-noise sub-variant, `scale_initial_state=False`)/`run_dynamics_ood_sweep`/
`run_noise_family_ood_sweep`/`run_constraint_ood_sweep` (all three of Axis
C's protocols)) -- mirroring `workbench.depth_sweep`'s own
`run_depth_sweep_study` + per-notebook-wrapper shape -- plus
`run_ood_depth_ablation` for the "over-thinking" ablation (Sec 3.6), which
drives NOMINAL TRAINING at several unfolding depths through the existing
per-contender cache before running a SCALE-axis sweep at each.

Five laws this module enforces structurally, not by convention:

* **Statistical rigor** (Sec 3.4) -- `OODSweepSpec` refuses ``n_seeds <
  MIN_OOD_SEEDS`` at construction.
* **Stochastic fairness** (Sec 3.5) -- every contender at a given (level,
  seed) point is evaluated against a batch spec seeded IDENTICALLY (the seed
  formula never depends on the contender label), so two contenders draw
  byte-identical realizations wherever their level grids overlap.
* **Floor caching** (Sec 3.7) -- `compute_box_constrained_floors` (imported
  verbatim from `applications.studies.nb04_box_constrained`, never
  re-derived) is memoized by the actual perturbed ``(A, B, process_noise_std,
  u_max)`` content, so every axis pays for exactly as many distinct SDP
  solves as it has distinct problem instances -- one for the whole HORIZON
  axis (the floor does not depend on horizon at all), one per level for
  SCALE, one per (level, perturbation seed) for the two DYNAMICS axes, and
  one per level for Axis C (the floor moves with the shifted bound --
  `_run_ood_sweep_core` computes it at each point's own `audit_u_max`, never
  the nominal `context.u_max` unconditionally).
* **Horizon-OOD law** (Sec 1.4/3.5) -- only the HORIZON axis rehosts BY
  HORIZON (`applications.ood.rehost.rehost_at_horizon`); every other axis
  evaluates the nominally-synthesized artifact exactly as trained, by
  construction (its `_OODPointInputs.rehost_horizon` is always ``None``).
* **Constraint-OOD law** (Sec 2.6/3.9) -- only Axis C's AWARE/NULL
  protocols rehost BY BOUND (`applications.ood.constraint_shift
  .rehost_at_bound`); BLIND deliberately does not (the artifact keeps its
  trained belief while the plant clips externally --
  `evaluate_under_shift`'s `applied_control_bound`), and no axis ever sets
  both `rehost_horizon` and `rehost_bound` on the same point.
"""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..applications.factories import GaussianBatchSpec, LQRProblemFactory
from ..applications.ood.constraint_shift import ControlBoundShift, rehost_at_bound
from ..applications.ood.distances import distance_to_nominal_gaussian
from ..applications.ood.noise import ExoticBatchSpec, NoiseFamily
from ..applications.ood.perturbations import (
    DynamicsPerturbation,
    DynamicsPerturbationKind,
    PerturbedLQRProblemFactory,
)
from ..applications.ood.rehost import rehost_at_horizon
from ..applications.studies import NB06Config, nb06_ood_generalization_experiment
from ..applications.studies.nb04_box_constrained import (
    BoxConstrainedFloors,
    compute_box_constrained_floors,
)
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.runtime import Backend, ComputeContext
from ..core.utils.signing import compute_signature_digest, hash_array
from ..experiments import CachePolicy, ContenderSpec
from ..experiments.cache import CacheReadOnlyMissError
from ..experiments.zero_shot import (
    ShiftMetrics,
    evaluate_under_shift,
    synthesize_nominal_contenders,
)
from ..models.guards import require_linear_quadratic
from ..models.lifecycle import SynthesizedController

#: The statistical-rigor floor (NB06 plan Sec 3.4): single-seed OOD
#: evaluation is forbidden, enforced at `OODSweepSpec` construction.
MIN_OOD_SEEDS = 5

#: Seed-derivation role offsets (Sec 3.4/3.5): the perturbation-draw seed and
#: the evaluation-batch seed are two independent roles, both indexed by the
#: SAME outer seed index but never numerically equal to each other.
_EVAL_SEED_OFFSET = 0
_PERTURBATION_SEED_OFFSET = 100_000


class OODAxis(StrEnum):
    """Which perturbation axis a sweep runs over (NB06 plan Sec 2)."""

    HORIZON = "horizon"
    SCALE = "scale"
    SCALE_NOISE_ONLY = "scale_noise_only"
    DYNAMICS_ADDITIVE = "dynamics_additive"
    DYNAMICS_ROTATION = "dynamics_rotation"
    NOISE_FAMILY = "noise_family"
    CONSTRAINT_BLIND = "constraint_blind"
    CONSTRAINT_AWARE = "constraint_aware"
    CONSTRAINT_NULL = "constraint_null"
    INTERACTION_SCALE_BOUND = "interaction_scale_bound"


class ConstraintProtocol(StrEnum):
    """Which of Axis C's three protocols `run_constraint_ood_sweep` runs
    (NB06 plan Sec 2.6/3.9)."""

    BLIND = "blind"
    AWARE = "aware"
    NULL = "null"


def _derive_seed(base_seed: int, seed_index: int, *, role: str) -> int:
    """One outer `seed_index` deterministically derives BOTH the
    perturbation-draw seed and the evaluation-batch seed (Sec 3.4), offset
    apart so the two roles never collide numerically."""
    offset = _EVAL_SEED_OFFSET if role == "eval" else _PERTURBATION_SEED_OFFSET
    return base_seed + offset + seed_index


@dataclass(frozen=True)
class OODSweepSpec:
    """One perturbation axis's sweep specification.

    Attributes:
        axis: Which axis this spec drives.
        levels: The perturbation levels to sweep, in order -- floats
            (epsilon/degrees/multiplier) for every axis except
            `OODAxis.NOISE_FAMILY`, whose levels are `NoiseFamily` members.
        n_seeds: Seeds per level (>= `MIN_OOD_SEEDS`, enforced below).
        base_seed: The base seed `_derive_seed` offsets from.
        eval_batches_per_seed: Evaluation batches drawn per (level, seed)
            point -- the within-point mean/std `ShiftMetrics` reports.
        contender_levels: Optional per-contender level SUBSET (the sparse
            grid for expensive contenders, e.g. COCP/COCP-LB); a label
            absent here sweeps the full `levels`.
        contender_seeds: Optional per-contender seed-COUNT override, same
            convention as `contender_levels`.
    """

    axis: OODAxis
    levels: tuple[Any, ...]
    n_seeds: int = 7
    base_seed: int = 0
    eval_batches_per_seed: int = 4
    contender_levels: Mapping[str, tuple[Any, ...]] | None = None
    contender_seeds: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        """Fail-fast guards: the statistical-rigor floor and a non-empty
        level grid.

        Raises:
            ValueError: If `n_seeds < MIN_OOD_SEEDS`, or `levels` is empty.
        """
        if self.n_seeds < MIN_OOD_SEEDS:
            raise ValueError(
                f"OODSweepSpec.n_seeds must be >= {MIN_OOD_SEEDS} (single-seed "
                f"OOD evaluation is forbidden), got {self.n_seeds}."
            )
        if not self.levels:
            raise ValueError("OODSweepSpec.levels must be non-empty.")

    def levels_for(self, label: str) -> tuple[Any, ...]:
        """This contender's level grid: its own entry in `contender_levels`
        if present, else the full `levels`."""
        if self.contender_levels is not None and label in self.contender_levels:
            return self.contender_levels[label]
        return self.levels

    def seeds_for(self, label: str) -> int:
        """This contender's seed count: its own entry in `contender_seeds`
        if present, else `n_seeds`."""
        if self.contender_seeds is not None and label in self.contender_seeds:
            return self.contender_seeds[label]
        return self.n_seeds


@dataclass(frozen=True)
class OODPoint:
    """One (level, seed) evaluation of one contender.

    Attributes:
        level: The perturbation level.
        seed: The evaluation seed this point drew its batches from.
        metrics: The `ShiftMetrics` this point scored.
        floor: The SDP floor for this point's own (possibly perturbed)
            problem instance; ``None`` where undefined (`NoiseFamily.CAUCHY`
            has no finite covariance to build a floor from).
    """

    level: Any
    seed: int
    metrics: ShiftMetrics
    floor: BoxConstrainedFloors | None


@dataclass(frozen=True)
class CurveWithBand:
    """A median curve with its IQR band, ready for `viz.plots.ood`'s
    renderers.

    Attributes:
        x: The x-axis coordinates (levels), in order.
        median: The median value at each `x`, across seeds.
        q25: The 25th percentile at each `x`.
        q75: The 75th percentile at each `x`.
    """

    x: tuple[Any, ...]
    median: tuple[float, ...]
    q25: tuple[float, ...]
    q75: tuple[float, ...]


def _level_order(points: tuple[OODPoint, ...]) -> tuple[Any, ...]:
    """The distinct levels in `points`, in FIRST-SEEN order (stable
    regardless of dict/set iteration, since levels may be an unhashable-
    unfriendly mix in principle -- floats/`NoiseFamily` members are both
    hashable here, but order is a presentation concern, not a set concern)."""
    order: list[Any] = []
    for point in points:
        if point.level not in order:
            order.append(point.level)
    return tuple(order)


def _band_over(
    points: tuple[OODPoint, ...], value_fn: Callable[[OODPoint], float | None]
) -> CurveWithBand:
    """`CurveWithBand` of `value_fn(point)` grouped by level, skipping any
    point where `value_fn` returns ``None`` (e.g. an undefined floor)."""
    grouped: dict[Any, list[float]] = {}
    for point in points:
        value = value_fn(point)
        if value is None:
            continue
        grouped.setdefault(point.level, []).append(value)
    present = [level for level in _level_order(points) if level in grouped]
    median = tuple(float(np.median(grouped[level])) for level in present)
    q25 = tuple(float(np.percentile(grouped[level], 25)) for level in present)
    q75 = tuple(float(np.percentile(grouped[level], 75)) for level in present)
    return CurveWithBand(x=tuple(present), median=median, q25=q25, q75=q75)


@dataclass(frozen=True)
class OODSweepResult:
    """One axis's full sweep, ready for `viz.plots.ood`'s renderers.

    Attributes:
        axis: Which axis this result came from.
        points: label -> every `OODPoint` computed for that contender (its
            OWN level/seed grid, per `OODSweepSpec.levels_for`/`seeds_for`).
    """

    axis: OODAxis
    points: Mapping[str, tuple[OODPoint, ...]]

    def cost_band(self, label: str) -> CurveWithBand:
        """The median-cost curve + IQR band across seeds, at each level."""
        return _band_over(self.points[label], lambda point: point.metrics.cost_median)

    def suboptimality_gap_band(self, label: str) -> CurveWithBand:
        """``(cost_median - J_SDP) / J_SDP`` curve + band; a point whose
        floor is undefined (Cauchy) is skipped, not zero-filled."""

        def gap(point: OODPoint) -> float | None:
            if point.floor is None:
                return None
            return (point.metrics.cost_median - point.floor.j_sdp) / point.floor.j_sdp

        return _band_over(self.points[label], gap)

    def saturation_band(self, label: str) -> CurveWithBand:
        """The saturation-rate curve + band (NB06 plan Sec 2.5's
        informative constraint metric)."""
        return _band_over(
            self.points[label], lambda point: point.metrics.saturation_rate
        )

    def tail_band(self, label: str) -> CurveWithBand:
        """The 90th-percentile pooled-per-trajectory-cost curve + band
        (NB06 plan Sec 13.2): zero-shot robustness is a tail property the
        median (`cost_band`) alone can hide."""
        return _band_over(self.points[label], lambda point: point.metrics.cost_q90)

    def cvar_band(self, label: str) -> CurveWithBand:
        """The CVaR-at-the-10%-tail curve + band (NB06 plan Sec 13.2): the
        mean cost of the worst 10% of trajectories at each level."""
        return _band_over(self.points[label], lambda point: point.metrics.cost_cvar10)

    def divergence_band(self, label: str) -> CurveWithBand:
        """The divergence-rate curve + band (NB06 plan Sec 13.2): the
        fraction of trajectories whose state norm exceeded the caller's own
        `divergence_reference_norm` at each level -- ``0.0`` everywhere if
        the sweep never requested it."""
        return _band_over(
            self.points[label], lambda point: point.metrics.divergence_rate
        )

    def normalized_cost_band(self, label: str, *, power: float) -> CurveWithBand:
        """`cost_band` divided by ``level**power`` at each point (NB06 plan
        Sec 2.2's ``J/s**2`` scale-equivariance view, and Sec 2.6/3.9's
        ``J/c**2`` C-null homogeneity check). For a policy still operating
        in its linear regime under Axis S, or for any C-aware-rehosted
        homogeneous contender under Axis C-null, this curve is flat by
        construction -- deviation from flat IS the finding.

        Args:
            label: The contender.
            power: The homogeneity degree to normalize by (``2`` for both
                of NB06's own uses -- LQR cost is quadratic).

        Returns:
            The normalized `CurveWithBand`.

        Raises:
            ZeroDivisionError: If any swept level is exactly ``0`` (the
                normalization is undefined there; NB06 never sweeps a zero
                scale/bound multiplier for this reason).
        """
        band = self.cost_band(label)
        return CurveWithBand(
            x=band.x,
            median=tuple(
                m / (x**power) for m, x in zip(band.median, band.x, strict=True)
            ),
            q25=tuple(q / (x**power) for q, x in zip(band.q25, band.x, strict=True)),
            q75=tuple(q / (x**power) for q, x in zip(band.q75, band.x, strict=True)),
        )

    def max_violation(self, label: str) -> float:
        """The feasibility audit: the worst `max_violation` across every
        point this contender was evaluated at."""
        return max(point.metrics.max_violation for point in self.points[label])

    def interaction_grid(
        self, label: str
    ) -> tuple[tuple[float, ...], tuple[float, ...], np.ndarray]:
        """Pivot an `OODAxis.INTERACTION_SCALE_BOUND` result's ``(scale,
        bound)``-pair levels into a 2-D median-cost grid (NB06 plan Sec
        13.7's interaction cell) -- the one place in this module a `level`
        is a pair rather than a scalar, so it needs its own aggregation
        rather than `_band_over`'s 1-D banding.

        Args:
            label: The contender.

        Returns:
            ``(scale_levels, bound_levels, grid)``, both level tuples
            sorted ascending; ``grid[i, j]`` is the median cost at
            ``(scale_levels[i], bound_levels[j])``, ``NaN`` where that
            combination was never swept for this contender.
        """
        points = self.points[label]
        grouped: dict[tuple[float, float], list[float]] = {}
        for point in points:
            scale, bound = point.level
            grouped.setdefault((float(scale), float(bound)), []).append(
                point.metrics.cost_median
            )
        scale_levels = tuple(sorted({key[0] for key in grouped}))
        bound_levels = tuple(sorted({key[1] for key in grouped}))
        grid = np.full((len(scale_levels), len(bound_levels)), np.nan)
        for i, scale in enumerate(scale_levels):
            for j, bound in enumerate(bound_levels):
                values = grouped.get((scale, bound))
                if values:
                    grid[i, j] = float(np.median(values))
        return scale_levels, bound_levels, grid


def _breakdown_level(gap_band: CurveWithBand) -> float:
    """The first level at which `gap_band`'s median exceeds 2x its
    magnitude at the mildest level -- NB06 plan Sec 5.3/11 G11's "breakdown
    level", the number that makes the degradation table rankable rather
    than only descriptive. ``NaN`` if the band is empty, if the mildest
    gap is (near) zero (undefined -- "2x of zero" is not a meaningful
    threshold), if the gap never crosses it, or if `gap_band.x` is not
    numeric (e.g. `OODAxis.NOISE_FAMILY`'s raw `NoiseFamily`-member levels,
    before `relabel_noise_family_by_distance` converts them to a Hellinger
    distance -- a "breakdown LEVEL" is not a meaningful concept over a
    categorical axis at all, so this degrades to `NaN` rather than raising).

    Args:
        gap_band: Typically `OODSweepResult.suboptimality_gap_band(label)`.

    Returns:
        The first ``x`` at which ``abs(median) > 2 * abs(median[0])``, or
        ``NaN``.
    """
    if not gap_band.x:
        return float("nan")
    baseline = abs(gap_band.median[0])
    if baseline < 1e-12:
        return float("nan")
    threshold = 2.0 * baseline
    for x, median in zip(gap_band.x, gap_band.median, strict=True):
        if abs(median) > threshold:
            try:
                return float(x)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def build_ood_degradation_table(
    results: Mapping[OODAxis, OODSweepResult],
) -> pd.DataFrame:
    """The degradation-summary table (NB06 plan Sec 5.3): per (axis,
    contender), the relative cost change from the mildest to the most
    severe swept level, the saturation rate at both ends, the feasibility
    audit, and the breakdown level -- the paper's Table 1 candidate.

    Args:
        results: axis -> that axis's `OODSweepResult` (e.g. one entry per
            call to `run_horizon_ood_sweep`/`run_scale_ood_sweep`/etc.).

    Returns:
        One row per (axis, contender) with data; columns ``"axis"``,
        ``"contender"``, ``"cost_mildest"``, ``"cost_severest"``,
        ``"relative_degradation"`` (``(severest - mildest) / mildest``,
        ``NaN`` if ``mildest == 0``), ``"saturation_mildest"``,
        ``"saturation_severest"``, ``"max_violation"``,
        ``"breakdown_level"`` (`_breakdown_level` on the suboptimality-gap
        band; ``NaN`` if the gap is undefined everywhere for this
        contender/axis, e.g. Cauchy). A contender with no computed points
        for a given axis contributes no row.
    """
    rows: list[dict[str, Any]] = []
    for axis, result in results.items():
        for label in result.points:
            cost_band = result.cost_band(label)
            if not cost_band.x:
                continue
            mildest, severest = cost_band.median[0], cost_band.median[-1]
            saturation_band = result.saturation_band(label)
            rows.append(
                {
                    "axis": axis.value,
                    "contender": label,
                    "cost_mildest": mildest,
                    "cost_severest": severest,
                    "relative_degradation": (
                        (severest - mildest) / mildest
                        if mildest != 0.0
                        else float("nan")
                    ),
                    "saturation_mildest": (
                        saturation_band.median[0] if saturation_band.x else float("nan")
                    ),
                    "saturation_severest": (
                        saturation_band.median[-1]
                        if saturation_band.x
                        else float("nan")
                    ),
                    "max_violation": result.max_violation(label),
                    "breakdown_level": _breakdown_level(
                        result.suboptimality_gap_band(label)
                    ),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class OODSweepContext:
    """Everything an OOD sweep needs about the nominal experiment, shared
    across every axis and every point.

    Attributes:
        artifacts: label -> the nominally-synthesized (or already-rehosted)
            controller, from `experiments.zero_shot.synthesize_nominal_contenders`.
        contender_specs: label -> the `ContenderSpec` each artifact came
            from -- `rehost_at_horizon`'s dispatch key.
        base_factory: The nominal (unperturbed) problem factory.
        nominal_batch: The nominal evaluation batch spec (its
            `process_noise_std`/`initial_state_std`/`batch_size`/`horizon`
            are the axes' own baselines, scaled or overridden per point).
        ctx: The compute context.
        u_max: The infinity-norm control bound every contender is projected
            onto and audited against.
    """

    artifacts: Mapping[str, SynthesizedController]
    contender_specs: Mapping[str, ContenderSpec]
    base_factory: LQRProblemFactory
    nominal_batch: GaussianBatchSpec
    ctx: ComputeContext
    u_max: float


@dataclass(frozen=True)
class _OODPointInputs:
    """What one (level, seed) point needs to be evaluated: the perturbed
    problem, the batch spec to draw evaluation realizations from, whether
    (and to what horizon/bound) the artifact must first be rehosted, the
    process-noise std the floor should be computed at (``None`` -> no floor,
    e.g. `NoiseFamily.CAUCHY`), and Axis C's own two additional knobs.

    Every axis but Axis C leaves `rehost_bound`/`audit_u_max`/
    `applied_control_bound` at their defaults (``None``), reproducing the
    original H/S/D/X behavior exactly. No axis this module builds ever sets
    BOTH `rehost_horizon` and `rehost_bound` on the same point (each axis
    rehosts at most one dimension at a time) -- `_run_ood_sweep_core`
    dispatches horizon-rehost first, bound-rehost only if horizon-rehost
    was not requested.

    Attributes:
        rehost_bound: Axis C-aware/C-null only (NB06 plan Sec 2.6/3.9): the
            shifted control bound to rehost the artifact at via
            `applications.ood.constraint_shift.rehost_at_bound`; ``None``
            elsewhere.
        audit_u_max: The bound `experiments.zero_shot.evaluate_under_shift`
            audits against; ``None`` defers to `OODSweepContext.u_max` (the
            nominal bound -- correct for every axis but Axis C, which
            audits against its own SHIFTED bound instead).
        applied_control_bound: Axis C-blind only: the bound the PLANT
            enforces via `evaluate_under_shift`'s commanded/applied split;
            ``None`` elsewhere (no external clip).
    """

    problem: OptimalControlProblem
    batch_spec: GaussianBatchSpec | ExoticBatchSpec
    rehost_horizon: int | None
    floor_process_noise_std: float | None
    rehost_bound: float | None = None
    audit_u_max: float | None = None
    applied_control_bound: float | None = None


def _cached_floor(
    cache: dict[tuple[str, str, float, float], BoxConstrainedFloors | None],
    problem: OptimalControlProblem,
    process_noise_std: float | None,
    u_max: float,
) -> BoxConstrainedFloors | None:
    """Memoized `compute_box_constrained_floors`, keyed by the ACTUAL
    perturbed `(A, B, process_noise_std, u_max)` content (Sec 3.7) -- callers
    across every axis and every seed share one cache, so a problem instance
    that recurs (e.g. every seed of the HORIZON/SCALE/NOISE_FAMILY axes,
    which never perturb dynamics) is solved exactly once."""
    if process_noise_std is None:
        return None
    system, _ = require_linear_quadratic(problem)
    key = (
        hash_array(system.A_t[0]),
        hash_array(system.B_t[0]),
        process_noise_std,
        u_max,
    )
    if key not in cache:
        cache[key] = compute_box_constrained_floors(
            problem, process_noise_std=process_noise_std, u_max=u_max
        )
    return cache[key]


def _run_ood_sweep_core(
    context: OODSweepContext,
    spec: OODSweepSpec,
    build_point_inputs: Callable[[Any, int, int], _OODPointInputs],
) -> OODSweepResult:
    """The generic sweep loop every axis wrapper delegates to: for each
    contender, at each of ITS OWN levels (`spec.levels_for`), at each of ITS
    OWN seeds (`spec.seeds_for`), build that point's inputs, rehost if
    needed, evaluate, and record.

    Args:
        context: The shared nominal-experiment context.
        spec: The sweep specification.
        build_point_inputs: ``(level, eval_seed, perturbation_seed) ->
            _OODPointInputs``, closing over whichever axis-specific
            construction (`applications.ood.perturbations`/`.noise`) this
            axis needs; the SAME closure is called for every contender, so
            the seed formula (and therefore the realized batches) never
            depends on the contender label -- the fairness law.

    Returns:
        The `OODSweepResult`.
    """
    floor_cache: dict[tuple[str, str, float, float], BoxConstrainedFloors | None] = {}
    points: dict[str, tuple[OODPoint, ...]] = {}

    for label, artifact in context.artifacts.items():
        contender_spec = context.contender_specs[label]
        label_points: list[OODPoint] = []
        for level in spec.levels_for(label):
            for seed_index in range(spec.seeds_for(label)):
                eval_seed = _derive_seed(spec.base_seed, seed_index, role="eval")
                perturbation_seed = _derive_seed(
                    spec.base_seed, seed_index, role="perturbation"
                )
                inputs = build_point_inputs(level, eval_seed, perturbation_seed)

                sampler, _ = inputs.batch_spec.build(
                    Backend.TORCH,
                    torch_dtype=context.ctx.torch_dtype,
                    torch_device=context.ctx.torch_device,
                )
                batches = tuple(sampler() for _ in range(spec.eval_batches_per_seed))

                artifact_to_use = artifact
                if inputs.rehost_horizon is not None:
                    artifact_to_use = rehost_at_horizon(
                        artifact,
                        contender_spec,
                        inputs.problem,
                        horizon=inputs.rehost_horizon,
                        ctx=context.ctx,
                    )
                elif inputs.rehost_bound is not None:
                    artifact_to_use = rehost_at_bound(
                        artifact,
                        contender_spec,
                        inputs.problem,
                        horizon=context.base_factory.horizon,
                        ctx=context.ctx,
                    )

                audit_bound = (
                    inputs.audit_u_max
                    if inputs.audit_u_max is not None
                    else context.u_max
                )
                metrics, _ = evaluate_under_shift(
                    artifact_to_use,
                    inputs.problem,
                    batches,
                    u_max=audit_bound,
                    applied_control_bound=inputs.applied_control_bound,
                )
                floor = _cached_floor(
                    floor_cache,
                    inputs.problem,
                    inputs.floor_process_noise_std,
                    audit_bound,
                )
                label_points.append(
                    OODPoint(level=level, seed=eval_seed, metrics=metrics, floor=floor)
                )
        points[label] = tuple(label_points)

    return OODSweepResult(axis=spec.axis, points=points)


def run_horizon_ood_sweep(
    context: OODSweepContext, spec: OODSweepSpec
) -> OODSweepResult:
    """Axis H (NB06 plan Sec 2.1): re-solve analytic structure / rehost
    learned parameters at each target horizon, dynamics and noise
    unperturbed. The floor is shared across every level (it does not depend
    on horizon at all)."""
    nominal_std = context.nominal_batch.process_noise_std

    def build(level: Any, eval_seed: int, _perturbation_seed: int) -> _OODPointInputs:
        horizon = int(level)
        problem = PerturbedLQRProblemFactory(
            base=context.base_factory,
            perturbation=DynamicsPerturbation(),
            horizon_override=horizon,
        ).build()
        batch = dataclasses.replace(
            context.nominal_batch, horizon=horizon, seed=eval_seed
        )
        return _OODPointInputs(
            problem=problem,
            batch_spec=batch,
            rehost_horizon=horizon,
            floor_process_noise_std=nominal_std,
        )

    return _run_ood_sweep_core(context, spec, build)


def run_scale_ood_sweep(
    context: OODSweepContext, spec: OODSweepSpec, *, scale_initial_state: bool = True
) -> OODSweepResult:
    """Axis S (NB06 plan Sec 2.2): scale process noise -- and, by default,
    initial-state spread -- jointly by `level`, dynamics unperturbed, no
    rehost.

    Args:
        context: The shared nominal-experiment context.
        spec: The sweep specification (its `axis` should be `OODAxis.SCALE`
            for the default joint variant, `OODAxis.SCALE_NOISE_ONLY` when
            `scale_initial_state=False` -- this function trusts `spec.axis`
            as given and does not override it, matching
            `run_dynamics_ood_sweep`'s own `kind`-parameterized precedent).
        scale_initial_state: If ``True`` (default), scale BOTH
            `process_noise_std` and `initial_state_std` by `level` (Axis
            S-joint). If ``False``, scale `process_noise_std` ONLY, leaving
            `initial_state_std` at its nominal value (Axis S-noise) --
            separates "cannot handle a noisier plant" from "cannot handle
            an unfamiliar initial condition", which S-joint alone confounds.

    Returns:
        The `OODSweepResult`.
    """

    def build(level: Any, eval_seed: int, _perturbation_seed: int) -> _OODPointInputs:
        multiplier = float(level)
        problem = PerturbedLQRProblemFactory(
            base=context.base_factory, perturbation=DynamicsPerturbation()
        ).build()
        scaled_process_std = context.nominal_batch.process_noise_std * multiplier
        scaled_initial_std = (
            context.nominal_batch.initial_state_std * multiplier
            if scale_initial_state
            else context.nominal_batch.initial_state_std
        )
        batch = dataclasses.replace(
            context.nominal_batch,
            process_noise_std=scaled_process_std,
            initial_state_std=scaled_initial_std,
            seed=eval_seed,
        )
        return _OODPointInputs(
            problem=problem,
            batch_spec=batch,
            rehost_horizon=None,
            floor_process_noise_std=scaled_process_std,
        )

    return _run_ood_sweep_core(context, spec, build)


def run_dynamics_ood_sweep(
    context: OODSweepContext,
    spec: OODSweepSpec,
    *,
    kind: DynamicsPerturbationKind,
    perturb_B: bool = False,
) -> OODSweepResult:
    """Axis D (NB06 plan Sec 2.3): perturb `(A, B)` by `level` (additive
    epsilon or rotation degrees, per `kind`), noise/horizon unperturbed, no
    rehost (the Dynamics-OOD law -- the artifact's internal model stays
    nominal while the true plant moves)."""
    nominal_std = context.nominal_batch.process_noise_std

    def build(level: Any, eval_seed: int, perturbation_seed: int) -> _OODPointInputs:
        problem = PerturbedLQRProblemFactory(
            base=context.base_factory,
            perturbation=DynamicsPerturbation(
                kind=kind,
                level=float(level),
                seed=perturbation_seed,
                perturb_B=perturb_B,
            ),
        ).build()
        batch = dataclasses.replace(context.nominal_batch, seed=eval_seed)
        return _OODPointInputs(
            problem=problem,
            batch_spec=batch,
            rehost_horizon=None,
            floor_process_noise_std=nominal_std,
        )

    return _run_ood_sweep_core(context, spec, build)


def run_noise_family_ood_sweep(
    context: OODSweepContext, spec: OODSweepSpec, *, student_t_df: float = 3.0
) -> OODSweepResult:
    """Axis X (NB06 plan Sec 2.4): draw evaluation noise from `level`
    (a `NoiseFamily` member) instead of Gaussian, dynamics/horizon
    unperturbed, no rehost. Every finite-variance family is
    variance-matched to the nominal std by construction
    (`applications.ood.noise.family_scale_parameter`), so they all share
    ONE floor; `NoiseFamily.CAUCHY` has none (undefined covariance)."""

    def build(level: Any, eval_seed: int, _perturbation_seed: int) -> _OODPointInputs:
        family = level
        assert isinstance(family, NoiseFamily)
        problem = PerturbedLQRProblemFactory(
            base=context.base_factory, perturbation=DynamicsPerturbation()
        ).build()
        batch = ExoticBatchSpec(
            state_dim=context.nominal_batch.state_dim,
            horizon=context.nominal_batch.horizon,
            batch_size=context.nominal_batch.batch_size,
            seed=eval_seed,
            process_noise_std=context.nominal_batch.process_noise_std,
            initial_state_std=context.nominal_batch.initial_state_std,
            family=family,
            student_t_df=student_t_df,
        )
        floor_std = (
            None
            if family is NoiseFamily.CAUCHY
            else context.nominal_batch.process_noise_std
        )
        return _OODPointInputs(
            problem=problem,
            batch_spec=batch,
            rehost_horizon=None,
            floor_process_noise_std=floor_std,
        )

    return _run_ood_sweep_core(context, spec, build)


def run_constraint_ood_sweep(
    context: OODSweepContext, spec: OODSweepSpec, *, protocol: ConstraintProtocol
) -> OODSweepResult:
    """Axis C (NB06 plan Sec 2.6/3.9): shift the control bound to
    ``level * nominal_u_max``, dynamics/horizon unperturbed, per `protocol`:

    * `ConstraintProtocol.BLIND` -- the artifact is NOT rehosted; the PLANT
      enforces the shifted bound (`experiments.zero_shot
      .evaluate_under_shift`'s `applied_control_bound`), and the audit is
      measured on the COMMANDED signal against that SAME shifted bound.
    * `ConstraintProtocol.AWARE` -- `applications.ood.constraint_shift
      .rehost_at_bound` re-parameterizes the projection/squash at the
      shifted bound, transplanting every learned parameter; the audit is
      measured directly (no external clip -- the artifact's own bound now
      matches).
    * `ConstraintProtocol.NULL` -- as `AWARE`, but `(sigma, sigma_0)` are
      ALSO scaled by the SAME multiplier (a joint `(u_max, sigma, sigma_0)`
      scale) -- the exact `multiplier**2`-homogeneity check
      (`OODSweepResult.normalized_cost_band(power=2)`), not an ordinary
      degradation sweep.

    Args:
        context: The shared nominal-experiment context.
        spec: The sweep specification; `levels` are bound MULTIPLIERS
            (``1.0`` is the nominal bound -- the degeneracy anchor for every
            protocol).
        protocol: Which of the three protocols to run.

    Returns:
        The `OODSweepResult`.
    """
    nominal_u_max = context.u_max

    def build(level: Any, eval_seed: int, _perturbation_seed: int) -> _OODPointInputs:
        multiplier = float(level)
        shifted_u_max = ControlBoundShift(multiplier=multiplier).bound(nominal_u_max)
        if protocol is ConstraintProtocol.NULL:
            scaled_process_std = context.nominal_batch.process_noise_std * multiplier
            scaled_initial_std = context.nominal_batch.initial_state_std * multiplier
        else:
            scaled_process_std = context.nominal_batch.process_noise_std
            scaled_initial_std = context.nominal_batch.initial_state_std

        problem = PerturbedLQRProblemFactory(
            base=context.base_factory,
            perturbation=DynamicsPerturbation(),
            u_max_override=shifted_u_max,
        ).build()
        batch = dataclasses.replace(
            context.nominal_batch,
            process_noise_std=scaled_process_std,
            initial_state_std=scaled_initial_std,
            seed=eval_seed,
        )
        rehost_bound = (
            shifted_u_max
            if protocol in (ConstraintProtocol.AWARE, ConstraintProtocol.NULL)
            else None
        )
        applied_bound = shifted_u_max if protocol is ConstraintProtocol.BLIND else None
        return _OODPointInputs(
            problem=problem,
            batch_spec=batch,
            rehost_horizon=None,
            floor_process_noise_std=scaled_process_std,
            rehost_bound=rehost_bound,
            audit_u_max=shifted_u_max,
            applied_control_bound=applied_bound,
        )

    return _run_ood_sweep_core(context, spec, build)


def run_interaction_scale_bound_sweep(
    context: OODSweepContext, spec: OODSweepSpec
) -> OODSweepResult:
    """The scale x bound interaction cell (NB06 plan Sec 13.7): `spec
    .levels` are ``(scale_multiplier, bound_multiplier)`` PAIRS -- does a
    noisier environment and a de-rated actuator degrade cost
    super-additively, or does each move it independently (an equally
    legible null result)? A BLIND deployment scenario, deliberately: the
    controller's trained belief about `u_max` is left untouched while the
    true environment (both the noise scale AND the bound the plant
    enforces) moves -- `experiments.zero_shot.evaluate_under_shift`'s
    `applied_control_bound`, exactly Axis C-blind's own mechanism, reused
    here rather than re-derived. No rehost.

    Args:
        context: The shared nominal-experiment context.
        spec: The sweep specification; `levels` are ``(scale, bound)``
            multiplier pairs (``(1.0, 1.0)`` is the nominal point).

    Returns:
        The `OODSweepResult` -- read via `OODSweepResult.interaction_grid`,
        not the 1-D band methods (a `level` here is a pair, not a scalar).
    """
    nominal_u_max = context.u_max

    def build(level: Any, eval_seed: int, _perturbation_seed: int) -> _OODPointInputs:
        scale_multiplier, bound_multiplier = level
        shifted_u_max = ControlBoundShift(multiplier=float(bound_multiplier)).bound(
            nominal_u_max
        )
        scaled_process_std = context.nominal_batch.process_noise_std * float(
            scale_multiplier
        )
        scaled_initial_std = context.nominal_batch.initial_state_std * float(
            scale_multiplier
        )
        problem = PerturbedLQRProblemFactory(
            base=context.base_factory,
            perturbation=DynamicsPerturbation(),
            u_max_override=shifted_u_max,
        ).build()
        batch = dataclasses.replace(
            context.nominal_batch,
            process_noise_std=scaled_process_std,
            initial_state_std=scaled_initial_std,
            seed=eval_seed,
        )
        return _OODPointInputs(
            problem=problem,
            batch_spec=batch,
            rehost_horizon=None,
            floor_process_noise_std=scaled_process_std,
            audit_u_max=shifted_u_max,
            applied_control_bound=shifted_u_max,
        )

    return _run_ood_sweep_core(context, spec, build)


def relabel_noise_family_by_distance(
    result: OODSweepResult, *, sigma: float, student_t_df: float = 3.0
) -> OODSweepResult:
    """Remap a `run_noise_family_ood_sweep` result's categorical
    `NoiseFamily` levels to their Hellinger distance from the nominal
    Gaussian (NB06 plan Sec 2.4/5.1): a family NAME is not a point on a
    numeric line, but its statistical distance from the nominal distribution
    is -- and that distance, not the family label, is what every noise-
    family figure plots against (`viz.plots.ood`'s renderers require a
    numeric ``x``, by construction, for every axis).

    Args:
        result: The `OODAxis.NOISE_FAMILY` result to relabel (any other
            axis's result already has numeric levels and does not need this).
        sigma: The nominal process-noise standard deviation the sweep's own
            families were standardized against (matches the value
            `run_noise_family_ood_sweep`'s caller used for
            `context.nominal_batch.process_noise_std`).
        student_t_df: `NoiseFamily.STUDENT_T`'s degrees of freedom, matching
            whatever `student_t_df` the sweep itself was run with.

    Returns:
        A new `OODSweepResult` with every `OODPoint.level` replaced by its
        Hellinger distance (a ``float``); every other field (`metrics`,
        `seed`, `floor`) is carried through unchanged.
    """
    remapped: dict[str, tuple[OODPoint, ...]] = {}
    for label, points in result.points.items():
        remapped_points = []
        for point in points:
            assert isinstance(point.level, NoiseFamily)
            hellinger = distance_to_nominal_gaussian(
                point.level, sigma=sigma, df=student_t_df
            ).hellinger
            remapped_points.append(dataclasses.replace(point, level=hellinger))
        remapped[label] = tuple(remapped_points)
    return OODSweepResult(axis=result.axis, points=remapped)


def run_ood_depth_ablation(
    base_config: NB06Config,
    depths: Sequence[int],
    scale_spec: OODSweepSpec,
    *,
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
    labels: Sequence[str] | None = None,
) -> dict[int, OODSweepResult]:
    """The "over-thinking" ablation (NB06 plan Sec 3.6): for each unfolding
    depth ``K`` in `depths`, train (or replay from cache) the nominal NB06
    experiment at that depth via the existing per-contender cache, then run
    `scale_spec`'s SCALE-axis sweep against its freshly-synthesized
    contenders -- the axis where the "a shallow unrolling may generalize
    better under heavy noise than a deep one" hypothesis is sharpest.

    Args:
        base_config: The NB06 configuration every depth is drawn from; only
            `num_unfolding_iterations` varies per depth (via
            `dataclasses.replace`) -- every other setting (dimensions,
            horizon, `u_max`, seeds, per-family plans) stays fixed.
        depths: The unfolding depths ``K`` to ablate, e.g. ``(3, 5, 10)``.
        scale_spec: The SCALE-axis `OODSweepSpec` run at every depth.
        root: The experiments/artifacts root nominal training caches under.
        policy: Per-invocation cache control for the nominal training pass;
            defaults to ``CachePolicy()`` (``"reuse"``).
        labels: Optional restriction to a subset of contender labels (e.g.
            just ``("unfolded_alpha", "unfolded_alpha_p")``, the only two
            contenders whose synthesis genuinely varies with depth) --
            every other NB06 contender is depth-invariant and would
            otherwise be OOD-re-evaluated identically at every depth for no
            reason (its nominal TRAINING is still served by the cache
            either way; only the redundant OOD sweep re-evaluation is
            avoided). ``None`` (the default) sweeps every contender.

    Returns:
        depth -> that depth's `OODSweepResult`.
    """
    results: dict[int, OODSweepResult] = {}
    for depth in depths:
        cfg = dataclasses.replace(base_config, num_unfolding_iterations=depth)
        experiment = nb06_ood_generalization_experiment(cfg)
        artifacts, _ = synthesize_nominal_contenders(
            experiment, root=root, policy=policy
        )
        contender_specs = {spec.resolved_label: spec for spec in experiment.contenders}
        if labels is not None:
            artifacts = {label: artifacts[label] for label in labels}
            contender_specs = {label: contender_specs[label] for label in labels}
        # `EvaluationProtocol.batch_spec` is typed at the wider `BatchSpec`
        # Protocol (NB07's own widening), but every NB06 declaration builds
        # it concretely as a `GaussianBatchSpec` (`nb06_ood_generalization_
        # experiment`) -- narrowed here since `OODSweepContext.nominal_batch`
        # needs the concrete dataclass for `dataclasses.replace` in every
        # axis wrapper above.
        nominal_batch = experiment.evaluation.batch_spec
        assert isinstance(nominal_batch, GaussianBatchSpec)
        context = OODSweepContext(
            artifacts=artifacts,
            contender_specs=contender_specs,
            base_factory=LQRProblemFactory(
                state_dim=cfg.state_dim,
                control_dim=cfg.control_dim,
                horizon=cfg.horizon,
                seed=cfg.seed,
                u_max=cfg.u_max,
            ),
            nominal_batch=nominal_batch,
            ctx=experiment.ctx,
            u_max=cfg.u_max,
        )
        results[depth] = run_scale_ood_sweep(context, scale_spec)
    return results


# --- OOD-result persistence (NB06 plan Sec 3.10) ----------------------------
#
# Nominal TRAINING is already content-keyed and cached (`run_experiment`);
# OOD *evaluation* is not -- every notebook re-run re-executes every sweep
# from scratch, which dominates wall time once Axis C's three protocols and
# the depth ablation are added. `run_ood_sweep_cached` closes that gap with
# the SAME `Signable`/digest discipline every other Tier-3 object in this
# project uses (`experiments.experiment.Experiment.signature_digest`), never
# a bespoke pickle: the signature is computed from `(context, spec)` BEFORE
# any rollout, so a cache hit costs nothing but a JSON read.


def _serializable_level(level: Any) -> Any:
    """`NoiseFamily` levels serialize as their plain string value; every
    other axis's levels are already JSON-native (`float`/`int`)."""
    return level.value if isinstance(level, NoiseFamily) else level


def _ood_sweep_signature(
    context: OODSweepContext, spec: OODSweepSpec
) -> dict[str, Any]:
    """The content signature `(context, spec)` determine, computed BEFORE
    running anything -- the `Signable` convention `Experiment.get_signature`
    already establishes, extended to an OOD sweep's own inputs: the axis,
    every spec field, the base problem/batch, the audited bound, and every
    contributing artifact's OWN signature (so retraining any contender
    invalidates every cached sweep that used it)."""
    return {
        "type": "OODSweepSignature",
        "axis": spec.axis.value,
        "levels": [_serializable_level(level) for level in spec.levels],
        "n_seeds": spec.n_seeds,
        "base_seed": spec.base_seed,
        "eval_batches_per_seed": spec.eval_batches_per_seed,
        "contender_levels": (
            {
                label: [_serializable_level(level) for level in levels]
                for label, levels in spec.contender_levels.items()
            }
            if spec.contender_levels is not None
            else None
        ),
        "contender_seeds": (
            dict(spec.contender_seeds) if spec.contender_seeds is not None else None
        ),
        "base_factory": context.base_factory.get_signature(),
        "nominal_batch": context.nominal_batch.get_signature(),
        "u_max": context.u_max,
        "artifacts": {
            label: artifact.get_signature()
            for label, artifact in context.artifacts.items()
        },
    }


def ood_sweep_signature_digest(context: OODSweepContext, spec: OODSweepSpec) -> str:
    """Compact composite identity of one `(context, spec)` sweep -- what
    `run_ood_sweep_cached` keys its persisted result under."""
    return compute_signature_digest(_ood_sweep_signature(context, spec))


def _serialize_shift_metrics(metrics: ShiftMetrics) -> dict[str, Any]:
    return dataclasses.asdict(metrics)


def _deserialize_shift_metrics(data: Mapping[str, Any]) -> ShiftMetrics:
    return ShiftMetrics(**data)


def _serialize_floor(floor: BoxConstrainedFloors | None) -> dict[str, Any] | None:
    if floor is None:
        return None
    return {
        "j_lqr": floor.j_lqr,
        "j_sdp": floor.j_sdp,
        "p_lqr": np.asarray(floor.p_lqr).tolist(),
        "p_sdp": np.asarray(floor.p_sdp).tolist(),
    }


def _deserialize_floor(
    data: Mapping[str, Any] | None,
) -> BoxConstrainedFloors | None:
    if data is None:
        return None
    return BoxConstrainedFloors(
        j_lqr=data["j_lqr"],
        j_sdp=data["j_sdp"],
        p_lqr=np.asarray(data["p_lqr"]),
        p_sdp=np.asarray(data["p_sdp"]),
    )


def serialize_ood_sweep_result(result: OODSweepResult) -> dict[str, Any]:
    """A JSON-serializable representation of `result` (NB06 plan Sec 3.10).

    Args:
        result: The `OODSweepResult` to serialize.

    Returns:
        A nested, JSON-native dict; round-trips exactly through
        `deserialize_ood_sweep_result`.
    """
    return {
        "axis": result.axis.value,
        "points": {
            label: [
                {
                    "level": _serializable_level(point.level),
                    "seed": point.seed,
                    "metrics": _serialize_shift_metrics(point.metrics),
                    "floor": _serialize_floor(point.floor),
                }
                for point in points
            ]
            for label, points in result.points.items()
        },
    }


def deserialize_ood_sweep_result(data: Mapping[str, Any]) -> OODSweepResult:
    """The inverse of `serialize_ood_sweep_result`.

    Args:
        data: A dict previously produced by `serialize_ood_sweep_result`
            (e.g. round-tripped through JSON).

    Returns:
        The reconstructed `OODSweepResult`.
    """
    axis = OODAxis(data["axis"])
    points = {
        label: tuple(
            OODPoint(
                level=(
                    NoiseFamily(entry["level"])
                    if axis is OODAxis.NOISE_FAMILY
                    else entry["level"]
                ),
                seed=entry["seed"],
                metrics=_deserialize_shift_metrics(entry["metrics"]),
                floor=_deserialize_floor(entry["floor"]),
            )
            for entry in entries
        )
        for label, entries in data["points"].items()
    }
    return OODSweepResult(axis=axis, points=points)


def run_ood_sweep_cached(
    sweep_fn: Callable[[OODSweepContext, OODSweepSpec], OODSweepResult],
    context: OODSweepContext,
    spec: OODSweepSpec,
    *,
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
) -> OODSweepResult:
    """Content-keyed persistence for one axis's sweep (NB06 plan Sec 3.10):
    the SAME cache discipline `run_experiment` applies to nominal training,
    applied here to OOD evaluation instead -- a genuinely separate cache
    (`<root>/ood_sweeps/`), never a reuse of the training cache's own store.

    Args:
        sweep_fn: One of `run_horizon_ood_sweep`/`run_scale_ood_sweep`/
            `run_dynamics_ood_sweep`/`run_noise_family_ood_sweep`/
            `run_constraint_ood_sweep`, called as ``sweep_fn(context,
            spec)`` -- pre-bind any extra keyword the axis needs (e.g.
            `kind=`/`protocol=`/`scale_initial_state=`) via
            `functools.partial` before passing it here.
        context: The shared nominal-experiment context.
        spec: The sweep specification.
        root: The experiments root the cache nests under (a sibling
            ``ood_sweeps/`` directory next to ``run_*``/``notebook_figures``).
        policy: `CachePolicy()` (``"reuse"``, default) replays a cached
            result without calling `sweep_fn` at all; ``"recompute"`` always
            calls `sweep_fn` and overwrites the cached entry; ``"readonly"``
            replays on a hit and raises `CacheReadOnlyMissError` on a miss.

    Returns:
        The `OODSweepResult` -- replayed from disk on a cache hit, computed
        (and persisted) fresh otherwise.

    Raises:
        CacheReadOnlyMissError: On a miss under `CachePolicy("readonly")`.
    """
    digest = ood_sweep_signature_digest(context, spec)
    cache_path = Path(root) / "ood_sweeps" / f"{spec.axis.value}__{digest}.json"

    if cache_path.exists() and policy.mode in ("reuse", "readonly"):
        with cache_path.open("r") as handle:
            return deserialize_ood_sweep_result(json.load(handle))

    if policy.mode == "readonly":
        raise CacheReadOnlyMissError(
            f"OOD sweep for axis {spec.axis.value!r} (digest {digest}) is not "
            "cached and the cache policy is readonly -- nothing may be "
            "computed. Rerun under CachePolicy('reuse') to compute and store it."
        )

    result = sweep_fn(context, spec)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        json.dump(serialize_ood_sweep_result(result), handle)
    return result
