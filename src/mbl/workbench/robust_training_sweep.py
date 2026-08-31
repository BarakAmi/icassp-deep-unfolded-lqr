"""The robust-training sweep engine (docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md
Sec 5.4/13): drives one uncertainty axis across its levels and seeds for
every one of NB07's SIX training-sweep contenders (`cocp` is excluded from
every sweep -- Sec 13 decision 2), and returns one frozen `RobustTrainingResult`
the notebook renders -- the established `run_depth_sweep_study` contract
(`workbench.depth_sweep`): a generic core parameterized by a per-level
builder callback, zero `for`-loops of its own in the caller.

**Why this does NOT reuse `run_experiment`** (a deliberate, documented
deviation from the plan's original `run_ood_sweep_study`-shaped pseudocode,
resolved with the user before implementation): `run_experiment` synthesizes
a contender and returns only its PERSISTED metrics/arrays -- never the live
`SynthesizedController` -- so it cannot serve the "evaluate the SAME trained
artifact on a SECOND (nominal) problem" need the price-of-robustness study
requires. It also has no seam for attaching an extra `Callback`
(`DomainRandomizationCallback`) to a recipe's internally-built `Runner`. This
module therefore drives `recipe.build_controller`/`build_engine` directly --
the same two calls `EngineTrainedSynthesizer.synthesize` makes internally --
and appends the DR callback to the returned `Runner.callbacks` list (a plain
mutable attribute) before calling ``.train()``. Both evaluation passes (on
the condition's own instance, and on the nominal instance) happen against the
SAME live artifact, in the SAME process, before anything is discarded.

Caching is NOT `run_experiment`'s (a full live-artifact rehydration loader
does not exist in the tree yet, per the NB06 plan's own P2 note) -- instead,
every training-and-evaluation POINT bakes both `ShiftMetrics` (condition and
nominal), the per-epoch learning curve, and (when present) the learned
Riccati-replacement matrix into ONE cached payload via the existing, reused
`TrackerBackedExperimentCache` primitive. A cache hit therefore needs no live
artifact at all -- every downstream figure reads only cached numbers.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..applications.recipes.base import (
    EngineHarness,
    ModelRecipe,
    TrainedControllerArtifact,
)
from ..applications.uncertainty.callback import DomainRandomizationCallback
from ..applications.uncertainty.perturbations import PerturbationDistribution
from ..applications.uncertainty.resync import supports_oracle_resync
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.runtime import ComputeContext
from ..core.utils.signing import compute_signature_digest
from ..engine.runner import Runner
from ..experiments.cache import (
    CachePolicy,
    CacheReadOnlyMissError,
    ContenderPayload,
    TrackerBackedExperimentCache,
    compose_cache_key,
)
from ..experiments.provenance import code_provenance_stamp
from ..experiments.run_logging import bind_run_log
from ..experiments.shifted_evaluation import ShiftMetrics, evaluate_under_shift

logger = logging.getLogger(__name__)

#: The mandate's statistical-rigor gate (NB07 plan Sec 5.4): single-seed
#: training is forbidden, enforced at ingress rather than left as a
#: convention someone can quietly violate.
MIN_TRAINING_SEEDS = 5


class UncertaintyAxis(StrEnum):
    """The six training-uncertainty axes (NB07 plan Sec 4.4)."""

    NOISE_SCALE = "noise_scale"
    NOISE_FAMILY = "noise_family"
    PLANT_ADDITIVE = "plant_additive"
    PLANT_ROTATION = "plant_rotation"
    TRAIN_HORIZON = "train_horizon"
    SAMPLE_SIZE = "sample_size"


class ModelAccess(StrEnum):
    """The model-access law (NB07 plan Sec 4.2): whether a learned
    contender's internal model is re-synchronized to the perturbed plant
    every epoch (`ORACLE`) or left describing the nominal plant while the
    simulated plant drifts underneath it (`NOMINAL`)."""

    ORACLE = "oracle"
    NOMINAL = "nominal"


@dataclass(frozen=True)
class TrainingCondition:
    """One fully-resolved (axis, level) training/evaluation condition -- the
    study module's responsibility to build; this module's responsibility to
    execute and cache.

    Attributes:
        training_problem: The problem each recipe trains against. For the
            two plant axes this is the SAME nominal problem object
            `dr_distribution` will mutate in place during training (and
            restore afterward, `DomainRandomizationCallback`'s own
            contract) -- never itself the perturbed instance.
        training_harness: The harness (batch spec + tracker + ctx) each
            recipe's training sampler and engine are built from.
        eval_problem: The FIXED instance every contender's freshly-trained
            artifact is scored against for the "trained-and-evaluated
            together" figure -- deliberately independent of whatever the
            live `training_problem.system` happens to hold post-training
            (see the module docstring's `DomainRandomizationCallback`
            restoration note).
        eval_batches: Evaluation batches drawn from `eval_problem`'s own
            noise law, under a seed distinct from the training data seed
            (the fairness law's eval-seed role, NB07 plan Sec 5.4).
        dr_distribution: The plant-perturbation stream for the two plant
            axes; ``None`` for every other axis (no per-epoch system
            mutation needed -- the level is already baked into
            `training_problem`/`training_harness`).
        nominal_override: ``(problem, eval_batches)`` to score the
            price-of-robustness pass against INSTEAD of the sweep's global
            ``nominal_problem``/``nominal_batches`` -- ``None`` for every
            axis except `TRAIN_HORIZON`. That axis's trained artifact is
            sized for its OWN level's horizon (a controller's gain/refinement
            stacks have exactly `level` entries), so evaluating it against
            the global nominal problem's own (different) horizon indexes
            past the end of those stacks; the override supplies a problem at
            the SAME horizon the artifact was actually built for, under
            otherwise-nominal noise/plant, so the pass stays well-posed
            instead of raising an `IndexError` partway through the sweep.
    """

    training_problem: OptimalControlProblem
    training_harness: EngineHarness
    eval_problem: OptimalControlProblem
    eval_batches: tuple[Any, ...]
    dr_distribution: PerturbationDistribution | None = None
    nominal_override: tuple[OptimalControlProblem, tuple[Any, ...]] | None = None


@dataclass(frozen=True)
class RobustTrainingSpec:
    """One axis's sweep declaration.

    Attributes:
        axis: Which uncertainty axis this sweep drives.
        levels: The severity levels to sweep, in order (their meaning is
            axis-specific: an epsilon, a degree angle, a noise family, an
            integer horizon, or a trajectory count).
        model_access: `ModelAccess.ORACLE` or `.NOMINAL` (NB07 plan Sec 4.2).
        n_seeds: Seeds per level; must be ``>= MIN_TRAINING_SEEDS``.
        base_seed: The seed the per-seed streams (train-data, perturbation,
            init) are derived from -- ``base_seed + seed_index``, kept
            distinct from the fixed, shared `eval_seed` (the fairness law).
    """

    axis: UncertaintyAxis
    levels: tuple[Any, ...]
    model_access: ModelAccess
    n_seeds: int
    base_seed: int

    def __post_init__(self) -> None:
        if self.n_seeds < MIN_TRAINING_SEEDS:
            raise ValueError(
                f"RobustTrainingSpec requires n_seeds >= {MIN_TRAINING_SEEDS} "
                f"(the statistical-rigor mandate, NB07 plan Sec 5.4/8); "
                f"got {self.n_seeds}."
            )


@dataclass(frozen=True)
class RobustTrainingPoint:
    """One (contender, level, seed) result -- either freshly computed or
    served from cache; both paths populate every field identically.

    Attributes:
        level: The severity level this point was trained/evaluated at.
        seed: The seed index (0-based, added to `RobustTrainingSpec.base_seed`).
        trained_metrics: Scored on `TrainingCondition.eval_problem` (the
            "trained-and-evaluated-together" figure).
        nominal_metrics: The SAME trained artifact, scored on the clean
            nominal problem (the price-of-robustness figure).
        learning_curve: Per-epoch training loss, in order.
        learned_p: The learned Riccati-replacement matrix (UF-alpha+P only),
            or ``None`` for every other family.
        disposition: ``"hit"`` (served from cache) or ``"fresh"`` (computed
            this call) -- the cache-reuse audit trail.
        effective_model_access: The `ModelAccess` this SPECIFIC point
            actually trained under. Equal to the sweep's declared
            `RobustTrainingSpec.model_access` for every contender except the
            plan's one declared exception (NB07 plan Sec 4.2): under
            `ORACLE`, a contender whose controller has no cheap
            resynchronizer (`applications.uncertainty.resync
            .supports_oracle_resync` is `False` -- COCP-backed families)
            trains `NOMINAL` for itself while the rest of the sweep stays
            `ORACLE`, rather than raising and aborting every other
            contender's run.
    """

    level: Any
    seed: int
    trained_metrics: ShiftMetrics
    nominal_metrics: ShiftMetrics
    learning_curve: tuple[float, ...]
    learned_p: np.ndarray | None
    disposition: str
    effective_model_access: ModelAccess


@dataclass(frozen=True)
class RobustTrainingResult:
    """The full sweep result for one axis, ready for Sec 6's renderers.

    Attributes:
        axis: The swept axis.
        model_access: The model-access mode every point was trained under.
        levels: The levels swept, in order.
        points: contender label -> one `RobustTrainingPoint` per
            ``(level, seed)`` pair, in level-major order.
    """

    axis: UncertaintyAxis
    model_access: ModelAccess
    levels: tuple[Any, ...]
    points: Mapping[str, tuple[RobustTrainingPoint, ...]]

    def points_at_level(
        self, label: str, level: Any
    ) -> tuple[RobustTrainingPoint, ...]:
        """Every seed's `RobustTrainingPoint` for `label` at `level`.

        Args:
            label: The contender label.
            level: The severity level.

        Returns:
            The matching points, in seed order.
        """
        return tuple(p for p in self.points[label] if p.level == level)

    def cost_band(self, label: str, *, on: str = "trained") -> "CurveWithBand":
        """Median + IQR (across seeds) of the per-point pooled median cost,
        one value per level (NB07 plan Sec 8's statistical protocol: median
        is the primary statistic everywhere, never the mean -- undefined
        under Cauchy noise).

        Args:
            label: The contender label.
            on: ``"trained"`` (Sec 6.1's learning-dynamics/degradation
                figures -- scored on the condition's own instance) or
                ``"nominal"`` (Sec 6.2's price-of-robustness figure --
                the SAME trained artifact, scored on the clean problem).

        Returns:
            The `CurveWithBand`, one entry per level present for `label`.

        Raises:
            ValueError: If `on` is neither ``"trained"`` nor ``"nominal"``.
        """
        if on not in ("trained", "nominal"):
            raise ValueError(f"on must be 'trained' or 'nominal', got {on!r}.")
        grouped: dict[Any, list[float]] = {}
        for point in self.points[label]:
            metrics = (
                point.trained_metrics if on == "trained" else point.nominal_metrics
            )
            grouped.setdefault(point.level, []).append(metrics.cost_median)
        return _band_over_levels(self.levels, grouped)

    def saturation_band(self, label: str, *, on: str = "trained") -> "CurveWithBand":
        """Median + IQR (across seeds) of the saturation rate, one value per
        level -- the constraint panel every cost figure is paired with
        (NB07 plan Sec 4.6).

        Args:
            label: The contender label.
            on: See `cost_band`.

        Returns:
            The `CurveWithBand`.
        """
        if on not in ("trained", "nominal"):
            raise ValueError(f"on must be 'trained' or 'nominal', got {on!r}.")
        grouped: dict[Any, list[float]] = {}
        for point in self.points[label]:
            metrics = (
                point.trained_metrics if on == "trained" else point.nominal_metrics
            )
            grouped.setdefault(point.level, []).append(metrics.saturation_rate)
        return _band_over_levels(self.levels, grouped)

    def learning_curve_band(self, label: str, level: Any) -> "CurveWithBand":
        """Median + IQR (across seeds) of the per-epoch training loss at one
        level -- Sec 6.1's learning-dynamics figure. Seeds' curves are
        truncated to their common minimum length before aggregating (every
        recipe in a sweep trains for the SAME declared epoch count, so a
        length mismatch only arises from a declared divergence, which is
        recorded upstream, never silently padded here).

        Args:
            label: The contender label.
            level: The severity level.

        Returns:
            The `CurveWithBand`, x-axis = 1-based epoch index.
        """
        curves = [
            np.asarray(point.learning_curve)
            for point in self.points_at_level(label, level)
            if point.learning_curve
        ]
        if not curves:
            return CurveWithBand(x=(), median=(), q25=(), q75=())
        min_len = min(len(curve) for curve in curves)
        trimmed = [curve[:min_len] for curve in curves]
        epochs = tuple(range(1, min_len + 1))
        grouped = {
            epoch: [curve[i] for curve in trimmed] for i, epoch in enumerate(epochs)
        }
        return _band_over_levels(epochs, grouped)

    def isotropy_band(self, label: str) -> "CurveWithBand":
        """Median + IQR (across seeds) of the learned-P isotropy index, one
        value per level -- Sec 6.4's spectral-analysis figure. Empty for a
        contender with no learned matrix (every family but UF-alpha+P).

        Args:
            label: The contender label.

        Returns:
            The `CurveWithBand`; empty (all-``()``) if `label` never
            reports a `learned_p`.
        """
        from .spectral_analysis import compute_p_spectrum

        grouped: dict[Any, list[float]] = {}
        for point in self.points[label]:
            if point.learned_p is None:
                continue
            spectrum = compute_p_spectrum(point.learned_p)
            grouped.setdefault(point.level, []).append(spectrum.isotropy_index)
        if not grouped:
            return CurveWithBand(x=(), median=(), q25=(), q75=())
        return _band_over_levels(self.levels, grouped)

    def spectral_norm_band(self, label: str) -> "CurveWithBand":
        """Median + IQR (across seeds) of the learned-P spectral norm, one
        value per level -- Sec 6.4's spectral-analysis figure. Empty for a
        contender with no learned matrix (every family but UF-alpha+P).

        Args:
            label: The contender label.

        Returns:
            The `CurveWithBand`; empty (all-``()``) if `label` never
            reports a `learned_p`.
        """
        from .spectral_analysis import compute_p_spectrum

        grouped: dict[Any, list[float]] = {}
        for point in self.points[label]:
            if point.learned_p is None:
                continue
            spectrum = compute_p_spectrum(point.learned_p)
            grouped.setdefault(point.level, []).append(spectrum.spectral_norm)
        if not grouped:
            return CurveWithBand(x=(), median=(), q25=(), q75=())
        return _band_over_levels(self.levels, grouped)

    def condition_number_band(self, label: str) -> "CurveWithBand":
        """Median + IQR (across seeds) of the learned-P condition number,
        one value per level -- Sec 6.4's spectral-analysis figure. A seed
        whose smallest eigenvalue magnitude is exactly zero reports
        `PSpectrum.condition_number == math.inf`; such points are excluded
        from the band (an infinite value has no meaningful median/IQR
        contribution), not silently treated as zero. Empty for a contender
        with no learned matrix.

        Args:
            label: The contender label.

        Returns:
            The `CurveWithBand`; empty (all-``()``) if `label` never
            reports a `learned_p`, or if every reported condition number
            is infinite.
        """
        from .spectral_analysis import compute_p_spectrum

        grouped: dict[Any, list[float]] = {}
        for point in self.points[label]:
            if point.learned_p is None:
                continue
            spectrum = compute_p_spectrum(point.learned_p)
            if not np.isfinite(spectrum.condition_number):
                continue
            grouped.setdefault(point.level, []).append(spectrum.condition_number)
        if not grouped:
            return CurveWithBand(x=(), median=(), q25=(), q75=())
        return _band_over_levels(self.levels, grouped)

    def eigenvalue_bands(self, label: str) -> tuple["CurveWithBand", ...]:
        """Median + IQR (across seeds) of each eigenvalue RANK of the
        learned P, one `CurveWithBand` per rank (Sec 6.4's "all n
        eigenvalues, banded" spectrum figure) -- index 0 is the largest
        eigenvalue's band, the last index the smallest. Empty tuple for a
        contender with no learned matrix.

        Args:
            label: The contender label.

        Returns:
            One `CurveWithBand` per eigenvalue rank, descending; `()` if
            `label` never reports a `learned_p`.
        """
        from .spectral_analysis import compute_p_spectrum

        grouped_by_rank: list[dict[Any, list[float]]] = []
        for point in self.points[label]:
            if point.learned_p is None:
                continue
            spectrum = compute_p_spectrum(point.learned_p)
            if not grouped_by_rank:
                grouped_by_rank = [{} for _ in spectrum.eigenvalues]
            for rank, value in enumerate(spectrum.eigenvalues):
                grouped_by_rank[rank].setdefault(point.level, []).append(value)
        return tuple(
            _band_over_levels(self.levels, grouped) for grouped in grouped_by_rank
        )

    def principal_angle_band(
        self, label: str, reference: np.ndarray
    ) -> "CurveWithBand":
        """Median + IQR (across seeds) of the principal angle (degrees,
        ``k=1`` -- the dominant eigenspace only) between the learned P and
        `reference` (typically the nominal DARE `P*`), one value per level
        (Sec 6.4). Empty for a contender with no learned matrix.

        Args:
            label: The contender label.
            reference: The fixed matrix every point's `learned_p` is
                compared against (e.g. the nominal problem's DARE solution).

        Returns:
            The `CurveWithBand`; empty (all-``()``) if `label` never
            reports a `learned_p`.
        """
        from .spectral_analysis import compute_p_spectrum

        grouped: dict[Any, list[float]] = {}
        for point in self.points[label]:
            if point.learned_p is None:
                continue
            spectrum = compute_p_spectrum(point.learned_p, reference=reference, k=1)
            angles = cast("tuple[float, ...]", spectrum.principal_angles_deg)
            grouped.setdefault(point.level, []).append(angles[0])
        if not grouped:
            return CurveWithBand(x=(), median=(), q25=(), q75=())
        return _band_over_levels(self.levels, grouped)


@dataclass(frozen=True)
class CurveWithBand:
    """A median curve with its IQR band, ready for `viz.plots
    .robust_training`'s renderers.

    Attributes:
        x: The x-axis coordinates (levels or epoch indices), in order.
        median: The median value at each `x`.
        q25: The 25th percentile at each `x`.
        q75: The 75th percentile at each `x`.
    """

    x: tuple[Any, ...]
    median: tuple[float, ...]
    q25: tuple[float, ...]
    q75: tuple[float, ...]


def _band_over_levels(
    levels: tuple[Any, ...], grouped: Mapping[Any, list[float]]
) -> CurveWithBand:
    """`CurveWithBand` from level -> list-of-per-seed-values, preserving
    `levels`' own order and skipping any level with no data."""
    present = [level for level in levels if level in grouped]
    medians = tuple(float(np.median(grouped[level])) for level in present)
    q25s = tuple(float(np.percentile(grouped[level], 25)) for level in present)
    q75s = tuple(float(np.percentile(grouped[level], 75)) for level in present)
    return CurveWithBand(x=tuple(present), median=medians, q25=q25s, q75=q75s)


def _condition_signature(  # noqa: PLR0913 -- every field participates in the point's cache key
    recipe: ModelRecipe,
    condition: TrainingCondition,
    *,
    axis: UncertaintyAxis,
    level: Any,
    seed: int,
    model_access: ModelAccess,
    u_max: float,
) -> dict[str, Any]:
    """The full content-signature tree this point's cache key is derived
    from -- every field that could change the computed result participates."""
    return {
        "recipe": recipe.get_signature(),
        "axis": axis.value,
        "level": level.value if isinstance(level, StrEnum) else level,
        "seed": seed,
        "model_access": model_access.value,
        "u_max": u_max,
        "training_problem": condition.training_problem.get_signature(),
        "eval_problem": condition.eval_problem.get_signature(),
        "dr_distribution": (
            condition.dr_distribution.get_signature()
            if condition.dr_distribution is not None
            else None
        ),
        "nominal_override": (
            condition.nominal_override[0].get_signature()
            if condition.nominal_override is not None
            else None
        ),
    }


def _learned_p_of(controller: Any) -> np.ndarray | None:
    """The learned Riccati-replacement matrix, if `controller` has one
    (UF-alpha+P only) -- read fresh, post-training."""
    config = getattr(controller, "config", None)
    parameters = getattr(config, "parameters", None)
    if parameters is not None and "riccati_matrix" in parameters:
        return cast(np.ndarray, parameters["riccati_matrix"].get_numpy())
    return None


def _extract_learning_curve(engine: Any) -> tuple[float, ...]:
    """The per-epoch training-loss history `MetricsHistoryCallback` (already
    attached by every recipe's standard callback list, `ModelRecipe
    ._standard_callbacks`) accumulated during `engine.train()`."""
    for callback in engine.callbacks:
        history = getattr(callback, "_history", None)
        if history is not None:
            metric_key = "loss" if history and "loss" in history[0] else "cost"
            return tuple(float(row.get(metric_key, float("nan"))) for row in history)
    return ()


def _shift_metrics_to_payload(prefix: str, metrics: ShiftMetrics) -> dict[str, float]:
    return {
        f"{prefix}_cost_mean": metrics.cost_mean,
        f"{prefix}_cost_median": metrics.cost_median,
        f"{prefix}_cost_q25": metrics.cost_q25,
        f"{prefix}_cost_q75": metrics.cost_q75,
        f"{prefix}_cost_std": metrics.cost_std,
        f"{prefix}_max_violation": metrics.max_violation,
        f"{prefix}_saturation_rate": metrics.saturation_rate,
        f"{prefix}_n_batches": float(metrics.n_batches),
        f"{prefix}_n_trajectories": float(metrics.n_trajectories),
    }


def _shift_metrics_from_payload(
    prefix: str, metrics: Mapping[str, float]
) -> ShiftMetrics:
    return ShiftMetrics(
        cost_mean=metrics[f"{prefix}_cost_mean"],
        cost_median=metrics[f"{prefix}_cost_median"],
        cost_q25=metrics[f"{prefix}_cost_q25"],
        cost_q75=metrics[f"{prefix}_cost_q75"],
        cost_std=metrics[f"{prefix}_cost_std"],
        max_violation=metrics[f"{prefix}_max_violation"],
        saturation_rate=metrics[f"{prefix}_saturation_rate"],
        n_batches=int(metrics[f"{prefix}_n_batches"]),
        n_trajectories=int(metrics[f"{prefix}_n_trajectories"]),
    )


def _train_and_score_point(  # noqa: PLR0913 -- one point's full identity + execution context
    label: str,
    recipe: ModelRecipe,
    condition: TrainingCondition,
    *,
    axis: UncertaintyAxis,
    level: Any,
    seed: int,
    model_access: ModelAccess,
    ctx: ComputeContext,
    u_max: float,
    nominal_problem: OptimalControlProblem,
    nominal_batches: tuple[Any, ...],
    cache: TrackerBackedExperimentCache,
    policy: CachePolicy,
    stamp: str,
) -> RobustTrainingPoint:
    """One (contender, level, seed) unit of work: cache-hit -> reconstruct
    from persisted metrics/arrays; cache-miss -> train, score twice, cache,
    return. See the module docstring for why this is not `run_experiment`."""
    content_digest = compute_signature_digest(
        _condition_signature(
            recipe,
            condition,
            axis=axis,
            level=level,
            seed=seed,
            model_access=model_access,
            u_max=u_max,
        )
    )
    key = compose_cache_key(content_digest, stamp)

    if policy.mode in ("reuse", "readonly"):
        cached = cache.get(key, content_digest=content_digest, stamp=stamp)
        if cached is not None:
            learned_p = cached.arrays.get("learned_p")
            cached_access = cached.params.get(
                "effective_model_access", model_access.value
            )
            return RobustTrainingPoint(
                level=level,
                seed=seed,
                trained_metrics=_shift_metrics_from_payload("trained", cached.metrics),
                nominal_metrics=_shift_metrics_from_payload("nominal", cached.metrics),
                learning_curve=tuple(cached.arrays["learning_curve"].tolist()),
                learned_p=learned_p,
                disposition="hit",
                effective_model_access=ModelAccess(cached_access),
            )
        if policy.mode == "readonly":
            raise CacheReadOnlyMissError(
                f"Point {label!r}/{axis.value}/{level!r}/seed={seed} is not "
                "cached and the cache policy is readonly."
            )

    controller = recipe.build_controller(condition.training_problem, ctx)
    # Every ModelRecipe.build_engine in the tree constructs a Runner (the
    # Engine Protocol's only implementation); a Runner's .callbacks is a
    # plain mutable list, which is the whole mechanism this module relies on
    # to attach the DR callback without any change to build_engine itself.
    engine = cast(
        Runner,
        recipe.build_engine(
            controller, condition.training_problem, condition.training_harness
        ),
    )
    effective_model_access = model_access
    if condition.dr_distribution is not None:
        resync_controller = model_access is ModelAccess.ORACLE
        if resync_controller and not supports_oracle_resync(controller):
            logger.info(
                "%s/%s has no ORACLE resynchronizer (controller type %s); "
                "training this contender NOMINAL instead (NB07 plan Sec "
                "4.2's declared COCP exception) while the rest of the "
                "%s/%r sweep stays ORACLE.",
                axis.value,
                label,
                type(controller).__name__,
                axis.value,
                level,
            )
            resync_controller = False
            effective_model_access = ModelAccess.NOMINAL
        engine.callbacks.append(
            DomainRandomizationCallback(
                condition.dr_distribution,
                ctx=ctx,
                resync_controller=resync_controller,
            )
        )
    final_metrics = dict(engine.train())
    artifact = TrainedControllerArtifact(
        controller=controller,
        context=ctx,
        synthesizer_signature=recipe.get_signature(),
        provenance={"final_metrics": final_metrics},
    )

    trained_metrics, _ = evaluate_under_shift(
        artifact, condition.eval_problem, condition.eval_batches, u_max=u_max
    )
    eval_nominal_problem, eval_nominal_batches = (
        condition.nominal_override
        if condition.nominal_override is not None
        else (nominal_problem, nominal_batches)
    )
    nominal_metrics, _ = evaluate_under_shift(
        artifact, eval_nominal_problem, eval_nominal_batches, u_max=u_max
    )
    learning_curve = _extract_learning_curve(engine)
    learned_p = _learned_p_of(controller)

    tracker = cache.open_run(f"NB07_{axis.value}__{label}")
    with bind_run_log(tracker.run_dir):
        arrays: dict[str, np.ndarray] = {
            "learning_curve": np.asarray(learning_curve, dtype=np.float64),
        }
        if learned_p is not None:
            arrays["learned_p"] = learned_p
        cache.put(
            key,
            tracker,
            ContenderPayload(
                content_digest=content_digest,
                stamp=stamp,
                params={
                    "contender_label": label,
                    "axis": axis.value,
                    "level": str(level),
                    "seed": seed,
                    "model_access": model_access.value,
                    "effective_model_access": effective_model_access.value,
                },
                metrics={
                    **_shift_metrics_to_payload("trained", trained_metrics),
                    **_shift_metrics_to_payload("nominal", nominal_metrics),
                },
                arrays=arrays,
            ),
        )

    return RobustTrainingPoint(
        level=level,
        seed=seed,
        trained_metrics=trained_metrics,
        nominal_metrics=nominal_metrics,
        learning_curve=learning_curve,
        learned_p=learned_p,
        disposition="fresh",
        effective_model_access=effective_model_access,
    )


def run_robust_training_sweep(  # noqa: PLR0913 -- mirrors run_depth_sweep_study's own keyword surface
    recipes_at: Callable[[Any], Mapping[str, ModelRecipe]],
    condition_at: Callable[[Any, int], TrainingCondition],
    spec: RobustTrainingSpec,
    *,
    ctx: ComputeContext,
    u_max: float,
    nominal_problem: OptimalControlProblem,
    nominal_batches: tuple[Any, ...],
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
) -> RobustTrainingResult:
    """Drive `spec`'s full ``levels x seeds`` grid for every contender,
    cached per (contender, level, seed) (`_train_and_score_point`).

    Args:
        recipes_at: Builds label -> `ModelRecipe` for one `level` (called
            once per level, reused across every seed at that level) -- the
            six training-sweep contenders (`cocp` excluded, NB07 plan Sec 13
            decision 2). A function of level, not a fixed mapping: every
            axis but `TRAIN_HORIZON` returns the SAME dict regardless of
            `level`; `TRAIN_HORIZON` must rebuild each unfolded/GRU recipe
            with ``horizon=level`` baked in, since a recipe's horizon is a
            specification-time field, not a per-call argument.
        condition_at: Builds the `TrainingCondition` for one
            ``(level, seed_index)`` pair; the study module's own
            axis-specific responsibility (batch-spec scaling, plant
            perturbation, horizon override, or finite-dataset construction).
        spec: The axis's sweep declaration.
        ctx: The compute context every recipe trains/evaluates under.
        u_max: The infinity-norm control bound (feasibility audit and
            saturation rate).
        nominal_problem: The clean problem the price-of-robustness pass
            evaluates against.
        nominal_batches: Evaluation batches drawn from `nominal_problem`.
        root: The experiments/artifacts root the per-point cache reads/writes.
        policy: Cache policy; defaults to ``CachePolicy("reuse")``.

    Returns:
        The aggregated `RobustTrainingResult`.
    """
    cache = TrackerBackedExperimentCache(root)
    stamp = code_provenance_stamp()
    labels = tuple(recipes_at(spec.levels[0]))
    points: dict[str, list[RobustTrainingPoint]] = {label: [] for label in labels}

    for level in spec.levels:
        recipes = recipes_at(level)
        for seed_index in range(spec.n_seeds):
            seed = spec.base_seed + seed_index
            condition = condition_at(level, seed)
            for label, recipe in recipes.items():
                point = _train_and_score_point(
                    label,
                    recipe,
                    condition,
                    axis=spec.axis,
                    level=level,
                    seed=seed,
                    model_access=spec.model_access,
                    ctx=ctx,
                    u_max=u_max,
                    nominal_problem=nominal_problem,
                    nominal_batches=nominal_batches,
                    cache=cache,
                    policy=policy,
                    stamp=stamp,
                )
                points[label].append(point)

    return RobustTrainingResult(
        axis=spec.axis,
        model_access=spec.model_access,
        levels=spec.levels,
        points={label: tuple(pts) for label, pts in points.items()},
    )
