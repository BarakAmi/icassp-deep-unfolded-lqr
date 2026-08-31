"""`run_experiment` — the single unified entry point for all research
execution (REFACTOR_PLAN v3, T3.d): bind the run's dedicated logger (T3.k) →
resolve contenders through the recipe registry → `synthesize` each (offline
phase) → evaluate every policy under the shared `EvaluationProtocol` on
common noise realizations (online phase) → assemble the `ExperimentReport` →
persist through the tracker-backed cache (T3.e) → append the history-registry
record (T3.j).

The per-contender cache interception happens *before any compute*: a hit
loads and returns (logged at INFO), a miss computes and stores. Hits and
misses alike appear in the history registry, so reuse is auditable.
"""

import logging
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from .cache import (
    CachePolicy,
    CacheReadOnlyMissError,
    ContenderPayload,
    ExperimentCache,
    TrackerBackedExperimentCache,
    compose_cache_key,
)
from .evaluation import evaluate_synthesized_controller
from .experiment import (
    ContenderResult,
    ContenderSpec,
    Experiment,
    ExperimentReport,
    problem_dimensions,
)
from .history import (
    ExperimentHistoryRegistry,
    HistoryRecord,
    pack,
    utc_timestamp,
)
from .provenance import code_provenance_stamp
from .run_logging import bind_run_log
from ..applications.recipes.base import EngineHarness
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.profiling import (
    current_peak_ram_bytes,
    peak_gpu_memory_bytes,
    reset_gpu_peak_memory,
    wall_clock,
)
from ..models.registry import ControllerRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionServices:
    """The injectable service overrides of `run_experiment` — every one
    optional, every one defaulting to the standard tracker-backed
    implementation rooted at `run_experiment`'s ``root``.

    Attributes:
        cache: The interception seam; defaults to
            `TrackerBackedExperimentCache(root)`.
        history: The logbook; defaults to `ExperimentHistoryRegistry` at the
            cache's root.
        recipe_registry: Recipe-resolution registry override (tests).
        log_level: The per-run ``experiment.log`` capture level (T3.k) —
            ``logging.DEBUG`` adds per-epoch metrics and hardware handshakes.
    """

    cache: ExperimentCache | None = None
    history: ExperimentHistoryRegistry | None = None
    recipe_registry: ControllerRegistry | None = None
    log_level: int = logging.INFO


@dataclass(frozen=True)
class _ContenderJob:
    """Internal per-contender execution context threaded from
    `run_experiment` into `_run_contender`: the resolved experiment
    materials plus the run-wide cache/provenance state."""

    experiment: Experiment
    problem: OptimalControlProblem
    eval_batches: Any
    cache: ExperimentCache
    policy: CachePolicy
    stamp: str
    experiment_signature: str
    recipe_registry: ControllerRegistry | None
    log_level: int


def run_experiment(
    experiment: Experiment,
    *,
    root: Path | str | None = None,
    policy: CachePolicy = CachePolicy(),
    services: ExecutionServices = ExecutionServices(),
) -> ExperimentReport:
    """Execute (or serve from cache) every contender of `experiment`.

    Args:
        experiment: The fully-bound, signable experiment declaration.
        root: The experiments/artifacts root; used to build the default
            tracker-backed cache and history registry. Required unless
            `services` injects both a cache and a history.
        policy: Per-invocation cache control (reuse / recompute / readonly).
        services: Optional service overrides — cache, history registry,
            recipe registry, and the per-run log capture level (see
            `ExecutionServices`).

    Returns:
        The assembled `ExperimentReport`.

    Raises:
        ValueError: If neither `root` nor an injected cache is given.
        CacheReadOnlyMissError: On a miss under a readonly policy.
    """
    cache, history = services.cache, services.history
    if cache is None:
        if root is None:
            raise ValueError("run_experiment needs either root= or a cache.")
        cache = TrackerBackedExperimentCache(root)
    if history is None:
        history_root = root if root is not None else getattr(cache, "root", None)
        if history_root is None:
            raise ValueError(
                "run_experiment needs root= (or a cache exposing .root) to "
                "place the history registry."
            )
        history = ExperimentHistoryRegistry(history_root)

    stamp = code_provenance_stamp()
    experiment_signature = experiment.signature_digest()
    logger.info(
        "Experiment %r started: signature %s, stamp %s, policy %r.",
        experiment.name,
        experiment_signature,
        stamp,
        policy.mode,
    )

    problem = experiment.problem.build()
    # Materialized ONCE: every contender is evaluated on these exact
    # realizations (the common-noise law of the EvaluationProtocol).
    eval_batches = experiment.evaluation.build_batches(experiment.ctx)

    job = _ContenderJob(
        experiment=experiment,
        problem=problem,
        eval_batches=eval_batches,
        cache=cache,
        policy=policy,
        stamp=stamp,
        experiment_signature=experiment_signature,
        recipe_registry=services.recipe_registry,
        log_level=services.log_level,
    )
    results: dict[str, ContenderResult] = {}
    for spec in experiment.contenders:
        results[spec.resolved_label] = _run_contender(spec, job)

    report = ExperimentReport(
        experiment_name=experiment.name,
        experiment_signature=experiment_signature,
        provenance_stamp=stamp,
        results=results,
    )
    _append_history(history, experiment, report)
    logger.info("Experiment %r finished: %s.", experiment.name, _dispositions(report))
    return report


def _run_contender(spec: ContenderSpec, job: _ContenderJob) -> ContenderResult:
    """One contender through the interception seam: hit → load; miss →
    synthesize (offline), evaluate (online), store."""
    experiment, problem = job.experiment, job.problem
    eval_batches = job.eval_batches
    cache, policy, stamp = job.cache, job.policy, job.stamp
    experiment_signature = job.experiment_signature
    recipe_registry, log_level = job.recipe_registry, job.log_level
    label = spec.resolved_label
    content_digest = experiment.contender_content_digest(spec)
    key = compose_cache_key(content_digest, stamp)

    if policy.mode in ("reuse", "readonly"):
        cached = cache.get(key, content_digest=content_digest, stamp=stamp)
        if cached is not None:
            logger.info(
                "Cache HIT for contender %r (key %s): serving %s.",
                label,
                key,
                cached.run_dir.name,
            )
            return ContenderResult(
                label=label,
                family=spec.family,
                cache_key=key,
                content_digest=content_digest,
                disposition="hit",
                run_dir=str(cached.run_dir),
                metrics=dict(cached.metrics),
                arrays=dict(cached.arrays),
            )
        if policy.mode == "readonly":
            raise CacheReadOnlyMissError(
                f"Contender {label!r} (key {key}) is not cached and the "
                "cache policy is readonly -- nothing may be computed. Rerun "
                "under CachePolicy('reuse') to compute and store it."
            )
        logger.info("Cache MISS for contender %r (key %s): computing.", label, key)
    else:
        logger.info(
            "Cache policy %r: recomputing contender %r (key %s).",
            policy.mode,
            label,
            key,
        )

    recipe = spec.resolve(recipe_registry)
    tracker = cache.open_run(f"{experiment.name}__{label}")
    with bind_run_log(tracker.run_dir, level=log_level):
        logger.info(
            "Contender %r (family %r) started: content digest %s, "
            "cache key %s, stamp %s.",
            label,
            recipe.family,
            content_digest,
            key,
            stamp,
        )
        harness = EngineHarness(
            batch_spec=experiment.evaluation.batch_spec,
            tracker=tracker,
            ctx=experiment.ctx,
        )
        synthesizer = recipe.build_synthesizer(problem, harness)
        artifact, synthesis_profile = _measure_synthesis(
            synthesizer, problem, experiment.ctx
        )
        logger.info(
            "Offline phase (synthesis) finished for %r; starting online "
            "evaluation on %d common batch(es).",
            label,
            len(eval_batches),
        )
        eval_metrics, eval_arrays = evaluate_synthesized_controller(
            artifact, problem, eval_batches
        )
        synthesis_metrics = {**synthesis_profile, **_synthesis_final_metrics(artifact)}
        metrics = {**synthesis_metrics, **eval_metrics}
        cache.put(
            key,
            tracker,
            ContenderPayload(
                content_digest=content_digest,
                stamp=stamp,
                params={
                    "application_name": experiment.name,
                    "experiment_name": experiment.name,
                    "experiment_signature": experiment_signature,
                    "contender_label": label,
                    "contender_family": recipe.family,
                    **problem_dimensions(experiment.problem),
                },
                metrics=metrics,
                arrays=eval_arrays,
            ),
        )
        logger.info("Contender %r persisted to %s.", label, tracker.run_dir.name)

    disposition = "recompute" if policy.mode == "recompute" else "fresh"
    return ContenderResult(
        label=label,
        family=recipe.family,
        cache_key=key,
        content_digest=content_digest,
        disposition=disposition,
        run_dir=str(tracker.run_dir),
        metrics=metrics,
        arrays=eval_arrays,
    )


def _measure_synthesis(
    synthesizer: Any, problem: OptimalControlProblem, ctx: Any
) -> tuple[Any, dict[str, float]]:
    """Run one contender's offline synthesis under a generic, family-agnostic
    wall-clock + peak-memory measurement -- the only profiling a direct
    solver (e.g. Riccati) ever gets, since it never enters a `Runner`/
    `ProfilingCallback` training loop. Trained families additionally log
    their own finer-grained `profile/train_wall_time_total_s` etc. via
    `_synthesis_final_metrics`; this measurement covers every family
    uniformly, so offline cost is comparable across all contenders.

    Args:
        synthesizer: The contender's resolved `Synthesizer`.
        problem: The `OptimalControlProblem` to synthesize a controller for.
        ctx: The compute context to synthesize on.

    Returns:
        The synthesized artifact, plus its ``profile/synthesis_*`` metrics
        (wall time always; peak VRAM/RAM when the underlying primitive is
        available on this platform).
    """
    reset_gpu_peak_memory()
    start = wall_clock()
    artifact = synthesizer.synthesize(problem, ctx)
    profile: dict[str, float] = {
        "profile/synthesis_wall_time_s": wall_clock() - start,
    }
    peak_vram = peak_gpu_memory_bytes()
    if peak_vram is not None:
        profile["profile/synthesis_peak_vram_mb"] = peak_vram / 1e6
    peak_ram = current_peak_ram_bytes()
    if peak_ram is not None:
        profile["profile/synthesis_peak_ram_mb"] = peak_ram / 1e6
    return artifact, profile


def _synthesis_final_metrics(artifact: Any) -> dict[str, float]:
    """The offline phase's final scalar metrics, read off the artifact's
    declared provenance when present (trained families); empty for direct
    solver artifacts, whose synthesis has no training curve."""
    provenance = getattr(artifact, "provenance", None) or {}
    final = provenance.get("final_metrics", {})
    return {
        key: float(value)
        for key, value in dict(final).items()
        if isinstance(value, (int, float))
    }


def _dispositions(report: ExperimentReport) -> dict[str, str]:
    return {label: r.disposition for label, r in report.results.items()}


def _append_history(
    history: ExperimentHistoryRegistry,
    experiment: Experiment,
    report: ExperimentReport,
) -> None:
    """One standardized logbook record per execution — hits included (T3.j)."""
    dims = problem_dimensions(experiment.problem)
    results = report.results
    history.append(
        HistoryRecord(
            experiment_signature=report.experiment_signature,
            timestamp_utc=utc_timestamp(),
            experiment_name=report.experiment_name,
            contenders=pack({r.label: r.family for r in results.values()}),
            horizon=dims["horizon"],
            state_dim=dims["state_dim"],
            control_dim=dims["control_dim"],
            final_costs=pack(
                {
                    r.label: str(r.metrics.get("eval_expected_cost", ""))
                    for r in results.values()
                }
            ),
            dispositions=pack({r.label: r.disposition for r in results.values()}),
            run_dirs=pack(
                {
                    r.label: (Path(r.run_dir).name if r.run_dir else "")
                    for r in results.values()
                }
            ),
            provenance_stamp=report.provenance_stamp,
        )
    )
