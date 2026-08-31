"""Offline/online benchmark measurement (T3.g *benchmarks*): phase-split
time+memory records out, nothing displayed — the presentation of these
records stays with the Tier-4 adapters.
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from .experiments import GDSolveSettings, run_analytic_controller_rollout
from .replay import load_contender_artifacts
from ..experiments import ContenderResult, ContenderSpec
from .setup import (
    DisturbanceRealization,
    GaussianBatchSpec,
    RunLocation,
    SamplerBundle,
    build_step_size_schedule,
    make_gaussian_batch_sampler,
)
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.profiling import numpy_array_bytes, track_stratified_memory, wall_clock
from ..core.runtime import ComputeContext
from ..models.analytic.iterative_gd import (
    AnalyticalIterativeGDController,
    SolveSpec,
    build_riccati_gd_refinement,
)
from ..models.analytic.riccati import RiccatiController
from ..models.iterative import (
    ControlInitializer,
    GaussSeidelSweep,
    JacobiSweep,
)
from ..models.lifecycle import Synthesizer
from ..models.samplers import ConstantDistribution
from ..viz.plots.benchmarking import BenchmarkRecord, MemoryBreakdown


def measure_riccati_synthesis_benchmark(
    location: RunLocation,
    problem: OptimalControlProblem,
    horizon: int,
    *,
    sampling: SamplerBundle,
) -> tuple[float, float, MemoryBreakdown, MemoryBreakdown]:
    """Asset 7 (Section 10): the Riccati-synthesis counterpart of
    `measure_analytic_gd_benchmark` below, phase-split into **offline
    synthesis** (solving the backward Riccati recursion once, building
    `RiccatiController`) and **online inference** (the closed-loop rollout
    under the resulting static gain ``u_t = -K_t x_t``, via the reused
    `run_analytic_controller_rollout` wrapper).

    A fresh `RiccatiController` is built here (rather than accepting an
    already-solved one) so `t_offline_s` actually times the backward
    recursion instead of crediting a synthesis that happened earlier in the
    notebook.

    Args:
        location: where the benchmark run is persisted (see
            `run_analytic_controller_rollout`).
        problem, horizon: the problem to synthesize the controller for.
        sampling: the Monte Carlo rollout's batch-sampling configuration
            (see `make_gaussian_batch_sampler`).

    Returns:
        ``(t_offline_s, t_online_s, offline_memory, online_memory)``:
        `t_offline_s` is the one-time backward-Riccati-recursion synthesis;
        `t_online_s` is the subsequent forward rollout; `offline_memory`/
        `online_memory` are each phase's stratified footprint
        (`core.profiling.track_stratified_memory`) as a `MemoryBreakdown`.
        `offline_memory.numpy_mb` is the solved `P_arr`/`K_arr`'s own
        structural footprint (registered explicitly, since NumPy's caching
        behavior otherwise leaves `track_stratified_memory`'s CPU-RSS-delta
        sampler alone to detect it). `online_memory.numpy_mb` reports the
        online phase's ABSOLUTE memory requirement, not merely its own
        growth: a controller executing ``u_t = -K_t x_t`` must hold `K_arr`
        resident in RAM regardless of when it was synthesized, so `K_arr` is
        registered explicitly here too, even though it was allocated during
        the offline phase and the rollout itself grows memory by almost
        nothing -- otherwise this phase would silently report ~0 MB,
        understating what the deployed controller actually needs resident.
        `torch_mb` stays at (or near) zero in both, since neither phase does
        any PyTorch work, unlike the GD solver below.
    """
    t0 = wall_clock()
    with track_stratified_memory() as offline_mem:
        riccati = RiccatiController(problem, horizon)
        offline_mem.track_numpy(riccati.P_arr, riccati.K_arr)
    t_offline_s = wall_clock() - t0

    t1 = wall_clock()
    with track_stratified_memory() as online_mem:
        run_analytic_controller_rollout(location, riccati, sampling=sampling)
        # K_arr was synthesized offline, so the rollout itself grows memory
        # by almost nothing -- but a controller executing u_t = -K_t x_t
        # must hold K_arr resident regardless, so its weight belongs in the
        # ONLINE report (absolute footprint), not just the offline one.
        online_mem.track_numpy(riccati.K_arr)
    t_online_s = wall_clock() - t1

    offline_report, online_report = offline_mem.report, online_mem.report
    assert offline_report is not None and online_report is not None  # ctx exited
    return (
        t_offline_s,
        t_online_s,
        MemoryBreakdown(
            numpy_mb=offline_report.numpy_cpu_bytes / 1e6,
            torch_mb=offline_report.torch_bytes / 1e6,
        ),
        MemoryBreakdown(
            numpy_mb=online_report.numpy_cpu_bytes / 1e6,
            torch_mb=online_report.torch_bytes / 1e6,
        ),
    )


def torch_tensor_bytes(*tensors: torch.Tensor) -> int:
    """Structural byte footprint of `tensors` -- the workbench's own PyTorch
    counterpart of `core.profiling.numpy_array_bytes` (``element_size() *
    nelement()`` per tensor), used to fold a precomputed parameter's
    resident weight into a benchmark phase's memory report even when the
    tensor was allocated during an earlier, already-measured phase (see
    `measure_analytic_gd_benchmark`'s ``online_memory.torch_mb``).
    """
    return sum(int(t.element_size() * t.nelement()) for t in tensors)


def measure_analytic_gd_benchmark(
    problem: OptimalControlProblem,
    horizon: int,
    realization: DisturbanceRealization,
    control_initializer: ControlInitializer,
    *,
    settings: GDSolveSettings,
    shared_riccati_offline_numpy_mb: float = 0.0,
) -> tuple[float, float, MemoryBreakdown, MemoryBreakdown]:
    """Asset 7 (Section 10): the GD counterpart of
    `measure_riccati_synthesis_benchmark`, phase-split into **offline
    synthesis** (`build_riccati_gd_refinement`'s one-time backward-Riccati
    solve that derives the local gradient's static ``M``/``C`` coefficients)
    and **online inference** (the real-time iterative macro-loop that
    refines ``U`` from `control_initializer` under those fixed
    coefficients).

    The refinement is built explicitly here and passed into `.solve` via its
    `refinement` argument -- `.solve` only builds one internally when
    `refinement` is omitted -- so `.solve` itself measures the online phase
    alone, with no double-counted synthesis.

    Args:
        problem, horizon: the instance to solve.
        realization: the fixed ``(x0, w)`` batch solved against (wrapped in
            `ConstantDistribution`, matching every other signal-space GD
            solve in this notebook).
        control_initializer: the starting signal (e.g. `ConstantInitializer`,
            `cold_start`).
        settings: the solver configuration -- step size, macro-iteration
            budget, dtype, and the update topology (`JacobiSweep`/
            `GaussSeidelSweep`) being benchmarked.
        shared_riccati_offline_numpy_mb: a known NumPy/CPU-layer floor (MB)
            for the offline phase, folded into `offline_memory.numpy_mb` via
            ``max``. `build_riccati_gd_refinement` internally solves the
            *exact same* backward Riccati recursion as
            `measure_riccati_synthesis_benchmark` (identical `system`,
            `cost`, `horizon` -- a deterministic computation, so its
            discarded ``P_arr`` is shape- and value-identical to that
            benchmark's own, already-measured one) before deriving its
            ``M``/``C`` coefficients, but never returns that intermediate
            ``P_arr`` for this function to register directly. Re-running the
            recursion here just to measure it would double `t_offline_s`
            against the theoretical "same offline cost" claim (Section 10);
            leaving it to `track_stratified_memory`'s own CPU-RSS-delta
            sampler alone is unreliable, since `P_arr`'s few-KB-to-tens-of-KB
            footprint at this notebook's dimensions can fall below the
            OS page-fault granularity `current_peak_ram_bytes` observes.
            Passing the caller's own already-measured floor sidesteps both.

    Returns:
        ``(t_offline_s, t_online_s, offline_memory, online_memory)`` -- each
        a `MemoryBreakdown`. `offline_memory.torch_mb` is the refinement's
        converted ``M_stack``/``C_stack`` tensors' footprint; `online_memory`
        reports the macro-iteration loop's ABSOLUTE memory requirement:
        `torch_mb` is the ``max`` of its own PyTorch-native allocation growth
        and ``M_stack``/``C_stack``'s own resident weight
        (`torch_tensor_bytes`) -- every macro-iteration reads those
        coefficients, so they must stay resident in RAM online just as much
        as `K_arr` must for the Riccati benchmark above, even though (like
        `K_arr`) they were allocated during the already-measured offline
        phase. `numpy_mb` is the returned `OptimizationResult`'s
        ``U_history``/``J_history`` (registered explicitly, both real NumPy
        arrays this function has direct access to) -- non-zero in both
        layers here, unlike the NumPy-only Riccati benchmark.
    """
    control_dim = problem.system.dimensions.control_dim
    step_size = build_step_size_schedule(
        control_dim,
        horizon,
        alpha=settings.alpha,
        num_iterations=settings.max_iters,
        dtype=settings.dtype,
    )
    spec = SolveSpec(
        horizon=horizon,
        x0_sampler=ConstantDistribution(realization.x0),
        noise_sampler=ConstantDistribution(realization.w),
        control_initializer=control_initializer,
        max_iters=settings.max_iters,
        tolerance=None,
        batch_size=realization.x0.shape[0],
        sweep_strategy=settings.sweep_strategy,
    )

    t0 = wall_clock()
    with track_stratified_memory() as offline_mem:
        refinement = build_riccati_gd_refinement(
            problem, step_size, spec, dtype=settings.dtype
        )
    t_offline_s = wall_clock() - t0

    t1 = wall_clock()
    with track_stratified_memory() as online_mem:
        result = AnalyticalIterativeGDController(step_size, dtype=settings.dtype).solve(
            problem, replace(spec, refinement=refinement)
        )
        online_mem.track_numpy(result.U_history, result.J_history)
    t_online_s = wall_clock() - t1

    offline_report, online_report = offline_mem.report, online_mem.report
    assert offline_report is not None and online_report is not None  # ctx exited
    offline_memory = MemoryBreakdown(
        numpy_mb=max(
            offline_report.numpy_cpu_bytes / 1e6, shared_riccati_offline_numpy_mb
        ),
        torch_mb=offline_report.torch_bytes / 1e6,
    )
    # M_stack/C_stack were allocated during the offline phase (inside
    # `refinement`), so the online loop's own allocation growth alone misses
    # their weight -- but every macro-iteration reads them, so they must
    # stay resident in RAM online. `max` (not a sum) mirrors both
    # `track_stratified_memory`'s own numpy_cpu_bytes fallback and this
    # function's `shared_riccati_offline_numpy_mb` handling above: the
    # reported footprint is never allowed to understate this known floor.
    online_memory = MemoryBreakdown(
        numpy_mb=online_report.numpy_cpu_bytes / 1e6,
        torch_mb=max(
            online_report.torch_bytes,
            torch_tensor_bytes(*refinement.static_parameters.values()),
        )
        / 1e6,
    )
    return t_offline_s, t_online_s, offline_memory, online_memory


#: The default `LocalExperimentTracker` name for the suite's Riccati rollout
#: benchmark run (kept as a module constant so notebooks can compose a
#: `RunLocation` without re-typing the historical name).
RICCATI_BENCHMARK_EXPERIMENT_NAME = "gd_lqr_signal_space_riccati_benchmark"


@dataclass(frozen=True)
class RiccatiBenchmarkSpec:
    """The Riccati rollout benchmark's own persistence + sampling
    configuration (its disturbance batch is drawn fresh, independent of the
    GD solves' fixed realization).

    Attributes:
        location: Where the Riccati benchmark run is persisted.
        batch_spec: The fresh Gaussian batch the rollout is scored on.
    """

    location: RunLocation
    batch_spec: GaussianBatchSpec


def run_benchmark_suite(
    problem: OptimalControlProblem,
    horizon: int,
    realization: DisturbanceRealization,
    control_initializer: ControlInitializer,
    *,
    settings: GDSolveSettings,
    riccati_benchmark: RiccatiBenchmarkSpec,
) -> list[BenchmarkRecord]:
    """Section 10 (Asset 7), measurement half: benchmarks Riccati synthesis
    against the signal-space iteration under both update orderings, each
    phase-split into **offline synthesis** (one-time pre-computation: the
    backward Riccati recursion, shared by both the closed-form controller and
    the GD refinement's gradient coefficients) and **online inference** (the
    real-time control-generation phase: a static-gain rollout for Riccati, an
    iterative macro-loop for GD), each phase's memory stratified by NumPy/CPU
    vs. PyTorch CPU/GPU layer (Micro-Prompt 5d). Returns data only -- the
    table/figure presentation lives with the Tier-4 adapters
    (`viz.adapters.notebook.report_benchmarks`).

    Args:
        problem, horizon: the primary instance being benchmarked.
        realization: the fixed disturbance batch each signal-space solve
            runs against (see `measure_analytic_gd_benchmark`).
        control_initializer: the starting signal each signal-space solve
            uses (e.g. a cold start).
        settings: the signal-space solver's configuration; its
            ``sweep_strategy`` is overridden per topology benchmarked.
        riccati_benchmark: the Riccati rollout benchmark's own persistence
            and (fresh, independent) sampling configuration.

    Returns:
        One `BenchmarkRecord` per method: Riccati, GD--Jacobi, and
        GD--Gauss-Seidel.
    """
    # `torch.profiler.profile` pays a one-time backend-activation cost
    # (~0.5-0.8s, independent of the profiled workload) on its very first use
    # in the process; a no-op warmup here pays that cost once, up front,
    # rather than letting it silently inflate whichever benchmark below
    # happens to call `track_stratified_memory` first (previously the
    # Riccati offline-synthesis benchmark, which then looked ~150x slower
    # than the identical recursion inside the GD refinement's own offline
    # benchmark).
    with track_stratified_memory():
        pass

    sampling = make_gaussian_batch_sampler(
        problem.system.dimensions.state_dim, horizon, riccati_benchmark.batch_spec
    )
    (
        riccati_offline_s,
        riccati_online_s,
        riccati_offline_mem,
        riccati_online_mem,
    ) = measure_riccati_synthesis_benchmark(
        riccati_benchmark.location,
        problem,
        horizon,
        sampling=sampling,
    )
    records = [
        BenchmarkRecord(
            label="Riccati",
            offline_s=riccati_offline_s,
            online_s=riccati_online_s,
            offline_memory=riccati_offline_mem,
            online_memory=riccati_online_mem,
        )
    ]

    max_iters = settings.max_iters
    batch_size = realization.x0.shape[0]
    for sweep_label, sweep_strategy in (
        ("Jacobi", JacobiSweep()),
        ("Gauss-Seidel", GaussSeidelSweep()),
    ):
        offline_s, online_s, offline_mem, online_mem = measure_analytic_gd_benchmark(
            problem,
            horizon,
            realization,
            control_initializer,
            settings=replace(settings, sweep_strategy=sweep_strategy),
            shared_riccati_offline_numpy_mb=riccati_offline_mem.numpy_mb or 0.0,
        )
        records.append(
            BenchmarkRecord(
                label=f"GD -- {sweep_label} (M={max_iters}, B={batch_size})",
                offline_s=offline_s,
                online_s=online_s,
                offline_memory=offline_mem,
                online_memory=online_mem,
            )
        )
    return records


def _synthesized_controller_memory(artifact: Any) -> MemoryBreakdown:
    """The frozen artifact's OWN resident weight: ``P_arr``/``K_arr`` for a
    closed-form `SynthesizedRiccatiController`, or `as_module()`'s
    registered parameters for a `TrainableController` -- folded into
    `measure_synthesized_controller_benchmark`'s online memory report, the
    same "a deployed controller must hold its own weights resident
    regardless of when they were synthesized" law
    `measure_riccati_synthesis_benchmark` already established for `K_arr`.
    """
    numpy_bytes = 0
    torch_bytes = 0
    for name in ("P_arr", "K_arr"):
        array = getattr(artifact, name, None)
        if isinstance(array, np.ndarray):
            numpy_bytes += numpy_array_bytes(array)
        elif isinstance(array, torch.Tensor):
            torch_bytes += torch_tensor_bytes(array)
    as_module = getattr(artifact, "as_module", None)
    if as_module is not None:
        tensors = list(as_module().parameters())
        if tensors:
            torch_bytes += torch_tensor_bytes(*tensors)
    return MemoryBreakdown(numpy_mb=numpy_bytes / 1e6, torch_mb=torch_bytes / 1e6)


@dataclass(frozen=True)
class InferenceBenchmarkSpec:
    """`measure_synthesized_controller_benchmark`'s online-phase timing
    configuration (PLR0913: keeps the function under the 6-argument cap).

    Attributes:
        label: The record's display label (e.g. the contender's).
        num_inference_calls: Timed online calls averaged over.
        inference_warmup_calls: Untimed calls before timing starts.
    """

    label: str
    num_inference_calls: int = 2000
    inference_warmup_calls: int = 50


@dataclass(frozen=True)
class OnlineInferenceRecord:
    """The online-only inference benchmark of a FROZEN, already-synthesized
    controller (NB03 Phase B, directive 3): no offline synthesis, no training
    -- the deployed per-control-step latency and resident footprint alone.

    Attributes:
        label: The contender's display label.
        online_latency_us: Mean per-``compute_action`` latency, microseconds.
        online_numpy_mb: Resident NumPy/CPU footprint of the frozen policy, MB.
        online_torch_mb: Resident PyTorch footprint of the frozen policy, MB.
    """

    label: str
    online_latency_us: float
    online_numpy_mb: float
    online_torch_mb: float


def _load_cached_parameters(controller: Any, result: ContenderResult) -> None:
    """Load a learned controller's TRAINED parameters back from its persisted
    ``ParameterSnapshotCallback`` artifacts (``parameter_<name>``), in place --
    the reconstruction that makes a frozen deployed policy available WITHOUT
    re-running training. A no-op for analytic controllers (``config is None``)
    and for the fixed unfolded configuration (no snapshot artifacts exist, so
    its already-correct init values stand).

    Args:
        controller: The freshly built controller (an `UnfoldedController` for
            the learned families; a closed-form controller otherwise).
        result: The contender's result, whose run directory holds the
            snapshot artifacts.
    """
    parameters = getattr(getattr(controller, "config", None), "parameters", None)
    if not parameters:
        return
    artifacts = load_contender_artifacts(result)
    for name, parameter in parameters.items():
        key = f"parameter_{name}"
        if key in artifacts:
            parameter.load(np.asarray(artifacts[key]))


def measure_frozen_inference_benchmark(
    contender_spec: ContenderSpec,
    result: ContenderResult,
    problem: OptimalControlProblem,
    ctx: ComputeContext,
    state_batch: torch.Tensor,
    *,
    spec: InferenceBenchmarkSpec,
) -> OnlineInferenceRecord:
    """Directive 3: measure a contender's ONLINE per-step latency + resident
    memory from its FROZEN policy, reconstructed from cache -- **never
    retraining**. The learned families' expensive ``synthesize`` (a full
    training run) is bypassed entirely: the controller is rebuilt (cheap: an
    init-state unfolded module, or a closed-form solve) and its trained
    parameters are loaded back from the persisted snapshot artifacts
    (`_load_cached_parameters`); only then is its deployed policy timed.

    The timed loop calls ``policy(0, state)`` over `spec.num_inference_calls`
    repetitions (after `spec.inference_warmup_calls` untimed warmups),
    bracketed by ``torch.cuda.synchronize()`` on CUDA. A one-shot finiteness
    guard rejects a policy that emits non-finite controls, so a NaN can never
    silently enter the reported metrics.

    Args:
        contender_spec: The contender declaration (resolves to its recipe).
        result: The contender's result from an `ExperimentReport` (its
            persisted parameter snapshots are read from here).
        problem: The problem the controller acts on.
        ctx: The compute context the controller is rebuilt under.
        state_batch: The fixed state every timed call evaluates the policy at
            -- batch 1 for the "one control decision at a time" scenario.
            Relocated to NumPy automatically for a closed-form (non-``Module``)
            controller.
        spec: The record's label plus the timing budget
            (`InferenceBenchmarkSpec`).

    Returns:
        The `OnlineInferenceRecord`; its memory folds the frozen controller's
        own resident weight in (`_synthesized_controller_memory`), so it never
        understates the deployed footprint.

    Raises:
        ValueError: If the reconstructed policy emits non-finite controls.
    """
    recipe = contender_spec.resolve()
    controller = recipe.build_controller(problem, ctx)
    _load_cached_parameters(controller, result)
    policy = controller.get_control_policy()

    is_module = isinstance(controller, torch.nn.Module)
    # A closed-form (numpy-native) controller's policy expects a numpy state;
    # a torch-native one expects the tensor as-is (on its own device).
    state: Any = state_batch if is_module else state_batch.detach().cpu().numpy()
    is_cuda = is_module and state_batch.is_cuda

    with torch.no_grad():
        sample = policy(0, state)
        sample_arr = np.asarray(
            sample.detach().cpu() if torch.is_tensor(sample) else sample
        )
        if not np.isfinite(sample_arr).all():
            raise ValueError(
                f"{spec.label}: reconstructed frozen policy emitted non-finite "
                "controls -- refusing to report a NaN benchmark."
            )
        for _ in range(spec.inference_warmup_calls):
            policy(0, state)
        if is_cuda:
            torch.cuda.synchronize()

        t0 = wall_clock()
        with track_stratified_memory() as online_mem:
            for _ in range(spec.num_inference_calls):
                policy(0, state)
            if is_cuda:
                torch.cuda.synchronize()
        t_total_s = wall_clock() - t0

    report = online_mem.report
    assert report is not None  # ctx exited
    resident = _synthesized_controller_memory(controller)
    return OnlineInferenceRecord(
        label=spec.label,
        online_latency_us=(t_total_s / spec.num_inference_calls) * 1e6,
        online_numpy_mb=max(report.numpy_cpu_bytes / 1e6, resident.numpy_mb or 0.0),
        online_torch_mb=max(report.torch_bytes / 1e6, resident.torch_mb or 0.0),
    )


def measure_synthesized_controller_benchmark(
    synthesizer: Synthesizer,
    problem: OptimalControlProblem,
    ctx: ComputeContext,
    state_batch: torch.Tensor,
    *,
    spec: InferenceBenchmarkSpec,
) -> BenchmarkRecord:
    """The offline-synthesis/online-inference benchmark, generalized from
    `measure_riccati_synthesis_benchmark`/`measure_analytic_gd_benchmark`
    (both specific to the signal-space GD family above) to ANY `Synthesizer`
    -- closed-form or gradient-trained alike -- via the shared
    `models.lifecycle` two-phase contract (``synthesize`` ->
    `SynthesizedController.make_policy`).

    **Offline** times/measures `synthesizer.synthesize` itself (a direct
    solve, or a full training run for a gradient-trained family).
    **Online** isolates the single per-control-step call ``policy(t, y_t)``
    (the deployed decision -- "compute_action"), timed over
    `spec.num_inference_calls` repeated calls on the same fixed `state_batch`
    after `spec.inference_warmup_calls` untimed warmup calls (so the timed
    loop measures steady-state cost, not one-time JIT/allocator/cache
    warmup); ``torch.cuda.synchronize()`` brackets both the warmup and the
    timed loop when `state_batch` is a CUDA tensor, since CUDA kernel
    launches are asynchronous and would otherwise be timed as
    near-instantaneous.

    Args:
        synthesizer: The contender's `Synthesizer` (e.g. `RiccatiSynthesizer`,
            or an `EngineTrainedSynthesizer`-backed recipe's).
        problem: The problem to synthesize the artifact for.
        ctx: The compute context `synthesize` executes under.
        state_batch: The fixed batched state ``y_t`` every timed inference
            call evaluates the policy at -- batch=1 for the realistic "one
            control decision at a time" deployment scenario this benchmark
            targets.
        spec: The record's label plus the online-phase timing budget (see
            `InferenceBenchmarkSpec`).

    Returns:
        A `BenchmarkRecord`: `offline_s` is the total synthesis wall time;
        `online_s` is the MEAN per-call latency (seconds/call, not the
        total timed-loop duration); `offline_memory`/`online_memory` are
        each phase's stratified footprint, with the online phase's
        absolute footprint additionally folded in via
        `_synthesized_controller_memory` (see its docstring).
    """
    t0 = wall_clock()
    with track_stratified_memory() as offline_mem:
        artifact = synthesizer.synthesize(problem, ctx)
    t_offline_s = wall_clock() - t0

    policy = artifact.make_policy()
    is_cuda = state_batch.is_cuda
    with torch.no_grad():
        for _ in range(spec.inference_warmup_calls):
            policy(0, state_batch)
        if is_cuda:
            torch.cuda.synchronize()

        t1 = wall_clock()
        with track_stratified_memory() as online_mem:
            for _ in range(spec.num_inference_calls):
                policy(0, state_batch)
            if is_cuda:
                torch.cuda.synchronize()
        t_online_total_s = wall_clock() - t1
    t_online_s = t_online_total_s / spec.num_inference_calls

    offline_report, online_report = offline_mem.report, online_mem.report
    assert offline_report is not None and online_report is not None  # ctx exited

    resident = _synthesized_controller_memory(artifact)
    offline_memory = MemoryBreakdown(
        numpy_mb=offline_report.numpy_cpu_bytes / 1e6,
        torch_mb=offline_report.torch_bytes / 1e6,
    )
    online_memory = MemoryBreakdown(
        numpy_mb=max(online_report.numpy_cpu_bytes / 1e6, resident.numpy_mb or 0.0),
        torch_mb=max(online_report.torch_bytes / 1e6, resident.torch_mb or 0.0),
    )
    return BenchmarkRecord(
        label=spec.label,
        offline_s=t_offline_s,
        online_s=t_online_s,
        offline_memory=offline_memory,
        online_memory=online_memory,
    )
