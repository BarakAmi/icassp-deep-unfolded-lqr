"""Thin execution assemblers (T3.g *experiments*): they run engines/solvers
and return data — reports, artifacts, `OptimizationResult`s — and never plot
or display anything.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from typing import Any

import numpy as np
import torch

from .analysis import validate_convergence
from .setup import (
    DisturbanceRealization,
    GaussianBatchSpec,
    ProblemDims,
    RunLocation,
    SamplerBundle,
    StochasticLQRExperiment,
    build_step_size_schedule,
    setup_stochastic_lqr_experiment,
)
from ..engine.callbacks import (
    ExperimentTrackingCallback,
    ParameterSnapshotCallback,
    ProblemSignatureCallback,
    ProfilingCallback,
    TrajectoryLoggingCallback,
)
from ..engine.config import TrainingConfig
from ..engine.runner import Runner, TrainingPhase
from ..core.optimal_control_problem import OptimalControlProblem
from ..engine.strategy import AnalyticalStrategy
from ..models.analytic.iterative_gd import (
    AnalyticalIterativeGDController,
    SolveSpec,
)
from ..models.base import Controller
from ..models.iterative import (
    ConstantInitializer,
    ControlInitializer,
    GaussSeidelSweep,
    OptimizationResult,
    SweepStrategy,
)
from ..models.samplers import ConstantDistribution
from ..persistence import LocalExperimentTracker


def run_analytic_controller_rollout(
    location: RunLocation,
    controller: Controller,
    *,
    sampling: SamplerBundle,
) -> None:
    """Persist a closed-form (non-learnable) Controller's Riccati matrices
    (`P_arr`/`K_arr`) and a Monte Carlo rollout under `location`, via
    `Runner` + `AnalyticalStrategy` -- a single simulate-and-score pass, no
    gradient training. Intended for `RiccatiController`/
    `TruncatedRiccatiController`-style controllers, which expose `.problem`,
    `.horizon`, `.P_arr`, and `.K_arr`.

    `sampling` is built by the caller (typically via
    `make_gaussian_batch_sampler`) rather than from raw `noise_std`/`seed`
    here (Phase 1F): the caller needs the same `distributions` objects
    *before* deciding whether to call this function at all, to compute the
    run's expected signature (`core.utils.signing.compute_run_signature`)
    and check it against any cached run -- building them internally would
    make that impossible without running the simulation first.
    """
    problem = controller.problem
    batch_sampler, distributions = sampling.sample, sampling.distributions

    def rollout_fn(batch: Any) -> Any:
        initial_state, process_noise, measurement_noise = batch
        policy = controller.get_control_policy()
        return problem.system.run(
            policy, initial_state, process_noise, measurement_noise
        )

    tracker = LocalExperimentTracker(location.root, location.name)
    # learning_rate is a required TrainingConfig field but AnalyticalStrategy
    # has no optimizer -- ExperimentTrackingCallback omits it (and the other
    # optimizer-only fields) from what actually gets logged, since
    # RunContext.is_trainable is False for an AnalyticalStrategy-only run.
    training_config = TrainingConfig(batch_size=sampling.batch_size, learning_rate=1.0)
    runner = Runner(
        model=controller,
        tracker=tracker,
        config=training_config,
        batch_sampler=batch_sampler,
        phases=[
            TrainingPhase(AnalyticalStrategy(controller), epochs=1, name="analytic")
        ],
        callbacks=[
            ExperimentTrackingCallback(),
            ProblemSignatureCallback(problem, controller, distributions),
            ProfilingCallback(),
            ParameterSnapshotCallback(
                # Analytic controllers (Riccati/truncated) expose their
                # solved stacks; the base protocol doesn't declare them.
                {
                    "P_arr": controller.P_arr,  # type: ignore[attr-defined]
                    "K_arr": controller.K_arr,  # type: ignore[attr-defined]
                }
            ),
            TrajectoryLoggingCallback(rollout_fn, batch_sampler),
        ],
    )
    runner.run()


@dataclass(frozen=True)
class GDSolveSettings:
    """The signal-space GD solver's configuration axis, shared verbatim by
    every solve a Global-Convergence-Diagnostics study performs.

    Attributes:
        alpha: The fixed per-macro-iteration step size (see
            `build_step_size_schedule`).
        max_iters: Macro-iteration budget every solve is driven to.
        dtype: torch dtype for the step-size schedule and the solver.
        sweep_strategy: The update topology; ``None`` selects the solver's
            default (`JacobiSweep`).
    """

    alpha: float
    max_iters: int
    dtype: torch.dtype
    sweep_strategy: SweepStrategy | None = None


def solve_signal_space_gd(
    problem: OptimalControlProblem,
    horizon: int,
    realization: DisturbanceRealization,
    control_initializer: ControlInitializer,
    *,
    settings: GDSolveSettings,
) -> OptimizationResult:
    """One direct signal-space gradient-descent solve (Eq. 7) on the fixed
    disturbance `realization`, driven to ``settings.max_iters``
    macro-iterations under `control_initializer`/``settings.sweep_strategy``
    -- the single solve primitive the Global Convergence Diagnostics
    experiments each sweep over a dict of variants.
    """
    control_dim = problem.system.dimensions.control_dim
    schedule = build_step_size_schedule(
        control_dim,
        horizon,
        alpha=settings.alpha,
        num_iterations=settings.max_iters,
        dtype=settings.dtype,
    )
    controller = AnalyticalIterativeGDController(schedule, dtype=settings.dtype)
    return controller.solve(
        problem,
        SolveSpec(
            horizon=horizon,
            x0_sampler=ConstantDistribution(realization.x0),
            noise_sampler=ConstantDistribution(realization.w),
            control_initializer=control_initializer,
            max_iters=settings.max_iters,
            tolerance=None,
            batch_size=realization.x0.shape[0],
            sweep_strategy=settings.sweep_strategy,
        ),
    )


def run_initializer_study(
    problem: OptimalControlProblem,
    horizon: int,
    realization: DisturbanceRealization,
    initializers: Mapping[str, ControlInitializer],
    *,
    settings: GDSolveSettings,
) -> dict[str, OptimizationResult]:
    """Experiment 1 (Global Convergence Diagnostics -- Initializers): solves
    the signal-space iteration (7) once per starting rule in `initializers`
    on the identical disturbance `realization`. Returns data only --
    validation/figure/table reporting lives with the Tier-4 adapter
    (`viz.adapters.notebook.report_convergence_study`).

    Args:
        problem: The `OptimalControlProblem` being solved.
        horizon: The control horizon ``T``.
        realization: The fixed Monte Carlo disturbance realization every
            starting rule is evaluated on.
        initializers: label -> `ControlInitializer`, e.g. cold/randomized/
            warm starts.
        settings: The `GDSolveSettings` shared by every variant.

    Returns:
        label -> `OptimizationResult`, in `initializers`' own order.
    """
    return {
        name: solve_signal_space_gd(
            problem, horizon, realization, initializer, settings=settings
        )
        for name, initializer in initializers.items()
    }


def run_topology_study(
    problem: OptimalControlProblem,
    horizon: int,
    realization: DisturbanceRealization,
    control_initializer: ControlInitializer,
    topologies: Mapping[str, SweepStrategy],
    *,
    settings: GDSolveSettings,
) -> dict[str, OptimizationResult]:
    """Experiment 2 (Global Convergence Diagnostics -- Sweep Topologies):
    solves the signal-space iteration (7) once per update ordering in
    `topologies`, holding the starting signal fixed at `control_initializer`,
    on the identical disturbance `realization` -- otherwise the exact
    counterpart of `run_initializer_study` (see its docstring for the shared
    data-only contract).

    Args:
        topologies: label -> `SweepStrategy` (e.g. `JacobiSweep`/
            `GaussSeidelSweep`), the update orderings compared;
            each overrides ``settings.sweep_strategy`` for its own solve.
        control_initializer: the starting signal held fixed across every
            topology.
        (all other args: see `run_initializer_study`.)

    Returns:
        label -> `OptimizationResult`, in `topologies`' own order.
    """
    return {
        name: solve_signal_space_gd(
            problem,
            horizon,
            realization,
            control_initializer,
            settings=replace(settings, sweep_strategy=sweep_strategy),
        )
        for name, sweep_strategy in topologies.items()
    }


@dataclass(frozen=True)
class M3ScatterStudy:
    """Section 9's solved three-control-dimension instance: everything the
    Tier-4 scatter-landscape adapter needs to render Asset 5, plus the
    study's own convergence diagnostics -- data only, nothing displayed.

    Attributes:
        lqr: the instance's full setup bundle (problem, Riccati baseline,
            fixed batch, figures directory).
        record: the solved signal-space GD result.
        t_star: the selected instant of highest cross-ensemble variance in
            the instance's optimal signal.
        relative_error: `record.J_final`'s relative error to the instance's
            own Riccati optimum `lqr.J_opt`.
    """

    lqr: StochasticLQRExperiment
    record: OptimizationResult
    t_star: int
    relative_error: float


def run_m3_scatter_study(
    dims: ProblemDims,
    *,
    batch_spec: GaussianBatchSpec,
    location: RunLocation,
    settings: GDSolveSettings,
    convergence_tolerance: float = 1e-3,
) -> M3ScatterStudy:
    """Section 9 (Asset 5's ``m=3`` instance), computation half: builds a
    THIRD, genuinely three-control-dimension LQR instance -- entirely
    separate from the primary and its own run directory -- via the exact same
    `setup_stochastic_lqr_experiment` boilerplate the primary instance uses,
    solves the signal-space iteration (7) on it with a Gauss-Seidel sweep
    from a cold start, validates convergence, and selects the instant of
    highest cross-ensemble variance in its optimal signal. Rendering lives
    with the Tier-4 adapter
    (`viz.adapters.notebook.render_local_cost_landscape_scatter_3d`).

    Args:
        dims: the instance's `ProblemDims` (``control_dim`` must be 3 for
            Asset 5's 3-component scatter).
        batch_spec: this instance's own disturbance draw, forwarded to
            `setup_stochastic_lqr_experiment`.
        location: this instance's own `LocalExperimentTracker` run, entirely
            separate from the primary instance's.
        settings: the signal-space gradient descent's configuration (step
            size, macro-iteration budget, dtype); the sweep topology is
            pinned to Gauss-Seidel regardless of ``settings.sweep_strategy``.
        convergence_tolerance: relative-error acceptance threshold the
            solved record's final cost must clear against this instance's
            own Riccati optimum (see `analysis.validate_convergence`).

    Returns:
        The solved `M3ScatterStudy`.
    """
    lqr = setup_stochastic_lqr_experiment(
        dims, batch_spec=batch_spec, location=location, dtype=settings.dtype
    )

    record = solve_signal_space_gd(
        lqr.problem,
        dims.horizon,
        lqr.realization,
        ConstantInitializer(0.0, dims.control_dim, settings.dtype, torch.device("cpu")),
        settings=replace(settings, sweep_strategy=GaussSeidelSweep()),
    )
    validate_convergence(
        {"m=3 signal-space GD": record}, lqr.J_opt, convergence_tolerance
    )
    relative_error = abs(record.J_final - lqr.J_opt) / abs(lqr.J_opt)
    t_star = int(np.argmax(lqr.U_opt.var(axis=0).sum(axis=-1)))
    return M3ScatterStudy(
        lqr=lqr, record=record, t_star=t_star, relative_error=relative_error
    )
