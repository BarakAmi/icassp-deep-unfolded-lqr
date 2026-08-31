"""Notebook setup assembly (T3.g *setup*): parameter specs in, ready-to-solve
experiment bundles out — the system draw, batch sampling, and Riccati-baseline
boilerplate the research notebooks need before any solver runs, with zero
display side effects.
"""

from dataclasses import dataclass
from pathlib import Path

from typing import cast

import numpy as np
import torch

from ..applications.rollout import RolloutModel
from ..core.cost.quadratic_cost import QuadraticCost
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.system.linear_system import LinearSystem
from ..engine.strategy import BatchSampler
from ..models.analytic.riccati import RiccatiController
from ..models.iterative import StepSizeSchedule
from ..models.samplers import (
    Distribution,
    NumpyGaussianDistribution,
    ZeroDistribution,
)
from ..persistence import LocalExperimentTracker


@dataclass(frozen=True)
class ProblemDims:
    """The three integers defining an LQR test instance's size.

    Attributes:
        state_dim: Dimension ``n`` of the state ``x_t``.
        control_dim: Dimension ``m`` of the control ``u_t``.
        horizon: Number of time steps ``T``.
    """

    state_dim: int
    control_dim: int
    horizon: int


@dataclass(frozen=True)
class RunLocation:
    """Where a run persists: the experiments root plus the run's name (the
    two constructor arguments of `LocalExperimentTracker`, as one value).

    Attributes:
        root: The experiments/artifacts root directory.
        name: The experiment name this run's tracker directory is keyed by.
    """

    root: Path | str
    name: str


@dataclass(frozen=True)
class GaussianBatchSpec:
    """The full specification of one Gaussian disturbance-batch draw (T3.c).

    Attributes:
        initial_state_std: Std of the Gaussian initial state
            ``x_0 ~ N(0, initial_state_std^2 I)``.
        noise_std: Std of the Gaussian process noise
            ``w_t ~ N(0, noise_std^2 I)`` (measurement noise is zero).
        batch_size: Monte Carlo ensemble size ``B``.
        seed: Seed for the single shared random stream.
    """

    initial_state_std: float
    noise_std: float
    batch_size: int
    seed: int


@dataclass(frozen=True)
class SamplerBundle:
    """A built batch sampler together with its signable ingredients.

    Attributes:
        sample: The `BatchSampler` closure — call with no arguments to draw
            one ``(initial_state, process_noise, measurement_noise)`` batch.
        distributions: The exact `Distribution` objects the closure calls —
            the single-source-of-truth objects `run_analytic_controller_rollout`
            passes to `ProblemSignatureCallback`, so what gets logged is never
            a parallel description of what gets sampled.
        batch_size: The ensemble size every draw produces.
    """

    sample: BatchSampler
    distributions: dict[str, Distribution]
    batch_size: int


@dataclass(frozen=True)
class DisturbanceRealization:
    """One fixed, already-drawn ``(x0, w)`` disturbance batch — the common
    input every signal-space GD variant in a study is evaluated on.

    Attributes:
        x0: Initial states, shape ``(batch, n)``.
        w: Process noise, shape ``(batch, T, n)``.
    """

    x0: np.ndarray
    w: np.ndarray


def make_gaussian_batch_sampler(
    state_dim: int,
    horizon: int,
    spec: GaussianBatchSpec,
) -> SamplerBundle:
    """Batch sampler for a fully observable system: i.i.d. Gaussian initial
    states and process noise (both `N(0, noise_std^2 I)`, sharing one seeded
    stream sequentially), zero measurement noise -- the `(initial_state,
    process_noise, measurement_noise)` shape `System.run` expects.

    Args:
        state_dim: Dimension ``n`` of the state (and noise) vectors.
        horizon: Number of time steps ``T`` each process-noise draw spans.
        spec: The `GaussianBatchSpec` describing the draw.

    Returns:
        A `SamplerBundle`: the sampler closure, the exact `Distribution`
        objects it calls, and the ensemble size.
    """
    # One shared rng: initial_state and process_noise must draw sequentially
    # from the SAME advancing stream (as this function's own single generator
    # always did), not two independent streams. Each carries its own std
    # (initial_state_std vs noise_std) so the initial-state spread and the
    # per-step process-noise spread can be configured independently.
    rng = np.random.default_rng(spec.seed)
    initial_state_dist = NumpyGaussianDistribution(
        std=spec.initial_state_std, seed=spec.seed, rng=rng
    )
    process_noise_dist = NumpyGaussianDistribution(
        std=spec.noise_std, seed=spec.seed, rng=rng
    )
    measurement_noise_dist = ZeroDistribution()

    def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0 = initial_state_dist(spec.batch_size, state_dim)
        w = process_noise_dist(spec.batch_size, horizon, state_dim)
        v = measurement_noise_dist(spec.batch_size, horizon, state_dim)
        return x0, w, v

    distributions: dict[str, Distribution] = {
        "initial_state": initial_state_dist,
        "process_noise": process_noise_dist,
        "measurement_noise": measurement_noise_dist,
    }
    return SamplerBundle(
        sample=sample, distributions=distributions, batch_size=spec.batch_size
    )


def generate_marginally_stable_system(
    state_dim: int, control_dim: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A random LTI (A, B) pair with A rescaled to spectral radius 1 --
    marginally stable, avoiding both divergent open-loop trajectories and a
    system so contractive that any controller performs adequately. The
    standard "hard but solvable" test instance for the LQR notebooks/
    applications in this project.

    Args:
        state_dim: Dimension n of A (n x n) and the state x_t.
        control_dim: Dimension m of B's columns (n x m).
        seed: Seed for the random generator, for reproducibility.

    Returns:
        A: (state_dim, state_dim), rescaled so max_i |lambda_i(A)| == 1.
        B: (state_dim, control_dim), standard normal entries.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((state_dim, state_dim))
    A /= np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.standard_normal((state_dim, control_dim))
    return A, B


@dataclass(frozen=True)
class StochasticLQRExperiment:
    """Everything a signal-space-optimization notebook needs to compare
    against the optimal feedback law: the problem itself, its closed-form
    (Riccati) baseline evaluated on one fixed disturbance batch, and that
    batch in both numpy and torch form -- the exact set `solve_primary`-style
    solvers and `RolloutModel` calls each expect as separate positional/
    keyword arguments, so callers destructure the fields they need rather
    than threading this dataclass itself through unrelated APIs.
    """

    system: LinearSystem
    cost: QuadraticCost
    problem: OptimalControlProblem
    riccati: RiccatiController
    x0_batch: np.ndarray
    w_batch: np.ndarray
    v_batch: np.ndarray
    x0_t: torch.Tensor
    w_t: torch.Tensor
    v_t: torch.Tensor
    X_opt: torch.Tensor
    U_opt: np.ndarray
    J_opt: float
    figures_dir: Path

    @property
    def realization(self) -> DisturbanceRealization:
        """The instance's fixed numpy ``(x0, w)`` batch, in the bundled shape
        every `solve_signal_space_gd`-family function consumes.

        Returns:
            The `DisturbanceRealization` over `x0_batch`/`w_batch`.
        """
        return DisturbanceRealization(x0=self.x0_batch, w=self.w_batch)


def setup_stochastic_lqr_experiment(
    dims: ProblemDims,
    *,
    batch_spec: GaussianBatchSpec,
    location: RunLocation,
    dtype: torch.dtype = torch.float32,
) -> StochasticLQRExperiment:
    """Builds a marginally-stable identity-weighted LQR instance, draws one
    fixed batch of `(x0, w)` realizations, and evaluates the optimal
    (Riccati) feedback law on it -- the complete Setup/Baseline boilerplate a
    signal-space-optimization notebook (e.g.
    `02_gradient_descent_lqr_poc.ipynb`) needs before any gradient-based
    solver runs, collected here so the notebook itself stays to the point
    (parameters in, `StochasticLQRExperiment` out).

    Args:
        dims: The instance's `ProblemDims` ``(n, m, T)``.
        batch_spec: The Gaussian disturbance draw; its `seed` is shared by
            the ``(A, B)`` system draw and the batch sampler.
        location: Where this run persists (`LocalExperimentTracker` root and
            name); the run's ``run_dir / "figures"`` is returned as
            `figures_dir`.
        dtype: torch dtype for the returned tensors.

    Returns:
        A `StochasticLQRExperiment` bundling the problem, its Riccati
        baseline, the drawn batch (numpy and torch), the optimal signal/cost,
        and this run's figures directory.
    """
    state_dim, control_dim, horizon = dims.state_dim, dims.control_dim, dims.horizon
    A, B = generate_marginally_stable_system(
        state_dim, control_dim, seed=batch_spec.seed
    )
    system = LinearSystem.fully_observable(A, B)
    # The backward Riccati recursion indexes Q[horizon]/R[k], so Q/R must be
    # genuinely time-stacked even though this problem's cost is time-invariant.
    Q = np.repeat(np.eye(state_dim)[None], horizon + 1, axis=0)
    R = np.repeat(np.eye(control_dim)[None], horizon, axis=0)
    cost = QuadraticCost(Q=Q, R=R)  # defaults: no terminal term, time-averaged
    problem = OptimalControlProblem(system=system, cost=cost)

    sampling = make_gaussian_batch_sampler(state_dim, horizon, batch_spec)
    x0_batch, w_batch, v_batch = sampling.sample()
    x0_t = torch.as_tensor(x0_batch, dtype=dtype)
    w_t = torch.as_tensor(w_batch, dtype=dtype)
    v_t = torch.as_tensor(v_batch, dtype=dtype)

    riccati = RiccatiController(problem, horizon)
    # x0/w/v are authored as tensors, so this rollout is torch-native.
    X_opt, _, U_opt_t = cast(
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
        problem.system.run(riccati.get_control_policy(), x0_t, w_t, v_t),
    )
    U_opt = U_opt_t.detach().cpu().numpy()
    # `.mean()`: the rollout returns one cost per trajectory since Annex 06
    # §4.3's correction, so the batch reduction is the caller's to make.
    J_opt = float(RolloutModel(riccati)(x0_t, w_t, v_t)[3].mean())

    tracker = LocalExperimentTracker(location.root, location.name)
    figures_dir = tracker.run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    return StochasticLQRExperiment(
        system=system,
        cost=cost,
        problem=problem,
        riccati=riccati,
        x0_batch=x0_batch,
        w_batch=w_batch,
        v_batch=v_batch,
        x0_t=x0_t,
        w_t=w_t,
        v_t=v_t,
        X_opt=X_opt,
        U_opt=U_opt,
        J_opt=J_opt,
        figures_dir=figures_dir,
    )


def build_step_size_schedule(
    control_dim: int,
    horizon: int,
    *,
    alpha: float,
    num_iterations: int,
    dtype: torch.dtype,
) -> StepSizeSchedule:
    """The fixed per-macro-iteration step size (7) as a `StepSizeSchedule`,
    shared by every signal-space gradient-descent solve in the GD-LQR PoC
    notebook (both `solve_signal_space_gd` and the standalone
    benchmarking/3-D-section solves that build an
    `AnalyticalIterativeGDController` directly) -- one construction, so a
    change to how `alpha` is broadcast across iterations/horizon/control
    dimension never has to be kept in sync across call sites.
    """
    return StepSizeSchedule(
        raw=torch.tensor(alpha, dtype=dtype),
        num_iterations=num_iterations,
        horizon=horizon,
        control_dim=control_dim,
    )
