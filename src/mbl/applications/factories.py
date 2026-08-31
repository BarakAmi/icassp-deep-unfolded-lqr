"""The single problem and sampler factories (REFACTOR_PLAN v3, T3.c):
`LQRProblemFactory` is the one home of the marginally-stable system draw and
the time-stacked identity-cost construction, and `GaussianBatchSpec` the one
home of the seeded Gaussian batch sampler -- each previously duplicated
verbatim across `StandardLQRApp`, `BoxConstraintLQRApp`, and
the notebook workbench layer (M1/M8). Both are frozen, signable specifications: the
factory is the `problem` leg of a run's signature, the batch spec part of
its `evaluation` leg.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from ..core.constraint.box_constraint import BoxConstraint
from ..core.cost.quadratic_cost import QuadraticCost
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.runtime import Backend
from ..core.system.linear_system import LinearSystem
from ..engine.strategy import BatchSampler
from ..models.samplers import (
    Distribution,
    NumpyGaussianDistribution,
    TorchGaussianDistribution,
    TorchZeroDistribution,
    ZeroDistribution,
)


@runtime_checkable
class BatchSpec(Protocol):
    """Structural seam (docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 2.6):
    anything that can build a signable evaluation/training batch sampler --
    widening `EvaluationProtocol.batch_spec`/`EngineHarness.batch_spec` from
    the single concrete `GaussianBatchSpec` to this Protocol is a TYPE-LEVEL
    change only (no runtime behavior, no signature content change for
    `GaussianBatchSpec` itself), exactly as `ProblemFactory` above documents
    for `LQRProblemFactory`/`LTVLQRProblemFactory` -- so it never invalidates
    a cache. `GaussianBatchSpec` and `applications.uncertainty.noise
    .ExoticBatchSpec` both satisfy it structurally, with no inheritance
    relationship between the two.
    """

    # Read-only `@property` stub, not a plain field annotation -- see
    # `ProblemFactory`'s identical note below on frozen-dataclass compliance.
    @property
    def batch_size(self) -> int: ...

    def build(
        self,
        backend: Backend,
        *,
        torch_dtype: torch.dtype = ...,
        torch_device: torch.device | None = None,
        batch_size: int | None = None,
        horizon: int | None = None,
    ) -> tuple[BatchSampler, dict[str, Distribution]]: ...

    def get_signature(self) -> dict[str, Any]: ...


@runtime_checkable
class ProblemFactory(Protocol):
    """Structural seam (NB05_LTV_BOX_CONSTRAINED_PLAN.md Sec 6.2): anything
    that can build a signable `OptimalControlProblem` of known
    ``(state_dim, control_dim, horizon)`` -- widening `Experiment.problem`/
    `CaseStudy.problem` from the single concrete `LQRProblemFactory` to this
    protocol is a TYPE-LEVEL change only (no runtime behavior, no signature
    content), so it never invalidates a cache. `LQRProblemFactory` and
    `applications.ltv_factories.LTVLQRProblemFactory` both satisfy it
    structurally, with no inheritance relationship between the two --
    exactly the point of a `Protocol` here (they share no implementation,
    only this shape).
    """

    # Read-only `@property` stubs, not plain field annotations: a plain
    # `state_dim: int` would require a SETTABLE attribute for structural
    # compliance, which a frozen dataclass's fields are not (mypy models
    # them as read-only) -- `LQRProblemFactory`/`LTVLQRProblemFactory` would
    # then fail this Protocol under strict mypy despite satisfying it at
    # runtime. A property stub matches both a `@property` implementation and
    # a frozen dataclass field.
    @property
    def state_dim(self) -> int: ...
    @property
    def control_dim(self) -> int: ...
    @property
    def horizon(self) -> int: ...

    def build(self) -> OptimalControlProblem: ...

    def get_signature(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LQRProblemFactory:
    """Builds the shared LQR benchmark problem: a seeded, marginally stable
    LTI system with time-stacked identity quadratic costs, optionally under
    an infinity-norm box bound on the control.

    Attributes:
        state_dim: State dimension ``n``.
        control_dim: Control dimension ``m``.
        horizon: The finite horizon ``N`` the cost matrices are stacked for.
        seed: Seed of the system draw.
        u_max: Optional box bound; ``None`` attaches no constraint.
    """

    state_dim: int
    control_dim: int
    horizon: int
    seed: int
    u_max: float | None = None

    def build(self) -> OptimalControlProblem:
        """Construct the problem this factory specifies.

        Returns:
            The `OptimalControlProblem` (with a single `BoxConstraint` when
            `u_max` is set).
        """
        n, m, horizon = self.state_dim, self.control_dim, self.horizon
        rng = np.random.default_rng(self.seed)

        A = rng.normal(size=(n, n))
        A /= np.max(
            np.abs(np.linalg.eigvals(A))
        )  # normalize to a marginally stable system
        B = rng.normal(size=(n, m))
        system = LinearSystem.fully_observable(A, B)

        # RiccatiController's finite_horizon_riccati indexes Q[horizon]/R[k],
        # so Q/R must be genuinely time-stacked even though this problem's
        # cost is itself time-invariant.
        Q = np.repeat(np.eye(n)[None], horizon + 1, axis=0)
        R = np.repeat(np.eye(m)[None], horizon, axis=0)
        cost = QuadraticCost(Q=Q, R=R)

        constraints = [BoxConstraint(u_max=self.u_max)] if self.u_max else None
        if constraints is None:
            return OptimalControlProblem(system=system, cost=cost)
        return OptimalControlProblem(system=system, cost=cost, constraints=constraints)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full problem specification.

        Returns:
            ``{"type": "LQRProblemFactory", **fields}``.
        """
        return {"type": type(self).__name__, **dataclasses.asdict(self)}


@dataclass(frozen=True)
class GaussianBatchSpec:
    """One specification for the seeded Gaussian evaluation/training batches
    both backends draw: standard-normal initial states, scaled Gaussian
    process noise, zero measurement noise.

    Attributes:
        state_dim: State (and noise) dimension ``n``.
        horizon: Number of noise steps per trajectory.
        batch_size: Trajectories per batch.
        seed: Seed of the shared random stream.
        process_noise_std: Standard deviation of the process noise.
        initial_state_std: Standard deviation of the initial state.
    """

    state_dim: int
    horizon: int
    batch_size: int
    seed: int
    process_noise_std: float
    initial_state_std: float = 1.0

    def build(
        self,
        backend: Backend,
        *,
        torch_dtype: torch.dtype = torch.float64,
        torch_device: torch.device | None = None,
        batch_size: int | None = None,
        horizon: int | None = None,
    ) -> tuple[BatchSampler, dict[str, Distribution]]:
        """Build the batch sampler for `backend`, returning alongside it the
        exact `Distribution` objects the closure calls -- the
        single-source-of-truth objects handed to `ProblemSignatureCallback`,
        so what gets logged is never a parallel description of what gets
        sampled.

        Args:
            backend: Which array backend the batches are authored on.
            torch_dtype: dtype of torch-backed batches (ignored for NumPy).
            torch_device: Residency of torch-backed batches (ignored for
                NumPy); defaults to CPU. Every draw still runs on a CPU
                generator regardless (`TorchGaussianDistribution`'s
                bit-for-bit-identical-for-a-given-seed contract), so this
                only changes where the result *lives*, never its values.
            batch_size: Optional per-family override of `self.batch_size`
                (e.g. COCP's smaller training batches).
            horizon: Optional per-family override of `self.horizon`.

        Returns:
            ``(sampler, distributions)``: a zero-argument batch sampler and
            the named `Distribution` map
            (``initial_state``/``process_noise``/``measurement_noise``).
        """
        effective_batch = batch_size or self.batch_size
        effective_horizon = horizon or self.horizon
        if backend is Backend.NUMPY:
            distributions = self._numpy_distributions()
        else:
            distributions = self._torch_distributions(torch_dtype, torch_device)

        n = self.state_dim
        initial_state = distributions["initial_state"]
        process_noise = distributions["process_noise"]
        measurement_noise = distributions["measurement_noise"]

        def sample() -> tuple[Any, Any, Any]:
            x0 = initial_state(effective_batch, n)
            w = process_noise(effective_batch, effective_horizon, n)
            v = measurement_noise(effective_batch, effective_horizon, n)
            return x0, w, v

        return sample, distributions

    def _numpy_distributions(self) -> dict[str, Distribution]:
        # One shared rng: initial_state and process_noise must draw
        # sequentially from the SAME advancing stream (as a single rng.normal
        # call per signal always has), not two independent streams.
        rng = np.random.default_rng(self.seed)
        return {
            "initial_state": NumpyGaussianDistribution(
                std=self.initial_state_std, seed=self.seed, rng=rng
            ),
            "process_noise": NumpyGaussianDistribution(
                std=self.process_noise_std, seed=self.seed, rng=rng
            ),
            "measurement_noise": ZeroDistribution(),
        }

    def _torch_distributions(
        self, dtype: torch.dtype, device: torch.device | None
    ) -> dict[str, Distribution]:
        # One shared generator, for the same reason as the numpy stream above.
        # Always constructed on CPU (never `device`): TorchGaussianDistribution
        # draws on the generator's own device and only relocates to `device`
        # afterward, so the sampled values stay bit-for-bit identical to the
        # CPU-only baseline regardless of where the caller wants them to live.
        generator = torch.Generator().manual_seed(self.seed)
        return {
            "initial_state": TorchGaussianDistribution(
                std=self.initial_state_std,
                seed=self.seed,
                dtype=dtype,
                device=device,
                generator=generator,
            ),
            "process_noise": TorchGaussianDistribution(
                std=self.process_noise_std,
                seed=self.seed,
                dtype=dtype,
                device=device,
                generator=generator,
            ),
            "measurement_noise": TorchZeroDistribution(dtype=dtype, device=device),
        }

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full sampling specification.

        Returns:
            ``{"type": "GaussianBatchSpec", **fields}``.
        """
        return {"type": type(self).__name__, **dataclasses.asdict(self)}
