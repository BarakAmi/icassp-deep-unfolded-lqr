"""`OpenLoopGDController`: a pure state holder for the decision variable ``U``,
the ``U``-holder for the **autograd-differentiable open-loop channel**
(`models.open_loop.differentiable`) -- see that subpackage's docstring for
the full channel doctrine (when to prefer it over the engine-free
`models.analytic.iterative_gd.AnalyticalIterativeGDController`). This
controller does not solve anything itself -- the Runner (via
`engine.strategy.GradientDescentStrategy` and
`differentiable.AnalyticalGradientDescent`) drives the solve, one `.step()`
per Runner epoch; this controller only holds ``U`` and exposes it as an
open-loop policy, satisfying the `models.base.Controller` protocol.
"""

from dataclasses import dataclass

import torch

from ..base import Config
from ..iterative.initializers import ControlInitializer
from ..iterative.step_size import StepSizeSchedule
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy
from ...core.utils import ensure_positive_integer
from ...core.utils.signing import hash_array


@dataclass(frozen=True)
class OpenLoopGDConfig(Config):
    """Configuration for `OpenLoopGDController`.

    Attributes:
        step_size: The fixed, analytical, (possibly multi-dimensional)
            step-size schedule the Runner-driven `AnalyticalGradientDescent`
            optimizer consumes.
        control_initializer: Builds the initial control ``u^(0)`` (broadcast
            across the horizon to build ``U^(0)``).
        horizon: The control horizon ``T``.
        num_iterations: ``M``, the number of gradient-descent iterations --
            must equal the Runner phase's ``epochs`` and `step_size`'s own
            `StepSizeSchedule.num_iterations`.
        control_dim: The control vector dimension ``m``.
        batch_size: Number of parallel trajectories ``U`` holds (the
            Part-13.1-approved deterministic convergence proof uses ``1``).
        stop_tolerance: Optional epsilon for an `engine.callbacks
            .EarlyStoppingCallback` wired up alongside this controller;
            stored here only as configuration metadata -- this controller
            does not consult it itself.
    """

    step_size: StepSizeSchedule
    control_initializer: ControlInitializer
    horizon: int
    num_iterations: int
    control_dim: int
    batch_size: int = 1
    stop_tolerance: float | None = None

    def __post_init__(self) -> None:
        """Fail-fast guards: positive dimensions, a positive `stop_tolerance`
        if given, and `step_size`'s own dimensions agreeing with this
        config's `num_iterations`/`horizon`/`control_dim`.

        Raises:
            ValueError: If `horizon`/`num_iterations`/`control_dim`/
                `batch_size` is not a positive ``int``, `stop_tolerance` is
                given and not positive, or `step_size` disagrees with this
                config's own dimensions.
        """
        ensure_positive_integer(self.horizon, "horizon")
        ensure_positive_integer(self.num_iterations, "num_iterations")
        ensure_positive_integer(self.control_dim, "control_dim")
        ensure_positive_integer(self.batch_size, "batch_size")
        if self.stop_tolerance is not None and self.stop_tolerance <= 0:
            raise ValueError(
                f"stop_tolerance must be positive, got {self.stop_tolerance}."
            )
        if self.step_size.num_iterations != self.num_iterations:
            raise ValueError(
                "step_size.num_iterations "
                f"({self.step_size.num_iterations}) must equal "
                f"num_iterations ({self.num_iterations})."
            )
        if self.step_size.horizon != self.horizon:
            raise ValueError(
                f"step_size.horizon ({self.step_size.horizon}) must equal "
                f"horizon ({self.horizon})."
            )
        if self.step_size.control_dim != self.control_dim:
            raise ValueError(
                f"step_size.control_dim ({self.step_size.control_dim}) must "
                f"equal control_dim ({self.control_dim})."
            )


class OpenLoopGDController:
    """Pure state holder for the open-loop decision variable ``U``, shape
    ``(batch, horizon, control_dim)``. Deliberately **not** an `nn.Module`:
    ``U`` is a plain leaf tensor (``requires_grad=True``), and
    `AnalyticalGradientDescent` accepts any iterable of such leaves --
    `torch.optim.Optimizer` does not require `nn.Parameter`/`nn.Module`.

    Attributes:
        problem: The `OptimalControlProblem` this controller's cost is scored
            against.
        config: This controller's `OpenLoopGDConfig`.
        U: The live decision variable, shape ``(batch, horizon, control_dim)``
            -- mutated in place by `AnalyticalGradientDescent.step`.
    """

    def __init__(
        self,
        problem: OptimalControlProblem,
        config: OpenLoopGDConfig,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` to solve.
            config: This controller's horizon/iteration/step-size configuration.
            dtype: ``U``'s dtype.
            device: ``U``'s device.
        """
        self.problem = problem
        self.config = config
        obs_dim = problem.system.dimensions.observation_dim
        u0 = config.control_initializer(
            0,
            torch.zeros(config.batch_size, obs_dim, dtype=dtype, device=device),
            None,
        )
        U0 = u0.unsqueeze(1).repeat(1, config.horizon, 1)
        self.U = U0.clone().detach().requires_grad_(True)

    def get_control_policy(self) -> ControlPolicy:
        """Return the open-loop policy: ignores the observation entirely and
        replays the live decision variable `self.U`.

        Returns:
            A `ControlPolicy` ``lambda t, y: self.U[:, t]``.
        """
        return lambda t, y: self.U[:, t]

    def get_signature(self) -> dict:
        """`Signable` member: this controller's own type/config, including the
        fixed step-size schedule's shape and content hash -- never the live,
        in-training `U` (that already has a home as a
        `engine.callbacks.ParameterSnapshotCallback` artifact).

        Returns:
            ``{"type": "OpenLoopGDController", "horizon":, "control_dim":,
            "num_iterations":, "batch_size":, "step_size_shape":,
            "step_size_hash":}``.
        """
        return {
            "type": type(self).__name__,
            "horizon": self.config.horizon,
            "control_dim": self.config.control_dim,
            "num_iterations": self.config.num_iterations,
            "batch_size": self.config.batch_size,
            "step_size_shape": tuple(self.config.step_size.raw.shape),
            "step_size_hash": hash_array(
                self.config.step_size.raw.detach().cpu().numpy()
            ),
        }
