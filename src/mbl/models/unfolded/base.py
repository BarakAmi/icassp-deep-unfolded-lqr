from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .parameters import UnfoldedParameter
from ..iterative.refinement import IterativeRefinement
from ..iterative.initializers import ControlInitializer
from ..base import Config
from ..registry import register_model
from ...core.optimal_control_problem import OptimalControlProblem
from ...core.system.state_space_system import ControlPolicy
from ...core.utils import ensure_positive_integer, to_numpy


@dataclass(frozen=True)
class UnfoldingConfig(Config):
    """Per-controller configuration for UnfoldedController.

    `horizon` lives here (like RiccatiController's own `horizon` constructor
    argument) rather than on OptimalControlProblem or Cost: a System/Cost pair
    can be rolled out for any horizon, so horizon is a property of *this
    controller's execution*, not of the problem itself. Deriving it from a
    cost matrix's shape (e.g. R.shape[0]) would silently break whenever Q/R
    are given in their time-invariant (2D) form.

    Attributes:
        parameters: Named learnable parameters (e.g. ``{"step_size": ...}``),
            shared by reference with `iterative_refinement`.
        control_initializer: Produces each time step's starting control u_0.
        iterative_refinement: Refines u_0 into the final control at each step.
        horizon: The rollout horizon length.
    """

    parameters: Mapping[str, UnfoldedParameter]
    control_initializer: ControlInitializer
    iterative_refinement: IterativeRefinement
    horizon: int

    def __post_init__(self) -> None:
        """Fail-fast guard: `horizon` must be a positive integer.

        Raises:
            ValueError: If `horizon` is not a positive ``int``.
        """
        ensure_positive_integer(self.horizon, "horizon")


@register_model("unfolded")
class UnfoldedController(nn.Module):
    """A deep-unfolded (unrolled gradient-descent) controller: at each time
    step, `config.control_initializer` proposes a starting control and
    `config.iterative_refinement` refines it via one or more learnable
    gradient-descent iterations -- the "unfolded" analogue of solving the LQR
    problem via gradient descent instead of the closed-form Riccati recursion.

    Attributes:
        problem: The `OptimalControlProblem` this controller solves.
        config: This controller's `UnfoldingConfig`.
    """

    def __init__(self, problem: OptimalControlProblem, config: UnfoldingConfig) -> None:
        """
        Args:
            problem: The `OptimalControlProblem` to solve.
            config: The controller's parameters, initializer, refinement
                strategy, and horizon.
        """
        super().__init__()
        self.problem = problem
        self.config = config
        self._parameter_module: nn.Module | None = None

    def as_module(self) -> nn.Module:
        """`TrainableController` seam (T2.b): the trainable ``nn.Module``
        through which the engine sees this family's parameters.

        This family's learnable parameters are structured `UnfoldedParameter`
        objects held in `config.parameters` (never registered as submodule
        attributes), so ``self.parameters()`` alone is empty -- the exact gap
        the pre-S3 applications patched with per-family ``p.get_raw()``
        extraction ladders. Here each parameter's raw ``nn.Parameter`` is
        registered, by its configured name, on one container module: optimizer
        construction (``as_module().parameters()``) and checkpointing
        (``as_module().state_dict()``) both flow through it, aliasing (never
        copying) the live tensors the refinement reads.

        The container is built once and cached so the optimizer, checkpoints,
        and any resume all observe the same module identity.

        Returns:
            The container ``nn.Module`` exposing every learnable parameter.
        """
        if self._parameter_module is None:
            module = nn.Module()
            for name, parameter in self.config.parameters.items():
                raw = parameter.get_raw()
                assert isinstance(raw, nn.Parameter)  # get_raw returns the live leaf
                module.register_parameter(name, raw)
            self._parameter_module = module
        return self._parameter_module

    def initialize(self) -> None:
        """Re-initialize every learnable parameter in `config.parameters` (see
        `UnfoldedParameter.initialize`).

        Invalidates the cached `as_module` container: re-initialization
        constructs fresh raw ``nn.Parameter`` objects, so a previously built
        container would silently alias the orphaned tensors."""
        for param in self.config.parameters.values():
            param.initialize()
        self._parameter_module = None

    def forward(
        self,
        initial_state: torch.Tensor,
        process_noise: torch.Tensor | None = None,
        measurement_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Forward pass with gradient-based control updates.

        Args:
            initial_state: Initial state x0, shape (batch_size, state_dim).
            process_noise: Process noise, shape (batch_size, horizon, state_dim);
                defaults to zeros if not given.
            measurement_noise: Measurement noise, shape
                (batch_size, horizon, observation_dim); defaults to zeros if
                not given.

        Returns:
            X: State trajectory (batch_size, horizon+1, state_dim), autograd-connected.
            Y: Observation trajectory (batch_size, horizon, observation_dim), autograd-connected.
            U: Control trajectory (batch_size, horizon, action_dim), autograd-connected.
            cost: Scalar (numpy, non-differentiable) cost from `problem.cost`
                -- see the note below on why this one value alone is not
                part of the autograd graph.
        """
        dtype, device = initial_state.dtype, initial_state.device
        batch_size, state_dim, horizon, observation_dim = (
            *initial_state.shape,
            self.config.horizon,
            self.problem.system.dimensions.observation_dim,
        )
        control_policy = self.get_control_policy()
        process_noise = (
            process_noise
            if process_noise is not None
            else torch.zeros(
                (batch_size, horizon, state_dim), device=device, dtype=dtype
            )
        )
        measurement_noise = (
            measurement_noise
            if measurement_noise is not None
            else torch.zeros(
                (batch_size, horizon, observation_dim), device=device, dtype=dtype
            )
        )
        # This family's rollout is torch-native end to end (autograd law);
        # System.run's static union narrows to tensors here.
        X, Y, U = cast(
            "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
            self.problem.system.run(
                policy=control_policy,
                initial_state=initial_state,
                process_noises=process_noise,
                measurement_noises=measurement_noise,
            ),
        )
        # Cost is defined over the state trajectory X (horizon+1 steps), not the
        # observation trajectory Y (horizon steps) -- passing Y here would always
        # fail QuadraticCost's own horizon validation.
        #
        # QuadraticCost is numpy-only (a separate, tracked issue -- it doesn't
        # accept torch.Tensor at all). X/U above now correctly preserve the
        # torch autograd graph after the System.run() fix, so detach+convert
        # just for this call; forward()'s returned X/Y/U are still the live,
        # gradient-carrying tensors -- only the `cost` value here is a
        # non-differentiable numpy result until QuadraticCost itself supports
        # torch tensors.
        cost = self.problem.cost(to_numpy(X), to_numpy(U))
        return X, Y, U, cost

    def get_control_policy(self) -> ControlPolicy:
        """
        Called once per forward pass → creates a fresh closure with prev_u = None.
        No manual reset needed; prev_u scope is exactly one trajectory.

        Returns:
            A `ControlPolicy`: ``(t, y_t) -> u_t``, refining
            `config.control_initializer`'s proposal via
            `config.iterative_refinement` at every step.
        """
        # Rollout boundary: let the refinement invalidate any per-forward cache
        # (e.g. RiccatiRefinement's batched horizon gradient matrices) so this
        # rollout rebuilds a fresh autograd graph from the current parameters.
        self.config.iterative_refinement.on_rollout_start()
        prev_u: torch.Tensor | None = None

        def policy(t: int, y: torch.Tensor) -> torch.Tensor:
            nonlocal prev_u
            u = self.config.control_initializer(t, y, prev_u)
            u = self.config.iterative_refinement(t, y, u)
            prev_u = u.detach()  # zero-copy; breaks the graph across time steps
            return u

        return cast("ControlPolicy", policy)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: type + horizon + every named parameter's own
        signature (init config, never a live/trained tensor -- see
        `UnfoldedParameter.get_signature`).

        The controlled `problem`'s signature is intentionally omitted --
        `ProblemSignatureCallback` already logs it once at the root
        (``problem.*``); embedding it again here would just duplicate every
        key under ``controller.problem.*``.

        Returns:
            ``{"type": "UnfoldedController", "horizon":, "parameters": {...}}``.
        """
        return {
            "type": type(self).__name__,
            "horizon": self.config.horizon,
            "parameters": {
                name: param.get_signature()
                for name, param in self.config.parameters.items()
            },
        }
