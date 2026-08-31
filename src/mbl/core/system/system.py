"""Abstract dynamical-system interface and its core type aliases.

All array-shaped types below use ``(batch, ...)`` conventions: a
`BatchedStateSpaceVector` is a single-time-step batch of vectors, shape
``(batch, dim)``; a `BatchedHorizonStateSpaceVector` is a full horizon of such
batches, shape ``(batch, horizon, dim)``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import Tensor

from ..utils import ensure_positive_integer

type BatchedStateSpaceVector = (
    np.ndarray[tuple[int, int], np.dtype[np.floating]] | Tensor
)
"""A batch of vectors at one time step: shape ``(batch, dim)``."""

type BatchedHorizonStateSpaceVector = (
    np.ndarray[tuple[int, int, int], np.dtype[np.floating]] | Tensor
)
"""A batch of per-time-step vectors over a full horizon: shape ``(batch, horizon, dim)``."""

type StateMap = Callable[
    [int, BatchedStateSpaceVector, BatchedStateSpaceVector, BatchedStateSpaceVector],
    BatchedStateSpaceVector,
]
"""State-transition map: ``(t, x_t, u_t, w_t) -> x_{t+1}``, all batched vectors."""

type ObservationMap = Callable[
    [int, BatchedStateSpaceVector, BatchedStateSpaceVector], BatchedStateSpaceVector
]
"""Observation map: ``(t, x_t, v_t) -> y_t``, all batched vectors."""

type ControlPolicy = Callable[[int, BatchedStateSpaceVector], BatchedStateSpaceVector]
"""Control policy: ``(t, y_t) -> u_t``, both batched vectors."""

type TimeStepCallback = Callable[
    [
        int,
        BatchedHorizonStateSpaceVector,
        BatchedHorizonStateSpaceVector,
        BatchedHorizonStateSpaceVector,
    ],
    None,
]
"""Per-step observer: ``(t, states, observations, controls) -> None``, invoked
during `System.run` with the trajectories accumulated so far."""


@dataclass(frozen=True)
class StateSpaceModelDimensions:
    """The vector dimensions defining one state-space model's I/O shapes.

    Attributes:
        state_dim: Dimension ``n`` of the state vector ``x_t``.
        control_dim: Dimension ``m`` of the control vector ``u_t``.
        process_noise_dim: Dimension of the process noise vector ``w_t``.
        observation_dim: Dimension ``p`` of the observation vector ``y_t``.
        measurement_noise_dim: Dimension of the measurement noise vector ``v_t``.
    """

    state_dim: int
    control_dim: int
    process_noise_dim: int
    observation_dim: int
    measurement_noise_dim: int

    def __post_init__(self) -> None:
        """Fail-fast guard: every dimension must be a positive integer.

        Raises:
            ValueError: If any field is not a strictly positive ``int``.
        """
        for field_name in self.__dataclass_fields__:
            ensure_positive_integer(getattr(self, field_name), field_name)


class System(ABC):
    """Abstract base class for dynamical systems.

    Concrete subclasses (e.g. `core.system.state_space_system.StateSpaceSystem`)
    implement `run` to simulate a batched trajectory under a given policy.

    Attributes:
        dimensions: The system's I/O vector dimensions -- every concrete
            system declares them (constructors take them explicitly), and
            interior code may rely on their presence.
    """

    dimensions: StateSpaceModelDimensions

    @abstractmethod
    def run(
        self,
        policy: ControlPolicy,
        initial_state: BatchedStateSpaceVector,
        process_noises: BatchedHorizonStateSpaceVector,
        measurement_noises: BatchedHorizonStateSpaceVector,
    ) -> tuple[
        BatchedHorizonStateSpaceVector,
        BatchedHorizonStateSpaceVector,
        BatchedHorizonStateSpaceVector,
    ]:
        """Execute one full simulation rollout.

        Args:
            policy: Maps ``(t, y_t) -> u_t`` at each time step.
            initial_state: Initial state ``x_0``, shape ``(batch, state_dim)``.
            process_noises: Process noise ``w_t`` for every step, shape
                ``(batch, horizon, process_noise_dim)``.
            measurement_noises: Measurement noise ``v_t`` for every step, shape
                ``(batch, horizon, measurement_noise_dim)``.

        Returns:
            A ``(states, observations, controls)`` tuple:
            states of shape ``(batch, horizon+1, state_dim)``,
            observations of shape ``(batch, horizon, observation_dim)``,
            controls of shape ``(batch, horizon, control_dim)``.

        Raises:
            NotImplementedError: Always, on this abstract base class.
        """
        raise NotImplementedError

    def get_signature(self) -> dict[str, Any]:
        """`Signable` default: just the concrete class name.

        Subclasses that carry actual matrices/dimensions (e.g.
        `core.system.state_space_system.StateSpaceSystem`) override this with
        richer content; this default lets any future `System` satisfy
        `Signable` immediately, even before it's updated (Open/Closed).

        Returns:
            ``{"type": <class name>}``.
        """
        return {"type": type(self).__name__}
