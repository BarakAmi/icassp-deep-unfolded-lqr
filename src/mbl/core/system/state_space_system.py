from dataclasses import asdict
from collections.abc import Callable
from typing import Any, Self, Sequence

from .system import (
    System,
    BatchedStateSpaceVector,
    BatchedHorizonStateSpaceVector,
    StateMap,
    ObservationMap,
    ControlPolicy,
    StateSpaceModelDimensions,
    TimeStepCallback,
)
from ..utils import (
    ensure_callable,
    stack_arrays,
)
from ..utils.shapes import enforce_tensor_shapes
from ..profiling import profiled


class StateSpaceSystem(System):
    """Represents a discrete-time state-space model with time-varying dynamics and observation maps.
    
    The model is defined by sequences of state transition maps and observation maps, which can vary at each time step. 
    The state transition map defines how the state evolves from time t to t+1 based on the current state, control input, and process noise. 
    The observation map defines how the observations are generated from the current state and measurement noise.
    
    $$x_{t+1} = f_t(x_t, u_t, w_t) \\
    y_t = h_t(x_t, v_t)$$

    Attributes:
        state_transition_map: A function that defines the state transition dynamics at each time step.
        observation_map: A function that defines the observation model at each time step.
        dimensions: An object that specifies the dimensions of the state, control, process noise, observation, and measurement noise vectors.
    """

    def __init__(
        self,
        state_transition_map: StateMap,
        observation_map: ObservationMap,
        dimensions: StateSpaceModelDimensions,
    ) -> None:
        """
        Args:
            state_transition_map: ``(t, x_t, u_t, w_t) -> x_{t+1}``.
            observation_map: ``(t, x_t, v_t) -> y_t``.
            dimensions: The model's vector dimensions.
        """
        self.state_transition_map = state_transition_map
        self.observation_map = observation_map
        self.dimensions = dimensions
        self._time_step_callbacks: list[TimeStepCallback] = []

    def run_with_control(
        self,
        controls: BatchedHorizonStateSpaceVector,
        initial_state: BatchedStateSpaceVector,
        process_noises: BatchedHorizonStateSpaceVector,
        measurement_noises: BatchedHorizonStateSpaceVector,
    ) -> tuple[
        BatchedHorizonStateSpaceVector,
        BatchedHorizonStateSpaceVector,
        BatchedHorizonStateSpaceVector,
    ]:
        """Roll out a *fixed*, pre-computed control sequence (open-loop),
        instead of a policy that reacts to each step's observation.

        Args:
            controls: Pre-computed controls for every step, shape
                ``(batch, horizon, control_dim)``.
            initial_state: Initial state ``x_0``, shape ``(batch, state_dim)``.
            process_noises: Process noise, shape
                ``(batch, horizon, process_noise_dim)``.
            measurement_noises: Measurement noise, shape
                ``(batch, horizon, measurement_noise_dim)``.

        Returns:
            The same ``(states, observations, controls)`` tuple as `run`; the
            returned controls equal `controls` (reproduced via a
            constant-per-step policy).
        """

        return self.run(
            lambda t, y: controls[:, t],
            initial_state,
            process_noises,
            measurement_noises,
        )

    @profiled("system.run")
    @enforce_tensor_shapes(
        initial_state=("B", "n"),
        process_noises=("B", "T", "w"),
        measurement_noises=("B", "T", "v"),
        returns=[("B", "T+1", "n"), ("B", "T", "p"), ("B", "T", "m")],
    )
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
        """Execute one full closed-loop simulation rollout.

        At each step ``t``: observes ``y_t = observation_map(t, x_t, v_t)``,
        queries ``u_t = policy(t, y_t)``, then advances
        ``x_{t+1} = state_transition_map(t, x_t, u_t, w_t)``.

        Args:
            policy: Maps ``(t, y_t) -> u_t``, shape ``(batch, control_dim)``.
            initial_state: Initial state ``x_0``, shape ``(batch, n)``.
            process_noises: Process noise for every step, shape
                ``(batch, T, process_noise_dim)``.
            measurement_noises: Measurement noise for every step, shape
                ``(batch, T, measurement_noise_dim)``.

        Returns:
            A ``(states, observations, controls)`` tuple: states of shape
            ``(batch, T+1, n)``, observations of shape ``(batch, T, p)``,
            controls of shape ``(batch, T, m)``.

        Raises:
            TypeError: If `policy` is not callable.
            ValueError: (via the `enforce_tensor_shapes` guard, when the
                ``SHAPE`` `GuardTier` is active) if `initial_state`,
                `process_noises`, or `measurement_noises` have the wrong rank,
                or disagree on batch size ``B``/horizon ``T``.
        """
        ensure_callable("policy", policy)

        horizon = process_noises.shape[1]
        states: list[Any] = [None] * (horizon + 1)
        observations: list[Any] = [None] * horizon
        controls: list[Any] = [None] * horizon
        states[0] = initial_state

        # Hoist per-step lookups out of the hot temporal loop: bind the maps to
        # locals and resolve the callback list once (an empty tuple when unset),
        # avoiding a `hasattr` probe on every one of the `horizon` iterations.
        observation_map = self.observation_map
        state_transition_map = self.state_transition_map
        callbacks = getattr(self, "_time_step_callbacks", ())

        for t in range(horizon):
            observations[t] = observation_map(t, states[t], measurement_noises[:, t])
            controls[t] = policy(t, observations[t])
            states[t + 1] = state_transition_map(
                t, states[t], controls[t], process_noises[:, t]
            )
            for callback in callbacks:
                callback(t, states, observations, controls)

        return (
            stack_arrays(states, axis=1),
            stack_arrays(observations, axis=1),
            stack_arrays(controls, axis=1),
        )

    def set_time_step_callbacks(
        self, callbacks: Sequence[Callable[[Self], TimeStepCallback]]
    ) -> None:
        """Set a custom callback function to be called at each time step during the run method.

        Args:
            callbacks: A sequence of functions that take the system instance and returns a TimeStepCallback function.
            The TimeStepCallback function will be called at each time step with the current time step, state trajectory, observation trajectory, and control trajectory.
        """
        self._time_step_callbacks = [cb(self) for cb in callbacks]

    @classmethod
    def fully_observable(
        cls, state_transition_map: StateMap, dimensions: StateSpaceModelDimensions
    ) -> "StateSpaceSystem":
        """Factory method for creating a fully observable state-space system
        where the observation is simply the state plus measurement noise
        (``y_t = x_t + v_t``, requiring ``observation_dim == state_dim``).

        Args:
            state_transition_map: ``(t, x_t, u_t, w_t) -> x_{t+1}``.
            dimensions: The model's vector dimensions.

        Returns:
            A `StateSpaceSystem` using the identity observation map.
        """

        def observation_map(
            t: int, x: BatchedStateSpaceVector, v: BatchedStateSpaceVector
        ) -> BatchedStateSpaceVector:
            return x + v

        return cls(state_transition_map, observation_map, dimensions)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` override: dimensions only.

        The state-transition/observation maps here are opaque closures with
        no matrices to hash in the general case; `LinearSystem` overrides this
        further with actual matrix content hashes.

        Returns:
            ``{"type": <class name>, **dimensions fields}``.
        """
        return super().get_signature() | asdict(self.dimensions)
