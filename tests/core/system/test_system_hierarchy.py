import numpy as np
import pytest

from mbl.core.system.linear_system import LinearSystem
from mbl.core.system.state_space_system import (
    StateSpaceModelDimensions,
    StateSpaceSystem,
)
from mbl.core.system.system import System


class _ConcreteSystem(System):
    def run(self, *args, **kwargs):
        return "ok"


def _build_dims() -> StateSpaceModelDimensions:
    return StateSpaceModelDimensions(
        state_dim=2,
        control_dim=1,
        process_noise_dim=2,
        observation_dim=2,
        measurement_noise_dim=2,
    )


def _build_basic_ssm() -> StateSpaceSystem:
    dims = _build_dims()
    return StateSpaceSystem(
        state_transition_map=lambda t, x, u, w: (
            x + np.repeat(u, dims.state_dim, axis=1) + w
        ),
        observation_map=lambda t, x, v: x + v,
        dimensions=dims,
    )


def test_system_is_abstract() -> None:
    with pytest.raises(TypeError):
        System()  # type: ignore[abstract]  # instantiating the ABC is the test


def test_concrete_system_run() -> None:
    assert _ConcreteSystem().run() == "ok"


def test_dimensions_validate_positive_integers() -> None:
    with pytest.raises(ValueError, match="state_dim"):
        StateSpaceModelDimensions(
            state_dim=0,
            control_dim=1,
            process_noise_dim=1,
            observation_dim=1,
            measurement_noise_dim=1,
        )


def test_state_space_run_shapes_and_values_deterministic() -> None:
    ssm = _build_basic_ssm()

    batch, horizon = 3, 4
    initial_state = np.zeros((batch, ssm.dimensions.state_dim))
    process_noises = np.zeros((batch, horizon, ssm.dimensions.process_noise_dim))
    measurement_noises = np.zeros(
        (batch, horizon, ssm.dimensions.measurement_noise_dim)
    )

    def policy(t, x):
        return np.ones((x.shape[0], ssm.dimensions.control_dim))

    states, observations, controls = ssm.run(
        policy, initial_state, process_noises, measurement_noises
    )

    assert states.shape == (batch, horizon + 1, ssm.dimensions.state_dim)
    assert observations.shape == (batch, horizon, ssm.dimensions.observation_dim)
    assert controls.shape == (batch, horizon, ssm.dimensions.control_dim)


def test_run_with_control_matches_run() -> None:
    ssm = _build_basic_ssm()

    batch, horizon = 2, 5
    initial_state = np.zeros((batch, ssm.dimensions.state_dim))
    process_noises = np.zeros((batch, horizon, ssm.dimensions.process_noise_dim))
    measurement_noises = np.zeros(
        (batch, horizon, ssm.dimensions.measurement_noise_dim)
    )
    controls = np.ones((batch, horizon, ssm.dimensions.control_dim))

    direct = ssm.run_with_control(
        controls, initial_state, process_noises, measurement_noises
    )
    via_policy = ssm.run(
        lambda t, x: controls[:, t], initial_state, process_noises, measurement_noises
    )

    for left, right in zip(direct, via_policy):
        assert np.allclose(left, right)


def test_run_rejects_non_callable_policy() -> None:
    ssm = _build_basic_ssm()
    initial_state = np.zeros((1, ssm.dimensions.state_dim))
    process_noises = np.zeros((1, 1, ssm.dimensions.process_noise_dim))
    measurement_noises = np.zeros((1, 1, ssm.dimensions.measurement_noise_dim))

    with pytest.raises(TypeError, match="policy"):
        ssm.run("not-callable", initial_state, process_noises, measurement_noises)


def test_linear_system_time_varying_and_time_invariant_produce_expected_shapes() -> (
    None
):
    """LinearSystem supports both a time-invariant construction (2D matrices, broadcast
    across the horizon) and a time-varying one (3D, per-step matrices) via TimeSeriesMatrix."""
    horizon = 3
    batch = 2

    A_t = np.repeat(np.eye(2)[None, :, :], horizon, axis=0)
    B_t = np.repeat(np.array([[[1.0], [0.0]]]), horizon, axis=0)
    C_t = np.repeat(np.array([[[1.0, 0.0]]]), horizon, axis=0)

    time_varying = LinearSystem(A_t=A_t, B_t=B_t, C_t=C_t)
    time_invariant = LinearSystem(
        A_t=np.eye(2), B_t=np.array([[1.0], [0.0]]), C_t=np.array([[1.0, 0.0]])
    )

    assert time_varying.is_time_invariant()
    assert time_invariant.is_time_invariant()

    controls = np.ones((batch, horizon, 1))
    initial_state = np.zeros((batch, 2))
    process_noises = np.zeros((batch, horizon, 2))
    measurement_noises = np.zeros((batch, horizon, 1))

    time_varying_out = time_varying.run_with_control(
        controls, initial_state, process_noises, measurement_noises
    )
    time_invariant_out = time_invariant.run_with_control(
        controls, initial_state, process_noises, measurement_noises
    )

    assert time_varying_out[0].shape == (batch, horizon + 1, 2)
    assert time_invariant_out[0].shape == (batch, horizon + 1, 2)
    assert np.allclose(time_varying_out[0], time_invariant_out[0])


def test_linear_system_fully_observable_factory() -> None:
    A_t = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    B_t = np.repeat(np.array([[[1.0], [0.0]]]), 3, axis=0)

    system = LinearSystem.fully_observable(A_t, B_t)

    batch, horizon = 2, 3
    initial_state = np.ones((batch, 2))
    process_noises = np.zeros((batch, horizon, 2))
    measurement_noises = np.zeros((batch, horizon, 2))
    controls = np.zeros((batch, horizon, 1))

    states, observations, _ = system.run_with_control(
        controls, initial_state, process_noises, measurement_noises
    )
    assert np.allclose(observations, states[:, :-1, :])


def test_system_run_signature_is_typed_and_named() -> None:
    """Regression test: System.run's abstract signature names its parameters (policy,
    initial_state, process_noises, measurement_noises) instead of *args/**kwargs, so
    callers that pass matching keyword arguments work against any System subclass."""
    dims = _build_dims()
    ssm = _build_basic_ssm()

    states, observations, controls = ssm.run(
        policy=lambda t, x: np.ones((x.shape[0], dims.control_dim)),
        initial_state=np.zeros((1, dims.state_dim)),
        process_noises=np.zeros((1, 2, dims.process_noise_dim)),
        measurement_noises=np.zeros((1, 2, dims.measurement_noise_dim)),
    )
    assert states.shape == (1, 3, dims.state_dim)
