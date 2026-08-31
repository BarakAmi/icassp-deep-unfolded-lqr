import numpy as np
import pytest

from mbl.core.system.linear_system import LinearSystem, TimeSeriesMatrix


def test_time_series_matrix_2d_input_is_time_invariant() -> None:
    matrix = np.eye(3)
    tsm = TimeSeriesMatrix("A", matrix)

    assert not tsm.is_time_varying()
    assert np.array_equal(tsm[0], matrix)
    assert np.array_equal(tsm[5], matrix)


def test_time_series_matrix_3d_input_with_all_identical_matrices_collapses_to_2d() -> (
    None
):
    stacked = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    tsm = TimeSeriesMatrix("A", stacked)

    assert not tsm.is_time_varying()
    assert np.array_equal(tsm[0], np.eye(2))
    assert np.array_equal(tsm[3], np.eye(2))


def test_time_series_matrix_3d_input_detects_periodic_pattern() -> None:
    m0, m1 = np.eye(2), 2 * np.eye(2)
    stacked = np.stack([m0, m1, m0, m1, m0, m1])
    tsm = TimeSeriesMatrix("A", stacked)

    assert tsm.is_time_varying()
    for t in range(6):
        expected = m0 if t % 2 == 0 else m1
        assert np.array_equal(tsm[t], expected)
    # Indexing past the stored horizon wraps around via modulo.
    assert np.array_equal(tsm[6], m0)


def test_time_series_matrix_run_length_compresses_piecewise_constant_sequence() -> None:
    """A piecewise-constant schedule [M0,M0,M0,M0,M1,M1,M1] is stored as two
    runs (not seven entries) while indexing every step correctly."""
    m0, m1 = np.eye(2), 2 * np.eye(2)
    stacked = np.stack([m0, m0, m0, m0, m1, m1, m1])
    tsm = TimeSeriesMatrix("A", stacked)

    assert tsm.is_time_varying()
    assert tsm.array.shape[0] == 2  # only the 2 unique matrices are stored
    for t in range(4):
        assert np.array_equal(tsm[t], m0)
    for t in range(4, 7):
        assert np.array_equal(tsm[t], m1)


def test_time_series_matrix_run_length_wraps_around_past_sequence_length() -> None:
    """Indexing beyond the defined sequence length replays the whole
    run-length schedule via modulo arithmetic (mandated wraparound)."""
    m0, m1 = np.eye(2), 2 * np.eye(2)
    tsm = TimeSeriesMatrix("A", np.stack([m0, m0, m0, m0, m1, m1, m1]))  # length 7

    assert np.array_equal(tsm[7], m0)  # 7 % 7 == 0
    assert np.array_equal(tsm[8], m0)  # 8 % 7 == 1
    assert np.array_equal(tsm[11], m1)  # 11 % 7 == 4
    assert np.array_equal(tsm[13], m1)  # 13 % 7 == 6


def test_time_series_matrix_fully_time_varying_degrades_to_per_step() -> None:
    """A sequence with no period and no runs stores one length-1 run per
    step and still wraps around correctly."""
    m0, m1, m2 = np.eye(2), 2 * np.eye(2), 3 * np.eye(2)
    tsm = TimeSeriesMatrix("A", np.stack([m0, m1, m2, m1]))

    for t, expected in enumerate([m0, m1, m2, m1]):
        assert np.array_equal(tsm[t], expected)
    assert np.array_equal(tsm[4], m0)  # 4 % 4 == 0 (wraparound)


def test_time_series_matrix_rejects_invalid_ndim() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        TimeSeriesMatrix("A", np.zeros((2, 2, 2, 2)))


def test_linear_system_rejects_non_square_a() -> None:
    with pytest.raises(ValueError, match="A_t must be square"):
        LinearSystem(np.ones((2, 3)), np.eye(3), np.eye(3))


def test_linear_system_rejects_b_state_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="B_t"):
        LinearSystem(np.eye(2), np.ones((3, 1)), np.eye(2))


def test_linear_system_rejects_c_state_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="C_t"):
        LinearSystem(np.eye(2), np.eye(2), np.ones((2, 3)))


def test_linear_system_signature_distinguishes_temporal_pattern() -> None:
    """Two systems with the SAME unique matrices arranged in DIFFERENT
    temporal patterns must hash differently (regression lock: hashing only
    the unique-matrix set made them collide)."""
    m0, m1, B = np.eye(2), 2 * np.eye(2), np.eye(2)
    alternating = LinearSystem(np.stack([m0, m1, m0, m1]), B, np.eye(2))
    blocked = LinearSystem(np.stack([m0, m0, m1, m1]), B, np.eye(2))

    assert alternating.get_signature()["A_hash"] != blocked.get_signature()["A_hash"]


def test_linear_system_signature_is_deterministic_for_identical_pattern() -> None:
    m0, m1, B = np.eye(2), 2 * np.eye(2), np.eye(2)
    a = LinearSystem(np.stack([m0, m1, m0, m1]), B, np.eye(2))
    b = LinearSystem(np.stack([m0, m1, m0, m1]), B, np.eye(2))

    assert a.get_signature()["A_hash"] == b.get_signature()["A_hash"]


def test_linear_system_time_invariant_signature_unchanged_by_schedule_hash() -> None:
    """A time-invariant matrix's content hash equals a plain hash of its
    single matrix -- the schedule-aware hashing must not perturb the
    overwhelmingly common time-invariant case."""
    from mbl.core.utils.signing import hash_array

    system = LinearSystem(np.eye(2), np.eye(2), np.eye(2))
    assert system.A_t.content_hash() == hash_array(system.A_t.array)


def _rollout(system: LinearSystem, batch: int, horizon: int):
    n, m, p = (
        system.dimensions.state_dim,
        system.dimensions.control_dim,
        system.dimensions.observation_dim,
    )
    initial_state = np.zeros((batch, n))
    process_noises = np.zeros((batch, horizon, n))
    measurement_noises = np.zeros((batch, horizon, p))
    controls = np.ones((batch, horizon, m))
    return system.run_with_control(
        controls, initial_state, process_noises, measurement_noises
    )


def test_linear_system_fully_observable_rollout_matches_manual_recursion() -> None:
    A, B = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.0], [1.0]])
    system = LinearSystem.fully_observable(A, B)

    horizon, batch = 4, 1
    states, observations, controls = _rollout(system, batch, horizon)

    expected = np.zeros((batch, 2))
    manual_state_list = [expected]
    for _ in range(horizon):
        expected = expected @ A.T + np.ones((batch, 1)) @ B.T
        manual_state_list.append(expected)
    manual_states = np.stack(manual_state_list, axis=1)

    assert np.allclose(states, manual_states)
    assert np.allclose(observations, states[:, :-1, :])


def test_linear_system_from_matrices_and_indices_builds_time_varying_system() -> None:
    A_t_with_indices = (np.array([np.eye(2)]), np.array([0, 0, 0]))
    B_t_with_indices = (
        np.array([[[1.0], [0.0]], [[2.0], [0.0]]]),
        np.array([0, 0, 1]),
    )

    system = LinearSystem.from_matrices_and_indices(A_t_with_indices, B_t_with_indices)

    assert not system.A_t.is_time_varying()
    assert system.B_t.is_time_varying()
    assert np.array_equal(system.B_t[0], np.array([[1.0], [0.0]]))
    assert np.array_equal(system.B_t[2], np.array([[2.0], [0.0]]))


def test_linear_system_get_signature_reports_type_dims_and_hashes() -> None:
    A, B = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.0], [1.0]])
    system = LinearSystem.fully_observable(A, B)

    signature = system.get_signature()

    assert signature["type"] == "LinearSystem"
    assert signature["state_dim"] == 2
    assert signature["control_dim"] == 1
    assert signature["is_time_invariant"] is True
    assert signature["A_hash"].startswith("sha256:")
    assert signature["B_hash"].startswith("sha256:")
    assert signature["C_hash"].startswith("sha256:")


def test_linear_system_get_signature_hash_changes_with_matrix_content() -> None:
    A1, B = np.eye(2), np.array([[1.0], [0.0]])
    A2 = A1 * 2.0
    sig1 = LinearSystem.fully_observable(A1, B).get_signature()
    sig2 = LinearSystem.fully_observable(A2, B).get_signature()
    assert sig1["A_hash"] != sig2["A_hash"]
    assert sig1["B_hash"] == sig2["B_hash"]


def test_linear_system_get_signature_is_stable_for_identical_content() -> None:
    A, B = np.eye(2), np.array([[1.0], [0.0]])
    sig1 = LinearSystem.fully_observable(A.copy(), B.copy()).get_signature()
    sig2 = LinearSystem.fully_observable(A.copy(), B.copy()).get_signature()
    assert sig1 == sig2
