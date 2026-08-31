import numpy as np
import pytest
import torch

from mbl.core.utils.predicates import is_positive_semi_definite
from mbl.models.unfolded.parameters import (
    MatrixModulationParameter,
    MatrixModulationParameterConfig,
    PerIterationRiccatiMatrixParameter,
    PerIterationRiccatiMatrixParameterConfig,
    RiccatiMatrixParameter,
    RiccatiMatrixParameterConfig,
    StepSizeParameter,
    StepSizeParameterConfig,
)

DTYPE = torch.float64


def test_step_size_parameter_get_signature_reports_init_config_not_live_value() -> None:
    param = StepSizeParameter(
        StepSizeParameterConfig(
            name="step_size",
            num_iterations=3,
            action_dim=2,
            alpha_init=0.2,
            alpha_max=1.0,
            dtype=DTYPE,
        )
    )
    signature = param.get_signature()

    assert signature["type"] == "StepSizeParameter"
    assert signature["name"] == "step_size"
    assert signature["num_iterations"] == 3
    assert signature["action_dim"] == 2
    assert signature["alpha_init"] == 0.2
    assert signature["alpha_max"] == 1.0
    # The live, trainable value must never appear in the signature.
    assert "rho" not in signature


def test_step_size_parameter_get_signature_is_unaffected_by_training() -> None:
    """The defining property of 'sign the config, not the live value': the
    signature must be identical before and after the raw parameter changes."""
    config = StepSizeParameterConfig(
        name="s",
        num_iterations=2,
        action_dim=1,
        alpha_init=0.1,
        alpha_max=1.0,
        dtype=DTYPE,
    )
    param = StepSizeParameter(config)
    before = param.get_signature()

    with torch.no_grad():
        param.get_raw().add_(5.0)  # simulate an optimizer step

    after = param.get_signature()
    assert before == after


def test_riccati_matrix_parameter_get_signature_hashes_array_init_field() -> None:
    L_init = np.array([[2.0, 0.0], [0.5, 1.0]])
    param = RiccatiMatrixParameter(
        RiccatiMatrixParameterConfig(
            name="riccati_matrix",
            num_iterations=1,
            state_dim=2,
            L_init=L_init,
            dtype=DTYPE,
        )
    )
    signature = param.get_signature()

    assert signature["type"] == "RiccatiMatrixParameter"
    assert signature["state_dim"] == 2
    assert signature["L_init"].startswith("sha256:")  # hashed, not embedded raw


def test_riccati_matrix_parameter_get_signature_handles_none_init() -> None:
    param = RiccatiMatrixParameter(
        RiccatiMatrixParameterConfig(
            name="riccati_matrix", num_iterations=1, state_dim=2
        )
    )
    assert param.get_signature()["L_init"] is None


def test_two_parameters_with_identical_config_have_identical_signatures() -> None:
    def build():
        return StepSizeParameter(
            StepSizeParameterConfig(
                name="s",
                num_iterations=2,
                action_dim=1,
                alpha_init=0.1,
                alpha_max=1.0,
                dtype=DTYPE,
            )
        )

    assert build().get_signature() == build().get_signature()


def _random_psd_stack(num_iterations: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((num_iterations, n, n))
    for j in range(num_iterations):
        A = rng.standard_normal((n, n))
        out[j] = A @ A.T + np.eye(n)  # strictly PD -> PSD, well away from singular
    return out


class TestPerIterationRiccatiMatrixParameter:
    """R2 (NB05 plan Sec 14.1-14.10): J fully independent learned matrices."""

    def test_get_signature_hashes_array_init_field(self) -> None:
        L_init = np.tile(np.eye(2), (3, 1, 1))
        param = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix",
                num_iterations=3,
                state_dim=2,
                L_init=L_init,
                dtype=DTYPE,
            )
        )
        signature = param.get_signature()
        assert signature["type"] == "PerIterationRiccatiMatrixParameter"
        assert signature["state_dim"] == 2
        assert signature["L_init"].startswith("sha256:")

    def test_default_init_is_identity_at_every_iteration(self) -> None:
        param = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=4, state_dim=3, dtype=DTYPE
            )
        )
        P = param.get_numpy()
        assert P.shape == (4, 3, 3)
        for j in range(4):
            np.testing.assert_allclose(P[j], np.eye(3), atol=1e-12)

    def test_get_is_psd_at_every_iteration_from_a_random_raw_factor(self) -> None:
        param = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=5, state_dim=3, dtype=DTYPE
            )
        )
        with torch.no_grad():
            param.get_raw().copy_(torch.randn(5, 3, 3, dtype=DTYPE))
        P = param.get_numpy()
        for j in range(5):
            assert is_positive_semi_definite(P[j])

    def test_load_round_trips_a_per_iteration_psd_stack(self) -> None:
        param = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=3, state_dim=4, dtype=DTYPE
            )
        )
        target = _random_psd_stack(3, 4, seed=0)
        param.load(target)
        np.testing.assert_allclose(param.get_numpy(), target, atol=1e-10)

    def test_load_rejects_a_non_psd_slice(self) -> None:
        param = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=2, state_dim=2, dtype=DTYPE
            )
        )
        not_psd = np.stack([np.eye(2), np.array([[1.0, 2.0], [2.0, 1.0]])])
        with pytest.raises(ValueError, match="iteration 1"):
            param.load(not_psd)


class TestMatrixModulationParameter:
    """R1.5 (NB05 plan Sec 14.9): one learned positive scalar c_j per
    unfolding iteration modulating a shared learned matrix."""

    def test_get_signature_reports_init_config(self) -> None:
        param = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=4,
                modulation_init=1.0,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        )
        signature = param.get_signature()
        assert signature["type"] == "MatrixModulationParameter"
        assert signature["num_iterations"] == 4
        assert signature["modulation_init"] == 1.0
        assert signature["modulation_max"] == 5.0
        assert "rho" not in signature

    def test_default_init_reproduces_modulation_init_at_every_row(self) -> None:
        param = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=6,
                modulation_init=1.0,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        )
        c = param.get_numpy()
        assert c.shape == (6,)
        np.testing.assert_allclose(c, 1.0, atol=1e-9)

    def test_get_stays_within_the_open_bound(self) -> None:
        param = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=4,
                modulation_init=2.5,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        )
        with torch.no_grad():
            param.get_raw().copy_(torch.tensor([10.0, -10.0, 0.0, 3.0], dtype=DTYPE))
        c = param.get_numpy()
        assert np.all(c > 0.0)
        assert np.all(c < 5.0)

    def test_load_round_trips_per_iteration_scales(self) -> None:
        param = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=5,
                modulation_init=1.0,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        )
        target = np.array([0.5, 1.0, 2.0, 3.5, 4.0])
        param.load(target)
        np.testing.assert_allclose(param.get_numpy(), target, atol=1e-8)
