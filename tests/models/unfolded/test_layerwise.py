"""NB03 acceptance tests for the layer-wise freeze plan (blueprint v2 §3.4):
`ParameterActivation` compiles into a `LayerFreeze` whose row masks and
whole-tensor `requires_grad` plan are checked against the LIVE parameter
shapes, not a bare integer -- the blueprint's schedule<->depth validation."""

import pytest
import torch

from mbl.models.unfolded.layerwise import (
    LayerFreezeBuilder,
    ParameterActivation,
)
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


def _step_size(num_iterations: int = 4, action_dim: int = 2) -> StepSizeParameter:
    return StepSizeParameter(
        StepSizeParameterConfig(
            name="step_size",
            num_iterations=num_iterations,
            action_dim=action_dim,
            alpha_init=0.1,
            alpha_max=1.0,
            dtype=DTYPE,
        )
    )


def _riccati_matrix(state_dim: int = 3) -> RiccatiMatrixParameter:
    return RiccatiMatrixParameter(
        RiccatiMatrixParameterConfig(
            name="riccati_matrix", num_iterations=1, state_dim=state_dim, dtype=DTYPE
        )
    )


def _per_iteration_matrix(
    num_iterations: int = 4, state_dim: int = 3
) -> PerIterationRiccatiMatrixParameter:
    return PerIterationRiccatiMatrixParameter(
        PerIterationRiccatiMatrixParameterConfig(
            name="riccati_matrix",
            num_iterations=num_iterations,
            state_dim=state_dim,
            dtype=DTYPE,
        )
    )


def _matrix_modulation(num_iterations: int = 4) -> MatrixModulationParameter:
    return MatrixModulationParameter(
        MatrixModulationParameterConfig(
            name="matrix_modulation",
            num_iterations=num_iterations,
            modulation_init=1.0,
            modulation_max=5.0,
            dtype=DTYPE,
        )
    )


class TestParameterActivationSignature:
    def test_sorts_rows_and_reports_all_literally(self) -> None:
        activation = ParameterActivation(
            step_size_rows=frozenset({2, 0, 1}), train_matrix=True
        )
        assert activation.get_signature() == {
            "type": "ParameterActivation",
            "step_size_rows": [0, 1, 2],
            "train_matrix": True,
        }
        all_rows = ParameterActivation(step_size_rows="all", train_matrix=False)
        assert all_rows.get_signature()["step_size_rows"] == "all"


class TestLayerFreezeBuilderStepSize:
    def test_single_row_activation_masks_only_that_row(self) -> None:
        step_size = _step_size(num_iterations=4, action_dim=2)
        activation = ParameterActivation(
            step_size_rows=frozenset({2}), train_matrix=False
        )
        freeze = LayerFreezeBuilder().build(activation, {"step_size": step_size})

        mask = freeze.grad_masks["step_size"]
        assert mask.shape == step_size.get_raw().shape
        assert mask.dtype == torch.bool
        expected = torch.zeros(4, 2, dtype=torch.bool)
        expected[2] = True
        assert torch.equal(mask, expected)
        # The whole tensor stays requires_grad-eligible; freezing is mask-only.
        assert freeze.trainable["step_size"] is True

    def test_cumulative_rows_mask_every_named_row(self) -> None:
        step_size = _step_size(num_iterations=4, action_dim=2)
        activation = ParameterActivation(
            step_size_rows=frozenset({0, 1, 2}), train_matrix=False
        )
        freeze = LayerFreezeBuilder().build(activation, {"step_size": step_size})
        mask = freeze.grad_masks["step_size"]
        assert mask[:3].all()
        assert not mask[3].any()

    def test_all_rows_produces_an_all_true_mask(self) -> None:
        step_size = _step_size(num_iterations=4, action_dim=2)
        activation = ParameterActivation(step_size_rows="all", train_matrix=False)
        freeze = LayerFreezeBuilder().build(activation, {"step_size": step_size})
        assert freeze.grad_masks["step_size"].all()

    def test_out_of_bounds_row_raises_against_the_live_parameter_shape(self) -> None:
        """The reframed A-7: bounds are checked against the PARAMETER's own
        actual row count (the ground truth), not a caller-supplied integer
        with no way to be verified against reality."""
        step_size = _step_size(num_iterations=4, action_dim=2)
        activation = ParameterActivation(
            step_size_rows=frozenset({5}), train_matrix=False
        )
        with pytest.raises(ValueError, match="outside its 4 actual row"):
            LayerFreezeBuilder().build(activation, {"step_size": step_size})

    def test_negative_row_index_raises(self) -> None:
        step_size = _step_size(num_iterations=4, action_dim=2)
        activation = ParameterActivation(
            step_size_rows=frozenset({-1}), train_matrix=False
        )
        with pytest.raises(ValueError, match="outside its 4 actual row"):
            LayerFreezeBuilder().build(activation, {"step_size": step_size})


class TestLayerFreezeBuilderMatrix:
    def test_train_matrix_true_marks_the_whole_tensor_trainable(self) -> None:
        matrix = _riccati_matrix()
        activation = ParameterActivation(step_size_rows="all", train_matrix=True)
        freeze = LayerFreezeBuilder().build(activation, {"riccati_matrix": matrix})
        assert freeze.trainable["riccati_matrix"] is True
        assert (
            "riccati_matrix" not in freeze.grad_masks
        )  # no partial-row masking exists

    def test_train_matrix_false_marks_the_whole_tensor_frozen(self) -> None:
        matrix = _riccati_matrix()
        activation = ParameterActivation(step_size_rows="all", train_matrix=False)
        freeze = LayerFreezeBuilder().build(activation, {"riccati_matrix": matrix})
        assert freeze.trainable["riccati_matrix"] is False


class TestLayerFreezeBuilderCombined:
    def test_builds_a_joint_plan_for_both_parameter_kinds(self) -> None:
        step_size = _step_size(num_iterations=3, action_dim=1)
        matrix = _riccati_matrix()
        activation = ParameterActivation(
            step_size_rows=frozenset({1}), train_matrix=True
        )
        freeze = LayerFreezeBuilder().build(
            activation, {"step_size": step_size, "riccati_matrix": matrix}
        )
        assert set(freeze.trainable) == {"step_size", "riccati_matrix"}
        assert freeze.trainable["riccati_matrix"] is True
        assert freeze.grad_masks["step_size"][1].all()

    def test_unknown_parameter_type_is_a_typed_refusal(self) -> None:
        class _Bogus:
            pass

        with pytest.raises(TypeError, match="no freeze rule"):
            LayerFreezeBuilder().build(
                ParameterActivation(step_size_rows="all", train_matrix=False),
                {"bogus": _Bogus()},  # type: ignore[dict-item]
            )


class TestLayerFreezeBuilderPerIterationMatrix:
    """R2 (NB05 plan Sec 14.7 item 4/5): a per-iteration matrix has the
    SAME row structure as step_size, so it reuses the identical row-mask
    mechanism (no new `ParameterActivation` field)."""

    def test_single_row_activation_masks_only_that_rows_whole_slice(self) -> None:
        matrix = _per_iteration_matrix(num_iterations=4, state_dim=3)
        activation = ParameterActivation(
            step_size_rows=frozenset({2}), train_matrix=False
        )
        freeze = LayerFreezeBuilder().build(activation, {"riccati_matrix": matrix})

        mask = freeze.grad_masks["riccati_matrix"]
        assert mask.shape == matrix.get_raw().shape == (4, 3, 3)
        expected = torch.zeros(4, 3, 3, dtype=torch.bool)
        expected[2] = True
        assert torch.equal(mask, expected)
        assert freeze.trainable["riccati_matrix"] is True

    def test_all_rows_produces_an_all_true_mask(self) -> None:
        matrix = _per_iteration_matrix()
        activation = ParameterActivation(step_size_rows="all", train_matrix=False)
        freeze = LayerFreezeBuilder().build(activation, {"riccati_matrix": matrix})
        assert freeze.grad_masks["riccati_matrix"].all()

    def test_out_of_bounds_row_raises_against_the_live_parameter_shape(self) -> None:
        matrix = _per_iteration_matrix(num_iterations=4)
        activation = ParameterActivation(
            step_size_rows=frozenset({5}), train_matrix=False
        )
        with pytest.raises(ValueError, match="outside its 4 actual row"):
            LayerFreezeBuilder().build(activation, {"riccati_matrix": matrix})


class TestLayerFreezeBuilderMatrixModulation:
    """R1.5 (NB05 plan Sec 14.9): the modulation scalar c_j shares
    step_size's exact row semantics."""

    def test_single_row_activation_masks_only_that_row(self) -> None:
        modulation = _matrix_modulation(num_iterations=4)
        activation = ParameterActivation(
            step_size_rows=frozenset({1}), train_matrix=False
        )
        freeze = LayerFreezeBuilder().build(
            activation, {"matrix_modulation": modulation}
        )

        mask = freeze.grad_masks["matrix_modulation"]
        assert mask.shape == modulation.get_raw().shape == (4,)
        expected = torch.zeros(4, dtype=torch.bool)
        expected[1] = True
        assert torch.equal(mask, expected)
        assert freeze.trainable["matrix_modulation"] is True

    def test_all_rows_produces_an_all_true_mask(self) -> None:
        modulation = _matrix_modulation()
        activation = ParameterActivation(step_size_rows="all", train_matrix=False)
        freeze = LayerFreezeBuilder().build(
            activation, {"matrix_modulation": modulation}
        )
        assert freeze.grad_masks["matrix_modulation"].all()


class TestLayerFreezeBuilderScalarModulatedJointPlan:
    """R1.5's full parameter set: step_size and matrix_modulation share the
    SAME active row per phase (both row-masked), while the shared base
    matrix keeps the existing whole-tensor `train_matrix` gate."""

    def test_step_size_and_modulation_share_the_same_active_row(self) -> None:
        step_size = _step_size(num_iterations=3, action_dim=1)
        base_matrix = _riccati_matrix()
        modulation = _matrix_modulation(num_iterations=3)
        activation = ParameterActivation(
            step_size_rows=frozenset({1}), train_matrix=True
        )
        freeze = LayerFreezeBuilder().build(
            activation,
            {
                "step_size": step_size,
                "riccati_matrix": base_matrix,
                "matrix_modulation": modulation,
            },
        )
        assert freeze.trainable["riccati_matrix"] is True
        assert torch.equal(
            freeze.grad_masks["step_size"][:, 0], freeze.grad_masks["matrix_modulation"]
        )
        assert freeze.grad_masks["matrix_modulation"][1]
        assert not freeze.grad_masks["matrix_modulation"][0]
        assert not freeze.grad_masks["matrix_modulation"][2]
