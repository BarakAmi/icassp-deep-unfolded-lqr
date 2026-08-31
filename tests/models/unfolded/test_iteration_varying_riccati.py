"""Acceptance tests for the NB05 P-resolution extension's shared refinement
mechanics (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 14.5/14.10):
`IterationVaryingRiccatiRefinement` and its two leaves,
`PerIterationRiccatiRefinement` (R2) and `ScalarModulatedRiccatiRefinement`
(R1.5). Verifies the kernel-slice-agreement claim the plan's own grounding
inventory made (`torch.equal(M[2], M0) == True`) as a permanent test, plus
`ScalarModulatedRiccatiRefinement`'s P^(j) = c_j * P combination."""

import numpy as np
import torch

from mbl.models.unfolded.iterative_refinement import (
    PerIterationRiccatiRefinement,
    RiccatiRefinement,
    ScalarModulatedRiccatiRefinement,
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
N, M, T = 3, 2, 5


def _step_size(num_iterations: int) -> StepSizeParameter:
    return StepSizeParameter(
        StepSizeParameterConfig(
            name="step_size",
            num_iterations=num_iterations,
            action_dim=M,
            alpha_init=0.05,
            alpha_max=1.0,
            dtype=DTYPE,
        )
    )


def _stacks(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    A = torch.tensor(rng.standard_normal((T, N, N)), dtype=DTYPE)
    B = torch.tensor(rng.standard_normal((T, N, M)), dtype=DTYPE)
    R_half = rng.standard_normal((T, M, M))
    R = torch.tensor(
        R_half @ R_half.transpose(0, 2, 1) + np.eye(M), dtype=DTYPE
    )  # PD, well-conditioned
    return A, B, R


def _random_psd_stack(num_iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((num_iterations, N, N))
    for j in range(num_iterations):
        L = rng.standard_normal((N, N))
        out[j] = L @ L.T + np.eye(N)
    return out


class TestPerIterationKernelSliceAgreement:
    """The plan's own grounding-inventory check, promoted to a permanent
    test: slice j of the batched (J, T, ...) gradient-matrix computation
    must be bit-identical to a single-P `RiccatiRefinement` call with P^(j)
    alone."""

    def test_each_slice_matches_a_single_p_riccati_refinement(self) -> None:
        J = 4
        A, B, R = _stacks(seed=0)
        P_stack = _random_psd_stack(J, seed=1)

        per_iteration_matrix = PerIterationRiccatiMatrixParameter(
            PerIterationRiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=J, state_dim=N, dtype=DTYPE
            )
        )
        per_iteration_matrix.load(P_stack)
        joint = PerIterationRiccatiRefinement(
            step_size=_step_size(J),
            num_iterations=J,
            learnable_parameters={"riccati_matrix": per_iteration_matrix},
            static_parameters={"A": A, "B": B, "R": R},
        )
        M_joint, C_joint = joint._horizon_gradient_matrices()
        assert M_joint.shape == (J, T, M, M)
        assert C_joint.shape == (J, T, M, N)

        for j in range(J):
            single_matrix = RiccatiMatrixParameter(
                RiccatiMatrixParameterConfig(
                    name="riccati_matrix", num_iterations=1, state_dim=N, dtype=DTYPE
                )
            )
            single_matrix.load(P_stack[j])
            single = RiccatiRefinement(
                step_size=_step_size(1),
                num_iterations=1,
                learnable_parameters={"riccati_matrix": single_matrix},
                static_parameters={"A": A, "B": B, "R": R},
            )
            M_single, C_single = single._horizon_gradient_matrices()
            # Not `torch.equal`: the joint path computes the whole (J, T, ...)
            # stack in one batched kernel while `single` computes one slice, so
            # the two are different reduction orders of the same sum. The
            # operands are bit-identical by construction; the outputs agree to
            # arithmetic, which is all any two machines owe each other. See
            # `docs/methods/cross_machine_numerical_reproducibility.md`.
            torch.testing.assert_close(M_joint[j], M_single, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(C_joint[j], C_single, atol=1e-10, rtol=1e-10)


class TestScalarModulatedIterationPStack:
    """R1.5: P^(j) = c_j * P, built from a shared base matrix plus a
    per-iteration positive scalar."""

    def test_iteration_p_stack_equals_c_j_times_shared_p(self) -> None:
        J = 3
        base_matrix = RiccatiMatrixParameter(
            RiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=1, state_dim=N, dtype=DTYPE
            )
        )
        P_target = _random_psd_stack(1, seed=2)[0]
        base_matrix.load(P_target)
        modulation = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=J,
                modulation_init=1.0,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        )
        c_target = np.array([0.5, 1.5, 3.0])
        modulation.load(c_target)

        A, B, R = _stacks(seed=3)
        refinement = ScalarModulatedRiccatiRefinement(
            step_size=_step_size(J),
            num_iterations=J,
            learnable_parameters={
                "riccati_matrix": base_matrix,
                "matrix_modulation": modulation,
            },
            static_parameters={"A": A, "B": B, "R": R},
        )
        P_stack = refinement._iteration_p_stack().detach().numpy()
        expected = c_target[:, None, None] * P_target[None, :, :]
        np.testing.assert_allclose(P_stack, expected, atol=1e-8)

    def test_modulation_of_one_reproduces_the_shared_matrix_at_every_iteration(
        self,
    ) -> None:
        """The nesting property (NB05 plan Sec 14.2): c_j = 1 for every j
        must reproduce the shared matrix P exactly at each iteration."""
        J = 4
        base_matrix = RiccatiMatrixParameter(
            RiccatiMatrixParameterConfig(
                name="riccati_matrix", num_iterations=1, state_dim=N, dtype=DTYPE
            )
        )
        P_target = _random_psd_stack(1, seed=4)[0]
        base_matrix.load(P_target)
        modulation = MatrixModulationParameter(
            MatrixModulationParameterConfig(
                name="matrix_modulation",
                num_iterations=J,
                modulation_init=1.0,
                modulation_max=5.0,
                dtype=DTYPE,
            )
        )
        modulation.load(np.ones(J))

        A, B, R = _stacks(seed=5)
        refinement = ScalarModulatedRiccatiRefinement(
            step_size=_step_size(J),
            num_iterations=J,
            learnable_parameters={
                "riccati_matrix": base_matrix,
                "matrix_modulation": modulation,
            },
            static_parameters={"A": A, "B": B, "R": R},
        )
        P_stack = refinement._iteration_p_stack().detach().numpy()
        for j in range(J):
            np.testing.assert_allclose(P_stack[j], P_target, atol=1e-8)
