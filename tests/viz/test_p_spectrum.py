"""Phase 5 acceptance tests for `workbench.spectral_analysis`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 6.4/11): the isotropy index
is 0 for ``cI`` and non-trivial for an anisotropic matrix; principal angles
are 0 against itself; symmetrization is applied; condition number matches
the direct eigenvalue ratio.
"""

import numpy as np
import pytest

from mbl.workbench.spectral_analysis import compute_p_spectrum, compute_principal_angles


class TestIsotropyIndex:
    def test_scalar_multiple_of_identity_is_exactly_zero(self) -> None:
        P = 3.7 * np.eye(5)
        spectrum = compute_p_spectrum(P)
        assert spectrum.isotropy_index == pytest.approx(0.0, abs=1e-12)

    def test_rank_one_matrix_matches_the_closed_form_extreme(self) -> None:
        # A rank-1 PSD matrix (eigenvalues 1,0,0,0 in n=4) is the extreme
        # anisotropic case: isotropy_index = sqrt((n-1)/n) exactly, the
        # largest value this ratio can take for a PSD matrix at this n
        # (never exactly 1 -- that would require an unbounded eigenvalue
        # spread, which a single nonzero eigenvalue does not produce).
        v = np.array([1.0, 0.0, 0.0, 0.0])
        P = np.outer(v, v)  # rank-1: eigenvalues (1, 0, 0, 0)
        spectrum = compute_p_spectrum(P)
        n = 4
        assert spectrum.isotropy_index == pytest.approx(np.sqrt((n - 1) / n), rel=1e-6)

    def test_generic_anisotropic_matrix_is_strictly_between(self) -> None:
        P = np.diag([1.0, 2.0, 3.0, 4.0])
        spectrum = compute_p_spectrum(P)
        assert 0.0 < spectrum.isotropy_index < 1.0

    def test_zero_matrix_is_defined_as_zero(self) -> None:
        spectrum = compute_p_spectrum(np.zeros((3, 3)))
        assert spectrum.isotropy_index == 0.0


class TestEigenvaluesAndConditionNumber:
    def test_eigenvalues_sorted_descending(self) -> None:
        P = np.diag([1.0, 5.0, 3.0])
        spectrum = compute_p_spectrum(P)
        assert spectrum.eigenvalues == pytest.approx((5.0, 3.0, 1.0))

    def test_condition_number_matches_direct_ratio(self) -> None:
        P = np.diag([2.0, 8.0])
        spectrum = compute_p_spectrum(P)
        assert spectrum.condition_number == pytest.approx(4.0)
        assert spectrum.spectral_norm == pytest.approx(8.0)

    def test_singular_matrix_gives_infinite_condition_number(self) -> None:
        P = np.diag([2.0, 0.0])
        spectrum = compute_p_spectrum(P)
        assert spectrum.condition_number == float("inf")

    def test_symmetrization_is_applied(self) -> None:
        # An asymmetric input whose symmetric part is diag(1,2).
        P = np.array([[1.0, 10.0], [-10.0, 2.0]])
        spectrum = compute_p_spectrum(P)
        assert spectrum.eigenvalues == pytest.approx((2.0, 1.0))


class TestPrincipalAngles:
    def test_angle_against_itself_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        M = rng.normal(size=(5, 5))
        M = M + M.T
        angles = compute_principal_angles(M, M, k=1)
        assert angles[0] == pytest.approx(0.0, abs=1e-8)

    def test_orthogonal_dominant_directions_give_90_degrees(self) -> None:
        # Dominant eigenvector of A is e1; dominant eigenvector of B is e2.
        A = np.diag([10.0, 1.0, 1.0])
        B = np.diag([1.0, 10.0, 1.0])
        angles = compute_principal_angles(A, B, k=1)
        assert np.degrees(angles[0]) == pytest.approx(90.0, abs=1e-6)

    def test_compute_p_spectrum_reports_principal_angles_when_reference_given(
        self,
    ) -> None:
        A = np.diag([10.0, 1.0, 1.0])
        B = np.diag([1.0, 10.0, 1.0])
        spectrum = compute_p_spectrum(A, reference=B, k=1)
        assert spectrum.principal_angles_deg is not None
        assert spectrum.principal_angles_deg[0] == pytest.approx(90.0, abs=1e-6)

    def test_no_reference_gives_none(self) -> None:
        spectrum = compute_p_spectrum(np.eye(3))
        assert spectrum.principal_angles_deg is None
