"""Phase 1 acceptance tests for `applications.uncertainty.perturbations`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 5.1/11): the degeneracy law
(level 0 / kind NONE reproduces the input bit-for-bit), the rotation
construction's exact canonical angles, the additive perturbation's spectral-
radius guard, and `PerturbationDistribution`'s resample-vs-fixed contract.
"""

import numpy as np
import pytest

from mbl.applications.uncertainty.perturbations import (
    PerturbationDistribution,
    PerturbationKind,
    PlantPerturbation,
    rotation_matrix,
)


def _canonical_angles(R: np.ndarray) -> list[float]:
    """Independent (non-reused) reference computation of a rotation's
    canonical angles, for cross-checking `rotation_matrix` under test."""
    eigenvalues = np.linalg.eigvals(R)
    return sorted(
        {round(abs(np.angle(e)), 10) for e in eigenvalues if abs(np.imag(e)) > 1e-9}
    )


class TestRotationCanonicalAngles:
    @pytest.mark.parametrize("n", [3, 4, 5, 7])
    @pytest.mark.parametrize("degrees", [15.0, 30.0, 45.0, 90.0])
    def test_every_canonical_angle_equals_theta(self, n: int, degrees: float) -> None:
        R = rotation_matrix(n, degrees, seed=7)
        angles = _canonical_angles(R)
        expected = round(np.deg2rad(degrees), 10)
        assert angles, "a nonzero rotation must have at least one complex pair"
        for angle in angles:
            assert angle == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_orthogonal(self, n: int) -> None:
        R = rotation_matrix(n, 37.0, seed=3)
        np.testing.assert_allclose(R @ R.T, np.eye(n), atol=1e-10)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_zero_degrees_is_exact_identity(self, n: int) -> None:
        R = rotation_matrix(n, 0.0, seed=11)
        np.testing.assert_array_equal(R, np.eye(n))


class TestDegeneracyLaw:
    """level == 0 (any kind) or kind == NONE reproduces the input bit-for-bit."""

    @pytest.fixture
    def system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        n, m = 4, 2
        A = rng.normal(size=(n, n))
        A /= np.max(np.abs(np.linalg.eigvals(A)))
        B = rng.normal(size=(n, m))
        W = 0.25 * np.eye(n)
        return A, B, W

    def test_kind_none(self, system: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        A, B, W = system
        perturbation = PlantPerturbation(kind=PerturbationKind.NONE, level=0.3)
        A2, B2, W2, realized = perturbation.draw(A, B, W, np.random.default_rng(1))
        np.testing.assert_array_equal(A2, A)
        np.testing.assert_array_equal(B2, B)
        np.testing.assert_array_equal(W2, W)
        assert realized["angle_degrees"] == 0.0

    @pytest.mark.parametrize(
        "kind", [PerturbationKind.ADDITIVE, PerturbationKind.ROTATION]
    )
    def test_zero_level(
        self,
        system: tuple[np.ndarray, np.ndarray, np.ndarray],
        kind: PerturbationKind,
    ) -> None:
        A, B, W = system
        perturbation = PlantPerturbation(kind=kind, level=0.0, seed=5)
        A2, B2, W2, _ = perturbation.draw(A, B, W, np.random.default_rng(2))
        np.testing.assert_array_equal(A2, A)
        np.testing.assert_array_equal(B2, B)
        np.testing.assert_array_equal(W2, W)


class TestAdditivePerturbation:
    @pytest.fixture
    def system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        n, m = 4, 2
        A = rng.normal(size=(n, n))
        A /= np.max(np.abs(np.linalg.eigvals(A)))
        B = rng.normal(size=(n, m))
        W = 0.25 * np.eye(n)
        return A, B, W

    @pytest.mark.parametrize("level", [0.05, 0.1, 0.2, 0.4, 1.0, 5.0])
    def test_spectral_radius_never_exceeds_one(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray], level: float
    ) -> None:
        A, B, W = system
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ADDITIVE,
            level=level,
            renormalize_spectral_radius=True,
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            A_pert, _, _, realized = perturbation.draw(A, B, W, rng)
            rho = float(np.max(np.abs(np.linalg.eigvals(A_pert))))
            assert rho <= 1.0 + 1e-9
            assert realized["spectral_radius"] == pytest.approx(rho)

    def test_b_untouched_unless_requested(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        perturbation = PlantPerturbation(kind=PerturbationKind.ADDITIVE, level=0.2)
        _, B_pert, _, _ = perturbation.draw(A, B, W, np.random.default_rng(0))
        np.testing.assert_array_equal(B_pert, B)

    def test_perturb_b_changes_it(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ADDITIVE, level=0.2, perturb_B=True
        )
        _, B_pert, _, _ = perturbation.draw(A, B, W, np.random.default_rng(0))
        assert not np.array_equal(B_pert, B)


class TestRotationPerturbation:
    @pytest.fixture
    def system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        n, m = 4, 2
        A = rng.normal(size=(n, n))
        A /= np.max(np.abs(np.linalg.eigvals(A)))
        B = rng.normal(size=(n, m))
        W = 0.25 * np.eye(n)
        return A, B, W

    def test_spectral_radius_preserved(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        nominal_rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ROTATION, level=30.0, seed=1
        )
        _, _, _, realized = perturbation.draw(A, B, W, np.random.default_rng(0))
        assert realized["spectral_radius"] == pytest.approx(nominal_rho, abs=1e-9)

    def test_w_is_rotated(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ROTATION, level=45.0, seed=2
        )
        _, _, W_pert, _ = perturbation.draw(A, B, W, np.random.default_rng(0))
        # W = sigma^2 I is rotation-invariant for this fixture; use a non-isotropic W instead.
        W_aniso = np.diag([0.1, 0.2, 0.3, 0.4])
        _, _, W_pert2, _ = perturbation.draw(A, B, W_aniso, np.random.default_rng(0))
        assert not np.allclose(W_pert2, W_aniso)
        assert np.trace(W_pert2) == pytest.approx(np.trace(W_aniso))

    def test_b_is_untouched_unless_requested(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """`perturb_B=False` must leave `B` alone — on the ROTATION branch too.

        The line read, literally, `B_pert = R @ B if self.perturb_B else R @ B`:
        both arms of a ternary were the same expression, so the flag decided
        nothing and `B` was co-rotated whatever the declaration said. The
        additive branch beside it honours the flag correctly, which is what
        made the defect invisible to a reader skimming the class.

        It is not cosmetic. With `B` co-rotated the transform is an exact
        orthogonal similarity — the same system in rotated coordinates — so
        the re-solved optimum does not move at all, and the campaign measured
        exactly that: **0.000e+00 in 11 of 15 (seed, angle) cells** and
        |1.5e-14| in the rest. The experiment a caller declaring
        `perturb_B=False` is asking for is the one where `B` stays put, and
        there the unconstrained optimum moves −4.5 % to +14.5 %.
        """
        A, B, W = system
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ROTATION, level=30.0, seed=1, perturb_B=False
        )
        _, B_pert, _, _ = perturbation.draw(A, B, W, np.random.default_rng(0))
        assert np.array_equal(B_pert, B)

    def test_requesting_it_co_rotates_b(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        # The other side of the same flag: a test only of the False case
        # would pass against an implementation that never rotated B at all.
        A, B, W = system
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ROTATION, level=30.0, seed=1, perturb_B=True
        )
        _, B_pert, _, _ = perturbation.draw(A, B, W, np.random.default_rng(0))
        assert not np.allclose(B_pert, B)
        # Co-ROTATED, not merely different: a rotation preserves every column
        # norm, which distinguishes it from any additive noise.
        assert np.linalg.norm(B_pert, axis=0) == pytest.approx(
            np.linalg.norm(B, axis=0)
        )

    def test_the_two_declarations_differ(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """The assertion the dead ternary could not satisfy, stated directly.

        Measured before the fix: both settings produced bit-identical `B`,
        with ||B' - B||_F = 1.066740e+00 either way.
        """
        A, B, W = system
        drawn = [
            PlantPerturbation(
                kind=PerturbationKind.ROTATION,
                level=30.0,
                seed=1,
                perturb_B=flag,
            ).draw(A, B, W, np.random.default_rng(0))[1]
            for flag in (False, True)
        ]
        assert not np.array_equal(drawn[0], drawn[1])


class TestPerturbationDistribution:
    @pytest.fixture
    def system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        n, m = 4, 2
        A = rng.normal(size=(n, n))
        A /= np.max(np.abs(np.linalg.eigvals(A)))
        B = rng.normal(size=(n, m))
        W = 0.25 * np.eye(n)
        return A, B, W

    def test_resample_per_epoch_draws_fresh(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        distribution = PerturbationDistribution(
            perturbation=PlantPerturbation(
                kind=PerturbationKind.ADDITIVE, level=0.2, seed=9
            ),
            resample_per_epoch=True,
        )
        stream = distribution.build_stream(A, B, W)
        A1, _, _, _ = stream()
        A2, _, _, _ = stream()
        assert not np.array_equal(A1, A2)

    def test_fixed_holds_the_same_draw(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        distribution = PerturbationDistribution(
            perturbation=PlantPerturbation(
                kind=PerturbationKind.ROTATION, level=30.0, seed=9
            ),
            resample_per_epoch=False,
        )
        stream = distribution.build_stream(A, B, W)
        A1, _, _, _ = stream()
        A2, _, _, _ = stream()
        np.testing.assert_array_equal(A1, A2)

    def test_reproducible_across_streams(
        self, system: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        A, B, W = system
        distribution = PerturbationDistribution(
            perturbation=PlantPerturbation(
                kind=PerturbationKind.ADDITIVE, level=0.2, seed=9
            ),
        )
        stream_a = distribution.build_stream(A, B, W)
        stream_b = distribution.build_stream(A, B, W)
        for _ in range(5):
            A_a, _, _, _ = stream_a()
            A_b, _, _, _ = stream_b()
            np.testing.assert_array_equal(A_a, A_b)


class TestSignatures:
    def test_perturbation_signature_covers_fields(self) -> None:
        perturbation = PlantPerturbation(
            kind=PerturbationKind.ADDITIVE, level=0.2, seed=3
        )
        signature = perturbation.get_signature()
        assert signature["kind"] == PerturbationKind.ADDITIVE
        assert signature["level"] == 0.2
        assert signature["seed"] == 3

    def test_distribution_signature_nests_perturbation(self) -> None:
        distribution = PerturbationDistribution(
            perturbation=PlantPerturbation(
                kind=PerturbationKind.ROTATION, level=30.0, seed=1
            ),
            resample_per_epoch=False,
        )
        signature = distribution.get_signature()
        assert signature["resample_per_epoch"] is False
        assert signature["perturbation"]["level"] == 30.0
