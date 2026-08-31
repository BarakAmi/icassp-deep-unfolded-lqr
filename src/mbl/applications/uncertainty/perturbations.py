"""Plant perturbations for training/evaluation under model mismatch
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.4/5.1): additive noise on
``A`` (optionally ``B``) and an exact-angle rotation of the whole system,
both sharing the legacy's spectral-radius safety guard.

`PlantPerturbation` is the pure, deterministic-given-``seed`` draw; a
`PerturbationDistribution` wraps it with the choice of whether an epoch loop
should redraw it fresh every call (the additive domain-randomization regime)
or draw it once and hold it fixed (the rotation regime, whose "distribution"
is a single fixed matrix once its angle is chosen). Neither type touches
torch or the engine -- this module is pure NumPy numerics, imported by the
`DomainRandomizationCallback` (``engine/callbacks.py``), never the reverse.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
from scipy.linalg import expm, schur

#: Numerical floor below which a real-Schur 2x2 block's off-diagonal entry is
#: treated as a genuine zero eigenvalue (only possible for odd state
#: dimension), not a rotation block.
_SCHUR_BLOCK_TOL = 1e-9


class PerturbationKind(StrEnum):
    """Which family of plant perturbation `PlantPerturbation.draw` applies."""

    NONE = "none"
    ADDITIVE = "additive"
    ROTATION = "rotation"


def _canonical_skew_generator(n: int, seed: int) -> np.ndarray:
    """A fixed, seed-drawn skew-symmetric generator whose EVERY canonical
    (real-Schur) angle is normalized to exactly 1 radian, so that
    ``expm(theta * generator)`` has every canonical angle equal to `theta`
    exactly (up to the real Schur decomposition's own floating-point
    precision, ~1e-14) -- the construction NB06's rotation-OOD axis and
    NB07's rotation-mismatch axis both share (NB07 plan Sec 4.4).

    A random skew-symmetric matrix is normal, so its real Schur form is
    exactly block-diagonal: one ``[[0, -sigma_k], [sigma_k, 0]]`` block per
    complex-conjugate eigenvalue pair ``+-i*sigma_k``, plus one 1-entry zero
    block if `n` is odd. Rescaling each block to ``sigma_k = 1`` in place
    (never touching the orthogonal Schur basis `Q`) makes every angle exactly
    `theta` once the whole generator is scaled by `theta` and exponentiated.

    Args:
        n: The state dimension.
        seed: Seed of the underlying random draw.

    Returns:
        The normalized skew-symmetric generator, shape ``(n, n)``.
    """
    rng = np.random.default_rng(seed)
    draw = rng.normal(size=(n, n))
    skew = draw - draw.T
    schur_form, basis = schur(skew, output="real")

    normalized = np.zeros_like(schur_form)
    i = 0
    while i < n:
        is_block = i + 1 < n and abs(schur_form[i + 1, i]) > _SCHUR_BLOCK_TOL
        if is_block:
            sign = np.sign(schur_form[i + 1, i])
            normalized[i, i + 1] = -sign
            normalized[i + 1, i] = sign
            i += 2
        else:
            i += 1
    return cast(np.ndarray, basis @ normalized @ basis.T)


def rotation_matrix(n: int, degrees: float, seed: int) -> np.ndarray:
    """The exact-canonical-angle rotation `R(theta) = exp(theta * S)` (NB07
    plan Sec 4.4): every canonical angle of the returned matrix equals
    `degrees` (converted to radians) exactly, and `degrees == 0` returns the
    exact identity without calling `expm` at all (the degeneracy law).

    Args:
        n: The state dimension.
        degrees: The rotation angle, in degrees.
        seed: Seed of the fixed skew-symmetric generator `S`.

    Returns:
        The orthogonal rotation matrix, shape ``(n, n)``.
    """
    if degrees == 0.0:
        return np.eye(n)
    theta = np.deg2rad(degrees)
    generator = _canonical_skew_generator(n, seed)
    return cast(np.ndarray, expm(theta * generator))


@dataclass(frozen=True)
class PlantPerturbation:
    """One perturbation draw of `(A, B, W)`, pure and deterministic given its
    own fields plus the `rng`/seed passed to `draw`.

    Attributes:
        kind: Which family of perturbation to apply.
        level: The perturbation's scalar severity -- an epsilon (Frobenius-
            normalized additive scale) for `ADDITIVE`, or a degree angle for
            `ROTATION`. `level == 0` (any kind) or `kind == NONE` reproduces
            the input bit-for-bit (the degeneracy law).
        seed: For `ROTATION`, the fixed generator seed (`rotation_matrix`'s
            `seed`) -- the SAME rotation matrix at a given `level` every
            call, since there is nothing left to redraw once `level` is
            fixed. Unused by `ADDITIVE`, whose randomness comes from the
            caller-supplied `rng` in `draw` instead (so a domain-
            randomization loop can advance it every epoch).
        perturb_B: Whether `ADDITIVE`/`ROTATION` also perturbs `B` (co-
            rotated for `ROTATION`; independently drawn and Frobenius-
            normalized for `ADDITIVE`).
        renormalize_spectral_radius: For `ADDITIVE` only: rescale the
            perturbed `A` so its spectral radius never exceeds 1 (the
            legacy's stability guard, `model_robustness_analysis.ipynb`
            cell 23) -- without it, a large `level` produces an exponentially
            diverging rollout whose cost overflow is an artifact of
            instability, not of controller quality.
    """

    kind: PerturbationKind
    level: float
    seed: int = 0
    perturb_B: bool = False
    renormalize_spectral_radius: bool = True

    def draw(
        self, A: np.ndarray, B: np.ndarray, W: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        """Apply this perturbation to `(A, B, W)`.

        Args:
            A: State transition matrix, shape ``(n, n)``.
            B: Control input matrix, shape ``(n, m)``.
            W: Process noise covariance, shape ``(n, n)``.
            rng: The generator `ADDITIVE` draws its Frobenius-normalized
                noise from; ignored by `ROTATION`/`NONE` (both fully
                determined by `self.level`/`self.seed`).

        Returns:
            ``(A', B', W', realized)``, where `realized` carries
            ``"spectral_radius"`` (of `A'`) and ``"angle_degrees"`` (0.0
            unless `kind == ROTATION`) for audit/logging.

        Raises:
            ValueError: If `kind` is not a recognized `PerturbationKind`.
        """
        if self.kind == PerturbationKind.NONE or self.level == 0.0:
            return (
                A.copy(),
                B.copy(),
                W.copy(),
                {
                    "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
                    "angle_degrees": 0.0,
                },
            )
        if self.kind == PerturbationKind.ADDITIVE:
            return self._draw_additive(A, B, W, rng)
        if self.kind == PerturbationKind.ROTATION:
            return self._draw_rotation(A, B, W)
        raise ValueError(f"Unknown PerturbationKind {self.kind!r}.")

    def _draw_additive(
        self, A: np.ndarray, B: np.ndarray, W: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        """Frobenius-normalized additive noise on `A` (and optionally `B`),
        re-normalized so the perturbed `A`'s spectral radius never exceeds 1
        (see `renormalize_spectral_radius`)."""
        noise = rng.normal(size=A.shape)
        noise /= np.linalg.norm(noise, ord="fro")
        A_pert = A + self.level * noise
        if self.renormalize_spectral_radius:
            rho = float(np.max(np.abs(np.linalg.eigvals(A_pert))))
            if rho > 1.0:
                A_pert = A_pert / rho
        B_pert = B.copy()
        if self.perturb_B:
            b_noise = rng.normal(size=B.shape)
            b_noise /= np.linalg.norm(b_noise, ord="fro")
            B_pert = B + self.level * b_noise
        realized_rho = float(np.max(np.abs(np.linalg.eigvals(A_pert))))
        return (
            A_pert,
            B_pert,
            W.copy(),
            {
                "spectral_radius": realized_rho,
                "angle_degrees": 0.0,
            },
        )

    def _draw_rotation(
        self, A: np.ndarray, B: np.ndarray, W: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        """`A_theta = R A R^T`, `W_theta = R W R^T`, and `B_theta = R B` only
        when `perturb_B` asks for it; a rotation never changes the spectral
        radius (orthogonal similarity), so no renormalization is needed or
        applied.

        **Whether `B` co-rotates decides whether the experiment exists.** With
        it co-rotated the whole transform is an exact orthogonal similarity —
        the same system written in rotated coordinates — so the re-solved
        optimum does not move: measured 0.000e+00 in 11 of 15 (seed, angle)
        cells and |1.5e-14| in the rest. With `B` left alone the plant really
        is mismatched and the unconstrained optimum moves −4.5 % to +14.5 %.
        The line below used to read `R @ B if self.perturb_B else R @ B`, both
        arms the same expression, so every caller got the degenerate one.
        """
        R = rotation_matrix(A.shape[0], self.level, self.seed)
        A_pert = R @ A @ R.T
        B_pert = R @ B if self.perturb_B else B.copy()
        W_pert = R @ W @ R.T
        realized_rho = float(np.max(np.abs(np.linalg.eigvals(A_pert))))
        return (
            A_pert,
            B_pert,
            W_pert,
            {
                "spectral_radius": realized_rho,
                "angle_degrees": float(self.level),
            },
        )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full perturbation specification.

        Returns:
            ``{"type": "PlantPerturbation", **fields}``.
        """
        return {"type": type(self).__name__, **dataclasses.asdict(self)}


@dataclass(frozen=True)
class PerturbationDistribution:
    """What a training epoch loop draws its plant FROM: either a fresh
    `PlantPerturbation.draw` every call (the additive domain-randomization
    regime -- NB07 plan Sec 4.4's D-add axis) or one draw held fixed for
    every subsequent call (the rotation regime's degenerate "distribution" --
    Sec 4.4's D-rot axis, where nothing is left to redraw once the angle is
    fixed).

    Attributes:
        perturbation: The underlying perturbation.
        resample_per_epoch: ``True`` redraws fresh every `build_stream`
            closure call (advancing one internal generator seeded from
            `perturbation.seed`); ``False`` draws once and returns the same
            result forever.
    """

    perturbation: PlantPerturbation
    resample_per_epoch: bool = True

    def build_stream(
        self, A: np.ndarray, B: np.ndarray, W: np.ndarray
    ) -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]]:
        """Build the zero-argument closure a training loop calls once per
        epoch to obtain that epoch's `(A, B, W)`.

        Args:
            A: The nominal state transition matrix.
            B: The nominal control input matrix.
            W: The nominal process noise covariance.

        Returns:
            A zero-argument callable returning ``(A', B', W', realized)``,
            fresh every call if `resample_per_epoch`, else the same frozen
            draw every call.
        """
        rng = np.random.default_rng(self.perturbation.seed)
        if not self.resample_per_epoch:
            frozen = self.perturbation.draw(A, B, W, rng)

            def _fixed() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
                return frozen

            return _fixed

        def _fresh() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
            return self.perturbation.draw(A, B, W, rng)

        return _fresh

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full distribution specification.

        Returns:
            ``{"type": "PerturbationDistribution", "perturbation": {...},
            "resample_per_epoch": bool}``.
        """
        return {
            "type": type(self).__name__,
            "perturbation": self.perturbation.get_signature(),
            "resample_per_epoch": self.resample_per_epoch,
        }
