"""Spectral analysis of a learned Riccati-replacement matrix `P`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 6.4): the quantitative
replacement for the legacy notebook's eyeballed "rotation DR pushes P toward
a scalar multiple of I" claim (`model_robustness_analysis.ipynb` cell 34) --
an isotropy index with a number and a band that can be WRONG, not a
heat-map triptych asserted from inspection.

Pure numerics: `P` arrives as a plain `np.ndarray` (already extracted from
`RobustTrainingPoint.learned_p`), never a torch tensor or a live controller.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class PSpectrum:
    """One learned matrix's spectral summary.

    Attributes:
        eigenvalues: Eigenvalues of ``(P + P^T) / 2``, sorted descending.
        spectral_norm: ``max(|eigenvalues|)``.
        condition_number: ``spectral_norm / min(|eigenvalues|)``; `math.inf`
            if the smallest eigenvalue magnitude is exactly zero.
        isotropy_index: ``||P_sym - (tr(P_sym)/n) I||_F / ||P_sym||_F``, in
            ``[0, 1]``: ``0`` for an EXACT scalar multiple of the identity
            (the rotation-DR fixed point the legacy notebook claimed by eye),
            larger for a more anisotropic matrix.
        principal_angles_deg: Principal angles (degrees, ascending) between
            `P`'s and a `reference` matrix's leading `k`-dimensional
            eigenspaces (see `compute_p_spectrum`'s `k`); ``None`` when no
            `reference` was given.
    """

    eigenvalues: tuple[float, ...]
    spectral_norm: float
    condition_number: float
    isotropy_index: float
    principal_angles_deg: tuple[float, ...] | None


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 0.5 * (matrix + matrix.T))


def compute_principal_angles(
    matrix_a: np.ndarray, matrix_b: np.ndarray, *, k: int = 1
) -> np.ndarray:
    """Principal angles (radians, ascending) between the `k`-dimensional
    leading eigenspaces of two symmetric matrices, via the singular values
    of their (orthonormal) eigenbases' inner product.

    Args:
        matrix_a: A symmetric matrix, shape ``(n, n)``.
        matrix_b: A symmetric matrix, shape ``(n, n)``.
        k: The leading-eigenspace dimension to compare (``k=1`` is the
            angle between the two dominant eigenvector directions).

    Returns:
        The `k` principal angles, ascending, in radians, each in
        ``[0, pi/2]``.
    """
    eigenvalues_a, eigenvectors_a = np.linalg.eigh(_symmetrize(matrix_a))
    eigenvalues_b, eigenvectors_b = np.linalg.eigh(_symmetrize(matrix_b))
    # np.linalg.eigh returns ASCENDING eigenvalues; the leading (largest-
    # magnitude, by convention here largest-VALUE) k columns are the last k.
    n = matrix_a.shape[0]
    top_a = eigenvectors_a[:, n - k :]
    top_b = eigenvectors_b[:, n - k :]
    del eigenvalues_a, eigenvalues_b
    cosines = np.linalg.svd(top_a.T @ top_b, compute_uv=False)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.asarray(np.sort(np.arccos(cosines)))


def compute_p_spectrum(
    matrix: np.ndarray, *, reference: np.ndarray | None = None, k: int = 1
) -> PSpectrum:
    """The spectral summary of one learned `P` (NB07 plan Sec 6.4).

    Args:
        matrix: The learned Riccati-replacement matrix, shape ``(n, n)``
            (need not already be symmetric -- symmetrized internally).
        reference: An optional reference matrix (typically the nominal DARE
            solution) to compute `PSpectrum.principal_angles_deg` against.
        k: The leading-eigenspace dimension for the principal-angle
            comparison (ignored when `reference` is ``None``).

    Returns:
        The `PSpectrum`.
    """
    symmetric = _symmetrize(matrix)
    eigenvalues = np.linalg.eigvalsh(symmetric)[::-1]  # descending
    magnitudes = np.abs(eigenvalues)
    spectral_norm = float(magnitudes.max())
    smallest = float(magnitudes.min())
    condition_number = spectral_norm / smallest if smallest > 0 else math.inf

    n = symmetric.shape[0]
    frobenius_norm = float(np.linalg.norm(symmetric, ord="fro"))
    if frobenius_norm > 0:
        isotropic_component = (np.trace(symmetric) / n) * np.eye(n)
        isotropy_index = float(
            np.linalg.norm(symmetric - isotropic_component, ord="fro") / frobenius_norm
        )
    else:
        isotropy_index = 0.0

    principal_angles_deg: tuple[float, ...] | None = None
    if reference is not None:
        angles = compute_principal_angles(symmetric, reference, k=k)
        principal_angles_deg = tuple(np.degrees(angles).tolist())

    return PSpectrum(
        eigenvalues=tuple(eigenvalues.tolist()),
        spectral_norm=spectral_norm,
        condition_number=condition_number,
        isotropy_index=isotropy_index,
        principal_angles_deg=principal_angles_deg,
    )


def get_signature(spectrum: PSpectrum) -> dict[str, Any]:
    """Small convenience: `PSpectrum` as a flat, JSON-serializable dict
    (e.g. for a results table row) -- NOT a `Signable` specification (this
    is a computed result, never a specification)."""
    return {
        "spectral_norm": spectrum.spectral_norm,
        "condition_number": spectrum.condition_number,
        "isotropy_index": spectrum.isotropy_index,
        "n_eigenvalues": len(spectrum.eigenvalues),
        "principal_angles_deg": spectrum.principal_angles_deg,
    }
