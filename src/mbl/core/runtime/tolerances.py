"""The project's named numeric-tolerance policy (T1.d).

One module defines the numeric contract by name; the validators and
predicates in ``core.utils`` take their defaults from here instead of
scattering magic literals. Additional named tolerances (relative-error
floors, reparameterization clamps, the palette module's stray
``DEFAULT_EPSILON``) migrate here as their owning modules are refactored
in later stages.

These values are the pre-S1 defaults, frozen by the golden-master harness:
naming them must not change any default behavior.
"""

from __future__ import annotations

SYMMETRY_ATOL: float = 1e-12
"""Absolute tolerance for matrix-symmetry checks (``np.allclose(M, M.T)``)."""

DEFINITENESS_TOL: float = 1e-9
"""Eigenvalue threshold for definiteness checks: positive definite requires
``min_eig > DEFINITENESS_TOL``; positive semi-definite requires
``min_eig > -DEFINITENESS_TOL`` (numerical slack around zero)."""
