"""Freeze the ICASSP campaign's problems to disk (Phase A of the campaign plan).

D19 makes a problem a frozen `.npz` rather than a generator call: identity is
derived from the matrices, so a study that re-drew them would silently change
what its models were trained on. This script is the authoring step that D19
implies and that the repository did not have -- the one committed problem was
hand-authored, and nothing could reproduce it.

Two things it does that the factory alone cannot.

**It targets a spectral radius and refuses to write when it misses.**
`LQRProblemFactory` divides `A` by `max|eigvals(A)|`, which is exactly 1 in
exact arithmetic and, measured, lands *strictly above* 1.0 on three of five
seeds at n = 4 (worst +2.89e-15, about 13 ULP, at n = 100). The campaign wants
ρ = 0.999: marginally stable, but genuinely settling inside a 100-step horizon,
because at ρ = 1 the closed loop was measured still unsettled at t = 99 on two
of five seeds and a time-averaged cost then reports a transient. The assertion
lives *here*, in the authoring step, and not in a `stability` gate -- the gate
is inert (`mbl.analysis.gates` routes a parse-stage gate to `DECIDED_AT_PARSE`
without ever looking at `A`), so declaring it would print a section asserting a
check nobody ran.

**It rotates `A` alone.** Figure 2 needs a plant whose dynamics are wrong by a
rotation while `B` is not. Neither existing generator can express that:
`ood.perturbations` returns `R @ A @ R.T, R @ B` unconditionally, and
`uncertainty.perturbations` reads, literally,
`B_pert = R @ B if self.perturb_B else R @ B` -- both branches identical, so
asking for `perturb_B=False` silently returns the co-rotated matrices. That
matters because co-rotating `B` makes the experiment a *similarity transform*,
under which the re-solved optimum was measured to move by 0.000e+00 on this
isotropic instance. With `B` left alone the same rotation moves the clipped box
cost by -12.3 % to +139.1 %.

The rotation matrix itself is **not** re-implemented: `build_rotation_matrix`
from `mbl.applications.ood.perturbations` is the repository's one construction
whose canonical angle in every invariant plane is exactly the requested one,
and a second convention here would be a second answer to "what is 30 degrees".

Usage::

    uv run python tools/author_icassp_problems.py --state-dim 4 --control-dim 2

Run with `--help` for the full surface. Writing is refused, not overwritten,
unless `--force` is given: a frozen problem is what a stored model's identity
points at, and replacing one in place detaches every measurement from the
matrices it was scored on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mbl.applications.factories import LQRProblemFactory
from mbl.applications.ood.perturbations import build_rotation_matrix
from mbl.spec.problem import GeneratorProvenance, ProblemData, ProblemSpec

#: The campaign's spectral-radius target. Marginally stable, and far enough
#: below 1 that the closed loop settles inside N = 100.
DEFAULT_RHO = 0.999

#: How close the written matrix must be to that target. This is the assertion
#: the inert `stability` gate cannot make. It is tight on purpose: the rescale
#: is one multiplication, so anything looser would be hiding an error rather
#: than allowing for arithmetic.
RHO_TOLERANCE = 1e-12

#: The campaign's rotation for Figure 2, in degrees. Measured mean cost gap by
#: angle: 0.39 % (2), 5.26 % (15), 31.9 % (30), 77.1 % (45), 419.9 % (90);
#: beyond 45 a mismatched controller destabilises a marginally stable plant.
DEFAULT_DEGREES = 30.0

#: Where frozen problems live: beside the study that names them, since
#: `[problem] path` is resolved relative to the study document.
DEFAULT_DESTINATION = Path("studies/icassp")


class AuthoringError(RuntimeError):
    """A problem could not be authored to the requested specification."""


def spectral_radius(matrix: NDArray[np.float64]) -> float:
    """`max |lambda_i|` of a square matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _require_spectral_radius(
    matrix: NDArray[np.float64], target: float, *, context: str
) -> None:
    """Refuse a matrix whose spectral radius is not the requested one.

    Raises:
        AuthoringError: If `|rho(matrix) - target|` exceeds `RHO_TOLERANCE`.
            Refused rather than warned about, because the whole reason this
            assertion lives in the authoring step is that nothing downstream
            makes it -- a warning here is a plant nobody checked.
    """
    achieved = spectral_radius(matrix)
    if abs(achieved - target) > RHO_TOLERANCE:
        raise AuthoringError(
            f"{context}: spectral radius is {achieved!r}, which misses the "
            f"target {target!r} by {abs(achieved - target):.3e} (tolerance "
            f"{RHO_TOLERANCE:.0e})"
        )


def _matrices(problem: Any) -> tuple[NDArray[np.float64], ...]:
    """`A`, `B`, `Q`, `R` read back off a built `OptimalControlProblem`.

    Read from the built object rather than re-drawn, so the factory stays the
    single author of the draw and this script only ever *rescales* what it
    produced. Re-implementing the draw here would be a second generator that
    has to agree with the first.
    """
    return (
        np.asarray(problem.system.A_t.array, dtype=np.float64),
        np.asarray(problem.system.B_t.array, dtype=np.float64),
        np.asarray(problem.cost.Q, dtype=np.float64),
        np.asarray(problem.cost.R, dtype=np.float64),
    )


def author_nominal(
    *,
    state_dim: int,
    control_dim: int,
    horizon: int,
    seed: int,
    u_max: float,
    rho: float = DEFAULT_RHO,
) -> ProblemSpec:
    """One nominal plant, drawn by the factory and rescaled to `rho`.

    The cost matrices are the factory's stacked identities and are carried
    through untouched. The 1/N of the time-averaged convention is applied by
    the cost kernel, not folded into `Q`: folding it would change the
    `ProblemID` for identical physics.

    Args:
        state_dim: `n`.
        control_dim: `m`.
        horizon: `N`, the number of steps the cost is stacked for.
        seed: The factory's draw seed.
        u_max: The box bound. Positive; an unconstrained problem is a
            different object and this script does not author one.
        rho: The spectral radius `A` is rescaled to.

    Returns:
        The frozen specification, with provenance naming this script.

    Raises:
        AuthoringError: If the rescaled `A` misses `rho`, or if `u_max` is not
            positive.
    """
    if u_max <= 0:
        raise AuthoringError(f"u_max must be positive, got {u_max!r}")
    if not np.isfinite(rho) or rho <= 0:
        raise AuthoringError(
            f"rho must be a positive finite target, got {rho!r}; scaling to "
            "zero produces the zero matrix, whose spectral radius IS zero, so "
            "the check below would pass on a plant with no dynamics at all"
        )
    A, B, Q, R = _matrices(
        LQRProblemFactory(
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=horizon,
            seed=seed,
            u_max=u_max,
        ).build()
    )
    # The factory has already normalised to rho = 1 (to within float64), so
    # this is a rescale of a normalised matrix rather than a second
    # normalisation of a raw one -- which is why the tolerance can be 1e-12.
    A = A * (rho / spectral_radius(A))
    _require_spectral_radius(A, rho, context=f"nominal n={state_dim} seed={seed}")
    return ProblemSpec(
        data=ProblemData(
            system={"A": A, "B": B},
            cost={"Q": Q, "R": R},
            horizon=horizon,
            control_bound=u_max,
        ),
        provenance=GeneratorProvenance(
            generator="tools/author_icassp_problems.py:author_nominal",
            params={
                "base": "LQRProblemFactory",
                "state_dim": state_dim,
                "control_dim": control_dim,
                "horizon": horizon,
                "seed": seed,
                "u_max": u_max,
                "rho": rho,
            },
        ),
    )


def author_rotated_a(
    source: ProblemSpec, *, degrees: float = DEFAULT_DEGREES, seed: int = 0
) -> ProblemSpec:
    """`source` with `A` rotated and everything else left alone.

    `A' = R A R^T`, with `B`, `Q`, `R` and the horizon carried through
    unchanged. Rotating `B` as well would make this a similarity transform, and
    on an isotropic instance a controller handed the transformed matrices
    attains the nominal cost exactly -- the experiment would measure nothing.

    The spectral radius is invariant under `A -> R A R^T` (similar matrices
    share eigenvalues), so the rotated plant inherits the source's marginal
    stability. That is asserted rather than assumed.

    Args:
        source: The nominal problem to rotate.
        degrees: The canonical angle, applied to every invariant plane.
        seed: Seeds the Haar-random plane structure. Held separate from the
            plant's own seed so the same rotation can be applied to different
            plants, and so a change of angle does not change which planes turn.

    Returns:
        The rotated specification.

    Raises:
        AuthoringError: If the rotation changed the spectral radius, which
            would mean the rotation matrix was not orthogonal.
    """
    A = np.asarray(source.data.system["A"], dtype=np.float64)
    if A.ndim != 2:
        raise AuthoringError(
            f"rotation is defined for a time-invariant A; got shape {A.shape}"
        )
    rotation = build_rotation_matrix(
        state_dim=A.shape[0], degrees=degrees, rng=np.random.default_rng(seed)
    )
    rotated = rotation @ A @ rotation.T
    _require_spectral_radius(
        rotated, spectral_radius(A), context=f"rotated {degrees} deg"
    )
    params = dict(getattr(source.provenance, "params", {}))
    return ProblemSpec(
        data=ProblemData(
            system={"A": rotated, "B": source.data.system["B"]},
            cost=dict(source.data.cost),
            horizon=source.data.horizon,
            control_bound=source.data.control_bound,
        ),
        provenance=GeneratorProvenance(
            generator="tools/author_icassp_problems.py:author_rotated_a",
            params={
                **params,
                "rotated_from": str(source.problem_id),
                "degrees": degrees,
                "rotation_seed": seed,
                "rotates": "A only (B, Q, R untouched)",
            },
        ),
    )


def author_rotated_ab(
    source: ProblemSpec, *, degrees: float = DEFAULT_DEGREES, seed: int = 0
) -> ProblemSpec:
    """`source` with `A` **and** `B` co-rotated: the similarity-transform control.

    `A' = R A R^T`, `B' = R B`, with `Q`, `R`, the bound and the horizon
    untouched. On this campaign's isotropic instance (`Q = I`, `R = I`,
    isotropic noise) the substitution `z = R^T x` maps this plant exactly onto
    the source, so it is the *same* plant in rotated coordinates — which is
    precisely why the author commissioned it (2026-08-08): an informed
    controller that re-derives fully must land on the nominal cost exactly,
    and any nonzero informed-analytic degradation on this plant is a defect in
    the pipeline, not a finding. Two invariances follow and are asserted by
    the authoring tests rather than assumed: the spectral radius (similarity),
    and the gradient's Lipschitz constant (`B'^T P' B' = B^T P B`), which is
    why the analytic PGD's nominal step literal is exactly the right step here
    and no per-plant override is declared for it.

    The rotation is the SAME `R` as `author_rotated_a` at the same `seed`, so
    the A-rotation is identical between the two shifted plants and the only
    difference between them is whether `B` co-rotates.

    Args:
        source: The nominal problem to rotate.
        degrees: The canonical angle, applied to every invariant plane.
        seed: Seeds the Haar-random plane structure, exactly as in
            `author_rotated_a`.

    Returns:
        The co-rotated specification.

    Raises:
        AuthoringError: If the rotation changed the spectral radius.
    """
    A = np.asarray(source.data.system["A"], dtype=np.float64)
    B = np.asarray(source.data.system["B"], dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise AuthoringError(
            "co-rotation is defined for time-invariant matrices; got shapes "
            f"A: {A.shape}, B: {B.shape}"
        )
    rotation = build_rotation_matrix(
        state_dim=A.shape[0], degrees=degrees, rng=np.random.default_rng(seed)
    )
    rotated = rotation @ A @ rotation.T
    _require_spectral_radius(
        rotated, spectral_radius(A), context=f"co-rotated {degrees} deg"
    )
    params = dict(getattr(source.provenance, "params", {}))
    return ProblemSpec(
        data=ProblemData(
            system={"A": rotated, "B": rotation @ B},
            cost=dict(source.data.cost),
            horizon=source.data.horizon,
            control_bound=source.data.control_bound,
        ),
        provenance=GeneratorProvenance(
            generator="tools/author_icassp_problems.py:author_rotated_ab",
            params={
                **params,
                "rotated_from": str(source.problem_id),
                "degrees": degrees,
                "rotation_seed": seed,
                "rotates": "A and B (Q, R untouched; the similarity control)",
            },
        ),
    )


def nominal_filename(
    *, state_dim: int, control_dim: int, horizon: int, u_max: float, seed: int
) -> str:
    """The name a nominal plant is frozen under.

    Every quantity that distinguishes two plants appears, because the file name
    is the only thing a reader sees in a study document -- `[problem] path` is
    what the study names, and a `ProblemID` is not legible.
    """
    # The decimal point is replaced in the bound's own fragment and nowhere
    # else: a blanket `.replace(".", "p")` over the whole name also eats the
    # suffix, turning `s0.npz` into `s0pnpz` -- which then fails the explicit
    # `.npz` check in `_write` rather than producing a wrong file, but only
    # because that check exists.
    bound = f"{u_max:g}".replace(".", "p")
    return f"icassp_n{state_dim}m{control_dim}_N{horizon}_u{bound}_s{seed}.npz"


def rotated_filename(nominal: str, *, degrees: float) -> str:
    """The rotated companion of `nominal`.

    The angle goes in the *name* because provenance is excluded from identity
    by construction: two angles produce two different `ProblemID`s, but nothing
    in the identifier says which is which.
    """
    return f"{nominal.removesuffix('.npz')}_rotA{degrees:g}.npz"


def rotated_ab_filename(nominal: str, *, degrees: float) -> str:
    """The co-rotated (similarity-control) companion of `nominal`."""
    return f"{nominal.removesuffix('.npz')}_rotAB{degrees:g}.npz"


def _write(spec: ProblemSpec, target: Path, *, force: bool) -> Path:
    """Freeze `spec` to `target`, refusing to replace an existing file.

    `ProblemSpec.save` appends `.npz` when the name lacks it and then returns
    the path it was *given*, so a caller that trusts the return value gets a
    path that does not exist. Every name this module builds already carries the
    suffix, and this is asserted rather than trusted.
    """
    if target.suffix != ".npz":
        raise AuthoringError(f"{target} must be named with an explicit .npz suffix")
    if target.exists() and not force:
        raise AuthoringError(
            f"{target} already exists; a frozen problem is what stored models "
            "point at, so replacing one detaches every measurement scored on "
            "it. Pass --force if that is genuinely intended"
        )
    spec.save(target)
    if not target.exists():
        raise AuthoringError(f"{target} was not written")
    return target


def require_matching_nominal(authored: ProblemSpec, frozen: Path) -> ProblemSpec:
    """The frozen nominal on disk must be the one `authored` reproduces.

    A rotation is *of* a particular plant. Authoring one against a nominal that
    differs from the frozen file every stored model points at would produce a
    companion nobody can compare with anything — silently, since both files
    would look perfectly well formed.

    Raises:
        AuthoringError: If the frozen plant is absent, or if any system or cost
            array differs from the authored one, naming the array.
    """
    if not frozen.exists():
        raise AuthoringError(
            f"{frozen} does not exist; a rotation is derived from a frozen "
            "nominal plant, so author that first (without --rotations-only)"
        )
    on_disk = ProblemSpec.load(frozen)
    for name, mine in (*authored.data.system.items(), *authored.data.cost.items()):
        theirs = (
            on_disk.data.system.get(name)
            if name in on_disk.data.system
            else on_disk.data.cost.get(name)
        )
        if theirs is None or not np.array_equal(np.asarray(mine), np.asarray(theirs)):
            raise AuthoringError(
                f"the frozen nominal {frozen} differs from the authored one in "
                f"{name!r}; a rotation of a different plant is not a companion "
                "to this one"
            )
    return on_disk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the ICASSP campaign's problems to .npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--control-dim", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--u-max", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--seed", type=int, default=0, help="the plant draw seed")
    parser.add_argument(
        "--degrees",
        type=float,
        default=DEFAULT_DEGREES,
        help="rotation angle for the Figure-2 companion; 0 writes no companion",
    )
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument(
        "--ab",
        action="store_true",
        help="also write the co-rotated (A and B) similarity-control companion",
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing frozen problem"
    )
    parser.add_argument(
        "--rotations-only",
        action="store_true",
        help=(
            "write only the rotated companions, leaving the frozen nominal "
            "untouched -- and verify the authored nominal reproduces it, so a "
            "companion is always a rotation OF the plant on disk"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    name = nominal_filename(
        state_dim=args.state_dim,
        control_dim=args.control_dim,
        horizon=args.horizon,
        u_max=args.u_max,
        seed=args.seed,
    )
    try:
        nominal = author_nominal(
            state_dim=args.state_dim,
            control_dim=args.control_dim,
            horizon=args.horizon,
            seed=args.seed,
            u_max=args.u_max,
            rho=args.rho,
        )
        if args.rotations_only:
            require_matching_nominal(nominal, args.destination / name)
            written = []
        else:
            written = [_write(nominal, args.destination / name, force=args.force)]
        if args.degrees:
            rotated = author_rotated_a(
                nominal, degrees=args.degrees, seed=args.rotation_seed
            )
            written.append(
                _write(
                    rotated,
                    args.destination / rotated_filename(name, degrees=args.degrees),
                    force=args.force,
                )
            )
            if args.ab:
                co_rotated = author_rotated_ab(
                    nominal, degrees=args.degrees, seed=args.rotation_seed
                )
                written.append(
                    _write(
                        co_rotated,
                        args.destination
                        / rotated_ab_filename(name, degrees=args.degrees),
                        force=args.force,
                    )
                )
    except AuthoringError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

    for path in written:
        spec = ProblemSpec.load(path)
        rho = spectral_radius(np.asarray(spec.data.system["A"]))
        print(
            f"{path}  n={spec.state_dim} m={spec.control_dim} "
            f"N={spec.horizon} rho={rho:.15f}  {spec.problem_id}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
