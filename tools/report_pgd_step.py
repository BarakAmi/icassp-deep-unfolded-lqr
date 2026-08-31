"""Report the analytic PGD's step size for a frozen problem, ready to paste.

The analytic PGD (`unfolded_fixed`, declared as `standard_pgd`) descends the
per-step LQR objective with a FIXED step, and on a convex quadratic projected
gradient descent converges only for a step strictly below ``2/L``. Every study
in this repository declares the literal ``step_size_init = 0.05``. Measured on
the campaign's own frozen plants, that literal is **above** ``2/L`` on four of
seven -- by 4.98x at n=15, 10.87x at n=20, 5.99x at n=30 and 23.87x at n=50 --
where the iteration limit-cycles and the cost-vs-depth curve goes non-monotone
while the box keeps ``max|u|`` pinned at ``u_max`` and hides it.

**Why a tool and not a build-time computation.** ``step_size_init``
participates in the `ModelID` at ONE-ULP granularity, and ``eigvalsh``,
``eigvals``, ``norm(., 2)``, ``svd`` and the closed 2x2 form disagree by exactly
one ULP on the n=4 plant. Resolving ``1/L`` while building a controller would
therefore make model identity a function of which LAPACK path ran -- two
machines, or two library versions, would disagree about which models they
already have. So the number is authored ONCE, here, and frozen into the study
document as a literal, exactly as D19 freezes the problem matrices themselves
rather than the generator call that produced them. (Thread count and
torch-vs-numpy `eigvalsh` were measured bit-identical, so the hazard is the
routine, not the machine -- which is why `gradient_lipschitz_constant` names
its routine in its docstring.)

A frozen literal cannot go stale silently, because the accompanying gate
recomputes ``L`` from the study's own problem and refuses a document whose
declared step is not ``1/L``.

Usage::

    uv run python tools/report_pgd_step.py studies/icassp/*.npz
    uv run python tools/report_pgd_step.py studies/icassp/icassp_n4m2_N100_u0p1_s0.npz --candidate 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mbl.models.analytic import problem_gradient_lipschitz_constant
from mbl.spec.problem import ProblemSpec


def report(path: Path, candidate: float | None) -> dict[str, object]:
    """`L`, `1/L`, `2/L` and where a candidate step sits, for one frozen problem.

    Args:
        path: A frozen problem `.npz`.
        candidate: A step size to judge against `2/L`, or None.

    Returns:
        A row of the printed table.
    """
    problem = ProblemSpec.load(path).build()
    lipschitz = problem_gradient_lipschitz_constant(problem)
    row: dict[str, object] = {
        "plant": path.name,
        "L": lipschitz,
        "one_over_L": 1.0 / lipschitz,
        "two_over_L": 2.0 / lipschitz,
    }
    if candidate is not None:
        row["candidate"] = candidate
        row["candidate_over_2_over_L"] = candidate / (2.0 / lipschitz)
        row["converges"] = candidate < 2.0 / lipschitz
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problems", type=Path, nargs="+", help="frozen problem .npz files"
    )
    parser.add_argument(
        "--candidate",
        type=float,
        default=0.05,
        help="a step size to judge against 2/L (default: the literal every study declares)",
    )
    arguments = parser.parse_args()

    rows = [report(path, arguments.candidate) for path in arguments.problems]

    print(
        f"{'plant':44s} {'L':>16s} {'1/L':>14s} {'2/L':>14s} {'cand/(2/L)':>11s}  converges"
    )
    for row in rows:
        print(
            f"{row['plant']:44s} {row['L']:16.9f} {row['one_over_L']:14.9f} "
            f"{row['two_over_L']:14.9f} {row['candidate_over_2_over_L']:11.4f}  "
            f"{'yes' if row['converges'] else 'NO -- DIVERGES'}"
        )

    # `repr`, not a format spec: the pasted literal must round-trip to the same
    # float, because one ULP of it moves the ModelID.
    print("\nPaste-ready `step_size_init` (repr-exact, round-trips bit-for-bit):")
    for row in rows:
        print(f"  # {row['plant']}\n  step_size_init = {row['one_over_L']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
