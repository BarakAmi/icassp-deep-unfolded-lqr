"""Figure 5's plant differs from Figure 3's in the box and in nothing else.

The whole economy of [the stress-test plan][plan] rests on one claim: narrowing
`u_max` moves the constraint and leaves the arithmetic alone. If it holds, the
declared step literals, the `step_is_inverse_lipschitz` gate and the step
ablation that chose `2/L` all carry over unedited; if it fails anywhere, every
one of them is quietly wrong and the figure is drawn against a plant nobody
checked.

**It is a property of two files on disk**, which is why it is a test rather than
a paragraph. `tools/author_icassp_problems.py` draws `A` and `B` from a seeded
factory and rescales to the target spectral radius; `u_max` takes no part in the
draw. That is the *intent* -- this asserts the outcome, bit for bit.

**`L` is asserted by exact equality, and that is the point.** The Lipschitz
constant is a function of `A`, `B`, `Q`, `R` and the horizon and the box does not
enter it, so the two plants must agree to the last bit rather than to a
tolerance. One ULP of `1/(2L)` moves a `ModelID`, so a check that passed within
`1e-12` would pass on a plant whose declared step is a different experiment.

[plan]: docs/planning/06_icassp_exact_convex/figure_five_stress_test/binding_box_stress_test_at_n100.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mbl.spec.gates import inverse_lipschitz_constant
from mbl.spec.problem import ProblemSpec

STUDIES = Path(__file__).resolve().parents[2] / "studies" / "icassp"

#: Figure 3's instance, and Figure 5's. The names carry the box because
#: `ProblemID` is content-derived and says nothing about which is which.
LOOSE = STUDIES / "icassp_n100m30_N100_u0p1_s0.npz"
BOUND = STUDIES / "icassp_n100m30_N100_u0p02_s0.npz"

#: The literals the exact-convex documents declare, and which this plant must
#: keep legitimate. Written out rather than derived, so a change to the kernel
#: that moved them would fail here instead of agreeing with itself.
LIPSCHITZ = 1386.686968045241
HALF_STEP = 0.0003605716441576073
DOUBLE_STEP = 0.0014422865766304293

#: What the authoring step targets, and the tolerance it refuses beyond.
RHO, RHO_TOLERANCE = 0.999, 1e-12


@pytest.fixture(scope="module")
def plants() -> tuple[ProblemSpec, ProblemSpec]:
    for path in (LOOSE, BOUND):
        assert path.exists(), f"{path} is not committed; Phase A did not run"
    return ProblemSpec.load(LOOSE), ProblemSpec.load(BOUND)


def test_every_matrix_is_bit_identical(
    plants: tuple[ProblemSpec, ProblemSpec],
) -> None:
    """`array_equal`, not `allclose`: the claim is that the draw did not move."""
    loose, bound = plants
    for group in ("system", "cost"):
        for name in sorted(getattr(loose.data, group)):
            mine = np.asarray(getattr(loose.data, group)[name])
            theirs = np.asarray(getattr(bound.data, group)[name])
            assert mine.shape == theirs.shape, f"{group}.{name} changed shape"
            assert np.array_equal(mine, theirs), (
                f"{group}.{name} differs; the box is not the only thing that "
                f"moved (max |diff| = {np.abs(mine - theirs).max():.3e})"
            )


def test_the_box_is_the_one_thing_that_moved(
    plants: tuple[ProblemSpec, ProblemSpec],
) -> None:
    loose, bound = plants
    assert loose.data.control_bound == 0.1
    assert bound.data.control_bound == 0.02
    assert loose.data.horizon == bound.data.horizon == 100
    assert (bound.data.state_dim, bound.data.control_dim) == (100, 30)


def test_the_two_plants_are_two_problems(
    plants: tuple[ProblemSpec, ProblemSpec],
) -> None:
    """The other half of the claim, and the one a reader forgets.

    `control_bound` is content, so the narrowed plant must sign *differently* --
    otherwise the new study would resolve to the old study's models and reuse a
    cast trained under a box it never met.
    """
    loose, bound = plants
    assert loose.problem_id != bound.problem_id


def test_the_lipschitz_constant_does_not_see_the_box(
    plants: tuple[ProblemSpec, ProblemSpec],
) -> None:
    loose, bound = plants
    assert inverse_lipschitz_constant(loose) == inverse_lipschitz_constant(bound)
    assert inverse_lipschitz_constant(bound) == LIPSCHITZ


def test_both_declared_step_literals_still_reproduce(
    plants: tuple[ProblemSpec, ProblemSpec],
) -> None:
    """Exact equality against the literals the documents carry.

    `1/(2L)` is what the three unfolded families declare and what the
    `step_is_inverse_lipschitz` gate compares against; `2/L` is Standard-PGD's,
    chosen by its own five-step ablation. Both are reproduced here from the
    narrowed plant, which is what lets the new document inherit them unedited.
    """
    _, bound = plants
    lipschitz = inverse_lipschitz_constant(bound)
    assert 1.0 / (2.0 * lipschitz) == HALF_STEP
    assert 2.0 / lipschitz == DOUBLE_STEP


def test_the_narrowed_plant_hits_the_campaign_s_spectral_radius(
    plants: tuple[ProblemSpec, ProblemSpec],
) -> None:
    """The authoring step's own assertion, re-made against the file it wrote."""
    _, bound = plants
    radius = float(np.max(np.abs(np.linalg.eigvals(bound.data.system["A"]))))
    assert abs(radius - RHO) <= RHO_TOLERANCE
