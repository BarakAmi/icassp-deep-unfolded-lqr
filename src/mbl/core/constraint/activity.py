"""How a control trajectory met its box — Annex 03 §A.3.1a.

**One home, because the project had two and they disagreed.** The saturated
fraction was computed in `experiments.shifted_evaluation` at a tolerance of
`1e-3` and in `tools/probes/box_binding` at `1e-6`, the second under a comment
asserting it held "the same number" as the first. Every saturation figure the
ICASSP campaign recorded came from the `1e-6` side, and nothing could have told
a reader which convention a number belonged to.

**The tolerance is not housekeeping.** A policy need not *write* the bound: an
interior-point solve satisfies an active constraint to solver tolerance, and a
smooth squashing function approaches it asymptotically. Measured across a
nine-contender cast at one box, moving the tolerance from `0` to `1e-3` moves
the exact convex policy from **76.32 % to 87.82 %** — from last place to third
— while the clipped baseline moves by a hundredth of a point. `1e-3` is chosen
on that measurement (Annex 01 §4.1): every contender is flat from `1e-4` to
`1e-3` and the whole swing lies below `1e-4`, so the value sits on a plateau
rather than on an edge.

**The two quantities answer different questions and neither replaces the
other.** `saturated` is the informative one — it says the study is not an
expensive repeat of the unconstrained problem — and it is the one that needs a
tolerance. `max_abs` is the falsifiable one: a feasible policy satisfies
``max|u| <= u_max`` or it does not, and no tolerance enters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

#: The relative width within which a control counts as *at* the bound, and the
#: value every stored statistic is counted at unless a caller says otherwise.
#: Measured rather than inherited -- see the module docstring and Annex 01
#: §4.1. It reaches the store beside the fraction it produced, because a
#: saturated fraction quoted without its tolerance says nothing.
SATURATION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ControlActivity:
    """One rollout's constraint activity, per trajectory.

    Attributes:
        max_abs: ``max_{t,i} |u_{t,i}|`` per trajectory, shape ``(batch,)``.
            Exact, and free of any tolerance.
        saturated: The fraction of the trajectory's scalar control entries at
            a bound, shape ``(batch,)``, counted at `tolerance`.
        tolerance: The relative width `saturated` was counted at, carried so
            that the fraction is never readable without it.
    """

    max_abs: torch.Tensor
    saturated: torch.Tensor
    tolerance: float


def control_activity(
    controls: torch.Tensor,
    *,
    u_max: Any,
    u_min: Any = None,
    tolerance: float = SATURATION_TOLERANCE,
) -> ControlActivity:
    """How often `controls` sits on its box, and how far it ever reached.

    Args:
        controls: The control trajectories, shape ``(batch, horizon, m)`` —
            the layout `system.run` returns.
        u_max: The upper bound: a scalar, or a per-dimension tensor
            broadcastable against the trailing axis.
        u_min: The lower bound, or `None` for the symmetric box ``-u_max``
            that `BoxConstraint` normalises to and `ProblemSpec` authors.
        tolerance: The relative width within which a control counts as at a
            bound. `0.0` asks the tolerance-free question — how many entries
            are at the *representable* bound — which is a different and also
            meaningful reading.

    Returns:
        The `ControlActivity`, reduced over ``(horizon, m)`` so that one value
        describes one trajectory. That is the unit Annex 03 §A.3 names, and it
        is what lets the statistic be re-reduced later rather than only read.

    Both bounds are tested, so an asymmetric box is counted on the side each
    entry is actually near. For the symmetric box this project authors the
    two-sided test is identically ``|u| >= (1 - tolerance) * u_max``, which is
    what the campaign's own record was measured with.
    """
    upper = torch.as_tensor(u_max, dtype=controls.dtype, device=controls.device)
    lower = (
        -upper
        if u_min is None
        else torch.as_tensor(u_min, dtype=controls.dtype, device=controls.device)
    )
    reduce_over = tuple(range(1, controls.ndim))
    # Inward from each bound by `tolerance` of THAT BOUND'S OWN magnitude.
    # Spelled this way rather than as `(1 - tolerance) * bound`, which is the
    # same expression for every box this project authors and wrong for one it
    # merely permits: with a lower bound above zero, scaling it moves the
    # threshold the wrong way, counting entries BELOW the bound as at it while
    # missing the ones actually on the face. Relative to the bound's magnitude
    # because a solver's slack is relative to the magnitude it computed, and a
    # bound of zero then admits only exact contact rather than a band.
    at_bound = (controls >= upper - tolerance * upper.abs()) | (
        controls <= lower + tolerance * lower.abs()
    )
    return ControlActivity(
        max_abs=controls.abs().amax(dim=reduce_over),
        saturated=at_bound.to(dtype=controls.dtype).mean(dim=reduce_over),
        tolerance=float(tolerance),
    )
