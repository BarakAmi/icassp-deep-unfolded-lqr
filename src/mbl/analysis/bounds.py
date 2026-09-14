"""Bounds an analysis may declare beside its contenders.

`Role.BOUND` names "a computed bound that is *not* an attained policy", and
until this module nothing could produce one: every use of the role in the
repository sat on a real, feasible policy whose realised cost is an **upper**
bound and whose identifier says `lower_bound` (Annex 01 §2.2, corrected
2026-08-20).

The absence was structural. A lower bound is not a rollout -- it is a property
of the *problem*, while a contender's measurement is a property of a trained
model scored on trajectories. So a bound enters here, computed from the study's
own problem and evaluation convention, and is **derived, never authored**: a
number typed into a document is the failure D19 exists to prevent, one tier up.

**The convention is the whole difficulty.** Figure 1 records omitting the
box-aware floor because the available one is an infinite-horizon steady-state
average while the figure draws a finite-horizon time-average, and *"plotting it
beside these numbers would bracket the truth with two different quantities"*.
Both are offered here and they are not interchangeable: `finite_horizon_box` is
the one a figure of time-averaged rollouts may draw.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from ..models.constrained.box_lagrangian import (
    BoxLQRData,
    DualBound,
    finite_horizon_box_bound,
    infinite_horizon_box_bound,
)
from ..spec.errors import SpecificationError
from ..spec.study import StudySpec


class _BoundKind(Protocol):
    """One computable bound, given the problem and the evaluation convention."""

    def __call__(self, data: BoxLQRData, horizon: int, x0_std: float) -> DualBound: ...


def _finite(data: BoxLQRData, horizon: int, x0_std: float) -> DualBound:
    """The box-aware floor in the convention a time-averaged figure evaluates."""
    covariance = x0_std**2 * np.eye(data.A.shape[0])
    return finite_horizon_box_bound(data, covariance, horizon)


def _infinite(data: BoxLQRData, horizon: int, x0_std: float) -> DualBound:
    """The box-aware floor on the steady-state average cost.

    Offered because it is the quantity the semidefinite program states, and
    **not** because a figure of finite-horizon rollouts may draw it.
    """
    return infinite_horizon_box_bound(data)


#: The declarable bounds. Named rather than inferred: which convention a figure
#: is in is the author's claim, and a default would make it the code's.
BOUND_KINDS: Mapping[str, _BoundKind] = {
    "finite_horizon_box": _finite,
    "infinite_horizon_box": _infinite,
}

#: What a READER is shown for each bound. A contender carries its own `display`
#: and a bound had none, so both figures that draw one printed the registry key
#: -- `finite_horizon_box`, an identifier, in a paper's figure. The names live
#: here rather than in the presentation tier so that adding a bound cannot
#: forget to name it: `test_every_bound_has_a_reader_facing_name` asserts the
#: two key sets are equal, which makes the omission structurally impossible
#: rather than merely discouraged.
#:
#: **Identity-free, exactly as a contender's `display` is** (Annex 01 §2.2.1):
#: renaming what a reader sees must not retrain anything or orphan a stored
#: result. These strings are read at render time and enter no identifier.
BOUND_DISPLAY: Mapping[str, str] = {
    "finite_horizon_box": "Finite-horizon box bound",
    "infinite_horizon_box": "Infinite-horizon box bound",
}


def _problem_data(study: StudySpec) -> tuple[BoxLQRData, int, float]:
    """The study's problem and evaluation convention, as a bound needs them.

    Args:
        study: The resolved study.

    Returns:
        ``(data, horizon, initial_state_std)``.

    Raises:
        SpecificationError: If the problem declares no control bound, so there
            is no box for a box-aware bound to be aware of.
    """
    problem = study.problem.data
    if problem.control_bound is None:
        raise SpecificationError(
            "a box-aware bound needs a control bound, and this study's problem "
            "declares none; an unconstrained problem's floor is its own optimum"
        )
    # `EvaluationProtocol.batch_spec` is typed `Signable` -- the seam that lets
    # an exotic noise family satisfy it structurally -- so the two convention
    # fields are read by name, and their absence is a real possibility rather
    # than a typing formality.
    protocol: Any = study.evaluation.protocol
    batch: Any = protocol.batch_spec
    noise = float(getattr(batch, "process_noise_std", 0.0))
    x0_std = float(getattr(batch, "initial_state_std", 0.0))
    n = problem.system["A"].shape[0]
    data = BoxLQRData(
        A=np.asarray(problem.system["A"], dtype=np.float64),
        B=np.asarray(problem.system["B"], dtype=np.float64),
        Q=np.asarray(problem.cost["Q"][0], dtype=np.float64),
        R=np.asarray(problem.cost["R"][0], dtype=np.float64),
        W=noise**2 * np.eye(n),
        u_max=float(problem.control_bound),
    )
    return data, int(problem.horizon), x0_std


def bound_rows(study: StudySpec, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The rows a study's declared bounds contribute to a tidy table.

    Args:
        study: The resolved study, which carries both the problem and the
            evaluation convention the bound must match.
        config: The analysis declaration; ``bounds`` names the kinds to compute.

    Returns:
        One row per declared bound, shaped like a contender's and carrying
        ``role = "bound"``. Empty when nothing is declared -- a default bound
        would apply a claim about convention to every study ever written.

    Raises:
        SpecificationError: If a declared kind is unknown, or the problem has
            no box.
    """
    declared = list(config.get("bounds", ()))
    if not declared:
        return []
    unknown = [kind for kind in declared if kind not in BOUND_KINDS]
    if unknown:
        raise SpecificationError(
            f"analysis declares unknown bound(s) {unknown}; available: "
            f"{sorted(BOUND_KINDS)}"
        )

    data, horizon, x0_std = _problem_data(study)
    rows: list[dict[str, Any]] = []
    for kind in declared:
        bound = BOUND_KINDS[kind](data, horizon, x0_std)
        rows.append(
            {
                "contender": kind,
                "role": "bound",
                "axis_path": "",
                # A bound does not vary with the swept axis, and says so the
                # same way a depth-invariant contender does.
                "axis_value": np.nan,
                "axis_label": "",
                # NOT a measurement: no seeds, no trajectories, no interval.
                # Reporting a spread here would invent a dispersion for a
                # quantity that has none.
                "n_seeds": 0,
                "n_trajectories": 0,
                "aggregate": bound.value,
                "interval_low": bound.value,
                "interval_high": bound.value,
                "within_seed_spread": 0.0,
                "across_seed_spread": 0.0,
                "seeds": [],
                "per_seed_aggregate": [],
            }
        )
    return rows


def bound_provenance(study: StudySpec, config: Mapping[str, Any]) -> dict[str, Any]:
    """What a declared bound's residuals were, for the analysis sidecar.

    A bound quoted without them is a number: this project has one on record at
    56.043029, reported by a solver that placed it *above* the optimum it
    claimed to bound.

    Args:
        study: The resolved study.
        config: The analysis declaration.

    Returns:
        Per-kind residuals, or an empty mapping when no bound is declared.
    """
    declared = list(config.get("bounds", ()))
    if not declared:
        return {}
    data, horizon, x0_std = _problem_data(study)
    out: dict[str, Any] = {}
    for kind in declared:
        bound = BOUND_KINDS[kind](data, horizon, x0_std)
        out[kind] = {
            "value": bound.value,
            "kkt_residual": bound.kkt_residual,
            "lmi_slack": bound.lmi_slack,
        }
    return out
