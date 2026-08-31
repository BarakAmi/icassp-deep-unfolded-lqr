"""Evaluate a synthesized controller's fresh policy against a (possibly
PERTURBED) problem and constraint bound (docs/planning/NB07_ROBUST_TRAINING_PLAN.md
Sec 4.6/5.3): the shared primitive both the price-of-robustness study and the
cross-evaluation table rest on. Reuses `evaluate_synthesized_controller`'s
exact cost path (`require_linear_quadratic` -> `time_invariant_slice` ->
`total_quadratic_cost` under the problem's declared conventions, under
`torch.no_grad`), so a robustly-trained model's re-scored cost can never
drift from the objective it was trained against (the C2 law) -- and adds
only the two constraint statistics (NB07 plan Sec 4.6's corrected metric
pair, shared with NB06's identical §2.5 correction) plus optional trajectory
capture for the single phase-portrait figure (NB07 plan Sec 6's spectral/
trajectory renderers).

`total_quadratic_cost`'s `CostReduction.PER_SAMPLE` (one cost per trajectory)
is used rather than `BATCH_MEAN`, so `ShiftMetrics`' median/IQR are computed
over the full POOLED per-trajectory distribution (every trajectory in every
batch), not over `n_batches` batch-mean values -- a materially richer
statistic at the same evaluation cost, since every `EvaluationBatch` already
carries the same `batch_size` trajectories.
"""

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from .experiment import EvaluationBatch
from ..core.constraint.activity import SATURATION_TOLERANCE
from ..core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from ..core.optimal_control_problem import OptimalControlProblem
from ..core.utils import to_numpy
from ..models.guards import require_linear_quadratic
from ..models.lifecycle import SynthesizedController

logger = logging.getLogger(__name__)

#: The saturation tolerance, now `core.constraint.activity`'s and no longer a
#: second copy of it. This module held `1e-3` and `tools/probes/box_binding`
#: held `1e-6` while asserting the two were the same number -- the divergence
#: this alias exists to make impossible.
#:
#: **The reason recorded here for the value did not survive measurement.** It
#: read "the GRU's `u_max * tanh(.)` can approach but never reach `u_max`
#: exactly"; measured on a nine-contender cast, the recurrent baseline reads
#: 94.25 % at tolerance ZERO and is the least tolerance-sensitive member of the
#: cast, while the QP families -- whose interior-point solutions land near an
#: active constraint rather than on it -- swing 11.5 points. The value stands
#: and is re-derived in Annex 01 §4.1; the justification is replaced.
_SATURATION_TOLERANCE = SATURATION_TOLERANCE


@dataclass(frozen=True)
class ShiftMetrics:
    """The scored outcome of one `evaluate_under_shift` call -- carried data
    (like `applications.studies.nb04_box_constrained.BoxConstrainedFloors`),
    never signed: it is a live computed result, not a specification.

    Attributes:
        cost_mean: Mean cost over every trajectory in every batch.
        cost_median: Median cost over the same pooled distribution -- the
            primary statistic reported everywhere (mean is undefined under
            Cauchy noise, NB07 plan Sec 4.5).
        cost_q25: 25th percentile of the pooled per-trajectory cost.
        cost_q75: 75th percentile.
        cost_std: Sample standard deviation (``ddof=1``); ``0.0`` when fewer
            than two trajectories were evaluated in total.
        max_violation: ``max|u| - u_max`` over every trajectory/batch -- the
            feasibility audit (should be ``<=`` float tolerance for every
            contender; a positive value is a falsifiable defect, not merely
            an informative number).
        saturation_rate: ``P[|u| >= (1 - tol) * u_max]`` over every
            trajectory/batch -- the active-set fraction (the informative
            constraint statistic).
        n_batches: Number of `EvaluationBatch`es scored.
        n_trajectories: Total pooled trajectory count (``n_batches *
            batch_size``).
    """

    cost_mean: float
    cost_median: float
    cost_q25: float
    cost_q75: float
    cost_std: float
    max_violation: float
    saturation_rate: float
    n_batches: int
    n_trajectories: int


def evaluate_under_shift(
    artifact: SynthesizedController,
    problem: OptimalControlProblem,
    batches: tuple[EvaluationBatch, ...],
    *,
    u_max: float,
    capture_trajectories: bool = False,
) -> tuple[ShiftMetrics, dict[str, np.ndarray]]:
    """Roll `artifact`'s policy out on every batch in `batches`, against
    `problem` (nominal OR perturbed), and score cost + constraint statistics.

    Args:
        artifact: The synthesized (offline) controller -- possibly one
            trained under a DIFFERENT condition than `problem` describes
            (the price-of-robustness study's whole point).
        problem: The problem to roll out against; may be a perturbed
            instance distinct from whatever `artifact` was trained on.
        batches: Evaluation batches, drawn from `problem`'s own noise law.
        u_max: The infinity-norm control bound the feasibility audit and
            saturation rate are measured against.
        capture_trajectories: If ``True``, the FIRST batch's state/control
            trajectories are additionally returned (for a phase-portrait
            figure) under ``"trajectory_states"``/``"trajectory_controls"``.

    Returns:
        ``(metrics, arrays)``: the `ShiftMetrics`, plus
        ``{"batch_costs": <per-trajectory pooled cost array>}`` and, when
        `capture_trajectories`, the two trajectory arrays above.
    """
    _, cost = require_linear_quadratic(problem)
    Q = torch.as_tensor(time_invariant_slice(cost.Q))
    R = torch.as_tensor(time_invariant_slice(cost.R))

    per_trajectory_costs: list[torch.Tensor] = []
    max_violations: list[float] = []
    saturation_hits: list[torch.Tensor] = []
    captured: dict[str, np.ndarray] = {}

    for batch_index, (initial_state, process_noise, measurement_noise) in enumerate(
        batches
    ):
        policy = artifact.make_policy()
        with torch.no_grad():
            X, _, U = cast(
                "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                problem.system.run(
                    policy, initial_state, process_noise, measurement_noise
                ),
            )
            batch_cost = cast(
                torch.Tensor,
                total_quadratic_cost(
                    Q.to(dtype=X.dtype, device=X.device),
                    R.to(dtype=U.dtype, device=U.device),
                    X,
                    U,
                    conventions=cost.conventions,
                    reduction=CostReduction.PER_SAMPLE,
                ),
            )
            abs_u = U.abs()
            max_violations.append(float(abs_u.max()) - u_max)
            saturation_hits.append(abs_u >= (1.0 - _SATURATION_TOLERANCE) * u_max)
        per_trajectory_costs.append(batch_cost)
        if capture_trajectories and batch_index == 0:
            captured["trajectory_states"] = to_numpy(X)
            captured["trajectory_controls"] = to_numpy(U)

    pooled_costs = torch.cat(per_trajectory_costs).cpu().numpy().astype(np.float64)
    saturation_rate = float(
        torch.cat([hits.reshape(-1) for hits in saturation_hits]).float().mean()
    )

    metrics = ShiftMetrics(
        cost_mean=float(pooled_costs.mean()),
        cost_median=float(np.median(pooled_costs)),
        cost_q25=float(np.percentile(pooled_costs, 25)),
        cost_q75=float(np.percentile(pooled_costs, 75)),
        cost_std=float(pooled_costs.std(ddof=1)) if pooled_costs.size > 1 else 0.0,
        max_violation=float(max(max_violations)),
        saturation_rate=saturation_rate,
        n_batches=len(batches),
        n_trajectories=int(pooled_costs.size),
    )
    logger.info("Shifted evaluation finished: %s", metrics)
    return metrics, {"batch_costs": pooled_costs, **captured}
