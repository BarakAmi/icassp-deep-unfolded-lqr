"""The shared ONLINE evaluation pass of `run_experiment` (T3.d): score one
synthesized controller's fresh policy on the experiment's common evaluation
batches, under the problem's *declared* cost conventions — the same
dual-backend kernel reduction `RolloutModel` trains against (C2 law), so the
evaluated objective and the trained objective can never drift apart.
"""

import logging
from typing import Any, cast

import numpy as np
import torch

from .experiment import EvaluationBatch
from ..core.constraint.activity import SATURATION_TOLERANCE, control_activity
from ..core.constraint.box_constraint import BoxConstraint
from ..core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from ..core.profiling import wall_clock
from ..core.optimal_control_problem import OptimalControlProblem
from ..models.guards import require_linear_quadratic
from ..models.lifecycle import SynthesizedController

logger = logging.getLogger(__name__)


def _box_of(problem: OptimalControlProblem) -> BoxConstraint | None:
    """The problem's box, or `None` when it declares none.

    `None` is the answer for an unconstrained problem and not a failure: the
    constraint-activity statistic is then **absent** from the measurement
    rather than zero, because a saturated fraction of zero is a claim about a
    box and there is no box (Annex 03 §A.3.1a).
    """
    for constraint in problem.constraints or ():
        if isinstance(constraint, BoxConstraint):
            return constraint
    return None


def _require_eval_mode(artifact: SynthesizedController) -> None:
    """Put the artifact's module into eval mode before it is scored.

    **`torch.no_grad()` is not eval mode, and the two are routinely
    confused.** The rollout below already runs under `no_grad`, which keeps
    evaluation out of the autograd graph and does nothing whatever about
    dropout or batch-norm statistics. A module left in training mode by
    `engine.train()` is therefore measured under live dropout, and the same
    policy on the same batch answers differently every call — measured on the
    campaign's GRU, **38.986540 against 38.988693**.

    That is not a tolerance question. The number entering the store under a
    `MeasurementID` would be one draw of a random variable, so re-running an
    identical study would disagree with itself, and §A.2's common-random-
    numbers law — which exists so two contenders can be compared on one test
    set — would be comparing them on two.

    Done HERE rather than in each recipe because this is the one seam every
    measurement passes through, so no family added later has to remember. The
    module is looked up on the *controller* rather than on the artifact, the
    same way `_weights_of` in the producer does and for the same reason:
    `TrainedControllerArtifact.as_module` exists unconditionally and delegates,
    so `hasattr(artifact, "as_module")` answers `True` even for a NumPy-native
    analytic controller on which calling it raises.
    """
    controller = getattr(artifact, "controller", artifact)
    as_module = getattr(controller, "as_module", None)
    if as_module is None:
        return
    try:
        module = as_module()
    except AttributeError:
        # An analytic controller reached through the delegating seam: it has
        # no module and nothing to switch, which is not an error.
        return
    module.eval()


def evaluate_synthesized_controller(
    artifact: SynthesizedController,
    problem: OptimalControlProblem,
    batches: tuple[EvaluationBatch, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Roll `artifact`'s policy out on every evaluation batch and score it.

    A *fresh* policy is drawn per batch (`make_policy`'s freshness law), the
    rollout runs under ``torch.no_grad`` (evaluation is never part of an
    autograd graph), and the cost honors the problem cost's declared
    ``include_terminal_cost``/``is_time_averaged`` flags.

    Args:
        artifact: The synthesized (offline) controller.
        problem: The shared optimal-control problem.
        batches: The experiment-wide common evaluation batches.

    Returns:
        ``(metrics, arrays)``: scalar metrics
        (``eval_expected_cost`` — mean across batches — plus
        ``eval_expected_cost_std`` when more than one batch was drawn) and the
        dense payload — ``eval_batch_costs``, one batch-mean cost per
        evaluation batch, ``eval_batch_wall_time_s``, what each of those
        batches cost in seconds, and ``eval_trajectory_costs``, shape
        ``(n_batches, batch)``, one total per evaluation trajectory.

        **When the problem declares a box**, three more metrics
        (``eval_max_abs_control``, ``eval_saturation_fraction`` and the
        ``eval_saturation_tolerance`` the second was counted at) and two more
        arrays of shape ``(n_batches, batch)``: ``eval_max_abs_control`` and
        ``eval_control_saturation``, Annex 03 §A.3.1a's retained content. On a
        problem with no box every one of them is **absent** rather than zero —
        a saturated fraction of zero is a claim about a box, and there is none.

    **Why both reductions are computed.** Annex 03 §A.3.1 makes the
    per-trajectory cost required content: §A.3 names evaluation trajectories as
    the within-seed aggregation unit and §A.4's paired procedures have no
    operand without them, so a stored measurement holding only batch means can
    never be re-reduced. The scalar nevertheless keeps ``BATCH_MEAN``:
    the two are *distinct named association orders* with separate
    golden-master heritage, and re-deriving ``eval_expected_cost`` from the
    per-trajectory vector would move every stored number in its last bits for
    nothing. The vector is an addition, never a replacement reduction.
    """
    _, cost = require_linear_quadratic(problem)
    _require_eval_mode(artifact)
    Q = torch.as_tensor(time_invariant_slice(cost.Q))
    R = torch.as_tensor(time_invariant_slice(cost.R))
    box = _box_of(problem)

    batch_costs: list[float] = []
    batch_times: list[float] = []
    trajectory_costs: list[np.ndarray] = []
    max_abs_controls: list[np.ndarray] = []
    saturations: list[np.ndarray] = []
    for initial_state, process_noise, measurement_noise in batches:
        # The online pass has never been timed (Annex 02 §2.2): a study's
        # provenance could report what its models cost to train and nothing at
        # all about what scoring them cost, which for the QP families is the
        # larger of the two. Timed per batch rather than per pass, so the trace
        # can show a first batch that pays for a warm-up the rest do not.
        started = wall_clock()
        policy = artifact.make_policy()
        with torch.no_grad():
            # Evaluation batches are always authored on the torch substrate
            # (see EvaluationProtocol.build_batches), so the rollout is
            # torch-native regardless of the contender's backend.
            X, _, U = cast(
                "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                problem.system.run(
                    policy, initial_state, process_noise, measurement_noise
                ),
            )
            Q_cast = Q.to(dtype=X.dtype, device=X.device)
            R_cast = R.to(dtype=U.dtype, device=U.device)
            batch_cost = total_quadratic_cost(
                Q_cast,
                R_cast,
                X,
                U,
                conventions=cost.conventions,
                reduction=CostReduction.BATCH_MEAN,
            )
            trajectory_cost = total_quadratic_cost(
                Q_cast,
                R_cast,
                X,
                U,
                conventions=cost.conventions,
                reduction=CostReduction.PER_SAMPLE,
            )
            if box is not None:
                # Off the control tensor this rollout already holds, inside the
                # same `no_grad` block: retention costs no extra rollout, which
                # is what made it a payload addition rather than a re-run.
                activity = control_activity(
                    U, u_max=box.u_max, u_min=box.u_min, tolerance=SATURATION_TOLERANCE
                )
                max_abs_controls.append(
                    activity.max_abs.detach().cpu().numpy().astype(np.float64)
                )
                saturations.append(
                    activity.saturated.detach().cpu().numpy().astype(np.float64)
                )
        batch_costs.append(float(batch_cost))
        trajectory_costs.append(
            cast("torch.Tensor", trajectory_cost)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        # After the materialisation above, deliberately: on an accelerator that
        # is where the queued work is waited on, so a time taken before it
        # would be the time to *enqueue* a rollout rather than to run one.
        batch_times.append(wall_clock() - started)

    costs = np.asarray(batch_costs, dtype=np.float64)
    metrics: dict[str, Any] = {"eval_expected_cost": float(costs.mean())}
    if costs.size > 1:
        metrics["eval_expected_cost_std"] = float(costs.std(ddof=1))
    # `np.stack` rather than a ragged object array: the protocol draws batches
    # of one declared size, and a ragged payload would break §A.4's pairing key
    # silently rather than here.
    arrays: dict[str, np.ndarray] = {
        "eval_batch_costs": costs,
        "eval_batch_wall_time_s": np.asarray(batch_times, dtype=np.float64),
        "eval_trajectory_costs": np.stack(trajectory_costs),
    }
    if box is not None:
        arrays["eval_max_abs_control"] = np.stack(max_abs_controls)
        arrays["eval_control_saturation"] = np.stack(saturations)
        # The scalars a gate and an audit read, and the tolerance the fraction
        # was counted at -- stored beside it because a saturated fraction
        # quoted without its tolerance says nothing (Annex 01 §4.1). The
        # maximum, not the mean, for `max_abs_control`: feasibility is a claim
        # about the worst entry, and a mean of per-trajectory maxima would
        # report a policy as feasible on the strength of its quiet trajectories.
        metrics["eval_max_abs_control"] = float(arrays["eval_max_abs_control"].max())
        metrics["eval_saturation_fraction"] = float(
            arrays["eval_control_saturation"].mean()
        )
        metrics["eval_saturation_tolerance"] = SATURATION_TOLERANCE
    logger.info("Online evaluation finished: %s", metrics)
    return metrics, arrays
