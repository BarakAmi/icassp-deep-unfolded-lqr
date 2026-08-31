"""Zero-shot OOD evaluation (docs/planning/NB06_OOD_GENERALIZATION_PLAN.md
Sec 3.2): score an already-synthesized artifact against a problem that is
NOT the one it was trained on -- the two-phase lifecycle
(`models.lifecycle.Synthesizer` / `SynthesizedController.make_policy`) was
built for exactly this separation, but nothing in the tree exercised it
before NB06 (`evaluate_synthesized_controller` already takes `artifact` and
`problem` as independent arguments; the gap is orchestration, not a missing
primitive).

`evaluate_under_shift` is a peer of `evaluation.evaluate_synthesized_controller`
built from the SAME cost primitives (`core.kernels.total_quadratic_cost`
under the problem's own declared `cost.conventions`) -- never a parallel
reimplementation, so the OOD-evaluated objective can never drift from the
nominally-scored one (the C2 law). It additionally reports the two
constraint metrics the OOD plan's corrected mandate needs (Sec 2.5):

* **Feasibility audit** (`max_violation`) -- ``max|u| - u_max`` across every
  evaluated entry. Every contender in this project projects or squashes onto
  its box by construction, so this should read at float tolerance for all of
  them; its value is as a falsifiable check (a projection bug, a rehost that
  drops the constraint), not a per-figure metric.
* **Saturation rate** (`saturation_rate`) -- the fraction of entries within
  ``(1 - _SATURATION_TOLERANCE) * u_max`` of the bound. This is the
  informative quantity: does a controller hold its OOD cost down by staying
  in the interior, or by pinning itself to the bound.

`synthesize_nominal_contenders` closes the other half of the gap: it runs
the ordinary (cached, auditable) nominal `Experiment` once, then resolves
and re-synthesizes each contender's recipe a SECOND time, in memory, for the
OOD pass. This double synthesis is a documented cost (Sec 3.2/7.5), not an
oversight -- the cache persists trained weights to safetensors but no loader
yet reconstructs a `SynthesizedController` from a run directory, so an
in-memory artifact is the only way to roll a trained contender out on a
DIFFERENT (perturbed) problem. Callers should call this ONCE per notebook
execution and hold the returned artifacts for the whole OOD sweep, never
re-synthesize per perturbation point.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from .cache import CachePolicy
from .experiment import EvaluationBatch, Experiment, ExperimentReport
from .runner import run_experiment
from ..applications.ood.constraint_shift import CommandRecorder
from ..applications.recipes.base import null_harness
from ..core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from ..core.optimal_control_problem import OptimalControlProblem
from ..models.guards import require_linear_quadratic
from ..models.lifecycle import SynthesizedController

#: How close to `u_max` (as a fraction of `u_max`) counts as "saturated" --
#: the GRU's ``u = u_max * tanh(...)`` can never reach `u_max` EXACTLY, so an
#: exact-equality test would always read zero for it; this small band gives
#: every family a well-defined, comparable saturation rate.
_SATURATION_TOLERANCE = 1e-3


def _conditional_value_at_risk(costs: np.ndarray, *, tail: float) -> float:
    """Mean of the worst `tail` fraction of `costs` (NB06 plan Sec 13.2) --
    the tail statistic that reads "how bad are the bad cases", which the
    median alone cannot show.

    Args:
        costs: Pooled per-trajectory costs, 1-D.
        tail: Fraction defining "worst" (e.g. ``0.10`` for the worst 10%).

    Returns:
        The mean of every entry at or above the ``100 * (1 - tail)``th
        percentile.
    """
    threshold = np.percentile(costs, 100.0 * (1.0 - tail))
    worst = costs[costs >= threshold]
    return float(worst.mean()) if worst.size > 0 else float(costs.max())


@dataclass(frozen=True)
class ShiftMetrics:
    """One (perturbation point, evaluation-batch-set)'s scored result (NB06
    plan Sec 3.2).

    Attributes:
        cost_mean: Mean of the per-batch mean costs -- UNCHANGED from every
            batch's own `core.kernels.CostReduction.BATCH_MEAN` reduction
            (never re-derived from the pooled per-trajectory costs below),
            so this is bit-for-bit the same quantity
            `evaluation.evaluate_synthesized_controller` reports as
            ``eval_expected_cost`` (the C2 law).
        cost_median: Median of the POOLED PER-TRAJECTORY costs -- the
            primary statistic (NB06 plan Sec 2.4/6/11 S3: robust under
            Cauchy, where the mean need not converge). Deliberately NOT the
            median of per-batch means: that statistic is itself
            batch-size-dependent under a heavy-tailed law (a batch mean of
            Cauchy-driven costs does not concentrate as batch size grows),
            so pooling every individual trajectory before taking quantiles
            is what makes this stable across batch geometry.
        cost_q25: 25th percentile of the pooled per-trajectory costs.
        cost_q75: 75th percentile of the pooled per-trajectory costs.
        cost_q90: 90th percentile of the pooled per-trajectory costs -- a
            tail statistic (NB06 plan Sec 13.2): zero-shot robustness is a
            tail property the median alone can hide.
        cost_cvar10: Mean of the worst 10% of pooled per-trajectory costs
            (Sec 13.2's other tail statistic).
        cost_std: Sample standard deviation of the pooled per-trajectory
            costs (``0.0`` when only one trajectory total was drawn).
        max_violation: The feasibility audit -- ``max|u| - u_max`` across
            every evaluated batch; should read at float tolerance for every
            contender in this project (Sec 2.5).
        saturation_rate: The pooled fraction of ``(t, batch, dim)`` control
            entries within `_SATURATION_TOLERANCE` of `u_max`.
        divergence_rate: Fraction of trajectories whose max-over-time state
            norm exceeded the caller's `divergence_reference_norm` (Sec
            13.2); ``0.0`` when that argument was ``None`` (not requested).
        n_batches: Number of evaluation batches this result was computed
            over.
    """

    cost_mean: float
    cost_median: float
    cost_q25: float
    cost_q75: float
    cost_q90: float
    cost_cvar10: float
    cost_std: float
    max_violation: float
    saturation_rate: float
    divergence_rate: float
    n_batches: int


def evaluate_under_shift(  # noqa: PLR0913 -- one evaluation call's full
    # identity: WHAT to roll out (artifact/problem/batches), the audit bound,
    # and three independently-optional reporting toggles (trajectory
    # capture, divergence threshold, the C-blind applied/commanded split) --
    # each a genuinely separate axis of "what this call additionally wants",
    # not a group that collapses into one options object without losing the
    # per-argument defaults every existing call site already relies on.
    artifact: SynthesizedController,
    problem: OptimalControlProblem,
    batches: tuple[EvaluationBatch, ...],
    *,
    u_max: float,
    capture_trajectories: bool = False,
    divergence_reference_norm: float | None = None,
    applied_control_bound: float | None = None,
) -> tuple[ShiftMetrics, dict[str, np.ndarray]]:
    """Roll `artifact`'s policy out on `problem` (which need not be the
    problem it was synthesized against) over every batch in `batches`, and
    score both cost and the constraint/tail metrics.

    Args:
        artifact: The (possibly rehosted) synthesized controller.
        problem: The problem to roll out against -- a perturbed instance for
            a genuine OOD evaluation, or the nominal one (reproducing
            `evaluation.evaluate_synthesized_controller` exactly) as a
            regression anchor.
        batches: The (seeded, shared-across-contenders) evaluation batches
            for this perturbation point.
        u_max: The infinity-norm control bound to audit against. When
            `applied_control_bound` is ``None`` this is also the bound the
            rollout itself is assumed to already respect (every family in
            this project projects/squashes internally); when
            `applied_control_bound` is set, `u_max` is what the feasibility
            audit is measured against on the COMMANDED (pre-clip) signal --
            callers pass the SAME shifted bound as both (NB06 plan Sec 2.6's
            C-blind protocol).
        capture_trajectories: If ``True``, additionally return the stacked
            per-batch state/control trajectories (for the single OOD phase
            portrait, NB06 plan Sec 5.5) -- off by default, since a full
            sweep must not accumulate every batch's raw trajectory in memory.
        divergence_reference_norm: If set, a trajectory whose max-over-time
            state norm exceeds this value counts toward `ShiftMetrics
            .divergence_rate` (NB06 plan Sec 13.2) -- e.g. a small multiple
            of the nominal steady-state state norm. ``None`` (default)
            reports ``divergence_rate=0.0`` and skips the extra reduction.
        applied_control_bound: If set, the policy is wrapped
            (`applications.ood.constraint_shift.CommandRecorder`) so the
            PLANT rolls out on the control clipped to
            ``[-applied_control_bound, applied_control_bound]`` (what is
            actually applied -- cost and saturation are computed from this),
            while `ShiftMetrics.max_violation` is computed from the
            COMMANDED, pre-clip signal against `u_max` instead (what the
            controller actually asked for) -- the two are genuinely
            different quantities under a blind de-rating shift (NB06 plan
            Sec 2.6's C-blind protocol), and conflating them would hide the
            phenomenon the protocol exists to measure. ``None`` (default)
            reproduces today's behavior exactly: the audit reads directly
            off the rolled-out `U`.

    Returns:
        ``(metrics, arrays)``: the `ShiftMetrics`, plus ``{"batch_costs":
        ...}`` (one `CostReduction.BATCH_MEAN` per batch, UNCHANGED),
        ``{"trajectory_costs": ...}`` (every individual trajectory's own
        cost via `CostReduction.PER_SAMPLE`, pooled across all batches --
        what `cost_median`/`cost_q25`/`cost_q75`/`cost_q90`/`cost_cvar10`/
        `cost_std` are computed from), and, if `capture_trajectories`,
        ``{"trajectory_states": (n_batches, B, T+1, n), "trajectory_controls":
        (n_batches, B, T, m)}``.
    """
    _, cost = require_linear_quadratic(problem)
    Q = torch.as_tensor(time_invariant_slice(cost.Q))
    R = torch.as_tensor(time_invariant_slice(cost.R))

    batch_costs: list[float] = []
    per_trajectory_costs: list[np.ndarray] = []
    max_violations: list[float] = []
    saturation_fractions: list[float] = []
    divergence_flags: list[np.ndarray] = []
    captured_states: list[np.ndarray] = []
    captured_controls: list[np.ndarray] = []

    for initial_state, process_noise, measurement_noise in batches:
        policy = artifact.make_policy()
        recorder: CommandRecorder | None = None
        if applied_control_bound is not None:
            recorder = CommandRecorder()
            policy = recorder.wrap(policy, bound=applied_control_bound)
        with torch.no_grad():
            X, _, U = cast(
                "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                problem.system.run(
                    policy, initial_state, process_noise, measurement_noise
                ),
            )
            batch_cost = total_quadratic_cost(
                Q.to(dtype=X.dtype, device=X.device),
                R.to(dtype=U.dtype, device=U.device),
                X,
                U,
                conventions=cost.conventions,
                reduction=CostReduction.BATCH_MEAN,
            )
            trajectory_cost = total_quadratic_cost(
                Q.to(dtype=X.dtype, device=X.device),
                R.to(dtype=U.dtype, device=U.device),
                X,
                U,
                conventions=cost.conventions,
                reduction=CostReduction.PER_SAMPLE,
            )
            abs_u = U.abs()
            if recorder is not None:
                commanded_max = max(
                    float(commanded.abs().max()) for commanded in recorder.commanded
                )
                max_violations.append(commanded_max - u_max)
            else:
                max_violations.append(float((abs_u.max() - u_max).item()))
            saturation_fractions.append(
                float(
                    (abs_u >= (1.0 - _SATURATION_TOLERANCE) * u_max)
                    .float()
                    .mean()
                    .item()
                )
            )
            if divergence_reference_norm is not None:
                max_state_norm = X.norm(dim=-1).amax(dim=1)
                divergence_flags.append(
                    (max_state_norm > divergence_reference_norm).float().cpu().numpy()
                )
        batch_costs.append(float(batch_cost))
        per_trajectory_costs.append(trajectory_cost.detach().cpu().numpy())
        if capture_trajectories:
            captured_states.append(X.detach().cpu().numpy())
            captured_controls.append(U.detach().cpu().numpy())

    costs = np.asarray(batch_costs, dtype=np.float64)
    trajectory_costs = np.concatenate(per_trajectory_costs).astype(np.float64)
    metrics = ShiftMetrics(
        cost_mean=float(costs.mean()),
        cost_median=float(np.median(trajectory_costs)),
        cost_q25=float(np.percentile(trajectory_costs, 25)),
        cost_q75=float(np.percentile(trajectory_costs, 75)),
        cost_q90=float(np.percentile(trajectory_costs, 90)),
        cost_cvar10=_conditional_value_at_risk(trajectory_costs, tail=0.10),
        cost_std=(
            float(trajectory_costs.std(ddof=1)) if trajectory_costs.size > 1 else 0.0
        ),
        max_violation=float(max(max_violations)),
        saturation_rate=float(np.mean(saturation_fractions)),
        divergence_rate=(
            float(np.mean(np.concatenate(divergence_flags)))
            if divergence_flags
            else 0.0
        ),
        n_batches=len(batches),
    )
    arrays: dict[str, np.ndarray] = {
        "batch_costs": costs,
        "trajectory_costs": trajectory_costs,
    }
    if capture_trajectories:
        arrays["trajectory_states"] = np.stack(captured_states)
        arrays["trajectory_controls"] = np.stack(captured_controls)
    return metrics, arrays


def synthesize_nominal_contenders(
    experiment: Experiment, *, root: Path | str, policy: CachePolicy = CachePolicy()
) -> tuple[dict[str, SynthesizedController], ExperimentReport]:
    """Train (or replay from cache) `experiment`'s contenders exactly as any
    other `run_experiment` call, then resolve and re-synthesize each one a
    second time, in memory, for the OOD pass (module docstring).

    Args:
        experiment: The nominal experiment (the SAME declaration a
            non-OOD notebook would run and cache).
        root: The experiments/artifacts root `run_experiment` caches under.
        policy: Per-invocation cache control for the nominal run; defaults
            to ``CachePolicy()`` (``"reuse"``).

    Returns:
        ``(artifacts, nominal_report)``: `artifacts` maps label -> the
        freshly in-memory-synthesized `SynthesizedController`, one per
        `experiment.contenders` entry -- callers hold this dict for the
        whole OOD sweep, it must not be recomputed per perturbation point.
        `nominal_report` is the cached `ExperimentReport` from the ordinary
        (non-OOD) run, e.g. for a level=0/nominal reference point on an OOD
        figure.
    """
    nominal_report = run_experiment(experiment, root=root, policy=policy)
    problem = experiment.problem.build()
    harness = null_harness(experiment.evaluation.batch_spec, experiment.ctx)

    artifacts: dict[str, SynthesizedController] = {}
    for spec in experiment.contenders:
        recipe = spec.resolve()
        synthesizer = recipe.build_synthesizer(problem, harness)
        artifacts[spec.resolved_label] = synthesizer.synthesize(problem, experiment.ctx)
    return artifacts, nominal_report
