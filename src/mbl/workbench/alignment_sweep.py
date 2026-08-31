"""The LTV "alignment" sweep (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md
Sec 0.2-0.3/7): drives NB05's experiment across several named
(regime, period_or_block, variation_strength) points -- the distinct-slice-
count ``D`` axis every contender's representational-alignment thesis is
plotted against -- at ONE fixed unfolding depth, mirroring
`workbench.depth_sweep`'s own "runs things, returns data, displays nothing"
contract but along a genuinely different axis: on `depth_sweep`'s axis
(unfolding depth ``K``), Truncated-Riccati's cost is axis-INVARIANT (the
same closed-form policy regardless of ``K``); on THIS axis every contender's
cost varies (a different ``D`` is a different LTV problem entirely), so
there is no baseline/curve split to make here -- every contender is
axis-dependent, and the reference floors (`applications.studies
.nb05_ltv_box_constrained.compute_ltv_floors`) are axis-dependent too, since
they are also problem-instance-specific.

This module is NB05-specific (unlike the three-notebook-shared
`depth_sweep`), so it does not generalize `run_depth_sweep_study`'s core --
forcing this axis's genuinely different shape (no depth-invariant baseline
concept) onto that generic function would only add branches nothing else
needs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..applications.ltv_factories import LTVLQRProblemFactory, LTVRegime
from ..applications.studies import (
    LTVFloors,
    NB05Config,
    compute_ltv_floors,
    nb05_ltv_experiment,
)
from ..experiments import CachePolicy, ExperimentReport, run_experiment
from ..models.guards import require_linear_quadratic


@dataclass(frozen=True)
class LTVAlignmentPoint:
    """One named point on the alignment sweep's axis: a specific
    (regime, period_or_block, variation_strength) combination -- e.g.
    "periodic p=4, eps=0.4" -- whose realized distinct-slice count ``D`` the
    sweep computes and reports alongside every contender's cost there.

    Attributes:
        label: Human-readable point identity (x-axis tick / legend entry),
            e.g. ``"periodic (p=4)"``.
        regime: Which time-variation regime this point draws.
        period_or_block: The period/block length; ``None`` only valid for
            `LTVRegime.FULLY_VARYING` (`LTVLQRProblemFactory`'s own rule).
        variation_strength: The perturbation strength ``epsilon`` at this
            point; ``0.0`` reproduces the NB04 (D=1) anchor regardless of
            `regime`/`period_or_block` (Sec 2.3's degeneracy law).
    """

    label: str
    regime: LTVRegime
    period_or_block: int | None
    variation_strength: float


@dataclass(frozen=True)
class LTVAlignmentSweepResult:
    """The full alignment sweep, ready for the alignment-curve and
    regime-comparison figures (`viz.adapters.notebook.render_alignment_curve`
    / `render_regime_comparison`).

    Attributes:
        points: The swept points, in the order evaluated.
        distinct_slice_counts: Each point's realized ``D``, same order as
            `points` -- the alignment curve's natural x-axis.
        costs: contender label -> cost per point (same order as `points`),
            for EVERY contender (unlike `workbench.depth_sweep
            .DepthSweepResult`, there is no depth-invariant-baseline split
            on this axis -- see module docstring).
        cost_stds: contender label -> standard error per point (`None` at a
            point evaluated with a single batch, i.e. `NB05Config
            .n_eval_batches == 1` there) -- the crossover estimate's own
            uncertainty input (`workbench.analysis.estimate_crossover`).
        floors: This point's `LTVFloors`, same order as `points` -- computed
            fresh per point (the problem itself changes point to point, so
            the floors are never shared across points the way NB03/NB04's
            own depth-invariant floor is shared across a K-sweep).
        reports: point.label -> `ExperimentReport`, for drill-down.
    """

    points: tuple[LTVAlignmentPoint, ...]
    distinct_slice_counts: tuple[int, ...]
    costs: Mapping[str, tuple[float, ...]]
    cost_stds: Mapping[str, tuple[float | None, ...]]
    floors: tuple[LTVFloors, ...]
    reports: Mapping[str, ExperimentReport]


def run_ltv_alignment_sweep_study(
    base_config: NB05Config,
    points: Sequence[LTVAlignmentPoint],
    *,
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
    compute_box_floor: bool = True,
) -> LTVAlignmentSweepResult:
    """Run `base_config` once per point in `points` -- only `regime`/
    `period_or_block`/`variation_strength` vary; `num_unfolding_iterations`
    and every other setting stay fixed at `base_config`'s own values, unlike
    `workbench.depth_sweep`'s axis (there, depth varies and the LTV knobs
    stay fixed) -- through the per-contender cache, computing each point's
    reference floors alongside its contenders' evaluated costs.

    Args:
        base_config: The NB05 configuration every point is drawn from.
        points: The named (regime, period_or_block, variation_strength)
            points to sweep, e.g. an NB04 (D=1) anchor plus one point per
            regime at a shared `variation_strength`.
        root: The experiments/artifacts root `run_experiment` caches under.
        policy: Per-invocation cache control; defaults to ``CachePolicy()``
            (``"reuse"``).
        compute_box_floor: Forwarded to `compute_ltv_floors` at every point;
            ``True`` by default. Set ``False`` to skip the (repeated,
            point-by-point) L-BFGS-B solves when only Floor A is needed.

    Returns:
        The aggregated `LTVAlignmentSweepResult`.

    Raises:
        ValueError: If `points` is empty, or two points share a `label`.
    """
    if not points:
        raise ValueError("points must be non-empty.")
    labels = [point.label for point in points]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Point labels must be unique, got {labels}.")

    reports: dict[str, ExperimentReport] = {}
    floors_list: list[LTVFloors] = []
    distinct_slice_counts: list[int] = []
    for point in points:
        cfg = replace(
            base_config,
            regime=point.regime,
            period_or_block=point.period_or_block,
            variation_strength=point.variation_strength,
        )
        report = run_experiment(nb05_ltv_experiment(cfg), root=root, policy=policy)
        reports[point.label] = report

        ltv_factory = LTVLQRProblemFactory(
            state_dim=cfg.state_dim,
            control_dim=cfg.control_dim,
            horizon=cfg.horizon,
            seed=cfg.seed,
            regime=point.regime,
            period_or_block=point.period_or_block,
            variation_strength=point.variation_strength,
            perturb_B=cfg.perturb_B,
            u_max=cfg.u_max,
        )
        problem = ltv_factory.build()
        # The REALIZED distinct-slice count of the BUILT system, not
        # `ltv_factory.distinct_slice_count()`'s structural D from the
        # schedule alone: at variation_strength == 0.0 (or any accidental
        # coincidence of unique slices), the degeneracy law collapses the
        # system to a single 2D matrix regardless of `period_or_block`, and
        # the alignment curve's x-axis must reflect what was actually BUILT
        # -- e.g. the NB04 (eps=0) anchor point must land at D=1, never at
        # whatever period_or_block happened to be configured alongside it.
        system, _ = require_linear_quadratic(problem)
        A_array = system.A_t.array
        realized_d = A_array.shape[0] if A_array.ndim == 3 else 1
        distinct_slice_counts.append(realized_d)
        floors_list.append(
            compute_ltv_floors(
                problem,
                process_noise_std=cfg.process_noise_std,
                initial_state_std=cfg.initial_state_std,
                u_max=cfg.u_max,
                compute_box_floor=compute_box_floor,
            )
        )

    all_labels = sorted(reports[points[0].label].results)
    costs = {
        label: tuple(
            reports[point.label].metric(label, "eval_expected_cost") for point in points
        )
        for label in all_labels
    }
    cost_stds = {
        label: tuple(
            reports[point.label].results[label].metrics.get("eval_expected_cost_std")
            for point in points
        )
        for label in all_labels
    }

    return LTVAlignmentSweepResult(
        points=tuple(points),
        distinct_slice_counts=tuple(distinct_slice_counts),
        costs=costs,
        cost_stds=cost_stds,
        floors=tuple(floors_list),
        reports=reports,
    )
