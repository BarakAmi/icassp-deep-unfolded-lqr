"""Depth-sweep orchestration for the "Cost vs. Unfolding Depth" figure (NB03
blueprint v2 SS5, closing Hazard 3 -- the missing orchestrator; generalized
for NB04, docs/planning/03_studies/nb04_box_constrained/box_constrained_lqr_benchmark.md Sec 3.2, and for NB05,
docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 6.5): a single workbench
function drives the per-contender cache across a sequence of unfolding
depths ``K`` and returns one frozen result the viz adapter consumes
directly -- the notebook cell is one line, with zero `for`-loops of its own
(the established `run_initializer_study`/`run_topology_study` workbench
contract: runs things, returns data, displays nothing).

`run_depth_sweep_study` is the generic core, parameterized by a per-depth
`Experiment` builder and which contender labels are depth-INVARIANT
baselines -- it knows nothing about NB03, NB04, or NB05 specifically.
`run_unfolding_depth_sweep_study` (NB03), `run_box_constrained_depth_sweep_study`
(NB04), and `run_ltv_depth_sweep_study` (NB05) are thin, notebook-facing
wrappers over that one core, so the aggregation/caching logic exists exactly
once regardless of how many study modules eventually sweep a depth axis.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..applications.studies import (
    NB03Config,
    NB04Config,
    NB05Config,
    nb03_unfolding_experiment,
    nb04_box_constrained_experiment,
    nb05_ltv_experiment,
)
from ..experiments import CachePolicy, Experiment, ExperimentReport, run_experiment

#: NB03 contender labels whose cost does NOT depend on the unfolding depth
#: `K` (the Riccati rollout baseline never unrolls) -- every other label in
#: the declaration is a K-DEPENDENT (iterative) curve.
BASELINE_LABELS = frozenset({"sim_riccati"})

#: NB04's analogue of `BASELINE_LABELS`: the saturated-Riccati baseline
#: (`truncated_riccati`) never unrolls, and neither does COCP (`cocp`) -- its
#: one-step convex-QP policy has no unfolding depth at all, so it is trained
#: exactly ONCE and its cost is K-invariant (NB04 COCP refinement plan Sec
#: 3.1 lever 1/Sec 3.3: this is also what makes it render as a flat
#: reference line with no renderer change, and what lets the per-contender
#: cache serve it across the whole depth sweep). COCP-LB (`cocp_lower_bound`)
#: joins them for the same reason -- an `AnalyticRecipe` with no unfolding
#: depth of its own (reference-bounds plan Sec 1.4). Every other NB04
#: contender label is a K-DEPENDENT curve.
NB04_BASELINE_LABELS = frozenset({"truncated_riccati", "cocp", "cocp_lower_bound"})

#: NB05's analogue of `BASELINE_LABELS`: the saturated-Riccati baseline
#: (`truncated_riccati`) never unrolls, exactly as in NB04. NB05 drops COCP/
#: COCP-LB entirely (NB05 plan Sec 0.4/4.4 -- both are structurally LTI), so
#: this set has only the one member; every other NB05 contender label is a
#: K-DEPENDENT curve.
NB05_BASELINE_LABELS = frozenset({"truncated_riccati"})


@dataclass(frozen=True)
class DepthSweepResult:
    """The full "Cost vs. Unfolding Depth" study, ready for
    `viz.plots.plot_cost_vs_unfolding_depth` (via the Tier-4
    `render_cost_vs_iterations` adapter).

    Attributes:
        k_values: The unfolding depths swept, in order.
        curves: label -> cost per `k_values` entry (same length/order), one
            entry for every K-DEPENDENT contender.
        reference_lines: label -> constant cost, one entry for every
            K-INDEPENDENT contender (the sweep's own `baseline_labels`,
            e.g. `BASELINE_LABELS` or `NB04_BASELINE_LABELS`) -- read from
            the FIRST depth's report (its cost cannot change with `K` by
            construction).
        reports: The full `ExperimentReport` per depth, for drill-down.
        dispositions: Per-depth, per-contender cache disposition
            (``"fresh"``/``"hit"``/``"recompute"``) -- the cache-reuse audit
            trail.
    """

    k_values: tuple[int, ...]
    curves: Mapping[str, tuple[float, ...]]
    reference_lines: Mapping[str, float]
    reports: Mapping[int, ExperimentReport]
    dispositions: Mapping[int, Mapping[str, str]]


def run_depth_sweep_study(
    experiment_at_depth: Callable[[int], Experiment],
    depths: Sequence[int],
    *,
    baseline_labels: frozenset[str],
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
) -> DepthSweepResult:
    """The generic depth-sweep core (NB04 plan Sec 3.2): run one `Experiment`
    per depth in `depths` -- built fresh from `experiment_at_depth`, which
    closes over whichever study's own config and `num_unfolding_iterations`
    replacement it needs -- through the per-contender cache, and aggregate
    every K-dependent contender's ``eval_expected_cost`` into one
    cost-vs-depth curve. Knows nothing about NB03 or NB04 specifically: it is
    parameterized entirely by the per-depth builder and which labels are
    depth-INVARIANT baselines.

    Every depth is (by construction of a correctly-written
    `experiment_at_depth`) a DISTINCT experiment signature
    (``num_unfolding_iterations`` participates in every unfolded contender's
    own ``get_signature``), so repeated sweeps over the same `depths` after
    the first compute are served entirely from cache under the default
    ``CachePolicy("reuse")`` -- `DepthSweepResult.dispositions` makes this
    auditable rather than invisible.

    Note: this returns ONLY the baselines already present among the
    experiment's own contenders (`baseline_labels`). A closed-form reference
    computed independently of `run_experiment` (e.g. NB03's Theo-Riccati) is
    a SEPARATE concern the caller merges into `DepthSweepResult
    .reference_lines` at render time -- keeping this orchestrator's one
    responsibility (drive the depth-indexed experiment cache) undiluted.

    Args:
        experiment_at_depth: Builds the `Experiment` for one depth ``K``
            (typically ``lambda k: some_experiment(dataclasses.replace(
            base_config, num_unfolding_iterations=k))``).
        depths: The unfolding depths ``K`` to sweep, e.g. ``[2, 4, 8, 16]``.
        baseline_labels: Contender labels whose cost does NOT depend on
            ``K`` -- everything else in the experiment is treated as a
            K-dependent curve.
        root: The experiments/artifacts root `run_experiment` caches under.
        policy: Per-invocation cache control; defaults to ``CachePolicy()``
            (``"reuse"``).

    Returns:
        The aggregated `DepthSweepResult`.

    Raises:
        ValueError: If `depths` is empty.
    """
    if not depths:
        raise ValueError("depths must be non-empty.")

    reports: dict[int, ExperimentReport] = {}
    dispositions: dict[int, dict[str, str]] = {}
    for k in depths:
        report = run_experiment(experiment_at_depth(k), root=root, policy=policy)
        reports[k] = report
        dispositions[k] = {
            label: result.disposition for label, result in report.results.items()
        }

    all_labels = set(reports[depths[0]].results)
    curve_labels = sorted(all_labels - baseline_labels)
    present_baseline_labels = sorted(all_labels & baseline_labels)
    curves = {
        label: tuple(reports[k].metric(label, "eval_expected_cost") for k in depths)
        for label in curve_labels
    }
    reference_lines = {
        label: reports[depths[0]].metric(label, "eval_expected_cost")
        for label in present_baseline_labels
    }

    return DepthSweepResult(
        k_values=tuple(depths),
        curves=curves,
        reference_lines=reference_lines,
        reports=reports,
        dispositions=dispositions,
    )


def run_unfolding_depth_sweep_study(
    base_config: NB03Config,
    depths: Sequence[int],
    *,
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
) -> DepthSweepResult:
    """NB03's depth sweep: a thin wrapper over `run_depth_sweep_study`
    (identical public signature/behavior to before the generalization --
    see `test_depth_sweep.py`), fixing the per-depth builder to
    `nb03_unfolding_experiment` and the baseline set to `BASELINE_LABELS`
    (Sim-Riccati's Monte-Carlo rollout cost; the closed-form Theo-Riccati
    baseline is a separate, purely analytical computation the notebook
    merges in at render time -- see `run_depth_sweep_study`'s docstring).

    Args:
        base_config: The NB03 configuration every depth is drawn from; only
            ``num_unfolding_iterations`` varies per depth (via
            `dataclasses.replace`) -- every other setting (dimensions,
            horizon, seeds, per-family plans) stays fixed across the sweep.
        depths: The unfolding depths ``K`` to sweep, e.g. ``[2, 4, 8, 16]``.
        root: The experiments/artifacts root `run_experiment` caches under.
        policy: Per-invocation cache control; defaults to ``CachePolicy()``
            (``"reuse"``).

    Returns:
        The aggregated `DepthSweepResult`.

    Raises:
        ValueError: If `depths` is empty.
    """
    return run_depth_sweep_study(
        lambda k: nb03_unfolding_experiment(
            replace(base_config, num_unfolding_iterations=k)
        ),
        depths,
        baseline_labels=BASELINE_LABELS,
        root=root,
        policy=policy,
    )


def run_box_constrained_depth_sweep_study(
    base_config: NB04Config,
    depths: Sequence[int],
    *,
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
) -> DepthSweepResult:
    """NB04's depth sweep (docs/planning/03_studies/nb04_box_constrained/box_constrained_lqr_benchmark.md Sec
    3.2): the box-constrained analogue of `run_unfolding_depth_sweep_study`,
    over the SAME generic `run_depth_sweep_study` core, fixing the per-depth
    builder to `nb04_box_constrained_experiment` and the baseline set to
    `NB04_BASELINE_LABELS` (Truncated-Riccati's Monte-Carlo rollout cost --
    unlike NB03's Sim-Riccati, it has no closed-form sibling: the saturated
    policy is nonlinear, so only Monte-Carlo estimation applies).

    Args:
        base_config: The NB04 configuration every depth is drawn from; only
            ``num_unfolding_iterations`` varies per depth -- every other
            setting (dimensions, horizon, `u_max`, seeds, per-family plans)
            stays fixed across the sweep.
        depths: The unfolding depths ``K`` to sweep.
        root: The experiments/artifacts root `run_experiment` caches under.
        policy: Per-invocation cache control; defaults to ``CachePolicy()``
            (``"reuse"``).

    Returns:
        The aggregated `DepthSweepResult`.

    Raises:
        ValueError: If `depths` is empty.
    """
    return run_depth_sweep_study(
        lambda k: nb04_box_constrained_experiment(
            replace(base_config, num_unfolding_iterations=k)
        ),
        depths,
        baseline_labels=NB04_BASELINE_LABELS,
        root=root,
        policy=policy,
    )


def run_ltv_depth_sweep_study(
    base_config: NB05Config,
    depths: Sequence[int],
    *,
    root: Path | str,
    policy: CachePolicy = CachePolicy(),
) -> DepthSweepResult:
    """NB05's depth sweep (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md
    Sec 6.5): the LTV analogue of `run_box_constrained_depth_sweep_study`,
    over the SAME generic `run_depth_sweep_study` core, fixing the per-depth
    builder to `nb05_ltv_experiment` and the baseline set to
    `NB05_BASELINE_LABELS` (Truncated-Riccati's Monte-Carlo rollout cost --
    as in NB04, no closed-form sibling: the saturated policy is nonlinear).

    Args:
        base_config: The NB05 configuration every depth is drawn from; only
            ``num_unfolding_iterations`` varies per depth -- every other
            setting (dimensions, horizon, `u_max`, the LTV regime/
            `period_or_block`/`variation_strength`/`perturb_B`, seeds,
            per-family plans) stays fixed across the sweep, so the sweep is
            run once PER regime/variation-strength point the notebook wants
            on its alignment curve (NB05 plan Sec 7's new figure), not swept
            jointly with depth.
        depths: The unfolding depths ``K`` to sweep.
        root: The experiments/artifacts root `run_experiment` caches under.
        policy: Per-invocation cache control; defaults to ``CachePolicy()``
            (``"reuse"``).

    Returns:
        The aggregated `DepthSweepResult`.

    Raises:
        ValueError: If `depths` is empty.
    """
    return run_depth_sweep_study(
        lambda k: nb05_ltv_experiment(replace(base_config, num_unfolding_iterations=k)),
        depths,
        baseline_labels=NB05_BASELINE_LABELS,
        root=root,
        policy=policy,
    )
