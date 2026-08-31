"""Disposition-agnostic artifact replay (NB03 blueprint v2 SS4, closing the
cache-hit/fresh asymmetry): a `run_experiment` cache HIT loads every one of
a contender's persisted artifacts eagerly (`TrackerBackedExperimentCache
.get` -> `_load_artifacts` scans the run's whole ``artifacts/`` folder), so
``ContenderResult.arrays`` already carries the full per-epoch
``metrics_history`` DataFrame `MetricsHistoryCallback` wrote. A FRESH
computation's `ContenderResult.arrays` carries only the online-evaluation
payload (`eval_batch_costs`) -- `MetricsHistoryCallback` (and every other
callback-written artifact) lands on disk under ``result.run_dir`` but is
never read back into the in-memory report. A notebook plotting learning
curves straight from ``result.arrays`` would therefore show curves on
cache-hit sessions and nothing on the very session that computed them.

`load_contender_artifacts`/`load_training_history` close that gap by
normalizing BOTH paths to the same disposition-agnostic view, reading the
run directory (the persisted source of truth) only for whatever a HIT
already supplied in memory. No edit to the locked `experiments` layer --
this reads the run directory the same way `persistence.run_reader` and the
cache itself do.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch

from ..engine.callbacks import format_structured_log_row
from ..experiments import ContenderResult
from ..models.analytic.riccati import LocalCostToGoModel, evaluate_local_cost_to_go
from ..models.constrained.cocp import COCPController
from ..persistence import list_run_artifacts, load_artifact

#: A `metrics_history` artifact with fewer rows than this carries no visible
#: trend -- e.g. a single-epoch `AnalyticalStrategy` run (Riccati, the fixed
#: unfolded configuration) always logs exactly one row -- so it is not
#: treated as a "training curve" by `load_training_history`.
_MIN_CURVE_LENGTH = 2


def load_contender_artifacts(result: ContenderResult) -> dict[str, Any]:
    """Every artifact belonging to `result`'s producing run, keyed by bare
    artifact name, identically whether `result` came from a cache HIT
    (already fully populated in ``result.arrays``) or a FRESH computation
    (only the online-evaluation payload is in memory; everything else,
    including ``metrics_history``, lives on disk under ``result.run_dir``
    until read here).

    Args:
        result: One contender's result from an `ExperimentReport`.

    Returns:
        bare artifact name -> loaded value. `result.arrays` entries take
        precedence (they are already the loaded objects, so nothing is
        re-read from disk for a HIT); any artifact present on disk but
        absent from `result.arrays` (the FRESH case) is loaded and added.
        Empty if `result.run_dir` is ``None`` and `result.arrays` is empty.
    """
    combined: dict[str, Any] = dict(result.arrays)
    if result.run_dir is not None:
        run_dir = Path(result.run_dir)
        for path in list_run_artifacts(run_dir):
            if path.stem not in combined:
                combined[path.stem] = load_artifact(path)
    return combined


def load_training_history(
    result: ContenderResult,
) -> pd.DataFrame | None:
    """This contender's per-epoch training curve (the ``metrics_history``
    artifact `MetricsHistoryCallback` writes), identically whether `result`
    was freshly computed or served from cache.

    Args:
        result: One contender's result from an `ExperimentReport`.

    Returns:
        The per-epoch history `DataFrame` (columns include ``"epoch"`` plus
        whatever the contender's `TrainingStrategy` returns, e.g.
        ``"loss"``), or ``None`` when no genuine training curve exists --
        either no ``metrics_history`` artifact at all, or one with fewer
        than `_MIN_CURVE_LENGTH` rows (a single-epoch analytical run, e.g.
        Sim-Riccati or Standard-GD, has nothing to plot a trend from).
    """
    history = load_contender_artifacts(result).get("metrics_history")
    if not isinstance(history, pd.DataFrame) or len(history) < _MIN_CURVE_LENGTH:
        return None
    return history


def _record_unfolding_depth(record: Any) -> int | None:
    """The unrolled depth ``J`` persisted with one training-log row, or
    ``None`` for a history written before the depth was recorded (or a run with
    no unrolling) -- so a replayed header matches the live one exactly when the
    depth is present, and is simply omitted when it is not."""
    depth = record.get("unfolding_depth")
    return None if depth is None or pd.isna(depth) else int(depth)


def render_training_log_replay(
    result: ContenderResult, *, stream: TextIO | None = None
) -> str:
    """Reconstruct the human-readable training trace
    `engine.callbacks.StructuredTrainingLogCallback` narrates live, from
    `result`'s persisted ``training_log_history`` artifact, and **flush it to
    stdout** -- the Cache Replay half of Zero-Black-Boxes transparency:
    loading a contender from cache (zero compute) must never render as a
    silent, blank notebook cell.

    Directive 2 (replay enforcement): this function itself PRINTS the trace
    (with ``flush=True``) rather than merely returning a string a caller might
    forget to display, so a cached notebook run always shows the historical
    trace dynamically. The text is also returned (for tests/further handling);
    call it bare -- ``render_training_log_replay(result)`` -- not wrapped in
    ``print(...)``, which would double it.

    Disposition-agnostic exactly like `load_training_history`: the artifact is
    read identically whether `result` came from a fresh computation or a cache
    HIT. Every row is rendered through the SAME `format_structured_log_row`
    the live callback uses, so a replayed trace (multi-line, one learned
    parameter per line, full ``P`` matrix included) is byte-identical to the
    live one modulo the `logging` timestamp prefix.

    Args:
        result: One contender's result from an `ExperimentReport`.
        stream: Where to flush the trace; defaults to ``sys.stdout``.

    Returns:
        A multi-line string: one identity header (label, family,
        disposition, run directory), followed by the per-epoch rows, or a
        one-line notice in place of the rows when this contender never
        trained (e.g. Sim-Riccati, a direct closed-form solve with no
        `StructuredTrainingLogCallback` and thus no artifact to replay).
    """
    header = (
        f"[Cache Replay] {result.label} (family={result.family}, "
        f"disposition={result.disposition}, run_dir={result.run_dir})"
    )
    history = load_contender_artifacts(result).get("training_log_history")
    if not isinstance(history, pd.DataFrame) or history.empty:
        text = (
            f"{header}\n  (no structured training log for this contender -- "
            "it never trained.)"
        )
    else:
        rows = (
            format_structured_log_row(
                int(record["epoch"]),
                float(record["cost"]),
                str(record["learned_parameters"]),
                unfolding_depth=_record_unfolding_depth(record),
            )
            for record in history.to_dict("records")
        )
        text = "\n".join([header, *rows])
    # Flush the trace to stdout ourselves (directive 2) via the stream's own
    # write/flush -- NOT the print() builtin, which library code below the
    # adapter layer must not call (architecture boundary T1.5.e).
    out = stream if stream is not None else sys.stdout
    out.write(text + "\n")
    out.flush()
    return text


@dataclass(frozen=True)
class UnfoldingLandscapeInputs:
    """Everything `viz.adapters.notebook.render_unfolding_landscape_animations`
    needs to reconstruct one unfolded contender's own inner-loop iterate
    trajectory at a frozen instant, read entirely from PERSISTED artifacts
    (never a live trained module) -- disposition-agnostic exactly like
    `load_training_history`: identical whether `result` came from a cache
    HIT or a fresh computation.

    Attributes:
        x_star: The frozen state the landscape/replay are evaluated at,
            shape ``(n,)`` -- ``trajectory_states[sample_index, t_star]``.
        step_size: The contender's trained per-iteration step size, shape
            ``(J, m)`` (`engine.callbacks.ParameterSnapshotCallback`'s
            ``"parameter_step_size"`` artifact, via
            `applications.recipes.unfolded.unfolded_convergence_parameters`).
        P_own: The contender's OWN cost-to-go matrix, used by its own inner
            update -- the learned ``"parameter_riccati_matrix"`` artifact if
            this contender learns one (``unfolded_alpha_p``), else the
            ``"parameter_P"`` fallback (the closed-form Riccati ``P``, for a
            contender that only learns the step size, e.g.
            ``unfolded_alpha``). May differ from the TRUE Riccati P the
            landscape backdrop is drawn from -- that divergence is exactly
            what the landscape figure visualizes.
        A: State transition matrix AT THE LANDSCAPE'S `t_star` (NB05 plan
            Sec 6.4), shape ``(n, n)`` -- the per-step-correct slice under
            LTV; identical to the problem's single matrix under LTI. Always
            2D, so every existing renderer (which only ever consumed a
            single frozen-instant matrix) is unaffected regardless of
            whether the producing artifact was LTI or LTV.
        B: Control input matrix AT THE LANDSCAPE'S `t_star`, shape ``(n, m)``
            (see `A`).
        R: Control cost matrix, shape ``(m, m)``.
    """

    x_star: np.ndarray
    step_size: np.ndarray
    P_own: np.ndarray
    A: np.ndarray
    B: np.ndarray
    R: np.ndarray


def unfolding_landscape_inputs(
    result: ContenderResult, *, t_star: int, sample_index: int = 0
) -> UnfoldingLandscapeInputs:
    """Bridge one unfolded contender's persisted artifacts into
    `render_unfolding_landscape_animations`'s input shape: `x_star` comes
    from the already-persisted ``trajectory_states`` artifact
    (`engine.callbacks.TrajectoryLoggingCallback`, unconditionally attached
    to every recipe -- no fresh rollout is performed here); the trained
    parameters come from ``ParameterSnapshotCallback``'s artifacts. Reads
    via `load_contender_artifacts`, so this is disposition-agnostic (a
    cache HIT and a fresh computation read identically).

    Args:
        result: One unfolded-family contender's result from an
            `ExperimentReport` (e.g. ``report.results["unfolded_alpha_p"]``).
        t_star: The frozen horizon instant to slice `x_star` at.
        sample_index: Which batch element of the persisted evaluation
            rollout to freeze the state from.

    Returns:
        The `UnfoldingLandscapeInputs` bundle.

    Raises:
        KeyError: If `result`'s artifacts are missing any of
            ``"trajectory_states"``, ``"parameter_step_size"``,
            ``"parameter_A"``, ``"parameter_B"``, ``"parameter_R"``, or both
            ``"parameter_riccati_matrix"``/``"parameter_P"`` -- i.e.
            `result` isn't an unfolded-family contender (`unfolded`/
            `unfolded_warmstart`), whose `ParameterSnapshotCallback` (the
            default ``artifact_prefix="parameter"``, see
            `applications.recipes.unfolded.unfolded_convergence_parameters`)
            names every artifact with that prefix.
        ValueError: If the persisted ``"parameter_riccati_matrix"`` artifact
            is not a single ``(n, n)`` matrix -- i.e. `result` is one of the
            NB05 P-resolution extension's two iteration-varying contenders
            (docs/planning/03_studies/nb05_ltv_box_constrained/ltv_box_constrained_lqr_benchmark.md Sec 14, rungs
            R1.5/R2), whose learned P has shape ``(J, n, n)``.
            `OneStepCostModel`/`compute_gradient_descent_path` below assume
            ONE fixed cost-to-go matrix for the whole inner loop; silently
            taking e.g. ``P_own[0]`` would draw a landscape path the
            deployed controller never actually walks (the same class of
            defect as NB04's unprojected-replay bug), so this raises
            instead of guessing -- a caller wanting to visualize one of
            these two contenders must explicitly pick a single iteration's
            matrix itself, not get one implicitly.
    """
    artifacts = load_contender_artifacts(result)
    states = np.asarray(artifacts["trajectory_states"])
    x_star = states[sample_index, t_star] if states.ndim == 3 else states[t_star]
    P_own = (
        artifacts["parameter_riccati_matrix"]
        if "parameter_riccati_matrix" in artifacts
        else artifacts["parameter_P"]
    )
    if np.asarray(P_own).ndim != 2:
        raise ValueError(
            "unfolding_landscape_inputs: this contender's learned P has "
            f"shape {np.asarray(P_own).shape}, not a single (n, n) matrix -- "
            "it varies across unfolding iterations (NB05 plan Sec 14, "
            "rungs R1.5/R2), which this iteration-agnostic replay does not "
            "support. Exclude this contender from the landscape figure "
            "rather than guessing which iteration's matrix to show."
        )
    # NB05 plan Sec 6.4: prefer the full per-step "A_t"/"B_t" stacks (present
    # on any run persisted after the LTV densification fix) sliced at
    # `t_star` -- the per-step-correct dynamics under LTV -- and fall back to
    # the single "A"/"B" matrix (every artifact persisted before that fix,
    # and every LTI run, where the two coincide) otherwise. Additive: no
    # existing cached artifact loses the ability to build these inputs.
    A = (
        artifacts["parameter_A_t"][t_star]
        if "parameter_A_t" in artifacts
        else artifacts["parameter_A"]
    )
    B = (
        artifacts["parameter_B_t"][t_star]
        if "parameter_B_t" in artifacts
        else artifacts["parameter_B"]
    )
    return UnfoldingLandscapeInputs(
        x_star=np.asarray(x_star),
        step_size=np.asarray(artifacts["parameter_step_size"]),
        P_own=np.asarray(P_own),
        A=np.asarray(A),
        B=np.asarray(B),
        R=np.asarray(artifacts["parameter_R"]),
    )


@dataclass(frozen=True)
class COCPReferencePoint:
    """COCP's realized control at a frozen state, evaluated on a caller-
    supplied cost-to-go bowl (NB04 COCP/viz refinement plan Sec 2.2/3.4) --
    NEVER COCP's own learned cost-to-go, so the result is directly
    comparable to another marker (e.g. the Riccati optimum) scored on the
    exact same bowl.

    Attributes:
        point: COCP's solved control ``u`` at the frozen state, shape ``(m,)``.
        value: the caller's bowl evaluated at `point`.
    """

    point: np.ndarray
    value: float


def cocp_reference_point(
    result: ContenderResult, x_star: np.ndarray, *, local_model: LocalCostToGoModel
) -> COCPReferencePoint:
    """Solve a persisted COCP contender's one-step QP at a frozen state,
    and score the result on `local_model` -- the SAME bowl a Riccati-optimum
    marker would be scored on, never COCP's own learned ``(P_sqrt, q)``
    cost-to-go, so the two markers are directly comparable on one figure.

    Reads COCP's `engine.callbacks.ParameterSnapshotCallback` artifacts
    (`applications.recipes.cocp.cocp_convergence_parameters`'s
    ``"parameter_P_sqrt"``/``"parameter_q"``/``"parameter_A"``/
    ``"parameter_B"``/``"parameter_R"``/``"parameter_u_max"``) via
    `load_contender_artifacts`, disposition-agnostic exactly like
    `unfolding_landscape_inputs`, and re-solves the QP via
    `models.constrained.cocp.COCPController.build_qp_layer` -- public
    exactly for this "only have saved (A, B, R, u_max, P_sqrt, q) values"
    use case. Builds the layer with the DEFAULT `COCPSolverSpec` (this
    project's long-standing DIFFCP/SCS choice): a one-off, single-state
    solve for a static figure has no need for MOREAU's training-loop speed,
    and re-running this project's solver-resolution/canary machinery here
    would duplicate what `applications.recipes.cocp.COCPRecipe` already
    does once at training time.

    Args:
        result: COCP's contender result from an `ExperimentReport`
            (e.g. ``report.results["cocp"]``).
        x_star: the frozen state to solve the QP at, shape ``(n,)`` --
            typically the same state a Riccati-optimum marker was scored at.
        local_model: the cost-to-go bowl to score COCP's solved control
            against (e.g. the landscape backdrop's own
            `models.analytic.riccati.LocalCostToGoModel`).

    Returns:
        The `COCPReferencePoint` bundle.

    Raises:
        KeyError: If `result`'s artifacts are missing any of COCP's
            snapshotted parameter names -- i.e. `result` isn't a COCP-family
            contender.
    """
    artifacts = load_contender_artifacts(result)
    P_sqrt = np.asarray(artifacts["parameter_P_sqrt"])
    q = np.asarray(artifacts["parameter_q"])
    A = np.asarray(artifacts["parameter_A"])
    B = np.asarray(artifacts["parameter_B"])
    R = np.asarray(artifacts["parameter_R"])
    u_max = float(np.asarray(artifacts["parameter_u_max"]))

    layer = COCPController.build_qp_layer(A, B, R, u_max)
    dtype = torch.float64
    y = torch.as_tensor(x_star, dtype=dtype).unsqueeze(0)  # (1, n)
    x = y.unsqueeze(-1)  # (1, n, 1) -- matches COCPController.get_control_policy
    P_sqrt_batch = torch.as_tensor(P_sqrt, dtype=dtype).unsqueeze(0)  # (1, n, n)
    q_batch = torch.as_tensor(q, dtype=dtype).unsqueeze(0)  # (1, n)
    with torch.no_grad():
        (u,) = layer(x, P_sqrt_batch, q_batch)
    point = u.squeeze(-1).squeeze(0).numpy()  # (m,)

    value = float(evaluate_local_cost_to_go(x_star, point[None, :], local_model)[0])
    return COCPReferencePoint(point=point, value=value)
