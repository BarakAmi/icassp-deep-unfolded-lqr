"""Run-reuse guards for notebook execution (T3.g *loading*): reuse a
persisted run's artifacts if a matching one already exists under an
experiments root, otherwise require the caller to explicitly opt in
(``force_run=True``) before running a (potentially slow) simulation — so
re-running a research notebook top to bottom never silently re-triggers a
multi-minute Monte Carlo rollout, and never blocks on stdin.

This is the proven Phase-1F machinery the Stage-S4 `ExperimentCache` (T3.e)
generalizes; it remains the notebook-facing guard for runs that predate the
`Experiment` entity.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..persistence import (
    RunSummary,
    discover_runs,
    list_run_artifacts,
    load_artifact,
    load_run_metadata,
)


def find_existing_run(
    root: Path | str, experiment_name: str, *, expected_signature: str | None = None
) -> RunSummary | None:
    """Most recent persisted run named exactly `experiment_name` (its run_id
    with the `run_YYYYMMDD_HHMMSS_` timestamp prefix stripped -- see
    `persistence.run_naming.generate_run_id`), or None if no such run exists
    yet under `root`.

    Args:
        root: Experiments root directory.
        experiment_name: The bare run name to match (timestamp prefix
            stripped), as `generate_run_id` builds it.
        expected_signature: When given, a name match is only accepted if the
            cached run's own persisted `params["signature"]` (see
            `ProblemSignatureCallback`) equals this value -- Phase 1F: a
            same-named run left over from a *different* problem/controller/
            sampling configuration (e.g. after changing `STATE_DIM` in a
            notebook) must not be silently reused just because its name
            still matches; scanning continues past a mismatch in case an
            older run under the same name still matches (e.g. after
            reverting the config change), rather than giving up outright.
    """
    for run in discover_runs(root):
        # run_id is "run_<8 digits>_<6 digits>_<experiment_name>"; splitting
        # on "_" with maxsplit=3 isolates the name exactly as
        # generate_run_id built it, regardless of underscores within the
        # name itself.
        parts = run.run_id.split("_", 3)
        if not (len(parts) == 4 and parts[0] == "run" and parts[3] == experiment_name):
            continue
        if expected_signature is None:
            return run
        cached_signature = load_run_metadata(run.run_dir)["params"].get("signature")
        if cached_signature == expected_signature:
            return run
        warnings.warn(
            f"Cached run '{run.run_id}' has signature {cached_signature!r}, "
            f"expected {expected_signature!r} -- its problem/controller/sampling "
            "configuration has changed since this run was cached; ignoring it.",
            stacklevel=2,
        )
    return None


def load_run_artifacts(run: RunSummary) -> dict[str, Any]:
    """Every artifact belonging to `run`, loaded eagerly and keyed by its
    bare name (e.g. "trajectory_states", "parameter_P_arr")."""
    return {path.stem: load_artifact(path) for path in list_run_artifacts(run.run_dir)}


def load_or_run(
    root: Path | str,
    experiment_name: str,
    run_fn: Callable[[], Any],
    *,
    force_run: bool = False,
    expected_signature: str | None = None,
) -> dict[str, Any]:
    """Reuse a persisted run's artifacts if one named `experiment_name`
    already exists under `root`; otherwise require `force_run=True` before
    calling `run_fn()` to execute the simulation.

    Args:
        root: Experiments root directory (e.g. "experiments").
        experiment_name: The run's name (see
            `persistence.run_naming.generate_run_id`) -- matched against
            every existing run's bare name (timestamp prefix stripped), so
            re-running this notebook on a different day still finds the
            same persisted experiment.
        run_fn: Executes the (potentially slow) simulation, itself
            responsible for persisting a run named `experiment_name` (e.g.
            by building its own LocalExperimentTracker + Engine and calling
            `.run()`). Called at most once, and only when `force_run=True`.
        force_run: Explicit opt-in to run the (potentially slow) simulation
            when no persisted run is found. Leave `False` by default so a
            fresh clone (no cached `experiments/`) fails loudly with
            instructions rather than silently kicking off a multi-minute
            rollout; set `True` deliberately once you intend to (re-)compute.
        expected_signature: When given, forwarded to `find_existing_run` so a
            same-named cached run is only reused if its persisted signature
            still matches the current problem/controller/sampling
            configuration (Phase 1F) -- see `core.utils.signing.
            compute_run_signature`. A mismatch is treated exactly like "no
            run found": `run_fn()` still requires `force_run=True`.

    Returns:
        Every artifact of the (existing or newly created) run, keyed by its
        bare artifact name.

    Raises:
        RuntimeError: if no matching run exists and `force_run` is `False`,
            or if `run_fn()` completes without producing a run named
            `experiment_name` with a matching signature.
    """
    existing = find_existing_run(
        root, experiment_name, expected_signature=expected_signature
    )
    if existing is not None:
        artifacts = load_run_artifacts(existing)
        if artifacts:
            return artifacts
        # metadata.json exists (e.g. committed to version control) but its
        # heavy artifacts/ directory doesn't (gitignored, so absent on a
        # fresh clone) -- nothing to reuse, fall through to (re-)running.

    if not force_run:
        raise RuntimeError(
            f"No existing run found for '{experiment_name}' under '{root}'. "
            "Pass force_run=True to compute it (this may take a while)."
        )

    run_fn()

    created = find_existing_run(
        root, experiment_name, expected_signature=expected_signature
    )
    if created is None:
        raise RuntimeError(
            f"run_fn() completed but no run named '{experiment_name}' with a "
            f"matching signature was found under '{root}' -- did it persist "
            "via a differently-named tracker, or with a different signature "
            "than expected?"
        )
    return load_run_artifacts(created)
