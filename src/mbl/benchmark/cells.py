"""One cost-grid cell, measured in the calling process.

The functions here are what `python -m mbl.benchmark.cell` runs after being
spawned fresh; they are importable so the acceptance tests can exercise the
measurement logic in-process at tiny shapes. The process-isolation guarantee
belongs to the CLI wrapper and the driver, not to these functions.

Offline (Phase F doctrine): for a trainable family, the REAL engine runs a
declared handful of epochs and the cell reports the per-epoch **median**
multiplied by the declared epoch count, plus the setup remainder — the
extrapolation the campaign plan validated at 1.6 % error. For an analytic
family the full synthesis is timed whole — and, being milliseconds, is
**repeated** and reduced by a median, with the cold first call kept beside it.
Nothing is published.

Online: the stored frozen artifact is loaded (never trained — an absent model
is refused by name), and the cell reports **setup** (load through first
control), **per-step at batch 1** (warmups then timed calls, the method
ported from the legacy `measure_frozen_inference_benchmark`), and the study's
own evaluation batches as a per-batch wall-time **distribution** through the
production evaluation seam.

Memory: peak RSS via `resource.getrusage` (one syscall, never `tracemalloc`)
plus current RSS from `/proc/self/statm` where "resident now" is the
quantity. The caller snapshots its baseline AFTER importing torch, so the
import's footprint is not billed to the phase.
"""

from __future__ import annotations

import dataclasses
import os
import resource
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..applications.recipes.base import EngineHarness, ModelRecipe, TrainableRecipe
from ..core.profiling import wall_clock
from ..engine.callbacks import METRICS_HISTORY
from ..experiments.bindings import DEFAULT_SPEC_BINDINGS
from ..experiments.evaluation import _require_eval_mode, evaluate_synthesized_controller
from ..persistence.recording_tracker import RecordingExperimentTracker
from ..runner.producer import load_trained_controller, training_batch_spec
from ..runner.seeds import derive_replicate_streams
from ..spec.errors import SpecificationError
from ..spec.loader import load_study
from ..spec.tiers import DEFAULT_TIER_CATALOGUE
from ..store.content_store import ModelStore

__all__ = [
    "CellAddress",
    "OnlineTimingSpec",
    "SynthesisTimingSpec",
    "measure_offline",
    "measure_online",
    "resolve_point",
]


@dataclasses.dataclass(frozen=True)
class CellAddress:
    """Which materialised point a cell measures (the S6 bundle idiom).

    Attributes:
        study_path: The study document.
        tier: The tier name — the catalogue's overlay applies exactly as
            `mbl run` applies it, so the benchmark times what the campaign
            trains.
        contender: The declared label.
        seed: The training replicate.
        axis_filter: Optional ``{axis_path: value}`` narrowing for a swept
            contender (e.g. the nominal depth).
    """

    study_path: Path | str
    tier: str
    contender: str
    seed: int = 0
    axis_filter: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class OnlineTimingSpec:
    """The online cell's timing budget (the S6 bundle idiom: two knobs that
    only ever travel together).

    Attributes:
        warmup_calls: Untimed calls before the clock starts.
        timed_calls: Timed batch-1 policy calls, split across `blocks`.
        blocks: How many independent blocks those calls are divided into. The
            reported per-step cost is the **smallest block median**, not the
            median of everything.

            **Because this machine's GPU is not dedicated.** Under WSL2 the
            card also drives the Windows desktop, so `nvidia-smi` reports
            memory in use and non-zero utilisation with "no running processes
            found": a batch-1 kernel queues behind a compositor this side
            cannot see. Measured, that is not a tail effect -- two of five
            blocks came back at **7x** the true cost while the clock sat at
            2467 MHz and the card at 50 C, so it is contention and not power.

            Contention can only ever make a call slower, never faster, so the
            cheapest block is the closest estimate of the uncontended cost.
            Measured across palindrome halves on that same shared card: the
            minimum block median disagreed by **0.1 %**, the median of block
            medians by 15 %, and the worst block by 589 %.
    """

    warmup_calls: int = 50
    timed_calls: int = 2000
    blocks: int = 5
    min_seconds: float = 3.0
    max_seconds: float = 20.0
    settle_blocks: int = 3
    settle_tolerance: float = 0.02


@dataclasses.dataclass(frozen=True)
class SynthesisTimingSpec:
    """How often a closed-form synthesis is repeated before it is reduced.

    A closed-form synthesis on this campaign's instances takes **2–700 ms** —
    it is by three orders of magnitude the smallest quantity in the grid, and
    it was the only timed quantity here estimated from ONE call while
    per-epoch is a median of five and per-step a median of two thousand.
    Measured 2026-08-13, on a machine with nothing else running: the first
    process of a palindrome read the unconstrained Riccati synthesis at
    47.97 ms and the last read it at 4.07 ms — a 91.5 % disagreement that
    refused the whole run, and a real fact about a cold interpreter rather
    than about the algorithm.

    So the synthesis is repeated. The cell's own first call is the warm-up
    (kept, as `first_synthesis_s`, because the cold cost is real and someone
    will ask), and the repeats run until both a minimum count and a wall-clock
    budget are met — the budget is what keeps a 700 ms SDP from being
    repeated fifty times.

    Attributes:
        min_calls: Timed repeats to run whatever the budget says.
        max_calls: Ceiling, so a microsecond-scale synthesis terminates.
        budget_s: Stop once the timed repeats have cost this much in total.
    """

    min_calls: int = 3
    max_calls: int = 2000
    budget_s: float = 2.0


def _peak_rss_bytes() -> int:
    """The process's high-water resident set, in bytes (Linux: KiB units)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _current_rss_bytes() -> int:
    """The process's resident set right now, from `/proc/self/statm`."""
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def resolve_point(address: CellAddress) -> Any:
    """The one materialised point a cell measures.

    Raises:
        SpecificationError: If no materialised point matches, naming what was
            asked for.
    """
    document = load_study(Path(address.study_path), bindings=DEFAULT_SPEC_BINDINGS)
    study = document.resolve(DEFAULT_TIER_CATALOGUE, address.tier, overrides={}).study
    wanted = dict(address.axis_filter or {})

    def _satisfies(point: Any) -> bool:
        # A contender that does not carry an axis (excluded by applies_to) is
        # invariant to it: the filter is satisfied vacuously, so one --axis
        # narrowing serves a mixed cast of swept and depth-invariant
        # contenders. A carried axis must match exactly.
        return all(
            path not in point.axis_values or point.axis_values[path] == value
            for path, value in wanted.items()
        )

    matches = [
        point
        for point in study.materialise()
        if point.contender.resolved_label == address.contender
        and point.seed == address.seed
        and _satisfies(point)
    ]
    if not matches:
        raise SpecificationError(
            f"no point of {study.id!r} at tier {address.tier!r} matches "
            f"contender {address.contender!r}, seed {address.seed}, axis "
            f"filter {wanted!r}"
        )
    if (
        len({str(point.model_id) for point in matches}) > 1
        or len({str(point.measurement_id) for point in matches}) > 1
    ):
        values = sorted(
            {
                str({k: v for k, v in point.axis_values.items() if k not in wanted})
                for point in matches
            }
        )
        raise SpecificationError(
            f"{len(matches)} points of {study.id!r} match contender "
            f"{address.contender!r}, seed {address.seed}, axis filter "
            f"{wanted!r}; a cell measures ONE point, and picking the first "
            "would pick by declaration order. Narrow the filter; the "
            f"unfiltered axis values are: {values}"
        )
    return matches[0]


def _repeat_synthesis(
    recipe: ModelRecipe,
    problem: Any,
    harness: EngineHarness,
    ctx: Any,
    *,
    timing: SynthesisTimingSpec,
) -> list[float]:
    """Time the whole synthesis again and again, warm.

    Each repeat rebuilds the synthesizer as well as running it: that is what
    the single call this replaces timed, and rebuilding is also what makes a
    repeat a genuine synthesis rather than a second reading of a cached one.
    """
    timings: list[float] = []
    spent = 0.0
    while len(timings) < timing.max_calls:
        started = wall_clock()
        recipe.build_synthesizer(problem, harness).synthesize(problem, ctx)
        duration = wall_clock() - started
        timings.append(duration)
        spent += duration
        if len(timings) >= timing.min_calls and spent >= timing.budget_s:
            break
    return timings


def measure_offline(
    address: CellAddress,
    *,
    bench_epochs: int = 5,
    synthesis: SynthesisTimingSpec | None = None,
    full_training: bool = False,
) -> dict[str, Any]:
    """The offline cell: what one dedicated machine pays to produce the
    artifact, decomposed as setup + per-epoch × declared epochs.

    Returns:
        ``kind`` ("trainable"/"analytic"), ``declared_epochs``,
        ``bench_epochs``, ``per_epoch_s`` (the bench's list),
        ``per_epoch_median_s``, ``setup_s`` (bench wall minus its epochs),
        ``offline_time_s`` (the extrapolated total), ``bench_wall_s``, and
        the two memory numbers.
    """
    point = resolve_point(address)
    recipe = point.contender.resolve()
    problem = point.problem.build()
    ctx = point.training.ctx
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    streams = derive_replicate_streams(point.training.data.seed, point.seed)
    declared_epochs: int | None = None
    if isinstance(recipe, TrainableRecipe):
        declared_epochs = int(recipe.plan.epochs)
        # `full_training` runs the declared budget instead of a bench sample,
        # so the total is TIMED rather than extrapolated -- and the
        # extrapolation is computed anyway, from the same run, so the two
        # routes can be compared instead of one being trusted.
        epochs = declared_epochs if full_training else bench_epochs
        # `recipe` is typed at the abstract `ModelRecipe`, which mypy cannot
        # know is a dataclass -- the same narrowing the producer's override
        # seam documents for the same call.
        recipe = dataclasses.replace(  # type: ignore[type-var]
            recipe, plan=dataclasses.replace(recipe.plan, epochs=epochs)
        )
    harness = EngineHarness(
        batch_spec=training_batch_spec(
            point.training, point.problem, DEFAULT_SPEC_BINDINGS, stream=streams.data
        ),
        tracker=tracker,
        ctx=ctx,
        microbatch=point.training.batch.microbatch,
    )
    torch.manual_seed(streams.weights)
    baseline_rss = _current_rss_bytes()
    started = wall_clock()
    synthesizer = recipe.build_synthesizer(problem, harness)
    synthesizer.synthesize(problem, ctx)
    bench_wall = wall_clock() - started

    result: dict[str, Any] = {
        "phase": "offline",
        "contender": address.contender,
        "seed": address.seed,
        # What was measured, not merely who measured it: without the axis a
        # cell taken at J = 7 and one taken at J = 3 are indistinguishable on
        # disk, and a table pooling them would be undetectably wrong.
        "study": str(address.study_path),
        "tier": address.tier,
        "axis_filter": dict(address.axis_filter or {}),
        "bench_wall_s": bench_wall,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": _peak_rss_bytes(),
        "current_rss_bytes": _current_rss_bytes(),
    }
    history = tracker.frame(METRICS_HISTORY)
    if declared_epochs is None or history is None or history.empty:
        # A closed-form family: the synthesis IS the offline phase, whole --
        # and it is milliseconds, so it is reduced from repeats like every
        # other timed quantity here (see `SynthesisTimingSpec`). The call
        # just made is the warm-up, and it is kept: the cold cost is real,
        # it is what a one-shot deployment pays, and hiding it would answer
        # a question nobody asked with a number nobody could reproduce.
        repeats = _repeat_synthesis(
            recipe, problem, harness, ctx, timing=synthesis or SynthesisTimingSpec()
        )
        warm = float(np.median(repeats))
        result |= {
            "kind": "analytic",
            "declared_epochs": None,
            "bench_epochs": 0,
            "per_epoch_s": [],
            "per_epoch_median_s": None,
            "first_synthesis_s": bench_wall,
            "per_synthesis_s": repeats,
            "synthesis_calls": len(repeats),
            "setup_s": warm,
            "offline_time_s": warm,
            # Re-read: the repeats allocate, and a peak taken before them
            # would understate the phase it labels.
            "peak_rss_bytes": _peak_rss_bytes(),
            "current_rss_bytes": _current_rss_bytes(),
        }
        return result
    per_epoch = [float(value) for value in history["wall_time_s"]]
    median = float(np.median(per_epoch))
    setup = max(0.0, bench_wall - float(np.sum(per_epoch)))
    extrapolated = setup + median * declared_epochs
    result |= {
        "kind": "trainable",
        "declared_epochs": declared_epochs,
        "bench_epochs": len(per_epoch),
        "per_epoch_s": per_epoch,
        "per_epoch_median_s": median,
        "setup_s": setup,
        "full_training": full_training,
        # Both routes, always. Under `full_training` the wall time IS the
        # answer and the extrapolation rides beside it as the thing being
        # validated; otherwise there is no measured total to report and the
        # field says so rather than repeating the estimate under a name that
        # would claim more than it is.
        "extrapolated_total_s": extrapolated,
        "measured_total_s": bench_wall if full_training else None,
        "offline_time_s": bench_wall if full_training else extrapolated,
    }
    return result


def _rebuild_frozen(point: Any, recipe: ModelRecipe, store_root: Path) -> Any:
    """The stored artifact, back as something that can be rolled out —
    the producer's own reuse semantics: a record with weights is a checkpoint
    (rebuild and load), one without is a receipt for a closed-form solve
    (re-derive through the synthesizer, bit-exact)."""
    models = ModelStore(store_root)
    model_id = str(point.model_id)
    if not models.exists(model_id):
        raise SpecificationError(
            f"model {model_id} ({point.contender.resolved_label!r}, seed "
            f"{point.seed}) is not in the store at {store_root}; the online "
            "cell measures a FROZEN artifact and never trains one"
        )
    record = models.get(model_id)
    problem = point.problem.build()
    if record.weights:
        return load_trained_controller(
            recipe.build_controller, problem, point.training.ctx, record
        )
    tracker = RecordingExperimentTracker(frames=())
    streams = derive_replicate_streams(point.training.data.seed, point.seed)
    harness = EngineHarness(
        batch_spec=training_batch_spec(
            point.training, point.problem, DEFAULT_SPEC_BINDINGS, stream=streams.data
        ),
        tracker=tracker,
        ctx=point.training.ctx,
        microbatch=point.training.batch.microbatch,
    )
    torch.manual_seed(streams.weights)
    synthesizer = recipe.build_synthesizer(problem, harness)
    return synthesizer.synthesize(problem, point.training.ctx)


def measure_online(
    address: CellAddress,
    *,
    store_root: Path | str,
    timing: OnlineTimingSpec | None = None,
) -> dict[str, Any]:
    """The online cell: what a fresh machine pays to deploy the artifact.

    Returns:
        ``setup_s`` (artifact load through first control), ``per_step_s``
        (median over `timed_calls` batch-1 policy calls; the p10/p90 ride
        along), ``batched_wall_s`` (one wall time per evaluation batch, the
        distribution §A.5 requires), ``rss_after_load_bytes``,
        ``peak_rss_bytes``, and the counts that make the numbers auditable.
    """
    budget = timing or OnlineTimingSpec()
    point = resolve_point(address)
    recipe = point.contender.resolve()
    baseline_rss = _current_rss_bytes()

    # The BUILD runs outside no_grad, exactly as the producer builds: the
    # convex policy's D17 canary validates its declared solver by comparing
    # GRADIENTS at build time, and a no_grad scope around it turns that
    # validation into a crash. Only the policy CALLS run under no_grad, the
    # way the production evaluation seam runs its rollouts -- a deployed
    # controller records no autograd graph, and timing one that does would
    # bill training machinery to the online cell.
    # A CUDA CALL RETURNS BEFORE ITS WORK IS DONE, so a clock stopped without
    # draining the queue times the launch and not the controller. Measured, that
    # is not a small error: the palindrome halves of one contender disagreed by
    # 61 % and the driver refused them, which is that gate doing exactly what it
    # exists for. On CPU this is a no-op and the numbers are unchanged.
    device = point.evaluation.ctx.torch_device

    def drain() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = wall_clock()
    artifact = _rebuild_frozen(point, recipe, Path(store_root))
    _require_eval_mode(artifact)
    policy = artifact.make_policy()
    # THROUGH THE CONTEXT'S OWN CONSTRUCTOR, which carries the device as well as
    # the dtype. Built with `torch.zeros(..., dtype=...)` this landed on the CPU
    # whatever the study declared, so the online cell could never measure a CUDA
    # study: the first policy call died on a device mismatch inside the
    # contender. Table 1 exists because Figure 1 is a CPU study; the first
    # attempt at the same table for a GPU one is what found this.
    probe_state = point.evaluation.ctx.zeros((1, point.problem.state_dim))
    with torch.no_grad():
        first = policy(0, probe_state)
    drain()
    setup_s = wall_clock() - started
    rss_after_load = _current_rss_bytes()
    # `.detach().cpu()` before numpy sees it: a control produced on the card is
    # a CUDA tensor, and the check that guards against a broken policy would
    # itself have raised. The same reason as the probe above -- this path had
    # only ever been run against a CPU study.
    probe_control = first.detach().cpu() if isinstance(first, torch.Tensor) else first
    if not bool(np.isfinite(np.asarray(probe_control)).all()):
        raise SpecificationError(
            f"{address.contender!r}'s frozen policy emitted a non-finite control "
            "at the probe state; a benchmark of a broken policy would "
            "file machine noise as a result"
        )

    with torch.no_grad():
        for _ in range(budget.warmup_calls):
            policy(0, probe_state)
        # Drain the warm-up before the first lap, or its tail is billed to it.
        drain()
        per_block = max(1, budget.timed_calls // budget.blocks)
        # KEEP SAMPLING UNTIL A QUIET WINDOW IS FOUND, within a budget.
        # `blocks` alone is not enough: measured, a contention episode on this
        # shared card lasts ABOUT A SECOND, and five blocks at the degraded rate
        # span less than half of one -- so a whole cell can sit inside a single
        # episode and every block comes back equally slow. Sampling past it is
        # the only way to see the uncontended cost; the budget is what stops a
        # permanently busy host from running forever.
        deadline = wall_clock() + budget.max_seconds
        block_laps: list[list[float]] = []
        best = float("inf")
        settled = 0
        while True:
            laps: list[float] = []
            for _ in range(per_block):
                tick = wall_clock()
                policy(0, probe_state)
                drain()
                laps.append(wall_clock() - tick)
            block_laps.append(laps)
            median_lap = float(np.median(laps))
            if median_lap < best * (1.0 - budget.settle_tolerance):
                best, settled = median_lap, 0
            else:
                best, settled = min(best, median_lap), settled + 1
            # A MINIMUM WALL-CLOCK WINDOW, not just a block count. Five blocks
            # of a fast policy span ~50 ms, and a contention episode here lasts
            # about a second -- so a cell can sit wholly inside one, see five
            # uniformly slow blocks, and "settle" on the contended cost. The
            # window has to be long enough to contain both states.
            elapsed = wall_clock() - (deadline - budget.max_seconds)
            enough = len(block_laps) >= budget.blocks and elapsed >= budget.min_seconds
            if enough and (settled >= budget.settle_blocks or wall_clock() >= deadline):
                break
        # THE CHEAPEST BLOCK, not the median of everything: see
        # `OnlineTimingSpec.blocks`. The block medians are kept so that the
        # contention this discards is visible in the record rather than
        # silently dropped -- a cell whose blocks disagree measured the host.
        block_medians = [float(np.median(block)) for block in block_laps]
        laps = block_laps[int(np.argmin(block_medians))]

    protocol = point.evaluation.protocol
    batches = protocol.build_batches(point.evaluation.ctx)
    _, arrays = evaluate_synthesized_controller(
        artifact, point.evaluation.problem.build(), batches
    )
    return {
        "phase": "online",
        "contender": address.contender,
        "seed": address.seed,
        "model_id": str(point.model_id),
        # As offline: the cell states what it measured, so a table cannot
        # pool two depths, two tiers or two documents without being refused.
        "study": str(address.study_path),
        "tier": address.tier,
        "axis_filter": dict(address.axis_filter or {}),
        "setup_s": setup_s,
        "per_step_s": float(np.median(laps)),
        "per_step_p10_s": float(np.percentile(laps, 10.0)),
        "per_step_p90_s": float(np.percentile(laps, 90.0)),
        "per_step_calls": per_block,
        "per_step_blocks": block_medians,
        "per_step_block_count": len(block_medians),
        "per_step_batch_size": 1,
        "batched_wall_s": [float(v) for v in arrays["eval_batch_wall_time_s"]],
        "batched_batch_size": int(protocol.batch_spec.batch_size),
        "baseline_rss_bytes": baseline_rss,
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
