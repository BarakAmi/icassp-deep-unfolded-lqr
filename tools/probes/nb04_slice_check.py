"""The end-to-end slice check: run NB04, analyse it, and disbelieve the result.

Everything the vertical slice built, exercised on the tracked study through the
shipped entry points, and then checked by something that is not the code under
test. Four things happen and only the first is a demonstration:

1. `mbl run` and `mbl analyse` at a chosen tier, into a throwaway store;
2. **one row is recomputed by hand** from the stored per-trajectory costs, and
   the difference from what the analysis emitted is printed. Independent
   recomputation is the only check here that can catch an aggregation that is
   plausible and wrong;
3. **negative control** — the analysis is re-run with `models/` deleted, and
   must produce the byte-identical table. An analysis that reads a model is one
   that cannot survive the deletion the whole design exists to permit;
4. **negative control** — a `smoke` store, whose measurements carry the
   `axis_subset` stamp, must be REFUSED. A gate tested only in the passing
   direction is a gate nobody has tested.

    uv run python tools/probes/nb04_slice_check.py --tier standard
    uv run python tools/probes/nb04_slice_check.py --tier standard --cheap

`--cheap` cuts the epoch counts and the batch to the smallest values that keep
all 27 points and their full evaluation protocol, which turns a four-minute run
into about fifteen seconds. It changes what the models ARE, so the costs it
prints are not science — every check above is about machinery, and each of them
can fail just as well on a badly-trained model.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
NB04_STUDY = REPO / "studies" / "box_lqr" / "depth_scaling.toml"
ANALYSIS_ID = "cost_by_depth"

#: Effort overrides that keep every point and every evaluation batch. They may
#: only touch `PERMITTED_TIER_PATHS`; anything else is content and the grammar
#: refuses it, which is the property that makes this switch safe to offer.
CHEAP = (
    "training.plan.epochs=2",
    "contenders.*.config.plan.epochs=2",
    "contenders.*.config.schedule.warmup_epochs_per_layer=1",
    "contenders.*.config.schedule.refinement_epochs=1",
    "training.batch.effective_size=128",
)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def resolve(tier: str, store: Path, overrides: tuple[str, ...]) -> Any:
    from mbl.cli.run import parse_overrides, plan_run

    resolved, preview = plan_run(
        NB04_STUDY,
        store=store,
        tier=tier,
        catalogue=None,
        overrides=parse_overrides(list(overrides)),
    )
    print(preview.render(previewed=True))
    return resolved


def recompute_one_row(store: Path, study: Any, table: Any) -> None:
    """Aggregate one contender at one depth by hand, from the stored samples.

    Deliberately not through `analysis/cost_vs_axis.py`: a recomputation that
    called the function under test would be a tautology. It reads the same
    `samples.parquet` the analysis reads, finds the measurement by re-deriving
    its identifier from the study, and takes the mean itself.
    """
    import numpy as np

    from mbl.store.content_store import MeasurementStore

    depth_path = "contenders.*.config.num_iterations"
    points = {
        (p.contender.resolved_label, p.axis_values.get(depth_path)): p
        for p in study.materialise()
    }
    target = ("unfolded_alpha_p", 20)
    point = points[target]
    samples = MeasurementStore(store).get(point.measurement_id).samples
    if samples is None:
        raise SystemExit("the measurement carries no per-trajectory samples")
    by_hand = float(np.mean(np.asarray(samples["trajectory_cost"], dtype=np.float64)))

    row = table[(table["contender"] == target[0]) & (table["axis_value"] == target[1])]
    if len(row) != 1:
        raise SystemExit(f"expected one row for {target}, found {len(row)}")
    emitted = float(row["aggregate"].iloc[0])
    print(f"  contender                {target[0]} at J={target[1]}")
    print(f"  trajectories             {len(samples)}")
    print(f"  n_trajectories in table  {int(row['n_trajectories'].iloc[0])}")
    print(f"  emitted by the analysis  {emitted!r}")
    print(f"  recomputed by hand       {by_hand!r}")
    print(f"  delta                    {abs(emitted - by_hand):.3e}")
    if abs(emitted - by_hand) > 1e-12:
        raise SystemExit("RECOMPUTATION DISAGREES")
    print("  OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="standard")
    parser.add_argument("--cheap", action="store_true")
    parser.add_argument("--keep", action="store_true", help="do not delete the store")
    args = parser.parse_args()

    from mbl.analysis.runner import run_analyses
    from mbl.cli.run import execute_run
    from mbl.spec.errors import SpecificationError
    from mbl.store.study_artifacts import StudyArtifactStore

    root = Path(tempfile.mkdtemp(prefix="mbl-slice-check-"))
    store = root / "store"
    overrides = CHEAP if args.cheap else ()
    try:
        banner(
            f"1. run + analyse at tier {args.tier}{' (cheap)' if args.cheap else ''}"
        )
        resolved = resolve(args.tier, store, overrides)
        started = time.perf_counter()
        outcome = execute_run(resolved, store=store, tier=args.tier)
        run_s = time.perf_counter() - started
        print(f"{outcome.render(previewed=False)}   [{run_s:.2f} s]")

        started = time.perf_counter()
        analyses = run_analyses(resolved.study, store=store)
        print(
            f"analysed {analyses[0].analysis_id}: {analyses[0].rows} rows"
            f"   [{time.perf_counter() - started:.2f} s]"
        )

        study_id = str(resolved.study.study_id)
        artifacts = StudyArtifactStore(store)
        stored = artifacts.get_analysis(study_id, ANALYSIS_ID)
        print()
        print(stored.table.to_string(index=False))
        print()
        for key, value in stored.sidecar.items():
            print(f"  {key}: {value}")

        banner("2. independent recomputation of one row")
        recompute_one_row(store, resolved.study, stored.table)

        banner("3. negative control -- the analysis must not read a model")
        before = (
            artifacts.analyses_root(study_id) / f"{ANALYSIS_ID}.parquet"
        ).read_bytes()
        shutil.rmtree(store / "models")
        print(f"  deleted {store / 'models'}")
        run_analyses(resolved.study, store=store)
        after = (
            artifacts.analyses_root(study_id) / f"{ANALYSIS_ID}.parquet"
        ).read_bytes()
        print(
            f"  re-ran with no models on disk; table byte-identical: {before == after}"
        )
        if before != after:
            raise SystemExit("THE TABLE CHANGED WITH THE MODELS DELETED")

        banner("4. negative control -- a subsetted `smoke` store must be REFUSED")
        smoke_store = root / "smoke-store"
        smoke = resolve("smoke", smoke_store, overrides)
        execute_run(smoke, store=smoke_store, tier="smoke")
        try:
            run_analyses(smoke.study, store=smoke_store)
        except SpecificationError as error:
            print(f"  REFUSED, as required:\n    {error}")
        else:
            raise SystemExit("A SUBSETTED SMOKE STORE WAS ANALYSED")

        banner("all four checks passed")
        return 0
    finally:
        if args.keep:
            print(f"\nstore kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
