"""Figure 4's cost-grid driver: every cell in its own process, in palindrome
order, behind a quiet-machine gate.

Usage::

    uv run python tools/run_cost_grid.py \
        --study studies/icassp/fig1_depth.toml --tier publication \
        --contenders truncated_riccati standard_pgd ... \
        --axis "contenders.*.config.num_iterations=7" \
        --out store/benchmarks/fig4_cost_grid

The driver spawns ``python -m mbl.benchmark.cell`` per (contender, phase,
half) — fresh interpreter, so no import, allocator or Riccati result is
shared — then requires the palindrome halves to agree within the declared
tolerance (a figure whose halves disagree measured the machine, not the
controllers) and emits one tidy parquet of merged cells plus the raw JSONs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mbl.benchmark.driver import (  # noqa: E402
    await_quiet_machine,
    merge_halves,
    palindrome_order,
    require_agreeing_halves,
)


def _loadavg() -> float:
    with open("/proc/loadavg") as handle:
        return float(handle.read().split()[0])


def _spawn(job, arguments, out_dir: Path) -> Path:
    out = (
        out_dir
        / "cells"
        / f"{job.contender}_{job.phase}_{job.half}_pass{arguments.pass_index}.json"
    )
    command = [
        sys.executable,
        "-m",
        "mbl.benchmark.cell",
        "--study",
        arguments.study,
        "--tier",
        arguments.tier,
        "--contender",
        job.contender,
        "--seed",
        str(arguments.seed),
        "--phase",
        job.phase,
        "--store",
        arguments.store,
        "--out",
        str(out),
        "--bench-epochs",
        str(arguments.bench_epochs),
        "--warmup-calls",
        str(arguments.warmup_calls),
        "--timed-calls",
        str(arguments.timed_calls),
    ]
    for pair in arguments.axis:
        command += ["--axis", pair]
    subprocess.run(command, check=True, cwd=REPO)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--contenders", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--store", default="store")
    parser.add_argument("--out", required=True)
    parser.add_argument("--axis", action="append", default=[])
    parser.add_argument("--bench-epochs", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=50)
    parser.add_argument("--timed-calls", type=int, default=2000)
    parser.add_argument("--load-threshold", type=float, default=1.5)
    parser.add_argument(
        "--settle-timeout-s",
        type=float,
        default=1800.0,
        help=(
            "how long to wait for the load average to fall below the "
            "threshold before refusing. A pass leaves its own load behind, so "
            "0 (refuse at once) cannot complete a multi-pass run"
        ),
    )
    parser.add_argument("--settle-poll-s", type=float, default=20.0)
    parser.add_argument(
        "--drift-tolerance",
        type=float,
        default=0.10,
        help=(
            "bound on the median SIGNED difference between the palindrome's "
            "halves, across the whole cast — the readout that the machine did "
            "not change during the pass"
        ),
    )
    parser.add_argument(
        "--gross-tolerance",
        type=float,
        default=0.60,
        help=(
            "per-cell structural limit: a stalled core or a swap event, not "
            "the platform's own process-to-process spread (measured at 33 %% "
            "on a sub-millisecond cell)"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "how many full palindrome passes to run. Every pass is subject to "
            "the same halves-agreement refusal and ALL of its samples are "
            "kept: the table's statistics want the distribution, and the "
            "single-pass merge was only ever a two-sample minimum."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["offline", "online"],
        choices=["offline", "online"],
    )
    arguments = parser.parse_args()

    out_dir = Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = palindrome_order(list(arguments.contenders), list(arguments.phases))
    (out_dir / "jobs.json").write_text(
        json.dumps([job.__dict__ for job in jobs], indent=1)
    )

    merged_rows = []
    for pass_index in range(1, arguments.repeats + 1):
        arguments.pass_index = pass_index
        # The quiet gate is re-asserted per pass, not once at the start: a
        # five-pass run is an hour long and the machine it began on is not
        # necessarily the machine it is on now. It WAITS rather than refusing
        # outright, because between passes the load it reads is the previous
        # pass's own, still decaying.
        load = await_quiet_machine(
            _loadavg,
            time.sleep,
            threshold=arguments.load_threshold,
            timeout_s=arguments.settle_timeout_s,
            poll_s=arguments.settle_poll_s,
            announce=lambda line: print(f"pass {pass_index}: {line}", flush=True),
        )
        print(f"pass {pass_index}: machine quiet (load {load:.2f})", flush=True)
        produced: dict[tuple[str, str, str], Path] = {}
        for index, job in enumerate(jobs, 1):
            print(
                f"[pass {pass_index}/{arguments.repeats}]"
                f"[{index}/{len(jobs)}] {job.contender} {job.phase} ({job.half})"
            )
            produced[(job.contender, job.phase, job.half)] = _spawn(
                job, arguments, out_dir
            )
        pairs = []
        for contender in arguments.contenders:
            for phase in arguments.phases:
                forward = json.loads(
                    produced[(contender, phase, "forward")].read_text()
                )
                reverse = json.loads(
                    produced[(contender, phase, "reverse")].read_text()
                )
                pairs.append((forward, reverse))
        # Per pass, and over the whole cast: drift is systematic, so it moves
        # the median of the signed differences, while the process-to-process
        # noise of a sub-millisecond cell does not (Annex 06 §7.4).
        drift = require_agreeing_halves(
            pairs,
            drift_tolerance=arguments.drift_tolerance,
            gross_tolerance=arguments.gross_tolerance,
        )
        for forward, reverse in pairs:
            row = merge_halves(forward, reverse)
            row["pass"] = pass_index
            row["pass_drift"] = drift
            merged_rows.append(row)
        print(
            f"pass {pass_index}: halves agree, median signed difference {drift:+.2%}",
            flush=True,
        )

    frame = pd.DataFrame(merged_rows)
    frame.to_parquet(out_dir / "cost_grid.parquet")
    print(
        f"{arguments.repeats} pass(es); {len(merged_rows)} rows "
        f"({len(merged_rows) // max(1, arguments.repeats)} cells each) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
