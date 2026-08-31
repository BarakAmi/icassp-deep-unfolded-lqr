"""The fresh-process entry point for one cost-grid cell.

Spawned by `tools/run_cost_grid.py` as ``python -m mbl.benchmark.cell`` so
every (contender, phase) pair pays its own imports, its own allocator and its
own Riccati solve — the own-dedicated-machine doctrine made literal. Writes
exactly one JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cells import (
    CellAddress,
    OnlineTimingSpec,
    SynthesisTimingSpec,
    measure_offline,
    measure_online,
)


def _axis_filter(pairs: list[str]) -> dict[str, Any]:
    """``path=value`` pairs; values parse as int, then float, else string."""
    out: dict[str, Any] = {}
    for pair in pairs:
        path, _, raw = pair.partition("=")
        if not path or not raw:
            raise SystemExit(f"--axis expects path=value, got {pair!r}")
        value: Any
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        out[path] = value
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--contender", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", choices=("offline", "online"), required=True)
    parser.add_argument("--store", default="store")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bench-epochs", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=50)
    parser.add_argument("--timed-calls", type=int, default=2000)
    parser.add_argument(
        "--synthesis-budget-s",
        type=float,
        default=2.0,
        help=(
            "how long the repeated closed-form syntheses may cost in total "
            "before the median is taken (a minimum of three repeats runs "
            "regardless)"
        ),
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help=(
            "run the declared epoch budget instead of a bench sample, so the "
            "offline total and the offline peak memory are MEASURED rather "
            "than extrapolated from five epochs"
        ),
    )
    parser.add_argument(
        "--axis", action="append", default=[], help="axis_path=value narrowing"
    )
    arguments = parser.parse_args(argv)

    address = CellAddress(
        study_path=arguments.study,
        tier=arguments.tier,
        contender=arguments.contender,
        seed=arguments.seed,
        axis_filter=_axis_filter(arguments.axis),
    )
    if arguments.phase == "offline":
        cell = measure_offline(
            address,
            bench_epochs=arguments.bench_epochs,
            synthesis=SynthesisTimingSpec(budget_s=arguments.synthesis_budget_s),
            full_training=arguments.full_training,
        )
    else:
        cell = measure_online(
            address,
            store_root=Path(arguments.store),
            timing=OnlineTimingSpec(
                warmup_calls=arguments.warmup_calls,
                timed_calls=arguments.timed_calls,
            ),
        )
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cell, indent=1))


if __name__ == "__main__":
    main()
