"""Does NB04 replay, from a warm store, inside the time bound?

Stage 6 Phase B's checkpoint has two halves and only one of them can be a test.
The refusal half needs nothing and lives in `tests/replay/test_notebook.py`.
This is the other half: *head to tail in under 60 seconds*, which needs the
study's real store — 3.43 h to produce at `publication` — and therefore cannot
be built inside `pytest` any more than the slice check can.

Four things are checked and only the first is a demonstration:

1. the committed notebook executes head to tail, and the elapsed time is
   compared against §7.4's own 60-second gate;
2. **independent recomputation** — the figure the notebook embedded is compared
   byte-for-byte against the artifact in the store, so "a figure appeared" is
   not mistaken for "the study's figure appeared";
3. **end-to-end closure** — every identifier the study materialises must appear
   in the rendered provenance section. A section that named the first model and
   stopped satisfies any assertion about the section existing;
4. **negative control** — every declared gate must carry a verdict, and none of
   them may read as passed without a measured value beside it.

    uv run python tools/probes/nb04_replay_check.py                 # ./store
    uv run python tools/probes/nb04_replay_check.py --store /data/mbl

A timing gate is only as honest as the machine it ran on, so the elapsed time
is reported next to the co-resident load rather than on its own: an earlier
phase of this project measured a 6.7x speed-up that was 3.10x, because
something else was on the machine.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "experiments" / "04_box_constrained_lqr_benchmark.ipynb"

#: Parent §7.4's gate, and Annex 04 §1.1's table: a replay notebook runs in
#: under a minute from a warm store. Cold it includes a run, which for this
#: study at `publication` is 3.43 h.
BUDGET_S = 60.0

sys.path.insert(0, str(ROOT / "src"))


def _outputs(executed: dict) -> str:
    """Every rendered output of the executed notebook, as one string."""
    chunks: list[str] = []
    for cell in executed["cells"]:
        for output in cell.get("outputs", ()):
            data = output.get("data", {})
            chunks.append("".join(data.get("text/markdown", "")))
            chunks.append("".join(data.get("text/plain", "")))
            chunks.append("".join(output.get("text", "")))
            if output.get("output_type") == "error":
                chunks.append(f"{output.get('ename')}: {output.get('evalue')}")
    return "\n".join(chunks)


def _execute(store: Path, destination: Path) -> tuple[int, str, float]:
    environment = dict(os.environ, MBL_STORE=str(store))
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(destination),
            str(NOTEBOOK),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    return (
        completed.returncode,
        completed.stdout + completed.stderr,
        time.monotonic() - started,
    )


def _co_resident() -> str:
    """What else was on the machine. Reported, never acted on."""
    try:
        top = subprocess.run(
            ["ps", "-eo", "pcpu,comm", "--sort=-pcpu"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[1:4]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return "; ".join(line.strip() for line in top)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="store", help="the warm store root")
    parser.add_argument(
        "--budget", type=float, default=BUDGET_S, help="the time bound, seconds"
    )
    args = parser.parse_args()

    store = Path(args.store).resolve()
    if not store.is_dir():
        print(f"FAIL  no store at {store}; this probe replays, it does not produce")
        return 1

    destination = store / "nb04_executed.ipynb"
    print(f"co-resident load before: {_co_resident()}")
    code, log, elapsed = _execute(store, destination)
    if code != 0:
        print(f"FAIL  the notebook did not execute against {store}")
        print(log[-3000:])
        return 1

    executed = json.loads(destination.read_text())
    rendered = _outputs(executed)
    failures: list[str] = []

    # 1 -- the time bound.
    verdict = "PASS" if elapsed < args.budget else "FAIL"
    print(f"{verdict}  head to tail in {elapsed:.1f} s (budget {args.budget:.0f} s)")
    if elapsed >= args.budget:
        failures.append("over the time budget")
    print(f"co-resident load after:  {_co_resident()}")

    from mbl.replay import load_study, survey  # noqa: PLC0415 - after sys.path

    tier = re.search(
        r'TIER = "([a-z]+)"',
        "".join(
            "".join(c["source"]) for c in executed["cells"] if c["cell_type"] == "code"
        ),
    )
    loaded = load_study(
        "box_lqr/depth_scaling", tier=tier.group(1) if tier else "standard"
    )
    report = survey(loaded, store=store)

    # 2 -- the figure it embedded is the one in the store.
    embedded = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", rendered)
    figures = store / "studies" / str(loaded.study.study_id) / "figures"
    stored = figures / "fig_cost_vs_depth.png"
    if embedded is None:
        failures.append("the notebook embedded no figure")
        print("FAIL  no embedded figure")
    elif not stored.is_file():
        failures.append(f"no stored figure at {stored}")
        print(f"FAIL  no stored figure at {stored}")
    else:
        same = base64.b64decode(embedded.group(1)) == stored.read_bytes()
        print(f"{'PASS' if same else 'FAIL'}  the embedded figure is the stored one")
        if not same:
            failures.append("the embedded figure is not the stored artifact")

    # 3 -- every identifier appears in provenance.
    points = loaded.study.materialise()
    absent = [
        p
        for p in points
        if str(p.model_id) not in rendered or str(p.measurement_id) not in rendered
    ]
    print(
        f"{'PASS' if not absent else 'FAIL'}  provenance names all "
        f"{len(points)} points ({len(absent)} absent)"
    )
    if absent:
        failures.append(f"{len(absent)} points missing from provenance")

    # 4 -- every declared gate carries a verdict.
    declared = [gate.kind.value for gate in loaded.study.gates]
    unreported = [name for name in declared if f"`{name}`" not in rendered]
    print(
        f"{'PASS' if not unreported else 'FAIL'}  all {len(declared)} declared "
        f"gate(s) reported ({', '.join(declared)})"
    )
    if unreported:
        failures.append(f"gates not reported: {', '.join(unreported)}")

    print(
        f"\nstudy {loaded.study.id} at {loaded.tier}: {report.points} points, "
        f"{report.measurements_held} measured"
    )
    if failures:
        print("\nFAILED: " + "; ".join(failures))
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
