"""The cost-grid driver's pure logic: job order, quietness, agreement.

The subprocess spawning lives in `tools/run_cost_grid.py`; everything here is
importable and unit-tested without timing anything. The palindrome is the
measurement's own integrity check — *an ordered sweep on a drifting machine
measures the drift* is on this project's record twice, once as a 6.7× speedup
that was really 3.10× — so the halves' disagreement FAILS the run rather than
widening an error bar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spec.errors import SpecificationError

__all__ = [
    "Job",
    "palindrome_order",
    "require_quiet_machine",
    "await_quiet_machine",
    "signed_differences",
    "require_agreeing_halves",
    "merge_halves",
]

#: The timed fields whose two palindrome readings must agree, per phase.
AGREEMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "offline": ("offline_time_s",),
    "online": ("setup_s", "per_step_s"),
}


@dataclass(frozen=True)
class Job:
    """One cell to measure: a (contender, phase) pair with its half tag."""

    contender: str
    phase: str
    half: str  # "forward" | "reverse"


def palindrome_order(contenders: list[str], phases: list[str]) -> list[Job]:
    """A…Z then Z…A, per phase: monotone drift enters the two halves with
    opposite sign, so agreement is a readout that the machine was quiet."""
    forward = [
        Job(contender, phase, "forward") for phase in phases for contender in contenders
    ]
    reverse = [
        Job(contender, phase, "reverse")
        for phase in reversed(phases)
        for contender in reversed(contenders)
    ]
    return forward + reverse


def require_quiet_machine(loadavg_1min: float, *, threshold: float) -> None:
    """Refuse to start on a machine already doing something else.

    Args:
        loadavg_1min: The 1-minute load average (`/proc/loadavg`, first field).
        threshold: The declared ceiling; above it the run is refused rather
            than silently contaminated.
    """
    if loadavg_1min > threshold:
        raise SpecificationError(
            f"1-minute load average is {loadavg_1min:.2f}, above the declared "
            f"quiet-machine threshold {threshold:.2f}; a cost grid measured on "
            "a busy machine files the machine's other work as a result. Wait, "
            "or raise --load-threshold deliberately"
        )


def await_quiet_machine(
    read_load: Callable[[], float],
    sleep: Callable[[float], None],
    *,
    threshold: float,
    timeout_s: float,
    poll_s: float,
    announce: Callable[[str], None] = lambda message: None,
) -> float:
    """Wait for the machine to go quiet, and refuse only if it never does.

    A multi-pass grid is its own worst neighbour: the load average a pass
    leaves behind is the previous pass's, and it decays over minutes, so a
    gate asserted between passes reads the measurement it is protecting.
    Measured 2026-08-13 — pass 1 ended at 5.57 against a threshold of 1.50 and
    refused the remaining four. Waiting is what the operator would have done;
    the refusal survives, at the far end of the timeout.

    Args:
        read_load: Reads the current 1-minute load average.
        sleep: Blocks for the given seconds (injected, so the wait is testable
            without one).
        threshold: The declared quiet ceiling.
        timeout_s: How long to wait in total before refusing. Zero makes this
            exactly `require_quiet_machine`.
        poll_s: The interval between readings.
        announce: Receives one line per reading while waiting.

    Returns:
        The load average that was low enough to start on.

    Raises:
        SpecificationError: If the machine is still busy when the wait runs
            out — naming the last reading and how long was spent waiting.
    """
    waited = 0.0
    load = read_load()
    while load > threshold:
        if waited >= timeout_s:
            raise SpecificationError(
                f"1-minute load average is {load:.2f}, above the declared "
                f"quiet-machine threshold {threshold:.2f}, after waiting "
                f"{waited:.0f} s for it to settle; something else is using "
                "this machine. Wait longer (--settle-timeout-s), or raise "
                "--load-threshold deliberately"
            )
        announce(
            f"load {load:.2f} > {threshold:.2f}; waited {waited:.0f} s of "
            f"{timeout_s:.0f} s"
        )
        sleep(poll_s)
        waited += poll_s
        load = read_load()
    return load


def signed_differences(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[str, float]]:
    """Every cell's SIGNED relative difference, reverse against forward.

    Returns:
        ``(name, difference)`` per (cell, field), where the name reads
        ``contender/phase/field`` and the difference is
        ``(reverse − forward) / max(|forward|, |reverse|)``. Drift over a run
        makes these systematically signed; process-to-process noise does not.
    """
    differences: list[tuple[str, float]] = []
    for forward, reverse in pairs:
        phase = str(forward["phase"])
        for field in AGREEMENT_FIELDS[phase]:
            first, second = float(forward[field]), float(reverse[field])
            scale = max(abs(first), abs(second))
            if scale == 0.0:
                continue
            name = f"{forward['contender']}/{phase}/{field}"
            differences.append((name, (second - first) / scale))
    return differences


def require_agreeing_halves(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    drift_tolerance: float,
    gross_tolerance: float,
) -> float:
    """The palindrome's readout for one pass, asserted over the whole cast.

    A per-cell bound was the first implementation and it was wrong twice
    (Annex 06 §7.4, amended 2026-08-13): it refuses process-to-process noise
    that no number of timed calls can shrink, and it passes the systematic
    5 % slope the palindrome exists to catch. Drift moves the MEDIAN of the
    signed differences; noise leaves it near zero.

    Args:
        pairs: Every (forward, reverse) cell pair measured in this pass.
        drift_tolerance: Bound on |median signed difference| across the cast.
        gross_tolerance: Per-cell bound, kept only for structural failures —
            a stalled core, a swap event — and set far above the platform's
            reproducibility rather than at it.

    Returns:
        The median signed difference, for the record: a pass reports the
        number it was judged on.

    Raises:
        SpecificationError: On systematic drift, naming the median and the
            cast; or on a gross per-cell failure, naming the cell.
    """
    differences = signed_differences(pairs)
    if not differences:
        raise SpecificationError(
            "the palindrome has no comparable readings; a pass that measured "
            "nothing cannot report that the machine was quiet"
        )
    for name, difference in differences:
        if abs(difference) > gross_tolerance:
            raise SpecificationError(
                f"palindrome halves disagree grossly on {name}: "
                f"{difference:+.1%} against a {gross_tolerance:.0%} structural "
                "limit; that is not process noise, and this number would be "
                "whatever the machine was doing instead"
            )
    drift = float(np.median([difference for _, difference in differences]))
    if abs(drift) > drift_tolerance:
        worst = ", ".join(
            f"{name} {difference:+.1%}"
            for name, difference in sorted(differences, key=lambda item: -abs(item[1]))[
                :3
            ]
        )
        raise SpecificationError(
            f"the palindrome's halves drifted: median signed difference "
            f"{drift:+.1%} across {len(differences)} readings, above the "
            f"declared {drift_tolerance:.0%}. The machine changed during the "
            f"pass and the sweep would file that change as a result "
            f"(worst: {worst})"
        )
    return drift


def merge_halves(forward: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    """One cell from its two agreeing readings: timed fields take the MINIMUM
    of the halves (the reading with the least interference — the standard
    benchmarking reduction), distributions and memory take the forward half,
    and both raw readings ride along for the audit."""
    merged = dict(forward)
    for field in AGREEMENT_FIELDS[str(forward["phase"])]:
        merged[field] = min(float(forward[field]), float(reverse[field]))
    merged["forward"] = {
        field: forward[field] for field in AGREEMENT_FIELDS[str(forward["phase"])]
    }
    merged["reverse"] = {
        field: reverse[field] for field in AGREEMENT_FIELDS[str(forward["phase"])]
    }
    return merged
