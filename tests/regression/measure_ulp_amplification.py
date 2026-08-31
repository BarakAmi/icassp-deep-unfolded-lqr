"""Measure how close each golden array comes to breaking its own contract when
the input moves by one ULP.

Not a test -- a tool, deliberately named so pytest does not collect it, because
one run re-executes every scenario four times. It exists because the question
"can this quantity be frozen at all?" has a cheap measured answer, and guessing
at it has cost four debugging sessions.

Run it before freezing any new quantity:

    uv run python tests/regression/measure_ulp_amplification.py [--write]

Full rationale, the classification rule and the obligations that follow:
`docs/methods/cross_machine_numerical_reproducibility.md`.

**The method.** Move EVERY entry of the process noise by one ULP of the
channel's declared precision, in independently-drawn directions -- which is what
two differently-built BLAS libraries do to each other -- and record, per frozen
array, the worst

    |fresh - golden| / (atol + rtol * |golden|)

under that channel's own tolerance. A value of 1 is exactly the contract; above
1 the array cannot meet it. Reporting this *violation ratio* rather than a bare
relative movement matters, because `assert_allclose` compares against
`atol + rtol*|desired|`: a near-zero quantity such as `w_mean` moves enormously
in relative terms and not at all in the terms it is actually judged by.

Reproducing another machine's arithmetic is neither possible nor necessary; what
matters is whether the computation amplifies a last-bit difference, and that is
a property of the computation alone.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Run as a script from anywhere: the repo root has to be importable before
# `tests.regression` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression import scenarios  # noqa: E402
from tests.regression.test_golden_master import (  # noqa: E402
    TOLERANCE_RECORD_KEY,
    TOLERANCES,
    Tolerance,
)

#: Sign patterns for the perturbation. Every entry is moved, so these seed
#: *which way* each moves rather than which entry is touched.
#:
#: Perturbing a single entry was the first version of this tool, and it
#: understates badly: two hosts disagree in the last bit of every entry at once.
#: Measured on `cold_jacobi__X_final_meansq`, one entry moved it by 5.5e-7 while
#: the real cross-runner disagreement was 2.5e-5 -- a factor of 46, which is the
#: accumulation a whole-array perturbation predicts over 409,600 entries and a
#: single-entry probe cannot see. That understatement shipped once and reddened
#: CI on the very next pull request.
SIGN_SEEDS = (0, 1, 2)


def _perturbing_sampler(seed: int, dtype: np.dtype) -> Callable[..., Any]:
    """`make_gaussian_batch_sampler`, with every entry of `w` moved one ULP.

    Patched at the *scenario* level rather than inside any solver, so the
    perturbation enters exactly where a cross-machine difference would: in the
    data the pipeline consumes. Directions are drawn from `seed`, because a
    uniform direction is a systematic bias rather than the independent last-bit
    disagreement two BLAS builds actually produce.
    """
    original = scenarios.make_gaussian_batch_sampler

    def patched(*args: Any, **kwargs: Any) -> Any:
        bundle = original(*args, **kwargs)
        sample = bundle.sample

        def perturbed_sample() -> Any:
            x0, w, v = sample()
            perturbed = np.array(w, copy=True, dtype=dtype)
            rng = np.random.default_rng(seed)
            towards = np.where(
                rng.integers(0, 2, size=perturbed.shape).astype(bool),
                dtype.type(np.inf),
                dtype.type(-np.inf),
            )
            return x0, np.nextafter(perturbed, towards), v

        # `SamplerBundle` is frozen, so the perturbed closure is swapped in by
        # rebuilding it rather than by assignment.
        return dataclasses.replace(bundle, sample=perturbed_sample)

    return patched


def _violation_ratio(fresh: Any, golden: Any, tolerance: Tolerance) -> float:
    """Worst `|delta| / (atol + rtol*|golden|)`: 1.0 is exactly the contract."""
    a = np.asarray(fresh, np.float64)
    b = np.asarray(golden, np.float64)
    if a.shape != b.shape:
        return float("inf")
    allowed = tolerance.atol + tolerance.rtol * np.abs(b)
    return float(np.max(np.abs(a - b) / allowed))


def measure(runner: Callable[[], tuple[Any, ...]], dtype_name: str) -> dict[str, float]:
    """Worst violation ratio per frozen array, over every sign pattern."""
    dtype = np.dtype(dtype_name)
    tolerance = TOLERANCES[dtype_name]
    base = runner()[0]

    worst: dict[str, float] = dict.fromkeys(base, 0.0)
    original = scenarios.make_gaussian_batch_sampler
    for seed in SIGN_SEEDS:
        scenarios.make_gaussian_batch_sampler = _perturbing_sampler(seed, dtype)
        try:
            trial = runner()[0]
        finally:
            scenarios.make_gaussian_batch_sampler = original
        for name in base:
            worst[name] = max(
                worst[name], _violation_ratio(trial[name], base[name], tolerance)
            )
    return worst


CHANNELS: dict[str, tuple[Callable[[], tuple[Any, ...]], str]] = {
    "standard_lqr": (scenarios.run_standard_lqr_scenario, "float64"),
    "signal_space_gd": (scenarios.run_signal_space_gd_scenario, "float32"),
    "signal_space_gd_3d": (scenarios.run_signal_space_gd_3d_scenario, "float32"),
}

#: Where `--write` puts the table the harness reads.
TABLE_PATH = Path(__file__).resolve().parent / "fixtures" / "ulp_movement.json"


def main() -> None:
    write = "--write" in sys.argv
    table: dict[str, dict[str, float]] = {}
    for channel, (runner, dtype_name) in CHANNELS.items():
        ratios = measure(runner, dtype_name)
        table[channel] = {n: float(f"{v:.4g}") for n, v in ratios.items()}
        print(f"\n=== {channel}  ({dtype_name}) ===")
        print(f"{'array':44s} {'violation ratio':>16s}")
        for name in sorted(ratios, key=lambda n: -ratios[n]):
            flag = "  <-- over contract" if ratios[name] > 1.0 else ""
            print(f"{name:44s} {ratios[name]:16.3e}{flag}")
    if write:
        # The ratios are only meaningful under the tolerances they were
        # computed with, so those travel in the same file: editing `TOLERANCES`
        # without re-running this tool leaves a table that silently describes a
        # contract nobody is asserting any more.
        table[TOLERANCE_RECORD_KEY] = {
            f"{name}.{field}": getattr(tolerance, field)
            for name, tolerance in TOLERANCES.items()
            for field in ("atol", "rtol")
        }
        TABLE_PATH.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
