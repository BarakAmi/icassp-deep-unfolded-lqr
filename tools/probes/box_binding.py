"""How hard the box binds on a frozen plant, and what a narrower one would buy.

A standing probe rather than a test, for the reason the two beside it are: it
runs the real recipes on the real frozen plants, it takes minutes, and it
answers a question no assertion on this branch can — *is this instance in a
regime where a box-aware controller has anything to win?*

**Two conventions, both reported, because the campaign's record carries both
and they disagree by eleven points.** `clipped` is the fraction of scalar
control entries at the box along the **clipped Riccati** loop — the trajectory
a feasible baseline actually flies, and what `evaluate_under_shift` calls the
saturation rate. `free` is the fraction of `|Kx|` entries at or above `u_max`
along the **unconstrained** loop, which is the quantity the campaign's Phase A
recorded as 77.61 % at n = 4. Neither is wrong and they are not comparable, so
a number quoted without its convention says nothing.

**Saturation is a property of a policy's trajectory, not of the plant.** The
frozen dual policy reads 83.2 % where the clipped baseline reads 86.4 % at the
same box, on the same plant. That is not a discrepancy to be reconciled.

**It is one knob.** Scale the state, the control, both noise scales and the box
by one factor: the closed loop is unchanged and every cost scales by its square.
So saturation depends on `u_max / sigma` alone, and `--sigma` exists to
*demonstrate* that rather than to offer a second dial — pass a ratio twice at
two scales and every ratio column reproduces while every cost column scales.

Usage::

    uv run python tools/probes/box_binding.py \\
        studies/icassp/icassp_n100m30_N100_u0p1_s0.npz \\
        --u-max 0.1,0.05,0.03,0.025,0.02 --frozen

Run with `--help` for the full surface.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mbl.applications.factories import GaussianBatchSpec
from mbl.applications.recipes.analytic import RiccatiRecipe, TruncatedRiccatiRecipe
from mbl.applications.recipes.cocp_exact import ExactCOCPLowerBoundRecipe
from mbl.applications.recipes.unfolded import FixedUnfoldedRecipe
from mbl.core.kernels import CostReduction, time_invariant_slice, total_quadratic_cost
from mbl.core.runtime.compute_context import ComputeContext
from mbl.models.constrained.box_lagrangian import BoxLQRData, finite_horizon_box_bound
from mbl.models.guards import require_linear_quadratic
from mbl.spec.gates import inverse_lipschitz_constant
from mbl.spec.problem import ProblemSpec

#: The probe's own tolerance, and DELIBERATELY not the production one.
#:
#: This constant used to carry a comment claiming it was "the tolerance
#: `experiments.shifted_evaluation` counts a control as saturated within ...
#: held here as the same number". It was not: that module counted at `1e-3`
#: and this one at `1e-6`, a factor of a thousand, and every saturation figure
#: the ICASSP campaign recorded came from this side of it. The claim is
#: withdrawn; the value is kept, because it is what the numbers on file were
#: measured with and moving it would silently restate them.
#:
#: The production statistic now lives in `core.constraint.activity` at the
#: tolerance Annex 01 §4.1 declares, and `tools/probes/saturation_tolerance.py`
#: is what measures the difference between the two rather than assuming it away.
SATURATION_TOLERANCE = 1e-6


def rebound(spec: ProblemSpec, u_max: float) -> ProblemSpec:
    """The same plant with a different box, and nothing else moved.

    `provenance` is dropped: the record would name parameters that no longer
    describe the matrices' box, and a probe must not write a provenance claim
    it did not author.
    """
    return replace(spec, data=replace(spec.data, control_bound=u_max), provenance=None)


def score(
    problem: Any, policy_source: Any, sample: Any, batches: int, u_max: float
) -> tuple[float, float]:
    """Mean cost and saturation over `batches` draws, through the shared kernel.

    Returns:
        `(mean cost, saturated fraction)`.
    """
    _, cost = require_linear_quadratic(problem)
    Q = torch.as_tensor(time_invariant_slice(cost.Q))
    R = torch.as_tensor(time_invariant_slice(cost.R))
    totals, hits, entries = [], 0, 0
    for _ in range(batches):
        initial_state, process_noise, measurement_noise = sample()
        with torch.no_grad():
            states, _, controls = problem.system.run(
                policy_source(), initial_state, process_noise, measurement_noise
            )
            totals.append(
                total_quadratic_cost(
                    Q.to(dtype=states.dtype),
                    R.to(dtype=controls.dtype),
                    states,
                    controls,
                    conventions=cost.conventions,
                    reduction=CostReduction.PER_SAMPLE,
                )
            )
        hits += int((controls.abs() >= (1.0 - SATURATION_TOLERANCE) * u_max).sum())
        entries += controls.numel()
    return float(torch.cat(totals).mean()), hits / entries


@dataclass(frozen=True)
class Draw:
    """How much of the evaluation distribution one row is scored over.

    One object rather than three parameters, and not only to satisfy the
    argument gate: `sigma` belongs with the counts because the three together
    are the *measurement*, and a row scored over a different draw is not
    comparable with its neighbours.
    """

    sigma: float
    batches: int
    batch_size: int


def _sampler(
    spec: ProblemSpec, sigma: float, batch_size: int, ctx: ComputeContext
) -> Any:
    """A fresh draw of the study's own evaluation distribution.

    Fresh per contender rather than shared, so every controller in a row meets
    the same trajectories — the campaign's batch-parity rule, and the reason
    two rows of this table may be compared at all.
    """
    return GaussianBatchSpec(
        state_dim=spec.data.state_dim,
        horizon=spec.data.horizon,
        batch_size=batch_size,
        seed=0,
        process_noise_std=sigma,
        initial_state_std=sigma,
    ).build(ctx.backend_enum, torch_dtype=torch.float64)[0]


def row(
    frozen: ProblemSpec,
    u_max: float,
    draw: Draw,
    *,
    depths: tuple[int, ...],
    dual: bool,
) -> dict[str, Any]:
    """Every quantity this probe reports, at one box width."""
    sigma, batches, batch_size = draw.sigma, draw.batches, draw.batch_size
    spec = rebound(frozen, u_max)
    problem = spec.build()
    horizon = spec.data.horizon
    state_dim = spec.data.state_dim
    ctx = ComputeContext(backend="torch", device="cpu", precision="float64")

    lipschitz = inverse_lipschitz_constant(spec)
    floor = finite_horizon_box_bound(
        BoxLQRData(
            A=np.asarray(spec.data.system["A"]),
            B=np.asarray(spec.data.system["B"]),
            Q=np.asarray(spec.data.cost["Q"][0]),
            R=np.asarray(spec.data.cost["R"][0]),
            W=sigma**2 * np.eye(state_dim),
            u_max=u_max,
        ),
        sigma**2 * np.eye(state_dim),
        horizon,
    ).value

    def measure(controller: Any) -> tuple[float, float]:
        cost, saturation = score(
            problem,
            controller.get_control_policy,
            _sampler(spec, sigma, batch_size, ctx),
            batches,
            u_max,
        )
        return float(cost), float(saturation)

    free_cost, free_saturation = measure(
        RiccatiRecipe(horizon=horizon).build_controller(problem, ctx)
    )
    clipped_cost, clipped_saturation = measure(
        TruncatedRiccatiRecipe(horizon=horizon).build_controller(problem, ctx)
    )
    projected = [
        measure(
            FixedUnfoldedRecipe(
                num_iterations=depth,
                # 2/L: the classical stability limit, and the step the campaign's
                # own ablation chose at this instance. It does not move with the
                # box, because L does not.
                step_size_init=2.0 / lipschitz,
                step_size_max=1.0,
                horizon=horizon,
            ).build_controller(problem, ctx)
        )[0]
        for depth in depths
    ]

    measured: dict[str, Any] = {
        "u_max": u_max,
        "ratio": u_max / sigma,
        "L": lipschitz,
        "saturation_clipped": clipped_saturation,
        "saturation_free": free_saturation,
        "floor": floor,
        "unconstrained": free_cost,
        "clipped": clipped_cost,
        "projected": projected,
    }
    if dual:
        started = time.perf_counter()
        controller = ExactCOCPLowerBoundRecipe(
            process_noise_std=sigma
        ).build_controller(problem, ctx)
        measured["dual_synthesis_s"] = time.perf_counter() - started
        measured["dual"], measured["saturation_dual"] = measure(controller)
    return measured


def _print(rows: list[dict[str, Any]], depths: tuple[int, ...], dual: bool) -> None:
    header = (
        f"{'u_max':>8} {'u/sig':>7} {'L':>16} {'sat|clip':>9} {'sat|free':>9} "
        f"{'floor':>10} {'unconstr':>10} {'clipped':>10} "
        + " ".join(f"{'PGD J=' + str(d):>10}" for d in depths)
    )
    if dual:
        header += f" {'dual':>10} {'clip/dual':>10}"
    print(header)
    for measured in rows:
        line = (
            f"{measured['u_max']:>8.4f} {measured['ratio']:>7.3f} "
            f"{measured['L']:>16.9f} {measured['saturation_clipped']:>8.1%} "
            f"{measured['saturation_free']:>8.1%} {measured['floor']:>10.4f} "
            f"{measured['unconstrained']:>10.4f} {measured['clipped']:>10.4f} "
            + " ".join(f"{value:>10.4f}" for value in measured["projected"])
        )
        if dual:
            line += (
                f" {measured['dual']:>10.4f} "
                f"{measured['clipped'] / measured['dual']:>10.4f}"
            )
        print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plant", type=Path, help="a frozen problem `.npz`")
    parser.add_argument(
        "--u-max",
        default="0.1",
        help="comma-separated box widths to sweep (the plant's own is ignored)",
    )
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--depths", default="1,3,10")
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="also score the dual-frozen convex policy, the untrained box-aware level",
    )
    args = parser.parse_args(argv)

    frozen = ProblemSpec.load(args.plant)
    depths = tuple(int(value) for value in args.depths.split(","))
    print(
        f"# {args.plant}  n={frozen.data.state_dim} m={frozen.data.control_dim} "
        f"N={frozen.data.horizon}  sigma={args.sigma}  "
        f"{args.batches} x {args.batch_size} trajectories",
        flush=True,
    )
    rows = [
        row(
            frozen,
            float(value),
            Draw(args.sigma, args.batches, args.batch_size),
            depths=depths,
            dual=args.frozen,
        )
        for value in args.u_max.split(",")
    ]
    _print(rows, depths, args.frozen)
    return 0


if __name__ == "__main__":  # pragma: no cover - a probe, not a library
    raise SystemExit(main())
