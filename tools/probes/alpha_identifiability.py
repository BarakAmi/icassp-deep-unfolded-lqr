"""How flat is the training objective in a learned step size, and where?

A standing probe rather than a test, for the reason the ones beside it are: it
builds the real recipe on the real frozen plant and answers a question no
assertion on this branch can -- *could an optimizer have identified this
parameter at all?*

**Measured at a shared reference point, never at a trained model's own
optimum.** A near-zero gradient at a converged model is what convergence looks
like, not what unidentifiability looks like. This evaluates every arm at the
declared initialisation, before any optimizer has acted, so the flatness it
reports is a property of the landscape rather than of a training run.

**Saturation here is the unfolded rollout's own**, at that reference alpha --
NOT the clipped-Riccati convention `box_binding.py` reports. The two are not
comparable and a number quoted without its convention says nothing: at
``u_max = 0.05`` this probe reads 49.2 % where the clipped convention reads
67.7 %.

**One knob, and it is the box.** ``A``, ``B``, ``Q`` and ``R`` are held; only
``control_bound`` moves, which is exactly how the campaign's two n = 100 plants
differ -- verified bit-for-bit before this probe was written.

What it found, and none of it was expected::

    u_max   saturation at J=10   median |dL/dalpha|, J=1 -> J=10   collapse
    0.10               11.5 %            4.31e+03 -> 4.89e+01          88x
    0.05               49.2 %            3.97e+03 -> 1.37e+01         290x
    0.02               83.6 %            1.52e+03 -> 1.36e+00        1114x

No component's gradient is ever exactly zero -- the *whole* gradient shrinks as
saturation climbs, by three orders of magnitude at the tightest box. There is a
structural reason: the projection's derivative is zero on a clamped coordinate,
so a saturated control contributes nothing through that path.

Why that matters, and why it is not the whole story: adam normalises by the
running gradient RMS, so a uniformly smaller gradient does not shorten its
steps -- it makes them noise-driven at a fixed learning rate. That predicted
what `learning_rate_sensitivity.py` then measured, and it is the reason the
Figure-5 study now selects its rate instead of inheriting one.

Usage::

    uv run python tools/probes/alpha_identifiability.py
    uv run python tools/probes/alpha_identifiability.py --boxes 0.1,0.02 --batch 4096

Read beside
[the Figure-5 plan](../../docs/planning/06_icassp_exact_convex/figure_five_stress_test/binding_box_stress_test_at_n100.md).
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import TYPE_CHECKING

import torch

from mbl.applications.factories import GaussianBatchSpec
from mbl.applications.recipes.unfolded import UnfoldedKind, UnfoldedRecipe
from mbl.applications.rollout import RolloutModel
from mbl.core.constraint.box_constraint import BoxConstraint
from mbl.core.optimal_control_problem import OptimalControlProblem
from mbl.core.runtime.compute_context import Backend, ComputeContext
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.spec.problem import ProblemSpec

if TYPE_CHECKING:
    from mbl.models.unfolded.base import UnfoldedController

#: The frozen n = 100 plant's gradient Lipschitz constant, authored by
#: `tools/report_pgd_step.py`. Only used to express steps in units of 1/L.
LIPSCHITZ = 1386.6869680452405

#: 0.5 x 1/L -- the campaign's initialisation law, and the shared reference
#: point every arm is evaluated at.
STEP_INIT = 0.0003605716441576073

DEFAULT_PLANT = "studies/icassp/icassp_n100m30_N100_u0p1_s0.npz"


def rebox(spec: ProblemSpec, u_max: float) -> OptimalControlProblem:
    """`spec`'s plant with its box replaced and nothing else touched."""
    problem = spec.build()
    return OptimalControlProblem(
        system=problem.system,
        cost=problem.cost,
        constraints=[BoxConstraint(u_max=u_max)],
    )


def controller_at_reference(
    depth: int, ctx: ComputeContext, problem: OptimalControlProblem
) -> UnfoldedController:
    """A learned-step controller at `STEP_INIT`, untrained.

    The plan is present because `UnfoldedRecipe` requires one; no optimizer is
    ever built from it, because this probe never trains.
    """
    recipe = UnfoldedRecipe(
        kind=UnfoldedKind.LEARNED_STEP_SIZE,
        plan=TrainingPlan(
            optimizer=OptimizerSpec(name="adam", learning_rate=0.05), epochs=1
        ),
        num_iterations=depth,
        step_size_init=STEP_INIT,
        step_size_max=1.0,
        horizon=100,
        label="unfolded_alpha",
    )
    return recipe.build_controller(problem, ctx)


@dataclasses.dataclass(frozen=True)
class Reading:
    """One (box, depth) arm."""

    u_max: float
    depth: int
    loss: float
    saturation: float
    gradient_median: float
    gradient_max: float


def measure(
    spec: ProblemSpec,
    u_max: float,
    depth: int,
    ctx: ComputeContext,
    batch: tuple[torch.Tensor, ...],
) -> Reading:
    """The objective's sensitivity to each alpha component, at the reference."""
    problem = rebox(spec, u_max)
    controller = controller_at_reference(depth, ctx, problem)
    model = RolloutModel(controller, problem)
    # The controller is built fresh for every arm, so its gradient is already
    # empty; zeroing it here would narrow the type to None for the rest of the
    # function and make the guard below unreachable.
    rho = dict(controller.as_module().named_parameters())["step_size"]

    _, _, controls, cost = model(*batch)
    loss = cost.mean()
    loss.backward()
    reduced = float(loss.detach())
    gradient_raw = rho.grad
    if gradient_raw is None:  # pragma: no cover -- backward() has just run
        raise RuntimeError("the step size received no gradient")

    alpha = torch.sigmoid(rho.detach())
    # dL/dalpha = dL/drho / sigma'(rho), and sigma' = alpha (1 - alpha) since
    # step_size_max is 1.0 here. Reported in alpha's units, not rho's, so the
    # number is comparable across arms.
    gradient = (gradient_raw / (alpha * (1.0 - alpha))).abs()
    saturated = (controls.detach().abs() >= u_max * (1.0 - 1e-6)).double().mean()
    return Reading(
        u_max=u_max,
        depth=depth,
        loss=reduced,
        saturation=float(saturated),
        gradient_median=float(gradient.median()),
        gradient_max=float(gradient.max()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plant", default=DEFAULT_PLANT)
    parser.add_argument("--boxes", default="0.10,0.05,0.02")
    parser.add_argument("--depths", default="1,3,5,10")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    boxes = [float(value) for value in args.boxes.split(",")]
    depths = [int(value) for value in args.depths.split(",")]

    torch.manual_seed(args.seed)
    ctx = ComputeContext(
        backend="torch",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        precision="float32",
    )
    spec = ProblemSpec.load(args.plant)
    sampler, _ = GaussianBatchSpec(
        state_dim=spec.state_dim,
        horizon=spec.horizon,
        batch_size=args.batch,
        seed=args.seed,
        process_noise_std=0.5,
        initial_state_std=0.5,
    ).build(Backend.TORCH, torch_dtype=ctx.torch_dtype, torch_device=ctx.torch_device)
    batch = sampler()

    inverse = 1.0 / LIPSCHITZ
    print(f"plant {args.plant}")
    print(
        f"n={spec.state_dim} m={spec.control_dim} batch={args.batch} ctx={ctx.device}"
    )
    print(f"every arm evaluated at the SAME alpha = {STEP_INIT!r} = 0.5 x 1/L")
    print("saturation is the UNFOLDED rollout's own convention, not the clipped one\n")
    header = f"{'u_max':>6} {'J':>3} {'loss':>12} {'sat %':>7} {'|dL/da| median':>16} {'max':>13}"
    print(header)
    print("-" * len(header))
    for u_max in boxes:
        readings = [measure(spec, u_max, depth, ctx, batch) for depth in depths]
        for reading in readings:
            print(
                f"{reading.u_max:>6.2f} {reading.depth:>3} {reading.loss:>12.4f} "
                f"{100 * reading.saturation:>6.1f}% {reading.gradient_median:>16.4e} "
                f"{reading.gradient_max:>13.4e}"
            )
        if len(readings) > 1:
            collapse = readings[0].gradient_median / readings[-1].gradient_median
            print(f"{'':>6} {'':>3} collapse across depth: {collapse:>8.0f}x")
        print()
    print(f"1/L = {inverse!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
