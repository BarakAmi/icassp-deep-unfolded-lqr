"""Does chunking do what it exists for, at a size that does not fit?

Annex 06 §4.3's gate, run where it means something. Not a test: it needs the
accelerator, it allocates tens of gigabytes, and it takes a minute -- `pytest`
must stay a thing you run without thinking.

**What it found, and why the annex's gate had to be corrected.** §4.3 asked for
"the unchunked run must fail with a CUDA out-of-memory error". On this host it
does not fail. WSL2's CUDA driver spills past the card's 24,467 MiB into shared
host memory, so the unchunked run *completes* -- and pays 26x in wall clock for
it. An OOM gate would therefore never fire here and would report the feature as
untestable rather than as working.

The honest gate is what chunking actually buys, measured against the unchunked
run at the same effective batch:

    same gradient, same loss, a fraction of the memory, a fraction of the time.

    uv run python tools/probes/microbatch_ceiling.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mbl.applications.rollout import RolloutModel  # noqa: E402
from mbl.core.cost.quadratic_cost import QuadraticCost  # noqa: E402
from mbl.core.optimal_control_problem import OptimalControlProblem  # noqa: E402
from mbl.core.runtime.compute_context import ComputeContext  # noqa: E402
from mbl.core.system.linear_system import LinearSystem  # noqa: E402
from mbl.engine.config import TrainingConfig  # noqa: E402
from mbl.engine.context import RunContext  # noqa: E402
from mbl.engine.strategy import GradientDescentStrategy  # noqa: E402
from mbl.experiments.bindings import DEFAULT_SPEC_BINDINGS as BINDINGS  # noqa: E402

#: Chosen so the unchunked autograd graph exceeds the card. Measured on this
#: host: 32,086 MiB peak against 24,467 MiB of VRAM.
STATE_DIM = 150
CONTROL_DIM = 75
HORIZON = 50
DEPTH = 20
BATCH = 65_536
MICROBATCH = 8_192


def _problem() -> OptimalControlProblem:
    """A *stable* random plant. An unscaled one diverges over 50 steps at this
    dimension and every loss comes back `nan`, which compares equal to nothing
    and would make the whole probe vacuous -- it was the first run's result."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(STATE_DIM, STATE_DIM)) / np.sqrt(STATE_DIM)
    a = a / (1.1 * max(abs(np.linalg.eigvals(a))))
    b = rng.normal(size=(STATE_DIM, CONTROL_DIM)) / np.sqrt(STATE_DIM)
    return OptimalControlProblem(
        system=LinearSystem.fully_observable(a, b),
        cost=QuadraticCost(
            Q=np.repeat(np.eye(STATE_DIM)[None], HORIZON + 1, axis=0),
            R=np.repeat((np.eye(CONTROL_DIM) * 0.1)[None], HORIZON, axis=0),
        ),
    )


def _rollout(problem: OptimalControlProblem) -> RolloutModel:
    plan = BINDINGS.build(
        "end_to_end",
        {"optimizer": "adam", "learning_rate": 0.01, "epochs": 1},
        {"state_dim": STATE_DIM, "horizon": HORIZON},
    )
    recipe = BINDINGS.registry.create(
        "unfolded",
        kind="learned_step_size",
        plan=plan,
        num_iterations=DEPTH,
        step_size_init=0.05,
        step_size_max=1.0,
        horizon=HORIZON,
    )
    ctx = ComputeContext(backend="torch", device="cuda", precision="float32")
    return RolloutModel(recipe.build_controller(problem, ctx)).to("cuda")


def main() -> int:
    if not torch.cuda.is_available():
        print("no accelerator: this probe has nothing to measure")
        return 2

    card = torch.cuda.get_device_properties(0).total_memory / 2**20
    problem = _problem()
    generator = torch.Generator(device="cuda").manual_seed(0)
    batch = (
        torch.randn(
            BATCH, STATE_DIM, generator=generator, device="cuda", dtype=torch.float32
        ),
        torch.randn(
            BATCH,
            HORIZON,
            STATE_DIM,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05,
        torch.zeros(BATCH, HORIZON, STATE_DIM, device="cuda", dtype=torch.float32),
    )
    context = RunContext(
        model=torch.nn.Module(),
        tracker=None,  # type: ignore[arg-type]
        config=TrainingConfig(batch_size=BATCH, learning_rate=0.01),
    )

    print(
        f"card {card:.0f} MiB | batch={BATCH} n={STATE_DIM} J={DEPTH} "
        f"N={HORIZON} float32"
    )
    results: dict[str | None, tuple[float, float, float, torch.Tensor]] = {}
    for microbatch in (None, MICROBATCH):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rollout = _rollout(problem)
        module = rollout.controller.as_module()
        strategy = GradientDescentStrategy(
            rollout,
            torch.optim.SGD(module.parameters(), lr=0.0),
            microbatch=microbatch,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = strategy.step(context, batch)["loss"]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**20
        grad = torch.cat(
            [
                p.grad.reshape(-1).clone()
                for p in module.parameters()
                if p.grad is not None
            ]
        )
        results[microbatch] = (elapsed, peak, loss, grad)
        fits = "spills past the card" if peak > card else "fits"
        print(
            f"  microbatch={str(microbatch):5s} {elapsed:7.2f} s  "
            f"peak {peak:8.0f} MiB ({fits})  loss {loss:.8f}"
        )
        del rollout, module, strategy

    whole, chunked = results[None], results[MICROBATCH]
    difference = float((whole[3] - chunked[3]).abs().max())
    print(f"  gradient max |diff| : {difference:.3e}")
    print(f"  loss   |diff|       : {abs(whole[2] - chunked[2]):.3e}")
    print(f"  speed-up            : {whole[0] / chunked[0]:.1f}x")
    print(f"  memory              : {whole[1] / chunked[1]:.1f}x less")

    ok = difference < 1e-8 and whole[1] > card and chunked[1] < card
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
