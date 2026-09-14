"""What the saturation tolerance is worth, per contender, on a real rung.

The tree carries **two** tolerances for one statistic and one of them claims to
be the other. `experiments.shifted_evaluation` counts a control saturated
within `1e-3` of the bound; `tools/probes/box_binding` uses `1e-6` under a
comment reading *"the tolerance `experiments.shifted_evaluation` counts a
control as saturated within ... held here as the same number"*. Every
saturation figure the ICASSP campaign has recorded -- 89.6 %, 32.3 %, 87.6 %,
82.9 % -- came from the `1e-6` side.

The production constant's stated reason is specific and testable: *"the GRU's
`u_max * tanh(.)` can approach but never reach `u_max` exactly."* This probe
tests it, by rolling every contender of a finished rung out of the store and
counting how often each sits on the box across a sweep of tolerances.

**Measured at `u_max = 0.02`, and the stated reason does not hold there.** The
recurrent baseline reaches the bound exactly and is the *least* tolerance-
sensitive member of the cast (94.25 % at tolerance zero, unmoved through
`1e-4`). The sensitive ones are the QP families, whose interior-point solutions
land near the bound rather than on it: `cocp_exact` swings **76.32 % -> 87.82 %**
over the same range. The tolerance therefore **reorders the cast** -- the exact
convex policy is last at tolerance zero and third at `1e-3` -- which is the
same lesson the campaign already learned about the two *conventions*: a
saturation number quoted without the tolerance it was counted at says nothing.

**Self-check first** (probe rule 6). The clipped Riccati baseline must
reproduce the campaign's recorded 89.6 % before any other row here is believed;
it reads 89.6523 %.

Usage::

    uv run python tools/probes/saturation_tolerance.py \\
        studies/icassp_exact_convex/fig5_stress_depth.toml --tier publication

Run with `--help` for the full surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch

from mbl.replay import load_study
from mbl.runner.producer import load_trained_controller
from mbl.spec.study import StudyPoint
from mbl.store.content_store import ModelStore

#: The two constants in the tree, the exact test, and two either side. `0.0`
#: is the one that is not a tolerance at all -- it asks how many entries are
#: *at* the representable bound, which is the only tolerance-free reading.
TOLERANCES = (0.0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2)

#: What the campaign's record says the clipped loop reads on this plant. The
#: instrument reproduces this or the rest of the table is measuring something
#: else (probe rule 6).
RECORDED_CLIPPED_SATURATION = 0.896

#: The contender the self-check is read off, and the width it is allowed to
#: miss the record by. Generous, because the record is quoted to one decimal.
SELF_CHECK_CONTENDER = "truncated_riccati"
SELF_CHECK_TOLERANCE = 5e-3


def selected(points: tuple[StudyPoint, ...], depth: int, seed: int) -> list[StudyPoint]:
    """One point per contender: replicate `seed`, and depth `depth` where swept.

    The depth is read off `axis_values` rather than out of the config, because
    a contender an `applies_to` excludes has no depth at all -- an absent key
    is the answer and not a gap, and reading the config back would silently
    give such a contender the recipe's own default.
    """
    chosen: dict[str, StudyPoint] = {}
    for point in points:
        if point.seed != seed:
            continue
        value = next(
            (v for k, v in point.axis_values.items() if k.endswith("num_iterations")),
            None,
        )
        if value is not None and int(value) != depth:
            continue
        chosen[point.contender.resolved_label] = point
    return [chosen[label] for label in sorted(chosen)]


def controller(point: StudyPoint, models: ModelStore) -> Any:
    """The stored model, back as something that can be rolled out.

    The producer's own two cases (`runner.producer._reuse`): a record carrying
    **weights** is a checkpoint and is rebuilt and loaded; a record carrying
    **none** is a receipt for a closed-form solve and is re-derived, which is
    bit-exact and costs a Riccati recursion.
    """
    recipe = point.contender.resolve()
    record = models.get(str(point.model_id))
    if record.weights:
        return load_trained_controller(
            recipe.build_controller, point.problem.build(), point.training.ctx, record
        )
    return recipe.build_controller(point.problem.build(), point.training.ctx)


def _policy_of(artifact: Any, device: Any, dtype: Any) -> Any:
    """The controller's policy, met where the batch actually lives.

    A re-derived analytic controller solves on NumPy and ignores the
    `ComputeContext` -- `runner.producer._reuse` records exactly this -- so its
    gains are host arrays while a CUDA study's batches are on the card. Only
    that case crosses to the host and back; every other contender's policy is
    called untouched, so its arithmetic is the stored run's. Either way the
    whole cast meets ONE draw of the evaluation distribution, which is the
    batch-parity rule that lets two rows be compared at all.
    """
    make_policy = getattr(artifact, "make_policy", None) or artifact.get_control_policy
    inner = make_policy()

    def bridged(t: int, x: torch.Tensor) -> torch.Tensor:
        try:
            return inner(t, x)
        except TypeError:
            return torch.as_tensor(inner(t, x.cpu())).to(device=device, dtype=dtype)

    return bridged


def _eval_mode(artifact: Any) -> None:
    """`.eval()` where there is a module, the way the measurement path does.

    Not decoration: under live dropout the same policy answers differently on
    the same batch, which the campaign measured at 38.986540 against 38.988693.
    """
    as_module = getattr(getattr(artifact, "controller", artifact), "as_module", None)
    if as_module is None:
        return
    try:
        as_module().eval()
    except AttributeError:
        return


def audit(point: StudyPoint, models: ModelStore, batches: int) -> dict[str, Any]:
    """Roll one contender out and count how often it sits on the box."""
    artifact = controller(point, models)
    _eval_mode(artifact)
    problem = point.evaluation.problem.build()
    u_max = float(point.evaluation.problem.data.control_bound)

    drawn = cast(Any, point.evaluation.protocol).build_batches(point.evaluation.ctx)
    peak = 0.0
    hits = dict.fromkeys(TOLERANCES, 0)
    entries = 0
    for initial_state, process_noise, measurement_noise in drawn[:batches]:
        policy = _policy_of(artifact, initial_state.device, initial_state.dtype)
        with torch.no_grad():
            _, _, U = cast(
                "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                problem.system.run(
                    policy, initial_state, process_noise, measurement_noise
                ),
            )
            abs_u = U.abs()
            peak = max(peak, float(abs_u.max()))
            entries += abs_u.numel()
            for tol in TOLERANCES:
                hits[tol] += int((abs_u >= (1.0 - tol) * u_max).sum())
    return {
        "label": point.contender.resolved_label,
        "role": point.contender.role.value,
        "u_max": u_max,
        "max_abs_u": peak,
        "violation": peak - u_max,
        "saturation": {tol: hits[tol] / entries for tol in TOLERANCES},
        "entries": entries,
    }


def _print(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'contender':24} {'role':10} {'max|u|':>13} {'max|u|-u_max':>14} "
        + " ".join(f"{'tol=' + format(tol, '.0e'):>10}" for tol in TOLERANCES),
        flush=True,
    )
    for row in rows:
        print(
            f"{row['label']:24} {row['role']:10} {row['max_abs_u']:>13.9f} "
            f"{row['violation']:>14.3e} "
            + " ".join(f"{row['saturation'][tol]:>9.2%}" for tol in TOLERANCES),
            flush=True,
        )


def _self_check(rows: list[dict[str, Any]]) -> None:
    """Reproduce the record, or say plainly that this instrument does not."""
    clipped = next((r for r in rows if r["label"] == SELF_CHECK_CONTENDER), None)
    if clipped is None:
        print(f"\n# self-check SKIPPED: no {SELF_CHECK_CONTENDER} in this cast")
        return
    measured = clipped["saturation"][1e-6]
    delta = abs(measured - RECORDED_CLIPPED_SATURATION)
    verdict = "REPRODUCES" if delta < SELF_CHECK_TOLERANCE else "DOES NOT REPRODUCE"
    print(
        f"\n# self-check: {SELF_CHECK_CONTENDER} at tol=1e-6 reads {measured:.4%}, "
        f"the record says {RECORDED_CLIPPED_SATURATION:.1%} -- {verdict} "
        f"(delta {delta:.4%})",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("document", type=Path, help="a study `.toml`, or its id")
    parser.add_argument("--tier", default="publication")
    parser.add_argument("--store", default="store")
    parser.add_argument("--depth", type=int, default=10, help="J, where swept")
    parser.add_argument("--seed", type=int, default=0, help="which replicate")
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args(argv)

    loaded = load_study(args.document, tier=args.tier)
    points = selected(loaded.study.materialise(), args.depth, args.seed)
    models = ModelStore(Path(args.store))

    print(
        f"# {args.document}  tier={args.tier}  J={args.depth}  seed={args.seed}  "
        f"{args.batches} evaluation batch(es)  -- indicative, one replicate; "
        "never to be set beside a publication number",
        flush=True,
    )
    rows = [audit(point, models, args.batches) for point in points]
    _print(rows)
    _self_check(rows)
    return 0


if __name__ == "__main__":  # pragma: no cover - a probe, not a library
    raise SystemExit(main())
