"""Which contenders depend on their learning rate, and by how much?

A standing probe rather than a test, because it trains real models against the
real grammar and takes hours. It exists because a study can inherit one
learning rate for a whole family of contenders and never be asked whether that
rate suits them -- and on this project, once, it did not.

**One knob.** The document it authors is the tracked study's own contender
block with a single swept axis, ``contenders.*.config.plan.optimizer
.learning_rate``. Everything else -- plant, box, depth, seeds, epochs, batch,
initialisation, evaluation protocol -- is copied, so the arms differ in the
rate and in nothing else.

**A control arm, re-run at the probe's own effort.** The incumbent rate is
always swept alongside the candidates rather than compared against the stored
publication number, which would be the cross-tier error this project refuses by
name. It is also the instrument check: at ``--tier publication`` the control
arm reproduced all three tracked Figure-5 values *bit-for-bit* --
UF-alpha 133.2149 (spread 1.0914), UF-alphaP 114.3096 (0.0011) and
UF-alphaP-j 114.2878 (0.0026).

What it found at ``u_max = 0.02``, J = 10, 5 seeds, which corrected a claim
this repository had already written down::

    family        lr=0.05     lr=0.01    lr=0.002   gain    seed spread @0.05
    UF-alpha     133.2149    131.2622    132.6624   1.9527             1.0914
    UF-alphaP    114.3096    114.2284    114.3646   0.0812             0.0011
    UF-alphaP-j  114.2878    114.2289    114.3479   0.0589             0.0026

All three select 0.01. **UF-alpha gains 24x more from that selection than
UF-alphaP does**, its across-seed spread collapses 210x, and at the inherited
rate its training loss was still *rising* at epoch 500 (+0.0021/epoch). The
instability the Figure-5 study reported as a property of learning a step alone
was an optimizer artifact. What survives is sharper: the step-only family
depends on the rate, the preconditioned families do not, and the step-only
family is still ~15 % worse once both are tuned.

**Re-run against the earlier figures, 2026-09-04, and the answer INVERTS.** The
same three families, same 5 seeds, same publication tier, at the campaign's
original box ``u_max = 0.1`` -- Figure 1 at J = 10 and Figure 2 at J = 3::

    figure  family        lr=0.002    lr=0.01    lr=0.05    selects
    fig1    UF-alpha      8.843700   8.828862   8.811526     0.05
    fig1    UF-alphaP     8.609028   8.279876   8.271590     0.05
    fig1    UF-alphaP-j   8.596147   8.279160   8.272428     0.05
    fig2    UF-alpha      8.842511   8.816321   8.806860     0.05
    fig2    UF-alphaP     8.336119   8.278231   8.271555     0.05
    fig2    UF-alphaP-j   8.335313   8.277868   8.271706     0.05

**All six select the incumbent 0.05, every bootstrap interval disjoint from the
runner-up's.** So the rate is not a property of the family, it is a property of
**the box**: at ``u_max = 0.02`` all three select 0.01 and UF-alpha's across-seed
spread at 0.05 is 1.0914; at ``u_max = 0.1`` the same family at the same rate
has a spread of **9.5e-05**, four orders of magnitude tighter. Figure 5 carrying
a different learning rate from Figures 1 and 2 is therefore a measured
consequence of its instance and not an inconsistency -- which is the form the
paper should state it in, because a reviewer will otherwise read it as one.

This also settles a constraint rather than merely observing one:
`tests/spec/test_icassp_exact_convex_reuse.py` pins every non-convex `ModelID`
across the two campaigns, so era 06's Figures 1 and 2 *cannot* change their rate
without silently retraining everything they reuse. The measurement and the
constraint agree, so nothing had to be traded.

**Coverage, stated because it is not total.** The probe keeps one axis and drops
the source document's own, so the Figure-2 run covers the arm trained on the
nominal plant. That is the whole of `fig2_angle_blind` and `fig2_angle_told`,
which train there and rotate only at evaluation; `fig2_angle_world` trains on
six plants and only its nominal arm is measured here.

Do not read the collapse of step components to zero as the pathology: at
lr = 0.01 -- the best arm -- 62.0 % of them still go to zero, slightly more
than at 0.05. Only lr = 0.002 avoids it, and that arm has barely left its
initialisation (median alpha 0.602/L against an init of 0.5/L), which is why
its cost is worse. Switching most steps off is what a good solution does here;
the pathology was the seeds disagreeing about *which* ones.

Usage::

    uv run python tools/probes/learning_rate_sensitivity.py \\
        studies/icassp_exact_convex/fig5_stress_depth.toml \\
        --contenders unfolded_alpha --depth 10 --tier publication

    # cheap smoke of the machinery, minutes rather than hours
    uv run python tools/probes/learning_rate_sensitivity.py \\
        studies/icassp_exact_convex/fig5_stress_depth.toml \\
        --contenders unfolded_alpha --depth 1 --tier standard --rates 0.05,0.01

Read beside
[the Figure-5 plan](../../docs/planning/06_icassp_exact_convex/figure_five_stress_test/binding_box_stress_test_at_n100.md)
and `alpha_identifiability.py`, which measured the landscape this explains.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _document(
    source: Path, contenders: list[str], depth: int, rates: list[float]
) -> str:
    """The probe document, as TOML text.

    Built by editing the source study's own parsed contender blocks rather than
    by re-declaring them, so a field nobody remembered is carried rather than
    dropped.
    """
    declared = tomllib.loads(source.read_text(encoding="utf-8"))
    keep = [c for c in declared["contenders"] if c["label"] in contenders]
    missing = set(contenders) - {c["label"] for c in keep}
    if missing:
        raise SystemExit(f"{source} declares no contender(s) named {sorted(missing)}")

    lines = [
        "# GENERATED by tools/probes/learning_rate_sensitivity.py -- not a tracked study.",
        f"# Source: {source.relative_to(REPO) if source.is_relative_to(REPO) else source}",
        'id = "probe/learning_rate_sensitivity"',
        "",
        "[problem]",
        f'path = "{Path(declared["problem"]["path"]).name}"',
        "",
    ]
    for section in ("compute", "training", "evaluation"):
        lines.append(_render(section, _for_kept(declared[section], contenders)))
    for contender in keep:
        block = dict(contender)
        config = dict(block.pop("config"))
        config["num_iterations"] = depth
        lines.append("[[contenders]]")
        for key, value in block.items():
            lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
        lines.append(_render("contenders.config", config))
    lines.append("[[sweep]]")
    lines.append('path = "contenders.*.config.plan.optimizer.learning_rate"')
    lines.append(f"values = {json.dumps(rates)}")
    lines.append(f"applies_to = {json.dumps(contenders)}")
    lines.append("")
    lines.append("[[analyses]]")
    lines.append('id = "cost_by_lr"')
    lines.append('kind = "cost_vs_axis"')
    lines.append('axis_path = "contenders.*.config.plan.optimizer.learning_rate"')
    lines.append("")
    for tier, overrides in declared.get("tier_overrides", {}).items():
        kept = {
            key: value
            for key, value in overrides.items()
            if _override_survives(key, contenders)
        }
        if not kept:
            continue
        lines.append(f"[tier_overrides.{tier}]")
        for key, value in kept.items():
            lines.append(f'"{key}" = {json.dumps(value)}')
        lines.append("")
    return "\n".join(lines) + "\n"


def _for_kept(section: dict[str, Any], contenders: list[str]) -> dict[str, Any]:
    """Drop per-contender entries addressing a contender the probe did not keep.

    The same law as `_override_survives`, applied to the one place it appears as
    data rather than as a dotted key: ``evaluation.rehost_overrides`` is a list of
    tables each naming a ``contender``, and the Figure-2 documents pin
    `standard_pgd`'s step size per rotated plant there. Carrying those into a
    document that no longer declares `standard_pgd` addresses nothing.
    """
    overrides = section.get("rehost_overrides")
    if not isinstance(overrides, list):
        return section
    kept = [
        entry
        for entry in overrides
        if not isinstance(entry, dict)
        or entry.get("contender") in contenders
        or "contender" not in entry
    ]
    trimmed = dict(section)
    if kept:
        trimmed["rehost_overrides"] = kept
    else:
        trimmed.pop("rehost_overrides")
    return trimmed


def _override_survives(key: str, contenders: list[str]) -> bool:
    """Whether a tier override still addresses something the probe document has.

    The probe keeps only the contenders it sweeps, so an override naming one it
    dropped -- ``contenders.neural.config.plan.epochs`` -- addresses a label that
    is no longer there, and the grammar refuses the whole document. Wildcards and
    non-contender overrides always survive; a concrete label survives only if it
    was kept, which is what stops this from quietly dropping the epoch budget of
    a contender the probe *is* measuring.
    """
    parts = key.split(".")
    if len(parts) < 2 or parts[0] != "contenders" or parts[1] == "*":
        return True
    return parts[1] in contenders


def _toml_value(value: Any) -> str:
    """Serialise one value as TOML, which JSON only coincides with for scalars.

    `json.dumps` renders a mapping as ``{"k": v}``, and TOML's inline table is
    ``{k = v}`` -- so an array of tables came out as a syntax error rather than
    as data. Nothing in the campaign hit it until `fig2_angle_told`, whose
    ``rehost_overrides`` is a list of tables; every other value in every other
    document is a scalar or an array of scalars, where the two agree.
    """
    if isinstance(value, dict):
        inner = ", ".join(
            f"{json.dumps(k)} = {_toml_value(v)}" for k, v in value.items()
        )
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(value)


def _render(header: str, table: dict[str, Any]) -> str:
    """One TOML table and its sub-tables, scalars before tables."""
    lines = [f"[{header}]"]
    for key, value in table.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in table.items():
        if isinstance(value, dict):
            lines.append("")
            lines.append(_render(f"{header}.{key}", value))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", type=Path)
    parser.add_argument("--contenders", required=True)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--rates", default="0.05,0.01,0.002")
    parser.add_argument("--tier", default="publication")
    parser.add_argument("--into", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="author and resolve, train nothing"
    )
    args = parser.parse_args()

    contenders = args.contenders.split(",")
    rates = [float(value) for value in args.rates.split(",")]
    root = args.into or Path("/tmp") / "mbl-probe-lr"
    root.mkdir(parents=True, exist_ok=True)

    source = args.study.resolve()
    declared = tomllib.loads(source.read_text(encoding="utf-8"))
    plant = (source.parent / declared["problem"]["path"]).resolve()
    shutil.copy2(plant, root / plant.name)
    document = root / "learning_rate_sensitivity.toml"
    document.write_text(_document(source, contenders, args.depth, rates), "utf-8")

    print(f"probe document : {document}")
    print(f"throwaway store: {root / 'store'}")
    print(f"contenders     : {contenders}  depth J={args.depth}  rates {rates}")
    print("the incumbent rate is swept as the control arm, never assumed\n")

    command = [
        "uv",
        "run",
        "mbl",
        "--store",
        str(root / "store"),
        "run",
        str(document),
        "--tier",
        args.tier,
    ]
    if args.dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=REPO, check=False)
    if completed.returncode or args.dry_run:
        return completed.returncode

    return subprocess.run(
        [
            "uv",
            "run",
            "mbl",
            "--store",
            str(root / "store"),
            "analyse",
            str(document),
            "--tier",
            args.tier,
        ],
        cwd=REPO,
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
