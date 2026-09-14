"""The learning-rate probe must author a document the grammar can still resolve.

The probe keeps only the contenders it sweeps. A tier override naming one it
dropped -- ``contenders.neural.config.plan.epochs`` -- then addresses a label
that is not there, and the whole document is refused at parse. The defect was
invisible for as long as the probe was only ever pointed at
`fig5_stress_depth.toml`, whose overrides are all wildcards; every Figure-2
document names `neural` explicitly and none of them could be probed at all.

The converse is the trap worth guarding: dropping too much is silent rather than
loud. An override naming a contender the probe *is* measuring carries its epoch
budget, and losing it would quietly re-run the comparison at a different budget
-- a fairness change reported as a learning-rate result.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

PROBE = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "probes"
    / "learning_rate_sensitivity.py"
)
STUDIES = Path(__file__).resolve().parents[2] / "studies" / "icassp_exact_convex"
SWEPT = ["unfolded_alpha", "unfolded_alpha_p", "unfolded_alpha_pj"]


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    """Import the probe by path; `tools/` is deliberately not a package."""
    spec = importlib.util.spec_from_file_location("lr_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "key,expected",
    [
        ("contenders.neural.config.plan.epochs", False),
        ("contenders.truncated_riccati.config.plan.epochs", False),
        ("contenders.unfolded_alpha.config.plan.epochs", True),
        ("contenders.unfolded_alpha_pj.config.plan.epochs", True),
        ("contenders.*.config.plan.epochs", True),
        ("training.batch_size", True),
        ("contenders", True),
    ],
)
def test_an_override_survives_exactly_when_it_still_addresses_something(
    probe: ModuleType, key: str, expected: bool
) -> None:
    """Wildcards and non-contender keys always survive; a label only if kept."""
    assert probe._override_survives(key, SWEPT) is expected


@pytest.mark.parametrize(
    "study",
    ["fig1_depth", "fig2_angle_blind", "fig2_angle_told", "fig2_angle_world"],
)
def test_every_campaign_document_authors_a_parseable_probe(
    probe: ModuleType, study: str
) -> None:
    """Every document the probe is pointed at must produce valid TOML.

    Parsing is not the whole gate -- the grammar refuses later -- but a dropped
    contender that leaves a dangling override shows up right here as a key
    naming a label the document no longer declares.
    """
    text = probe._document(STUDIES / f"{study}.toml", SWEPT, depth=3, rates=[0.05])
    authored = tomllib.loads(text)
    declared = {c["label"] for c in authored["contenders"]}
    assert declared == set(SWEPT)
    for tier, overrides in authored.get("tier_overrides", {}).items():
        for key in overrides:
            parts = key.split(".")
            if parts[0] == "contenders" and parts[1] != "*":
                assert parts[1] in declared, (
                    f"{study} [{tier}] overrides {parts[1]!r}, which the probe dropped"
                )


def test_the_epoch_budget_of_a_swept_contender_is_not_dropped(
    probe: ModuleType,
) -> None:
    """Filtering must not cost a measured contender the budget it was given.

    `fig2_angle_world` gives the unfolded families 200 epochs against the
    wildcard's 100. Losing that would change the comparison and report it as a
    learning-rate finding.
    """
    text = probe._document(
        STUDIES / "fig2_angle_world.toml", SWEPT, depth=3, rates=[0.05]
    )
    publication = tomllib.loads(text)["tier_overrides"]["publication"]
    source = tomllib.loads((STUDIES / "fig2_angle_world.toml").read_text())
    expected = source["tier_overrides"]["publication"]
    for label in SWEPT:
        key = f"contenders.{label}.config.plan.epochs"
        assert publication[key] == expected[key]
    assert "contenders.neural.config.plan.epochs" not in publication


def test_a_rehost_override_for_a_dropped_contender_is_not_carried(
    probe: ModuleType,
) -> None:
    """`fig2_angle_told` pins `standard_pgd`'s step size per rotated plant.

    That contender is not in the probe, and neither are the rotated plants --
    the probe sweeps one axis and drops the source's own. Carrying the block
    addresses nothing; it is the same law as the tier overrides, met as data
    rather than as a dotted key.
    """
    source = tomllib.loads((STUDIES / "fig2_angle_told.toml").read_text())
    assert source["evaluation"]["rehost_overrides"], "the fixture lost its premise"

    text = probe._document(
        STUDIES / "fig2_angle_told.toml", SWEPT, depth=3, rates=[0.05]
    )
    evaluation = tomllib.loads(text)["evaluation"]
    for entry in evaluation.get("rehost_overrides", []):
        assert entry["contender"] in SWEPT


def test_an_array_of_tables_survives_the_round_trip(probe: ModuleType) -> None:
    """TOML's inline table is `{k = v}`; JSON's is `{"k": v}`.

    A serialiser that confuses them turns data into a syntax error, which is how
    `fig2_angle_told` became unprobeable. Asserted on a value the campaign does
    not currently contain, so the guard outlives the document that exposed it.
    """
    rendered = probe._render(
        "evaluation",
        {"rehost_overrides": [{"contender": "x", "config": {"step_size_init": 0.5}}]},
    )
    assert tomllib.loads(rendered)["evaluation"]["rehost_overrides"] == [
        {"contender": "x", "config": {"step_size_init": 0.5}}
    ]
