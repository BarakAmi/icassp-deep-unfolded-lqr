"""Acceptance tests for the TOML surface — Stage 2 Phase E3.

Written before the implementation. The checkpoint class is **degeneracy**:
malformed, contradictory and empty specifications must each fail with a message
naming the offending field — not a `KeyError`, not a `TOMLDecodeError` handed
straight through, and never a default silently substituted. A grammar whose
errors are unreadable is a grammar nobody uses, so the errors are a deliverable
here rather than a by-product, and they are what most of this module asserts.

Two properties beyond the errors carry the phase:

* **Neither surface is privileged** (Annex 01 §3). A study composed in Python
  and the same study written in TOML must derive the *same* `StudyID`. That
  single assertion covers every field the loader touches at once, and it fails
  loudly if the reader quietly drops one.
* **NB04 re-expressed declaratively is the same study.** Verified by
  independent recomputation against the live `nb04_box_constrained` module
  rather than against a tree this file writes down — the same rule Phase B
  followed for the registry.

The loader also discharges `require_supported_precision`'s **second and last**
mandated call site (plan §9.2b): a COCP contender under a float32 context must
fail here, at parse time, and not at the first gradient.
"""

from __future__ import annotations

import dataclasses

from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from mbl.applications.studies import nb04_box_constrained as nb04
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import OptimizerSpec, TrainingPlan
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.experiments import EvaluationProtocol
from mbl.applications.factories import GaussianBatchSpec
from mbl.spec.analysis import AnalysisSpec
from mbl.spec.contender import ContenderSpec, Role
from mbl.spec.data import DataSpec
from mbl.spec.errors import SpecificationError
from mbl.spec.evaluation import EvaluationSpec
from mbl.spec.figure import FigureSpec
from mbl.spec.gates import GateKind, GateSpec
from mbl.spec.loader import (
    BUILD_KEY,
    StudyDocument,
    load_study,
    load_tier_catalogue,
)
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import StudySpec, SweepAxis
from mbl.spec.tiers import DEFAULT_TIER_CATALOGUE
from mbl.spec.training import BatchPlan, TrainingSpec

from .test_contender import REGISTRY

N, M, HORIZON = 4, 2, 6

#: The tracked study catalogue this repository ships.
STUDIES = Path(__file__).resolve().parents[2] / "studies"


def _problem_file(tmp_path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    spec = ProblemSpec(
        ProblemData(
            system={"A": rng.normal(size=(N, N)), "B": rng.normal(size=(N, M))},
            cost={
                "Q": np.repeat(np.eye(N)[None], HORIZON + 1, axis=0),
                "R": np.repeat(np.eye(M)[None], HORIZON, axis=0),
            },
            horizon=HORIZON,
            control_bound=0.5,
        )
    )
    path = tmp_path / f"problem_{seed}.npz"
    spec.save(path)
    return path


#: A complete, valid study. Every degenerate document below is this one with a
#: single thing broken, so a failure is attributable to that thing.
GOOD = """
id = "probe/depth_scaling"

[problem]
path = "problem_0.npz"

[compute]
backend = "torch"
device = "cpu"
precision = "float64"

[training]
seeds = [0, 1]

[training.data]
kind = "gaussian"
seed = 1
process_noise_std = 0.5

[training.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.001
epochs = 3

[training.batch]
effective_size = 256
microbatch = 64

[evaluation]
metrics = ["expected_cost"]

[evaluation.protocol]
_build = "protocol"
n_batches = 2

[evaluation.protocol.batch_spec]
_build = "gaussian"
batch_size = 64
seed = 0
process_noise_std = 0.5

[[contenders]]
label = "unfolded_a"
family = "unfolded"
role = "contender"

[contenders.config]
kind = "learned_step_size"
num_iterations = 3
step_size_init = 0.1
step_size_max = 1.0
horizon = 6

[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.001
epochs = 3

[[contenders]]
label = "baseline"
family = "truncated_riccati"
role = "baseline"

[contenders.config]
horizon = 6

[[sweep]]
path = "contenders.*.config.num_iterations"
values = [2, 4, 8]
applies_to = ["unfolded_a"]

[[gates]]
kind = "constraint_binds"
min_fraction = 0.01

[tier_overrides.smoke]
"training.plan.epochs" = 1
"""


#: A well-formed COCP contender, appended where a test needs the one family
#: with a precision requirement. Spelled out rather than produced by swapping a
#: family name in `GOOD`: COCP takes a `plan` that no other contender there
#: does, and a swap leaves it unresolvable -- which then masks the very error
#: the precision test exists to observe.
COCP_CONTENDER = """
[[contenders]]
label = "cocp"
family = "cocp"

[contenders.config]
solver_eps = 1e-8

[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.1
epochs = 2
"""


def _write(tmp_path: Path, document: str, name: str = "study.toml") -> Path:
    _problem_file(tmp_path)
    path = tmp_path / name
    path.write_text(document)
    return path


def _load(tmp_path: Path, document: str = GOOD) -> StudyDocument:
    return load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)


# --------------------------------------------------------------------------
# The happy path — what the document says is what the study is
# --------------------------------------------------------------------------


def test_a_complete_document_loads(tmp_path: Path) -> None:
    study = _load(tmp_path).study
    assert study.id == "probe/depth_scaling"
    assert [c.resolved_label for c in study.contenders] == ["unfolded_a", "baseline"]
    assert study.training.seeds == (0, 1)
    assert study.training.batch == BatchPlan(effective_size=256, microbatch=64)
    assert study.evaluation.metrics == ("expected_cost",)
    assert study.sweep[0].values == (2, 4, 8)
    assert study.gates[0].kind is GateKind.CONSTRAINT_BINDS


def test_the_loaded_study_materialises(tmp_path: Path) -> None:
    """The end of the pipeline, not the middle: a document that parses into an
    object that cannot expand is a document that has not been loaded."""
    points = _load(tmp_path).study.materialise()
    assert (
        len(points) == (3 + 1) * 2
    )  # (swept contender x 3 depths + baseline) x 2 seeds
    assert len({p.model_id for p in points}) == len(points)


def test_the_batch_dimensions_come_from_the_problem(tmp_path: Path) -> None:
    """`state_dim` and `horizon` are the problem's, and the document has no key
    for them. Letting a document restate a dimension is letting it disagree
    with the matrices, which is a whole class of error deleted by omission."""
    study = _load(tmp_path).study
    batch_spec = study.evaluation.protocol.batch_spec
    assert (batch_spec.state_dim, batch_spec.horizon) == (N, HORIZON)
    assert "state_dim" not in GOOD


def test_the_evaluation_problem_defaults_to_the_training_problem(
    tmp_path: Path,
) -> None:
    study = _load(tmp_path).study
    assert study.evaluation.problem.problem_id == study.problem.problem_id


def test_a_declared_evaluation_problem_makes_it_a_shifted_study(
    tmp_path: Path,
) -> None:
    """The one field that turns an ordinary study into a distribution-shift
    study, reachable from the document."""
    _problem_file(tmp_path, seed=7)
    document = GOOD.replace(
        "[evaluation]\nmetrics",
        '[evaluation.problem]\npath = "problem_7.npz"\n\n[evaluation]\nmetrics',
    )
    study = load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS).study
    assert study.evaluation.problem.problem_id != study.problem.problem_id


# --------------------------------------------------------------------------
# Neither surface is privileged (Annex 01 §3)
# --------------------------------------------------------------------------


def test_the_toml_and_python_surfaces_derive_the_same_study(tmp_path: Path) -> None:
    """The strongest single assertion available here: it covers every field the
    loader touches at once, and it fails if the reader drops one, coerces one,
    or reaches a default the Python author did not write."""
    loaded = _load(tmp_path).study
    problem = ProblemSpec.load(tmp_path / "problem_0.npz")
    ctx = ComputeContext(
        backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64
    )
    plan = TrainingPlan(
        optimizer=OptimizerSpec(name="adam", learning_rate=0.001), epochs=3
    )
    composed = StudySpec(
        id="probe/depth_scaling",
        problem=problem,
        contenders=(
            ContenderSpec(
                family="unfolded",
                config={
                    "kind": "learned_step_size",
                    "num_iterations": 3,
                    "step_size_init": 0.1,
                    "step_size_max": 1.0,
                    "horizon": 6,
                    "plan": plan,
                },
                label="unfolded_a",
                registry=REGISTRY,
            ),
            ContenderSpec(
                family="truncated_riccati",
                config={"horizon": 6},
                label="baseline",
                role=Role.BASELINE,
                registry=REGISTRY,
            ),
        ),
        training=TrainingSpec(
            data=DataSpec(kind="gaussian", seed=1, params={"process_noise_std": 0.5}),
            plan=plan,
            batch=BatchPlan(effective_size=256, microbatch=64),
            ctx=ctx,
            seeds=(0, 1),
        ),
        evaluation=EvaluationSpec(
            problem=problem,
            protocol=EvaluationProtocol(
                batch_spec=GaussianBatchSpec(
                    state_dim=N,
                    horizon=HORIZON,
                    batch_size=64,
                    seed=0,
                    process_noise_std=0.5,
                ),
                n_batches=2,
            ),
            ctx=ctx,
        ),
        sweep=(
            SweepAxis(
                path="contenders.*.config.num_iterations",
                values=(2, 4, 8),
                applies_to=("unfolded_a",),
            ),
        ),
        gates=(GateSpec(GateKind.CONSTRAINT_BINDS, {"min_fraction": 0.01}),),
    )
    assert loaded.study_id == composed.study_id


# --------------------------------------------------------------------------
# THE CHECKPOINT — degeneracy, every class, each naming its own offender
# --------------------------------------------------------------------------

#: `(what is broken, the document, the token the message must contain)`.
#: Grouped by the three classes the checkpoint names: malformed, empty and
#: contradictory. Each entry breaks exactly one thing in `GOOD`.
DEGENERATE: list[tuple[str, str, str]] = [
    # -- malformed -------------------------------------------------------
    ("not toml at all", "id = 'unterminated\n[[[", "study.toml"),
    ("a table opened twice", GOOD + "\n[training]\nseeds = [9]\n", "study.toml"),
    # -- empty -----------------------------------------------------------
    ("an empty document", "", "is empty"),
    ("no id", GOOD.replace('id = "probe/depth_scaling"', ""), "id"),
    ("no contenders", GOOD.split("[[contenders]]")[0], "contenders"),
    (
        "no problem",
        GOOD.replace('[problem]\npath = "problem_0.npz"', ""),
        "problem is missing",
    ),
    ("an empty seed list", GOOD.replace("seeds = [0, 1]", "seeds = []"), "seeds"),
    (
        "no training distribution",
        GOOD.replace(
            '[training.data]\nkind = "gaussian"\nseed = 1\nprocess_noise_std = 0.5\n\n',
            "",
        ),
        "training.data",
    ),
    # -- contradictory ---------------------------------------------------
    (
        "an axis naming a contender that does not exist",
        GOOD.replace('applies_to = ["unfolded_a"]', 'applies_to = ["unfolded_b"]'),
        "unfolded_b",
    ),
    (
        "two contenders sharing a label",
        GOOD.replace('label = "baseline"', 'label = "unfolded_a"'),
        "unfolded_a",
    ),
    (
        "an unknown contender family",
        GOOD.replace('family = "truncated_riccati"', 'family = "riccati_truncated"'),
        "riccati_truncated",
    ),
    (
        "an unknown precision",
        GOOD.replace('precision = "float64"', 'precision = "float128"'),
        "precision",
    ),
    (
        "an unknown backend",
        GOOD.replace('backend = "torch"', 'backend = "jax"'),
        "backend",
    ),
    (
        "a problem file that is not there",
        GOOD.replace('path = "problem_0.npz"', 'path = "absent.npz"'),
        "absent.npz",
    ),
    (
        "an unknown gate kind",
        GOOD.replace('kind = "constraint_binds"', 'kind = "constraint_bound"'),
        "constraint_bound",
    ),
    (
        "a gate that can never fire",
        GOOD.replace("min_fraction = 0.01", "min_fraction = 0.0"),
        "min_fraction",
    ),
    (
        "an unknown composition on an axis",
        GOOD.replace("values = [2, 4, 8]", 'values = [2, 4, 8]\ncompose = "cartesian"'),
        "compose",
    ),
    (
        "composition declared on the study, where it used to live",
        GOOD.replace(
            'id = "probe/depth_scaling"',
            'id = "probe/depth_scaling"\ncompose = "product"',
        ),
        "compose",
    ),
    (
        "a tier override outside the whitelist",
        GOOD.replace('"training.plan.epochs" = 1', '"sweep.0.values" = [1]'),
        "sweep.0.values",
    ),
    (
        "an unknown thing to build",
        GOOD.replace(
            '_build = "end_to_end"\noptimizer', '_build = "endtoend"\noptimizer'
        ),
        "available",
    ),
    (
        "a plan missing its epochs",
        GOOD.replace(
            '_build = "end_to_end"\noptimizer = "adam"\nlearning_rate = 0.001\nepochs = 3\n\n[training.batch]',
            '_build = "end_to_end"\noptimizer = "adam"\nlearning_rate = 0.001\n\n[training.batch]',
        ),
        "needs 'epochs'",
    ),
    (
        "a batch declaration restating a problem dimension",
        GOOD.replace("batch_size = 64", "batch_size = 64\nstate_dim = 99"),
        "state_dim",
    ),
    (
        "an enum value no family knows",
        GOOD.replace('kind = "learned_step_size"', 'kind = "learned_step"'),
        "learned_step",
    ),
]


@pytest.mark.parametrize(
    ("document", "token"),
    [(document, token) for _, document, token in DEGENERATE],
    ids=[name for name, _, _ in DEGENERATE],
)
def test_a_degenerate_document_fails_naming_the_offender(
    tmp_path: Path, document: str, token: str
) -> None:
    """Every class of broken document, each asserted to name the thing that is
    wrong with it. The message is the deliverable: a `KeyError` from three
    frames down tells an author that something is missing, but not what, and a
    silently substituted default tells them nothing at all."""
    with pytest.raises(SpecificationError) as error:
        load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)
    assert token in str(error.value)


def test_the_reference_document_is_not_itself_degenerate(tmp_path: Path) -> None:
    """Anti-vacuity for the whole table above. Every entry is `GOOD` with one
    thing broken, so if `GOOD` stopped loading, all of them would pass for the
    wrong reason."""
    assert _load(tmp_path).study.id == "probe/depth_scaling"


@pytest.mark.parametrize(
    ("name", "document"),
    [(name, document) for name, document, _ in DEGENERATE],
    ids=[name for name, _, _ in DEGENERATE],
)
def test_every_degenerate_document_actually_differs_from_the_reference(
    name: str, document: str
) -> None:
    """The other half of the anti-vacuity guard, and it has already fired once.

    Each entry is built by `GOOD.replace(...)`, which is silent when its
    pattern stops matching -- so moving a key out of the reference document
    turned one of these into an exact copy of a *valid* study, and the case
    went on passing until the key it targeted was removed entirely. The same
    failure the mutation harness reports as PATTERN NOT FOUND rather than as a
    pass.
    """
    assert document != GOOD, f"{name!r} is byte-identical to the valid document"


def test_a_cocp_contender_at_float32_fails_at_parse_time(tmp_path: Path) -> None:
    """`require_supported_precision`'s **second and last** mandated call site
    (plan §9.2b). COCP's cone solver loses its correctness canary at float32
    and the failure is silent -- wrong gradients, not an exception -- so the
    pairing is refused while reading the document rather than at the first
    backward pass."""
    document = (
        GOOD.replace('precision = "float64"', 'precision = "float32"') + COCP_CONTENDER
    )
    with pytest.raises(SpecificationError) as error:
        load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)
    message = str(error.value)
    assert "cocp" in message and "float32" in message and "float64" in message


def test_a_cocp_contender_at_float64_is_accepted(tmp_path: Path) -> None:
    """The positive direction, and it is not decoration: the refusal above
    would also pass if COCP were simply unloadable, which -- before the
    contender was given a plan -- is exactly what it was."""
    study = _load(tmp_path, GOOD + COCP_CONTENDER).study
    assert "cocp" in {c.resolved_label for c in study.contenders}


# --------------------------------------------------------------------------
# The three editing levels, in increasing locality (Annex 01 §2.3.1)
# --------------------------------------------------------------------------


def test_the_per_study_overlay_beats_the_catalogue(tmp_path: Path) -> None:
    document = _load(tmp_path)
    resolved = document.resolve(DEFAULT_TIER_CATALOGUE, "smoke")
    assert resolved.study.training.plan.epochs == 1  # the study's, not smoke's 5


def test_the_invocation_beats_the_per_study_overlay(tmp_path: Path) -> None:
    document = _load(tmp_path)
    resolved = document.resolve(
        DEFAULT_TIER_CATALOGUE, "smoke", overrides={"training.plan.epochs": 42}
    )
    assert resolved.study.training.plan.epochs == 42


def test_a_tier_the_study_does_not_override_still_applies(tmp_path: Path) -> None:
    """The overlay is a refinement of the catalogue, not a replacement for it."""
    resolved = _load(tmp_path).resolve(DEFAULT_TIER_CATALOGUE, "standard")
    assert resolved.study.training.plan.epochs == 50


def test_the_per_study_overlay_obeys_the_whitelist(tmp_path: Path) -> None:
    """Annex 01 §2.3.1: all three levels write to the same closed set of paths.
    Asserted at load time, so a study file cannot ship a forbidden override and
    fail only for whoever runs that tier."""
    document = GOOD.replace('"training.plan.epochs" = 1', '"problem.data.horizon" = 2')
    with pytest.raises(SpecificationError) as error:
        load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)
    assert "problem.data.horizon" in str(error.value)


# --------------------------------------------------------------------------
# The tracked catalogue
# --------------------------------------------------------------------------


def test_the_tracked_catalogue_loads() -> None:
    catalogue = load_tier_catalogue(STUDIES / "_tiers.toml")
    assert set(catalogue.tiers) == set(DEFAULT_TIER_CATALOGUE.tiers)


def test_the_tracked_catalogue_is_the_shipped_default() -> None:
    """The code constant is the fallback for the tracked file, so the two
    disagreeing means a run's tier depends on whether the file was found."""
    catalogue = load_tier_catalogue(STUDIES / "_tiers.toml")
    for name, tier in DEFAULT_TIER_CATALOGUE.tiers.items():
        assert dict(catalogue.tiers[name].overrides) == dict(tier.overrides)
        assert catalogue.tiers[name].smoke_subset == tier.smoke_subset


def test_a_catalogue_may_not_smuggle_a_subset_into_another_tier(
    tmp_path: Path,
) -> None:
    path = tmp_path / "_tiers.toml"
    path.write_text(
        '[publication]\n"training.seeds" = 5\n\n'
        "[publication.smoke_subset]\nmax_points_per_axis = 2\n"
    )
    with pytest.raises(SpecificationError) as error:
        load_tier_catalogue(path)
    assert "publication" in str(error.value)


# --------------------------------------------------------------------------
# NB04, declaratively — independent recomputation against the live module
# --------------------------------------------------------------------------


def test_nb04_declared_in_toml_is_the_same_contender_set() -> None:
    """Parent §7.2's Stage 2 deliverable: *express one existing study
    declaratively*. Checked against what `nb04_box_constrained` builds today,
    contender by contender and by resolved signature -- not against a tree this
    file writes down, which would only assert that the transcription matches
    itself."""
    document = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    )
    today = nb04.nb04_box_constrained_experiment(nb04.NB04Config())
    declared = {c.resolved_label: c for c in document.study.contenders}
    expected = {c.resolved_label: c for c in today.contenders}
    assert set(declared) == set(expected)
    for label, spec in expected.items():
        assert declared[label].get_signature() == spec.get_signature(), label


def test_nb04_sweeps_the_three_depth_bearing_contenders() -> None:
    """NB04's real shape, which is what made E1's defect visible: six
    contenders over eight depths, of which only three carry a depth, is 27
    points and not 48."""
    study = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    ).study
    points = study.materialise()
    assert len({p.model_id for p in points}) == 3 * 8 + 3


def test_nb04_declares_the_solver_its_gradients_come_from() -> None:
    """The reference document says which conic solver produces its gradients.

    A study *may* leave `solver_backend` to the default and still be correct —
    the default is itself a declaration, and it is signed. This document spells
    it out because it is the reference for what the grammar can say, and the
    one field whose omission hides something: which backend a COCP model was
    trained with used to be chosen by probing the machine and then discarded,
    which is how this very study once trained `cocp` on DIFFCP and
    `cocp_lower_bound` on MOREAU (Stage 2 Phase G-4, D17).

    Asserted so that dropping the declaration in a later edit is a failing test
    rather than a quiet return to an invisible default.
    """
    study = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    ).study
    declaring = {
        contender.resolved_label: dict(contender.config)
        for contender in study.contenders
        if contender.family in {"cocp", "cocp_lower_bound"}
    }
    assert set(declaring) == {"cocp", "cocp_lower_bound"}
    for label, config in declaring.items():
        assert config["solver_backend"] == "MOREAU", label
        assert config["solver_device"] == "cpu", label


def test_spelling_the_backend_out_derives_the_same_study() -> None:
    """Independent recomputation of "a default is a default", on the tracked
    document rather than a fixture: removing the two explicit declarations must
    give the same `StudyID` and the same points, since the values spelled out
    are the defaults.

    This is what makes the declaration above free, and it is checked rather than
    asserted in a comment — the same claim was measured with the cross-branch
    probe when the declaration was added, and this keeps it true.
    """
    source = (STUDIES / "box_lqr" / "depth_scaling.toml").read_text()
    stripped = source.replace('solver_backend = "MOREAU"\nsolver_device = "cpu"\n', "")
    assert stripped != source, "the declaration anchor moved; this test strips nothing"
    # The *assignments*, not every mention: the prose above the contender
    # explains why the field is spelled out and must survive the strip.
    assert stripped.count("solver_backend = ") == 0

    declared = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    ).study
    with TemporaryDirectory() as directory:
        path = Path(directory) / "depth_scaling.toml"
        path.write_text(stripped)
        copyfile(
            STUDIES / "box_lqr" / "box_lqr_n4m2_N50_u0.5_s0.npz",
            path.parent / "box_lqr_n4m2_N50_u0.5_s0.npz",
        )
        omitted = load_study(path, bindings=DEFAULT_SPEC_BINDINGS).study

    assert declared.study_id == omitted.study_id
    assert [str(p.model_id) for p in declared.materialise()] == [
        str(p.model_id) for p in omitted.materialise()
    ]


def test_the_build_key_is_reserved_and_never_reaches_a_recipe(tmp_path: Path) -> None:
    """`_build` is the one reserved key in a configuration table. Leaving it in
    the configuration handed to a recipe would put it into the contender's
    signature, and every id derived from this surface would differ from the
    same study composed in Python."""
    study = _load(tmp_path).study
    unfolded = next(c for c in study.contenders if c.resolved_label == "unfolded_a")
    assert BUILD_KEY not in unfolded.config
    assert BUILD_KEY not in repr(unfolded.get_signature())


def test_the_bindings_are_injected_not_imported() -> None:
    """Tier 3 may not reach into `engine`/`experiments`/`applications`, so the
    loader cannot construct a plan or a protocol itself. The concrete binding
    is supplied by Tier 4, exactly as the recipe registry already is."""
    with pytest.raises(TypeError):
        load_study(STUDIES / "box_lqr" / "depth_scaling.toml")  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# The injection seam itself
# --------------------------------------------------------------------------


@dataclass
class RecordingBindings:
    """A `SpecBindings` that is not the default one, and remembers.

    Every other test in this module passes `DEFAULT_SPEC_BINDINGS`, which is
    also what a caller would reach for -- so none of them can tell whether the
    argument is consulted at all. This one differs from the default in the
    property under test, which is the rule that the "explicit argument" trap
    taught: an override supplied as the very object the implementation would
    have defaulted to holds whether or not it is used.
    """

    registry: object = REGISTRY
    seen: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def build(
        self,
        kind: str,
        declaration: Mapping[str, object],
        context: Mapping[str, object],
    ) -> object:
        self.seen.append((kind, dict(declaration)))
        return DEFAULT_SPEC_BINDINGS.build(kind, declaration, context)


def test_the_reserved_key_never_reaches_a_builder(tmp_path: Path) -> None:
    """`_build` says what to make of a table; it is not part of the
    declaration. A builder that received it would be one refactor away from
    forwarding it into a constructed object's signature, at which point every
    id derived from a document would differ from the same study written in
    Python -- silently, since both would still be self-consistent."""
    bindings = RecordingBindings()
    load_study(_write(tmp_path, GOOD), bindings=bindings)
    assert bindings.seen, "the injected bindings were never called"
    assert {kind for kind, _ in bindings.seen} == {"end_to_end", "gaussian", "protocol"}
    for kind, declaration in bindings.seen:
        assert BUILD_KEY not in declaration, kind


def test_a_missing_study_file_is_refused_by_name(tmp_path: Path) -> None:
    """Distinct from a missing *problem* file, which has its own check: this is
    the study document itself, and it is the first thing a mistyped `--study`
    hits."""
    with pytest.raises(SpecificationError) as error:
        load_study(tmp_path / "absent.toml", bindings=DEFAULT_SPEC_BINDINGS)
    assert "absent.toml" in str(error.value)


def test_a_missing_catalogue_file_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(SpecificationError) as error:
        load_tier_catalogue(tmp_path / "absent.toml")
    assert "absent.toml" in str(error.value)


def test_a_catalogue_declaring_no_tiers_is_refused(tmp_path: Path) -> None:
    """Degeneracy for the catalogue reader. A file holding only a version is
    not a catalogue, and accepting it would make every `--tier` fail later with
    "no tier named ..." while the real fault is one file away."""
    path = tmp_path / "_tiers.toml"
    path.write_text('version = "1"\n')
    with pytest.raises(SpecificationError) as error:
        load_tier_catalogue(path)
    assert "no tiers" in str(error.value)


def test_the_catalogue_carries_its_own_version(tmp_path: Path) -> None:
    """`tier_catalogue_version` is recorded per model so that "why did this
    retrain?" has an answer (Annex 01 §2.3.1). A catalogue that dropped its own
    version would answer it with an empty string."""
    catalogue = load_tier_catalogue(STUDIES / "_tiers.toml")
    assert catalogue.version == "1"
    resolved = catalogue.resolve(_load(tmp_path).study, "smoke")
    assert resolved.provenance["tier_catalogue_version"] == "1"


def test_a_bad_enum_value_names_the_field_and_the_alternatives(
    tmp_path: Path,
) -> None:
    """The message an author gets for the single most likely typo in a
    contender configuration. Asserted on content rather than on type, because
    the recipe constructors raise `ValueError` for a dozen other reasons."""
    document = GOOD.replace('kind = "learned_step_size"', 'kind = "learned_step"')
    with pytest.raises(SpecificationError) as error:
        load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)
    message = str(error.value)
    assert "kind" in message and "learned_step" in message
    assert "learned_step_size" in message, "the message must show what was meant"


# --------------------------------------------------------------------------
# F2a — the tracked catalogue against the tracked study
# --------------------------------------------------------------------------


def _executed_plans(study: StudySpec) -> dict[str, Any]:
    """What each contender will actually be trained under.

    Read off `contenders.*.config`, because that is the plan
    `TrainableRecipe.build_engine` hands to the optimiser. Reading
    `training.plan` is exactly what made this defect invisible: the study plan
    dropped to 2 epochs at `smoke` while every learned contender trained for
    its declared 200.
    """
    executed: dict[str, Any] = {}
    for contender in study.contenders:
        config = dict(contender.config)
        plan = config.get("plan") or config.get("schedule")
        if plan is not None:
            executed[contender.resolved_label] = plan.get_signature()
    return executed


def test_the_tracked_tiers_reduce_the_tracked_study_s_real_effort() -> None:
    """**The decisive check of Phase F2a**, against the two documents this
    repository actually ships rather than any fixture.

    Before F2a this study resolved at `smoke` to a 2-epoch study plan while
    `unfolded_alpha` still declared 200, `unfolded_alpha_p` 25 + 100 and `cocp`
    100 — the tier moved every `ModelID` and changed no training effort at all,
    which inverts D15 rather than bending it. A `smoke` run of NB04 through the
    producer cost **146.9 s** as a result.
    """
    document = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    )
    catalogue = load_tier_catalogue(STUDIES / "_tiers.toml")

    declared = _executed_plans(document.study)
    smoked = _executed_plans(document.resolve(catalogue, "smoke").study)

    assert (
        set(declared) == set(smoked) == {"unfolded_alpha", "unfolded_alpha_p", "cocp"}
    )
    assert declared != smoked, (
        "the tier reached none of the plans that execute, which is the whole "
        "of the defect this phase exists to fix"
    )
    for label in declared:
        assert _epoch_budget(smoked[label]) < _epoch_budget(declared[label]), (
            f"{label} trains for no fewer epochs at smoke than as declared"
        )


def _epoch_budget(signature: Mapping[str, Any]) -> int:
    """Total epochs a plan signature buys, end-to-end or layer-wise.

    The two plan types share no epoch field — `TrainingPlan` has `epochs`,
    `LayerwiseTrainingPlan` has `warmup_epochs_per_layer` and
    `refinement_epochs` — which is the reason `TrainingSpec.plan` is typed
    `Signable` in the first place, and the reason the catalogue needs a
    separate path for each.
    """
    if "epochs" in signature:
        return int(signature["epochs"])
    return int(signature["warmup_epochs_per_layer"]) + int(
        signature["refinement_epochs"]
    )


def test_nb04_s_own_overlay_beats_the_catalogue_on_its_own_contenders() -> None:
    """Annex 01 §2.3.1's increasing locality, on the tracked pair of documents.

    NB04 declares a smoke overlay because COCP's QP is far slower per epoch
    than an unrolling, so the catalogue's shared epoch count is the wrong
    wiring check for *this* study. That overlay wrote `training.plan.epochs`
    until F2a and therefore reduced nothing at all.

    Asserted against the catalogue's own value rather than a literal, so the
    test states the *relationship* -- the overlay wins -- and does not have to
    be edited whenever either document is retuned.
    """
    document = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    )
    catalogue = load_tier_catalogue(STUDIES / "_tiers.toml")

    overlay = document.tier_overrides["smoke"]
    shared = catalogue.get("smoke").overrides
    path = "contenders.*.config.plan.epochs"
    assert path in overlay and path in shared, (
        "this study no longer overlays the executed plan, so the precedence "
        "below is not being exercised"
    )
    assert overlay[path] != shared[path], (
        "the overlay repeats the catalogue's value, so it could be deleted "
        "with no effect and this test would not notice"
    )

    resolved = document.resolve(catalogue, "smoke").study
    for label in ("cocp", "unfolded_alpha"):
        contender = next(c for c in resolved.contenders if c.resolved_label == label)
        assert dict(contender.config)["plan"].epochs == overlay[path]


def _tracked_studies() -> list[Path]:
    """Every study document this repository ships.

    Discovered rather than listed: the ICASSP campaign adds one document per
    figure, and a hand-maintained list would let a new one ship unresolvable
    at a tier nobody ran it at.
    """
    return sorted(path for path in STUDIES.glob("*/*.toml"))


def test_there_are_tracked_studies_to_check() -> None:
    """The discovery guards itself — a glob that stopped matching would make
    the parametrisation below vacuous while the suite stayed green."""
    assert len(_tracked_studies()) >= 2, _tracked_studies()


@pytest.mark.parametrize(
    "document_path", _tracked_studies(), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_every_tracked_tier_still_resolves_every_tracked_study(
    document_path: Path,
) -> None:
    """A catalogue entry naming a path no contender has must not stop a study
    from running: the catalogue is shared, and a study with no layer-wise
    contender still has to resolve at every tier.

    Every tracked document, at every tier, because the failure this catches is
    silent until someone runs that combination — and the campaign's documents
    are run at `publication` weeks after they are written.
    """
    document = load_study(document_path, bindings=DEFAULT_SPEC_BINDINGS)
    catalogue = load_tier_catalogue(STUDIES / "_tiers.toml")
    for tier in sorted(catalogue.tiers):
        assert document.resolve(catalogue, tier).study.materialise(), tier


def test_a_study_overlay_naming_no_contender_is_refused_at_load(
    tmp_path: Path,
) -> None:
    """The strict half, on the surface an author edits. `[tier_overrides.…]` is
    written against one study, so a wildcard matching nobody is a typo — and it
    must fail when the tier is applied, naming the path."""
    document = _with_overlay_key("contenders.*.config.plan.epcohs")
    loaded = load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)
    with pytest.raises(SpecificationError) as error:
        loaded.resolve(DEFAULT_TIER_CATALOGUE, "smoke")
    assert "contenders.*.config.plan.epcohs" in str(error.value)


def _with_overlay_key(path: str) -> str:
    """`GOOD` with one more key in the smoke overlay it already declares.

    Asserted to have actually changed the document: a fixture derived by
    `str.replace` goes silently inert when its pattern stops matching, and a
    "must be refused" case that has quietly become a byte-identical copy of a
    valid document goes on passing.
    """
    anchor = '[tier_overrides.smoke]\n"training.plan.epochs" = 1'
    derived = GOOD.replace(anchor, f'{anchor}\n"{path}" = 1')
    assert derived != GOOD, "the overlay anchor moved; this fixture writes nothing"
    return derived


def test_the_same_overlay_spelled_correctly_resolves(tmp_path: Path) -> None:
    """Anti-vacuity for the refusal above."""
    document = _with_overlay_key("contenders.*.config.plan.epochs")
    loaded = load_study(_write(tmp_path, document), bindings=DEFAULT_SPEC_BINDINGS)
    resolved = loaded.resolve(DEFAULT_TIER_CATALOGUE, "smoke").study
    learned = next(c for c in resolved.contenders if "plan" in dict(c.config))
    assert dict(learned.config)["plan"].epochs == 1


# --------------------------------------------------------------------------
# The study's default plan — Stage 2 Phase G-2
# --------------------------------------------------------------------------


def _without_the_contender_plan(document: str = GOOD) -> str:
    """`GOOD`, with `unfolded_a`'s own plan removed.

    Derived by substitution, so it asserts that it actually changed something:
    a fixture whose pattern stops matching goes silently inert and a "this must
    now inherit" case becomes a copy of a document that already declared one.
    """
    anchor = """[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.001
epochs = 3
"""
    derived = document.replace(anchor, "", 1)
    assert derived != document, "the contender plan anchor moved"
    return derived


def test_a_contender_that_declares_no_plan_inherits_the_study_s(
    tmp_path: Path,
) -> None:
    """The document this makes writable: one fitting procedure, declared once,
    shared by every contender that trains. Before G-2 it was unwritable --
    `plan` is a required field of every family that takes one, so omitting it
    failed at parse time with a constructor error."""
    study = _load(tmp_path, _without_the_contender_plan()).study
    learned = next(c for c in study.contenders if c.resolved_label == "unfolded_a")
    assert dict(learned.config)["plan"].epochs == 3
    assert learned.resolve().plan.epochs == 3


def test_omitting_the_contender_plan_derives_the_same_study(tmp_path: Path) -> None:
    """Independent recomputation on the document surface: a study that inherits
    the default and one that spells the identical plan into the contender are
    the same study, `StudyID` and every point."""
    inherited = _load(tmp_path / "a", _without_the_contender_plan()).study
    spelled = _load(tmp_path / "b", GOOD).study
    assert inherited.study_id == spelled.study_id
    assert [str(p.model_id) for p in inherited.materialise()] == [
        str(p.model_id) for p in spelled.materialise()
    ]


def test_a_contender_that_cannot_be_composed_is_still_refused_by_position(
    tmp_path: Path,
) -> None:
    """The loader's parse-time schema check runs *after* materialisation now,
    and must not have lost the contender's position on the way: without it, a
    bad configuration surfaces as a bare constructor error at the first
    `get_signature()`, which under a study runner is an hour into the run."""
    broken = _without_the_contender_plan().replace(
        "num_iterations = 3", "num_iterations = 3\nnot_a_field = 1", 1
    )
    with pytest.raises(SpecificationError, match=r"contenders\[0\]"):
        _load(tmp_path, broken)


def test_a_layerwise_contender_is_not_given_the_study_plan(tmp_path: Path) -> None:
    """The plan-vs-schedule rule, on the document surface: a study-level
    end-to-end plan cannot be injected into a layer-wise family, and the tracked
    NB04 document is exactly that case."""
    document = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    )
    flagship = next(
        c for c in document.study.contenders if c.resolved_label == "unfolded_alpha_p"
    )
    config = dict(flagship.config)
    assert "plan" not in config
    assert config["schedule"].warmup_epochs_per_layer == 25


# --------------------------------------------------------------------------
# `[[analyses]]` and `[[figures]]` — Annex 01 §2.6 (slice Phase B-1)
# --------------------------------------------------------------------------

REPORTING = """
[[analyses]]
id = "cost_by_depth"
kind = "cost_vs_axis"
axis = "contenders.*.config.num_iterations"
aggregate = "mean"

[[figures]]
id = "fig_cost_vs_depth"
kind = "axis_scaling"
source = "cost_by_depth"
caption = "Cost against unrolling depth."
"""


def test_a_document_may_declare_analyses_and_figures(tmp_path: Path) -> None:
    study = _load(tmp_path, GOOD + REPORTING).study

    assert [spec.id for spec in study.analyses] == ["cost_by_depth"]
    assert study.analyses[0].kind == "cost_vs_axis"
    assert study.figures[0].source == "cost_by_depth"


def test_the_reserved_keys_are_lifted_out_of_the_config(tmp_path: Path) -> None:
    """`id`, `kind` and `source` are structure, not configuration.

    Leaving `source` in `config` would make a load-bearing reference -- the
    one that decides which table the figure renders -- indistinguishable from
    a style option, and a renderer reading `config["source"]` would work by
    accident until someone reordered the keys.
    """
    study = _load(tmp_path, GOOD + REPORTING).study

    assert set(study.analyses[0].config) == {"axis", "aggregate"}
    assert set(study.figures[0].config) == {"caption"}


def test_a_document_declaring_neither_is_unchanged(tmp_path: Path) -> None:
    study = _load(tmp_path).study
    assert study.analyses == ()
    assert study.figures == ()


def test_a_figure_naming_an_undeclared_analysis_is_refused(tmp_path: Path) -> None:
    broken = GOOD + REPORTING.replace('source = "cost_by_depth"', 'source = "typo"')
    with pytest.raises(SpecificationError, match="typo"):
        _load(tmp_path, broken)


def test_an_analysis_without_a_kind_is_refused_by_position(tmp_path: Path) -> None:
    broken = GOOD + '\n[[analyses]]\nid = "a"\n'
    with pytest.raises(SpecificationError, match=r"analyses\[0\]"):
        _load(tmp_path, broken)


def test_a_figure_without_a_source_is_refused_by_position(tmp_path: Path) -> None:
    broken = GOOD + '\n[[figures]]\nid = "f"\nkind = "axis_scaling"\n'
    with pytest.raises(SpecificationError, match=r"figures\[0\]"):
        _load(tmp_path, broken)


def test_declaring_them_moves_no_identifier_on_the_document_surface(
    tmp_path: Path,
) -> None:
    """The B-1 property, asserted where an author actually edits.

    The spec-level test proves the exclusion; this one proves the *document*
    reaches it. A loader that dropped `analyses` on the floor would satisfy
    the spec test and this one alike, so the declaration is read back as well
    as the identifier compared.
    """
    bare = _load(tmp_path, GOOD).study
    reported = _load(tmp_path / "b", GOOD + REPORTING).study

    assert reported.analyses, "the loader dropped the declaration"
    assert reported.study_id == bare.study_id
    assert [str(p.model_id) for p in reported.materialise()] == [
        str(p.model_id) for p in bare.materialise()
    ]


def test_the_tracked_study_can_declare_them_without_moving_its_identity(
    tmp_path: Path,
) -> None:
    """B-1 on real data: the document that will actually be re-run.

    `dataclasses.replace` rather than a copied file, because the tracked
    document's `[problem] path` is relative to it and a copy would need the
    `.npz` beside it -- and because re-running `__post_init__` is precisely
    what has to be exercised: the figure's source is validated there.
    """
    del tmp_path
    document = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    )
    declared = document.study
    reported = dataclasses.replace(
        declared,
        analyses=(AnalysisSpec(id="cost_by_depth", kind="cost_vs_axis"),),
        figures=(
            FigureSpec(
                id="fig_cost_vs_depth",
                kind="axis_scaling",
                source="cost_by_depth",
            ),
        ),
    )

    assert reported.study_id == declared.study_id
    assert len(reported.materialise()) == len(declared.materialise())
    assert [str(p.model_id) for p in reported.materialise()] == [
        str(p.model_id) for p in declared.materialise()
    ]


def test_the_tracked_study_declares_its_analysis_and_its_figure() -> None:
    """What NB04 computes and draws, asserted on the tracked document.

    Mutation testing deleted the `[[figures]]` table from the study and the
    whole suite still passed: every figure test built its own document. The
    slice's deliverable is that THIS study produces a figure, so this study is
    what asserts it.
    """
    study = load_study(
        STUDIES / "box_lqr" / "depth_scaling.toml", bindings=DEFAULT_SPEC_BINDINGS
    ).study

    assert [spec.id for spec in study.analyses] == ["cost_by_depth"]
    assert study.analyses[0].kind == "cost_vs_axis"
    assert [spec.id for spec in study.figures] == ["fig_cost_vs_depth"]
    assert study.figures[0].kind == "axis_scaling"
    assert study.figures[0].source == "cost_by_depth"
    # §B.2.1: a figure may say what it is ABOUT and not how wide it is.
    assert not {"width_in", "figsize", "font_size"} & set(study.figures[0].config)
