"""Acceptance tests for the replicate — Stage 2 Phase G-3.

Written before the implementation. The defect being closed is that
`training.seeds` reached `ModelID` and nothing else: a five-seed study derived
five identifiers for five **byte-identical** models and would have reported a
zero-width confidence interval. Annex 01 §2.3.3 states the correction, and it
is a scientific definition rather than a wiring detail — *a replicate redraws
everything a repetition of the experiment would redraw*.

The checkpoint is therefore **two-directional**, because a replicate reaches a
model through two different mechanisms and a fix covering one of them looks
complete:

* a family whose *construction* is random inherits the replicate through the
  global torch RNG — the reason the fix is not an enumeration of per-family
  seed fields, which would leave a family added later silently unseeded;
* a family whose construction is deterministic but whose *training data* is not
  inherits it through the batch sampler. Four of the eight registered families
  declare no seed field at all and two of those four train, so this is not the
  minority case.

Two controls sit under both: every contender at one replicate must draw
**identical batches**, or a comparison between contenders is a comparison of
their luck; and the same replicate run twice must be **bit-identical**, or the
store's whole premise is gone.

The composition itself is tested structurally as well as behaviourally. It must
read `(data.seed, replicate)` and nothing else — a version that consulted the
contender would satisfy every behavioural assertion here except the one about
identical batches, and would pass it too for a study whose contenders happen to
be alike.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mbl.applications.recipes.neural import NeuralRecipe
from mbl.applications.recipes.unfolded import (
    FixedUnfoldedRecipe,
    UnfoldedKind,
    UnfoldedRecipe,
    WarmStartUnfoldedRecipe,
)
from mbl.core.runtime import Backend, ComputeContext, Precision
from mbl.engine.training_plan import LayerwiseTrainingPlan, OptimizerSpec, TrainingPlan
from mbl.experiments import DEFAULT_SPEC_BINDINGS
from mbl.models.iterative.initializers import ControlInitMethod
from mbl.runner.producer import run_study
from mbl.runner.seeds import ReplicateStreams, derive_replicate_streams
from mbl.spec.contender import RecipeRegistry
from mbl.spec.errors import SpecificationError
from mbl.spec.loader import SpecBindings, load_study
from mbl.spec.problem import ProblemData, ProblemSpec
from mbl.spec.study import StudySpec
from mbl.store.content_store import ModelStore

N, M, HORIZON = 4, 2, 6

#: The study's declared training stream. Deliberately **not** zero: at
#: replicate 0 the data child *is* the declared seed, so a zero would make
#: "the stream reached the sampler" and "the composition did nothing" the same
#: observation.
TRAIN_STREAM = 11

#: Enough replicates that "they differ" is a statement about a family of
#: streams rather than about one pair.
REPLICATES = (0, 1, 2, 3, 4, 5, 6, 7)


# -- the composition ---------------------------------------------------------


def test_the_composition_reads_the_replicate_and_nothing_else() -> None:
    """Structural, and it carries the fairness law on its own.

    Every contender at one replicate must train on identical batches, and the
    behavioural test of that can only check the contenders a fixture happens to
    declare. A composition with no parameter through which a contender, a
    family or a recipe could arrive cannot violate it for any study that will
    ever be written — the same reason `derive_model_id`'s absent evaluation
    parameter is stronger than any behavioural test of the split.
    """
    parameters = inspect.signature(derive_replicate_streams).parameters
    assert list(parameters) == ["data_seed", "replicate"], (
        "the composition takes the declared stream and the replicate, exactly; "
        f"got {list(parameters)}"
    )
    # A named pair rather than a bare `tuple[int, int]`: the two streams are
    # interchangeable by position and nothing downstream would notice them
    # swapped -- the batches would still be reproducible, and still wrong.
    assert isinstance(derive_replicate_streams(TRAIN_STREAM, 1), ReplicateStreams)


def test_replicate_zero_draws_the_stream_its_document_names() -> None:
    """A single-replicate study must draw exactly what `training.data.seed`
    says, so that the common case is legible: the document says 11 and the
    sampler is seeded 11."""
    for data_seed in (0, 1, 11, 4242):
        assert derive_replicate_streams(data_seed, 0).data == data_seed


def test_each_replicate_gets_its_own_pair() -> None:
    """The whole point: eight replicates are sixteen distinct streams, not one
    repeated. Checked across two declared roots as well, since two studies
    differing only in `training.data.seed` must not share a replicate family."""
    drawn: list[int] = []
    for data_seed in (0, 11):
        for replicate in REPLICATES:
            streams = derive_replicate_streams(data_seed, replicate)
            drawn.extend((streams.data, streams.weights))
    assert len(set(drawn)) == len(drawn), "two replicates share a stream"


def test_the_two_streams_are_never_one_integer() -> None:
    """Two generators fed the same integer are owed independence by nothing,
    and today's producer does exactly that whenever `training.data.seed` is 0 —
    which is every tracked study."""
    for data_seed in (0, 1, 11, 4242):
        for replicate in REPLICATES:
            streams = derive_replicate_streams(data_seed, replicate)
            assert streams.data != streams.weights, (
                f"seed={data_seed} replicate={replicate} seeds the sampler and "
                "the global RNG with one value"
            )


def test_a_stream_seeds_both_generators_this_project_uses() -> None:
    """A derived stream is useless if the samplers refuse it."""
    for replicate in REPLICATES:
        streams = derive_replicate_streams(TRAIN_STREAM, replicate)
        for stream in (streams.data, streams.weights):
            torch.Generator().manual_seed(stream)
            np.random.default_rng(stream)


def test_a_stream_fits_a_signed_64_bit_integer() -> None:
    """The width is **this project's** contract, not the generators'.

    Measured: `torch.Generator.manual_seed` and `numpy.random.default_rng` both
    accept the full *unsigned* 64-bit range, so nothing downstream of a
    generator would object. What would object is everything downstream of a
    *number*: SQLite's `INTEGER` is signed 64-bit and a JSON integer beyond
    2**63 is a portability hazard for any reader that is not Python. Without
    this assertion `STREAM_BITS` is a declared invariant nothing enforces —
    which is how the surviving mutant of this suite's first pass found it.
    """
    for data_seed in (0, 1, TRAIN_STREAM, 4242):
        for replicate in REPLICATES:
            streams = derive_replicate_streams(data_seed, replicate)
            for stream in (streams.data, streams.weights):
                assert 0 <= stream < 2**63, f"{stream} does not fit an int64"


def test_the_composition_is_deterministic_across_processes(tmp_path: Path) -> None:
    """Recomputing in the building process only shows determinism; a stored
    model is re-derived in another process, on another day (Phase A's lesson).
    """
    expected = [
        (streams.data, streams.weights)
        for streams in (
            derive_replicate_streams(TRAIN_STREAM, replicate)
            for replicate in REPLICATES
        )
    ]
    script = tmp_path / "recompute.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from mbl.runner.seeds import derive_replicate_streams
            for replicate in {list(REPLICATES)!r}:
                streams = derive_replicate_streams({TRAIN_STREAM}, replicate)
                print(streams.data, streams.weights)
            """
        )
    )
    output = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    ).stdout.split()
    assert [(int(a), int(b)) for a, b in zip(output[::2], output[1::2])] == expected


@pytest.mark.parametrize(
    ("data_seed", "replicate", "offender"),
    [(-1, 0, "training.data.seed"), (11, -3, "seed")],
)
def test_a_negative_seed_is_refused_naming_the_field(
    data_seed: int, replicate: int, offender: str
) -> None:
    """Degeneracy. `np.random.default_rng(-1)` already raises, from inside a
    sampler several frames from the document that caused it, and says nothing
    about which of the two seeds was wrong."""
    with pytest.raises(SpecificationError, match=offender):
        derive_replicate_streams(data_seed, replicate)


# -- the pin -----------------------------------------------------------------


def _ctx() -> ComputeContext:
    return ComputeContext(
        backend=Backend.TORCH, device="cpu", precision=Precision.FLOAT64
    )


def _plan() -> TrainingPlan:
    return TrainingPlan(optimizer=OptimizerSpec("adam", 1e-2), epochs=1)


def _built_problem() -> Any:
    return _problem_spec(seed=0).build()


#: Every recipe that declares a seed field, at a configuration where the field
#: is actually consumed. A negative control must cover every instance the fix
#: claims to cover, not one representative: `random_init_seed` is inert under
#: the default cold start, so each unfolded entry is built `RANDOMIZED`.
PINNABLE: dict[str, tuple[type, str, dict[str, Any]]] = {
    "neural": (NeuralRecipe, "init_seed", {"hidden_dim": 8, "plan": _plan()}),
    "unfolded": (
        UnfoldedRecipe,
        "random_init_seed",
        {
            "kind": UnfoldedKind.LEARNED_STEP_SIZE,
            "plan": _plan(),
            "num_iterations": 3,
            "step_size_init": 0.1,
            "step_size_max": 1.0,
            "horizon": HORIZON,
            "init_method": ControlInitMethod.RANDOMIZED,
        },
    ),
    "unfolded_fixed": (
        FixedUnfoldedRecipe,
        "random_init_seed",
        {
            "num_iterations": 3,
            "step_size_init": 0.1,
            "step_size_max": 1.0,
            "horizon": HORIZON,
            "init_method": ControlInitMethod.RANDOMIZED,
        },
    ),
    "unfolded_warmstart": (
        WarmStartUnfoldedRecipe,
        "random_init_seed",
        {
            "kind": UnfoldedKind.LEARNED_STEP_SIZE,
            "schedule": LayerwiseTrainingPlan(
                optimizer=OptimizerSpec("adam", 1e-2),
                warmup_epochs_per_layer=1,
                refinement_epochs=1,
            ),
            "num_iterations": 3,
            "step_size_init": 0.1,
            "step_size_max": 1.0,
            "horizon": HORIZON,
            "init_method": ControlInitMethod.RANDOMIZED,
        },
    ),
}


def _initial_control(recipe: Any, problem: Any, ambient: int) -> torch.Tensor:
    """What this recipe constructs under a given ambient global stream.

    Read through the controller rather than through the parameters, because for
    the unfolded families the randomness is in the *initial control iterate*
    and not in a registered parameter: a `state_dict` comparison would be
    identical for all four and would pass however the pin behaved.
    """
    torch.manual_seed(ambient)
    controller = recipe.build_controller(problem, _ctx())
    initializer = getattr(
        getattr(controller, "config", None), "control_initializer", None
    )
    if initializer is not None:
        observation = torch.zeros(1, N, dtype=torch.float64)
        return initializer(0, observation).detach().reshape(-1)
    return torch.cat(
        [value.detach().reshape(-1) for value in controller.state_dict().values()]
    )


@pytest.mark.parametrize("family", sorted(PINNABLE))
def test_a_declared_seed_pins_the_construction(family: str) -> None:
    """The pin's positive direction, for **every** recipe that has one."""
    recipe_cls, field, config = PINNABLE[family]
    recipe = recipe_cls(**config, **{field: 7})
    problem = _built_problem()
    pinned = _initial_control(recipe, problem, ambient=1)
    assert torch.equal(pinned, _initial_control(recipe, problem, ambient=2)), (
        f"{family} declares {field}=7, so the replicate must not reach it"
    )


@pytest.mark.parametrize("family", sorted(PINNABLE))
def test_an_undeclared_seed_inherits_the_replicate(family: str) -> None:
    """The negative direction, and the one that makes the pin test capable of
    failing: unset, the field must *not* fix the construction."""
    recipe_cls, field, config = PINNABLE[family]
    recipe = recipe_cls(**config)
    assert getattr(recipe, field) is None, (
        f"{field} must default to inheriting the replicate, not to a constant"
    )
    problem = _built_problem()
    assert not torch.equal(
        _initial_control(recipe, problem, ambient=1),
        _initial_control(recipe, problem, ambient=2),
    ), f"{family} ignores the ambient stream, so a replicate cannot reach it"


# -- the producer ------------------------------------------------------------

STUDY = """
id = "probe/replicates"

[problem]
path = "problem.npz"

[compute]
backend = "torch"
device = "cpu"
precision = "float64"

[training]
seeds = {seeds}

[training.data]
kind = "gaussian"
seed = {train_stream}
process_noise_std = 0.4
initial_state_std = 0.8

[training.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 2

[training.batch]
effective_size = 16

[evaluation]
metrics = ["expected_cost"]

[evaluation.protocol]
_build = "protocol"
n_batches = 1

[evaluation.protocol.batch_spec]
_build = "gaussian"
batch_size = 16
seed = 5
process_noise_std = 0.4
initial_state_std = 1.0

[[contenders]]
label = "unfolded_a"
family = "unfolded"

[contenders.config]
kind = "learned_step_size"
num_iterations = 2
step_size_init = 0.1
step_size_max = 1.0
horizon = 6

[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 2

[[contenders]]
label = "policy"
family = "neural"

[contenders.config]
hidden_dim = 4

[contenders.config.plan]
_build = "end_to_end"
optimizer = "adam"
learning_rate = 0.05
epochs = 2
"""


def _problem_spec(seed: int) -> ProblemSpec:
    rng = np.random.default_rng(seed)
    return ProblemSpec(
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


def _study(
    directory: Path, *, seeds: tuple[int, ...] = (0, 1), bindings: Any = None
) -> StudySpec:
    directory.mkdir(parents=True, exist_ok=True)
    _problem_spec(seed=0).save(directory / "problem.npz")
    path = directory / "study.toml"
    path.write_text(STUDY.format(seeds=list(seeds), train_stream=TRAIN_STREAM))
    return load_study(path, bindings=bindings or DEFAULT_SPEC_BINDINGS).study


class RecordingBindings:
    """The real vocabulary, with every declaration it builds written down.

    Wraps `DEFAULT_SPEC_BINDINGS` and calls through, for the same reason the
    producer's synthesis double does: a replacement would let the suite assert
    a property of the replacement. The seed the training sampler is built with
    is not otherwise observable from outside the producer, and asserting on
    the *weights* instead would confound the sampler with everything else the
    replicate touches.
    """

    def __init__(self) -> None:
        self.built: list[tuple[str, Mapping[str, Any]]] = []

    @property
    def registry(self) -> RecipeRegistry:
        return DEFAULT_SPEC_BINDINGS.registry

    def build(
        self, kind: str, declaration: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Any:
        self.built.append((kind, dict(declaration)))
        return DEFAULT_SPEC_BINDINGS.build(kind, declaration, context)

    def streams(self) -> list[int]:
        """The seeds handed to a sampler, in the order they were built."""
        return [
            int(declaration["seed"])
            for kind, declaration in self.built
            if kind == "gaussian"
        ]


def _run(study: StudySpec, store: Path, bindings: SpecBindings | None = None) -> Any:
    return run_study(study, store=store, bindings=bindings or DEFAULT_SPEC_BINDINGS)


def _weights(study: StudySpec, store: Path, label: str) -> dict[int, Any]:
    """Every replicate's stored weights for one contender, by replicate."""
    models = ModelStore(store)
    return {
        point.seed: dict(models.get(str(point.model_id)).weights)
        for point in study.materialise()
        if point.contender.resolved_label == label
    }


def _differ(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    assert set(first) == set(second) and first, "nothing to compare"
    return any(not torch.equal(first[key], second[key]) for key in first)


def test_every_contender_at_one_replicate_draws_the_same_batches(
    tmp_path: Path,
) -> None:
    """The fairness law, measured at the sampler the trainer is handed.

    Two contenders of different families, two replicates: the streams must
    group by replicate and not by contender. A composition consulting the
    family would give four distinct streams and every other test here would
    still pass.
    """
    bindings = RecordingBindings()
    study = _study(tmp_path / "doc", bindings=bindings)
    bindings.built.clear()  # the document's own protocol was built at load time
    report = _run(study, tmp_path / "store", bindings=bindings)

    streams = bindings.streams()
    assert len(streams) == len(report.outcomes) == 4
    by_replicate: dict[int, set[int]] = {}
    for outcome, stream in zip(report.outcomes, streams, strict=True):
        by_replicate.setdefault(outcome.seed, set()).add(stream)
    assert all(len(seen) == 1 for seen in by_replicate.values()), (
        f"contenders at one replicate drew different batches: {by_replicate}"
    )
    assert len({stream for seen in by_replicate.values() for stream in seen}) == 2, (
        "both replicates drew the same batches, so the replicate reached nothing"
    )


def test_the_declared_stream_is_the_root_and_not_the_stream_of_every_replicate(
    tmp_path: Path,
) -> None:
    """The composition must be observable in what the sampler actually got:
    replicate 0 draws the declared seed and replicate 1 does not."""
    bindings = RecordingBindings()
    study = _study(tmp_path / "doc", bindings=bindings)
    bindings.built.clear()
    report = _run(study, tmp_path / "store", bindings=bindings)

    by_replicate = {
        outcome.seed: stream
        for outcome, stream in zip(report.outcomes, bindings.streams(), strict=True)
    }
    assert by_replicate[0] == TRAIN_STREAM
    assert by_replicate[1] != TRAIN_STREAM
    assert by_replicate[1] == derive_replicate_streams(TRAIN_STREAM, 1).data


def test_a_replicate_redraws_a_deterministically_constructed_model(
    tmp_path: Path,
) -> None:
    """Direction one. `unfolded` at the default cold start builds bit-identical
    controllers whatever the RNG says, so its replicates can only differ
    through the training data — which is the case a per-family seed field could
    never have covered, and which four of the eight families are in."""
    study = _study(tmp_path / "doc")
    _run(study, tmp_path / "store")
    weights = _weights(study, tmp_path / "store", "unfolded_a")

    assert sorted(weights) == [0, 1]
    assert _differ(weights[0], weights[1]), (
        "two replicates of a cold-started unfolded contender trained to the "
        "same weights, so `training.seeds` still reaches nothing but the id"
    )


def test_a_replicate_redraws_a_randomly_constructed_model(tmp_path: Path) -> None:
    """Direction two, through the global torch RNG rather than the sampler."""
    study = _study(tmp_path / "doc")
    _run(study, tmp_path / "store")
    weights = _weights(study, tmp_path / "store", "policy")

    assert sorted(weights) == [0, 1]
    assert _differ(weights[0], weights[1])


def test_the_same_replicate_twice_is_bit_identical(tmp_path: Path) -> None:
    """The determinism control. Without it every assertion above is satisfied
    by a producer that simply seeds from the clock."""
    study = _study(tmp_path / "doc")
    _run(study, tmp_path / "first")
    _run(study, tmp_path / "second")

    for label in ("unfolded_a", "policy"):
        first = _weights(study, tmp_path / "first", label)
        second = _weights(study, tmp_path / "second", label)
        for replicate in sorted(first):
            assert not _differ(first[replicate], second[replicate]), (
                f"{label} replicate {replicate} is not reproducible"
            )


def test_the_replicates_are_one_study_and_not_two(tmp_path: Path) -> None:
    """End-to-end closure: a two-seed run publishes two models per contender
    under two identifiers, and re-running trains none of them."""
    study = _study(tmp_path / "doc")
    store = tmp_path / "store"
    first = _run(study, store)
    second = _run(study, store)

    assert len(first.trained) == 4
    assert len(second.trained) == 0
    assert len({str(point.model_id) for point in study.materialise()}) == 4
