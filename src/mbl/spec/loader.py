"""The TOML surface over the grammar (Stage 2 Phase E3).

A study is a *value* (P1), so it must be writable as a document — catalogued in
git, reviewed in a diff, and read back months later. This module is the reader,
and the errors it raises are its main deliverable: a grammar whose failures are
unreadable is a grammar nobody uses, so every refusal names the key that caused
it and, where the format itself is broken, the line.

**Why TOML and not YAML** (Annex 01 §3, ratified 2026-08-01). Not portability
across operating systems — both are plain UTF-8 text and neither carries an OS
constraint. `tomllib` implements TOML 1.0, which has one published semantics,
while the available YAML parser implements YAML 1.1, where `no` and `off` are
booleans and `12:30` is 750. Identity is derived from these documents; a format
whose meaning depends on which library opened it is the wrong substrate. And
`tomllib` is standard library, so the surface costs no dependency at all.

**The one seam: Tier 3 parses but never constructs what it may not import.**
A document says ``optimizer = "adam"``, ``epochs = 200``, ``n_batches = 8``.
Turning those into a `TrainingSpec` means building `engine.TrainingPlan` and
`experiments.EvaluationProtocol`, both above this tier. Rather than three
injected builders, there is one vocabulary: any table carrying the reserved key
`_build` is handed to the injected `SpecBindings.build`, which knows what the
name means and this module does not. That is the same dependency inversion the
recipe registry already uses (plan §8.1), collected into a single point, and it
means extending the buildable vocabulary is a Tier-4 change that never touches
the grammar.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ..core.runtime import Backend, ComputeContext, Precision
from ..core.utils.signing import Signable
from .analysis import AnalysisSpec
from .contender import ContenderSpec, RecipeRegistry, Role
from .data import DEFAULT_DATA_KIND, DataSpec
from .errors import SpecificationError
from .evaluation import (
    DEFAULT_METRICS,
    REHOST_BLIND,
    EvaluationSpec,
    RehostOverride,
)
from .figure import FigureSpec
from .gates import GateSpec
from .problem import ProblemSpec
from .study import Composition, StudySpec, SweepAxis
from .tiers import (
    ResolvedStudy,
    SmokeSubset,
    Tier,
    TierCatalogue,
    require_permitted_paths,
)
from .training import BatchPlan, TrainingSpec, require_supported_precision

#: The reserved key marking a table this tier cannot build itself. Everything
#: else in such a table is the declaration; `_build` names what to make of it.
#: It is stripped before the table is used, because leaving it in a contender's
#: configuration would put it into that contender's signature and make every id
#: derived from this surface differ from the same study composed in Python.
BUILD_KEY = "_build"

#: The key under which a study declares its per-tier overlay (Annex 01 §2.3.1,
#: the middle of the three editing levels).
TIER_OVERRIDES_KEY = "tier_overrides"


class SpecBindings(Protocol):
    """What Tier 3 needs from the tiers above it to read a document.

    Supplied by the caller; `mbl.experiments.DEFAULT_SPEC_BINDINGS` is this
    project's own. The grammar may not import `engine`, `experiments` or
    `applications`, so it declares what it needs of them structurally and lets
    the concrete objects be injected -- the same resolution `ContenderSpec`
    already uses for the recipe registry.
    """

    @property
    def registry(self) -> RecipeRegistry:
        """The recipe registry contender families resolve through."""

    def build(
        self, kind: str, declaration: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Any:
        """Construct the object `kind` names from its declaration.

        Args:
            kind: The `_build` value, e.g. ``"end_to_end"`` or ``"gaussian"``.
            declaration: The rest of the table.
            context: What the builder may need but the document must not
                restate -- the problem's `state_dim` and `horizon`. Letting a
                document declare a dimension is letting it disagree with the
                matrices.

        Raises:
            SpecificationError: If `kind` is unknown or the declaration is
                incomplete, naming the offender.
        """


# -- reading primitives, each naming the key it refuses ---------------------


def _table(document: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = document.get(key)
    if value is None:
        raise SpecificationError(f"{_at(where, key)} is missing; it is required")
    if not isinstance(value, dict):
        raise SpecificationError(
            f"{_at(where, key)} must be a table, got {type(value).__name__}"
        )
    return value


def _at(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _string(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise SpecificationError(
            f"{_at(where, key)} must be a non-empty string, got {value!r}"
        )
    return value


def _optional_string(table: Mapping[str, Any], key: str, where: str) -> str | None:
    """A key that may be absent, but may not be present and malformed.

    Absent is `None`. Present goes through `_string`, so an empty string or a
    number is refused with the same message a required key would give -- an
    optional field whose bad value is silently ignored is the shape that lets a
    document say something the run does not do.
    """
    if key not in table:
        return None
    return _string(table, key, where)


def _sequence(table: Mapping[str, Any], key: str, where: str) -> tuple[Any, ...]:
    value = table.get(key, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SpecificationError(f"{_at(where, key)} must be a list, got {value!r}")
    return tuple(value)


def _enum(kind: type[StrEnum], value: str, where: str) -> Any:
    """A string into one of the grammar's own enums, naming the alternatives."""
    try:
        return kind(value)
    except ValueError:
        raise SpecificationError(
            f"{where} is {value!r}, which is not one of "
            f"{sorted(member.value for member in kind)}"
        ) from None


# -- the buildable vocabulary ----------------------------------------------


def _build_from(
    table: Mapping[str, Any],
    where: str,
    bindings: SpecBindings,
    context: Mapping[str, Any],
) -> Any:
    """A table that *must* declare what to build, built."""
    _string(table, BUILD_KEY, where)
    return _resolve_buildables(table, where, bindings, context)


def _resolve_buildables(
    value: Any, where: str, bindings: SpecBindings, context: Mapping[str, Any]
) -> Any:
    """Replace every `_build` table anywhere inside `value` with the object it
    names, leaving everything else untouched.

    **Depth first.** A protocol declares a batch spec, which is itself built,
    so a table's own children must be resolved before it is handed to its
    builder -- otherwise a builder receives a raw dictionary where it expects
    the object, and the only place that could be repaired is inside every
    builder, one by one.

    Applied to a contender's configuration too, because that is where today's
    recipes carry their own training plan -- `unfolded` takes a `plan`,
    `unfolded_warmstart` takes a `schedule` -- and the grammar re-expresses
    today's contenders rather than redefining them (plan §8.1).
    """
    if isinstance(value, dict):
        resolved = {
            key: _resolve_buildables(item, _at(where, key), bindings, context)
            for key, item in value.items()
            if key != BUILD_KEY
        }
        if BUILD_KEY in value:
            return bindings.build(_string(value, BUILD_KEY, where), resolved, context)
        return resolved
    if isinstance(value, list):
        return [
            _resolve_buildables(item, f"{where}[{index}]", bindings, context)
            for index, item in enumerate(value)
        ]
    return value


# -- the study document -----------------------------------------------------


@dataclass(frozen=True)
class StudyDocument:
    """One loaded study file: the study, plus the middle editing level.

    Attributes:
        study: The study as declared, before any tier is applied.
        tier_overrides: Tier name to the overlay that study declares for it.
            Validated against the whitelist at load time, so a study file
            cannot ship a forbidden override and fail only for whoever happens
            to run that tier.
        source: Where it was read from, for messages.
    """

    study: StudySpec
    tier_overrides: Mapping[str, Mapping[str, Any]]
    source: Path

    def resolve(
        self,
        catalogue: TierCatalogue,
        tier: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> ResolvedStudy:
        """Apply all three editing levels, in increasing locality.

        Args:
            catalogue: The tracked tier catalogue (the outermost level).
            tier: Which tier to run at.
            overrides: The per-invocation level (`--set`), the innermost.

        Returns:
            The study as it will run, plus what to record about how.
        """
        overlay = {
            **self.tier_overrides.get(tier, {}),
            **(overrides or {}),
        }
        return catalogue.resolve(self.study, tier, overrides=overlay)


def _read(path: Path) -> Mapping[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise SpecificationError(f"no such specification: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise SpecificationError(f"{path} is not valid TOML: {error}") from error


def load_study(path: Path | str, *, bindings: SpecBindings) -> StudyDocument:
    """Read a study document.

    Args:
        path: The `.toml` file. Paths inside it -- the problem `.npz`, an
            evaluation problem -- are resolved relative to it, so a study
            directory can be moved or copied whole.
        bindings: The Tier-4 objects this tier may not import.

    Returns:
        The `StudyDocument`.

    Raises:
        SpecificationError: On any malformed, empty or contradictory
            document, naming the offending key.
    """
    source = Path(path)
    document = _read(source)
    if not document:
        raise SpecificationError(
            f"{source} is empty; a study needs at least an id, a problem and "
            "a contender"
        )
    # `id` first, so that a document with several things wrong is reported
    # about a study that can be named rather than about an anonymous file.
    study_id = _string(document, "id", "")
    if "compose" in document:
        # It used to live here, one rule for the whole sweep. Silently ignoring
        # a key an author wrote is how a study quietly stops meaning what its
        # file says -- the same failure the batch-dimension keys were refused
        # for.
        raise SpecificationError(
            "compose is declared on each [[sweep]] axis, not on the study: it "
            "says how that axis joins the ones before it, so a single "
            "study-wide rule cannot express an axis coupled to one neighbour "
            "and crossed with another"
        )
    problem = _problem(document, source, "problem")
    ctx = _context(document, "compute")
    context = {"state_dim": problem.state_dim, "horizon": problem.data.horizon}

    # `[training]` is read BEFORE `[[contenders]]`, which is not the order the
    # document is written in. `_contenders` resolves each contender while the
    # file is open so that a bad configuration is refused with its *position*
    # (Annex 01 §4), and since G-2 a contender that declares no plan cannot be
    # resolved until the study's default is in it. The materialisation itself
    # lives in `StudySpec.__post_init__`, which is its single home and runs
    # again below; this call exists only so that the parse-time message can
    # still name `contenders[i]`.
    training = _training(document, ctx, bindings, context, source)
    contenders = _contenders(document, bindings, context, plan=training.plan)
    for contender in contenders:
        # Call site 2 of 2 for the precision rule (plan §9.2b). This is the
        # other place a contender family and a training context are both in
        # hand, and "fails at parse time" is a property of this surface.
        require_supported_precision(contender.family, ctx.precision_enum)

    return StudyDocument(
        study=StudySpec(
            id=study_id,
            problem=problem,
            contenders=contenders,
            training=training,
            evaluation=_evaluation(document, problem, ctx, source, bindings, context),
            sweep=_sweep(document, source),
            gates=_gates(document),
            analyses=_analyses(document),
            figures=_figures(document),
        ),
        tier_overrides=_tier_overrides(document),
        source=source,
    )


#: Axis paths whose value is a whole `ProblemSpec` rather than a scalar inside
#: one (Annex 01 §2.5.1). Measured: both crash `materialise()` with
#: `AttributeError: 'str' object has no attribute 'problem_id'` when the value
#: arrives as a raw TOML string, while every NESTED problem path already works
#: (`problem.data.control_bound` materialises 6 points and 6 ModelIDs) — so the
#: gap is exactly these two and the list is closed rather than a prefix rule.
#:
#: They sit on the identity levels their roots already classify, which is the
#: whole point: sweeping `evaluation.problem` scores one model on several
#: plants (zero trainings), sweeping `problem` trains one model per plant, and
#: sweeping `training.problem` (Annex 01 §2.3.5, D25) trains one model per
#: WORLD while the contender is told the study plant throughout.
PROBLEM_VALUED_AXIS_PATHS = frozenset(
    {"problem", "evaluation.problem", "training.problem"}
)


def _problem(document: Mapping[str, Any], source: Path, key: str) -> ProblemSpec:
    table = _table(document, key, "")
    return _load_problem(_string(table, "path", key), source, f"{key}.path")


def _load_problem(relative: str, source: Path, where: str) -> ProblemSpec:
    """One frozen `.npz`, resolved beside the document that names it.

    Shared by `[problem].path`, `[evaluation].problem.path` and a spec-valued
    sweep axis, so all three resolve relatively the same way and refuse a
    missing file in the same words. A second reading of "where does problem
    data live" is how D19's answer comes to have two answers.
    """
    path = (source.parent / relative).resolve()
    if not path.exists():
        raise SpecificationError(
            f"{where} names {relative!r}, which does not exist next to "
            f"{source.name}. Problem data is committed alongside the study that "
            "uses it (D19) rather than regenerated"
        )
    return ProblemSpec.load(path)


def _context(document: Mapping[str, Any], key: str) -> ComputeContext:
    table = _table(document, key, "")
    # `backend` and `precision` are checked here, one at a time, rather than
    # left to `ComputeContext` to reject: its message names the offending value
    # and the enum's class name, and the author needs the *document key* to go
    # and edit. `device` is left to it, because resolving hardware loudly is
    # exactly what it exists to do and it has facts this tier does not.
    backend = _enum(Backend, _string(table, "backend", key), _at(key, "backend"))
    precision = _enum(
        Precision, _string(table, "precision", key), _at(key, "precision")
    )
    try:
        return ComputeContext(
            backend=backend, device=_string(table, "device", key), precision=precision
        )
    except (ValueError, RuntimeError) as error:
        raise SpecificationError(f"{_at(key, 'device')}: {error}") from error


def _contenders(
    document: Mapping[str, Any],
    bindings: SpecBindings,
    context: Mapping[str, Any],
    *,
    plan: Signable,
) -> tuple[ContenderSpec, ...]:
    # An empty or absent `contenders` is deliberately NOT refused here:
    # `StudySpec.__post_init__` already refuses it, and its message names the
    # study, which this function cannot. A second check would only be
    # distinguishable from its own absence by the wording.
    declared = _sequence(document, "contenders", "")
    specs: list[ContenderSpec] = []
    for index, entry in enumerate(declared):
        where = f"contenders[{index}]"
        if not isinstance(entry, dict):
            raise SpecificationError(f"{where} must be a table, got {entry!r}")
        # An unknown family is deliberately NOT pre-checked against the
        # registry here. `_require_resolvable` below already refuses it, and
        # the registry's own message names the value *and* lists every
        # registered family -- so a check here would duplicate the refusal
        # while creating a second source of truth about what exists, free to
        # drift from what `create` actually accepts.
        family = _string(entry, "family", where)
        config = _resolve_buildables(
            entry.get("config", {}), f"{where}.config", bindings, context
        )
        spec = ContenderSpec(
            family=family,
            config=config,
            label=_string(entry, "label", where),
            # Read explicitly. `_contenders` folds nothing implicitly, so a key
            # this function does not name is dropped in silence -- and a
            # display name that vanishes between the document and the figure
            # reads as "the feature does not work" rather than as a typo.
            display=_optional_string(entry, "display", where),
            role=_enum(Role, str(entry.get("role", Role.CONTENDER)), f"{where}.role"),
            registry=bindings.registry,
        ).with_default_plan(plan)
        _require_resolvable(spec, where)
        specs.append(spec)
    return tuple(specs)


def _require_resolvable(spec: ContenderSpec, where: str) -> None:
    """Force the family's own schema check while the document is being read.

    Annex 01 §4 puts this under *semantic* validation: a family configuration
    is checked against that family's declared schema "at parse time with an
    actionable message rather than at layer construction". A `ContenderSpec`
    resolves its recipe lazily, so without this a document naming a key the
    family does not take, or an enum value it does not know, is accepted here
    and fails much later -- at the first `get_signature()`, several frames away
    from anything that could name the document.

    The recipe constructors already raise in their own vocabulary, which is
    right for them and useless to someone editing a file. What this adds is the
    contender's position and family.

    Raises:
        SpecificationError: If the family cannot be composed from this
            configuration.
    """
    try:
        spec.resolve()
    except SpecificationError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise SpecificationError(
            f"{where} cannot be composed as family {spec.family!r}: {error}"
        ) from error


def _training(
    document: Mapping[str, Any],
    ctx: ComputeContext,
    bindings: SpecBindings,
    context: Mapping[str, Any],
    source: Path,
) -> TrainingSpec:
    table = _table(document, "training", "")
    batch = _table(table, "batch", "training")
    # As with `contenders` above: `TrainingSpec.__post_init__` already refuses
    # an empty seed collection, in the same words.
    seeds = _sequence(table, "seeds", "training")
    return TrainingSpec(
        data=_data(_table(table, "data", "training"), "training.data"),
        plan=_build_from(
            _table(table, "plan", "training"), "training.plan", bindings, context
        ),
        batch=BatchPlan(
            effective_size=_integer(batch, "effective_size", "training.batch"),
            microbatch=_optional_integer(batch, "microbatch", "training.batch"),
        ),
        ctx=ctx,
        seeds=tuple(int(seed) for seed in seeds),
        # The training WORLD (Annex 01 §2.3.5, D25): the plant the training
        # trajectories are generated by, when it is not the plant the
        # contender is told. Optional; resolved beside the document exactly
        # as the study problem is, so the two cannot disagree about where
        # problem data lives.
        problem=(
            _load_problem(
                _string(
                    _table(table, "problem", "training"), "path", "training.problem"
                ),
                source,
                "training.problem.path",
            )
            if "problem" in table
            else None
        ),
    )


def _data(table: Mapping[str, Any], where: str) -> DataSpec:
    """A distribution declaration.

    `kind` and `seed` are named keys; everything else is the family's own
    parameter, passed through. The family's schema belongs with the sampler,
    not with the grammar -- the closed thing here is the set of families, and
    an unknown one is refused where it is built.
    """
    reserved = {"kind", "seed"}
    return DataSpec(
        kind=str(table.get("kind", DEFAULT_DATA_KIND)),
        seed=_integer(table, "seed", where) if "seed" in table else 0,
        params={key: value for key, value in table.items() if key not in reserved},
    )


def _evaluation(
    document: Mapping[str, Any],
    problem: ProblemSpec,
    ctx: ComputeContext,
    source: Path,
    bindings: SpecBindings,
    context: Mapping[str, Any],
) -> EvaluationSpec:
    table = _table(document, "evaluation", "")
    #: An evaluation problem is optional, and declaring one is the *entire*
    #: mechanism for a distribution-shift study: whether that shifted
    #: measurement also *informs* the controller is `rehost`'s concern
    #: (Annex 01 §2.4.1), read explicitly below -- before it was a named key,
    #: a `rehost` line here would have been dropped in silence, this project's
    #: recurring inert-declaration class.
    scored_on = _problem(table, source, "problem") if "problem" in table else problem
    return EvaluationSpec(
        problem=scored_on,
        protocol=_build_from(
            _table(table, "protocol", "evaluation"),
            "evaluation.protocol",
            bindings,
            {"state_dim": scored_on.state_dim, "horizon": scored_on.data.horizon},
        ),
        # Its own context by default equal to the training one, so that
        # contenders trained at different precisions are still scored at one.
        ctx=_context(table, "compute") if "compute" in table else ctx,
        metrics=tuple(_sequence(table, "metrics", "evaluation")) or DEFAULT_METRICS,
        # Validated by `EvaluationSpec.__post_init__`, whose refusal names the
        # permitted modes; validating here too would be a second source of
        # truth about what exists.
        rehost=(
            _string(table, "rehost", "evaluation")
            if "rehost" in table
            else REHOST_BLIND
        ),
        rehost_overrides=_rehost_overrides(table, source),
    )


def _rehost_overrides(
    table: Mapping[str, Any], source: Path
) -> tuple[RehostOverride, ...]:
    """`[[evaluation.rehost_overrides]]` -- per-plant construction literals.

    The plant is resolved beside the document exactly as an axis value is
    (Annex 01 §2.5.1), so an override and the sweep it belongs to cannot
    disagree about where problem data lives. Whether the override's contender
    exists, and whether any point ever evaluates on its plant, are
    `StudySpec`'s checks -- this function has neither the contender list nor
    the expanded points.
    """
    overrides: list[RehostOverride] = []
    for index, entry in enumerate(_sequence(table, "rehost_overrides", "evaluation")):
        where = f"evaluation.rehost_overrides[{index}]"
        if not isinstance(entry, dict):
            raise SpecificationError(f"{where} must be a table, got {entry!r}")
        config = entry.get("config")
        if not isinstance(config, dict) or not config:
            raise SpecificationError(
                f"{where}.config must be a non-empty table of the fields the "
                f"aware rebuild replaces, got {config!r}"
            )
        overrides.append(
            RehostOverride(
                contender=_string(entry, "contender", where),
                problem=_load_problem(
                    _string(entry, "problem", where), source, f"{where}.problem"
                ),
                config=tuple(sorted(config.items())),
            )
        )
    return tuple(overrides)


def _sweep(document: Mapping[str, Any], source: Path) -> tuple[SweepAxis, ...]:
    axes: list[SweepAxis] = []
    for index, entry in enumerate(_sequence(document, "sweep", "")):
        where = f"sweep[{index}]"
        if not isinstance(entry, dict):
            raise SpecificationError(f"{where} must be a table, got {entry!r}")
        path = _string(entry, "path", where)
        axes.append(
            SweepAxis(
                path=path,
                values=_axis_values(entry, path, source, where),
                applies_to=tuple(
                    str(label) for label in _sequence(entry, "applies_to", where)
                ),
                compose=_enum(
                    Composition,
                    str(entry.get("compose", Composition.PRODUCT.value)),
                    f"{where}.compose",
                ),
                labels=tuple(str(label) for label in _sequence(entry, "labels", where)),
            )
        )
    return tuple(axes)


def _axis_values(
    entry: Mapping[str, Any], path: str, source: Path, where: str
) -> tuple[Any, ...]:
    """An axis's declared values, built into specs where the path needs one.

    Annex 01 §2.5.1. Everything else stays a raw TOML scalar, which is what
    `replace_at` writes into the spec tree and what every existing axis has
    always been; only the two whole-spec paths are constructed here, and they
    are constructed HERE rather than in `SweepAxis` because resolving a file
    beside the document is the loader's job and Tier 3 has no `source`.
    """
    values = _sequence(entry, "values", where)
    if path not in PROBLEM_VALUED_AXIS_PATHS:
        return values
    built: list[Any] = []
    for index, value in enumerate(values):
        if str(value) == "":
            # The matched-case sentinel (Annex 01 §2.3.5): on a
            # training.problem axis the empty string writes the field's
            # ABSENCE, so the position signs nothing and its points are the
            # matched models bit for bit. The other two whole-spec paths have
            # no "absent" -- a study is trained on something and scored on
            # something -- so the spelling is refused there by name.
            if path != "training.problem":
                raise SpecificationError(
                    f'{where}.values[{index}] is "" on axis {path!r}; the '
                    "matched-case spelling exists only for training.problem, "
                    "whose absence means <the study problem>. A study or "
                    "evaluation problem has no absent case"
                )
            built.append(None)
            continue
        built.append(_load_problem(str(value), source, f"{where}.values[{index}]"))
    return tuple(built)


def _gates(document: Mapping[str, Any]) -> tuple[GateSpec, ...]:
    gates: list[GateSpec] = []
    for index, entry in enumerate(_sequence(document, "gates", "")):
        where = f"gates[{index}]"
        if not isinstance(entry, dict):
            raise SpecificationError(f"{where} must be a table, got {entry!r}")
        gates.append(
            GateSpec(
                kind=_string(entry, "kind", where),  # type: ignore[arg-type]
                config={k: v for k, v in entry.items() if k != "kind"},
            )
        )
    return tuple(gates)


def _analyses(document: Mapping[str, Any]) -> tuple[AnalysisSpec, ...]:
    """`[[analyses]]` -- Annex 01 §2.6.

    Reserved keys are lifted out and everything else is the kind's own
    configuration, exactly as `_gates` treats `kind`. The open remainder is
    deliberate: the closed thing is the *kind*, and its schema belongs with
    the analysis rather than with the grammar.
    """
    analyses: list[AnalysisSpec] = []
    for index, entry in enumerate(_sequence(document, "analyses", "")):
        where = f"analyses[{index}]"
        if not isinstance(entry, dict):
            raise SpecificationError(f"{where} must be a table, got {entry!r}")
        analyses.append(
            AnalysisSpec(
                id=_string(entry, "id", where),
                kind=_string(entry, "kind", where),
                config={k: v for k, v in entry.items() if k not in ("id", "kind")},
            )
        )
    return tuple(analyses)


def _figures(document: Mapping[str, Any]) -> tuple[FigureSpec, ...]:
    """`[[figures]]` -- Annex 01 §2.6.

    `source` is reserved alongside `id` and `kind` because it is structural:
    it names the analysis table this figure renders, and Annex 03 §A.6 makes
    that table the figure's own `.data.parquet`. Leaving it in `config` would
    make a load-bearing reference indistinguishable from a style option.
    """
    figures: list[FigureSpec] = []
    for index, entry in enumerate(_sequence(document, "figures", "")):
        where = f"figures[{index}]"
        if not isinstance(entry, dict):
            raise SpecificationError(f"{where} must be a table, got {entry!r}")
        # `table` is a FIELD, not a config key. Without this line
        # `[figures.table]` would parse, land in `config`, and be ignored —
        # the seventh declared-everywhere-executed-nowhere this project would
        # have shipped, and the one that silently omits an artifact.
        table = entry.get("table")
        if table is not None and not isinstance(table, dict):
            raise SpecificationError(
                f"{where}.table must be a table of table options "
                f"(`columns`, `precision`, `caption`), got {table!r}"
            )
        figures.append(
            FigureSpec(
                id=_string(entry, "id", where),
                kind=_string(entry, "kind", where),
                source=_string(entry, "source", where),
                config={
                    k: v
                    for k, v in entry.items()
                    if k not in ("id", "kind", "source", "table")
                },
                table=table,
            )
        )
    return tuple(figures)


def _tier_overrides(
    document: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    declared = document.get(TIER_OVERRIDES_KEY, {})
    if not isinstance(declared, dict):
        raise SpecificationError(
            f"{TIER_OVERRIDES_KEY} must be a table of tier name to overrides"
        )
    overrides: dict[str, dict[str, Any]] = {}
    for tier, overlay in declared.items():
        where = f"{TIER_OVERRIDES_KEY}.{tier}"
        if not isinstance(overlay, dict):
            raise SpecificationError(f"{where} must be a table of path to value")
        require_permitted_paths(overlay, f"the {tier!r} overlay of this study")
        overrides[tier] = dict(overlay)
    return overrides


def _integer(table: Mapping[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecificationError(f"{_at(where, key)} must be an integer, got {value!r}")
    return value


def _optional_integer(table: Mapping[str, Any], key: str, where: str) -> int | None:
    return None if key not in table else _integer(table, key, where)


# -- the tier catalogue -----------------------------------------------------


def load_tier_catalogue(path: Path | str, *, version: str = "") -> TierCatalogue:
    """Read `studies/_tiers.toml`.

    Args:
        path: The catalogue file. Each top-level table is one tier, whose keys
            are the dotted paths of Annex 01 §2.3.2 -- quoted in the file so
            TOML keeps them literal rather than expanding them into nested
            tables. A tier's vocabulary *is* the whitelist, and a catalogue
            needing to be flattened before it could be checked against that
            list would put the two one transformation apart.
        version: Recorded as `tier_catalogue_version`; defaults to the file's
            own `version` key.

    Returns:
        The `TierCatalogue`.

    Raises:
        SpecificationError: On malformed TOML, an override outside the
            whitelist, or `smoke_subset` declared by any tier but `smoke`.
    """
    document = _read(Path(path))
    declared_version = str(document.get("version", ""))
    tiers: dict[str, Tier] = {}
    for name, table in document.items():
        if not isinstance(table, dict):
            continue  # a top-level scalar such as `version`
        subset = table.get("smoke_subset")
        if subset is not None and not isinstance(subset, dict):
            raise SpecificationError(f"{name}.smoke_subset must be a table")
        tiers[name] = Tier(
            name=name,
            overrides={k: v for k, v in table.items() if k != "smoke_subset"},
            smoke_subset=None
            if subset is None
            else SmokeSubset(
                max_points_per_axis=_integer(
                    subset, "max_points_per_axis", f"{name}.smoke_subset"
                )
            ),
        )
    if not tiers:
        raise SpecificationError(f"{path} declares no tiers")
    return TierCatalogue(tiers=tiers, version=version or declared_version)
