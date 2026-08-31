"""Execution tiers as editable data (Stage 2 Phase E2, decision **D15**).

A study declares its *scientific* content; how hard it is run is a separable
choice. A named tier scales the effort knobs together, so that a wiring check
and a thesis figure are one word apart.

**A tier scales effort; it never changes content.** That is the whole of D15,
and it is enforced structurally rather than by convention: a tier may write
only to `PERMITTED_TIER_PATHS`, a closed whitelist, and anything it does not
name is refused whether or not it was foreseen. The forbidden classes are the
point of the rule -- `problem.*`, `contenders.*.family`, `sweep.*.values` and
`gates.*` -- because a tier able to truncate a swept axis or relax a gate would
let `publication` quietly mean something the reader of the resulting figure
does not assume.

**The tier is never an opaque label.** It participates in identity only through
the fields it writes, so two differently named tiers writing the same effective
fields produce the same models -- and re-running a study at a higher tier
reuses every model the lower tier already produced whose effective fields are
unchanged. The tier name and the catalogue version go to provenance, where they
answer "why did this retrain?", and to nothing else.

**The one exemption, granted to `smoke` alone.** A wiring check over NB04's
real shape is 27 trainings; over two depths it is 9 -- the difference between a
check that runs before every heavy batch and one that is skipped. It is safe
only because smoke models are disposable by construction: `smoke` also cuts
epochs and batch size, so its models differ in effective fields and are never
reused by the later full run. Two guards keep it honest, and the second one is
`require_analysable_measurement` at the bottom of this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .errors import SpecificationError
from .paths import replace_at, replace_at_all
from .study import StudySpec, SweepAxis

#: Paths a tier may write, from [Annex 01 §2.3.2]. A trailing ``.*`` permits
#: the subtree **strictly below** that path and not the path itself: replacing
#: `training.plan` wholesale could swap an end-to-end plan for a layer-wise
#: one, which is content, not effort.
#:
#: `evaluation.protocol` is named by its two leaves rather than as a subtree on
#: purpose. The subtree would also carry
#: `evaluation.protocol.batch_spec.process_noise_std`, and a tier able to
#: rewrite the evaluation noise level changes what is measured, not how hard it
#: is measured. `training.plan.*` stays a subtree because every leaf under it
#: is an effort knob.
#:
#: `statistics.*` and `profiling.timing_repeats` are forward-looking: they are
#: Tier 6/7 fields arriving in Stages 4-5, whitelisted now so that adding them
#: later is a grammar change and not also a policy change. Until then a tier
#: writing one fails naming the missing field.
#:
#: `contenders.*.config.plan.*` and `…schedule.*` are the fitting effort that
#: actually EXECUTES: `TrainableRecipe.build_engine` trains for the plan in the
#: contender's own config, and `training.plan` -- which is also permitted, and
#: which the shipped catalogue keeps in step so a resolved study does not
#: declare 200 epochs while running 5 -- is read by no family at all. Before
#: F2a, `smoke` therefore moved every ModelID and changed no effort whatsoever.
#:
#: By leaf-subtree and never by contender subtree:
#: `contenders.*.config.num_iterations` is the unrolling depth, which is the
#: study's independent variable, and `contenders.*.config.plan` without a leaf
#: would let a tier swap an end-to-end plan for a layer-wise one. Both content.
PERMITTED_TIER_PATHS: tuple[str, ...] = (
    "training.seeds",
    "training.plan.*",
    "contenders.*.config.plan.*",
    "contenders.*.config.schedule.*",
    "training.batch.effective_size",
    "training.batch.microbatch",
    "evaluation.protocol.n_batches",
    "evaluation.protocol.batch_spec.batch_size",
    "statistics.*",
    "profiling.timing_repeats",
)

#: The one path whose tier value is a *count* rather than the value itself: the
#: catalogue writes `training.seeds: 5` against a `tuple[int, ...]` field. This
#: is the only per-path coercion, deliberately -- a coercion table over paths
#: would be a second grammar.
SEED_COUNT_PATH = "training.seeds"

#: The provenance key a subsetted run stamps into every measurement it
#: produces, and the key `require_analysable_measurement` reads. A by-name
#: contract between the producer and the analysis tier, in the same family as
#: `seeds` and `id`.
AXIS_SUBSET_KEY = "axis_subset"

#: The one tier the axis-subset exemption is granted to.
SUBSETTING_TIER = "smoke"


def _is_permitted(path: str) -> bool:
    """Whether the whitelist names this path, wildcard or label-addressed.

    A `*` INSIDE a pattern matches either the wildcard or one contender's
    label, because the whitelist is about WHICH FIELD a tier may write and not
    about how many contenders it reaches: `contenders.neural.config.plan.epochs`
    and `contenders.*.config.plan.epochs` write the same field, so admitting one
    and refusing the other would be the rule enforcing a spelling. What stays
    refused is refused by the field, exactly as before —
    `contenders.neural.family` matches no pattern, since `contenders.*.family`
    is not on the list.

    A trailing `.*` still means "this subtree", which is a different quantifier
    and is matched by prefix as it always was.
    """
    segments = path.split(".")
    for pattern in PERMITTED_TIER_PATHS:
        if pattern.endswith(".*") and path.startswith(pattern[:-1]):
            return True
        expected = pattern.split(".")
        if len(expected) == len(segments) and all(
            want == "*" or want == got
            for want, got in zip(expected, segments, strict=True)
        ):
            return True
    return False


def require_permitted_paths(overrides: Mapping[str, Any], source: str) -> None:
    """Refuse any write outside the whitelist.

    Called by every one of Annex 01 §2.3.1's three editing levels -- the
    catalogue, the per-study overlay and the per-invocation `--set` -- because
    a level that escaped the whitelist would make the others' enforcement
    decorative.

    Args:
        overrides: Dotted path to value.
        source: What is doing the writing, for the message.

    Raises:
        SpecificationError: On the first path the whitelist does not name.
    """
    for path in overrides:
        if not _is_permitted(path):
            raise SpecificationError(
                f"{source} may not write {path!r}: a tier scales effort and never "
                f"changes content. Permitted: {list(PERMITTED_TIER_PATHS)}"
            )


@dataclass(frozen=True)
class SmokeSubset:
    """How much of each swept axis a smoke run keeps.

    Attributes:
        max_points_per_axis: Values kept per axis, at evenly spaced indices and
            always including both endpoints -- so 2 is the endpoints, 3 adds
            the midpoint, and an axis at or below the cap is left whole.
            Endpoints because the extremes are where plumbing breaks: a depth
            of 1 and a depth of 20 exercise different code paths, two adjacent
            middle values exercise one.
    """

    max_points_per_axis: int = 2

    def __post_init__(self) -> None:
        if self.max_points_per_axis < 1:
            raise SpecificationError(
                "max_points_per_axis must be at least 1, got "
                f"{self.max_points_per_axis}; an axis subset to nothing is a "
                "sweep that sweeps nothing"
            )

    def indices(self, length: int) -> tuple[int, ...]:
        """Which positions of a `length`-value axis a smoke run keeps.

        Exposed so values and `labels` are subset by the SAME indices: the
        join between a label and its value is position, and subsetting one
        without the other would silently rename the surviving categories.
        """
        cap = self.max_points_per_axis
        if length <= cap:
            return tuple(range(length))
        if cap == 1:
            return (0,)
        last = length - 1
        return tuple(sorted({round(step * last / (cap - 1)) for step in range(cap)}))

    def subset(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        """The values kept, in the axis's own order."""
        return tuple(values[index] for index in self.indices(len(values)))


@dataclass(frozen=True)
class Tier:
    """One named effort level.

    Attributes:
        name: What `--tier` selects it by.
        overrides: Dotted path to value, validated against the whitelist at
            construction -- parse time, before any compute.
        smoke_subset: The axis-subset exemption, readable only by the `smoke`
            tier. `None` everywhere else.
    """

    name: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    smoke_subset: SmokeSubset | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecificationError("a tier needs a name; --tier selects it by one")
        require_permitted_paths(self.overrides, f"tier {self.name!r}")
        if self.smoke_subset is not None and self.name != SUBSETTING_TIER:
            raise SpecificationError(
                f"tier {self.name!r} declares smoke_subset, which only the "
                f"{SUBSETTING_TIER!r} tier may: the exemption is justified solely "
                "by smoke models being disposable, since smoke also cuts epochs "
                "and batch size and its models are never reused by the full run. "
                f"No other tier can say that, so {self.name!r} would truncate an "
                "axis and keep the results"
            )


@dataclass(frozen=True)
class ResolvedStudy:
    """A study with a tier applied, and what the producer must record about it.

    Attributes:
        study: The study as it will actually run. An ordinary `StudySpec`: the
            tier has written its fields and is otherwise gone, which is what
            makes the tier identity-bearing without being identity itself.
        tier: The tier's name. Provenance only.
        catalogue_version: The catalogue it came from. Provenance only --
            informational, since the effective fields already carry identity.
        axis_subset: Whether an axis was **actually** truncated.
    """

    study: StudySpec
    tier: str
    catalogue_version: str
    axis_subset: bool

    @property
    def provenance(self) -> dict[str, Any]:
        """What the producer stamps into every measurement of this run."""
        return {
            "tier": self.tier,
            "tier_catalogue_version": self.catalogue_version,
            AXIS_SUBSET_KEY: self.axis_subset,
        }


@dataclass(frozen=True)
class TierCatalogue:
    """The tracked set of tiers a study may be run at.

    Attributes:
        tiers: Name to tier. Each entry must be filed under its own name; a
            catalogue is edited by hand, and a tier under the wrong key would
            be reported in provenance as one nobody can look up.
        version: Recorded per model as `tier_catalogue_version`, so that "why
            did this retrain?" has an answer. Never signed.
    """

    tiers: Mapping[str, Tier]
    version: str = ""

    def __post_init__(self) -> None:
        for name, tier in self.tiers.items():
            if tier.name != name:
                raise SpecificationError(
                    f"tier {tier.name!r} is filed under {name!r}; a catalogue entry "
                    "must carry the name it is looked up by"
                )

    def get(self, name: str) -> Tier:
        """The named tier.

        Raises:
            SpecificationError: If no such tier is catalogued. The message
                lists the ones that are, because a mistyped `--tier` is the
                most likely way to reach this.
        """
        try:
            return self.tiers[name]
        except KeyError:
            raise SpecificationError(
                f"no tier named {name!r} in this catalogue; available: "
                f"{sorted(self.tiers)}"
            ) from None

    def resolve(
        self,
        study: StudySpec,
        tier: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> ResolvedStudy:
        """Apply a tier, and any per-invocation overlay, to a study.

        Args:
            study: The study as declared.
            tier: The catalogued tier to run it at.
            overrides: The per-invocation level (`--set`), subject to the same
                whitelist and taking precedence over the catalogue.

        Returns:
            The study as it will run, plus what to record about how.

        Raises:
            SpecificationError: If the tier is unknown, if any override is
                outside the whitelist, or if a permitted path names a field
                this study does not have.
        """
        selected = self.get(tier)
        overlay = dict(overrides or {})
        require_permitted_paths(overlay, "an override")

        resolved = study
        # The catalogue first, the overlay second, so the more local level wins
        # where both name one path. They differ in one further respect: see
        # `_write`.
        for path, value in selected.overrides.items():
            resolved = _write(resolved, path, value, source=f"tier {tier!r}")
        for path, value in overlay.items():
            resolved = _write(resolved, path, value, source="an override", strict=True)

        truncated = False
        subset = selected.smoke_subset
        if subset is not None and resolved.sweep:
            axes = _subset_axes(resolved.sweep, subset)
            # Compared by LENGTH, never by equality: an axis may sweep whole
            # `ProblemSpec`s, whose dataclass `__eq__` compares NumPy arrays and
            # raises on the resulting array's truth value.
            truncated = any(
                len(new.values) < len(old.values)
                for new, old in zip(axes, resolved.sweep, strict=True)
            )
            if truncated:
                resolved = replace_at(resolved, ["sweep"], axes)
                # An override whose sweep position the subset removed is the
                # truncation's dead weight, not the author's error: left in
                # place, the resolved study would refuse ITSELF at
                # materialise under the gate-firability rule. Pruned by
                # index, because a RehostOverride holds a ProblemSpec and
                # comparing those equates NumPy arrays.
                dead = set(resolved.unfired_override_indices())
                if dead:
                    survivors = tuple(
                        override
                        for index, override in enumerate(
                            resolved.evaluation.rehost_overrides
                        )
                        if index not in dead
                    )
                    resolved = replace_at(
                        resolved, ["evaluation", "rehost_overrides"], survivors
                    )
        return ResolvedStudy(
            study=resolved,
            tier=tier,
            catalogue_version=self.version,
            axis_subset=truncated,
        )


def _write(
    study: StudySpec, path: str, value: Any, *, source: str, strict: bool = False
) -> StudySpec:
    """One override, written into `study`.

    Args:
        study: The study so far.
        path: The dotted path, possibly carrying a `*` fan-out.
        value: What to write, before per-path coercion.
        source: Which editing level is writing, for the message.
        strict: Whether a fan-out matching **nothing** is an error.

            This is the one place the three editing levels of Annex 01 §2.3.1
            genuinely differ. A catalogue is shared across every study and
            cannot know which contender families any one of them declares, so
            `contenders.*.config.schedule.refinement_epochs` matching nobody is
            ordinary there -- refusing would stop a study with no layer-wise
            contender from running at all. A per-study overlay and a `--set`
            are written by someone looking at *this* study, so a path matching
            nobody is a typo; accepting it silently would leave a knob nobody
            turns that provenance reports as one that was, which is the whole
            of the defect F2a exists to close.

    Returns:
        The rebuilt study.

    Raises:
        SpecificationError: If the path names no such field, or -- under
            `strict` -- if its fan-out matched nothing. The path is named here
            rather than in `paths`, which cannot know it.
    """
    segments = path.split(".")
    coerced = _coerce(path, value)
    try:
        resolved, matched = replace_at_all(study, segments, coerced)
    except SpecificationError as error:
        raise SpecificationError(f"{source} cannot write {path!r}: {error}") from error
    if strict and matched == 0:
        raise SpecificationError(
            f"{source} writes {path!r}, which no contender of study "
            f"{study.id!r} accepts, so it turns nothing. Contenders that "
            "declare no such field are skipped by design -- an analytic "
            "baseline has no training plan -- but a path matching none of them "
            "is a misspelling, and it would be recorded in provenance as a knob "
            "this run turned"
        )
    # `replace_at_all` traverses duck-typed over frozen dataclasses and cannot
    # promise more than `Any` about what it rebuilds; what went in was a study.
    return cast("StudySpec", resolved)


def _subset_axes(
    sweep: tuple[SweepAxis, ...], subset: SmokeSubset
) -> tuple[SweepAxis, ...]:
    return tuple(
        SweepAxis(
            path=axis.path,
            values=subset.subset(axis.values),
            applies_to=axis.applies_to,
            # Carried through deliberately: this rebuilds every axis, and an
            # omitted `compose` would default to PRODUCT, silently turning a
            # coupled pair into a grid -- a *scientific* change made by a tier,
            # which is the one thing D15 forbids outright.
            compose=axis.compose,
            # Subset by the SAME indices as the values, because the join
            # between a label and its value is position. The first rebuild
            # here dropped `labels` altogether -- harmless while no labelled
            # axis was ever longer than the cap, and a silent renaming of
            # every surviving category the moment one was.
            labels=(
                tuple(axis.labels[index] for index in subset.indices(len(axis.values)))
                if axis.labels
                else ()
            ),
        )
        for axis in sweep
    )


def _coerce(path: str, value: Any) -> Any:
    """The one per-path coercion: a seed *count* into that many seeds."""
    if path != SEED_COUNT_PATH:
        return value
    if isinstance(value, bool):
        raise SpecificationError(
            f"{SEED_COUNT_PATH} got {value!r}; a boolean is an int in Python and "
            "a hand-written tier file is exactly where a stray yes/no becomes "
            "one seed by accident"
        )
    if isinstance(value, int):
        if value < 1:
            raise SpecificationError(
                f"{SEED_COUNT_PATH} must ask for at least one seed, got {value}; "
                "zero seeds trains nothing, which is a specification error rather "
                "than a very cheap tier"
            )
        return tuple(range(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    raise SpecificationError(
        f"{SEED_COUNT_PATH} takes a seed count or an explicit sequence of seeds, "
        f"got {value!r}"
    )


def require_analysable_measurement(provenance: Mapping[str, Any]) -> None:
    """Refuse a measurement produced under a truncated sweep axis.

    D15's second guard, and the reason the first one is worth having: no
    analysis, table or figure may consume a subsetted run, so a truncated axis
    cannot physically reach a notebook -- it fails loudly here rather than
    rendering a plausible, wrong figure.

    A free function over a provenance mapping rather than a method, for the
    same reason `require_supported_precision` is one: the caller that has the
    data is the **analysis tier**, which does not exist until Stage 4 and which
    Tier 3 must not grow. The rule is delivered and tested here; the call site
    is owed by Stage 4.

    An absent stamp means unsubsetted. Refusing absence would retire every
    measurement predating the stamp, and the guard exists to protect figures,
    not to invalidate the store.

    Args:
        provenance: The measurement's provenance mapping.

    Raises:
        SpecificationError: If the measurement is stamped `axis_subset: true`.
    """
    if not provenance.get(AXIS_SUBSET_KEY, False):
        return
    tier = provenance.get("tier", SUBSETTING_TIER)
    raise SpecificationError(
        f"this measurement is stamped {AXIS_SUBSET_KEY}=true (tier {tier!r}), so a "
        "swept axis was truncated to check plumbing. It cannot be analysed, "
        "tabulated or plotted: the figure would be plausible and wrong. Re-run "
        "the study at a tier that does not subset"
    )


#: The four standard tiers of Annex 01 §2.3, with `publication`'s values taken
#: from the annex's own catalogue block. E3 moves these into a tracked
#: `studies/_tiers.yaml` and leaves this as its fallback.
DEFAULT_TIER_CATALOGUE = TierCatalogue(
    version="1",
    tiers={
        "smoke": Tier(
            name="smoke",
            overrides={
                "training.seeds": 1,
                "training.plan.epochs": 5,
                # The plan that EXECUTES. `training.plan` above is kept in
                # step so a resolved study does not declare one effort and
                # run another, but no family reads it (plan §9.6.6).
                "contenders.*.config.plan.epochs": 5,
                "contenders.*.config.schedule.warmup_epochs_per_layer": 1,
                "contenders.*.config.schedule.refinement_epochs": 2,
                "training.batch.effective_size": 128,
                "evaluation.protocol.n_batches": 1,
            },
            smoke_subset=SmokeSubset(max_points_per_axis=2),
        ),
        "standard": Tier(
            name="standard",
            overrides={
                "training.seeds": 1,
                "training.plan.epochs": 50,
                # The plan that EXECUTES. `training.plan` above is kept in
                # step so a resolved study does not declare one effort and
                # run another, but no family reads it (plan §9.6.6).
                "contenders.*.config.plan.epochs": 50,
                "contenders.*.config.schedule.warmup_epochs_per_layer": 6,
                "contenders.*.config.schedule.refinement_epochs": 25,
                "training.batch.effective_size": 2048,
                "evaluation.protocol.n_batches": 4,
            },
        ),
        "publication": Tier(
            name="publication",
            overrides={
                "training.seeds": 5,
                "training.plan.epochs": 200,
                # The plan that EXECUTES. `training.plan` above is kept in
                # step so a resolved study does not declare one effort and
                # run another, but no family reads it (plan §9.6.6).
                "contenders.*.config.plan.epochs": 200,
                "contenders.*.config.schedule.warmup_epochs_per_layer": 25,
                "contenders.*.config.schedule.refinement_epochs": 100,
                "training.batch.effective_size": 8192,
                "evaluation.protocol.n_batches": 8,
                "evaluation.protocol.batch_spec.batch_size": 4096,
            },
        ),
        "publication_b16k": Tier(
            name="publication_b16k",
            overrides={
                "training.seeds": 5,
                "training.plan.epochs": 200,
                # The plan that EXECUTES. `training.plan` above is kept in
                # step so a resolved study does not declare one effort and
                # run another, but no family reads it (plan §9.6.6).
                "contenders.*.config.plan.epochs": 200,
                "contenders.*.config.schedule.warmup_epochs_per_layer": 25,
                "contenders.*.config.schedule.refinement_epochs": 100,
                # The one field that differs from `publication` (Annex 01
                # §2.3, added 2026-08-09): the ICASSP revision's training
                # batch, costed by probe at x1.8-2.0 per epoch. Tier names
                # are never signed -- this entry orphans nothing.
                "training.batch.effective_size": 16384,
                "evaluation.protocol.n_batches": 8,
                "evaluation.protocol.batch_spec.batch_size": 4096,
            },
        ),
        "comprehensive": Tier(
            name="comprehensive",
            overrides={
                "training.seeds": 10,
                "training.plan.epochs": 400,
                # The plan that EXECUTES. `training.plan` above is kept in
                # step so a resolved study does not declare one effort and
                # run another, but no family reads it (plan §9.6.6).
                "contenders.*.config.plan.epochs": 400,
                "contenders.*.config.schedule.warmup_epochs_per_layer": 50,
                "contenders.*.config.schedule.refinement_epochs": 200,
                "training.batch.effective_size": 8192,
                "evaluation.protocol.n_batches": 16,
                "evaluation.protocol.batch_spec.batch_size": 4096,
            },
        ),
    },
)
