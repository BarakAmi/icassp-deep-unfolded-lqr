"""One study, and the sweep algebra that expands it (Stage 2 Phase E1).

A `StudySpec` is one problem, a set of contenders, one training declaration,
one evaluation declaration, and the axes along which they vary. It is the
generic primitive that replaces five bespoke sweep drivers.

**The algebra's job is to know what each axis costs.** Parent §4.3:

| An axis touching | perturbs | and costs |
|---|---|---|
| the problem, a recipe parameter, a training field, the train seed | `ModelID` | one training per point |
| the evaluation problem or protocol | `MeasurementID` only | one rollout per point |

That second row is the economic claim the whole re-architecture rests on. An
out-of-distribution study over ten shifted problems is *one* training and ten
rollouts, because the evaluation cannot reach a `ModelID` -- and it cannot
reach one because `derive_model_id` has no parameter through which it could.
The classification here is not a heuristic; it is a projection of that
structural fact onto the axis path, and `materialise` is where the two are
reconciled by actually deriving the identifiers.

**What this module deliberately does not carry.** `analyses`, `figures` and
`statistics` are `StudySpec` fields in Annex 01 §2, and they are -- but they
are consumed by Tiers 6 and 7, which are Stages 4 and 5. Nothing here could
validate one beyond checking it is a string, and hashing an unvalidated blob
into `StudyID` would bake in a shape those stages will change.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
from typing import Any

from ..store.ids import MeasurementID, ModelID, StudyID, study_id
from .analysis import AnalysisSpec, require_unique_analysis_ids
from .contender import ContenderSpec
from .errors import SpecificationError
from .evaluation import REHOST_AWARE, REHOST_BLIND, REHOST_FULL, EvaluationSpec
from .figure import FigureSpec, require_resolvable_sources, require_unique_figure_ids
from .gates import GateKind, GateSpec, inverse_lipschitz_step, require_unique_kinds
from .identity import derive_measurement_id, derive_model_id
from .paths import replace_at
from .problem import ProblemSpec
from .training import TrainingSpec


class AxisLevel(StrEnum):
    """Which identity level an axis perturbs, and therefore what it costs."""

    #: One training per point.
    MODEL = "model"
    #: One rollout per point; the model is trained once for the whole axis.
    MEASUREMENT = "measurement"


#: The root segment of an axis path decides its level, because the roots *are*
#: the identity split: everything `derive_model_id` consumes is model-level and
#: everything only `derive_measurement_id` sees is measurement-level. Adding a
#: root here without adding it to one of those two functions would be a lie the
#: `materialise` counts would immediately expose.
AXIS_ROOTS: dict[str, AxisLevel] = {
    "problem": AxisLevel.MODEL,
    "contenders": AxisLevel.MODEL,
    "training": AxisLevel.MODEL,
    "evaluation": AxisLevel.MEASUREMENT,
}


#: The axis path that cannot vary anything, and the one it is confused with.
#: `training.plan` is a *template*, materialised into every contender that
#: accepts one when the study is built (Annex 01 §2.3.4), so an axis writing it
#: afterwards reaches nobody: every point would derive identical identifiers,
#: `materialise` would deduplicate them, and an eight-value sweep would expand
#: to one point while the document says it swept eight. A *tier* writing the
#: same path stays permitted -- a tier is keeping a declaration honest, an axis
#: is claiming to vary something.
TEMPLATE_PATH = "training.plan"
EXECUTED_PLAN_PATH = "contenders.*.config.plan"


class Composition(StrEnum):
    """How several axes combine."""

    #: Every combination. The default, and what a scaling study wants.
    PRODUCT = "product"
    #: Positionwise. Expresses a diagonal, which a product cannot say.
    ZIP = "zip"


@dataclass(frozen=True)
class SweepAxis:
    """One independent variable.

    Attributes:
        path: Dotted path into the study, e.g.
            ``contenders.*.config.num_iterations``. The `*` matches every
            contender the axis applies to.
        values: The values to sweep. At least one.
        applies_to: Contender labels this axis perturbs; empty means all of
            them. A depth axis applying to an analytic baseline that has no
            depth would otherwise be a silent no-op.
        compose: How this axis joins the ones **before** it. `ZIP` binds it
            positionwise to the group immediately preceding; `PRODUCT` opens a
            new group, and the groups combine by Cartesian product. The first
            axis has nothing to join, so its value is unread.
        labels: What a **reader** is shown for each value, positionwise
            (Annex 01 §2.5.1). Empty for a scalar axis, where `2` and `4` are
            their own names; a spec-valued axis needs them, because
            `str(ProblemSpec)` is a repr and a filename would name a category
            `icassp_n4m2_N100_u0p1_s0_rotA30` in a paper.

            **Needed is not the same as required HERE, and that distinction is
            a correction.** Enforcing it in this constructor was tried and
            refuted: this module's own suite builds a spec-valued
            `evaluation.problem` axis purely to assert that it classifies as
            measurement-level, and a producer-only sweep never shows a value to
            anybody. Requiring a label to materialise a point would put a
            figure's concern inside the identity grammar. The refusal
            therefore lives in the analysis that reads a label
            (Annex 03 §A.6.1).

            **It takes no part in any identifier**, which is the contract a
            contender's `display` has (§2.2.1): renaming what a reader sees
            must not retrain anything or orphan a stored result. The join to a
            value is POSITION, which the sweep algebra already orders by.
    """

    path: str
    values: tuple[Any, ...]
    applies_to: tuple[str, ...] = ()
    compose: Composition = Composition.PRODUCT
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.labels and len(self.labels) != len(self.values):
            raise SpecificationError(
                f"axis {self.path!r} declares {len(self.labels)} labels for "
                f"{len(self.values)} values; a label is joined to its value by "
                "POSITION, so the two lists have to be the same length or the "
                "figure names one category after another's number"
            )
        if not self.values:
            raise SpecificationError(
                f"axis {self.path!r} declares no values, so it sweeps nothing; "
                "an axis with a single fixed value belongs in the spec instead"
            )
        if self.path == TEMPLATE_PATH or self.path.startswith(f"{TEMPLATE_PATH}."):
            raise SpecificationError(
                f"axis {self.path!r} sweeps the study's default training plan, "
                "which is materialised into every contender before any axis is "
                "applied -- so it reaches nobody and every point of this axis "
                "would derive the same identifiers. Sweep "
                f"{EXECUTED_PLAN_PATH}.* instead, which is the plan each "
                "contender is actually trained under"
            )

    @property
    def root(self) -> str:
        return self.path.split(".", 1)[0]

    @property
    def level(self) -> AxisLevel:
        """Which identifier this axis moves.

        Raises:
            SpecificationError: If the path's root is not part of the grammar.
                Named rather than defaulted, because a misspelled root would
                otherwise classify as something and cost either a wrong count
                of trainings or a collision.
        """
        try:
            return AXIS_ROOTS[self.root]
        except KeyError:
            raise SpecificationError(
                f"axis path {self.path!r} starts at {self.root!r}, which is not "
                f"part of a study; expected one of {sorted(AXIS_ROOTS)}"
            ) from None


@dataclass(frozen=True)
class StudyPoint:
    """One materialised point: everything needed to train and score once.

    Attributes:
        contender: The contender, with every applicable axis already applied.
        training: How it is fitted.
        evaluation: How it is scored.
        problem: What it is trained on.
        seed: Which replicate this point is.
        axis_values: The axis value this point was bound at, per axis path,
            for **only** the axes that applied to this contender. Empty for a
            study with no sweep, and empty *per axis* for a contender an
            `applies_to` excludes -- which is not a gap but the answer: a
            depth-invariant contender has no depth, and an analysis draws it
            as a flat reference rather than at some depth it never had.

            **It is not identity and never could be.** Every identifier below
            is derived from the four specs, into which the value has already
            been written; this mapping records *which declared value* put it
            there, so an analysis can label an axis without re-deriving it
            from a path -- and so an axis over whole objects (an evaluation
            problem, say) is still reportable, which reading a scalar back
            from the spec tree could not manage.
    """

    contender: ContenderSpec
    training: TrainingSpec
    evaluation: EvaluationSpec
    problem: ProblemSpec
    seed: int
    axis_values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> ModelID:
        return derive_model_id(self.problem, self.contender, self.training, self.seed)

    @property
    def measurement_id(self) -> MeasurementID:
        return derive_measurement_id(self.model_id, self.evaluation)


@dataclass(frozen=True)
class StudySpec:
    """One study: what is compared, how it is fitted, how it is scored, and
    what varies.

    Attributes:
        id: What the study is filed under. **Excluded from `StudyID`** -- and
            emitted under exactly the name Tier 4's
            `STUDY_KEYS_EXCLUDED_FROM_STUDY_ID` strips, which is a by-name
            contract rather than a coincidence.
        problem: The problem contenders are trained on.
        contenders: The competing controller families. Labels must be unique;
            they index every result.
        training: How learned contenders are fitted.
        evaluation: How every contender is scored.
        sweep: The independent variables. Each carries its own `compose`,
            which is how it joins the axes before it.
        gates: The scientific checks this study claims to meet. Opt-in -- a
            default gate would apply a claim to every study ever written,
            including the ones it is wrong for -- and part of `StudyID`,
            because D15 forbids a tier from relaxing one.
        analyses: The tidy tables to compute from this study's measurements
            (Annex 01 §2.6). **Excluded from `StudyID`**, under exactly the
            name Tier 4 strips: an analysis is a derivation *from* what the
            study computed, so declaring one must not invalidate a single
            stored measurement.
        figures: The figures to render from those tables. Excluded from
            `StudyID` for the same reason, and each one's `source` is checked
            against `analyses` here -- this is the only object holding both.
    """

    id: str
    problem: ProblemSpec
    contenders: tuple[ContenderSpec, ...]
    training: TrainingSpec
    evaluation: EvaluationSpec
    sweep: tuple[SweepAxis, ...] = ()
    gates: tuple[GateSpec, ...] = ()
    analyses: tuple[AnalysisSpec, ...] = ()
    figures: tuple[FigureSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.contenders:
            raise SpecificationError(
                f"study {self.id!r} declares no contenders, so it compares nothing"
            )
        self._materialise_training_plan()
        labels = [spec.resolved_label for spec in self.contenders]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise SpecificationError(
                f"study {self.id!r} has duplicate contender labels "
                f"{sorted(duplicates)}; labels index every result, so two "
                "contenders sharing one would make the output ambiguous"
            )
        # And the same refusal on the reader's side, for a stronger reason: two
        # series sharing one *drawn* name make a legend nobody can read, and the
        # greyscale gate would then report the collision by naming one string
        # twice.
        shown = [spec.resolved_display for spec in self.contenders]
        shared = {name for name in shown if shown.count(name) > 1}
        if shared:
            raise SpecificationError(
                f"study {self.id!r} shows {sorted(shared)} for more than one "
                "contender; a legend with two identical entries names neither, "
                "and the greyscale refusal would print the same string twice"
            )
        self._require_known_labels()
        self._require_coherent_rehost()
        require_unique_kinds(self.gates, self.id)
        self._require_firable_gates()
        require_unique_analysis_ids(self.analyses, self.id)
        require_unique_figure_ids(self.figures, self.id)
        require_resolvable_sources(
            self.figures, [spec.id for spec in self.analyses], self.id
        )

    def _materialise_training_plan(self) -> None:
        """Give the study's default plan to every contender that accepts one.

        **The single home of Annex 01 §2.3.4**, and it is here rather than in
        the loader for a reason Phase E3 already paid for: neither surface is
        privileged, and a study composed in Python must be the same study as the
        identical document read from TOML. A loader-only step would give the two
        different contenders and therefore different identifiers.

        It runs before the label checks below, because a contender that declares
        no plan cannot be resolved -- and therefore cannot be *named* -- until it
        has one.

        Idempotent by construction: after this every contender that takes a plan
        declares one, so re-running it (a tier writes `training.plan.epochs`
        through `dataclasses.replace`, which re-enters `__post_init__`) changes
        nothing. That is also why a tier writing only the template changes no
        effort, which is F2a's finding and the reason the shipped catalogue
        writes `contenders.*.config.plan.*` as well.
        """
        object.__setattr__(
            self,
            "contenders",
            tuple(
                spec.with_default_plan(self.training.plan) for spec in self.contenders
            ),
        )

    def _require_firable_gates(self) -> None:
        """Refuse a gate whose verdict is fixed before any compute runs.

        A gate the study cannot fail is the same defect as a gate nobody wrote,
        with the added cost that the study file says otherwise. Only the checks
        decidable from the specification live here; the rest are the runner's
        and the analysis tier's (see `gates.GateStage`).

        Raises:
            SpecificationError: If `constraint_binds` is declared on an
                unconstrained problem, or `dimension_robust` over a set that
                excludes the study's own control dimension.
        """
        for gate in self.gates:
            if (
                gate.kind is GateKind.CONSTRAINT_BINDS
                and self.problem.data.control_bound is None
            ):
                raise SpecificationError(
                    f"study {self.id!r} declares a {gate.kind.value!r} gate but its "
                    "problem has control_bound=None; the gate asks what fraction "
                    "of the time the box is active, and with no box there is no "
                    "fraction -- the study would run to completion reporting a "
                    "check it never made"
                )
            if gate.kind is GateKind.DIMENSION_ROBUST:
                declared = tuple(gate.config["control_dims"])
                own = self.problem.control_dim
                if own not in declared:
                    raise SpecificationError(
                        f"study {self.id!r} declares a {gate.kind.value!r} gate over "
                        f"control dimensions {list(declared)}, which excludes its "
                        f"own dimension {own}; the study runs at {own} whatever "
                        "else it sweeps, so the gate claims robustness over "
                        "dimensions never visited and says nothing about the one "
                        "that is"
                    )
            if gate.kind is GateKind.STEP_IS_INVERSE_LIPSCHITZ:
                known = {spec.resolved_label for spec in self.contenders}
                for label in gate.config["contenders"]:
                    if label not in known:
                        raise SpecificationError(
                            f"study {self.id!r} declares a {gate.kind.value!r} "
                            f"gate naming {label!r}, which no contender "
                            f"declaration carries; declared labels: "
                            f"{sorted(known)}"
                        )
                    spec = next(s for s in self.contenders if s.resolved_label == label)
                    if "step_size_init" not in spec.config:
                        raise SpecificationError(
                            f"study {self.id!r} declares a {gate.kind.value!r} "
                            f"gate naming {label!r}, whose config carries no "
                            "step_size_init; the gate would check nothing for "
                            "it, and a check the study file claims and nobody "
                            "performs is the inert-declaration defect"
                        )

    def _require_coherent_rehost(self) -> None:
        """Refuse rehost declarations nothing can read (Annex 01 §2.4.1).

        Two shapes, both the inert-declaration class this project keeps
        rediscovering: an override naming a contender the study does not
        have, and overrides in a document that is blind everywhere — blind
        points never read them, so unless some axis sweeps
        `evaluation.rehost`, the declaration decorates a study it cannot
        touch. Whether an override's *plant* is ever evaluated on needs the
        expanded points and is `materialise`'s check.
        """
        known = {spec.resolved_label for spec in self.contenders}
        for override in self.evaluation.rehost_overrides:
            if override.contender not in known:
                raise SpecificationError(
                    f"rehost override names contender {override.contender!r}, "
                    f"which study {self.id!r} does not declare; available: "
                    f"{sorted(known)}"
                )
        sweeps_mode = any(axis.path == "evaluation.rehost" for axis in self.sweep)
        if (
            self.evaluation.rehost == REHOST_BLIND
            and self.evaluation.rehost_overrides
            and not sweeps_mode
        ):
            raise SpecificationError(
                f"study {self.id!r} declares rehost_overrides while blind "
                "everywhere: a blind point never reads an override, and no "
                "axis sweeps evaluation.rehost, so the declaration decorates "
                "a study it cannot touch (Annex 01 §2.4.1)"
            )
        if (
            self.training.problem is not None
            and self.training.problem.problem_id == self.problem.problem_id
        ):
            raise SpecificationError(
                f"study {self.id!r} declares training.problem equal to its own "
                "problem; that is the matched case wearing a declaration, and "
                "signing it would mint a second ModelID for physics the store "
                "already holds under the first (Annex 01 §2.3.5). Leave it out"
            )

    def _require_matched_overrides(self, points: tuple[StudyPoint, ...]) -> None:
        """The gate-firability rule, applied to overrides at expansion.

        An override whose (contender, plant) pair no aware point carries can
        never fire; running the study anyway would report a document claiming
        a replacement nobody performed.
        """
        for override in self.evaluation.rehost_overrides:
            if not self._fires(override, points):
                raise SpecificationError(
                    f"rehost override for {override.contender!r} on problem "
                    f"{override.problem.problem_id} matches no point: no aware "
                    f"point of study {self.id!r} evaluates that contender on "
                    "that plant, so the override can never fire"
                )

    def zip_group_paths(self, axis_path: str) -> tuple[str, ...]:
        """The paths of the zipped group containing `axis_path`, in order.

        The partition is `_groups()`'s — the same one `_combinations` executes
        — exposed because a zipped group is **one effective axis**: an analysis
        keying its categories by a single member's value would collapse two
        positions that share it (measured: the two rehost modes of one plant
        landed in one group and were refused as repeated seeds). A product
        axis, or the only axis, is a group of itself.

        Raises:
            SpecificationError: If the path is not a swept axis of this study.
        """
        for group in self._groups():
            paths = tuple(axis.path for axis in group)
            if axis_path in paths:
                return paths
        raise SpecificationError(
            f"axis {axis_path!r} is not swept by study {self.id!r}; declared "
            f"axes: {', '.join(axis.path for axis in self.sweep) or '(none)'}"
        )

    # -- the algebra ------------------------------------------------------

    def _groups(self) -> list[list[SweepAxis]]:
        """The sweep partitioned into zipped groups (Annex 01 §2.5).

        An axis declaring `ZIP` joins the group immediately before it; one
        declaring `PRODUCT` opens a new group. The first axis always opens one,
        whatever it declares -- it has nothing to join, and treating a leading
        `ZIP` as an error would refuse a study that is perfectly well formed.
        """
        groups: list[list[SweepAxis]] = []
        for axis in self.sweep:
            if groups and axis.compose is Composition.ZIP:
                groups[-1].append(axis)
            else:
                groups.append([axis])
        return groups

    def _zipped(self, group: list[SweepAxis]) -> list[tuple[Any, ...]]:
        """One group's value tuples, paired positionwise.

        Raises:
            SpecificationError: If the group's axes are not all the same
                length. Zip pairs positionwise, and silently truncating to the
                shortest would drop points the author wrote down.
        """
        lengths = {len(axis.values) for axis in group}
        if len(lengths) != 1:
            raise SpecificationError(
                f"study {self.id!r} zips axes of different length: "
                + ", ".join(f"{a.path}={len(a.values)}" for a in group)
                + "; zip pairs positionwise, and silently truncating to the "
                "shortest would drop points the author wrote down"
            )
        return list(zip(*(axis.values for axis in group), strict=True))

    def _combinations(self) -> Iterator[tuple[Any, ...]]:
        """Value tuples, one entry per axis, in declaration order.

        The composition is **per axis**, so a study may product some axes and
        zip others -- which is the whole reason the field exists. A control
        bound swept alongside a coupled "rehosted" companion must vary with it
        and not independently of it, and a study forced to choose one rule for
        every axis at once could only fake that with a Cartesian product and
        post-hoc filtering.
        """
        if not self.sweep:
            yield ()
            return
        groups = self._groups()
        for combination in product(*(self._zipped(group) for group in groups)):
            # `product` yields one tuple per group; flatten back to one entry
            # per axis, in the order the axes were declared, because that is
            # what `_apply` zips against.
            yield tuple(value for bundle in combination for value in bundle)

    def _require_known_labels(self) -> None:
        """Refuse an axis naming a contender the study does not have.

        Checked at construction rather than at expansion. Annex 01 §4 puts
        `applies_to` naming contenders that exist under *structural*
        validation, which "runs at parse time, before any compute" -- and an
        axis that named nothing was previously accepted by a `StudySpec` and
        only refused by `materialise`, so a study document could be loaded,
        catalogued and reviewed while carrying an axis that applies to nobody.
        """
        known = {spec.resolved_label for spec in self.contenders}
        for axis in self.sweep:
            unknown = set(axis.applies_to) - known
            if unknown:
                raise SpecificationError(
                    f"axis {axis.path!r} applies_to names {sorted(unknown)}, "
                    f"which are not contenders of study {self.id!r}; "
                    f"available: {sorted(known)}"
                )

    def materialise(self) -> tuple[StudyPoint, ...]:
        """Expand the sweep into the points a runner would execute.

        `_expand` plus the expansion-dependent refusal: an override no point
        can fire is refused here rather than silently carried (the
        gate-firability rule). The expansion itself is separate so that the
        one caller with a legitimate reason to *remove* a dead override — the
        smoke subset, whose truncation is what killed it — can ask which
        without tripping the refusal it exists to repair.

        Raises:
            SpecificationError: If an axis names an unknown contender, zips
                axes of unequal length, resolves to no such field, or if a
                declared rehost override matches no point.
        """
        expanded = self._expand()
        self._require_matched_overrides(expanded)
        self._require_distinct_worlds(expanded)
        self._require_inverse_lipschitz_steps(expanded)
        return expanded

    def _require_inverse_lipschitz_steps(self, points: tuple[StudyPoint, ...]) -> None:
        """The `step_is_inverse_lipschitz` gate, decided over the POSITIONS.

        Evaluated on the materialised points rather than on the document,
        because the Figure-3 shape carries the literal as the plant axis's
        zipped companion — the gate's subject is the (plant, step) pair as it
        will actually run, and reading the declaration instead would go blind
        the moment a reordering unpaired them (the plan names this as the
        gate's blind spot).

        Exact equality, deliberately: the literal is authored by the same
        `eigvalsh` kernel this check recomputes with
        (`spec.gates.inverse_lipschitz_step`), measured bit-stable across
        thread counts and backends, and one ULP of a declared step moves the
        `ModelID` — a tolerance would readmit exactly the two-identifiers
        defect the strict `step_size_max` refusal closed.

        Raises:
            SpecificationError: If a named contender's declared step at any
                position is not that position's plant's exact ``1/L``, quoting
                the expected repr so the fix is a paste.
        """
        named: set[str] = set()
        fraction = 1.0
        for gate in self.gates:
            if gate.kind is GateKind.STEP_IS_INVERSE_LIPSCHITZ:
                named.update(gate.config["contenders"])
                # Absent after canonicalisation means the default (Annex 01
                # §4's conditional signing); kinds are unique per study, so
                # one fraction governs every named contender.
                fraction = float(gate.config.get("fraction", 1.0))
        if not named:
            return
        expected_by_problem: dict[str, float] = {}
        for point in points:
            label = point.contender.resolved_label
            if label not in named:
                continue
            declared = point.contender.config.get("step_size_init")
            if declared is None:  # pragma: no cover — `_require_firable_gates`
                # refused a named contender without the field at construction,
                # and an axis can replace a key but never remove one.
                continue
            problem_key = str(point.problem.problem_id)
            if problem_key not in expected_by_problem:
                # `fraction ×` the kernel's value, NEVER a re-derived
                # 1/(fraction⁻¹·L): the admitted fractions are exact in
                # binary64 (Annex 01 §4), so bit equality survives; the
                # other spelling can differ in its last ULP.
                expected_by_problem[problem_key] = fraction * inverse_lipschitz_step(
                    point.problem
                )
            expected = expected_by_problem[problem_key]
            if float(declared) != expected:
                law = "1/L" if fraction == 1.0 else f"{fraction} x 1/L"
                raise SpecificationError(
                    f"study {self.id!r}: contender {label!r} declares "
                    f"step_size_init = {declared!r} on plant {problem_key} "
                    f"(n = {point.problem.state_dim}), whose exact {law} is "
                    f"{expected!r}; the step_is_inverse_lipschitz gate refuses "
                    "any other value. Paste the literal authored by "
                    "tools/report_pgd_step.py for this plant"
                    + ("" if fraction == 1.0 else ", halved exactly")
                )

    def _require_distinct_worlds(self, points: tuple[StudyPoint, ...]) -> None:
        """Refuse a point whose training world equals the plant it is told.

        The study-level form of this refusal lives in `__post_init__`; an axis
        can write the same defect per point, and there the author meant the
        matched case and has a spelling for it (Annex 01 §2.3.5): `""` writes
        the absence, signs nothing, and reuses the matched models bit for bit.
        """
        for point in points:
            world = point.training.problem
            if world is not None and world.problem_id == point.problem.problem_id:
                raise SpecificationError(
                    f"a sweep position gives contender "
                    f"{point.contender.resolved_label!r} a training world equal "
                    "to the plant it is told; that would sign a second ModelID "
                    'for the matched physics. Spell the matched case "" on the '
                    "training.problem axis (Annex 01 §2.3.5)"
                )

    def _expand(self) -> tuple[StudyPoint, ...]:
        """The sweep as points, checks that need the expansion excluded.

        Each point carries the fully-resolved contender, training, evaluation
        and problem for one combination, so the identifiers it derives are the
        identifiers that would actually be produced. That is what makes the
        acceptance suite's counts a measurement rather than a restatement of
        the classification.

        **Points are deduplicated by identity**, and that is not an
        optimisation. An axis with `applies_to` narrower than the contender set
        is a no-op for every contender it does not name, so the naive expansion
        emits that contender once per axis value -- byte-identical points, the
        same model and the same measurement, differing in nothing. NB04's own
        shape makes this concrete: six contenders over eight depths, of which
        only three carry a depth, expands to 48 points of which 21 are exact
        repeats. Emitting them would have the runner score the analytic
        baseline eight times to get eight identical numbers.

        Returns:
            One `StudyPoint` per distinct (contender, training, evaluation,
            problem, seed), in declaration order.

        Raises:
            SpecificationError: If an axis names an unknown contender, zips
                axes of unequal length, or resolves to no such field.
        """
        self._require_known_labels()
        points: list[StudyPoint] = []
        seen: set[tuple[str, str]] = set()
        for combination in self._combinations():
            for contender in self.contenders:
                resolved = self._apply(combination, contender)
                for seed in resolved.training.seeds:
                    point = StudyPoint(
                        contender=resolved.contender,
                        training=resolved.training,
                        evaluation=resolved.evaluation,
                        problem=resolved.problem,
                        seed=seed,
                        axis_values=resolved.axis_values,
                    )
                    # Keyed on the derived identifiers rather than on the point
                    # itself: a `StudyPoint` holds NumPy arrays and is not
                    # hashable, and the identifiers are precisely the definition
                    # of "the same work" the store will apply anyway.
                    key = (str(point.model_id), str(point.measurement_id))
                    if key in seen:
                        continue
                    seen.add(key)
                    points.append(point)
        return tuple(points)

    def unfired_override_indices(self) -> tuple[int, ...]:
        """Which declared rehost overrides no point of this study can fire.

        By index into `evaluation.rehost_overrides`, because a `RehostOverride`
        holds a `ProblemSpec` and comparing those equates NumPy arrays. Exposed
        for the one caller with a legitimate reason to *remove* an unfirable
        override rather than refuse it: a tier's smoke subset drops sweep
        positions, and an override whose position was truncated away is the
        truncation's dead weight, not the author's error.
        """
        expanded = self._expand()
        return tuple(
            index
            for index, override in enumerate(self.evaluation.rehost_overrides)
            if not self._fires(override, expanded)
        )

    @staticmethod
    def _fires(override: Any, points: tuple[StudyPoint, ...]) -> bool:
        wanted = str(override.problem.problem_id)
        return any(
            point.evaluation.rehost in (REHOST_AWARE, REHOST_FULL)
            and point.contender.resolved_label == override.contender
            and str(point.evaluation.problem.problem_id) == wanted
            for point in points
        )

    def _apply(self, combination: tuple[Any, ...], contender: ContenderSpec) -> _Bound:
        """One combination, applied to one contender's slice of the study.

        The `continue` below is what makes `axis_values` a record of what
        actually happened rather than of what was declared: an axis an
        `applies_to` excludes writes nothing into the specs and therefore
        contributes no entry, so the two can never disagree.
        """
        bound = _Bound(
            contender=contender,
            training=self.training,
            evaluation=self.evaluation,
            problem=self.problem,
        )
        for axis, value in zip(self.sweep, combination, strict=True):
            if axis.applies_to and contender.resolved_label not in axis.applies_to:
                continue
            bound = bound.with_axis(axis, value)
        return bound

    # -- identity ---------------------------------------------------------

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member. `id` is present so Tier 4 can strip it by name."""
        return {
            "type": type(self).__name__,
            "id": self.id,
            "problem": self.problem.get_signature(),
            "contenders": {
                spec.resolved_label: spec.get_signature() for spec in self.contenders
            },
            "training": self.training.get_signature(),
            "training_ctx": self.training.ctx.get_signature(),
            "evaluation": self.evaluation.get_signature(),
            "evaluation_problem": str(self.evaluation.problem.problem_id),
            "sweep": [
                {
                    "path": axis.path,
                    "values": [_as_json(value) for value in axis.values],
                    "applies_to": sorted(axis.applies_to),
                    # The FIRST axis's composition is unread by the algebra, so
                    # it must not be signed either -- two studies that expand to
                    # the same points would otherwise be two studies.
                    "compose": (
                        Composition.PRODUCT.value if index == 0 else axis.compose.value
                    ),
                }
                for index, axis in enumerate(self.sweep)
            ],
            # Sorted by kind, for the reason `metrics` is sorted: two authors
            # listing the same checks in a different order must collide rather
            # than diverge. `require_unique_kinds` makes the key total.
            "gates": [
                gate.get_signature()
                for gate in sorted(self.gates, key=lambda gate: gate.kind.value)
            ],
            # Emitted so the signature stays a faithful dump, and stripped by
            # name in `store.ids.study_id` -- the same contract `id` has, and
            # for the same reason: every identity exclusion in this project is
            # auditable in one list rather than in a module's silence.
            "analyses": [spec.get_signature() for spec in self.analyses],
            "figures": [spec.get_signature() for spec in self.figures],
        }

    @property
    def study_id(self) -> StudyID:
        """This study's identity, independent of the name it is filed under."""
        return study_id(self.get_signature())


@dataclass(frozen=True)
class _Bound:
    """A study's four specs, part-way through having a combination applied."""

    contender: ContenderSpec
    training: TrainingSpec
    evaluation: EvaluationSpec
    problem: ProblemSpec
    #: What each applied axis was bound to, accumulated by `with_axis` so that
    #: it records the writes that actually happened.
    axis_values: Mapping[str, Any] = field(default_factory=dict)

    #: Which field of this object each axis root addresses.
    _ROOTS: dict[str, str] = field(
        default_factory=lambda: {
            "problem": "problem",
            "contenders": "contender",
            "training": "training",
            "evaluation": "evaluation",
        },
        repr=False,
        compare=False,
    )

    def with_axis(self, axis: SweepAxis, value: Any) -> _Bound:
        axis.level  # classify first, so an unknown root fails before mutation
        attribute = self._ROOTS[axis.root]
        segments = axis.path.split(".")[1:]
        # `contenders.*.…`: the wildcard selects the contender already in hand.
        if segments and segments[0] == "*":
            segments = segments[1:]
        target = getattr(self, attribute)
        replaced = value if not segments else replace_at(target, segments, value)
        changes: dict[str, Any] = {attribute: replaced}
        if attribute == "problem" and not segments:
            followed = self._evaluation_following(value)
            if followed is not None:
                changes["evaluation"] = followed
        return dataclasses.replace(
            self,
            **changes,
            axis_values={**self.axis_values, axis.path: value},
        )

    def _evaluation_following(self, problem: ProblemSpec) -> EvaluationSpec | None:
        """The matched evaluation, moved with the swept root problem.

        Sweeping `problem` moves the WHOLE study problem — the matched case at
        every axis position — so an evaluation that was matched as declared
        follows: its problem becomes the position's plant and its batch
        dimensions re-derive from that plant, exactly as the loader derives
        them from the declared plant at load. An evaluation the author pinned
        to a different plant is a shifted study and stays pinned; `None` says
        so. Measured before this existed: the second point of a two-plant root
        sweep was scored on the DECLARED plant with the declared plant's batch
        shapes — an n-plant model on the 4-plant instance.

        The condition compares against `self.problem` *before* the axis
        replaces it, which inside one `_apply` is the study's declared plant —
        so an `evaluation.problem` axis that explicitly bound the declared
        plant (the matched position of a fig2-style zip) follows too, which is
        correct because an explicit-equal declaration signs identically to the
        matched default (measured in Phase D).
        """
        if str(self.evaluation.problem.problem_id) != str(self.problem.problem_id):
            return None
        protocol = replace_at(
            self.evaluation.protocol, ["batch_spec", "state_dim"], problem.state_dim
        )
        protocol = replace_at(protocol, ["batch_spec", "horizon"], problem.data.horizon)
        return dataclasses.replace(self.evaluation, problem=problem, protocol=protocol)


def _as_json(value: Any) -> Any:
    """Axis values are arbitrary spec objects; a signature needs JSON."""
    if hasattr(value, "get_signature"):
        return value.get_signature()
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value
