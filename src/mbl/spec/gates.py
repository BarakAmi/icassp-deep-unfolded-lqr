"""The declarative scientific gates a study inherits (Stage 2 Phase E2).

Annex 01 §4. Today these are hand-written assertion cells that each notebook
has to remember to write; here they are part of the grammar, so every study
carries them whether or not its author thought of them.

**The failure they prevent is on this project's record.** A box constraint so
wide it never bound, silently degenerating a constrained study into an
expensive repeat of the unconstrained one -- a result that looked entirely
plausible and answered a different question from the one asked.

**A gate misspelled into silence is worse than no gate**, because it reads in
the study file as a check that is being performed. Both the kind set and each
kind's key set are therefore closed and validated at parse time, and two gates
of one kind are a declaration error rather than a conjunction: a study carrying
two `stability` thresholds says nothing about which one it claims to meet.

**Gates participate in `StudyID`.** They are part of what a study claims, not
decoration on it, and D15's rule forbidding a tier from writing `gates.*` --
"a cheaper run must not be a less-checked one" -- would be much weaker if two
studies differing only in how strictly they are checked shared an identity.

**What this module does not do is run them.** The kinds divide by what each
needs in hand, which is why `GateStage` exists: `stability` and
`dimension_robust` are decidable from the specification, while
`constraint_binds` and `contenders_separate` cannot be decided before training
at all -- whether two contenders separate is what the training was run to find
out, and how often a controller sits on the box is a property of that
controller's own trajectory. Evaluation belongs to the analysis tier (Tier 6);
the declaration and its validation belong here.

**`constraint_binds` moved from `PREFLIGHT` to `POST_HOC` on 2026-08-22**
(Annex 01 §4.1). Its pre-flight definition nominated one cheap rollout of a
non-learned contender as a proxy, which buys an abort before a night of
training and pays for it with the wrong quantity. The stage is **not signed** --
`get_signature` emits `kind` and `config` and nothing else -- so the move
orphans no stored record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .errors import SpecificationError


class GateKind(StrEnum):
    """The gates Annex 01 §4 declares. A closed set."""

    #: The box constraint must actually be active often enough to matter.
    CONSTRAINT_BINDS = "constraint_binds"
    #: The contenders must be distinguishable, or the study measures noise.
    CONTENDERS_SEPARATE = "contenders_separate"
    #: The claim must hold across the listed control dimensions.
    DIMENSION_ROBUST = "dimension_robust"
    #: The closed loop must not be driven unstable.
    STABILITY = "stability"
    #: Every named contender's declared step must be exactly the plant's 1/L.
    STEP_IS_INVERSE_LIPSCHITZ = "step_is_inverse_lipschitz"


class GateStage(StrEnum):
    """When a gate's verdict can be reached."""

    #: Decidable from the specification alone, before any compute.
    PARSE = "parse"
    #: Needs one cheap rollout, but no training.
    PREFLIGHT = "preflight"
    #: Needs the measurements themselves; a gate on publication.
    POST_HOC = "post_hoc"


def _fraction(name: str, value: Any) -> float:
    """A proportion in (0, 1]. Zero is refused because a `min_fraction` of zero
    is exactly the historical failure: a constraint-binding check that passes
    on a constraint which never binds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpecificationError(f"{name} must be a number, got {value!r}")
    if not 0 < float(value) <= 1:
        raise SpecificationError(
            f"{name} must lie in (0, 1], got {value!r}; a threshold outside its "
            "own range is a gate that either never fires or always does"
        )
    return float(value)


def _positive(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpecificationError(f"{name} must be a number, got {value!r}")
    if float(value) <= 0:
        raise SpecificationError(f"{name} must be positive, got {value!r}")
    return float(value)


def _dimensions(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise SpecificationError(
            f"{name} must be a sequence of dimensions, got {value!r}"
        )
    dims = tuple(value)
    if not dims:
        raise SpecificationError(
            f"{name} is empty, so this gate asserts robustness over nothing"
        )
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise SpecificationError(
                f"{name} must hold positive integer dimensions, got {dim!r}"
            )
    return dims


#: The step-law fractions Annex 01 §4 admits. Closed on purpose: every member is
#: exact in binary64 (multiplying by a power of two is a pure exponent shift
#: wherever the result is normal), which is what lets the gate keep comparing
#: with bit equality; an inexact fraction would quietly turn that into a
#: tolerance nobody declared.
#:
#: Widened from `(1.0, 0.5)` to the powers of two on 2026-08-12. The old pair
#: admitted only the stable half of the classical range, so a study sweeping the
#: step to find where stability *ends* was refused by the gate meant to keep its
#: literals honest -- the alternative being no gate at all. `4.0` is `2/L`, the
#: classical limit itself; `8.0` is twice past it. The gate asserts the literal
#: is exactly the declared multiple of the plant's own `1/L`, never that the
#: step is stable, so this widens what a document can SAY and not what it may
#: get away with.
EXACT_STEP_FRACTIONS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _step_fraction(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SpecificationError(f"{name} must be a number, got {value!r}")
    if float(value) not in EXACT_STEP_FRACTIONS:
        raise SpecificationError(
            f"{name} must be one of {list(EXACT_STEP_FRACTIONS)}, got {value!r}; "
            "the closed set is what keeps the gate's bit-equality law exact "
            "(Annex 01 §4)"
        )
    return float(value)


def _labels(name: str, value: Any) -> tuple[str, ...]:
    """A non-empty sequence of distinct, non-empty contender labels."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise SpecificationError(f"{name} must be a sequence of labels, got {value!r}")
    labels = tuple(value)
    if not labels:
        raise SpecificationError(
            f"{name} is empty, so this gate checks the step of nobody; a gate "
            "that can decide nothing is a declaration error"
        )
    for label in labels:
        if not isinstance(label, str) or not label:
            raise SpecificationError(
                f"{name} must hold non-empty contender labels, got {label!r}"
            )
    if len(set(labels)) != len(labels):
        raise SpecificationError(f"{name} names a contender twice: {labels}")
    return labels


#: kind -> (when it can be decided, key -> the validator that normalises it).
#: The value functions do double duty: they refuse the out-of-range case and
#: they canonicalise, so two spellings of one threshold (`1` and `1.0`) cannot
#: become two `StudyID`s.
GATE_SCHEMA: dict[GateKind, tuple[GateStage, dict[str, Any]]] = {
    GateKind.CONSTRAINT_BINDS: (GateStage.POST_HOC, {"min_fraction": _fraction}),
    GateKind.CONTENDERS_SEPARATE: (GateStage.POST_HOC, {"min_relative_gap": _positive}),
    GateKind.DIMENSION_ROBUST: (GateStage.PARSE, {"control_dims": _dimensions}),
    GateKind.STABILITY: (GateStage.PARSE, {"max_spectral_radius": _positive}),
    GateKind.STEP_IS_INVERSE_LIPSCHITZ: (GateStage.PARSE, {"contenders": _labels}),
}

#: kind -> optional keys, each with a (validator, default) pair. An absent key
#: means the default; a key spelled AT its default is dropped after validation
#: so the two spellings of one claim share one `StudyID` (conditional signing,
#: the D24 pattern) -- and so a document written before the key existed keeps
#: its identity bit-for-bit.
GATE_OPTIONAL_SCHEMA: dict[GateKind, dict[str, tuple[Any, Any]]] = {
    GateKind.STEP_IS_INVERSE_LIPSCHITZ: {"fraction": (_step_fraction, 1.0)},
}


def inverse_lipschitz_constant(problem: Any) -> float:
    """``L`` for a frozen problem, from its matrices alone.

    The spec-tier re-expression of `models/analytic`'s
    `problem_gradient_lipschitz_constant`: same recursion, same
    gradient-coefficient stack, same `eigvalsh` — all resolved through the
    single-home kernels in `core.kernels.riccati`, which is what lets the
    `step_is_inverse_lipschitz` gate compare against an authored literal with
    exact equality. `models` is `FORBIDDEN_FOR_SPEC`, so the narrowing that
    `require_linear_quadratic` performs there is re-expressed here against
    `ProblemSpec.data` directly.

    Args:
        problem: A `ProblemSpec` (typed `Any` to keep this module free of the
            import cycle with `problem.py`).

    Returns:
        ``L`` over the problem's own horizon.
    """
    import numpy as np

    from ..core.kernels.riccati import (
        gradient_lipschitz_constant,
        riccati_recursion,
    )

    data = problem.data
    A_2d = np.asarray(data.system["A"])
    B_2d = np.asarray(data.system["B"])
    Q = np.asarray(data.cost["Q"])
    R = np.asarray(data.cost["R"])
    horizon = int(data.horizon)
    # Densified per step, the same per-step-indexable contract the kernel
    # documents; for a time-invariant plant this is `horizon` views of one
    # matrix and the arithmetic is bit-identical to the models path.
    A = np.broadcast_to(A_2d, (horizon, *A_2d.shape))
    B = np.broadcast_to(B_2d, (horizon, *B_2d.shape))
    P, _ = riccati_recursion(A, B, Q, R, horizon)
    return gradient_lipschitz_constant(P, A, B, R)


def inverse_lipschitz_step(problem: Any) -> float:
    """``1/L`` — the exact float the gate expects a document to declare,
    computed the way `tools/report_pgd_step.py` authors it."""
    return 1.0 / inverse_lipschitz_constant(problem)


@dataclass(frozen=True)
class GateSpec:
    """One declared check.

    Attributes:
        kind: What is being checked. A `GateKind`; a plain string is accepted
            and coerced, since the YAML surface has nothing else to give.
        config: The kind's thresholds. Every key required, no key spare, each
            value normalised by the schema.
    """

    kind: GateKind
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            kind = GateKind(self.kind)
        except ValueError:
            raise SpecificationError(
                f"unknown gate kind {self.kind!r}; expected one of "
                f"{sorted(member.value for member in GateKind)}"
            ) from None
        object.__setattr__(self, "kind", kind)

        _, schema = GATE_SCHEMA[kind]
        optional = GATE_OPTIONAL_SCHEMA.get(kind, {})
        missing = sorted(set(schema) - set(self.config))
        if missing:
            raise SpecificationError(
                f"gate {kind.value!r} is missing {', '.join(missing)}; it needs "
                f"{sorted(schema)}. A gate with no threshold has no verdict, and "
                "defaulting one would invent a claim the author never made"
            )
        spare = sorted(set(self.config) - set(schema) - set(optional))
        if spare:
            raise SpecificationError(
                f"gate {kind.value!r} does not accept {', '.join(spare)}; "
                f"accepted: {sorted(schema) + sorted(optional)}. An unrecognised "
                "key is how a real gate turns into a decorative one"
            )
        config = {key: check(key, self.config[key]) for key, check in schema.items()}
        for key, (check, default) in optional.items():
            if key not in self.config:
                continue
            value = check(key, self.config[key])
            # Conditional signing: a key spelled AT its default is the same
            # claim as an absent key, so it must derive the same StudyID --
            # dropped here, after validation, exactly once.
            if value != default:
                config[key] = value
        object.__setattr__(self, "config", config)

    @property
    def stage(self) -> GateStage:
        """When this gate's verdict can be reached."""
        return GATE_SCHEMA[self.kind][0]

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member.

        Keys are sorted so that two authors writing one gate's thresholds in a
        different order collide rather than diverge.

        `control_dims` becomes a list so the tree survives a JSON round trip
        unchanged -- `spec.json` gives a list back, and a tree that compared
        unequal to itself after storage would be a nasty thing to debug. It is
        **not** what makes the tree hashable: `json.dumps` serialises a tuple
        and a list identically, and the two spellings were measured to give the
        same digest. The strict-JSON check is carried by the enum values, which
        pass only because `GateKind` is a `StrEnum`."""
        return {
            "kind": self.kind.value,
            "config": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in sorted(self.config.items())
            },
        }


def require_unique_kinds(gates: tuple[GateSpec, ...], study_id: str) -> None:
    """Refuse a study declaring one kind twice.

    Args:
        gates: The declared gates.
        study_id: The study's name, for the message.

    Raises:
        SpecificationError: If any kind appears more than once.
    """
    kinds = [gate.kind.value for gate in gates]
    repeated = sorted({kind for kind in kinds if kinds.count(kind) > 1})
    if repeated:
        raise SpecificationError(
            f"study {study_id!r} declares {', '.join(repeated)} more than once; "
            "gates of one kind are not a conjunction, and a study carrying two "
            "thresholds says nothing about which one it claims to meet"
        )
