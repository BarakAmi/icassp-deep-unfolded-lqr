"""How every contender is scored (Stage 2 Phase D).

An `EvaluationSpec` is deliberately a *sibling* of the training declaration
rather than a part of it. That separation is the whole of the re-architecture's
central fix: today a model's cache key contains its evaluation protocol, so
raising an evaluation batch count or adding a metric invalidates every trained
model and retrains it. Here an evaluation cannot reach a `ModelID` at all,
because `derive_model_id` has no parameter through which one could be passed.

**The evaluation problem may differ from the training problem, and that is the
point.** When it does, the measurement *is* a distribution-shift result --
there is no flag, no second code path and no out-of-distribution subsystem,
just a different `ProblemSpec` in this object and one extra rollout.

**The evaluation carries its own compute context (D20).** Contenders may be
*trained* at different precisions -- the COCP family requires float64 while the
rest default to float32 -- but they must be *scored* at one common precision, or
the comparison measures arithmetic rather than control.

**What the signature omits, and why it differs from `TrainingSpec`.** One rule
covers both: a spec omits exactly what the Tier-4 identifier function takes on
its own axis. `measurement_id(model, eval_problem, eval_protocol)` has an
`eval_problem` parameter, so `problem` is absent here. It has no `ctx`
parameter, so the context *is* present -- and must be, since scoring at float32
and at float64 gives different numbers. `TrainingSpec` is the mirror image:
`model_id` has a `ctx` parameter, so its tree omits the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.runtime import ComputeContext
from ..core.utils.signing import Signable
from .errors import SpecificationError
from .problem import ProblemSpec

#: What a study measures when it says nothing more specific.
DEFAULT_METRICS: tuple[str, ...] = ("expected_cost",)

#: The two rehost modes (Annex 01 §2.4.1, D24). Blind is the prior behaviour
#: and the default; aware rebuilds the controller against the evaluation
#: problem. A module-level tuple rather than an Enum because the mode is an
#: ordinary sweepable scalar, and `"aware"` in a TOML axis must be the value
#: itself rather than a name to be looked up.
REHOST_BLIND = "blind"
REHOST_AWARE = "aware"
#: Annex 01 §2.4.1 as amended 2026-08-09: the doctrine's second half. A
#: family that trains nothing has no offline stage to freeze, so under
#: ``full`` an analytic recipe re-synthesises whole against the evaluation
#: plant (overrides included), while a trainable recipe rebuilds by the
#: identical algorithm as ``aware``. A NEW mode rather than a correction to
#: ``aware``, because the rebuild algorithm is not signed into
#: `MeasurementID` and re-defining a mode in place would re-caption every
#: stored record built under it.
REHOST_FULL = "full"
REHOST_MODES: tuple[str, ...] = (REHOST_BLIND, REHOST_AWARE, REHOST_FULL)


@dataclass(frozen=True)
class RehostOverride:
    """One construction literal, replaced at an aware rebuild (§2.4.1).

    A declared constant can encode a problem-derived quantity — the analytic
    PGD's step is the frozen 1/L of a *named plant* — so an aware rehost onto
    another plant carries that plant's own authored literal. The override
    signs by the resolved plant's `ProblemID`, never by filename or axis
    label, and participates in `MeasurementID` only (§5 rule 3).

    Attributes:
        contender: The declared contender's `label`.
        problem: The evaluation plant this override belongs to, resolved
            beside the document (D19).
        config: Sorted (field, value) pairs replaced on the resolved recipe
            at the aware rebuild.
    """

    contender: str
    problem: ProblemSpec
    config: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.config:
            raise SpecificationError(
                f"rehost override for {self.contender!r} declares no config "
                "fields; an override that replaces nothing is a declaration "
                "error rather than a valid study"
            )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the contender, the plant's id, the pairs."""
        return {
            "contender": self.contender,
            "problem": str(self.problem.problem_id),
            "config": [[key, value] for key, value in sorted(self.config)],
        }


@dataclass(frozen=True)
class EvaluationSpec:
    """One scoring protocol, shared by every contender in a study.

    Attributes:
        problem: The problem to score on. **May** differ from the problem the
            model was trained on; when it does, this is a shifted measurement.
            Excluded from `get_signature` -- `measurement_id` takes it on its
            own axis.
        protocol: The batch specification and batch count, held through
            `Signable` rather than as `experiments.EvaluationProtocol`, which
            would pull `experiments` and `applications` into Tier 3.
        ctx: The scoring context. Its **own**, deliberately, so that contenders
            trained at different precisions are still compared at one (D20).
        metrics: What to compute. Named rather than passed as callables, so an
            evaluation is data and its identity is derivable.
        rehost: Whether the controller is rebuilt against the training problem
            (blind, the default) or against `problem` (aware) — Annex 01
            §2.4.1, D24. Signs only off its default, so a blind spec's
            signature is byte-identical to one predating the field.
        rehost_overrides: Per-(contender, plant) construction literals applied
            at an aware rebuild. Read only under aware; a document that is
            blind everywhere and declares one is refused by `StudySpec`.
    """

    problem: ProblemSpec
    protocol: Signable
    ctx: ComputeContext
    metrics: tuple[str, ...] = field(default=DEFAULT_METRICS)
    rehost: str = REHOST_BLIND
    rehost_overrides: tuple[RehostOverride, ...] = ()

    def __post_init__(self) -> None:
        if not self.metrics:
            raise SpecificationError(
                "metrics is empty, so this evaluation measures nothing; an "
                "expensive rollout whose result is discarded is a "
                "specification error rather than a valid study"
            )
        if len(set(self.metrics)) != len(self.metrics):
            raise SpecificationError(
                f"metrics contains duplicates: {self.metrics}. Each metric is "
                "computed once; a repeat is a declaration error"
            )
        if self.rehost not in REHOST_MODES:
            raise SpecificationError(
                f"rehost is {self.rehost!r}; the modes are "
                f"{list(REHOST_MODES)} (Annex 01 §2.4.1)"
            )
        pairs = [
            (override.contender, str(override.problem.problem_id))
            for override in self.rehost_overrides
        ]
        if len(set(pairs)) != len(pairs):
            raise SpecificationError(
                "rehost_overrides declares a (contender, plant) pair more "
                "than once; each pair is applied once, so a repeat is a "
                "declaration error"
            )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: protocol, scoring context and metrics.

        `problem` is absent by design (see the module docstring); it reaches
        `MeasurementID` as its own `ProblemID`, so including it here would hash
        one value twice and give it two homes that could disagree.

        Metrics are sorted, so two authors who list the same measurements in a
        different order collide rather than diverge.

        `rehost` signs **only off its default** (§2.4.1): a blind spec's
        signature is byte-identical to one predating the field, so the grammar
        growing orphans no stored measurement. The overrides ride the same
        condition — a blind point never reads them.
        """
        signature: dict[str, Any] = {
            "type": type(self).__name__,
            "protocol": self.protocol.get_signature(),
            "ctx": self.ctx.get_signature(),
            "metrics": sorted(self.metrics),
        }
        if self.rehost != REHOST_BLIND:
            signature["rehost"] = self.rehost
            signature["rehost_overrides"] = [
                override.get_signature()
                for override in sorted(
                    self.rehost_overrides,
                    key=lambda override: (
                        override.contender,
                        str(override.problem.problem_id),
                    ),
                )
            ]
        return signature
