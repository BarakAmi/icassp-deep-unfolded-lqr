"""How a learned contender is fitted (Stage 2 Phase C, decision **D20**).

D20 draws a line through the middle of the training declaration, and the whole
of this module is that line.

**Effective batch size is scientific content.** It says how many trajectories
the gradient is averaged over, which changes the estimator and therefore the
model. It participates in `ModelID`.

**The microbatch is not.** At n ~ 100-200 the training autograd graph
(batch x horizon x J x n) exceeds 24 GB, so the batch must be processed in
chunks with the gradients accumulated. Which chunk size fits is a property of
the machine that happened to run the job, not of the experiment: it is chosen
by the runner from available memory, recorded in provenance, and **excluded**
from identity. The alternative was measured against D15 and rejected — putting
the chunk size in identity would force every execution tier to be re-tuned per
problem size, which destroys the tier as a reusable abstraction and breaks D15's
rule that a tier must never change scientific content.

The accepted cost is stated rather than hidden: two runs of one `ModelID` may
produce weights differing at float-summation tolerance (~1e-7), because summing
in a different order is what accumulation does. Content addressing *notices* —
a second `put` of differing bytes under an existing id raises
`ContentConflictError` rather than overwriting — so the reuse path must check
`exists()` before training rather than publish-and-reconcile after.

**Precision is content, and the COCP family requires float64.** Its solver is
ill-conditioned at float32, so the pairing is refused rather than run. Precision
already lives in `ComputeContext` and therefore already participates in
`ModelID`: a float32 model and a float64 model are different models, correctly.

**One fitting procedure, one home (G-2).** A study declares `plan` and every
trainable recipe declares its own, and only the contender's was ever executed —
`TrainableRecipe.build_engine` trains for that one. Both were signed anyway, so
two studies differing only in the study-level plan derived different `ModelID`s
for models identical in every respect, including their epoch count. `plan` is
now the study's **default**, materialised into every contender that accepts one
when the `StudySpec` is built, and `get_signature` omits it: identity comes from
the effective plan, once, through the contender (Annex 01 §2.3.4).

**The tier rule this module obeys, and what the plan is held *for*.** The
training plan is held through the `Signable` protocol rather than by importing
`engine.training_plan`, which would pull `torch`, `engine.config` and
`models.unfolded.layerwise` into Tier 3. It is also the only description both
plan types satisfy, since `TrainingPlan` has `epochs` while
`LayerwiseTrainingPlan` has `warmup_epochs_per_layer` and `refinement_epochs`.
What changed with G-2 is *why* the grammar holds it: not to sign it, but to hand
it out. `Signable` is kept because it remains the narrowest structural
description of an object this tier may not construct — and because the object
still has to be signable, inside the contender it lands in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..core.runtime import ComputeContext, Precision
from ..core.utils.signing import Signable
from .data import DataSpec
from .errors import SpecificationError
from .problem import ProblemSpec

#: Families whose numerics require a specific working precision. COCP's
#: differentiable cone solver loses its correctness canary at float32, and the
#: failure is silent — wrong gradients, not an exception — so the pairing is
#: refused at specification time instead. Checked by
#: `require_supported_precision`, which the composing surfaces call; a family
#: absent from this table accepts any precision.
PRECISION_REQUIREMENTS: dict[str, Precision] = {"cocp": Precision.FLOAT64}


def require_supported_precision(family: str, precision: Precision) -> None:
    """Refuse a family/precision pairing its numerics cannot support.

    A free function rather than a `TrainingSpec` validator, because a
    `TrainingSpec` has no contender and must not acquire one: a study declares
    one training plan for many contenders, so a family field would make the
    type unshareable. The check therefore belongs where the two meet — the
    identity derivation and the study loader — and each of those needs its own
    test, since "fails at parse time" is a property of the composing surface.

    Args:
        family: The contender family's registry name.
        precision: The working precision declared by the training context.

    Raises:
        SpecificationError: If `family` declares a requirement and `precision`
            is not it. The message names all three, because the caller has to
            know which to change.
    """
    required = PRECISION_REQUIREMENTS.get(family)
    if required is not None and required is not precision:
        raise SpecificationError(
            f"contender family {family!r} requires {required.value} but the "
            f"training context declares {precision.value}; the pairing is "
            "refused here rather than producing silently wrong gradients"
        )


@dataclass(frozen=True)
class BatchPlan:
    """How a batch is *executed*, as opposed to how large it is.

    Attributes:
        effective_size: Trajectories the gradient is averaged over. Scientific
            content; participates in `ModelID`.
        microbatch: Trajectories per forward/backward pass, or `None` for "the
            runner chooses". A resource decision: recorded in provenance,
            excluded from identity (D20).
    """

    effective_size: int
    microbatch: int | None = None

    def __post_init__(self) -> None:
        if self.effective_size < 1:
            raise SpecificationError(
                f"effective_size must be positive, got {self.effective_size}"
            )
        if self.microbatch is not None and self.microbatch < 1:
            raise SpecificationError(
                f"microbatch must be positive or None, got {self.microbatch}; "
                "None means the runner chooses from available memory"
            )

    @property
    def accumulation_steps(self) -> int:
        """Forward/backward passes one effective batch costs.

        `ceil`, not `floor`: a plan that dropped the remainder would train on
        fewer trajectories than the specification declares, which is a silent
        change to the estimator rather than a rounding detail. A microbatch
        larger than the batch is one whole pass, not a fraction of one.
        """
        if self.microbatch is None:
            return 1
        return max(1, math.ceil(self.effective_size / self.microbatch))

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the scientific content only.

        `microbatch` and the `accumulation_steps` it implies are deliberately
        absent — not merely equal-hashing, absent. This one omission is the
        whole of D20's resource half, which is why the acceptance suite checks
        it structurally as well as behaviourally: a constant microbatch in the
        tree would shift every derived identifier uniformly and no behavioural
        test would notice.
        """
        return {"type": type(self).__name__, "effective_size": self.effective_size}


@dataclass(frozen=True)
class TrainingSpec:
    """One fitting procedure, shared by every learned contender in a study.

    Attributes:
        data: The distribution the training trajectories are drawn from. Its
            own object, distinct from the evaluation's, which is the structural
            fix for parent §2.2's defect -- and it carries no batch size, for
            the reason in `spec/data.py`.
        plan: The **default** optimiser/schedule declaration, held structurally
            (see the module docstring). `engine.TrainingPlan` and
            `engine.LayerwiseTrainingPlan` both satisfy it. Materialised into
            every contender that accepts one when the study is built, and
            therefore excluded from `get_signature`: a contender that declares
            its own keeps it, and identity comes from whichever plan ends up in
            the contender.
        batch: The batch declaration, whose D20 split is the point of this
            module.
        ctx: Backend, device and precision. A **field**, because the trainer
            and `derive_model_id` both need it — but deliberately not part of
            `get_signature`, since `model_id` takes it on its own axis and
            hashing it twice would give one value two homes.
        seeds: The seed collection this study declares. Excluded from
            `ModelID` by `TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID`: a study
            declaring five seeds is five models, and adding a sixth must train
            one, not six.
    """

    data: DataSpec
    plan: Signable
    batch: BatchPlan
    ctx: ComputeContext
    seeds: tuple[int, ...] = field(default=(0,))
    problem: ProblemSpec | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise SpecificationError(
                "seeds is empty, so this specification trains nothing; declare "
                "at least one seed"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise SpecificationError(
                f"seeds contains duplicates: {self.seeds}. Each seed is one "
                "model, so a repeat would publish identical bytes twice and "
                "surface as a store conflict far from this declaration"
            )

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: distribution, batch and the declared seeds.

        **`plan` is absent, and not merely equal-hashing — absent.** By the time
        an identifier is derived it has been materialised into every contender
        that accepts one, so signing it here would hash the effective plan twice
        for those contenders and a template that governs nothing for the rest.
        The observed consequence of the latter was two identifiers for one
        model (Annex 01 §2.3.4).

        `ctx` is absent by design too (see the class docstring). `seeds` is
        emitted under exactly the name `TRAINING_KEYS_EXCLUDED_FROM_MODEL_ID`
        strips, which is a contract with Tier 4 rather than a coincidence:
        renaming the key here would silently put the whole collection back into
        every model's identity.
        """
        signature: dict[str, Any] = {
            "type": type(self).__name__,
            "data": self.data.get_signature(),
            "batch": self.batch.get_signature(),
            "seeds": list(self.seeds),
        }
        # The training WORLD (Annex 01 §2.3.5, D25) signs **only when
        # declared**, by the plant's own identity: absent, the signature is
        # byte-identical to one predating the field, so the grammar growing
        # moves no stored ModelID. Declared, every model fitted in that world
        # is a different model, which is the point.
        if self.problem is not None:
            signature["problem"] = str(self.problem.problem_id)
        return signature
