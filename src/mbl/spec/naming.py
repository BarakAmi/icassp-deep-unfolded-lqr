"""A stored model's human-legible name, projected from its specification (F3).

A content hash is correct and unusable: nobody recognises `74ad4e8f36bfbb15`.
Annex 02 §3 gives every record a deterministic name derived from what it is,
with the short digest appended — this one is NB04's, copied off a real store
rather than composed here:

    box_lqr-n4m2-N50-u0.5-s0/unfolded-learned_step_size-J1/adam-lr1e-2-ep2-b128-seed0#e592e3bd

**Two deliberate deviations from the annex's illustration**, which renders
`boxlqr-…/unfolded-aP-J10`. The underscores survive because `store.naming`'s
slug treats `_` as safe, and the family label is the author's own spelling of
their study id rather than a transformation of it. And the kind is slugged as
declared rather than abbreviated to `aP`, because producing `aP` from
`learned_step_size_and_matrix` needs a per-family abbreviation table — a second
source of truth about what a family is, of exactly the kind Phase E3 deleted.

Tier 4 has held `ProblemName`/`ContenderName`/`TrainingName` since the
store was built, with nothing to fill them. The descriptors are grammar-level
concepts, so the projection could only be written once the grammar existed.
This module is that projection, and it is the last thing Stage 2 owed.

**The name is a label, never an identity, and that is why this is not in
`identity.py`.** The plan's §9.4 filed it beside `derive_model_id`; the
correction is recorded in §F3. Nothing here participates in any identifier,
which is precisely what makes the omissions below safe: a name is lossy by
construction, ordered by discriminating power, and the digest is what
guarantees uniqueness. Two models the name cannot separate share a name and
differ in the digest.

**Reading a family's own keys is permitted here, and was refused three times
elsewhere.** `kind` and `num_iterations` are looked up in the resolved recipe's
signature, which is a per-family assumption of exactly the kind Phase E3 deleted
from the loader, Phase G-2 refused for the training plan and Phase G-3 refused
for the seed. The difference is what the lookup decides. There it fed an
identifier or a build, so a missing key produced a wrong model or a silent
no-op; here a missing key costs one segment of *description*. The rule is not
"never read a family's keys" — it is "never let a family's keys decide something
that must be right".

**What D19 costs, and the one place it shows.** `ProblemSpec` holds frozen
matrices and an optional provenance record and no name at all, so the family
label cannot come from the problem. It comes from the study that names it —
`StudySpec.id`'s first path segment — and the problem's own seed comes from
provenance when there is any, and is omitted when there is not, rather than
inventing one for a hand-specified problem.
"""

from __future__ import annotations

from typing import Any

from ..store.naming import ContenderName, ProblemName, TrainingName, semantic_name
from .study import StudyPoint, StudySpec

#: Recipe-configuration keys the contender segment renders when a family
#: happens to have them. Absent keys are omitted; see the module docstring for
#: why that is safe here and is not elsewhere.
KIND_KEY, DEPTH_KEY = "kind", "num_iterations"

#: Where the problem's own seed lives when the problem was generated rather
#: than hand-specified. Optional by D19.
PROBLEM_SEED_KEY = "seed"

#: How much of the identifier the name carries. Long enough to separate the
#: models one study produces, short enough to type; the full identifier is
#: always one `mbl models show` away.
DIGEST_WIDTH = 8


def derive_semantic_name(point: StudyPoint, study: StudySpec) -> str:
    """The name a stored model is filed under, beside its identifier.

    Args:
        point: The materialised point the model belongs to. Supplies the
            problem, the resolved contender, the effective training plan and
            the replicate.
        study: The study that names the problem. Needed for one field only —
            the problem family label, which D19 removed from `ProblemSpec` and
            which is therefore the study's to supply.

    Returns:
        `problem/contender/training#digest`, per Annex 02 §3.
    """
    return semantic_name(
        problem=_problem_name(point, study),
        contender=_contender_name(point),
        training=_training_name(point),
        digest=str(point.model_id)[:DIGEST_WIDTH],
    )


def _problem_name(point: StudyPoint, study: StudySpec) -> ProblemName:
    """The problem segment: the study's label plus the problem's own shape."""
    problem = point.problem
    return ProblemName(
        family=study.id.split("/", 1)[0],
        state_dim=problem.state_dim,
        control_dim=problem.control_dim,
        horizon=problem.horizon,
        u_max=problem.data.control_bound,
        seed=_problem_seed(point),
    )


def _problem_seed(point: StudyPoint) -> int | None:
    """The seed the problem was generated from, when it was generated at all.

    `None` for a hand-specified problem, which renders as an omitted field
    rather than as `s0` — a problem nobody seeded and a problem seeded zero are
    different things, and only one of them can be regenerated.
    """
    provenance = point.problem.provenance
    if provenance is None:
        return None
    seed = dict(provenance.params).get(PROBLEM_SEED_KEY)
    return int(seed) if isinstance(seed, int) else None


def _contender_name(point: StudyPoint) -> ContenderName:
    """The contender segment, read from the **resolved** recipe.

    Resolved rather than declared, so a default the study never spelled out
    still appears — the same reason `ContenderSpec.get_signature` resolves
    before signing.
    """
    signature = point.contender.get_signature()
    return ContenderName(
        family=point.contender.family,
        kind=_slug_value(signature.get(KIND_KEY)),
        depth=_as_int(signature.get(DEPTH_KEY)),
    )


def _training_name(point: StudyPoint) -> TrainingName:
    """The training segment: the plan that **executes**, plus the replicate.

    The plan comes from the contender, not from `TrainingSpec.plan`: G-2 made
    the study-level plan a default materialised into each contender, and a name
    reporting the template would say what the model was *not* fitted with.

    A layer-wise schedule has `warmup_epochs_per_layer` and `refinement_epochs`
    and no `epochs` at all, so the epoch field is omitted rather than summed —
    a sum would be a number the specification never states. An analytic
    contender has no plan and still has a seed, so its segment is the seed
    alone rather than the empty placeholder: two replicates of an analytic
    contender are two models and must be two names.
    """
    plan = _executed_plan(point)
    optimizer = plan.get("optimizer") if plan else None
    return TrainingName(
        optimizer=_slug_value(optimizer.get("name")) if optimizer else None,
        learning_rate=_as_float(optimizer.get("learning_rate")) if optimizer else None,
        epochs=_as_int(plan.get("epochs")) if plan else None,
        batch_size=point.training.batch.effective_size if plan else None,
        seed=point.seed,
    )


def _executed_plan(point: StudyPoint) -> dict[str, Any]:
    """The signature of whichever plan this contender is actually trained by.

    Both spellings are read — `plan` for an end-to-end family, `schedule` for a
    layer-wise one — because the two are a different shape rather than a
    narrower one, and a contender with neither does not train at all.
    """
    signature = point.contender.get_signature()
    for key in ("plan", "schedule"):
        declared = signature.get(key)
        if isinstance(declared, dict):
            return declared
    return {}


def _slug_value(value: Any) -> str | None:
    """A signature value as a name fragment, or `None` when there is none.

    `str(value)` rather than `value`: an enum member reaches a signature as
    itself, and every one in this tree is a `StrEnum` whose `str` is its value.
    """
    return None if value is None else str(value)


def _as_int(value: Any) -> int | None:
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    )


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
