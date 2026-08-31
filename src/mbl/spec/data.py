"""The distribution a batch of trajectories is drawn from (Stage 2 Phase F1).

**The defect this closes has been live since Phase C.** `TrainingSpec` had no
training distribution at all, so two studies fitting the same contender on
trajectories with different process-noise levels derived the *same* `ModelID` —
and the second would silently be served the first out of the store. Worse, the
legacy cache key this grammar replaces *does* distinguish them, because the
training batch spec reaches it through `evaluation`: the identity split as built
had lost a term the defective thing it replaces still carries.

**A `DataSpec` carries no batch size, and that is a decision.** The count lives
in `BatchPlan`, beside the microbatch it is split into, because the relationship
between those two is the whole of D20. Giving one quantity two homes would be
bad on its own; here it is also incompatible with D15. A tier writing
`training.batch.effective_size` rebuilds the entire `TrainingSpec` through
`replace_at`, so any `__post_init__` guard keeping two homes consistent would
fire on **every tier application** and make the whitelist's most-used path
unusable. Tested, so the rejected design cannot come back by accident.

**Concrete Tier-3 data, not another injected `Signable`.** Every other
collaborator in this tier is injected because naming its type would drag `torch`
or `engine` in behind it. A distribution declaration does not: it is a name, a
seed and a few floats. Being concrete means it can be *validated* at parse time
rather than merely hashed, which is the point of having a grammar at all. The
sampler the `kind` names is still built through the injected `_build`
vocabulary, exactly as a training plan or an evaluation protocol is.

**Which half a tier may write.** `training.batch.effective_size` is effort and
is whitelisted; `training.data.*` is content — the noise level a model was
fitted at is part of what the study claims — and is not. That is the same line
Annex 01 §2.3.2 draws between `evaluation.protocol.batch_spec.batch_size` and
`…batch_spec.process_noise_std`, and it is why that path is whitelisted leaf by
leaf rather than as a subtree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import SpecificationError

#: What a study draws from when it says nothing more specific.
DEFAULT_DATA_KIND = "gaussian"


@dataclass(frozen=True)
class DataSpec:
    """One distribution of trajectories, as data.

    Attributes:
        kind: Which sampler family to draw from — a registry key, exactly as a
            contender's `family` is. The concrete sampler is built by the
            injected bindings, so adding a family never touches this tier.
        seed: The **root** of this study's replicate family, and at replicate
            0 the stream itself. Not the replicate: the producer spawns two
            independent children of `(seed, replicate)`, one seeding the
            trajectories and one the global RNG a family's construction draws
            from, so a replicate redraws both (Annex 01 §2.3.3,
            `runner.seeds.derive_replicate_streams`). This docstring said the
            opposite until Phase G-3 -- that this seed picks the trajectories
            and the training seed picks the weights -- and that reading is
            what let a five-seed study train five identical models.
        params: The family's own parameters — `process_noise_std`,
            `initial_state_std`, and whatever a new family needs. Deliberately
            open, because the closed thing here is the *family*, and its
            schema belongs with the sampler rather than with the grammar.
    """

    kind: str = DEFAULT_DATA_KIND
    seed: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise SpecificationError(
                "a distribution needs a kind; it is the registry key the "
                "sampler is built from, and an unnamed one builds nothing"
            )
        for name, value in self.params.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError) as error:
                # Refused here rather than at the first `model_id`, which is
                # several frames and, in a study, one whole phase away from the
                # declaration that caused it.
                raise SpecificationError(
                    f"distribution parameter {name!r} is not serialisable "
                    f"({error}); a parameter identity is derived from has to "
                    "survive being written down"
                ) from error

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the whole declaration.

        Parameters are sorted, for the reason `metrics` and `gates` are: two
        authors writing the same distribution in a different order must collide
        rather than diverge.
        """
        return {
            "type": type(self).__name__,
            "kind": self.kind,
            "seed": self.seed,
            "params": {key: self.params[key] for key in sorted(self.params)},
        }
