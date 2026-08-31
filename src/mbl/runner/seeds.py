"""What a training replicate is, as executable arithmetic (Stage 2 Phase G-3).

**The defect this closes.** `TrainingSpec.seeds` was signed into `ModelID` and
consumed by nothing else. `training.data.seed` picked the trajectories, a
recipe's own `random_init_seed`/`init_seed` picked the initial iterate — but
only under `init_method = "randomized"`, which no study declares — and the
replicate index itself reached the identifier and stopped. A five-seed study
therefore derived five identifiers for five *byte-identical* models, and the
first thing that would have surfaced it is a zero-width confidence interval in
a figure.

**Why this is not a per-family seed field.** The obvious repair — every recipe
declares which of its fields is "the seed", and the producer writes the
replicate into it — was measured against the registry and rejected (plan §G-3).
Four of the eight registered families declare no such field, `cocp` among them,
and `cocp` trains: a replicate that only re-initialises weights is a no-op for
half the registry. It also generalises badly, since the next family added would
have to remember to declare one. What a replicate must reach is the *streams*,
and every source of randomness in this project draws from exactly two of them.

**What this module is, and is not.** It is Tier 5. Nothing here participates in
identity: `derive_model_id` already takes the replicate on its own axis and
`training.data.seed` is already inside `canon(TrainingSpec)`, so both terms are
signed before this function is called. What it decides is what the producer
*does* with two values that are already written down — which is execution, and
putting it in the grammar would invite the belief that a stream is part of what
a model *is*.

**The residual, because one property here is not free.** Replicate 0's data
child *is* the declared seed, so a single-replicate study draws exactly the
stream its document names and the tracked studies keep their numbers. The cost
is that at replicate 0 the two streams are a declared integer and a spawned
child rather than two children of one root, so their mutual independence rests
on the child being a well-mixed 63-bit value rather than on `SeedSequence`'s
own guarantee. That is strictly better than what it replaces — today both
streams are literally the integer `0` — and the alternative was to move every
study's replicate-0 trajectories for a property that only bites where a family
draws its construction from the global stream. Recorded here so that choosing
differently later is a decision rather than a discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spec.errors import SpecificationError

#: The spawn key of each stream, under the replicate's own key. Distinct
#: integers rather than one value reused: `SeedSequence` owes independence to
#: distinct keys and to nothing else.
DATA_CHILD, WEIGHTS_CHILD = 0, 1

#: Width of a derived stream. Measured, rather than assumed: both
#: `torch.Generator.manual_seed` and `numpy.random.default_rng` accept the full
#: unsigned 64-bit range, so this bit is given up deliberately and not because
#: a generator demands it. A 63-bit value is the largest that fits the *signed*
#: 64-bit integer everything downstream of a generator can hold — SQLite's
#: `INTEGER`, a JSON number read by anything that is not Python — and a stream
#: is a number that gets logged and written down.
STREAM_BITS = 63


@dataclass(frozen=True)
class ReplicateStreams:
    """The two independent streams one replicate of a study draws from.

    Attributes:
        data: Seeds the training batch sampler — the trajectories the model is
            fitted on.
        weights: Seeds the **global** torch RNG, which is where `nn.Module`
            initialisation and any future family's construction randomness
            draw from. Deliberately not a per-family field: a family added
            later inherits correct replicate behaviour while declaring
            nothing.
    """

    data: int
    weights: int


def derive_replicate_streams(data_seed: int, replicate: int) -> ReplicateStreams:
    """The streams one replicate of one study draws from.

    **The parameters are the whole specification of the fairness law.** There
    is no parameter through which a contender, a family or a recipe could
    arrive, so every contender at one replicate necessarily draws identical
    batches — the property a comparison between contenders means nothing
    without, and one that a behavioural test could only check for the
    contenders some fixture happened to declare.

    Args:
        data_seed: `TrainingSpec.data.seed`, the root of this study's replicate
            family. At replicate 0 it is also the data stream itself.
        replicate: The declared seed of the point being executed — one member
            of `TrainingSpec.seeds`, not its index, since a study may name its
            replicates explicitly.

    Returns:
        The `ReplicateStreams` for this point.

    Raises:
        SpecificationError: If either value is negative. `SeedSequence` and
            `numpy.random.default_rng` both refuse one, several frames inside a
            sampler and naming neither the document key nor which of the two
            seeds was wrong.
    """
    if data_seed < 0:
        raise SpecificationError(
            f"training.data.seed is {data_seed}; a stream root must be "
            "non-negative, since it is the entropy every replicate of this "
            "study is spawned from"
        )
    if replicate < 0:
        raise SpecificationError(
            f"a training seed is {replicate}; a replicate must be non-negative, "
            "since it is the spawn key that makes replicates independent"
        )
    if replicate == 0:
        # The declared stream, untouched: a single-replicate study draws
        # exactly what its document says, which is the common case and the one
        # that has to stay legible. See the module docstring's residual.
        return ReplicateStreams(
            data=data_seed, weights=_child(data_seed, replicate, WEIGHTS_CHILD)
        )
    return ReplicateStreams(
        data=_child(data_seed, replicate, DATA_CHILD),
        weights=_child(data_seed, replicate, WEIGHTS_CHILD),
    )


def _child(entropy: int, replicate: int, stream: int) -> int:
    """One grandchild of `entropy`, as an integer both generators accept.

    `SeedSequence(entropy=e, spawn_key=(r, s))` is exactly the `s`-th child of
    the `r`-th child of `e`, so the tree is addressable without holding the
    intermediate objects — which matters because a producer derives these one
    point at a time, in any order, and must not depend on how many it has
    derived already.
    """
    state = np.random.SeedSequence(entropy=entropy, spawn_key=(replicate, stream))
    return int(state.generate_state(1, dtype=np.uint64)[0] >> (64 - STREAM_BITS))
