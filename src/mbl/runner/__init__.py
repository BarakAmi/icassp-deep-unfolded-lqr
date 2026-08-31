"""Tier 5 — execution. A study specification, executed into the store.

Stage 2 Phase F2 delivers the minimal serial half of this tier: one study, in
declaration order, with no DAG, no parallelism, no resource classes and no
queue. Those are Stage 3's (Annex 06), and the split is deliberate — parent
§7.2's gate ("train, then evaluate differently, and assert zero training") is a
property of the *reuse path*, so it cannot be asserted against anything until
a real producer exists, and a producer is the one part of Tier 5 that Stage 2
can finish.

Unlike Tier 3, this tier has no import restriction: it is the layer where the
grammar, the store, the experiment machinery and the applications are finally
allowed to meet. `spec/` must never import *this*, which
`tests/architecture/test_boundaries.py` enforces in both directions.
"""

from .producer import (
    Disposition,
    PointOutcome,
    RunReport,
    default_synthesize,
    load_trained_controller,
    require_scorable_context,
    run_study,
    training_batch_spec,
)
from .seeds import ReplicateStreams, derive_replicate_streams

__all__ = [
    "Disposition",
    "PointOutcome",
    "ReplicateStreams",
    "RunReport",
    "default_synthesize",
    "derive_replicate_streams",
    "load_trained_controller",
    "require_scorable_context",
    "run_study",
    "training_batch_spec",
]
