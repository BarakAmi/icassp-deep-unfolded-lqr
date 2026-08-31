"""Where things sit under the store root (Annex 02 §2).

Separated from `content_store` so that path arithmetic costs nothing to import.
Reading the store -- listing models, sizing a problem, finding an abandoned
resume artifact -- needs these names and no tensor library; only opening a
record needs PyTorch, pandas and safetensors. Keeping the layout here is what
lets the `mbl` command line answer a query in milliseconds.

`content_store` imports these constants rather than restating them, so there is
one definition of the layout and no way for the two to drift.
"""

from __future__ import annotations

#: Subdirectory holding model records.
MODELS_DIR = "models"

#: Subdirectory holding measurement records.
MEASUREMENTS_DIR = "measurements"

#: Subdirectory holding study records.
STUDIES_DIR = "studies"

#: Where a record's members are assembled before publication.
STAGING_DIR = ".staging"

#: Per-member SHA-256 digests, written last inside the staging directory. What
#: makes `verify` able to detect corruption and `put` able to tell a genuine
#: content conflict from a race.
MANIFEST = "manifest.json"

#: A resume artifact for an in-flight training, never a result (Annex 02 §7.2).
#: Invisible to every consumer: a directory holding only this is not a record.
PARTIAL = ".partial"

# -- payload members (Annex 02 §2.2) ----------------------------------------
#
# Content a record carries about how it came to be, rather than about what it
# is. None of them takes part in an identifier, and -- structurally, not as a
# matter of care -- none of them may ever join a store's `required` tuple: a
# record published before a payload existed would then read as *absent*, and
# absent means recompute. `tests/store/test_payloads_are_optional.py` is the
# gate.

#: Per-epoch training history: one row per epoch.
TRAINING_HISTORY = "training.parquet"

#: The offline phase's captured log, plain UTF-8 text.
SYNTHESIS_LOG = "synthesis.log"

#: Per-trajectory evaluation detail: one row per evaluation trajectory.
SAMPLES = "samples.parquet"

#: Per-batch evaluation detail: one row per evaluation batch.
EVALUATION_TRACE = "evaluation.parquet"

#: The online phase's captured log, plain UTF-8 text.
EVALUATION_LOG = "evaluation.log"

#: The derived index, rebuildable from the trees above.
INDEX_FILE = "index.sqlite"
