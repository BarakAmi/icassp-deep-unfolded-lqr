"""An `ExperimentTracker` that keeps the named artifact frames in memory.

The gap it closes, in one sentence: `MetricsHistoryCallback` has built the
per-epoch history frame on every trainable run since the v3 refactor, and the
Tier-5 producer binds a `NullExperimentTracker`, so the frame is built, handed
to `save_artifact`, and discarded -- 162 stored model records, zero
`training.parquet` (Annex 02 §2.2).

**Why a whitelist rather than "keep whatever you are given".** The same callback
list hands `save_artifact` the dense state and control trajectories of
`TrajectoryLoggingCallback` -- megabytes per model, and precisely the derived
bulk the store exists to keep out (Annex 02 §2: 53.2 GB of `trajectory_*.npz`
against 1.3 MB of actual models). A tracker that kept everything would put that
back, one epoch at a time, through the front door.

**And why `frames` has no default.** The artifact name belongs to the callback
that writes it, one tier down; a default here would be a second home for it, and
this module would then have to import `engine` to keep the two in step --
`engine.callbacks` already imports `persistence.tracker`, so that import is a
cycle. The caller names what it wants.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from .null_tracker import NullExperimentTracker


class RecordingExperimentTracker(NullExperimentTracker):
    """Collects params/metrics in memory like its parent, and additionally keeps
    the artifacts named in `frames` when they are `DataFrame`s.

    Attributes:
        frames: The artifact names worth keeping.
        recorded: The kept frames, by artifact name.
    """

    def __init__(self, frames: Collection[str]) -> None:
        """
        Args:
            frames: Artifact names to retain (e.g.
                `engine.callbacks.METRICS_HISTORY`). Anything else is
                discarded, as is anything under these names that is not a
                `DataFrame` -- a caller that renamed its artifact gets no frame
                rather than an object of the wrong type filed under the right
                name.
        """
        super().__init__()
        self.frames = frozenset(frames)
        self.recorded: dict[str, pd.DataFrame] = {}

    def save_artifact(self, name: str, obj: Any) -> Path:
        """Keep `obj` if it is a wanted frame; discard it otherwise.

        Returns:
            The placeholder `artifacts_dir` -- nothing is written to disk here,
            exactly as for the null tracker. The store publishes the frame as a
            member of the record, atomically, and a file written beside the
            record would be a second home for it.
        """
        if name in self.frames and isinstance(obj, pd.DataFrame):
            self.recorded[name] = obj
        return super().save_artifact(name, obj)

    def frame(self, name: str) -> pd.DataFrame | None:
        """The recorded frame, or `None` where nothing was recorded.

        An **empty** frame is also reported as `None`: a run that produced no
        epoch has no history, and a zero-row parquet reads back as a frame with
        no columns, which would make every consumer branch on shape instead of
        on presence.
        """
        recorded = self.recorded.get(name)
        return None if recorded is None or recorded.empty else recorded
