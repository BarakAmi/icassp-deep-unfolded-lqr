"""The tracker that keeps the training history and nothing else.

Annex 02 §2.2. The property under test is as much what it *discards* as what it
keeps: the same callback list that produces the per-epoch frame also hands
`save_artifact` the dense state and control trajectories, which are megabytes
per model and are the derived bulk the store exists to keep out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbl.engine.callbacks import METRICS_HISTORY
from mbl.persistence import RecordingExperimentTracker


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {"epoch": [0, 1], "wall_time_s": [0.5, 0.4], "loss": [2.0, 1.0]}
    )


def test_a_named_frame_is_kept() -> None:
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    tracker.save_artifact(METRICS_HISTORY, _history())

    kept = tracker.frame(METRICS_HISTORY)
    assert kept is not None
    pd.testing.assert_frame_equal(kept, _history())


def test_an_unnamed_artifact_is_discarded() -> None:
    """The trajectory arrays, by name. Keeping "whatever it is given" is how
    53.2 GB of `trajectory_*.npz` came back through the front door."""
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    tracker.save_artifact("trajectory_states", np.zeros((8, 64, 4)))
    tracker.save_artifact("trajectory_controls", np.zeros((8, 64, 2)))

    assert tracker.recorded == {}
    assert tracker.frame("trajectory_states") is None
    # ... and the parent's behaviour is intact: the names are still recorded.
    assert tracker.artifact_names == ["trajectory_states", "trajectory_controls"]


def test_a_named_artifact_that_is_not_a_frame_is_discarded() -> None:
    """A caller that reused the name for something else gets nothing, rather
    than an object of the wrong type filed under the right name."""
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    tracker.save_artifact(METRICS_HISTORY, {"epoch": [0, 1]})

    assert tracker.frame(METRICS_HISTORY) is None


def test_an_empty_frame_reads_as_absent() -> None:
    """A run that produced no epoch has no history. A zero-row parquet reads
    back as a frame with no columns, which would make every consumer branch on
    shape instead of on presence."""
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    tracker.save_artifact(METRICS_HISTORY, pd.DataFrame())

    assert tracker.frame(METRICS_HISTORY) is None


def test_nothing_is_written_to_disk(tmp_path) -> None:
    """The store publishes the frame as a member of the record, atomically. A
    file written beside the record would be a second home for it."""
    tracker = RecordingExperimentTracker(frames=(METRICS_HISTORY,))
    tracker.save_artifact(METRICS_HISTORY, _history())

    assert list(tmp_path.iterdir()) == []
    assert not tracker.artifacts_dir.is_absolute()


def test_params_and_metrics_still_behave_as_the_null_tracker() -> None:
    tracker = RecordingExperimentTracker(frames=())
    tracker.log_params({"lr": 0.05})
    tracker.log_metrics({"final_loss": 1.5})

    assert tracker.params == {"lr": 0.05}
    assert tracker.metrics == {"final_loss": 1.5}
