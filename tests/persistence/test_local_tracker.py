import json
from datetime import datetime

import numpy as np

from mbl.persistence.local_tracker import LocalExperimentTracker
from mbl.persistence.tracker import ExperimentTracker


def _fixed_clock(timestamp: datetime):
    return lambda: timestamp


def test_tracker_satisfies_the_experiment_tracker_protocol(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "demo", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    assert isinstance(tracker, ExperimentTracker)


def test_tracker_creates_expected_directory_structure(tmp_path) -> None:
    timestamp = datetime(2026, 7, 3, 10, 30, 0)
    tracker = LocalExperimentTracker(
        tmp_path, "covert_lqr", clock=_fixed_clock(timestamp)
    )

    expected_run_dir = tmp_path / "run_20260703_103000_covert_lqr"
    assert tracker.run_dir == expected_run_dir
    assert tracker.artifacts_dir == expected_run_dir / "artifacts"
    assert tracker.run_dir.is_dir()
    assert tracker.artifacts_dir.is_dir()
    assert (tracker.run_dir / "metadata.json").exists()


def test_tracker_writes_params_metrics_and_timestamps_to_metadata(tmp_path) -> None:
    created_at = datetime(2026, 7, 3, 10, 30, 0)
    tracker = LocalExperimentTracker(tmp_path, "exp", clock=_fixed_clock(created_at))

    tracker.log_params({"horizon": 10, "learning_rate": 0.01})
    tracker.log_metrics({"final_cost": 3.14})

    metadata = json.loads((tracker.run_dir / "metadata.json").read_text())
    assert metadata["params"] == {"horizon": 10, "learning_rate": 0.01}
    assert metadata["metrics"] == {"final_cost": 3.14}
    assert metadata["created_at"] == created_at.isoformat()
    assert metadata["updated_at"] == created_at.isoformat()
    assert metadata["total_duration_s"] == 0.0


def test_tracker_writes_total_duration_s_as_elapsed_seconds(tmp_path) -> None:
    created_at = datetime(2026, 7, 3, 10, 30, 0)
    ticks = iter([created_at, created_at, datetime(2026, 7, 3, 10, 30, 5)])
    tracker = LocalExperimentTracker(tmp_path, "exp", clock=lambda: next(ticks))

    tracker.log_metrics({"final_cost": 3.14})  # advances the clock to +5s

    metadata = json.loads((tracker.run_dir / "metadata.json").read_text())
    assert metadata["created_at"] == created_at.isoformat()
    assert metadata["updated_at"] == datetime(2026, 7, 3, 10, 30, 5).isoformat()
    assert metadata["total_duration_s"] == 5.0


def test_tracker_log_params_and_log_metrics_merge_incrementally(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "exp", clock=_fixed_clock(datetime(2026, 1, 1))
    )

    tracker.log_params({"a": 1})
    tracker.log_params({"b": 2})
    tracker.log_metrics({"loss": 1.0})
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.9})

    metadata = json.loads((tracker.run_dir / "metadata.json").read_text())
    assert metadata["params"] == {"a": 1, "b": 2}
    assert metadata["metrics"] == {"loss": 0.5, "accuracy": 0.9}


def test_tracker_save_artifact_writes_under_artifacts_dir(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "exp", clock=_fixed_clock(datetime(2026, 1, 1))
    )

    array = np.array([1, 2, 3])
    out_path = tracker.save_artifact("trajectory", array)

    assert out_path == tracker.artifacts_dir / "trajectory.npz"
    with np.load(out_path) as payload:
        assert np.array_equal(payload["array"], array)
    # Artifacts must never live next to metadata.json, only under artifacts_dir.
    assert not (tracker.run_dir / "trajectory.npz").exists()


def test_two_runs_with_different_names_do_not_collide(tmp_path) -> None:
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    tracker_a = LocalExperimentTracker(tmp_path, "run_a", clock=_fixed_clock(timestamp))
    tracker_b = LocalExperimentTracker(tmp_path, "run_b", clock=_fixed_clock(timestamp))

    assert tracker_a.run_dir != tracker_b.run_dir
    assert tracker_a.run_dir.is_dir()
    assert tracker_b.run_dir.is_dir()
