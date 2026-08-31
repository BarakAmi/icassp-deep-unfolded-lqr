from datetime import datetime

import numpy as np
import pandas as pd

from mbl.persistence.local_tracker import LocalExperimentTracker
from mbl.persistence.run_reader import (
    RunSummary,
    discover_runs,
    list_run_artifacts,
    load_artifact,
    load_run_metadata,
)


def _fixed_clock(timestamp: datetime):
    return lambda: timestamp


def test_discover_runs_returns_empty_list_for_missing_root(tmp_path) -> None:
    assert discover_runs(tmp_path / "does_not_exist") == []


def test_discover_runs_ignores_non_run_directories(tmp_path) -> None:
    (tmp_path / "not_a_run").mkdir()
    (tmp_path / "stray_file.txt").write_text("noise")

    assert discover_runs(tmp_path) == []


def test_discover_runs_lists_runs_most_recent_first(tmp_path) -> None:
    LocalExperimentTracker(
        tmp_path, "first", clock=_fixed_clock(datetime(2026, 1, 1, 10, 0, 0))
    )
    LocalExperimentTracker(
        tmp_path, "second", clock=_fixed_clock(datetime(2026, 1, 2, 10, 0, 0))
    )

    runs = discover_runs(tmp_path)

    assert [r.run_id for r in runs] == [
        "run_20260102_100000_second",
        "run_20260101_100000_first",
    ]
    assert all(isinstance(r, RunSummary) for r in runs)
    assert runs[0].artifacts_dir == runs[0].run_dir / "artifacts"


def test_load_run_metadata_reads_params_and_metrics(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "exp", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    tracker.log_params({"horizon": 200, "learning_rate": 0.01})
    tracker.log_metrics({"final_cost": 3.14})

    metadata = load_run_metadata(tracker.run_dir)

    assert metadata["params"] == {"horizon": 200, "learning_rate": 0.01}
    assert metadata["metrics"] == {"final_cost": 3.14}


def test_list_run_artifacts_returns_empty_list_when_no_artifacts_dir(tmp_path) -> None:
    empty_run_dir = tmp_path / "run_without_artifacts"
    empty_run_dir.mkdir()

    assert list_run_artifacts(empty_run_dir) == []


def test_list_run_artifacts_and_load_artifact_round_trip(tmp_path) -> None:
    tracker = LocalExperimentTracker(
        tmp_path, "exp", clock=_fixed_clock(datetime(2026, 1, 1))
    )
    states = np.random.default_rng(0).normal(size=(50, 201, 4))
    costs = pd.DataFrame({"epoch": range(20), "loss": np.linspace(2.0, 0.2, 20)})

    tracker.save_artifact("states", states)
    tracker.save_artifact("cost_history", costs)

    artifacts = list_run_artifacts(tracker.run_dir)
    assert {p.name for p in artifacts} == {"states.npz", "cost_history.csv"}

    loaded_by_name = {p.stem: load_artifact(p) for p in artifacts}
    assert np.array_equal(loaded_by_name["states"], states)
    pd.testing.assert_frame_equal(loaded_by_name["cost_history"], costs)
