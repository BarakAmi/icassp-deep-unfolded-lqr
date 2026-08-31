"""Common interface for experiment tracking, decoupling callers (training loops,
workflows) from the concrete storage backend."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    """Tracks one experiment run: lightweight metadata (params/metrics, tracked by
    Git) versus heavy artifacts (arrays, tensors, dataframes, figures, ignored by Git).
    """

    run_dir: Path
    artifacts_dir: Path

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record configuration/hyperparameters, merged into the run's metadata."""
        ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record scalar metrics, merged into the run's metadata."""
        ...

    def save_artifact(self, name: str, obj: Any) -> Path:
        """Persist a heavy payload (array/tensor/dataframe/figure/...) under
        artifacts_dir and return the path actually written."""
        ...
