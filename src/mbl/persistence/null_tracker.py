"""An in-memory, no-persistence `ExperimentTracker`: satisfies the protocol
while writing nothing to disk. Used where the engine chassis is executed
outside a tracked run -- e.g. an engine-backed `Synthesizer` invoked
standalone (Stage S3) -- and by tests that assert on logged values without
a filesystem."""

from pathlib import Path
from collections.abc import Mapping
from typing import Any


class NullExperimentTracker:
    """Collects params/metrics in memory and discards artifacts.

    Attributes:
        run_dir: A placeholder path (never created or written).
        artifacts_dir: A placeholder path (never created or written).
        params: Every mapping passed to `log_params`, merged.
        metrics: Every mapping passed to `log_metrics`, merged.
        artifact_names: The names passed to `save_artifact`, in order.
    """

    def __init__(self) -> None:
        self.run_dir = Path(".")
        self.artifacts_dir = Path(".")
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifact_names: list[str] = []

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Merge `params` into the in-memory record."""
        self.params.update(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Merge `metrics` into the in-memory record."""
        self.metrics.update(metrics)

    def save_artifact(self, name: str, obj: Any) -> Path:
        """Record `name` and discard `obj`.

        Returns:
            The placeholder `artifacts_dir` (nothing is written).
        """
        self.artifact_names.append(name)
        return self.artifacts_dir
