"""Read-side companion to `LocalExperimentTracker`: discovers runs under an
experiments root and loads their metadata/artifacts back, so callers (the
Streamlit dashboard, notebooks, ad-hoc scripts) never need to know the
on-disk run layout themselves."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .artifact_loaders import ArtifactLoaderRegistry, default_artifact_loader_registry


@dataclass(frozen=True)
class RunSummary:
    """Identifies one persisted run without eagerly loading its contents."""

    run_id: str
    run_dir: Path

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"


def discover_runs(root: Path | str) -> list[RunSummary]:
    """List runs under `root`, most recent first.

    Run directory names are `run_YYYYMMDD_HHMMSS_<name>` (see
    `run_naming.generate_run_id`), so a reverse lexicographic sort is already
    a reverse chronological sort -- no timestamp parsing needed.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    return [
        RunSummary(run_id=path.name, run_dir=path)
        for path in sorted(root.iterdir(), reverse=True)
        if path.is_dir() and (path / "metadata.json").exists()
    ]


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Read `metadata.json` (params/metrics/timestamps) for a run."""
    return cast("dict[str, Any]", json.loads((run_dir / "metadata.json").read_text()))


def list_run_artifacts(run_dir: Path) -> list[Path]:
    """List artifact file paths for a run, sorted by name."""
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.is_dir():
        return []
    return sorted(p for p in artifacts_dir.iterdir() if p.is_file())


def load_artifact(path: Path, *, registry: ArtifactLoaderRegistry | None = None) -> Any:
    """Load a single artifact file, dispatching on its suffix."""
    registry = registry or default_artifact_loader_registry()
    return registry.load(path)
