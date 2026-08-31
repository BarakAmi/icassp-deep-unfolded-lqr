import json
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifact_serializers import (
    ArtifactSerializerRegistry,
    default_artifact_serializer_registry,
)
from .run_naming import generate_run_id


class LocalExperimentTracker:
    """Filesystem-backed ExperimentTracker.

    Layout per run, under `root`:
        run_YYYYMMDD_HHMMSS_<name>/
            metadata.json   -- params, metrics, timestamps (tracked by Git)
            artifacts/      -- heavy payloads (ignored by Git)
    """

    def __init__(
        self,
        root: Path | str,
        name: str,
        *,
        clock: Callable[[], datetime] = datetime.now,
        serializer_registry: ArtifactSerializerRegistry | None = None,
    ) -> None:
        self._clock = clock
        self._created_at = self._clock()
        self._serializer_registry = (
            serializer_registry or default_artifact_serializer_registry()
        )

        run_id = generate_run_id(name, timestamp=self._created_at)
        self.run_dir = Path(root) / run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_path = self.run_dir / "metadata.json"
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._write_metadata()

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._params.update(params)
        self._write_metadata()

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._metrics.update(metrics)
        self._write_metadata()

    def save_artifact(self, name: str, obj: Any) -> Path:
        return self._serializer_registry.save(self.artifacts_dir, name, obj)

    def _write_metadata(self) -> None:
        updated_at = self._clock()
        metadata = {
            "params": self._params,
            "metrics": self._metrics,
            "created_at": self._created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "total_duration_s": (updated_at - self._created_at).total_seconds(),
        }
        self._metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
