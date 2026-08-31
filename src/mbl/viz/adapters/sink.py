"""The dual-format `FigureSink` (REFACTOR_PLAN v3, T4.e -- closes G2).

**The "No Figure Unbacked" law:** no figure persists without its data. Every
`FigureSink.save` emits two coupled artifacts:

1. **The publication face** -- the high-contrast vector graphic (``.pdf``),
   exactly as the project's style discipline produces.
2. **The data sidecar** -- a machine-readable file (``.json`` for light
   payloads, ``.npz`` for dense arrays, chosen by payload size; never
   pickle) carrying the EXACT plotted series plus the labeling/configuration
   needed to re-render: the serialized renderer input, the renderer's
   registered name, and a sidecar schema version.

Because every renderer in `viz.plots`/`viz.landscape` is a pure function of
frozen Result data (T4.b), the sidecar IS the serialized renderer input --
no second "figure data model" exists, so figure and sidecar cannot drift.

**Restyle without recompute:** `re_render` consumes a sidecar and reproduces
the figure -- with any cosmetic overrides desired -- without importing
solvers, touching the experiment cache, or re-running anything. Renderers
announce themselves via `register_sidecar_renderer` (see
`viz.adapters.notebook`, which registers every notebook-facing renderer at
import time).

Animations get the same backing: `FigureSink.save_animation` persists the
GIF/MP4 together with a sidecar of the animation's serializable inputs, so
the frames' underlying data outlives the encoded raster.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..style import save_raster_figure, save_vector_figure

#: Bumped whenever the sidecar layout changes shape; `load_sidecar` refuses
#: versions it does not understand rather than mis-reading them.
SIDECAR_SCHEMA_VERSION = 1

#: Payloads whose arrays total at most this many elements are written as
#: human-readable ``.json``; anything denser becomes compressed ``.npz``.
_JSON_ELEMENT_LIMIT = 512

#: Key separator used to flatten nested payload mappings into flat ``.npz``
#: entry names. Payload keys must therefore never contain it.
_KEY_SEPARATOR = "/"

#: The reserved ``.npz`` entry holding the sidecar's JSON header (renderer
#: name, config, non-array data leaves).
_NPZ_HEADER_KEY = "__sidecar__"


@dataclass(frozen=True)
class SavedFigure:
    """The coupled artifact pair one `FigureSink.save` produces."""

    figure_path: Path
    sidecar_path: Path


@dataclass(frozen=True)
class FigureSidecar:
    """A loaded sidecar: everything needed to re-render its figure.

    Attributes:
        schema_version: the `SIDECAR_SCHEMA_VERSION` it was written under.
        renderer: the registered renderer name `re_render` dispatches on.
        data: the exact plotted series (nested mapping of arrays/scalars) --
            the serialized renderer input.
        config: the labeling/styling kwargs (titles, axis labels, layout
            keys) the renderer was called with.
    """

    schema_version: int
    renderer: str
    data: dict[str, Any]
    config: dict[str, Any]


#: renderer name -> ``(data, config) -> Figure``. Populated via
#: `register_sidecar_renderer`; consulted by `re_render`.
_SIDECAR_RENDERERS: dict[
    str, Callable[[Mapping[str, Any], Mapping[str, Any]], Figure]
] = {}


def register_sidecar_renderer(
    name: str, render: Callable[[Mapping[str, Any], Mapping[str, Any]], Figure]
) -> None:
    """Register the pure re-render function for sidecars written under
    `name`: it receives the sidecar's `data` and `config` mappings and must
    return a Figure without touching solvers or the filesystem. Idempotent
    (re-registration overwrites), so import-time registration is safe."""
    _SIDECAR_RENDERERS[name] = render


def register_kwarg_renderer(name: str, plot_fn: Callable[..., Figure]) -> None:
    """Convenience registration for renderers whose signature is plain
    keyword arguments: the sidecar's `data` and `config` entries are splatted
    directly into `plot_fn`."""

    def render(data: Mapping[str, Any], config: Mapping[str, Any]) -> Figure:
        return plot_fn(**data, **config)

    register_sidecar_renderer(name, render)


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into ``a/b/c`` keyed leaves (arrays, scalars,
    strings, or lists of scalars)."""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"Sidecar payload keys must be str, got {key!r}.")
        if _KEY_SEPARATOR in key:
            raise ValueError(
                f"Sidecar payload key {key!r} contains the reserved "
                f"separator {_KEY_SEPARATOR!r}."
            )
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{path}{_KEY_SEPARATOR}"))
        else:
            flat[path] = value
    return flat


def _unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of `_flatten`."""
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        parts = path.split(_KEY_SEPARATOR)
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def _leaf_to_jsonable(value: Any) -> Any:
    """Encode one flattened leaf for the JSON face of a sidecar. Arrays are
    tagged with their dtype so `_leaf_from_jsonable` restores them exactly."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_leaf_to_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        # Mappings nested inside a list leaf (e.g. a list of benchmark
        # records) -- top-level/nested dict structure is flattened by
        # `_flatten` before ever reaching here.
        return {k: _leaf_to_jsonable(v) for k, v in value.items()}
    raise TypeError(
        f"Sidecar payloads accept arrays, scalars, strings, and lists "
        f"thereof (T3.i: structured formats only, never pickle); got "
        f"{type(value).__name__}."
    )


def _leaf_from_jsonable(value: Any) -> Any:
    """Inverse of `_leaf_to_jsonable`."""
    if isinstance(value, dict) and "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    if isinstance(value, dict):
        return {k: _leaf_from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_leaf_from_jsonable(v) for v in value]
    return value


def _normalize_leaves(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce torch tensors / array-likes to numpy up front so the format
    choice and both serializers see one canonical representation."""
    normalized: dict[str, Any] = {}
    for key, value in flat.items():
        if hasattr(value, "detach") and hasattr(value, "cpu"):  # torch.Tensor
            value = value.detach().cpu().numpy()
        normalized[key] = value
    return normalized


def write_sidecar(
    base_path: Path,
    *,
    renderer: str,
    data: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write the data sidecar for a figure whose stem is `base_path`,
    choosing ``.json`` (light payloads) or ``.npz`` (dense arrays) by total
    array size. Returns the sidecar's resolved path.
    """
    config = dict(config or {})
    flat = _normalize_leaves(_flatten(dict(data)))
    total_elements = sum(v.size for v in flat.values() if isinstance(v, np.ndarray))

    base_path.parent.mkdir(parents=True, exist_ok=True)
    if total_elements <= _JSON_ELEMENT_LIMIT:
        sidecar_path = base_path.with_suffix(".json")
        document = {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "renderer": renderer,
            "config": {k: _leaf_to_jsonable(v) for k, v in config.items()},
            "data": {k: _leaf_to_jsonable(v) for k, v in flat.items()},
        }
        sidecar_path.write_text(json.dumps(document, indent=2))
        return sidecar_path.resolve()

    sidecar_path = base_path.with_suffix(".npz")
    arrays = {k: v for k, v in flat.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in flat.items() if not isinstance(v, np.ndarray)}
    header = json.dumps(
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "renderer": renderer,
            "config": {k: _leaf_to_jsonable(v) for k, v in config.items()},
            "data_scalars": {k: _leaf_to_jsonable(v) for k, v in scalars.items()},
        }
    )
    np.savez_compressed(
        sidecar_path,
        **{_NPZ_HEADER_KEY: np.asarray(header)},  # type: ignore[arg-type]  # numpy stubs mistype **kwds
        **arrays,  # type: ignore[arg-type]
    )
    return sidecar_path.resolve()


def load_sidecar(path: Path | str) -> FigureSidecar:
    """Load a ``.json``/``.npz`` sidecar back into a `FigureSidecar`, with
    every array leaf restored as `np.ndarray` and the nested payload shape
    reconstructed.

    Raises:
        ValueError: on an unknown suffix or an unsupported schema version.
    """
    path = Path(path)
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        version = document["schema_version"]
        flat = {k: _leaf_from_jsonable(v) for k, v in document["data"].items()}
        config = {k: _leaf_from_jsonable(v) for k, v in document["config"].items()}
        renderer = document["renderer"]
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            header = json.loads(str(archive[_NPZ_HEADER_KEY]))
            flat = {k: archive[k] for k in archive.files if k != _NPZ_HEADER_KEY}
        version = header["schema_version"]
        flat.update(
            {k: _leaf_from_jsonable(v) for k, v in header["data_scalars"].items()}
        )
        config = {k: _leaf_from_jsonable(v) for k, v in header["config"].items()}
        renderer = header["renderer"]
    else:
        raise ValueError(
            f"Sidecars are .json or .npz files; got suffix {path.suffix!r}."
        )

    if version != SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            f"Sidecar {path} has schema version {version}; this code "
            f"understands version {SIDECAR_SCHEMA_VERSION}."
        )
    return FigureSidecar(
        schema_version=version,
        renderer=renderer,
        data=_unflatten(flat),
        config=config,
    )


def re_render(sidecar: FigureSidecar | Path | str, **config_overrides: Any) -> Figure:
    """THE restyle-without-recompute entry point (T4.e): reproduce a
    persisted figure from its data sidecar alone -- optionally with cosmetic
    `config_overrides` (a new title, different labels) -- without importing
    solvers, touching the experiment cache, or re-running anything.

    Args:
        sidecar: a loaded `FigureSidecar` or the path of one on disk.
        **config_overrides: config entries overriding the sidecar's own
            (e.g. ``title="Restyled"``).

    Returns:
        The re-rendered Figure.

    Raises:
        KeyError: if the sidecar's renderer name has no registered
            re-render function (import `viz.adapters.notebook` to register
            the standard set).
    """
    if not isinstance(sidecar, FigureSidecar):
        sidecar = load_sidecar(sidecar)
    try:
        render = _SIDECAR_RENDERERS[sidecar.renderer]
    except KeyError:
        raise KeyError(
            f"No renderer registered under {sidecar.renderer!r} "
            f"(known: {sorted(_SIDECAR_RENDERERS)}); import "
            "mbl.viz.adapters.notebook to register the standard set."
        ) from None
    return render(sidecar.data, {**sidecar.config, **config_overrides})


#: The parent folder every content-keyed notebook figure store nests under,
#: kept out of the per-contender ``run_*`` namespace.
_NOTEBOOK_FIGURES_ROOT = "notebook_figures"


def content_keyed_figures_dir(root: Path | str, name: str, digest: str) -> Path:
    """A per-experiment figures directory keyed by its content `digest`, so
    two distinct experiment configurations never overwrite each other's
    persisted figures, and an identical re-run reuses the same directory
    (the "No Figure Overwrite" law -- figures flow into reports/papers, so
    they must be reliable and permanent).

    Args:
        root: The experiments root the store nests under.
        name: The experiment's identity (e.g. ``experiment.name``) -- a
            human-readable prefix on an otherwise opaque digest.
        digest: The experiment's content signature
            (e.g. ``experiment.signature_digest()``): everything that
            changes the plotted DATA folds into it, so a changed problem,
            contender, evaluation, or device yields a fresh directory.

    Returns:
        ``<root>/notebook_figures/<name>__<digest>`` (not yet created; the
        `FigureSink` constructor materializes it).
    """
    return Path(root) / _NOTEBOOK_FIGURES_ROOT / f"{name}__{digest}"


def _existing_sidecar_path(base_path: Path) -> Path | None:
    """The ``.json``/``.npz`` sidecar already on disk for the figure stem
    `base_path`, or ``None`` if neither exists."""
    for suffix in (".json", ".npz"):
        candidate = base_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _payload_fingerprint(
    renderer: str, data: Mapping[str, Any], config: Mapping[str, Any] | None
) -> str:
    """A stable content fingerprint of one figure's exact renderer input --
    ``(renderer name, plotted data, styling config)`` -- computed identically
    for a fresh render and for a persisted sidecar, so the two can be compared
    to decide a cache hit WITHOUT re-encoding anything.

    Arrays contribute their dtype, shape, and raw bytes; the config
    contributes its canonical JSON. A false miss (fingerprints differ when the
    figures would be identical) merely re-renders -- safe; a false hit is made
    effectively impossible by hashing the full array bytes.

    Args:
        renderer: The registered renderer name.
        data: The renderer input series (nested mapping of arrays/scalars) --
            normalized (torch -> numpy) and flattened exactly as
            `write_sidecar` does, so a re-loaded sidecar fingerprints the same.
        config: The labeling/styling kwargs.

    Returns:
        A hex SHA-256 digest.
    """
    flat = _normalize_leaves(_flatten(dict(data)))
    hasher = hashlib.sha256()
    hasher.update(renderer.encode())
    for key in sorted(flat):
        value = flat[key]
        hasher.update(b"\x00")
        hasher.update(key.encode())
        if isinstance(value, np.ndarray):
            hasher.update(b"\x01")
            hasher.update(str(value.dtype).encode())
            hasher.update(str(value.shape).encode())
            hasher.update(np.ascontiguousarray(value).tobytes())
        else:
            hasher.update(b"\x02")
            hasher.update(repr(value).encode())
    encoded_config = {k: _leaf_to_jsonable(v) for k, v in (config or {}).items()}
    hasher.update(json.dumps(encoded_config, sort_keys=True).encode())
    return hasher.hexdigest()


def _sidecar_matches(
    base_path: Path,
    renderer: str,
    data: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> bool:
    """Whether a persisted sidecar for `base_path` carries the EXACT same
    renderer input as ``(renderer, data, config)`` -- the load-from-store hit
    predicate. A missing or unreadable sidecar (e.g. an older schema version)
    is a miss, never an error, so the sink always falls back to rendering."""
    sidecar_path = _existing_sidecar_path(base_path)
    if sidecar_path is None:
        return False
    try:
        existing = load_sidecar(sidecar_path)
    except (ValueError, OSError, KeyError):
        return False
    return _payload_fingerprint(renderer, data, config) == _payload_fingerprint(
        existing.renderer, existing.data, existing.config
    )


def _display(obj: Any) -> None:
    """Inline-display `obj` when IPython is available; silently a no-op in
    plain-script/pytest contexts, so the sink's persistence law never
    depends on a notebook frontend being present."""
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover - IPython is a dev dependency
        return
    display(obj)


class FigureSink:
    """The one component allowed to persist figures (viz layer L3), and the
    enforcement point of the "No Figure Unbacked" law: `save` has no
    figure-only mode -- callers MUST hand over the plotted data, and every
    persisted PDF gains a `.json`/`.npz` sidecar beside it.

    Args:
        figures_dir: directory the artifact pairs are written under
            (created on demand) -- typically a run's own ``figures/`` folder,
            so sidecars live beside their PDFs in the run's artifact
            repository.
        show: display each saved figure/animation inline (IPython) as it is
            saved -- the notebook-facing default; pass False in scripts and
            tests.
    """

    def __init__(self, figures_dir: Path | str, *, show: bool = True) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.show = show

    def save(
        self,
        fig: Figure,
        name: str,
        *,
        renderer: str,
        data: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        close: bool = True,
    ) -> SavedFigure:
        """Persist `fig` as ``<name>.pdf`` (publication vector art) plus a
        ``<name>.png`` raster preview and its ``<name>.json``/``.npz`` data
        sidecar, display it inline (see `show`), and close it.

        Load-from-store: if a persisted sidecar for `name` already carries the
        EXACT same ``(renderer, data, config)`` and both the ``.pdf`` and
        ``.png`` are on disk, the existing artifacts are reused (nothing is
        re-encoded, so a figure that went into a report stays byte-stable
        across re-runs) and only the inline display happens.

        Args:
            fig: the rendered Figure (from a pure `viz.plots`/`viz.landscape`
                renderer).
            name: the artifact stem (no suffix).
            renderer: the registered renderer name a future `re_render` will
                dispatch on.
            data: the exact renderer input series (nested mapping of
                arrays/scalars) -- serialized verbatim into the sidecar.
            config: the labeling/styling kwargs the renderer was called with.
            close: close `fig` after persisting (keeps long notebook
                sessions from accumulating open figures).

        Returns:
            The `SavedFigure` artifact pair (`figure_path` is the ``.pdf``).
        """
        base = self.figures_dir / name
        pdf_path = base.with_suffix(".pdf")
        if (
            pdf_path.is_file()
            and base.with_suffix(".png").is_file()
            and _sidecar_matches(base, renderer, data, config)
        ):
            existing_sidecar = _existing_sidecar_path(base)
            assert existing_sidecar is not None  # _sidecar_matches found it
            figure_path, sidecar_path = pdf_path.resolve(), existing_sidecar.resolve()
        else:
            figure_path = save_vector_figure(fig, pdf_path)
            save_raster_figure(fig, base.with_suffix(".png"))
            sidecar_path = write_sidecar(
                base, renderer=renderer, data=data, config=config
            )
        if self.show:
            _display(fig)
        if close:
            plt.close(fig)
        return SavedFigure(figure_path=figure_path, sidecar_path=sidecar_path)

    def save_animation(
        self,
        name: str,
        write: Callable[[Path], object],
        *,
        renderer: str,
        data: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        suffix: str = ".gif",
    ) -> SavedFigure:
        """The animation face of the same law: `write` encodes the animation
        to ``<name><suffix>`` (e.g. ``lambda path: animator.save(path,
        fps=12)``), and the exact serializable inputs it was built from are
        persisted beside it as the data sidecar. Displayed inline when
        `show` is set.

        Load-from-store: if a persisted sidecar for `name` already carries the
        EXACT same ``(renderer, data, config)`` and the encoded animation is
        on disk, `write` is NOT called -- the expensive re-encoding is skipped
        and the stored animation is displayed as-is. This is the store's main
        time-saver (GIF encoding dominates a notebook's figure cost).

        Returns:
            The `SavedFigure` pair (`figure_path` is the animation file).
        """
        base = self.figures_dir / name
        animation_path = self.figures_dir / f"{name}{suffix}"
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        if animation_path.is_file() and _sidecar_matches(base, renderer, data, config):
            existing_sidecar = _existing_sidecar_path(base)
            assert existing_sidecar is not None  # _sidecar_matches found it
            sidecar_path = existing_sidecar.resolve()
        else:
            write(animation_path)
            sidecar_path = write_sidecar(
                base, renderer=renderer, data=data, config=config
            )
        if self.show:
            try:
                from IPython.display import Image
            except ImportError:  # pragma: no cover
                Image = None  # type: ignore[assignment, misc]  # sentinel when IPython is absent
            if Image is not None and animation_path.suffix == ".gif":
                _display(Image(filename=str(animation_path)))
        return SavedFigure(
            figure_path=animation_path.resolve(), sidecar_path=sidecar_path
        )
