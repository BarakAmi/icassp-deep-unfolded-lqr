"""The dual-format `FigureSink` and its sidecar round-trip (REFACTOR_PLAN
v3, T4.e / §7.12): every persisted figure is a PDF + raw-data sidecar pair,
and `re_render` reproduces the original figure's data exactly from the
sidecar alone -- no solver, cache, or recomputation involved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mbl.viz.adapters.notebook  # noqa: F401  registers the standard sidecar renderers
from mbl.viz.adapters.sink import (
    SIDECAR_SCHEMA_VERSION,
    FigureSink,
    content_keyed_figures_dir,
    load_sidecar,
    re_render,
    write_sidecar,
)
from mbl.viz.plots import CostComparisonStyle, plot_empirical_vs_theoretical_cost


def _small_curves(horizon: int = 12) -> dict[str, np.ndarray]:
    t = np.arange(1, horizon + 1, dtype=float)
    theoretical = 2.0 + 1.0 / t
    empirical = theoretical * (1.0 + 0.01 * np.sin(t))
    return {"empirical_cost": empirical, "theoretical_cost": theoretical}


def _saved_pair(tmp_path, curves):
    sink = FigureSink(tmp_path, show=False)
    fig = plot_empirical_vs_theoretical_cost(
        **curves, style=CostComparisonStyle(title="Original")
    )
    return sink.save(
        fig,
        "cost_comparison",
        renderer="empirical_vs_theoretical_cost",
        data=curves,
        config={"title": "Original"},
    )


def test_save_persists_the_pdf_and_json_sidecar_pair(tmp_path) -> None:
    saved = _saved_pair(tmp_path, _small_curves())

    assert saved.figure_path.is_file()
    assert saved.figure_path.suffix == ".pdf"
    assert saved.figure_path.read_bytes().startswith(b"%PDF-")
    assert saved.sidecar_path.is_file()
    assert saved.sidecar_path.suffix == ".json"
    assert saved.sidecar_path.parent == saved.figure_path.parent


def test_save_has_no_figure_only_mode(tmp_path) -> None:
    """The "No Figure Unbacked" law is structural: `save` cannot even be
    called without the plotted data."""
    sink = FigureSink(tmp_path, show=False)
    fig, _ = plt.subplots()
    with pytest.raises(TypeError):
        sink.save(fig, "unbacked", renderer="anything")  # type: ignore[call-arg]  # the omission is the test
    plt.close(fig)


def test_dense_payloads_become_npz_sidecars(tmp_path) -> None:
    dense = {"cost_grid": np.arange(4096, dtype=float).reshape(64, 64)}
    path = write_sidecar(
        tmp_path / "dense", renderer="local_cost_landscape", data=dense
    )
    assert path.suffix == ".npz"

    sidecar = load_sidecar(path)
    np.testing.assert_array_equal(sidecar.data["cost_grid"], dense["cost_grid"])


def test_json_sidecar_round_trips_nested_payloads_exactly(tmp_path) -> None:
    data: dict = {
        "curves": {
            "cold start": {"J_history": np.array([3.0, 1.5, 1.1]), "J_final": 1.1},
            "warm start": {"J_history": np.array([2.0, 1.2, 1.05]), "J_final": 1.05},
        },
        "J_opt": 1.0,
    }
    path = write_sidecar(
        tmp_path / "convergence",
        renderer="convergence_curves",
        data=data,
        config={"title": "Study"},
    )
    sidecar = load_sidecar(path)

    assert sidecar.schema_version == SIDECAR_SCHEMA_VERSION
    assert sidecar.renderer == "convergence_curves"
    assert sidecar.config == {"title": "Study"}
    np.testing.assert_array_equal(
        sidecar.data["curves"]["cold start"]["J_history"], [3.0, 1.5, 1.1]
    )
    assert sidecar.data["curves"]["warm start"]["J_final"] == 1.05
    assert sidecar.data["J_opt"] == 1.0


def test_npz_sidecar_round_trips_nested_payloads_exactly(tmp_path) -> None:
    rng = np.random.default_rng(0)
    data: dict = {
        "grids": {
            "0": rng.standard_normal((40, 40)),
            "1": rng.standard_normal((40, 40)),
        },
        "spec": {"timestep": 7, "components": [0, 1]},
    }
    path = write_sidecar(
        tmp_path / "landscape", renderer="local_cost_landscape", data=data
    )
    assert path.suffix == ".npz"

    sidecar = load_sidecar(path)
    np.testing.assert_array_equal(sidecar.data["grids"]["0"], data["grids"]["0"])
    np.testing.assert_array_equal(sidecar.data["grids"]["1"], data["grids"]["1"])
    assert sidecar.data["spec"]["timestep"] == 7
    assert sidecar.data["spec"]["components"] == [0, 1]


def test_re_render_reproduces_the_original_figures_data_exactly(tmp_path) -> None:
    """§7.12: sidecar-driven re-rendering reproduces the original figure's
    plotted data bit-for-bit, without any solver in sight."""
    curves = _small_curves()
    original = plot_empirical_vs_theoretical_cost(
        **curves, style=CostComparisonStyle(title="Original")
    )
    saved = _saved_pair(tmp_path, curves)

    rebuilt = re_render(saved.sidecar_path)

    original_lines = [line.get_ydata() for line in original.axes[0].get_lines()]
    rebuilt_lines = [line.get_ydata() for line in rebuilt.axes[0].get_lines()]
    assert len(original_lines) == len(rebuilt_lines) > 0
    for original_y, rebuilt_y in zip(original_lines, rebuilt_lines):
        np.testing.assert_array_equal(original_y, rebuilt_y)
    plt.close(original)
    plt.close(rebuilt)


def test_re_render_applies_cosmetic_config_overrides(tmp_path) -> None:
    saved = _saved_pair(tmp_path, _small_curves())

    restyled = re_render(saved.sidecar_path, title="Journal Restyle")

    assert restyled.axes[0].get_title() == "Journal Restyle"
    plt.close(restyled)


def test_re_render_rejects_unregistered_renderers(tmp_path) -> None:
    path = write_sidecar(
        tmp_path / "orphan", renderer="not_a_renderer", data={"x": 1.0}
    )
    with pytest.raises(KeyError, match="not_a_renderer"):
        re_render(path)


def test_save_animation_backs_the_gif_with_a_data_sidecar(tmp_path) -> None:
    sink = FigureSink(tmp_path, show=False)
    frames = {"baseline_U": np.zeros((5, 2)), "candidate_history": np.ones((3, 5, 2))}

    saved = sink.save_animation(
        "signals_animated",
        lambda path: path.write_bytes(b"GIF89a"),
        renderer="signal_animation",
        data=frames,
        config={"fps": 12},
    )

    assert saved.figure_path.name == "signals_animated.gif"
    assert saved.figure_path.read_bytes().startswith(b"GIF89a")
    sidecar = load_sidecar(saved.sidecar_path)
    np.testing.assert_array_equal(
        sidecar.data["candidate_history"], frames["candidate_history"]
    )


def test_sidecar_rejects_reserved_separator_in_keys(tmp_path) -> None:
    with pytest.raises(ValueError, match="separator"):
        write_sidecar(tmp_path / "bad", renderer="r", data={"a/b": 1.0})


def test_sidecar_rejects_unserializable_leaves(tmp_path) -> None:
    with pytest.raises(TypeError, match="never pickle"):
        write_sidecar(tmp_path / "bad", renderer="r", data={"obj": object()})


def test_load_sidecar_rejects_unknown_schema_versions(tmp_path) -> None:
    path = write_sidecar(tmp_path / "versioned", renderer="r", data={"x": 1.0})
    document = path.read_text().replace(
        f'"schema_version": {SIDECAR_SCHEMA_VERSION}', '"schema_version": 999'
    )
    path.write_text(document)
    with pytest.raises(ValueError, match="schema version"):
        load_sidecar(path)


def test_torch_tensors_are_normalized_to_numpy_in_sidecars(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    data = {"J_history": torch.tensor([3.0, 2.0, 1.0])}
    path = write_sidecar(tmp_path / "tensor", renderer="r", data=data)

    sidecar = load_sidecar(path)
    assert isinstance(sidecar.data["J_history"], np.ndarray)
    np.testing.assert_array_equal(sidecar.data["J_history"], [3.0, 2.0, 1.0])


def test_re_render_reconstructs_a_landscape_from_its_npz_sidecar(tmp_path) -> None:
    """The dataclass-input renderers round-trip too: the sidecar codec
    rebuilds `CostField` (including `reference_U`) and `TrajectoryOverlay`
    verbatim, then re-dispatches to the pure landscape renderer."""
    from mbl.viz.adapters.notebook import _landscape_sidecar_data
    from mbl.viz.landscape import GridProjector, SliceSpec, TrajectoryOverlay

    resolution = 24
    baseline = np.zeros((5, 2))
    spec = SliceSpec(
        timestep=2,
        components=(0, 1),
        ranges=(
            np.linspace(-1.0, 1.0, resolution),
            np.linspace(-1.0, 1.0, resolution),
        ),
    )
    field = GridProjector().project(
        baseline, lambda U: (U[:, 2, :] ** 2).sum(axis=-1), spec
    )
    history = np.linspace(1.0, 0.0, 4)[:, None, None] * np.ones((1, 5, 2))
    overlay = TrajectoryOverlay(
        history,
        iteration_costs=np.linspace(2.0, 0.0, 4),
        baseline_U=baseline,
        optimum_value=0.0,
    )
    path = write_sidecar(
        tmp_path / "landscape",
        renderer="local_cost_landscape",
        data=_landscape_sidecar_data(field, overlay),
        config={"kind": "contour", "title": "L"},
    )
    assert path.suffix == ".npz"

    fig = re_render(path)

    assert fig.axes  # rendered without touching any solver
    loaded = load_sidecar(path)
    np.testing.assert_array_equal(loaded.data["cost_grid"], field.cost_grid)
    np.testing.assert_array_equal(loaded.data["reference_U"], field.reference_U)
    plt.close(fig)


# --- The content-keyed figure store: PNG preview + load-from-store on a hit ---


def test_content_keyed_figures_dir_isolates_experiments(tmp_path) -> None:
    """Two distinct experiment signatures resolve to distinct directories --
    the "No Figure Overwrite" law -- while sharing one parent folder."""
    a = content_keyed_figures_dir(tmp_path, "NB03", "digestAAA")
    b = content_keyed_figures_dir(tmp_path, "NB03", "digestBBB")

    assert a != b
    assert a.parent == b.parent == tmp_path / "notebook_figures"
    assert a.name == "NB03__digestAAA"


def test_save_also_writes_a_png_preview(tmp_path) -> None:
    """Every static figure gains a ``.png`` raster face beside its ``.pdf`` --
    a report-/web-embeddable artifact and the load-from-store display asset."""
    saved = _saved_pair(tmp_path, _small_curves())
    png = saved.figure_path.with_suffix(".png")

    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_identical_resave_reuses_the_stored_artifacts(tmp_path) -> None:
    """An identical ``(renderer, data, config)`` re-save is a hit: the ``.pdf``
    and ``.png`` are NOT re-encoded, so a figure that went into a report stays
    byte-stable across re-runs."""
    curves = _small_curves()
    first = _saved_pair(tmp_path, curves)
    png = first.figure_path.with_suffix(".png")
    pdf_mtime, png_mtime = first.figure_path.stat().st_mtime_ns, png.stat().st_mtime_ns

    second = _saved_pair(tmp_path, curves)  # identical inputs -> a hit

    assert second.figure_path == first.figure_path
    assert second.figure_path.stat().st_mtime_ns == pdf_mtime  # not re-written
    assert png.stat().st_mtime_ns == png_mtime


def test_cosmetic_config_change_forces_a_re_render(tmp_path) -> None:
    """The same data with a different styling `config` is a MISS -- the figure
    is re-rendered and the sidecar rewritten to the new config."""
    curves = _small_curves()
    sink = FigureSink(tmp_path, show=False)

    fig_a = plot_empirical_vs_theoretical_cost(
        **curves, style=CostComparisonStyle(title="A")
    )
    sink.save(
        fig_a,
        "cost",
        renderer="empirical_vs_theoretical_cost",
        data=curves,
        config={"title": "A"},
    )
    fig_b = plot_empirical_vs_theoretical_cost(
        **curves, style=CostComparisonStyle(title="B")
    )
    saved_b = sink.save(
        fig_b,
        "cost",
        renderer="empirical_vs_theoretical_cost",
        data=curves,
        config={"title": "B"},
    )

    assert load_sidecar(saved_b.sidecar_path).config == {"title": "B"}


def test_animation_hit_skips_re_encoding(tmp_path) -> None:
    """The store's headline saving: an identical animation re-save does NOT
    invoke `write` again -- the expensive GIF encoding is skipped."""
    sink = FigureSink(tmp_path, show=False)
    frames = {"baseline_U": np.zeros((5, 2)), "candidate_history": np.ones((3, 5, 2))}
    encode_calls: list = []

    def write(path):
        encode_calls.append(path)
        path.write_bytes(b"GIF89a")

    kwargs = dict(renderer="signal_animation", data=frames, config={"fps": 12})
    sink.save_animation("anim", write, **kwargs)
    assert len(encode_calls) == 1

    sink.save_animation("anim", write, **kwargs)  # identical inputs -> a hit
    assert len(encode_calls) == 1  # write NOT invoked again
