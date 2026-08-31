import pytest
from matplotlib.figure import Figure

from mbl.viz.plots.benchmarking import (
    BenchmarkRecord,
    MemoryBreakdown,
    plot_cartesian_performance_tradeoff,
    plot_computational_benchmarks,
)

LABELS = ["Riccati synthesis", "GD (Jacobi)", "GD (Gauss-Seidel)"]
OFFLINE_S = [0.2, 0.25, 0.25]
ONLINE_S = [0.002, 1.2, 0.9]

# Offline: both a NumPy-only closed-form solver and a PyTorch-native
# iterative solver synthesize (and thus resident-store) the same Riccati
# recursion, so both carry a comparable, non-zero NumPy layer; GD's offline
# phase additionally carries a small PyTorch layer (its converted gradient
# coefficients).
OFFLINE_MEMORY = [
    MemoryBreakdown(numpy_mb=5.0, torch_mb=0.0),
    MemoryBreakdown(numpy_mb=4.8, torch_mb=0.5),
    MemoryBreakdown(numpy_mb=4.8, torch_mb=0.4),
]
# Online: Riccati's rollout stays NumPy-only and minimal; GD's macro-
# iteration loop is PyTorch-native and dominates.
ONLINE_MEMORY = [
    MemoryBreakdown(numpy_mb=0.1, torch_mb=0.0),
    MemoryBreakdown(numpy_mb=0.2, torch_mb=40.0),
    MemoryBreakdown(numpy_mb=0.2, torch_mb=38.0),
]


def _records(offline_memory=None):
    return [
        BenchmarkRecord(
            label=label,
            offline_s=offline_s,
            online_s=online_s,
            offline_memory=offline_mem,
            online_memory=online_mem,
        )
        for label, offline_s, online_s, offline_mem, online_mem in zip(
            LABELS,
            OFFLINE_S,
            ONLINE_S,
            offline_memory if offline_memory is not None else OFFLINE_MEMORY,
            ONLINE_MEMORY,
        )
    ]


RECORDS = _records()


def test_plot_computational_benchmarks_returns_figure_with_four_axes() -> None:
    fig = plot_computational_benchmarks(RECORDS)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4


def test_cartesian_performance_tradeoff_returns_2x2_with_microsecond_online() -> None:
    """Directive 4: the 2x2 Cartesian barplot (offline/online x time/memory);
    the online-time panel is expressed per-control-step in microseconds and
    scales the input seconds by 1e6."""
    fig = plot_cartesian_performance_tradeoff(RECORDS)

    assert isinstance(fig, Figure)
    offline_time_ax, online_latency_ax, _, _ = fig.axes
    assert "microseconds" in online_latency_ax.get_ylabel()
    # offline-time bars are the raw seconds; online-latency bars are seconds*1e6
    assert [b.get_height() for b in offline_time_ax.containers[0]] == OFFLINE_S
    online_heights = [b.get_height() for b in online_latency_ax.containers[0]]
    assert online_heights == pytest.approx([s * 1e6 for s in ONLINE_S])


def test_cartesian_performance_tradeoff_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_cartesian_performance_tradeoff([])


def test_plot_computational_benchmarks_time_panels_have_one_bar_per_label() -> None:
    fig = plot_computational_benchmarks(RECORDS)
    offline_time_ax, online_time_ax, _, _ = fig.axes

    assert len(offline_time_ax.patches) == len(LABELS)
    assert len(online_time_ax.patches) == len(LABELS)


def test_plot_computational_benchmarks_memory_panels_have_two_segments_per_label() -> (
    None
):
    fig = plot_computational_benchmarks(RECORDS)
    _, _, offline_memory_ax, online_memory_ax = fig.axes

    assert len(offline_memory_ax.patches) == 2 * len(LABELS)
    assert len(online_memory_ax.patches) == 2 * len(LABELS)


def test_plot_computational_benchmarks_time_panel_values_match_inputs() -> None:
    fig = plot_computational_benchmarks(RECORDS)
    offline_time_ax, online_time_ax, _, _ = fig.axes

    for ax, values in ((offline_time_ax, OFFLINE_S), (online_time_ax, ONLINE_S)):
        for bar, value in zip(ax.patches, values):
            assert bar.get_y() == pytest.approx(0.0)
            assert bar.get_height() == pytest.approx(value)


def test_plot_computational_benchmarks_memory_panel_segments_are_stacked_correctly() -> (
    None
):
    fig = plot_computational_benchmarks(RECORDS)
    _, _, offline_memory_ax, _ = fig.axes

    n = len(LABELS)
    numpy_bars = offline_memory_ax.patches[:n]
    torch_bars = offline_memory_ax.patches[n:]
    for numpy_bar, torch_bar, breakdown in zip(numpy_bars, torch_bars, OFFLINE_MEMORY):
        assert numpy_bar.get_y() == pytest.approx(0.0)
        assert numpy_bar.get_height() == pytest.approx(breakdown.numpy_mb)
        assert torch_bar.get_y() == pytest.approx(breakdown.numpy_mb)
        assert torch_bar.get_height() == pytest.approx(breakdown.torch_mb)


def test_plot_computational_benchmarks_annotates_exact_values() -> None:
    fig = plot_computational_benchmarks(RECORDS)
    offline_time_ax, online_time_ax, offline_memory_ax, online_memory_ax = fig.axes

    assert "0.2 s" in {t.get_text() for t in offline_time_ax.texts}
    assert "1.2 s" in {t.get_text() for t in online_time_ax.texts}
    # Memory panels annotate the STACK TOTAL (numpy_mb + torch_mb), not
    # either layer alone.
    assert "5 MB" in {t.get_text() for t in offline_memory_ax.texts}  # 5.0 + 0.0
    assert "40.2 MB" in {t.get_text() for t in online_memory_ax.texts}  # 0.2 + 40.0


def test_plot_computational_benchmarks_renders_na_for_missing_memory_total() -> None:
    offline_memory = [
        MemoryBreakdown(numpy_mb=5.0, torch_mb=0.0),
        MemoryBreakdown(numpy_mb=None, torch_mb=0.5),
        MemoryBreakdown(numpy_mb=float("nan"), torch_mb=0.4),
    ]
    fig = plot_computational_benchmarks(_records(offline_memory=offline_memory))

    offline_memory_ax = fig.axes[2]
    memory_texts = [t.get_text() for t in offline_memory_ax.texts]
    assert memory_texts.count("N/A") == 2


def test_plot_computational_benchmarks_applies_hatching_and_distinct_colors() -> None:
    fig = plot_computational_benchmarks(RECORDS)

    for ax in fig.axes:
        n = len(LABELS)
        # Compare only the first (per-method) segment set on memory panels,
        # since the second segment set is the SAME per-method colors/hatches
        # deliberately lightened/sparsified -- a second, distinct layer, not
        # a second method.
        hatches = [bar.get_hatch() for bar in ax.patches[:n]]
        colors = [bar.get_facecolor() for bar in ax.patches[:n]]
        assert len(set(hatches)) == len(LABELS)
        assert len(set(colors)) == len(LABELS)


def test_plot_computational_benchmarks_memory_layers_are_visually_distinct() -> None:
    fig = plot_computational_benchmarks(RECORDS)
    _, _, offline_memory_ax, _ = fig.axes

    n = len(LABELS)
    for numpy_bar, torch_bar in zip(
        offline_memory_ax.patches[:n], offline_memory_ax.patches[n:]
    ):
        assert numpy_bar.get_alpha() != torch_bar.get_alpha()
        assert numpy_bar.get_hatch() != torch_bar.get_hatch()


def test_plot_computational_benchmarks_same_method_same_color_across_panels() -> None:
    fig = plot_computational_benchmarks(RECORDS)
    n = len(LABELS)

    for method_index in range(n):
        # Time panels: one bar per method, indexed directly. Memory panels:
        # the first n patches are each method's NumPy-layer segment, so the
        # same index still isolates that method's bar.
        patches_at_method = [ax.patches[method_index] for ax in fig.axes]
        colors = {p.get_facecolor() for p in patches_at_method}
        hatches = {p.get_hatch() for p in patches_at_method}
        assert len(colors) == 1
        assert len(hatches) == 1


def test_plot_computational_benchmarks_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_computational_benchmarks([])


def test_plot_computational_benchmarks_uses_custom_title() -> None:
    fig = plot_computational_benchmarks(RECORDS, title="Custom Benchmark Title")
    assert fig._suptitle.get_text() == "Custom Benchmark Title"


def test_memory_breakdown_total_mb_sums_both_layers() -> None:
    breakdown = MemoryBreakdown(numpy_mb=3.0, torch_mb=2.5)
    assert breakdown.total_mb == pytest.approx(5.5)


def test_memory_breakdown_total_mb_is_none_when_a_layer_is_missing() -> None:
    assert MemoryBreakdown(numpy_mb=None, torch_mb=1.0).total_mb is None
    assert MemoryBreakdown(numpy_mb=1.0, torch_mb=float("nan")).total_mb is None
