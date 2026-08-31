import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mbl.core.profiling import (
    COLLECTOR,
    PROFILING,
    ProfileResult,
    aggregate_results,
    cpu_clock,
    current_peak_ram_bytes,
    numpy_array_bytes,
    profile_block,
    profiled,
    track_stratified_memory,
    track_torch_memory,
    wall_clock,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _restore_profiling_state():
    """Snapshot/restore the global profiling switch + collector around each test."""
    enabled = PROFILING.enabled
    COLLECTOR.reset()
    yield
    PROFILING.enabled = enabled
    COLLECTOR.reset()


def test_clocks_are_monotonic_nonnegative() -> None:
    assert wall_clock() >= 0.0
    assert cpu_clock() >= 0.0


def test_profile_block_measures_finite_latency() -> None:
    with profile_block("unit") as handle:
        sum(range(10_000))
    result = handle.result
    assert result is not None
    assert result.name == "unit"
    assert result.wall_time_s >= 0.0
    assert result.cpu_time_s >= 0.0


def test_profile_block_cpu_only_has_no_cuda_time() -> None:
    import torch

    if torch.cuda.is_available():  # pragma: no cover - CI is CPU-only
        pytest.skip("CUDA present; this asserts the CPU-only contract")
    PROFILING.enabled = True
    with profile_block("cpu") as handle:
        _ = [0] * 1000
    assert handle.result is not None
    assert handle.result.cuda_time_ms is None
    assert handle.result.peak_vram_bytes is None


def test_profiled_records_into_collector_when_enabled() -> None:
    PROFILING.enabled = True

    @profiled("adder")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    aggregates = COLLECTOR.aggregate()
    assert "adder" in aggregates
    assert aggregates["adder"].count == 1


def test_profiled_is_a_noop_recorder_when_disabled() -> None:
    PROFILING.enabled = False

    @profiled("adder")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5  # still computes correctly
    assert COLLECTOR.aggregate() == {}  # but records nothing (zero-overhead opt-out)


def test_collector_aggregates_multiple_calls() -> None:
    PROFILING.enabled = True

    @profiled("loop")
    def work() -> None:
        return None

    for _ in range(5):
        work()
    aggregate = COLLECTOR.aggregate()["loop"]
    assert aggregate.count == 5
    assert aggregate.wall_total_s >= 0.0
    assert aggregate.wall_p95_s >= aggregate.wall_p50_s


def test_aggregate_results_percentiles_and_maxima() -> None:
    results = [
        ProfileResult("b", wall_time_s=w, cpu_time_s=w, peak_ram_bytes=r)
        for w, r in [(0.1, 100), (0.2, 300), (0.3, 200)]
    ]
    agg = aggregate_results("b", results)
    assert agg.count == 3
    assert agg.wall_total_s == pytest.approx(0.6)
    assert agg.wall_mean_s == pytest.approx(0.2)
    assert agg.peak_ram_bytes == 300  # max, not last


def test_profile_result_as_metrics_omits_absent_device_fields() -> None:
    result = ProfileResult("x", wall_time_s=0.5, cpu_time_s=0.4)
    metrics = result.as_metrics()
    assert metrics["profile/x/wall_time_s"] == 0.5
    assert "profile/x/cuda_time_ms" not in metrics
    assert "profile/x/peak_vram_mb" not in metrics


def test_track_torch_memory_reports_nonzero_for_real_tensor_ops() -> None:
    import torch

    with track_torch_memory() as handle:
        x = torch.randn(1000, 1000, dtype=torch.float64)
        _ = x @ x
    assert handle.bytes > 0


def test_track_torch_memory_reports_zero_for_pure_numpy_work() -> None:
    with track_torch_memory() as handle:
        _ = np.random.randn(1000, 1000)
    assert handle.bytes == 0


def test_peak_ram_bytes_is_positive_on_linux() -> None:
    ram = current_peak_ram_bytes()
    if ram is not None:  # None only where `resource` is unavailable
        assert ram > 0


def test_numpy_array_bytes_sums_structural_footprint() -> None:
    a = np.zeros((10, 10), dtype=np.float64)  # 800 bytes
    b = np.zeros((5,), dtype=np.float32)  # 20 bytes
    assert numpy_array_bytes(a, b) == 820


def test_numpy_array_bytes_of_no_arrays_is_zero() -> None:
    assert numpy_array_bytes() == 0


def test_track_stratified_memory_reports_numpy_layer_for_registered_arrays() -> None:
    with track_stratified_memory() as handle:
        p_arr = np.random.randn(200, 40, 40)  # ~5.1 MB, well above RSS noise
        handle.track_numpy(p_arr)
    report = handle.report
    assert report is not None
    assert report.numpy_cpu_bytes >= p_arr.nbytes
    assert report.torch_bytes == 0


def test_track_stratified_memory_reports_torch_layer_for_real_tensor_ops() -> None:
    import torch

    with track_stratified_memory() as handle:
        x = torch.randn(1000, 1000, dtype=torch.float64)
        _ = x @ x
    report = handle.report
    assert report is not None
    assert report.torch_bytes > 0


def test_track_stratified_memory_reports_zero_numpy_layer_when_nothing_registered() -> (
    None
):
    with track_stratified_memory() as handle:
        _ = 1 + 1
    report = handle.report
    assert report is not None
    assert report.numpy_cpu_bytes == 0


def test_track_stratified_memory_ignores_rss_delta_once_torch_work_is_detected() -> (
    None
):
    """Regression test: a block mixing a small registered NumPy array with
    substantial PyTorch CPU tensor allocation must not let the PyTorch
    allocation's own (much larger) RSS growth inflate the NumPy layer -- the
    RSS-delta fallback is only safe for blocks with no detectable PyTorch
    activity at all (see `track_stratified_memory`'s docstring)."""
    import torch

    small_array = np.zeros((10,), dtype=np.float64)  # 80 bytes

    with track_stratified_memory() as handle:
        handle.track_numpy(small_array)
        # A large-ish PyTorch allocation, so the process's RSS high-water
        # mark grows by (at least) several MB during this block -- far more
        # than the registered array's 80 bytes.
        x = torch.randn(2000, 2000, dtype=torch.float64)
        _ = x @ x
    report = handle.report

    assert report is not None
    assert report.torch_bytes > 0
    assert report.numpy_cpu_bytes == numpy_array_bytes(small_array)


def test_stratified_memory_report_total_bytes_sums_both_layers() -> None:
    with track_stratified_memory() as handle:
        handle.track_numpy(np.zeros((100,), dtype=np.float64))
    report = handle.report
    assert report is not None
    assert report.total_bytes == report.numpy_cpu_bytes + report.torch_bytes


def test_sc_profile_env_var_enables_profiling_in_subprocess() -> None:
    script = (
        "from mbl.core.profiling import PROFILING, profiled, COLLECTOR\n"
        "assert PROFILING.enabled is True\n"
        "@profiled('x')\n"
        "def f():\n"
        "    return 1\n"
        "f()\n"
        "assert COLLECTOR.aggregate()['x'].count == 1\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "SC_PROFILE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
