"""Lightweight, opt-in performance profiling.

Captures two metric families around a block of work:

* **Latency** -- wall-clock (``perf_counter``), CPU (``process_time``), and, on
  CUDA, device time (CUDA events + ``synchronize``).
* **Memory** -- peak process RAM (``resource.ru_maxrss`` high-water mark) and,
  on CUDA, peak VRAM (``torch.cuda.max_memory_allocated``).

Two entry points, matching the Phase 1A guard discipline:

* :func:`profile_block` / :func:`profiled` -- fine-grained block/function timing
  that records into the process-global :data:`COLLECTOR`. **Strictly opt-in**
  via ``SC_PROFILE`` (default off). :func:`profiled` is meant only for
  epoch/rollout-level boundaries (``forward``/``run``), never inner loops, so
  its single per-call ``PROFILING.enabled`` check is negligible while remaining
  runtime-toggleable (unlike the decoration-time guard/shape decorators).
* The epoch/run-level primitives (:func:`wall_clock`, :func:`cpu_clock`,
  :func:`current_peak_ram_bytes`, :func:`reset_gpu_peak_memory`,
  :func:`peak_gpu_memory_bytes`) are always cheap and are used by
  ``engine.callbacks.ProfilingCallback`` to log latency/memory into every run's
  ``metadata.json`` regardless of ``SC_PROFILE``.

A third memory primitive, :func:`track_stratified_memory`, layers
:func:`numpy_array_bytes` (exact structural byte-counting of a block's own
NumPy output) and a CPU-RSS delta sampler on top of :func:`track_torch_memory`
so a block's footprint is reported per heterogeneous device/library layer
(NumPy/CPU vs. PyTorch CPU+CUDA) rather than as one number that silently
reads as zero for whichever layer a single-backend tracker can't see -- e.g.
a NumPy-only Riccati recursion vs. a PyTorch-native gradient-descent solver.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

import numpy as np

try:  # POSIX only; absent on native Windows (this project runs on WSL/Linux).
    import resource
except ImportError:  # pragma: no cover - platform fallback
    resource = None  # type: ignore[assignment]

import torch

F = TypeVar("F", bound=Callable[..., Any])

_TRUTHY = {"1", "true", "yes", "on"}
# On Linux ru_maxrss is in kilobytes; on macOS it is already bytes.
_RU_MAXRSS_TO_BYTES = 1024 if sys.platform.startswith("linux") else 1


class ProfilingConfig:
    """Process-global opt-in switch for block/function profiling, seeded once
    from ``SC_PROFILE``."""

    __slots__ = ("enabled",)

    def __init__(self) -> None:
        self.enabled = os.environ.get("SC_PROFILE", "").strip().lower() in _TRUTHY


PROFILING = ProfilingConfig()


# --- always-cheap measurement primitives (used by ProfilingCallback) ---------


def wall_clock() -> float:
    """Current wall-clock time in seconds, from a monotonic, high-resolution
    clock (``time.perf_counter``). Only meaningful as a difference between two
    calls, not as an absolute timestamp.

    Returns:
        Seconds, as a ``float``.
    """
    return time.perf_counter()


def cpu_clock() -> float:
    """Current process CPU time in seconds (``time.process_time``), summing
    system and user CPU time but excluding time spent sleeping/blocked on I/O.
    Only meaningful as a difference between two calls.

    Returns:
        Seconds, as a ``float``.
    """
    return time.process_time()


def cuda_available() -> bool:
    """Return whether a CUDA device is available in this process.

    Returns:
        ``torch.cuda.is_available()``.
    """
    return torch.cuda.is_available()


def current_peak_ram_bytes() -> int | None:
    """Process peak resident set size (high-water mark) in bytes, or ``None``
    where ``resource`` is unavailable.

    Returns:
        The process's peak RSS in bytes (via ``resource.ru_maxrss``, converted
        from kilobytes on Linux), or ``None`` on platforms without the
        ``resource`` module (e.g. native Windows).
    """
    if resource is None:
        return None  # type: ignore[unreachable]  # reachable on non-POSIX hosts
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RU_MAXRSS_TO_BYTES


def reset_gpu_peak_memory() -> None:
    """Reset CUDA's peak-memory-allocated counter for the current device
    (no-op if no CUDA device is available)."""
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_bytes() -> int | None:
    """Peak CUDA memory allocated on the current device since the last
    `reset_gpu_peak_memory` call.

    Returns:
        Peak allocated bytes (via ``torch.cuda.max_memory_allocated``, after a
        ``torch.cuda.synchronize()`` to ensure all queued kernels have run), or
        ``None`` if no CUDA device is available.
    """
    if not cuda_available():
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def current_gpu_allocated_bytes() -> int:
    """CUDA memory currently allocated on the current device, in bytes.

    The live-allocation baseline a measured block starts from. This is NOT
    zero in general: PyTorch retains e.g. per-stream cuBLAS/cuSOLVER
    workspaces (~tens of MB after any ``linalg``/matmul call) as *allocated*
    — not merely cached — for the life of the process, and
    ``torch.cuda.reset_peak_memory_stats`` re-seeds the peak counter to
    exactly this baseline. Peak-based block attribution must therefore
    subtract it (see `track_stratified_memory`), or every block measured
    after the process's first CUDA mathematics inherits the baseline as
    phantom footprint.

    Returns:
        ``torch.cuda.memory_allocated()``, or ``0`` if no CUDA device is
        available.
    """
    if not cuda_available():
        return 0
    return int(torch.cuda.memory_allocated())


@contextmanager
def _suppress_native_profiler_noise() -> Iterator[None]:
    """Silences `torch.profiler`'s ``__enter__``/``__exit__`` chatter: the
    "Profiler clears events..." `UserWarning` it emits every call, and
    libkineto's own ``USDT: ... ActivityProfilerController.cpp ...
    profiler_start``/``profiler_stop`` lines. The latter are written by C++
    glog-style logging straight to the process's stderr file descriptor,
    bypassing `sys.stderr` entirely -- a `warnings` filter alone leaves them
    on-screen, so this also duplicates fd 2 to `os.devnull` for the duration
    of the wrapped call.
    """
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


class _TorchMemoryHandle:
    """Mutable receiver so a ``with track_torch_memory() as h:`` caller can
    read ``h.bytes`` after the block (only known at block exit)."""

    __slots__ = ("bytes",)

    def __init__(self) -> None:
        self.bytes: int = 0


def numpy_array_bytes(*arrays: np.ndarray) -> int:
    """Structural byte footprint of `arrays`: the sum of each array's own
    ``.nbytes``.

    An exact, allocator-independent measurement -- unlike
    `current_peak_ram_bytes` (a process-wide RSS *high-water mark* that
    never decreases and can miss a block entirely if the process already
    peaked earlier), this counts precisely the bytes a NumPy computation's
    own *output* arrays occupy, regardless of when or how the process's
    resident set actually grew.

    Args:
        arrays: The arrays whose footprint should be counted (e.g. a
            Riccati recursion's ``P_arr``/``K_arr``).

    Returns:
        The sum of ``a.nbytes`` over `arrays` (``0`` if none given).
    """
    return sum(int(a.nbytes) for a in arrays)


@contextmanager
def track_torch_memory() -> Iterator[_TorchMemoryHandle]:
    """PyTorch-native allocation footprint of a block, in bytes.

    Process RSS (`current_peak_ram_bytes`) is a *high-water mark*: it never
    decreases within a process, and PyTorch's caching allocator routinely
    services short-lived tensor allocations out of memory it already holds --
    so a block that does real tensor work can easily report a 0-byte RSS
    delta simply because the process's peak was already reached earlier.
    This instead wraps the block in `torch.profiler` with
    ``profile_memory=True`` and sums each op's own (non-negative) allocation
    -- CPU and, when available, CUDA -- giving a non-zero, PyTorch-native
    footprint even when nothing moves the process-wide RSS high-water mark.

    Yields:
        A `_TorchMemoryHandle` whose ``.bytes`` is populated once the
        ``with`` block exits, including on an exception.
    """
    handle = _TorchMemoryHandle()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    prof = torch.profiler.profile(activities=activities, profile_memory=True)
    with _suppress_native_profiler_noise():
        prof.__enter__()
    try:
        yield handle
    finally:
        # `key_averages()` only sees finalized events after `__exit__` --
        # calling it while still inside the `with` silently returns zeros.
        with _suppress_native_profiler_noise():
            prof.__exit__(None, None, None)
        handle.bytes = sum(
            max(event.self_cpu_memory_usage, 0) + max(event.self_device_memory_usage, 0)
            for event in prof.key_averages()
        )


@dataclass(frozen=True)
class StratifiedMemoryReport:
    """A block's memory footprint, stratified by the heterogeneous
    device/library layer that actually holds the bytes -- so a block mixing
    NumPy/CPU-only computation (e.g. a Riccati recursion) with PyTorch
    CPU/GPU tensor allocation (e.g. a gradient-descent solver) never
    collapses both into one monolithic number that silently reads as zero
    for whichever layer a single-backend tracker (`track_torch_memory`
    alone) cannot see.

    Attributes:
        numpy_cpu_bytes: NumPy/CPU-resident footprint, in bytes.
        torch_bytes: PyTorch-native (CPU + CUDA) allocation footprint, in
            bytes (see `track_torch_memory`).
    """

    numpy_cpu_bytes: int
    torch_bytes: int

    @property
    def total_bytes(self) -> int:
        """`numpy_cpu_bytes + torch_bytes`."""
        return self.numpy_cpu_bytes + self.torch_bytes


class _StratifiedMemoryHandle:
    """Mutable receiver for `track_stratified_memory`: the caller registers
    the NumPy arrays a block itself produced via `track_numpy` (only known
    from inside the block), and reads the resulting `.report` afterward
    (only known at block exit)."""

    __slots__ = ("numpy_arrays", "report")

    def __init__(self) -> None:
        self.numpy_arrays: list[np.ndarray] = []
        self.report: StratifiedMemoryReport | None = None

    def track_numpy(self, *arrays: np.ndarray) -> None:
        """Register `arrays` as this block's own synthesized NumPy output, so
        their structural byte count (`numpy_array_bytes`) contributes to
        `.report.numpy_cpu_bytes` once the block exits.

        Args:
            arrays: The arrays to register (e.g. a freshly solved
                controller's ``P_arr``/``K_arr``).
        """
        self.numpy_arrays.extend(arrays)


@contextmanager
def track_stratified_memory() -> Iterator[_StratifiedMemoryHandle]:
    """Stratified memory footprint of a block, split by the heterogeneous
    device/library layer that actually holds the bytes.

    * **NumPy/CPU layer** (``.report.numpy_cpu_bytes``): the structural byte
      count (`numpy_array_bytes`) of whichever arrays the block registers via
      ``handle.track_numpy(...)``. When the block does no detectable PyTorch
      work (``torch_bytes == 0``), this is additionally ``max``-combined with
      the block's own CPU-RSS delta (`current_peak_ram_bytes` sampled at
      entry and exit, clamped to non-negative) -- a same-block companion that
      catches short-lived NumPy temporaries (e.g. matrix-inverse scratch
      space) that never make it into the block's registered *output* arrays.
      The RSS delta is a *process-wide* signal, though, blind to which
      backend actually grew the process's resident set -- folding it in
      whenever the block also does real PyTorch work would misattribute
      PyTorch's own CPU tensor growth to the NumPy layer (observed
      empirically: an RSS delta in the gigabytes for a macro-iteration loop
      whose registered NumPy output was only tens of MB), so it is dropped
      entirely once `torch_bytes` is non-zero, leaving the registered
      arrays' exact structural count as the sole (and still honest, if more
      conservative) NumPy-layer estimate.
    * **PyTorch layer** (``.report.torch_bytes``): the ``max`` of (a)
      `track_torch_memory`'s profiler-based CPU+CUDA op sum, and (b), when a
      CUDA device is present, the CUDA allocator's own peak-allocated delta
      (`reset_gpu_peak_memory`/`peak_gpu_memory_bytes`, i.e. PyTorch's native
      ``torch.cuda.max_memory_allocated`` hook) -- the profiler-based sum
      alone can undercount allocator-level retention that hook reports
      directly.

    A block that is pure NumPy (e.g. a Riccati recursion) therefore reports a
    non-zero `numpy_cpu_bytes` and a zero `torch_bytes`; a block that is pure
    PyTorch reports the reverse -- unlike `track_torch_memory` alone, which
    silently reports zero bytes for the former.

    Yields:
        A `_StratifiedMemoryHandle`; call ``handle.track_numpy(...)`` from
        inside the block to register its own NumPy output, then read
        ``handle.report`` (a `StratifiedMemoryReport`) once the block exits.
    """
    handle = _StratifiedMemoryHandle()
    reset_gpu_peak_memory()
    # The CUDA leg must be a *delta above the block's entry allocation*:
    # reset_peak_memory_stats re-seeds the peak to whatever is currently
    # allocated, and PyTorch keeps e.g. cuBLAS workspaces allocated for the
    # process's lifetime after its first CUDA matmul/linalg call — an
    # absolute peak would attribute that standing baseline to every
    # subsequently measured block (observed: a pure-NumPy block "growing"
    # 32 MiB of torch bytes merely because CUDA mathematics had run earlier
    # in the process).
    gpu_entry_bytes = current_gpu_allocated_bytes()
    rss_before = current_peak_ram_bytes()
    with track_torch_memory() as torch_handle:
        yield handle
    rss_after = current_peak_ram_bytes()
    rss_delta = (
        max(0, rss_after - rss_before)
        if rss_before is not None and rss_after is not None
        else 0
    )
    structural_bytes = numpy_array_bytes(*handle.numpy_arrays)
    cuda_peak_bytes = max(0, (peak_gpu_memory_bytes() or 0) - gpu_entry_bytes)
    torch_bytes = max(torch_handle.bytes, cuda_peak_bytes)
    numpy_cpu_bytes = (
        max(structural_bytes, rss_delta) if torch_bytes == 0 else structural_bytes
    )
    handle.report = StratifiedMemoryReport(
        numpy_cpu_bytes=numpy_cpu_bytes,
        torch_bytes=torch_bytes,
    )


# --- block / function profiling ----------------------------------------------


@dataclass(frozen=True)
class ProfileResult:
    """One measurement of a single profiled block/call.

    Attributes:
        name: The block's identifier (e.g. ``"rollout.forward"``).
        wall_time_s: Elapsed wall-clock time, in seconds.
        cpu_time_s: Elapsed process CPU time, in seconds.
        cuda_time_ms: Elapsed CUDA device time in milliseconds, or ``None`` if
            the block ran without CUDA timing (CPU-only, or profiling disabled).
        peak_ram_bytes: Peak process RSS sampled at block exit, in bytes, or
            ``None`` if not sampled (profiling disabled, or unsupported platform).
        peak_vram_bytes: Peak CUDA memory allocated during the block, in bytes,
            or ``None`` if no CUDA device was used.
        device: ``"cuda"`` if CUDA timing/memory were captured, else ``"cpu"``.
        n_calls: Number of calls this result represents (always 1 for a single
            `profile_block`/`profiled` measurement).
    """

    name: str
    wall_time_s: float
    cpu_time_s: float
    cuda_time_ms: float | None = None
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    device: str = "cpu"
    n_calls: int = 1

    def as_metrics(self, prefix: str = "profile/") -> dict[str, float]:
        """Flatten this result into a scalar-metrics dict suitable for an
        `ExperimentTracker.log_metrics` call.

        Args:
            prefix: Namespace prefix for every metric key.

        Returns:
            A dict with keys like ``f"{prefix}{name}/wall_time_s"``; fields
            that are ``None`` (e.g. no CUDA) are omitted rather than logged as
            ``None``.
        """
        metrics = {
            f"{prefix}{self.name}/wall_time_s": self.wall_time_s,
            f"{prefix}{self.name}/cpu_time_s": self.cpu_time_s,
        }
        if self.cuda_time_ms is not None:
            metrics[f"{prefix}{self.name}/cuda_time_ms"] = self.cuda_time_ms
        if self.peak_ram_bytes is not None:
            metrics[f"{prefix}{self.name}/peak_ram_mb"] = self.peak_ram_bytes / 1e6
        if self.peak_vram_bytes is not None:
            metrics[f"{prefix}{self.name}/peak_vram_mb"] = self.peak_vram_bytes / 1e6
        return metrics


class _ProfileHandle:
    """Mutable receiver so a ``with profile_block(...) as h:`` caller can read
    ``h.result`` after the block (the result is only known at block exit)."""

    __slots__ = ("result",)

    def __init__(self) -> None:
        self.result: ProfileResult | None = None


@contextmanager
def profile_block(name: str, *, device: str | None = None) -> Iterator[_ProfileHandle]:
    """Measure latency (and, when profiling is enabled, memory) of a block.

    When ``PROFILING.enabled`` is False this is a thin wall/CPU-clock pass-through
    (no CUDA sync, no memory sampling); when enabled it additionally samples peak
    RAM and, on CUDA, device time + peak VRAM.

    Args:
        name: Identifier stored on the resulting `ProfileResult.name`.
        device: Fallback device label used when CUDA timing was not active
            (e.g. profiling disabled, or no CUDA device present). Ignored if
            CUDA timing *is* active, in which case ``"cuda"`` is used.

    Yields:
        A `_ProfileHandle` whose ``.result`` is populated (a `ProfileResult`)
        once the ``with`` block exits, including on an exception.
    """
    handle = _ProfileHandle()
    detailed = PROFILING.enabled
    use_cuda = detailed and cuda_available()

    if use_cuda:
        reset_gpu_peak_memory()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start, cpu_start = wall_clock(), cpu_clock()
    try:
        yield handle
    finally:
        wall_elapsed = wall_clock() - wall_start
        cpu_elapsed = cpu_clock() - cpu_start

        cuda_ms: float | None = None
        peak_vram: int | None = None
        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            cuda_ms = start_event.elapsed_time(end_event)
            peak_vram = int(torch.cuda.max_memory_allocated())

        peak_ram = current_peak_ram_bytes() if detailed else None
        handle.result = ProfileResult(
            name=name,
            wall_time_s=wall_elapsed,
            cpu_time_s=cpu_elapsed,
            cuda_time_ms=cuda_ms,
            peak_ram_bytes=peak_ram,
            peak_vram_bytes=peak_vram,
            device="cuda" if use_cuda else (device or "cpu"),
        )


def profiled(name: str | None = None) -> Callable[[F], F]:
    """Wrap a function so each call is timed into :data:`COLLECTOR` -- but only
    while profiling is enabled. The intended targets are epoch/rollout-level
    boundaries (``RolloutModel.forward``, ``StateSpaceSystem.run``), never inner
    loops, so the per-call opt-in check is negligible.

    Args:
        name: Block name to record under; defaults to the wrapped function's
            qualified name.

    Returns:
        A decorator producing a wrapper that behaves identically to the
        original function (same arguments, return value, and exceptions) but
        additionally records a `ProfileResult` into `COLLECTOR` when
        `PROFILING.enabled` is ``True``.
    """

    def decorate(func: F) -> F:
        block_name: str = name or str(getattr(func, "__qualname__", func.__name__))

        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> Any:
            if not PROFILING.enabled:
                return func(*args, **kwargs)
            with profile_block(block_name) as handle:
                result = func(*args, **kwargs)
            if handle.result is not None:
                COLLECTOR.record(handle.result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorate


# --- aggregation / collection ------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """Summary statistics over multiple `ProfileResult` samples for one named
    block.

    Attributes:
        name: The block's identifier.
        count: Number of samples aggregated.
        wall_total_s: Sum of `ProfileResult.wall_time_s` over all samples.
        wall_mean_s: Mean wall time per sample.
        wall_p50_s: Median wall time.
        wall_p95_s: 95th-percentile wall time.
        cpu_total_s: Sum of `ProfileResult.cpu_time_s` over all samples.
        cuda_total_ms: Sum of CUDA device time, or ``None`` if no sample
            recorded CUDA timing.
        peak_ram_bytes: Maximum `ProfileResult.peak_ram_bytes` across samples,
            or ``None`` if no sample recorded it.
        peak_vram_bytes: Maximum `ProfileResult.peak_vram_bytes` across
            samples, or ``None`` if no sample recorded it.
    """

    name: str
    count: int
    wall_total_s: float
    wall_mean_s: float
    wall_p50_s: float
    wall_p95_s: float
    cpu_total_s: float
    cuda_total_ms: float | None
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None

    def as_metrics(self, prefix: str = "profile/") -> dict[str, float]:
        """Flatten this aggregate into a scalar-metrics dict.

        Args:
            prefix: Namespace prefix for every metric key.

        Returns:
            A dict with keys like ``f"{prefix}{name}/wall_mean_s"``; fields
            that are ``None`` are omitted.
        """
        base = f"{prefix}{self.name}/"
        metrics = {
            f"{base}count": float(self.count),
            f"{base}wall_total_s": self.wall_total_s,
            f"{base}wall_mean_s": self.wall_mean_s,
            f"{base}wall_p50_s": self.wall_p50_s,
            f"{base}wall_p95_s": self.wall_p95_s,
            f"{base}cpu_total_s": self.cpu_total_s,
        }
        if self.cuda_total_ms is not None:
            metrics[f"{base}cuda_total_ms"] = self.cuda_total_ms
        if self.peak_ram_bytes is not None:
            metrics[f"{base}peak_ram_mb"] = self.peak_ram_bytes / 1e6
        if self.peak_vram_bytes is not None:
            metrics[f"{base}peak_vram_mb"] = self.peak_vram_bytes / 1e6
        return metrics


def aggregate_results(name: str, results: list[ProfileResult]) -> Aggregate:
    """Summarize a non-empty list of same-block `ProfileResult` samples.

    Args:
        name: Block identifier to attach to the returned `Aggregate`.
        results: A non-empty list of `ProfileResult`, typically all sharing
            the same `.name` (not itself enforced).

    Returns:
        An `Aggregate` with wall/CPU totals and percentiles, and CUDA/RAM/VRAM
        summaries where any sample recorded them. Passing an empty `results`
        yields ``nan`` wall statistics (numpy's empty-mean/percentile behavior)
        rather than raising; callers (e.g. `ProfilingCollector.aggregate`)
        avoid this by only aggregating non-empty result lists.
    """
    walls = np.array([r.wall_time_s for r in results], dtype=float)
    cuda_values = [r.cuda_time_ms for r in results if r.cuda_time_ms is not None]
    ram_values = [r.peak_ram_bytes for r in results if r.peak_ram_bytes is not None]
    vram_values = [r.peak_vram_bytes for r in results if r.peak_vram_bytes is not None]
    return Aggregate(
        name=name,
        count=len(results),
        wall_total_s=float(walls.sum()),
        wall_mean_s=float(walls.mean()),
        wall_p50_s=float(np.percentile(walls, 50)),
        wall_p95_s=float(np.percentile(walls, 95)),
        cpu_total_s=float(sum(r.cpu_time_s for r in results)),
        cuda_total_ms=float(sum(cuda_values)) if cuda_values else None,
        peak_ram_bytes=max(ram_values) if ram_values else None,
        peak_vram_bytes=max(vram_values) if vram_values else None,
    )


class ProfilingCollector:
    """Thread-safe accumulation of :class:`ProfileResult` keyed by block name,
    drained and persisted by ``ProfilingCallback`` at train end."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[str, list[ProfileResult]] = {}

    def record(self, result: ProfileResult) -> None:
        """Append `result` to the list for its block name.

        Args:
            result: The measurement to record; grouped by ``result.name``.
        """
        with self._lock:
            self._results.setdefault(result.name, []).append(result)

    def aggregate(self) -> dict[str, Aggregate]:
        """Summarize every recorded block.

        Returns:
            A dict from block name to `Aggregate`, one entry per distinct
            name recorded via `record` since the last `reset` (empty if
            nothing has been recorded).
        """
        with self._lock:
            return {
                name: aggregate_results(name, results)
                for name, results in self._results.items()
                if results
            }

    def reset(self) -> None:
        """Discard all recorded results (e.g. at the start of a new run)."""
        with self._lock:
            self._results.clear()


COLLECTOR = ProfilingCollector()
