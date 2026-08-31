"""Performance-profiling subsystem: per-block/per-epoch wall- & CPU-time and
peak RAM/VRAM measurement, aggregation, and a process-global collector.

Promoted out of ``core.utils`` (where it was a large, fully self-contained
subsystem sharing nothing with the other utilities) into its own package.
The public surface is re-exported here so callers import from
``core.profiling`` directly (e.g. ``from ...core.profiling import profiled``).
"""

from .profiling import (
    COLLECTOR,
    PROFILING,
    Aggregate,
    ProfileResult,
    ProfilingCollector,
    ProfilingConfig,
    StratifiedMemoryReport,
    aggregate_results,
    cpu_clock,
    cuda_available,
    current_peak_ram_bytes,
    numpy_array_bytes,
    peak_gpu_memory_bytes,
    profile_block,
    profiled,
    reset_gpu_peak_memory,
    track_stratified_memory,
    track_torch_memory,
    wall_clock,
)

__all__ = [
    "COLLECTOR",
    "PROFILING",
    "Aggregate",
    "ProfileResult",
    "ProfilingCollector",
    "ProfilingConfig",
    "StratifiedMemoryReport",
    "aggregate_results",
    "cpu_clock",
    "cuda_available",
    "current_peak_ram_bytes",
    "numpy_array_bytes",
    "peak_gpu_memory_bytes",
    "profile_block",
    "profiled",
    "reset_gpu_peak_memory",
    "track_stratified_memory",
    "track_torch_memory",
    "wall_clock",
]
