"""Figure 4's cost-grid benchmark (the campaign plan's Phase F).

Each controller is measured as though it had its own dedicated machine, one
per phase — {time, memory} × {offline, online} — in a FRESH interpreter per
cell (`python -m mbl.benchmark.cell`), so no import, allocator, BLAS plan or
Riccati result is ever shared between two controllers' numbers. The offline
cell re-performs the computation to time it and publishes nothing; the online
cell loads the stored frozen artifact and never trains.

Tier 8: this package may import spec, store, runner, applications and
experiments — it is a consumer of the whole stack, like the notebooks.
"""

from .cells import measure_offline, measure_online

__all__ = ["measure_offline", "measure_online"]
