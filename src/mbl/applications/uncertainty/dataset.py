"""The finite, replayed training-trajectory dataset
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.3/5.1) -- the fix for the
plan's single most important correction: `GaussianBatchSpec`'s (and
`ExoticBatchSpec`'s) sampler closure draws FRESH i.i.d. data on every call
(``applications.factories.GaussianBatchSpec.build``'s ``sample()`` closure),
so the effective training-set size is unboundedly large and there is no
finite sample to overfit -- a curve produced by varying only ``batch_size``
would measure gradient-noise level, not sample complexity.

`FiniteTrajectoryDatasetSpec` fixes this: it materializes exactly
``n_trajectories`` trajectories ONCE, then serves reshuffled mini-batches
from that fixed set on every subsequent call -- so a model trained against
it can genuinely overfit, and the sample-complexity axis measures what it
claims to.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .noise import ExoticBatchSpec
from ...core.runtime import Backend
from ...core.utils import ensure_positive_integer
from ...engine.strategy import BatchSampler
from ...models.samplers import Distribution


class _CyclicIndexSampler:
    """Draws index batches from ``range(n)`` without replacement WITHIN a
    pass: each full pass through the shuffled indices is exhausted before a
    fresh permutation is drawn, so every trajectory is seen once per ``n``
    draws, not resampled with a biased frequency. A requested count larger
    than ``n`` spans multiple passes (i.e. the batch tiles more than one
    shuffled epoch), which is exactly the correct semantics when the
    requested training batch size exceeds the finite dataset size (the small
    end of the sample-complexity ladder, e.g. ``n_trajectories=16`` under a
    ``batch_size=256`` training plan)."""

    def __init__(self, n: int, seed: int) -> None:
        self._n = n
        self._rng = np.random.default_rng(seed)
        self._buffer = np.empty(0, dtype=np.int64)

    def draw(self, count: int) -> np.ndarray:
        """Return `count` indices into ``range(self._n)``, refilling the
        internal shuffled buffer as needed.

        Args:
            count: Number of indices to draw.

        Returns:
            An ``int64`` array of shape ``(count,)``.
        """
        while self._buffer.size < count:
            self._buffer = np.concatenate(
                (self._buffer, self._rng.permutation(self._n))
            )
        batch, self._buffer = self._buffer[:count], self._buffer[count:]
        return batch


def _index(data: Any, indices: np.ndarray) -> Any:
    """Index `data` (NumPy array or torch Tensor) along its leading axis by
    `indices`, preserving the input's own array type."""
    if isinstance(data, torch.Tensor):
        return data[torch.as_tensor(indices, dtype=torch.long, device=data.device)]
    return data[indices]


@dataclass(frozen=True)
class FiniteTrajectoryDatasetSpec:
    """A `BatchSpec`-compatible specification whose `build` materializes
    exactly `n_trajectories` trajectories once from `source`, then serves
    reshuffled mini-batches of them on every subsequent sampler call.

    Attributes:
        source: The noise law (and dimensions/seed) the finite trajectory
            set is drawn from -- its own ``batch_size`` is ignored; the
            materialized set size is always `n_trajectories`.
        n_trajectories: The fixed training-set size (the sample-complexity
            axis's x-coordinate).
        shuffle_seed: Seed of the mini-batch reshuffling stream -- kept
            distinct from `source.seed` (which seeds which trajectories
            exist) so the two concerns (WHICH trajectories, and in WHAT
            order they are replayed) vary independently.
    """

    source: ExoticBatchSpec
    n_trajectories: int
    shuffle_seed: int

    def __post_init__(self) -> None:
        ensure_positive_integer(self.n_trajectories, "n_trajectories")

    @property
    def batch_size(self) -> int:
        """`BatchSpec` protocol conformance: the mini-batch size `build`
        serves absent an explicit override -- proxies `source.batch_size`
        (never `n_trajectories`, which is the materialized SET size, not the
        per-call serving size)."""
        return self.source.batch_size

    def build(
        self,
        backend: Backend,
        *,
        torch_dtype: torch.dtype = torch.float64,
        torch_device: torch.device | None = None,
        batch_size: int | None = None,
        horizon: int | None = None,
    ) -> tuple[BatchSampler, dict[str, Distribution]]:
        """Materialize the finite trajectory set once, then return a sampler
        that serves reshuffled mini-batches from it.

        Args:
            backend: Which array backend the materialized set is authored on.
            torch_dtype: dtype of torch-backed batches (ignored for NumPy).
            torch_device: Residency of torch-backed batches (ignored for
                NumPy).
            batch_size: The mini-batch size each subsequent `sample()` call
                returns; defaults to `source.batch_size`. May exceed
                `n_trajectories` (see `_CyclicIndexSampler`).
            horizon: Optional per-family override of `source.horizon`.

        Returns:
            ``(sampler, distributions)``: `sampler()` returns a mini-batch
            drawn (without full-pass replacement) from the fixed set;
            `distributions` is `source`'s own declared distributions (the
            single-source-of-truth object for what was actually drawn to
            build the fixed set).
        """
        full_sampler, distributions = self.source.build(
            backend,
            torch_dtype=torch_dtype,
            torch_device=torch_device,
            batch_size=self.n_trajectories,
            horizon=horizon,
        )
        x0_full, w_full, v_full = full_sampler()
        effective_batch = batch_size or self.source.batch_size
        index_sampler = _CyclicIndexSampler(self.n_trajectories, self.shuffle_seed)

        def sample() -> tuple[Any, Any, Any]:
            indices = index_sampler.draw(effective_batch)
            return (
                _index(x0_full, indices),
                _index(w_full, indices),
                _index(v_full, indices),
            )

        return sample, distributions

    def get_signature(self) -> dict[str, Any]:
        """`Signable` member: the full dataset specification -- every rung
        of the sample-complexity ladder is its own cache identity because
        `n_trajectories` participates here.

        Returns:
            ``{"type": "FiniteTrajectoryDatasetSpec", "source": {...},
            "n_trajectories":, "shuffle_seed":}``.
        """
        return {
            "type": type(self).__name__,
            "source": self.source.get_signature(),
            "n_trajectories": self.n_trajectories,
            "shuffle_seed": self.shuffle_seed,
        }
