"""`GridProjector`: the batched meshgrid engine mapping a `SliceSpec` plus a
fixed reference control backdrop through a `CostOracle` to a `CostField` --
the generalization of `workbench.analysis.compute_loss_grid` /
`workbench.analysis.compute_stage_cost_grid` from a scalar-per-point
Python loop to one (chunked) vectorized oracle call, so an arbitrarily
expensive oracle (a rollout under `torch.no_grad()`) never pays a
per-grid-point Python dispatch cost.
"""

import numpy as np
import torch

from .types import CostField, CostOracle, SliceSpec, ensure_slice_matches_control
from ...core.utils import ensure_positive_integer, to_numpy

#: Balances peak memory (O(chunk * T * n)) against Python-call overhead;
#: comfortably covers the default 2D grid (G=100 -> 10_000 points) in three
#: chunks and the default 3D grid (G=40 -> 64_000 points) in sixteen.
_DEFAULT_CHUNK_SIZE = 4096


class GridProjector:
    """Projects the cost landscape onto a 2D or 3D slice of one timestep's
    control components, holding every other ``(t, j)`` entry fixed at a
    reference backdrop (Phase 2B plan, Part 3)."""

    def __init__(self, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> None:
        """
        Args:
            chunk_size: maximum number of grid points evaluated per
                `cost_oracle` call; the grid is evaluated in row chunks of
                this size, transparently to the oracle.
        """
        ensure_positive_integer(chunk_size, "chunk_size")
        self.chunk_size = chunk_size

    def project(
        self,
        reference_U: np.ndarray | torch.Tensor,
        cost_oracle: CostOracle,
        spec: SliceSpec,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> CostField:
        """Evaluate `cost_oracle` over every node of `spec`'s coordinate
        grid, varying only `spec.components` of `reference_U[spec.timestep]`
        and holding every other control entry fixed.

        Args:
            reference_U: the fixed backdrop control sequence, shape
                ``(T, m)`` (use `types.normalize_control` first for a
                batched input).
            cost_oracle: scores a ``(P, T, m)`` batch to a ``(P,)`` cost
                vector (see `types.CostOracle`).
            spec: which timestep/components/ranges to project.
            dtype: dtype grid batches are cast to before calling
                `cost_oracle`.
            device: device grid batches are cast to before calling
                `cost_oracle`.

        Returns:
            The evaluated `CostField`.

        Raises:
            ValueError: If `reference_U` is not 2D, or `spec` doesn't agree
                with `reference_U`'s ``(T, m)`` (via
                `types.ensure_slice_matches_control`).
        """
        reference = to_numpy(reference_U)
        if reference.ndim != 2:
            raise ValueError(
                "reference_U must be (T, m) -- use types.normalize_control "
                f"first for a batched input; got shape {reference.shape}."
            )
        ensure_slice_matches_control(spec, reference)

        mesh = np.meshgrid(
            *spec.ranges, indexing="xy" if len(spec.components) == 2 else "ij"
        )
        grid_shape = mesh[0].shape
        num_points = mesh[0].size

        # P independent copies of the fixed backdrop; only the varied
        # (t*, component) entries are overwritten below -- every other
        # (t, j) entry of every row stays at reference_U's value.
        U_batch = np.repeat(reference[None, :, :], num_points, axis=0)  # (P, T, m)
        for component, coord_grid in zip(spec.components, mesh):
            U_batch[:, spec.timestep, component] = coord_grid.ravel()

        cost_flat = self._evaluate_in_chunks(U_batch, cost_oracle, dtype, device)
        cost_grid = cost_flat.reshape(grid_shape)

        return CostField(
            grids=tuple(mesh), cost_grid=cost_grid, spec=spec, reference_U=reference
        )

    def _evaluate_in_chunks(
        self,
        U_batch: np.ndarray,
        cost_oracle: CostOracle,
        dtype: torch.dtype,
        device: torch.device,
    ) -> np.ndarray:
        """Call `cost_oracle` on `U_batch` in row chunks of `self.chunk_size`,
        concatenating the per-chunk cost vectors -- transparent to the
        oracle, which never sees the chunking.

        Args:
            U_batch: the full grid batch, shape ``(P, T, m)``.
            cost_oracle: scores a chunk to a ``(chunk,)`` cost vector.
            dtype: dtype each chunk is cast to before the call.
            device: device each chunk is cast to before the call.

        Returns:
            The concatenated per-point cost, shape ``(P,)``.
        """
        num_points = U_batch.shape[0]
        chunks: list[np.ndarray] = []
        for start in range(0, num_points, self.chunk_size):
            chunk = U_batch[start : start + self.chunk_size]
            chunk_tensor = torch.as_tensor(chunk, dtype=dtype, device=device)
            cost_chunk = cost_oracle(chunk_tensor)
            chunks.append(to_numpy(cost_chunk))
        return np.concatenate(chunks, axis=0)
