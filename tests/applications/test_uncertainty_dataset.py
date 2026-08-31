"""Phase 1 acceptance tests for `applications.uncertainty.dataset`
(docs/planning/03_studies/nb07_robust_training/robust_training_under_model_mismatch.md Sec 4.3/5.1/11): exactly
`n_trajectories` distinct trajectories are ever served; every draw is a
(tiled) permutation, never fresh i.i.d. data; and the mini-batch stream is
reproducible given `shuffle_seed`.
"""

import numpy as np
import pytest
import torch

from mbl.applications.uncertainty.dataset import FiniteTrajectoryDatasetSpec
from mbl.applications.uncertainty.noise import ExoticBatchSpec, NoiseFamily
from mbl.core.runtime import Backend

_N = 3
_HORIZON = 5


def _source(n_trajectories_hint_batch_size: int = 999) -> ExoticBatchSpec:
    # source.batch_size is deliberately irrelevant to the materialized set
    # size (n_trajectories overrides it in FiniteTrajectoryDatasetSpec.build).
    return ExoticBatchSpec(
        state_dim=_N,
        horizon=_HORIZON,
        batch_size=n_trajectories_hint_batch_size,
        seed=7,
        process_noise_std=0.5,
        family=NoiseFamily.GAUSSIAN,
    )


class TestFiniteSetSize:
    @pytest.mark.parametrize("backend", [Backend.NUMPY, Backend.TORCH])
    def test_exactly_n_trajectories_distinct_x0(self, backend: Backend) -> None:
        n_traj = 16
        spec = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=1
        )
        sample, _ = spec.build(backend, batch_size=n_traj)
        x0, _, _ = sample()
        assert x0.shape[0] == n_traj

    def test_large_batch_never_exceeds_the_finite_set(self) -> None:
        n_traj = 4
        spec = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=3
        )
        sample, _ = spec.build(Backend.NUMPY, batch_size=64)
        x0, _, _ = sample()
        assert x0.shape[0] == 64
        # Every row of the 64-batch must equal one of the n_traj distinct rows.
        unique_rows = np.unique(x0, axis=0)
        assert unique_rows.shape[0] <= n_traj

    def test_every_trajectory_seen_once_per_full_pass(self) -> None:
        n_traj = 8
        spec = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=5
        )
        sample, _ = spec.build(Backend.NUMPY, batch_size=n_traj)
        first_pass, _, _ = sample()
        second_pass, _, _ = sample()
        # A full pass is a permutation of the same n_traj distinct rows both
        # times -- as sets, the two passes are identical.
        assert np.array_equal(
            np.unique(first_pass, axis=0), np.unique(second_pass, axis=0)
        )


class TestNotFreshData:
    def test_repeated_full_passes_never_introduce_new_trajectories(self) -> None:
        n_traj = 5
        spec = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=9
        )
        sample, _ = spec.build(Backend.NUMPY, batch_size=n_traj)
        first, _, _ = sample()
        seen = {tuple(row) for row in first}
        for _ in range(10):
            batch, _, _ = sample()
            seen |= {tuple(row) for row in batch}
        assert len(seen) == n_traj


class TestReproducibility:
    def test_same_shuffle_seed_reproduces_the_stream(self) -> None:
        n_traj = 6
        spec_a = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=11
        )
        spec_b = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=11
        )
        sample_a, _ = spec_a.build(Backend.NUMPY, batch_size=3)
        sample_b, _ = spec_b.build(Backend.NUMPY, batch_size=3)
        for _ in range(5):
            a, _, _ = sample_a()
            b, _, _ = sample_b()
            np.testing.assert_array_equal(a, b)

    def test_different_shuffle_seed_changes_the_order(self) -> None:
        n_traj = 20
        spec_a = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=1
        )
        spec_b = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=n_traj, shuffle_seed=2
        )
        sample_a, _ = spec_a.build(Backend.NUMPY, batch_size=n_traj)
        sample_b, _ = spec_b.build(Backend.NUMPY, batch_size=n_traj)
        a, _, _ = sample_a()
        b, _, _ = sample_b()
        assert not np.array_equal(a, b)


class TestTorchIndexing:
    def test_torch_backend_preserves_dtype_and_device(self) -> None:
        spec = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=8, shuffle_seed=2
        )
        sample, _ = spec.build(Backend.TORCH, torch_dtype=torch.float64, batch_size=4)
        x0, w, v = sample()
        assert isinstance(x0, torch.Tensor)
        assert x0.dtype == torch.float64
        assert x0.shape == (4, _N)
        assert w.shape == (4, _HORIZON, _N)


class TestSignature:
    def test_n_trajectories_participates(self) -> None:
        sig_16 = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=16, shuffle_seed=0
        ).get_signature()
        sig_32 = FiniteTrajectoryDatasetSpec(
            source=_source(), n_trajectories=32, shuffle_seed=0
        ).get_signature()
        assert sig_16 != sig_32
        assert sig_16["source"]["type"] == "ExoticBatchSpec"

    def test_rejects_nonpositive_n_trajectories(self) -> None:
        with pytest.raises(ValueError):
            FiniteTrajectoryDatasetSpec(
                source=_source(), n_trajectories=0, shuffle_seed=0
            )
