from typing import Any, cast

import numpy as np

from .system import BatchedStateSpaceVector
from .state_space_system import StateSpaceSystem, StateSpaceModelDimensions
from ..utils import match_array_type
from ..utils.signing import hash_array


class TimeSeriesMatrix:
    """Compact storage for a (possibly time-varying) sequence of matrices, e.g.
    a `LinearSystem`'s ``A_t``/``B_t``/``C_t``.

    Accepts either a single 2D matrix (time-invariant) or a 3D time-stack
    ``(horizon, *matrix_shape)``, and stores only the *unique* matrices plus a
    compact **run-length-encoded schedule** mapping each time step to one of
    them. Four regimes fall out of the same representation:

    1. **Time-invariant** -- a 2D input, or a 3D stack whose slices are all
       identical: stored as a single 2D matrix, no schedule.
    2. **Periodic (single occurrence)** -- a short repeating cycle such as
       ``[M0, M1, M0, M1, ...]``: the cycle is first collapsed to its shortest
       period (``[M0, M1]``) before run-length encoding.
    3. **Run-length / piecewise-constant (NEW)** -- consecutive repeats such as
       ``[M0, M0, M0, M0, M1, M1, M1]``: stored as two runs, not seven entries.
    4. **Fully time-varying** -- no exploitable pattern: run-length encoding
       degrades gracefully to one length-1 run per step (no worse than a raw
       index array).

    In every non-time-invariant regime, indexing beyond the stored schedule
    length **wraps around via modulo arithmetic** (`__getitem__`): the whole
    (period- and run-length-compressed) schedule replays, so a rollout horizon
    may exceed the length the matrix was defined over.
    """

    def __init__(self, name: str, array: np.ndarray) -> None:
        """
        Args:
            name: Human-readable identifier used in error messages (e.g. ``"A_t"``).
            array: A single 2D matrix (time-invariant), or a 3D time-stack
                ``(horizon, *matrix_shape)``.

        Raises:
            ValueError: If `array` is not 2D or 3D.
        """
        self.name = name
        # `array`: the unique matrices (2D when time-invariant, else a 3D stack).
        # `_run_values`/`_run_ends`: the run-length-encoded schedule -- for run i,
        # `_run_values[i]` indexes into `array` and `_run_ends[i]` is that run's
        # cumulative (exclusive) end step. `_cycle_length == _run_ends[-1]` is the
        # schedule length used for modulo wraparound. All three are ``None`` iff
        # time-invariant.
        self.array: np.ndarray
        self._run_values: np.ndarray | None
        self._run_ends: np.ndarray | None
        self._cycle_length: int | None
        self._decompose(array)

    def _decompose(self, array: np.ndarray) -> None:
        """Route by rank: 2D is time-invariant; 3D is unique-matrix extraction
        followed by period + run-length compression of the schedule."""
        if array.ndim == 2:
            self._set_time_invariant(array)
            return
        if array.ndim == 3:
            self._prepare_time_stacked(array)
            return
        raise ValueError(
            f"{self.name} must be either 2D or 3D, but got shape {array.shape}."
        )

    def _set_time_invariant(self, matrix: np.ndarray) -> None:
        self.array = matrix
        self._run_values = None
        self._run_ends = None
        self._cycle_length = None

    def _prepare_time_stacked(self, array: np.ndarray) -> None:
        """Extract the unique matrices and build the compressed schedule."""
        unique_matrices, inverse_indices = np.unique(array, axis=0, return_inverse=True)
        inverse_indices = np.asarray(inverse_indices).ravel()

        if unique_matrices.shape[0] == 1:
            # All slices identical: collapse to a single 2D matrix.
            self._set_time_invariant(unique_matrices[0])
            return

        # Collapse a short repeating cycle to its shortest period first, then
        # run-length encode -- period detection wins for short cycles (which
        # run-length alone would not compress), run-length wins for long runs.
        period = self._detect_period(inverse_indices)
        schedule = period if period is not None else inverse_indices

        self.array = unique_matrices
        self._run_values, self._run_ends = self._run_length_encode(schedule)
        self._cycle_length = int(self._run_ends[-1])

    @staticmethod
    def _detect_period(inverse_indices: np.ndarray) -> np.ndarray | None:
        """Return the shortest prefix of ``inverse_indices`` whose repetition
        reproduces the full sequence, or ``None`` if it is not periodic within
        half its length (e.g. ``[0,1,0,1,0,1] -> [0,1]``)."""
        doubled = np.concatenate((inverse_indices, inverse_indices))
        for length in range(1, inverse_indices.size // 2 + 1):
            if np.array_equal(
                doubled[length : length + inverse_indices.size], inverse_indices
            ):
                return inverse_indices[:length]
        return None

    @staticmethod
    def _run_length_encode(
        sequence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run-length encode a 1D index sequence into ``(values, ends)``, where
        ``values[i]`` is the (unique-matrix) index of run ``i`` and ``ends[i]``
        is that run's cumulative exclusive end step (e.g.
        ``[0,0,0,0,1,1,1] -> values=[0,1], ends=[4,7]``). ``ends`` is
        non-decreasing, so `__getitem__` can locate a step via binary search."""
        change_points = np.flatnonzero(np.diff(sequence)) + 1
        ends = np.append(change_points, sequence.size)
        values = sequence[np.append(0, change_points)]
        return values, ends

    def is_time_varying(self) -> bool:
        """Return whether this matrix genuinely varies over time.

        Returns:
            ``False`` if the constructor input was 2D or a 3D stack that
            collapsed to a single matrix; ``True`` otherwise.
        """
        return not self.is_time_invariant()

    def is_time_invariant(self) -> bool:
        """Return whether this matrix is time-invariant.

        Returns:
            ``True`` if the constructor input was 2D or a 3D stack that
            collapsed to a single matrix; ``False`` otherwise.
        """
        return self._run_values is None

    def __getitem__(self, t: int) -> np.ndarray:
        """Return the matrix slice active at time step `t`.

        Args:
            t: The (non-negative) time step. For any time-varying regime,
                `t` wraps via modulo arithmetic over the compressed schedule
                length -- indexing beyond the stored horizon replays the
                period/run-length schedule from the start.

        Returns:
            The matrix active at time `t`, shape ``(rows, cols)``.
        """
        if self.is_time_invariant():
            return self.array
        # Non-time-invariant: the schedule triple is set (see _decompose).
        assert self._cycle_length is not None
        assert self._run_ends is not None and self._run_values is not None
        step = t % self._cycle_length
        run = int(np.searchsorted(self._run_ends, step, side="right"))
        return cast(np.ndarray, self.array[self._run_values[run]])

    def content_hash(self) -> str:
        """A content hash covering BOTH the unique matrices AND their temporal
        schedule, so two matrices sharing the same unique set but arranged
        differently in time hash *differently* (a plain hash of the unique
        matrices alone cannot tell them apart, yet they induce different
        dynamics). A time-invariant matrix hashes exactly as
        ``signing.hash_array`` would, so its signature is unchanged.

        Returns:
            A ``"sha256:..."`` content-hash string.
        """
        matrices_hash = hash_array(self.array)
        if self.is_time_invariant():
            return matrices_hash
        assert self._run_values is not None and self._run_ends is not None
        schedule: np.ndarray = np.concatenate((self._run_values, self._run_ends))
        combined = f"{matrices_hash}|{hash_array(schedule)}".encode()
        return hash_array(np.frombuffer(combined, dtype=np.uint8))

    def __repr__(self) -> str:
        """Human-readable dump of the stored matrix/matrices and, if
        time-varying, the run-length schedule."""
        if self.is_time_invariant():
            return f"{self.name} (2D):\n{self.array}"
        return (
            f"{self.name} (3D):\n{self.array}\n"
            f"Run values:\n{self._run_values}\nRun ends:\n{self._run_ends}"
        )


class LinearSystem(StateSpaceSystem):
    """A special case of StateSpaceModel where the dynamics and observation maps are linear functions of the state, control, and noise.
    
    The linear state-space model is defined by sequences of matrices A_t, B_t, C_t that specify the linear dynamics and observation maps:
    
    $$x_{t+1} = A_t x_t + B_t u_t + w_t \\
    y_t = C_t x_t + v_t$$
    
    Attributes:
        A_t: State transition matrices, shape (state_dim, state_dim) or (horizon, state_dim, state_dim)
        B_t: Control input matrices, shape (state_dim, control_dim) or (horizon, state_dim, control_dim)
        C_t: Observation matrices for the state, shape (observation_dim, state_dim) or (horizon, observation_dim, state_dim)
        dimensions: An object that specifies the dimensions of the state, control, process noise, observation, and measurement noise vectors.
    """

    def __init__(self, A_t: np.ndarray, B_t: np.ndarray, C_t: np.ndarray) -> None:
        """
        Args:
            A_t: State transition matrices, shape ``(n, n)`` or ``(horizon, n, n)``.
            B_t: Control input matrices, shape ``(n, m)`` or ``(horizon, n, m)``.
            C_t: Observation matrices, shape ``(p, n)`` or ``(horizon, p, n)``.
                Dimensions ``n``/``m``/``p`` are inferred from the trailing
                shapes of `A_t`/`B_t`/`C_t` respectively.

        Raises:
            ValueError: If any of `A_t`/`B_t`/`C_t` is not 2D or 3D (via
                `TimeSeriesMatrix`), or the three matrices are mutually
                dimensionally incoherent -- ``A_t`` not square, ``B_t``'s row
                count not equal to ``A_t``'s state dimension, or ``C_t``'s
                column count not equal to ``A_t``'s state dimension (a
                fail-fast cross-matrix check: an incoherent triple would
                otherwise only surface deep inside a rollout's matmul).
        """
        self._ensure_dimensional_coherence(A_t, B_t, C_t)
        dimensions = StateSpaceModelDimensions(
            state_dim=A_t.shape[-1],
            control_dim=B_t.shape[-1],
            observation_dim=C_t.shape[-2],
            process_noise_dim=A_t.shape[-1],
            measurement_noise_dim=C_t.shape[-2],
        )
        self.A_t = TimeSeriesMatrix("A_t", A_t)
        self.B_t = TimeSeriesMatrix("B_t", B_t)
        self.C_t = TimeSeriesMatrix("C_t", C_t)
        super().__init__(
            state_transition_map=self._state_transition_map,
            observation_map=self._observation_map,
            dimensions=dimensions,
        )

    @staticmethod
    def _ensure_dimensional_coherence(
        A_t: np.ndarray, B_t: np.ndarray, C_t: np.ndarray
    ) -> None:
        """Fail-fast on a mutually-incoherent ``(A_t, B_t, C_t)`` triple, using
        only their trailing 2D shapes (so time-invariant 2D and time-stacked
        3D inputs are validated identically). The state dimension ``n`` is
        taken from ``A_t``'s trailing columns; ``B_t`` must have ``n`` rows and
        ``C_t`` must have ``n`` columns.

        Raises:
            ValueError: If ``A_t`` is not square, ``B_t``'s row count differs
                from ``n``, or ``C_t``'s column count differs from ``n``.
        """
        n = A_t.shape[-1]
        if A_t.shape[-2] != n:
            raise ValueError(
                f"A_t must be square in its trailing dims, got {A_t.shape[-2:]}."
            )
        if B_t.shape[-2] != n:
            raise ValueError(
                f"B_t's trailing row count {B_t.shape[-2]} must equal A_t's "
                f"state dimension {n}, got B_t trailing shape {B_t.shape[-2:]}."
            )
        if C_t.shape[-1] != n:
            raise ValueError(
                f"C_t's trailing column count {C_t.shape[-1]} must equal A_t's "
                f"state dimension {n}, got C_t trailing shape {C_t.shape[-2:]}."
            )

    def _state_transition_map(
        self,
        t: int,
        x: BatchedStateSpaceVector,
        u: BatchedStateSpaceVector,
        w: BatchedStateSpaceVector,
    ) -> BatchedStateSpaceVector:
        """Linear dynamics: ``x_{t+1} = A_t x_t + B_t u_t + w_t``.

        Args:
            t: The current time step (indexes `A_t`/`B_t`).
            x: State ``x_t``, shape ``(batch, n)``.
            u: Control ``u_t``, shape ``(batch, m)``.
            w: Process noise ``w_t``, shape ``(batch, n)``.

        Returns:
            Next state ``x_{t+1}``, shape ``(batch, n)``.
        """
        # A_t/B_t/C_t are always plain numpy (TimeSeriesMatrix is framework-agnostic
        # storage); match_array_type converts them to match x/u when the rollout is
        # torch-based, since torch.Tensor @ numpy.ndarray raises RuntimeError the
        # moment the tensor requires grad (not just a silent, gradient-losing
        # conversion -- an outright crash for any genuinely learnable rollout).
        A_t = match_array_type(x, self.A_t[t])
        B_t = match_array_type(u, self.B_t[t])
        return x @ A_t.T + u @ B_t.T + w

    def _observation_map(
        self,
        t: int,
        x: BatchedStateSpaceVector,
        v: BatchedStateSpaceVector,
    ) -> BatchedStateSpaceVector:
        """Linear observation: ``y_t = C_t x_t + v_t``.

        Args:
            t: The current time step (indexes `C_t`).
            x: State ``x_t``, shape ``(batch, n)``.
            v: Measurement noise ``v_t``, shape ``(batch, p)``.

        Returns:
            Observation ``y_t``, shape ``(batch, p)``.
        """
        C_t = match_array_type(x, self.C_t[t])
        return x @ C_t.T + v

    def is_time_invariant(self) -> bool:
        """Return whether all of `A_t`/`B_t`/`C_t` are time-invariant.

        Returns:
            ``True`` iff none of `A_t`, `B_t`, `C_t` varies over time (see
            `TimeSeriesMatrix.is_time_varying`).
        """
        return all(
            matrix.is_time_invariant() for matrix in [self.A_t, self.B_t, self.C_t]
        )

    @classmethod
    def fully_observable(  # type: ignore[override]  # deliberate factory specialization: matrices in, maps derived
        cls, A_t: np.ndarray, B_t: np.ndarray
    ) -> "LinearSystem":
        """Factory method for creating a fully observable linear system where the observation is simply the state plus measurement noise.

        Args:
            A_t: State transition matrices, shape (horizon, state_dim, state_dim)
            B_t: Control input matrices, shape (horizon, state_dim, control_dim)

        Returns:
            LinearSystem: A fully observable linear system.
        """
        C = np.eye(*A_t.shape[-2:])
        return cls(A_t, B_t, C)

    @classmethod
    def from_matrices_and_indices(
        cls,
        A_t_with_indices: tuple[np.ndarray, np.ndarray],
        B_t_with_indices: tuple[np.ndarray, np.ndarray],
        C_t_with_indices: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "LinearSystem":
        """Factory method for creating a LinearSystem from pre-expanded
        ``(unique_matrices, time_indices)`` pairs, e.g. as recovered from a
        previously constructed `TimeSeriesMatrix` (``unique_matrices[time_indices]``
        reconstructs the full per-step stack).

        Args:
            A_t_with_indices: ``(unique_A_matrices, time_indices)``; the full
                ``A_t`` stack is reconstructed as ``unique_A_matrices[time_indices]``.
            B_t_with_indices: Same convention as `A_t_with_indices`, for ``B_t``.
            C_t_with_indices: Same convention, for ``C_t``. If ``None``, the
                system is built fully observable (see `fully_observable`)
                instead of using an explicit ``C_t``.

        Returns:
            A `LinearSystem` built from the reconstructed `A_t`/`B_t`/`C_t`.
        """
        A_t = A_t_with_indices[0][A_t_with_indices[1]]
        B_t = B_t_with_indices[0][B_t_with_indices[1]]
        if C_t_with_indices is not None:
            C_t = C_t_with_indices[0][C_t_with_indices[1]]
            return cls(A_t, B_t, C_t)
        return cls.fully_observable(A_t, B_t)

    def get_signature(self) -> dict[str, Any]:
        """`Signable` override: dimensions + time-invariance + content hashes
        of each matrix's unique slices **and its temporal schedule**.

        Uses `TimeSeriesMatrix.content_hash` rather than hashing the raw
        `.array` (unique slices) alone: the latter would give two systems with
        the same unique matrices in *different* time orderings an identical
        signature despite inducing different dynamics. A time-invariant matrix
        still hashes exactly as before (once, not once per horizon step), so
        existing time-invariant signatures are unchanged.

        Returns:
            ``{"type": "LinearSystem", **dimensions fields, "is_time_invariant":
            bool, "A_hash":, "B_hash":, "C_hash":}``.
        """
        return super().get_signature() | {
            "is_time_invariant": self.is_time_invariant(),
            "A_hash": self.A_t.content_hash(),
            "B_hash": self.B_t.content_hash(),
            "C_hash": self.C_t.content_hash(),
        }
