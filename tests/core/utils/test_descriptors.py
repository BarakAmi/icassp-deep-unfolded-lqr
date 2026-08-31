from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pytest

from mbl.core.utils.descriptors import (
    BoundedFloat,
    PositiveInt,
    TimeStackedPSD,
    validated_dataclass,
)
from mbl.core.utils.validation import validate_square_matrix


class _Host:
    Q = TimeStackedPSD()
    count = PositiveInt()
    rate = BoundedFloat(low=0.0, high=1.0, inclusive=False)

    def __init__(self, Q: np.ndarray, count: int, rate: float) -> None:
        self.Q = Q
        self.count = count
        self.rate = rate


def test_descriptors_accept_and_store_valid_values() -> None:
    host = _Host(Q=np.eye(2), count=3, rate=0.5)
    assert np.array_equal(host.Q, np.eye(2))
    assert host.count == 3
    assert host.rate == 0.5


def test_psd_descriptor_rejects_invalid_matrix_on_assignment() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        _Host(Q=np.array([[1.0, 2.0], [3.0, 4.0]]), count=1, rate=0.5)


def test_positive_int_descriptor_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="count"):
        _Host(Q=np.eye(2), count=0, rate=0.5)


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.5, 2.0])
def test_bounded_float_descriptor_rejects_out_of_range(rate: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        _Host(Q=np.eye(2), count=1, rate=rate)


def test_descriptor_is_write_once_immutable() -> None:
    host = _Host(Q=np.eye(2), count=1, rate=0.5)
    with pytest.raises(AttributeError, match="validated-immutable"):
        host.Q = np.eye(2)


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@validated_dataclass
@dataclass(frozen=True)
class _Config:
    width: Annotated[int, _positive]
    kernel: Annotated[np.ndarray, validate_square_matrix]


def test_validated_dataclass_runs_annotated_validators() -> None:
    cfg = _Config(width=4, kernel=np.eye(3))
    assert cfg.width == 4

    with pytest.raises(ValueError, match="width"):
        _Config(width=0, kernel=np.eye(3))
    with pytest.raises(ValueError, match="square"):
        _Config(width=4, kernel=np.zeros((2, 3)))


@validated_dataclass
@dataclass(frozen=True)
class _ConfigWithPostInit:
    value: Annotated[int, _positive]
    _checked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "_checked", True)


def test_validated_dataclass_chains_existing_post_init() -> None:
    cfg = _ConfigWithPostInit(value=2)
    assert cfg._checked is True  # original __post_init__ still ran
    with pytest.raises(ValueError, match="value"):
        _ConfigWithPostInit(value=-1)
