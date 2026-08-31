import pytest

from mbl.engine.config import TrainingConfig


def test_training_config_constructs_with_valid_values() -> None:
    config = TrainingConfig(batch_size=32, learning_rate=1e-3)
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.optimizer_name == "adam"
    assert config.optimizer_kwargs == {}
    assert config.log_every == 1
    assert config.seed is None


def test_training_config_get_config_returns_dict_of_fields() -> None:
    config = TrainingConfig(batch_size=8, learning_rate=0.01, seed=0)
    assert config.get_config() == {
        "batch_size": 8,
        "learning_rate": 0.01,
        "optimizer_name": "adam",
        "optimizer_kwargs": {},
        "log_every": 1,
        "seed": 0,
    }


@pytest.mark.parametrize("batch_size", [0, -5])
def test_training_config_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=batch_size, learning_rate=0.01)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1])
def test_training_config_rejects_non_positive_learning_rate(
    learning_rate: float,
) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(batch_size=8, learning_rate=learning_rate)


def test_training_config_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(batch_size=8, learning_rate=0.01, seed=-1)


def test_training_config_rejects_non_positive_log_every() -> None:
    with pytest.raises(ValueError, match="log_every"):
        TrainingConfig(batch_size=8, learning_rate=0.01, log_every=0)
