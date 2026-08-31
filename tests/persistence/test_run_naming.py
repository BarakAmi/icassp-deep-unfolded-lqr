from datetime import datetime

from mbl.persistence.run_naming import generate_run_id


def test_generate_run_id_formats_timestamp_and_name() -> None:
    timestamp = datetime(2026, 7, 3, 14, 5, 9)
    assert (
        generate_run_id("covert_lqr", timestamp=timestamp)
        == "run_20260703_140509_covert_lqr"
    )


def test_generate_run_id_is_deterministic_for_same_timestamp() -> None:
    timestamp = datetime(2026, 1, 1, 0, 0, 0)
    first = generate_run_id("exp", timestamp=timestamp)
    second = generate_run_id("exp", timestamp=timestamp)
    assert first == second


def test_generate_run_id_differs_across_timestamps() -> None:
    name = "exp"
    first = generate_run_id(name, timestamp=datetime(2026, 1, 1, 0, 0, 0))
    second = generate_run_id(name, timestamp=datetime(2026, 1, 1, 0, 0, 1))
    assert first != second
