"""The in-memory log capture behind `synthesis.log` / `evaluation.log`.

Annex 02 §2.2, §8. Two properties carry it, and only one of them is about what
gets captured:

* it captures what the block emits, at `INFO`, formatted the way the run log
  already formats records; and
* it leaves the root logger exactly as it found it -- **including when the
  block raises**, because a handler leaked by a failed phase attaches itself to
  every later phase in the same process, and a producer runs 135 of them.
"""

from __future__ import annotations

import logging

import pytest

from mbl.experiments.run_logging import capture_log

logger = logging.getLogger("mbl.tests.capture")


def _root_state() -> tuple[list[logging.Handler], int]:
    root = logging.getLogger()
    return list(root.handlers), root.level


def test_it_captures_what_the_block_emits() -> None:
    with capture_log() as captured:
        logger.info("synthesizing on %d-trajectory batches", 256)

    text = captured.getvalue()
    assert "synthesizing on 256-trajectory batches" in text
    assert "mbl.tests.capture" in text, "the formatter's logger name is missing"


def test_it_captures_nothing_emitted_outside_the_block() -> None:
    """The buffer is a record of one phase, not of the process."""
    logger.info("before the block")
    with capture_log() as captured:
        logger.info("inside the block")
    logger.info("after the block")

    text = captured.getvalue()
    assert "inside the block" in text
    assert "before the block" not in text
    assert "after the block" not in text


def test_a_debug_record_is_below_the_capture_level() -> None:
    """`INFO` per T1.5.e. Per-epoch metrics are logged at DEBUG, and a payload
    that grew with the epoch count is the derived bulk the store keeps out."""
    with capture_log() as captured:
        logger.debug("per-epoch metrics: %s", {"loss": 1.0})
        logger.info("kept")

    assert "per-epoch metrics" not in captured.getvalue()
    assert "kept" in captured.getvalue()


def test_the_root_logger_is_left_exactly_as_it_was() -> None:
    before = _root_state()

    with capture_log():
        logger.info("anything")

    assert _root_state() == before


def test_the_root_logger_is_restored_when_the_block_raises() -> None:
    """A handler leaked by a failed phase attaches itself to every later phase
    in the same process, and a publication run has 135 of them."""
    before = _root_state()

    with pytest.raises(RuntimeError, match="the solver diverged"):
        with capture_log():
            raise RuntimeError("the solver diverged")

    assert _root_state() == before


def test_a_failing_block_leaves_its_traceback_in_the_buffer() -> None:
    """The phase that died is the one whose log has to explain it."""
    with pytest.raises(RuntimeError):
        with capture_log() as captured:
            logger.info("starting")
            raise RuntimeError("the solver diverged")

    text = captured.getvalue()
    assert "starting" in text
    assert "the solver diverged" in text
    assert "Traceback" in text


def test_the_buffer_is_empty_when_nothing_was_logged() -> None:
    """What the producer turns into `log=None`: a phase that said nothing has
    no log, and an empty member would claim otherwise."""
    with capture_log() as captured:
        pass

    assert captured.getvalue() == ""
