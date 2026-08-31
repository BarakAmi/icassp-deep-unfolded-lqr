"""Per-run logging binding (REFACTOR_PLAN v3, T3.k, closes G5's execution
half): every `run_experiment` contender execution binds a dedicated file
logger writing ``experiment.log`` inside that run's own artifact folder.

Lifecycle: the handler is bound before contender resolution and released
after persistence — on success *and* failure. Failures write the full
traceback to the run's log before release, so a dead run's folder explains
itself. Console mirroring remains an entry-point concern; the run log is
always written regardless of console verbosity. Log content never enters
signatures (T1.5.c).

`capture_log` is the same binding with a buffer for a destination, for the
Tier-5 producer: a record is published atomically, so its log has to be *in
hand* when the record is written and cannot be a file the record does not own
yet (Annex 02 §2.2, §8).
"""

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: The per-run log file's name inside a run directory (T3.k contract).
RUN_LOG_FILENAME = "experiment.log"

_FORMATTER = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")


@contextmanager
def bind_run_log(
    run_dir: Path, *, level: int = logging.INFO, filename: str = RUN_LOG_FILENAME
) -> Iterator[Path]:
    """Attach a file handler capturing every library logger's records into
    `run_dir / filename` for the duration of the block.

    The handler attaches to the root logger (all module loggers propagate
    to it — the T1.5.e "module loggers only" policy means nothing below the
    adapters ever attaches handlers of its own), and the root's level is
    temporarily lowered so INFO/DEBUG records actually flow; both are
    restored on exit.

    Args:
        run_dir: The run's directory (must exist; the tracker created it).
        level: The capture level — ``INFO`` per the T1.5.e content contract,
            ``DEBUG`` for per-epoch metrics and hardware handshakes.
        filename: The log file's name inside `run_dir`.

    Yields:
        The bound log file's path.

    Raises:
        Whatever the block raises — after writing the full traceback to the
        run's log, so the run folder explains its own death.
    """
    log_path = Path(run_dir) / filename
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_FORMATTER)

    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    if root.getEffectiveLevel() > level:
        root.setLevel(level)
    try:
        yield log_path
    except BaseException:
        logging.getLogger(__name__).exception(
            "Run failed; full traceback follows (T3.k failure contract)."
        )
        raise
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.close()


@contextmanager
def capture_log(*, level: int = logging.INFO) -> Iterator[io.StringIO]:
    """Capture every library log record emitted inside the block, in memory.

    The same binding as `bind_run_log` with a different destination, and for the
    same reason it exists at all: a phase that fails must leave something that
    explains it. The Tier-5 producer publishes a record atomically, so the log
    has to be in hand when the record is written -- a file beside the record
    would be a second home for a member the record owns (Annex 02 §2.2).

    **This is process-global while the block runs.** `run_study` is serial, so a
    captured log belongs to exactly one point; Annex 06 §4.3's executor makes
    that false the day it lands, and the capture then needs a filter on the
    worker rather than a handler on the root.

    Args:
        level: The capture level -- ``INFO`` per the T1.5.e content contract.
            A payload that grew with anything but the number of phases would be
            the derived bulk Annex 02 §2 exists to keep out of the store.

    Yields:
        The buffer being written to. Read it with ``.getvalue()`` **after** the
        block: the handler is flushed on release, and a value taken inside can
        be missing the last record.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(level)
    handler.setFormatter(_FORMATTER)

    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    if root.getEffectiveLevel() > level:
        root.setLevel(level)
    try:
        yield buffer
    except BaseException:
        logging.getLogger(__name__).exception(
            "Phase failed; full traceback follows (T3.k failure contract)."
        )
        raise
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        handler.flush()
