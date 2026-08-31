from datetime import datetime


def generate_run_id(name: str, *, timestamp: datetime) -> str:
    """Build a unique, sortable run directory name: run_YYYYMMDD_HHMMSS_<name>.

    The timestamp is passed in explicitly (rather than read from the clock here)
    so callers control the single source of truth for "when did this run start"
    and the name stays deterministic and testable.
    """
    return f"run_{timestamp:%Y%m%d_%H%M%S}_{name}"
