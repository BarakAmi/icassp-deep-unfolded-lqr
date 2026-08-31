"""Stage S0 golden-value capture tool: executes the frozen scenarios in
``tests.regression.scenarios`` on the CURRENT (untouched) tree and persists
their outputs as the golden-master fixtures the regression harness
(``tests/regression/test_golden_master.py``) verifies against.

Run from the repository root:

    uv run python -m tests.regression.capture_golden

Serialization policy (REFACTOR_PLAN v3 §5 T3.i, binding for this harness):
NumPy arrays ride compressed ``.npz`` (``np.savez_compressed``), torch
tensors ride ``safetensors``, scalars/digests/metadata ride plain JSON.
No ``pickle`` in any form -- ``np.load`` is invoked by the tests with
``allow_pickle=False``, and ``torch.save``'s zip-pickle envelope is never
used.

The capture is intentionally NOT re-runnable by accident: existing fixtures
are only overwritten when ``--force`` is passed, because regenerating them
against a drifted tree would silently re-baseline the very numerics the
harness exists to freeze.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors

from tests.regression import scenarios

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

STANDARD_LQR_NPZ = FIXTURES_DIR / "standard_lqr_golden.npz"
SIGNAL_SPACE_GD_NPZ = FIXTURES_DIR / "signal_space_gd_golden.npz"
SIGNAL_SPACE_GD_3D_NPZ = FIXTURES_DIR / "signal_space_gd_3d_golden.npz"
SIGNAL_SPACE_GD_SAFETENSORS = FIXTURES_DIR / "signal_space_gd_golden.safetensors"
MANIFEST_JSON = FIXTURES_DIR / "golden_manifest.json"

ALL_FIXTURE_PATHS = (
    STANDARD_LQR_NPZ,
    SIGNAL_SPACE_GD_NPZ,
    SIGNAL_SPACE_GD_3D_NPZ,
    SIGNAL_SPACE_GD_SAFETENSORS,
    MANIFEST_JSON,
)


def _describe(arrays: dict) -> dict[str, dict]:
    """Human-auditable per-array shape/dtype ledger for the manifest."""
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def capture(force: bool = False) -> dict:
    """Execute all scenarios and write every fixture. Returns the manifest."""
    existing = [p for p in ALL_FIXTURE_PATHS if p.exists()]
    if existing and not force:
        raise SystemExit(
            "Golden fixtures already exist; refusing to re-baseline them "
            "without --force:\n  " + "\n  ".join(str(p) for p in existing)
        )
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Capturing scenario: standard_lqr (notebook 01) ...", flush=True)
    lqr_arrays, lqr_scalars = scenarios.run_standard_lqr_scenario()
    np.savez_compressed(STANDARD_LQR_NPZ, **lqr_arrays)

    print("Capturing scenario: signal_space_gd (notebook 02, m=2) ...", flush=True)
    gd_arrays, gd_tensors, gd_scalars = scenarios.run_signal_space_gd_scenario()
    np.savez_compressed(SIGNAL_SPACE_GD_NPZ, **gd_arrays)
    save_safetensors(gd_tensors, SIGNAL_SPACE_GD_SAFETENSORS)

    print("Capturing scenario: signal_space_gd_3d (notebook 02, m=3) ...", flush=True)
    gd3d_arrays, gd3d_scalars = scenarios.run_signal_space_gd_3d_scenario()
    np.savez_compressed(SIGNAL_SPACE_GD_3D_NPZ, **gd3d_arrays)

    manifest = {
        "stage": "S0",
        "purpose": (
            "Golden-master regression baselines frozen on the untouched "
            "pre-refactor tree (REFACTOR_PLAN v3 §7.1). Regenerate only via "
            "`uv run python -m tests.regression.capture_golden --force`, and "
            "only as a deliberate, reviewed re-baselining."
        ),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "known_defect_exceptions": {
            "C1": "models/base.py Controller.get_signature protocol body "
            "embeds problem.* despite its own docstring; sentinel in "
            "test_known_defect_sentinels.py, fixed in S2 (T2.a).",
            "C2": "applications/rollout.py RolloutModel._differentiable_cost "
            "ignores include_terminal_cost/is_time_averaged; the frozen "
            "J_opt coincides with the declared (False, True) default flags "
            "and must survive the S2 kernel fix unchanged; divergence under "
            "non-default flags is frozen as a sentinel, fixed in S2 (T2.d).",
            "C3": "core/utils/validation.py validate_positive_(semi_)definite "
            "does not forward atol to the eigenvalue check; sentinel frozen, "
            "fixed in S1 (T1.d).",
            "C4": "both apps' build_engine logs unfolded_learning_rate for "
            "every model family; sentinel frozen, fixed in S3 (T3.b).",
        },
        "scenarios": {
            "standard_lqr": {
                "params": scenarios.STANDARD_LQR_PARAMS,
                "scalars": lqr_scalars,
                "arrays": _describe(lqr_arrays),
                "npz": STANDARD_LQR_NPZ.name,
            },
            "signal_space_gd": {
                "params": scenarios.SIGNAL_SPACE_GD_PARAMS,
                "variants": list(scenarios.GD_VARIANTS),
                "scalars": gd_scalars,
                "arrays": _describe(gd_arrays),
                "torch_tensors": _describe(gd_tensors),
                "npz": SIGNAL_SPACE_GD_NPZ.name,
                "safetensors": SIGNAL_SPACE_GD_SAFETENSORS.name,
            },
            "signal_space_gd_3d": {
                "params": scenarios.SIGNAL_SPACE_GD_3D_PARAMS,
                "scalars": gd3d_scalars,
                "arrays": _describe(gd3d_arrays),
                "npz": SIGNAL_SPACE_GD_3D_NPZ.name,
            },
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for path in ALL_FIXTURE_PATHS:
        print(f"  wrote {path.name}: {path.stat().st_size / 1024:.1f} KiB")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden fixtures (deliberate re-baselining only).",
    )
    capture(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
