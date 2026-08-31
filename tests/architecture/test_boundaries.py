"""Cross-cutting architecture boundaries (REFACTOR_PLAN v3, §7.2), completing
the enforcement suite Stage S5 started with `test_viz_layering`:

* **Core purity** — `src/core` imports nothing from
  `models|engine|experiments|applications|viz|persistence|workbench`.
* **No-pickle law (T3.i)** — `pickle` (and `torch.save`/`torch.load`,
  whose default envelope is zip-pickle) appears nowhere on the modern
  write/read paths outside the single quarantined legacy loader module.
* **Print/handler hygiene (T1.5.e)** — no `print(...)` calls anywhere in
  `src`, and logging handlers are attached only by the sanctioned per-run
  binding (`experiments.run_logging`, T3.k); every other module may only
  obtain module loggers.
* **Pydantic boundary rule (T1.5.a)** — `pydantic` is imported only by the
  designated boundary-validation module, never by `core`/`models`/`engine`
  interiors.
* **Conversion boundary (T1.f)** — NumPy<->Torch materialization primitives
  (`torch.as_tensor`/`torch.tensor`/`torch.from_numpy`, `.numpy()`,
  `.cpu()`, `core.utils.to_numpy`) appear only in the reviewed allowlist of
  designated boundary modules below. The list codifies today's audited set;
  a conversion appearing in any NEW module fails this test and forces an
  explicit review (extend the allowlist deliberately or route the code
  through an existing boundary).

Static source checks only: nothing here executes `src` code.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mbl"

#: The frozen legacy package (never scanned; scheduled for deletion).
LEGACY_PREFIX = "lqr"

#: T3.i: the one module allowed to *read* legacy pickle formats, kept solely
#: so the dashboard can open grandfathered pre-v3 runs; unreachable from
#: `ExperimentCache` (asserted behaviorally in tests/experiments).
PICKLE_QUARANTINE = {"persistence/artifact_loaders.py"}

#: T3.k / notebook entry point: the src modules allowed to attach a logging
#: handler. `run_logging.py` binds the per-run ``experiment.log``;
#: `notebook_bootstrap.py` is the notebook's explicit entry-point helper, whose
#: `configure_notebook_logging` installs the console handler ONLY when called
#: (never at import time), so importing it as a library stays side-effect-free.
HANDLER_SANCTIONED = {
    "experiments/run_logging.py",
    "experiments/notebook_bootstrap.py",
}

#: T1.5.e: the adapter layer this rule is named after. A command line's output
#: *is* its return value, so routing it through a logger would put the answer
#: behind a handler configuration. The carve-out is deliberately one module
#: deep: `cli/render.py` builds every string and returns it, and stays subject
#: to the rule, which is what keeps the layout testable against literals.
ADAPTER_LAYER = {"cli/app.py"}

#: T1.5.a: the one boundary-validation module that may import pydantic.
PYDANTIC_BOUNDARY = {"core/runtime/validators.py"}

#: T1.f: the reviewed conversion-boundary allowlist. Grouped by role; every
#: entry is a module whose NumPy<->Torch materializations were audited as
#: intentional ingress/egress (array authorship, persistence codecs,
#: solver-ingress casting, or presentation-layer ingestion).
CONVERSION_BOUNDARY = {
    # -- array authorship & signing (the context/utility boundary) --------
    "core/runtime/compute_context.py",
    "core/utils/array_ops.py",
    "core/utils/signing.py",
    "core/constraint/box_constraint.py",  # dual-backend projection ingress
    "core/constraint/activity.py",  # same ingress, one module over: the box's own bound
    # -- persistence codecs ------------------------------------------------
    "persistence/artifact_serializers.py",
    # -- engine/solver ingress (operands authored onto the torch substrate)
    "engine/callbacks.py",
    "engine/strategy.py",
    "experiments/evaluation.py",
    "experiments/shifted_evaluation.py",  # same solver-ingress category as evaluation.py
    "experiments/zero_shot.py",  # same solver-ingress category as evaluation.py (NB06)
    "models/analytic/iterative_gd.py",
    "models/analytic/riccati.py",
    "models/constrained/cocp.py",
    "models/constrained/cocp_exact.py",  # plant ingress: the QP's (A, B, R) become buffers once, at construction
    "models/constrained/solver_resolution.py",
    "models/iterative/initializers.py",
    "models/iterative/refinement.py",
    "models/open_loop/gd_controller.py",
    "models/unfolded/base.py",
    "models/unfolded/parameters.py",
    "applications/ood/noise.py",  # NB06: numpy-drawn exotic families cast to torch at the sampler boundary, same category as dataset.py
    "applications/recipes/cocp.py",  # cocp_parameter_log_summaries: same category as unfolded.py's own log summaries
    "applications/recipes/cocp_exact.py",  # exact_cocp_parameter_log_summaries: same category as cocp.py's
    "applications/recipes/unfolded.py",
    "applications/rollout.py",
    "applications/uncertainty/dataset.py",  # torch indexing for finite-set mini-batches, same category as rollout.py
    "applications/uncertainty/resync.py",  # rebuilds refinement static_parameters tensor stacks, same category as unfolded.py
    "workbench/analysis.py",
    "workbench/benchmarks.py",  # bridges numpy-native + torch-native frozen controllers
    "workbench/replay.py",  # cocp_reference_point: solver ingress, same category as cocp.py
    "benchmark/cells.py",  # egress for the non-finite guard: a control produced on the card must reach numpy to be checked
    "workbench/setup.py",
    # -- presentation-layer ingestion (renderers/adapters consume results) -
    "viz/adapters/notebook.py",
    "viz/adapters/sink.py",
    "viz/landscape/grids.py",
    "viz/landscape/oracles.py",
    "viz/landscape/types.py",
    "viz/plots/convergence.py",
    "viz/plots/cost_analysis.py",
    "viz/plots/loss_landscape.py",
    "viz/plots/matrix_evolution.py",
    "viz/plots/training_curves.py",
    "viz/plots/trajectories.py",
}

#: Tier-3 purity: the packages `src/mbl/spec` must never reach into. The
#: grammar exists to be reusable by the runner, the analysis tier and the
#: command line alike, and it derives identity for all of them; a dependency on
#: `engine`/`models`/`experiments`/`applications` is exactly what made
#: `experiments.Experiment` un-reusable and is what this stage is replacing.
#: `core` and `store` are permitted -- the grammar is built on the identifier
#: types and the signing primitives.
FORBIDDEN_FOR_SPEC = (
    "mbl.models",
    "mbl.engine",
    "mbl.experiments",
    "mbl.applications",
    "mbl.viz",
    "mbl.workbench",
    "mbl.persistence",
    "mbl.cli",
    # Tier 5, added with the producer (Stage 2 F2). The grammar is what the
    # runner is written against; a dependency the other way would make the
    # execution layer's imports -- engine, applications, torch -- reachable
    # from every consumer of a `ProblemSpec`.
    "mbl.runner",
    # Tier 6, added with `cost_vs_axis` (slice Phase B). The analysis tier is
    # written against the grammar for the same reason the runner is, and the
    # grammar reaching it would put pandas and scipy behind every `StudySpec`.
    "mbl.analysis",
    # Tier 7 and Tier 8, added with the replay seam (Stage 6 Phase A). Same
    # rule, sixth occurrence: `present` puts matplotlib behind every
    # `StudySpec`, and `replay` puts all of them at once, since it is the one
    # module that sees the whole pipeline.
    "mbl.present",
    "mbl.replay",
)

#: The top tier. `replay/` is what a notebook imports and it may reach anything
#: below it; **nothing below it may reach back**, or the seam stops being a
#: seam and becomes a cycle. `cli` is not exempt and does not need to be: the
#: command line and the notebook are two peers over one pipeline, not layers.
TOP_TIER = "mbl.replay"

#: Core purity: the packages `src/core` must never reach into.
FORBIDDEN_FOR_CORE = (
    "mbl.models",
    "mbl.engine",
    "mbl.experiments",
    "mbl.applications",
    "mbl.viz",
    "mbl.persistence",
    "mbl.workbench",
    "mbl.runner",
)

_CONVERSION_PATTERN = re.compile(
    r"torch\.as_tensor\(|torch\.tensor\(|torch\.from_numpy\(|\.numpy\(\)"
    r"|\.cpu\(\)|to_numpy\("
)


def _modern_modules() -> list[Path]:
    return sorted(
        path
        for path in SRC_ROOT.rglob("*.py")
        if not path.relative_to(SRC_ROOT).parts[0].startswith(LEGACY_PREFIX)
    )


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC_ROOT)).replace("\\", "/")


def _resolved_imports(path: Path) -> list[str]:
    """Absolute dotted names of every import in `path`, with relative
    imports resolved against the module's own package."""
    tree = ast.parse(path.read_text())
    package_parts = path.relative_to(SRC_ROOT.parents[0]).parent.parts
    resolved: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            resolved.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                resolved.append(node.module or "")
            else:
                base = package_parts[: len(package_parts) - node.level + 1]
                suffix = (node.module,) if node.module else ()
                resolved.append(".".join((*base, *suffix)))
    return resolved


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_core_imports_no_higher_tier(path: Path) -> None:
    if _rel(path).split("/")[0] != "core":
        pytest.skip("core-purity rule")
    for module in _resolved_imports(path):
        for forbidden in FORBIDDEN_FOR_CORE:
            if module == forbidden or module.startswith(forbidden + "."):
                pytest.fail(f"core module {_rel(path)} imports {module}")


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_spec_imports_no_higher_tier(path: Path) -> None:
    if _rel(path).split("/")[0] != "spec":
        pytest.skip("Tier-3 purity rule")
    for module in _resolved_imports(path):
        for forbidden in FORBIDDEN_FOR_SPEC:
            if module == forbidden or module.startswith(forbidden + "."):
                pytest.fail(
                    f"spec module {_rel(path)} imports {module}; Tier 3 may use "
                    "core and store only"
                )


def test_the_spec_tier_is_actually_being_scanned() -> None:
    """Negative control. The rule above skips every module outside `spec/`, so
    if the package were renamed or the tier were empty it would pass by
    skipping everything while claiming to enforce a boundary."""
    scanned = [p for p in _modern_modules() if _rel(p).split("/")[0] == "spec"]
    assert scanned, "no spec/ modules found; the Tier-3 purity rule guards nothing"
    assert any(_rel(p).endswith("problem.py") for p in scanned)


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_nothing_below_the_top_tier_imports_it(path: Path) -> None:
    """`replay/` may reach anything; nothing may reach back.

    The seam exists so a notebook has exactly one import, and an import in the
    other direction would make every consumer of a `StudySpec` pay for the
    whole pipeline — which is the cost `FORBIDDEN_FOR_SPEC` was written to
    prevent, arriving through a different door.
    """
    if _rel(path).split("/")[0] == "replay":
        pytest.skip("the top tier may import itself")
    for module in _resolved_imports(path):
        if module == TOP_TIER or module.startswith(TOP_TIER + "."):
            pytest.fail(
                f"{_rel(path)} imports {module}; nothing may import the top tier"
            )


def test_the_top_tier_is_actually_being_scanned() -> None:
    """Negative control, for the same reason the spec one exists: the rule
    above skips `replay/` itself, so an empty or renamed package would pass by
    guarding nothing."""
    scanned = [p for p in _modern_modules() if _rel(p).split("/")[0] == "replay"]
    assert scanned, "no replay/ modules found; the top-tier rule guards nothing"
    assert any(_rel(p).endswith("resolution.py") for p in scanned)


def test_importing_the_spec_tier_pulls_in_no_higher_tier() -> None:
    """Runtime companion to `test_spec_imports_no_higher_tier`, and stronger in
    the one direction that matters.

    The AST scan is per-module and *direct*: it sees `spec/x.py` importing
    `applications`, but not `spec/x.py` importing something in `core` that
    imports `applications`. This check is transitive, because the cost the rule
    exists to prevent is transitive -- `build_default_recipe_registry()` imports
    every recipe module, and with it the engine, the models and the persistence
    layer, which is what made `experiments.Experiment` un-reusable.

    Every module in the package is imported, discovered from the filesystem, so
    a Tier-3 module added by a later phase cannot drift out of coverage.
    """
    modules = sorted(
        "mbl.spec." + path.stem if path.stem != "__init__" else "mbl.spec"
        for path in (SRC_ROOT / "spec").glob("*.py")
    )
    assert "mbl.spec" in modules, "the spec package vanished; this check guards nothing"
    probe = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"forbidden = {FORBIDDEN_FOR_SPEC!r}\n"
        "print(','.join(sorted(m for m in sys.modules if any("
        "m == f or m.startswith(f + '.') for f in forbidden))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", (
        f"importing {modules} transitively loads {result.stdout.strip()}; Tier 3 "
        "may use core and store only, and what it needs from above must be "
        "injected rather than imported"
    )


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_no_pickle_on_modern_paths(path: Path) -> None:
    if _rel(path) in PICKLE_QUARANTINE:
        pytest.skip("the quarantined legacy loader")
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name == "pickle" or name.startswith("pickle."):
                pytest.fail(f"{_rel(path)} imports pickle (T3.i prohibition)")
    for call in re.finditer(r"torch\.(save|load)\(", source):
        pytest.fail(
            f"{_rel(path)} calls torch.{call.group(1)} (zip-pickle envelope; "
            "persist torch weights via the safetensors codec instead)"
        )


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_no_print_below_the_adapter_layer(path: Path) -> None:
    if _rel(path) in ADAPTER_LAYER:
        pytest.skip("the command line is the adapter layer; stdout is its output")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            pytest.fail(
                f"{_rel(path)}:{node.lineno} calls print() (T1.5.e: library "
                "code narrates through module loggers only)"
            )


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_no_handler_attachment_outside_the_run_binding(path: Path) -> None:
    if _rel(path) in HANDLER_SANCTIONED:
        pytest.skip("the sanctioned per-run experiment.log binding (T3.k)")
    source = path.read_text()
    for needle in ("addHandler", "basicConfig", "StreamHandler", "FileHandler"):
        if needle in source:
            pytest.fail(
                f"{_rel(path)} touches logging handlers ({needle}); handler "
                "attachment is an entry-point/per-run-binding right (T1.5.e)"
            )


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_pydantic_only_at_the_boundary(path: Path) -> None:
    if _rel(path) in PYDANTIC_BOUNDARY:
        pytest.skip("the designated boundary-validation module")
    for module in _resolved_imports(path):
        if module == "pydantic" or module.startswith("pydantic."):
            pytest.fail(
                f"{_rel(path)} imports pydantic outside the boundary "
                "module (T1.5.a placement discipline)"
            )


@pytest.mark.parametrize("path", _modern_modules(), ids=_rel)
def test_conversions_only_in_designated_boundaries(path: Path) -> None:
    if _rel(path) in CONVERSION_BOUNDARY:
        pytest.skip("reviewed conversion-boundary module")
    # Strip comments/docstrings so prose mentioning a primitive never trips
    # the scan; only executable source counts.
    tree = ast.parse(path.read_text())
    executable = ast.unparse(tree)
    match = _CONVERSION_PATTERN.search(executable)
    if match:
        pytest.fail(
            f"{_rel(path)} performs the NumPy<->Torch conversion "
            f"{match.group(0)!r} outside the T1.f boundary allowlist -- "
            "route it through a designated boundary or extend the reviewed "
            "list in tests/architecture/test_boundaries.py deliberately"
        )


@pytest.mark.parametrize(
    "name",
    sorted(
        ADAPTER_LAYER
        | CONVERSION_BOUNDARY
        | HANDLER_SANCTIONED
        | PICKLE_QUARANTINE
        | PYDANTIC_BOUNDARY
    ),
)
def test_every_exemption_names_a_module_that_exists(name: str) -> None:
    """Each carve-out above suspends a rule for one named file. A stale entry --
    left behind by a rename or a deletion -- reads as a reviewed exception while
    protecting nothing, and quietly licenses the next module that happens to
    take the same path. Every list is checked, not a representative one.
    """
    assert (SRC_ROOT / name).is_file(), (
        f"{name} is exempted in tests/architecture/test_boundaries.py but does "
        "not exist; remove the entry or correct the path"
    )
