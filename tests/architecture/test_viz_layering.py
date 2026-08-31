"""Import-graph enforcement of the `viz` layering (REFACTOR_PLAN v3, §7.2):

* `viz.style` (L0) imports nothing project-internal outside itself;
* `viz.plots` (L1) and `viz.landscape` (L2) never import `engine`, `models`,
  or `applications` -- renderers are pure functions of frozen Result data --
  with exactly one documented exemption: `viz.landscape.oracles` adapts the
  closed-form local Bellman kernel (`models.analytic.riccati
  .evaluate_local_cost_to_go`) into a `CostOracle`;
* only `viz.adapters` may import `IPython`/`streamlit`;
* no module below the adapter layer selects a matplotlib backend
  (`matplotlib.use`) at import time (N9).

Static AST checks: nothing here executes `src` code.
"""

import ast
from pathlib import Path

import pytest

VIZ_ROOT = Path(__file__).resolve().parents[2] / "src" / "mbl" / "viz"

#: The one blessed downward reach: the oracle adapter for the closed-form
#: local Bellman cost-to-go (relocated verbatim by T4.a).
ORACLE_EXEMPTION = "landscape/oracles.py"

FORBIDDEN_BELOW_ADAPTERS = ("mbl.engine", "mbl.models", "mbl.applications")
DISPLAY_ONLY_IN_ADAPTERS = ("IPython", "streamlit")


def _viz_modules() -> list[Path]:
    return sorted(VIZ_ROOT.rglob("*.py"))


def _resolved_imports(path: Path) -> list[str]:
    """Absolute dotted names of every import in `path`, with relative
    imports resolved against the module's own package (e.g. ``..style``
    inside ``src/viz/plots/foo.py`` -> ``src.viz.style``)."""
    tree = ast.parse(path.read_text())
    package_parts = path.relative_to(
        VIZ_ROOT.parents[1]
    ).parent.parts  # ("src", "viz", ...)
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


def _layer(path: Path) -> str:
    relative = path.relative_to(VIZ_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else "__root__"


@pytest.mark.parametrize(
    "path", _viz_modules(), ids=lambda p: str(p.relative_to(VIZ_ROOT))
)
def test_style_imports_nothing_project_internal(path: Path) -> None:
    if _layer(path) != "style":
        pytest.skip("style-layer rule")
    for module in _resolved_imports(path):
        if module.startswith("mbl.") and not module.startswith("mbl.viz.style"):
            pytest.fail(f"{path.name} (L0 style) imports project-internal {module}")


@pytest.mark.parametrize(
    "path", _viz_modules(), ids=lambda p: str(p.relative_to(VIZ_ROOT))
)
def test_renderers_never_import_engines_models_or_applications(path: Path) -> None:
    if _layer(path) not in ("plots", "landscape"):
        pytest.skip("renderer-layer rule")
    if str(path.relative_to(VIZ_ROOT)) == ORACLE_EXEMPTION:
        pytest.skip("the documented local-Bellman oracle exemption")
    for module in _resolved_imports(path):
        for forbidden in FORBIDDEN_BELOW_ADAPTERS:
            if module == forbidden or module.startswith(forbidden + "."):
                pytest.fail(
                    f"{path.relative_to(VIZ_ROOT)} (renderer layer) imports "
                    f"{module} -- renderers must be pure functions of frozen "
                    "Result data (T4.b)"
                )


@pytest.mark.parametrize(
    "path", _viz_modules(), ids=lambda p: str(p.relative_to(VIZ_ROOT))
)
def test_only_adapters_import_display_frontends(path: Path) -> None:
    if _layer(path) == "adapters":
        pytest.skip("adapters are the display layer")
    for module in _resolved_imports(path):
        for frontend in DISPLAY_ONLY_IN_ADAPTERS:
            if module == frontend or module.startswith(frontend + "."):
                pytest.fail(
                    f"{path.relative_to(VIZ_ROOT)} imports {frontend} -- only "
                    "viz.adapters may touch display frontends (T4.a)"
                )


@pytest.mark.parametrize(
    "path", _viz_modules(), ids=lambda p: str(p.relative_to(VIZ_ROOT))
)
def test_no_backend_selection_below_the_adapter_layer(path: Path) -> None:
    if _layer(path) == "adapters":
        pytest.skip("backend selection is an adapter/entry-point concern")
    assert "matplotlib.use(" not in path.read_text(), (
        f"{path.relative_to(VIZ_ROOT)} selects a matplotlib backend at import "
        "time (N9: backend selection belongs to viz.adapters.dashboard / "
        "tests/conftest.py)"
    )


def test_oracle_exemption_still_exists() -> None:
    """If the oracle module moves, the exemption above must move with it
    rather than silently exempting nothing."""
    assert (VIZ_ROOT / ORACLE_EXEMPTION).is_file()
