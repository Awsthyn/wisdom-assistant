import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
LLM_DIR = APP_DIR / "llm"
FORBIDDEN_ROOTS = ("anthropic", "google.genai")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            if node.module == "google":
                modules.update(f"google.{alias.name}" for alias in node.names)
    return modules


def test_only_app_llm_imports_llm_sdks() -> None:
    """Protects the provider boundary; rejects any module outside app/llm importing an LLM SDK."""
    offenders: list[str] = []
    for path in APP_DIR.rglob("*.py"):
        if LLM_DIR in path.parents:
            continue
        for module in _imported_modules(path):
            if any(module == root or module.startswith(f"{root}.") for root in FORBIDDEN_ROOTS):
                offenders.append(f"{path.relative_to(APP_DIR.parent)}: {module}")

    assert offenders == []
