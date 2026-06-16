from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def bootstrap() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _remove_foreign_editable_finders()
    return REPO_ROOT


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _remove_foreign_editable_finders() -> None:
    filtered = []
    for finder in sys.meta_path:
        finder_module = getattr(finder, "__module__", "")
        if finder_module.startswith("__editable___veritas_"):
            continue
        filtered.append(finder)
    sys.meta_path[:] = filtered
