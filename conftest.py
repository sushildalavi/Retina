from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _remove_foreign_editable_finders() -> None:
    filtered = []
    for finder in sys.meta_path:
        finder_module = getattr(finder, "__module__", "")
        if finder_module.startswith("__editable___veritas_"):
            continue
        filtered.append(finder)
    sys.meta_path[:] = filtered


root = str(REPO_ROOT)
if root not in sys.path:
    sys.path.insert(0, root)

_remove_foreign_editable_finders()
