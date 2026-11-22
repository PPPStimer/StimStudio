"""``.stimproj`` project files (JSON, UTF-8).

Media paths are stored relative to the project file, with the absolute path as
a fallback (see ``core.media``); in memory they are always absolute.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.media import resolve_paths, store_paths
from ..core.model import Project

EXTENSION = ".stimproj"
_DEFAULT = object()


def save_project(project: Project, path, base_dir=_DEFAULT) -> None:
    """Write atomically: a crash mid-save never leaves a truncated project.

    Media paths are made relative to *base_dir* — by default the folder of
    *path*; ``None`` keeps them absolute (autosave, which lives elsewhere).
    """
    path = Path(path)
    if base_dir is _DEFAULT:
        base_dir = path.parent
    data = store_paths(project.to_dict(), base_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_project(path, base_dir=_DEFAULT) -> Project:
    """Read a project; relative media paths resolve against *base_dir* (by
    default the project's own folder)."""
    path = Path(path)
    if base_dir is _DEFAULT:
        base_dir = path.parent
    data = json.loads(path.read_text(encoding="utf-8"))
    return Project.from_dict(resolve_paths(data, base_dir))
