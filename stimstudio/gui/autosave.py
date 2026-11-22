"""Autosave and crash recovery.

Every ``interval`` seconds a modified project is written to the autosave
folder (not over the user's file). A normal save or a clean close removes the
autosave; whatever is left at the next start is offered for recovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QStandardPaths, QTimer

from ..io.project_file import save_project

AUTOSAVE_INTERVAL_S = 120


def default_autosave_dir() -> Path:
    env = os.environ.get("STIMSTUDIO_AUTOSAVE_DIR")
    if env:
        return Path(env)
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(base or Path.home() / ".stimstudio") / "autosave"


class Autosave(QObject):
    def __init__(self, doc, folder: Path | None = None, interval_s: int = AUTOSAVE_INTERVAL_S,
                 parent=None):
        super().__init__(parent)
        self.doc = doc
        self.folder = Path(folder) if folder else default_autosave_dir()
        self.timer = QTimer(self)
        self.timer.setInterval(int(interval_s * 1000))
        self.timer.timeout.connect(self.save_now)
        self.timer.start()
        self._session = hashlib.sha1(f"{os.getpid()}-{time.time()}".encode()).hexdigest()[:8]
        doc.dirtyChanged.connect(lambda dirty: None if dirty else self.discard())

    def _stem(self) -> str:
        name = Path(self.doc.path).stem if self.doc.path else "untitled"
        return f"{name}-{self._session}"

    @property
    def project_path(self) -> Path:
        return self.folder / f"{self._stem()}.stimproj"

    def save_now(self) -> Path | None:
        """Write the autosave if the project has unsaved changes."""
        if not self.doc.dirty:
            return None
        self.folder.mkdir(parents=True, exist_ok=True)
        for old in self.folder.glob(f"*-{self._session}.*"):   # renamed project: drop stale copy
            if old.stem != self._stem():
                old.unlink(missing_ok=True)
        # media paths relative to the real project (the autosave lives elsewhere)
        base = Path(self.doc.path).parent if self.doc.path else None
        save_project(self.doc.project, self.project_path, base_dir=base)
        meta = {"original": self.doc.path, "saved": time.time()}
        self.project_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return self.project_path

    def discard(self) -> None:
        for p in self.folder.glob(f"*-{self._session}.*"):
            p.unlink(missing_ok=True)

    def stop(self) -> None:
        self.timer.stop()
        self.discard()


def find_recoverable(folder: Path | None = None) -> list[tuple[Path, dict]]:
    """Autosaves left behind by sessions that did not close cleanly, newest first."""
    folder = Path(folder) if folder else default_autosave_dir()
    out = []
    for proj in folder.glob("*.stimproj") if folder.is_dir() else []:
        meta_path = proj.with_suffix(".json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {"original": None, "saved": proj.stat().st_mtime}
        out.append((proj, meta))
    return sorted(out, key=lambda pm: pm[1].get("saved", 0), reverse=True)


def remove_autosave(path: Path) -> None:
    Path(path).unlink(missing_ok=True)
    Path(path).with_suffix(".json").unlink(missing_ok=True)
