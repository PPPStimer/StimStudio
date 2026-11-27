"""File → Project media: every file the project uses, missing-file relinking
and *Collect files* (copy everything next to the project)."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QPushButton, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout)

from ..core import media as md
from . import metrics, theme
from .audio_import import run_job

MISSING_COLOR = "#f48771"


def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class _Signals(QObject):
    progress = pyqtSignal(str, float)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class CollectJob(QRunnable):
    """Copies the files of a collect plan; result: ``old → new`` path mapping."""

    def __init__(self, items):
        super().__init__()
        self.setAutoDelete(False)
        self.items = items
        self.signals = _Signals()
        self._cancel = False
        self.partial: dict[str, str] = {}

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = md.run_collect(self.items, lambda name, f: self.signals.progress.emit(
                f"Copying {name}", f), lambda: self._cancel)
        except md.Cancelled:
            self.signals.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit(result)


def collect_files(parent, doc, media_dir: str) -> int | None:
    """Copy the project's files into *media_dir* and relink them (one undo step).
    Returns the number of clips relinked, or None if cancelled."""
    items = md.plan_collect(doc.project, Path(doc.path).parent, media_dir)
    if any(i.action == md.COPY for i in items):
        mapping = run_job(parent, CollectJob(items), "Collecting files", "Collect files")
        if mapping is None:
            return None
    else:
        mapping = {i.file.path: i.target for i in items if i.action == md.REUSE}
    return doc.relink_media(mapping, "Collect files")


class CollectDialog(QDialog):
    """Confirms the target folder and shows what will be copied."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Collect files")
        self.project_dir = Path(doc.path).parent
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Copy every file the project uses into one folder next to the "
                             "project, so it can be moved or shared as a whole.\nThe "
                             "originals are not touched; undo points the clips back at them."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Folder:"))
        self.folder = QLineEdit(str(self.project_dir / md.DEFAULT_MEDIA_DIR))
        self.folder.textChanged.connect(self._update)
        row.addWidget(self.folder, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        lay.addLayout(row)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.save_after = QCheckBox("Save the project afterwards")
        self.save_after.setChecked(True)
        lay.addWidget(self.save_after)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Collect")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.resize(metrics.cap_to_screen(560, parent), 0)
        self._update()

    def media_dir(self) -> str:
        return self.folder.text().strip()

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Collect into", self.media_dir())
        if path:
            self.folder.setText(path)

    def _update(self):
        folder = self.media_dir()
        ok = bool(folder)
        if ok:
            items = md.plan_collect(self.doc.project, self.project_dir, folder)
            count = {a: [i for i in items if i.action == a]
                     for a in (md.COPY, md.REUSE, md.INSIDE, md.MISSING)}
            size = sum(i.file.size for i in count[md.COPY])
            parts = [f"<b>{len(count[md.COPY])}</b> file(s) to copy ({format_size(size)})"]
            if count[md.REUSE]:
                parts.append(f"{len(count[md.REUSE])} already there")
            if count[md.INSIDE]:
                parts.append(f"{len(count[md.INSIDE])} already in the project folder")
            if count[md.MISSING]:
                parts.append(f"<span style='color:{MISSING_COLOR}'>{len(count[md.MISSING])} "
                             "missing (skipped)</span>")
            self.summary.setText(" · ".join(parts))
            ok = bool(count[md.COPY] or count[md.REUSE])
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)


class ProjectMediaDialog(QDialog):
    """List of the project's files with relink and collect actions.

    *save* is a callable that saves the project (asking for a name if it has
    none) and returns True on success.
    """

    COLUMNS = ("File", "Type", "Used by", "Size", "Status", "Folder")

    def __init__(self, doc, parent=None, save=None):
        super().__init__(parent)
        self.doc = doc
        self.save = save
        self.setWindowTitle("Project media")
        lay = QVBoxLayout(self)
        self.where = QLabel()
        self.where.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.where.setWordWrap(True)
        lay.addWidget(self.where)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(self.COLUMNS)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemDoubleClicked.connect(lambda *_: self.locate())
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.find_btn = QPushButton("Find missing in folder…")
        self.find_btn.setToolTip("Search a folder (and its subfolders) for the missing files "
                                 "by name")
        self.find_btn.clicked.connect(lambda: self.find_in_folder())
        self.locate_btn = QPushButton("Locate…")
        self.locate_btn.setToolTip("Choose the new location of the selected file")
        self.locate_btn.clicked.connect(lambda: self.locate())
        self.collect_btn = QPushButton("Collect files…")
        self.collect_btn.setToolTip("Copy every file into a folder next to the project")
        self.collect_btn.clicked.connect(lambda: self.collect())
        for b in (self.find_btn, self.locate_btn, self.collect_btn):
            row.addWidget(b)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        metrics.fit_to_screen(self, 820, 360)
        doc.modelChanged.connect(self.refresh)
        doc.pathChanged.connect(lambda _p: self.refresh())
        self.refresh()

    # ── list ────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        self.files = md.media_files(self.doc.project)
        project_dir = Path(self.doc.path).parent if self.doc.path else None
        if project_dir is None:
            self.where.setText("The project is not saved yet: paths are stored relative to the "
                               "project file once it has a name.")
        else:
            self.where.setText(f"Project folder: {project_dir}  ·  paths are saved relative "
                               "to it, with the absolute path as a fallback.")
        selected = {it.data(0, Qt.ItemDataRole.UserRole) for it in self.tree.selectedItems()}
        self.tree.clear()
        for mf in self.files:
            users = sorted({r.clip_name or "clip" for r in mf.refs})
            if mf.exists:
                inside = project_dir is not None and mf.inside(project_dir)
                status = "OK · in project folder" if inside else "OK"
            else:
                status = "Missing"
            it = QTreeWidgetItem([Path(mf.path).name, mf.kind,
                                  ", ".join(users) if len(users) <= 3 else
                                  f"{len(mf.refs)} clips",
                                  format_size(mf.size) if mf.exists else "—", status,
                                  str(Path(mf.path).parent)])
            it.setData(0, Qt.ItemDataRole.UserRole, mf.path)
            it.setToolTip(0, mf.path)
            if not mf.exists:
                for c in range(len(self.COLUMNS)):
                    it.setForeground(c, QColor(MISSING_COLOR))
            self.tree.addTopLevelItem(it)
            it.setSelected(mf.path in selected)
        for c in range(len(self.COLUMNS) - 1):
            self.tree.resizeColumnToContents(c)
        if not self.files:
            it = QTreeWidgetItem(["(the project uses no media files)"])
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.tree.addTopLevelItem(it)
        self._update_buttons()

    def missing(self) -> list[md.MediaFile]:
        return [m for m in self.files if not m.exists]

    def selected_files(self) -> list[md.MediaFile]:
        paths = {it.data(0, Qt.ItemDataRole.UserRole) for it in self.tree.selectedItems()}
        return [m for m in self.files if m.path in paths]

    def _update_buttons(self) -> None:
        self.find_btn.setEnabled(bool(self.missing()))
        self.locate_btn.setEnabled(len(self.selected_files()) == 1)
        self.collect_btn.setEnabled(any(m.exists for m in self.files))

    # ── actions ─────────────────────────────────────────────────────────────
    def find_in_folder(self, folder: str | None = None) -> int:
        """Search *folder* for the missing files by name; returns how many were found."""
        missing = self.missing()
        if not missing:
            return 0
        if folder is None:
            start = str(Path(self.doc.path).parent) if self.doc.path else ""
            folder = QFileDialog.getExistingDirectory(self, "Search for missing files in", start)
            if not folder:
                return 0
        mapping = md.relink_by_name(missing, folder)
        if mapping:
            self.doc.relink_media(mapping, "Relink missing files")
        left = len(missing) - len(mapping)
        if self.isVisible():
            QMessageBox.information(self, "Find missing files",
                                    f"Found {len(mapping)} of {len(missing)} missing file(s)."
                                    + (f"\n{left} still missing." if left else ""))
        return len(mapping)

    def locate(self, new_path: str | None = None) -> bool:
        sel = self.selected_files()
        if len(sel) != 1:
            return False
        mf = sel[0]
        if new_path is None:
            ext = Path(mf.path).suffix
            start = str(Path(mf.path).parent) if Path(mf.path).parent.is_dir() else (
                str(Path(self.doc.path).parent) if self.doc.path else "")
            new_path, _ = QFileDialog.getOpenFileName(
                self, f"Locate {Path(mf.path).name}", start,
                f"{Path(mf.path).name} (*{ext});;All files (*)")
            if not new_path:
                return False
        self.doc.relink_media({mf.path: new_path}, "Relink file")
        if os.path.isfile(new_path):
            others = self.missing()
            folder = str(Path(new_path).parent)
            if others and md.relink_by_name(others, folder):
                # the other missing files probably moved to the same place
                if self.isVisible():
                    r = QMessageBox.question(self, "Relink file",
                                             f"Other missing files are in {folder} too. "
                                             "Relink them as well?")
                    if r == QMessageBox.StandardButton.Yes:
                        self.find_in_folder(folder)
        return True

    def collect(self, media_dir: str | None = None, save_after: bool | None = None) -> int | None:
        if not self.doc.path:
            if self.save is None or not self.save():
                return None
        if media_dir is None:
            dlg = CollectDialog(self.doc, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            media_dir, save_after = dlg.media_dir(), dlg.save_after.isChecked()
        try:
            n = collect_files(self, self.doc, media_dir)
        except RuntimeError as e:
            QMessageBox.critical(self, "Collect files", str(e))
            return None
        if n is not None and save_after and self.save is not None:
            self.save()
        return n
