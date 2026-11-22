"""Browser dock: generators and clip presets. Drag an entry onto a track to add a
clip there, or double-click it to add it at the edit cursor."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QMimeData, QStandardPaths, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QHBoxLayout, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ...generators import list_generators
from ...io.presets import BUNDLED_DIR, list_presets
from ..song.view import GENERATOR_MIME, PRESET_MIME

CATEGORIES = (
    ("activity", "Howl activities"),
    ("shape", "Shapes"),
    ("rhythm", "Rhythm"),
    ("stim", "Electrotherapy"),
    ("illusion", "Tactile illusions"),
    ("chaos", "Noise & chaos"),
    ("body", "Breath & body"),
    ("triphase", "Triphase (3 electrodes)"),
    ("manual", "Manual"),
    ("import", "Import"),
)

_ROLE = Qt.ItemDataRole.UserRole


def user_preset_dir() -> Path:
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(docs or Path.home()) / "StimStudio" / "Presets"


class _DragTree(QTreeWidget):
    """Tree whose leaves carry a payload (generator id or preset path) under *mime*."""

    def __init__(self, mime: str, parent=None):
        super().__init__(parent)
        self._mime = mime
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)

    def mimeData(self, items):
        mime = QMimeData()
        payload = items[0].data(0, _ROLE) if items else None
        if payload:
            mime.setData(self._mime, str(payload).encode("utf-8"))
        return mime

    def mimeTypes(self):
        return [self._mime]


class GeneratorBrowser(QWidget):
    generatorActivated = pyqtSignal(str)   # double-click: add at the cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tree = _DragTree(GENERATOR_MIME)
        lay.addWidget(self.tree)
        groups = {}
        for cat, title in CATEGORIES:
            g = QTreeWidgetItem([title])
            g.setFlags(Qt.ItemFlag.ItemIsEnabled)
            groups[cat] = g
            self.tree.addTopLevelItem(g)
        for gen in list_generators():
            it = QTreeWidgetItem([gen.label])
            it.setData(0, _ROLE, gen.type_id)
            tags = [gen.type_id] + (["random (seed)"] if gen.stochastic else []) + \
                (["follows BPM"] if gen.tempo_synced else [])
            it.setToolTip(0, " · ".join(tags))
            groups.get(gen.category, groups["shape"]).addChild(it)
        for g in groups.values():
            g.setExpanded(True)
            g.setHidden(g.childCount() == 0)
        self.tree.itemDoubleClicked.connect(self._on_double)
        self.select("waveform")

    def current_type(self) -> str:
        it = self.tree.currentItem()
        return (it.data(0, _ROLE) if it else None) or "waveform"

    def select(self, type_id: str) -> None:
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for j in range(g.childCount()):
                if g.child(j).data(0, _ROLE) == type_id:
                    self.tree.setCurrentItem(g.child(j))
                    return

    def _on_double(self, item, _col):
        if item.data(0, _ROLE):
            self.generatorActivated.emit(item.data(0, _ROLE))


class PresetBrowser(QWidget):
    presetActivated = pyqtSignal(str)      # double-click: add at the cursor (path)

    def __init__(self, user_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.user_dir = Path(user_dir) if user_dir else user_preset_dir()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tree = _DragTree(PRESET_MIME)
        self.tree.itemDoubleClicked.connect(
            lambda it, _c: it.data(0, _ROLE) and self.presetActivated.emit(it.data(0, _ROLE)))
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        row.setContentsMargins(4, 2, 4, 4)
        for text, slot in (("Refresh", self.refresh), ("Open folder", self.open_folder)):
            b = QPushButton(text)
            b.setObjectName("secondary")
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def refresh(self) -> None:
        self.tree.clear()
        for title, folder in (("Bundled", BUNDLED_DIR), ("My presets", self.user_dir)):
            root = QTreeWidgetItem([title])
            root.setFlags(Qt.ItemFlag.ItemIsEnabled)
            root.setToolTip(0, str(folder))
            self.tree.addTopLevelItem(root)
            subgroups: dict[Path, QTreeWidgetItem] = {}
            for path in list_presets(folder):
                parent = root
                rel_dir = path.parent.relative_to(folder)
                if rel_dir != Path("."):
                    parent = subgroups.get(rel_dir)
                    if parent is None:
                        parent = subgroups[rel_dir] = QTreeWidgetItem([str(rel_dir)])
                        parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        root.addChild(parent)
                        parent.setExpanded(True)
                it = QTreeWidgetItem([path.stem])
                it.setData(0, _ROLE, str(path))
                it.setToolTip(0, str(path))
                parent.addChild(it)
            root.setExpanded(True)
            if root.childCount() == 0:
                hint = QTreeWidgetItem(["(save a clip as preset to add one)"])
                hint.setFlags(Qt.ItemFlag.NoItemFlags)
                root.addChild(hint)

    def open_folder(self) -> None:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.user_dir)))


class Browser(QTabWidget):
    """The left dock: Generators and Presets tabs."""

    def __init__(self, user_preset_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.generators = GeneratorBrowser()
        self.presets = PresetBrowser(user_preset_dir)
        self.addTab(self.generators, "Generators")
        self.addTab(self.presets, "Presets")
        self.setDocumentMode(True)

    # generator API used by the main window / song editor
    def current_type(self) -> str:
        return self.generators.current_type()

    def select(self, type_id: str) -> None:
        self.generators.select(type_id)
