"""Beat timestamps import dialog: pick the CSV tracks that become beat-marks
tracks (one block per pattern group / run of standalone beats)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                             QLabel, QMessageBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from ..io.beat_csv import blocks_to_clips, read_beat_timestamps
from . import theme


def load_beat_files(paths) -> tuple[list[tuple[str, object]], list[str]]:
    """``[(display name, BeatTrack)]`` of every file, and the errors."""
    tracks, errors = [], []
    for p in paths:
        try:
            for t in read_beat_timestamps(p):
                label = t.name if len(paths) == 1 else f"{Path(p).stem}: {t.name}"
                tracks.append((label, t))
        except (OSError, ValueError) as exc:
            errors.append(f"{Path(p).name}: {exc}")
    return tracks, errors


def beat_track_items(tracks, offset_s: float = 0.0, selected=None):
    """``[(track name, [beat-marks clips])]`` for the chosen CSV tracks (by
    index; default: every track with beats)."""
    items = []
    for i, (label, t) in enumerate(tracks):
        if selected is not None and i not in selected:
            continue
        if selected is None and not t.beats:
            continue
        items.append((label, blocks_to_clips(t.blocks(), label, offset_s)))
    return items


class BeatImportDialog(QDialog):
    def __init__(self, paths, offset_s: float = 0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import beat timestamps")
        self.setMinimumWidth(520)
        self.tracks, errors = load_beat_files(paths)
        if errors:
            QMessageBox.warning(self, "Import beat timestamps", "\n".join(errors))
        lay = QVBoxLayout(self)
        info = QLabel("Every checked CSV track becomes a track of beat-marks blocks: one block "
                      "per pattern group (pattern start … pattern end) and one per run of "
                      "standalone beats. Turn a block into a pattern from its context menu.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(info)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["CSV track", "Beats", "Blocks", "Span"])
        self.tree.setRootIsDecorated(False)
        for label, t in self.tracks:
            span = (theme.format_time(t.beats[-1][0] - t.beats[0][0]) if t.beats else "—")
            it = QTreeWidgetItem([label, str(len(t.beats)), str(len(t.blocks())), span])
            it.setCheckState(0, Qt.CheckState.Checked if t.beats else Qt.CheckState.Unchecked)
            self.tree.addTopLevelItem(it)
        for c in range(4):
            self.tree.resizeColumnToContents(c)
        lay.addWidget(self.tree, 1)

        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItem("Standard tracks", "standard")
        self.kind.addItem("Triphase tracks", "triphase")
        form.addRow("Create", self.kind)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(0.0, 24 * 3600.0)
        self.offset.setDecimals(3)
        self.offset.setSuffix(" s")
        self.offset.setValue(max(0.0, offset_s))
        self.offset.setToolTip("Timeline position of CSV time 0 (the start of the audio)")
        form.addRow("Time 0 at", self.offset)
        lay.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def options(self) -> dict:
        selected = [i for i in range(self.tree.topLevelItemCount())
                    if self.tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked]
        return {"track_type": self.kind.currentData(), "offset_s": self.offset.value(),
                "tracks": selected}
