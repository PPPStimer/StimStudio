"""Export dialog: format, range, mix or per-track stems, audio options."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
                             QMessageBox, QPushButton, QRadioButton, QVBoxLayout, QWidget)

from ..core.export import AudioSettings, export_mix, export_stems, stem_tracks
from . import theme

FORMATS = (("hwl", "Howl .hwl (pulses)"), ("wav", "WAV audio"), ("flac", "FLAC audio"))
RATES = (44100, 48000, 96000)


class ExportDialog(QDialog):
    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window = window
        self.doc = window.doc
        self.setWindowTitle("Export")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)

        self.format = QComboBox()
        for key, label in FORMATS:
            self.format.addItem(label, key)
        self.format.currentIndexChanged.connect(self._sync)
        form.addRow("Format", self.format)

        self.range = QComboBox()
        self.range.addItem(f"Whole project ({theme.format_time(self.doc.project.duration_s(), 1)})",
                           None)
        loop = window.playback.loop_region
        if loop is not None:
            self.range.addItem(f"Loop region ({theme.format_time(loop[0], 1)} – "
                               f"{theme.format_time(loop[1], 1)})", loop)
        span = window.editor.selection_span()
        if span is not None:
            self.range.addItem(f"Selected clips ({theme.format_time(span[0], 1)} – "
                               f"{theme.format_time(span[1], 1)})", span)
        form.addRow("Range", self.range)

        target = QWidget()
        th = QHBoxLayout(target)
        th.setContentsMargins(0, 0, 0, 0)
        self.mix = QRadioButton("Mix (one file)")
        n = len(stem_tracks(self.doc.project))
        self.stems = QRadioButton(f"Stems (one file per track, {n})")
        self.stems.setEnabled(n > 0)
        group = QButtonGroup(self)
        for b in (self.mix, self.stems):
            group.addButton(b)
            th.addWidget(b)
        self.mix.setChecked(True)
        self.mix.toggled.connect(self._sync)
        form.addRow("Output", target)

        self.mode = QComboBox()
        self.mode.addItems(["continuous", "wavelet"])
        self.rate = QComboBox()
        for r in RATES:
            self.rate.addItem(f"{r} Hz", r)
        self.rate.setCurrentIndex(1)
        self.use_panel = QCheckBox("Use the Output panel's frequency range and sound settings")
        panel = window.device_panel
        if panel.current_key() in ("continuous", "wavelet"):
            self.mode.setCurrentText(panel.current_key())
            self.use_panel.setChecked(True)
        form.addRow("Audio mode", self.mode)
        form.addRow("Sample rate", self.rate)
        form.addRow("", self.use_panel)

        path_row = QHBoxLayout()
        self.path = QLineEdit()
        browse = QPushButton("…")
        browse.setObjectName("secondary")
        browse.setMaximumWidth(36)
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path, 1)
        path_row.addWidget(browse)
        form.addRow("Save to", path_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.run)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._sync()

    # ── helpers ─────────────────────────────────────────────────────────────
    @property
    def fmt(self) -> str:
        return self.format.currentData()

    def _base(self) -> str:
        return Path(self.doc.path).stem if self.doc.path else "untitled"

    def _sync(self, *_):
        audio = self.fmt != "hwl"
        for w in (self.mode, self.rate, self.use_panel):
            w.setEnabled(audio)
        folder = Path(self.window._last_dir())
        current = self.path.text().strip()
        if current:
            p = Path(current)
            folder = p if (p.is_dir() or not p.suffix) else p.parent
        self.path.setText(str(folder if self.stems.isChecked()
                              else folder / f"{self._base()}.{self.fmt}"))

    def _browse(self):
        if self.stems.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Folder for stems", self.path.text())
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export", self.path.text(),
                                                  f"*.{self.fmt}")
        if path:
            self.path.setText(path)

    def audio_settings(self) -> AudioSettings:
        panel = self.window.device_panel
        use = self.use_panel.isChecked() and panel.current_key() == self.mode.currentText()
        return AudioSettings(
            mode=self.mode.currentText(), sample_rate=self.rate.currentData(),
            freq_hz=(panel.freq_min.value(), panel.freq_max.value()) if use else None,
            device_settings=panel.settings_dict() if use else None)

    # ── run ─────────────────────────────────────────────────────────────────
    def run(self) -> list[Path] | None:
        target = self.path.text().strip()
        if not target:
            return None
        rng = self.range.currentData()
        start, end = (rng if rng else (None, None))
        audio = self.audio_settings() if self.fmt != "hwl" else None
        if audio is not None or self.stems.isChecked():
            self.window.playback.stop()   # this holds the GUI thread: nothing would read Esc
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self.stems.isChecked():
                written = export_stems(self.doc.project, target, self._base(), self.fmt,
                                       start, end, audio, self.doc.cache)
            else:
                path = Path(target)
                if path.suffix.lower() != f".{self.fmt}":
                    path = path.with_suffix(f".{self.fmt}")
                export_mix(self.doc.project, path, start, end, audio, self.doc.cache)
                written = [path]
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export", str(e))
            return None
        QApplication.restoreOverrideCursor()
        self.window._remember_dir(str(written[0]))
        self.window.statusBar().showMessage(
            f"Exported {len(written)} file(s) → {written[0].parent}", 6000)
        self.written = written
        self.accept()
        return written
