"""Export dialog: format, range, mix or per-track stems, audio / funscript options.

For audio formats a read-only "Signal" box lists what will be written —
file format, rate, length, size, mode, frequencies, triphase, level — and
follows every option change, so Export / Cancel is decided with it in view.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                             QRadioButton, QVBoxLayout, QWidget)

from ..core.export import (FUNSCRIPT_FORMAT, AudioSettings, FunscriptSettings, describe_audio,
                           export_mix, export_stems, stem_tracks, written_paths)
from ..io.funscript_export import LAYOUT_SEPARATE, LAYOUT_SINGLE
from . import metrics, theme
from .settings import app_settings

FORMATS = (("hwl", "Howl .hwl (pulses)"), ("wav", "WAV audio"), ("flac", "FLAC audio"),
           (FUNSCRIPT_FORMAT, "Funscript (amp + freq, channels A / B)"))
AUDIO = ("wav", "flac")
RATES = (44100, 48000, 96000)


class ExportDialog(QDialog):
    def __init__(self, window, parent=None, fmt: str | None = None):
        super().__init__(parent or window)
        self.window = window
        self.doc = window.doc
        self.settings = app_settings()
        self.setWindowTitle("Export")
        self.setMinimumWidth(metrics.cap_to_screen(560, window))
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)

        self.format = QComboBox()
        for key, label in FORMATS:
            self.format.addItem(label, key)
        if fmt is not None and self.format.findData(fmt) >= 0:
            self.format.setCurrentIndex(self.format.findData(fmt))
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
        self.range.currentIndexChanged.connect(self._update_summary)
        form.addRow("Range", self.range)

        target = QWidget()
        th = QHBoxLayout(target)
        th.setContentsMargins(0, 0, 0, 0)
        self.mix = QRadioButton("Mix (one file)")
        self.n_stems = len(stem_tracks(self.doc.project))
        self.stems = QRadioButton(f"Stems (one file per track, {self.n_stems})")
        self.stems.setEnabled(self.n_stems > 0)
        group = QButtonGroup(self)
        for b in (self.mix, self.stems):
            group.addButton(b)
            th.addWidget(b)
        self.mix.setChecked(True)
        self.mix.toggled.connect(self._sync)
        form.addRow("Output", target)

        # audio options
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
        for w in (self.mode, self.rate):
            w.currentIndexChanged.connect(self._update_summary)
        self.use_panel.toggled.connect(self._update_summary)
        self.audio_rows = [("Audio mode", self.mode), ("Sample rate", self.rate),
                           ("", self.use_panel)]
        for label, w in self.audio_rows:
            form.addRow(label, w)

        # funscript options
        fs_row = QWidget()
        fh = QHBoxLayout(fs_row)
        fh.setContentsMargins(0, 0, 0, 0)
        self.fs_layout = QComboBox()
        self.fs_layout.addItem("4 files (.amp_a / .amp_b / .freq_a / .freq_b)", LAYOUT_SEPARATE)
        self.fs_layout.addItem("one multi-axis file", LAYOUT_SINGLE)
        idx = self.fs_layout.findData(str(self.settings.value("export/funscript_layout",
                                                              LAYOUT_SEPARATE)))
        self.fs_layout.setCurrentIndex(max(0, idx))
        self.fs_layout.currentIndexChanged.connect(self._sync)
        fh.addWidget(self.fs_layout, 1)
        fh.addWidget(QLabel("Tolerance"))
        self.fs_tol = QDoubleSpinBox()
        self.fs_tol.setRange(0.0, 10.0)
        self.fs_tol.setSingleStep(0.5)
        self.fs_tol.setDecimals(1)
        self.fs_tol.setValue(float(self.settings.value("export/funscript_tolerance", 1.0)))
        self.fs_tol.setToolTip("Drop points that lie within this many position units (of 100) "
                               "of a straight line — 0 keeps every change")
        fh.addWidget(self.fs_tol)
        self.fs_row = fs_row
        form.addRow("Funscript", fs_row)

        path_row = QHBoxLayout()
        self.path = QLineEdit()
        browse = QPushButton("…")
        browse.setObjectName("secondary")
        browse.setMaximumWidth(36)
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path, 1)
        path_row.addWidget(browse)
        form.addRow("Save to", path_row)
        self.form = form

        # what the audio file will contain
        self.summary_box = QGroupBox("Signal")
        self.summary = QFormLayout(self.summary_box)
        self.summary.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.summary_box)

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
        audio = self.fmt in AUDIO
        for label, w in self.audio_rows:
            self.form.setRowVisible(w, audio)
        self.form.setRowVisible(self.fs_row, self.fmt == FUNSCRIPT_FORMAT)
        folder = Path(self.window._last_dir())
        current = self.path.text().strip()
        if current:
            p = Path(current)
            folder = p if (p.is_dir() or not p.suffix) else p.parent
        self.path.setText(str(folder if self.stems.isChecked()
                              else folder / f"{self._base()}.{self.fmt}"))
        self._update_summary()
        self.adjustSize()
        metrics.fit_wrapped_height(self)

    def _update_summary(self, *_):
        self.summary_box.setVisible(self.fmt in AUDIO)
        if self.fmt not in AUDIO:
            return
        while self.summary.rowCount():
            self.summary.removeRow(0)
        rng = self.range.currentData()
        start, end = rng if rng else (None, None)
        render = self.doc.last_render
        n = render.n if render is not None else self.doc.project.n_pulses()
        for label, value in describe_audio(
                self.doc.project, n, start, end, self.audio_settings(), self.fmt,
                self.n_stems if self.stems.isChecked() else 0, self._panel_used()):
            text = QLabel(value)
            text.setWordWrap(True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.summary.addRow(f"{label}:", text)
        metrics.fit_wrapped_height(self)

    def _browse(self):
        if self.stems.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Folder for stems", self.path.text())
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export", self.path.text(),
                                                  f"*.{self.fmt}")
        if path:
            self.path.setText(path)

    def _panel_used(self) -> bool:
        return (self.use_panel.isChecked()
                and self.window.device_panel.current_key() == self.mode.currentText())

    def audio_settings(self) -> AudioSettings:
        panel = self.window.device_panel
        use = self._panel_used()
        return AudioSettings(
            mode=self.mode.currentText(), sample_rate=self.rate.currentData(),
            freq_hz=(panel.freq_min.value(), panel.freq_max.value()) if use else None,
            device_settings=panel.settings_dict() if use else None)

    def funscript_settings(self) -> FunscriptSettings:
        return FunscriptSettings(layout=self.fs_layout.currentData(),
                                 tolerance=self.fs_tol.value())

    # ── run ─────────────────────────────────────────────────────────────────
    def run(self) -> list[Path] | None:
        target = self.path.text().strip()
        if not target:
            return None
        rng = self.range.currentData()
        start, end = (rng if rng else (None, None))
        audio = self.audio_settings() if self.fmt in AUDIO else None
        funscript = None
        if self.fmt == FUNSCRIPT_FORMAT:
            funscript = self.funscript_settings()
            self.settings.setValue("export/funscript_layout", funscript.layout)
            self.settings.setValue("export/funscript_tolerance", funscript.tolerance)
        if audio is not None or self.stems.isChecked():
            self.window.playback.stop()   # this holds the GUI thread: nothing would read Esc
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self.stems.isChecked():
                written = export_stems(self.doc.project, target, self._base(), self.fmt,
                                       start, end, audio, self.doc.cache, funscript)
            else:
                path = Path(target)
                if path.suffix.lower() != f".{self.fmt}":
                    path = path.with_suffix(f".{self.fmt}")
                export_mix(self.doc.project, path, start, end, audio, self.doc.cache, funscript)
                written = written_paths(path, self.fmt, funscript)
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
