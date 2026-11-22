"""Triphase audio: a phase-coded three-phase stereo file → a stim clip.

Open (or drop) the file; it is demodulated once on the thread pool
(``audio.triphase_decode``, cached per file). Every option afterwards only
recombines the per-pulse result, so the preview follows at once:

* **Signal** — an unbroken carrier or bursts of carrier (found automatically);
* **Output** — a standard track whose A/B amplitudes are the channel levels
  (what a triphase track sends to Howl) or the sensation at electrodes L / R,
  or a triphase track with the volume and position recovered;
* **Frequency** — the carrier pitch, the burst rate, or a constant, mapped
  from a range in Hz onto the 0..1 lane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QVBoxLayout)

from ...audio import triphase_decode as td
from ...audio.loader import AUDIO_EXTENSIONS, is_audio_file
from ...core.pulse import FS
from ...core.triphase import TRI_LANE_NAMES, Calibration
from .. import theme
from ..pattern.curves import Y_TICKS, lane_pen
from ..settings import app_settings
from ..widgets.master_preview import TimeAxisItem
from .base import ToolWindow

SIGNALS = (("auto", "Auto"), ("continuous", "Unbroken carrier"), ("pulsed", "Bursts"))
OUTPUTS = (("channels", "Standard track — A/B = channel levels"),
           ("electrodes", "Standard track — A/B = electrodes L / R"),
           ("triphase", "Triphase track — volume + position"))
FREQ_SOURCES = (("carrier", "Carrier pitch"), ("burst", "Burst rate"), ("constant", "Constant"))
AB_LANE_NAMES = ("Amp A", "Amp B", "Freq A", "Freq B")
GOOD_COHERENCE = 0.9
POOR_COHERENCE = 0.6
SETTINGS_GROUP = "tools/triphase_audio"


def _combo(items, tip: str) -> QComboBox:
    c = QComboBox()
    for key, label in items:
        c.addItem(label, key)
    c.setToolTip(tip)
    return c


def _hz_spin(value: float) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(0.0, 20000.0)
    w.setDecimals(2)
    w.setSingleStep(1.0)
    w.setValue(value)
    w.setSuffix(" Hz")
    return w


class TriphaseAudioWindow(ToolWindow):
    title = "Triphase audio → stim clip"

    def __init__(self, main_window):
        super().__init__(main_window)
        self.path: str | None = None
        self.start_s: float | None = None        # where the clip lands (None: edit cursor)
        self.analysis: td.TriAudioAnalysis | None = None
        self._loading = True
        self._build()
        self._restore()
        self._loading = False
        self.setAcceptDrops(True)
        self.resize(980, 640)
        self._sync()

    # ── layout ──────────────────────────────────────────────────────────────
    def _build(self) -> None:
        lay = QVBoxLayout(self)
        src = QHBoxLayout()
        src.addWidget(QLabel("Audio:"))
        self.source_label = QLabel("—")
        self.source_label.setTextFormat(Qt.TextFormat.PlainText)
        src.addWidget(self.source_label, 1)
        self.file_btn = QPushButton("Open file…")
        self.file_btn.setToolTip("A stereo three-phase file (or drop one on this window)")
        self.file_btn.clicked.connect(lambda: self.open_file())
        src.addWidget(self.file_btn)
        lay.addLayout(src)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.info)

        # drawn like the pattern editor: every pulse, peak downsampling, clip to view
        self.plot = pg.PlotWidget(background=theme.BG,
                                  axisItems={"bottom": TimeAxisItem(orientation="bottom")})
        self.plot.setMinimumHeight(220)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setYRange(0.0, 1.02, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getAxis("left").setTicks([Y_TICKS])
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.plot.addLegend(offset=(8, 8))
        self.curves = [self.plot.plot(pen=lane_pen(i, width=1.2), name=AB_LANE_NAMES[i])
                       for i in range(4)]
        lay.addWidget(self.plot, 1)

        box = QGroupBox("Clip")
        row = QHBoxLayout(box)
        f1 = QFormLayout()
        self.signal = _combo(SIGNALS, "Unbroken carrier: level and phase averaged over each "
                                      "1/40 s.\nBursts: the level is each burst's peak, held "
                                      "until the next burst.")
        f1.addRow("Signal", self.signal)
        self.output = _combo(OUTPUTS, (
            "Channel levels: A = |L|, B = |R| — what a triphase track sends to Howl "
            "(A → L, B → R, both returns on N).\n"
            "Electrodes L / R: the current through electrodes L and R, phase included — "
            "closer to how the file feels on L and R, the third electrode N is left out.\n"
            "Triphase track: volume and position recovered (Track → Triphase calibration "
            "is taken into account); plays with true phase on continuous audio."))
        f1.addRow("Output", self.output)
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0.0, 2.0)
        self.smooth.setDecimals(2)
        self.smooth.setSingleStep(0.025)
        self.smooth.setSuffix(" s")
        self.smooth.setToolTip("Moving average of level and phase")
        f1.addRow("Smoothing", self.smooth)
        self.normalize = QCheckBox("Scale the maximum to 1")
        self.normalize.setChecked(True)
        self.normalize.setToolTip("Otherwise the file's full scale is 1")
        f1.addRow(self.normalize)
        row.addLayout(f1, 3)

        f2 = QFormLayout()
        self.freq_source = _combo(FREQ_SOURCES, "What the frequency lanes follow")
        f2.addRow("Frequency", self.freq_source)
        rng = QHBoxLayout()
        lo, hi = td.default_hz_range("carrier")
        self.hz_lo, self.hz_hi = _hz_spin(lo), _hz_spin(hi)
        for w in (self.hz_lo, self.hz_hi):
            w.setToolTip("These frequencies become 0 and 1 on the lane (Carrier pitch: "
                         "set so the lowest carrier found in the file is 0.1, the highest 0.9)")
        rng.addWidget(self.hz_lo)
        rng.addWidget(QLabel(".."))
        rng.addWidget(self.hz_hi)
        f2.addRow("0..1 =", rng)
        self.const_hz = _hz_spin(50.0)
        f2.addRow("Constant", self.const_hz)
        row.addLayout(f2, 2)

        self.create_btn = QPushButton("Create clip")
        self.create_btn.setDefault(True)
        self.create_btn.setToolTip("A new track with the clip at the edit cursor")
        self.create_btn.clicked.connect(self.create_clip)
        row.addWidget(self.create_btn, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addWidget(box)

        for c in (self.signal, self.output):
            c.currentIndexChanged.connect(lambda _i: self._changed())
        self.freq_source.currentIndexChanged.connect(lambda _i: self._source_changed())
        for w in (self.smooth, self.hz_lo, self.hz_hi, self.const_hz):
            w.valueChanged.connect(lambda _v: self._changed())
        self.normalize.toggled.connect(lambda _on: self._changed())

    # ── settings ────────────────────────────────────────────────────────────
    def _restore(self) -> None:
        s = app_settings()
        s.beginGroup(SETTINGS_GROUP)
        for combo, key in ((self.signal, "signal"), (self.output, "output"),
                           (self.freq_source, "freq_source")):
            i = combo.findData(s.value(key, combo.currentData()))
            if i >= 0:
                combo.setCurrentIndex(i)
        self.smooth.setValue(float(s.value("smooth", 0.0)))
        self.normalize.setChecked(str(s.value("normalize", "true")).lower() == "true")
        self.const_hz.setValue(float(s.value("const_hz", 50.0)))
        lo, hi = td.default_hz_range(self.freq_source.currentData())
        self.hz_lo.setValue(float(s.value("hz_lo", lo)))
        self.hz_hi.setValue(float(s.value("hz_hi", hi)))
        s.endGroup()

    def _save(self) -> None:
        s = app_settings()
        s.beginGroup(SETTINGS_GROUP)
        o = self.options()
        for key in ("signal", "output", "freq_source"):
            s.setValue(key, o["mode" if key == "signal" else key])
        s.setValue("smooth", o["smooth_s"])
        s.setValue("normalize", "true" if o["normalize"] else "false")
        s.setValue("const_hz", o["const_hz"])
        s.setValue("hz_lo", o["hz_range"][0])
        s.setValue("hz_hi", o["hz_range"][1])
        s.endGroup()

    # ── source ──────────────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and is_audio_file(urls[0].toLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.open_file(event.mimeData().urls()[0].toLocalFile())

    def open_file(self, path: str | None = None, start_s: float | None = None) -> bool:
        """Decode *path* (asked for when None); the clip will land at *start_s*
        (File → Import audio passes the drop position), else at the edit cursor."""
        if path is None:
            exts = " ".join(f"*{e}" for e in AUDIO_EXTENSIONS)
            path, _ = QFileDialog.getOpenFileName(self, "Three-phase audio",
                                                  self.main_window._last_dir(),
                                                  f"Audio ({exts});;All files (*)")
            if not path:
                return False
        self.main_window._remember_dir(path)
        from ..audio_import import AudioJob, run_job
        try:
            analysis = run_job(self, AudioJob(path, "triphase"),
                               f"Decoding {Path(path).name}", self.title)
        except RuntimeError as exc:
            QMessageBox.warning(self, self.title, f"Could not decode {path}:\n{exc}")
            return False
        if analysis is None:
            return False
        self.set_analysis(path, analysis)
        self.start_s = start_s
        return True

    def set_analysis(self, path: str, analysis: td.TriAudioAnalysis) -> None:
        self.path, self.analysis = path, analysis
        self.source_label.setText(Path(path).name)
        auto = self.signal.findData("auto")
        self.signal.setItemText(auto, "Auto ({})".format(
            "bursts" if analysis.detected_mode == "pulsed" else "unbroken carrier"))
        self.plot.setXRange(0.0, max(analysis.duration_s, 0.1), padding=0.01)
        if self.freq_source.currentData() == "carrier":
            self._set_range(*self.hz_range_for("carrier"))
            self._save()
        self._sync()

    # ── options ─────────────────────────────────────────────────────────────
    def options(self) -> dict:
        return {"mode": self.signal.currentData(), "output": self.output.currentData(),
                "freq_source": self.freq_source.currentData(),
                "hz_range": (self.hz_lo.value(), self.hz_hi.value()),
                "const_hz": self.const_hz.value(), "normalize": self.normalize.isChecked(),
                "smooth_s": self.smooth.value()}

    def calibration(self) -> Calibration:
        return Calibration.from_dict(self.doc.project.triphase_cal)

    def hz_range_for(self, freq_source: str) -> tuple[float, float]:
        """Carrier: the range that puts the lowest carrier found in the file at 0.1
        and the highest at 0.9 on the lane; otherwise the output's usual range."""
        fitted = (td.fitted_hz_range(self.analysis, freq_source)
                  if freq_source == "carrier" and self.analysis is not None else None)
        if fitted is None:
            return td.default_hz_range(freq_source)
        return round(fitted[0], 2), round(fitted[1], 2)    # the spin boxes' 0.01 Hz

    def _set_range(self, lo: float, hi: float) -> None:
        for w, v in ((self.hz_lo, lo), (self.hz_hi, hi)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _source_changed(self) -> None:
        if not self._loading:
            self._set_range(*self.hz_range_for(self.freq_source.currentData()))
        self._changed()

    def _changed(self) -> None:
        if self._loading:
            return
        self._save()
        self._sync()

    def pulses(self) -> np.ndarray | None:
        if self.analysis is None:
            return None
        return td.to_pulses(self.analysis, cal=self.calibration(), **self.options())

    def _sync(self) -> None:
        self.const_hz.setEnabled(self.freq_source.currentData() == "constant")
        self.create_btn.setEnabled(self.analysis is not None)
        self._update_info()
        self.update_preview()

    def _update_info(self) -> None:
        an = self.analysis
        if an is None:
            self.info.setText(f"<span style='color:{theme.TEXT_DIM}'>Open or drop a stereo "
                              "three-phase file: one carrier in both channels, the position "
                              "in their levels and phase difference.</span>")
            return
        coh = an.mean_coherence
        parts = [theme.format_time(an.duration_s), f"{an.sr} Hz",
                 "bursts" if an.detected_mode == "pulsed" else "unbroken carrier"]
        if an.carrier_median:
            parts.append(f"carrier ≈ {an.carrier_median:.0f} Hz")
        if an.burst_median:
            parts.append(f"bursts ≈ {an.burst_median:.1f} / s, "
                         f"{an.duty * 100:.0f} % of the time on")
        parts.append(f"phase coherence {coh:.2f}")
        text = " · ".join(parts)
        if coh < POOR_COHERENCE:
            text += (f"<br><span style='color:{theme.DANGER}'>The channels are not one "
                     "phase-shifted carrier — this does not look like a three-phase file; "
                     "phase and position will be noise.</span>")
        elif coh < GOOD_COHERENCE:
            text += (f"<br><span style='color:{theme.TEXT_DIM}'>Partly coherent: noise or "
                     "several carriers in the file — check the preview.</span>")
        if self.freq_source.currentData() == "burst" and not an.burst_median:
            text += (f"<br><span style='color:{theme.TEXT_DIM}'>No bursts found: the burst "
                     "rate falls back to the middle of the range.</span>")
        self.info.setText(text)

    def update_preview(self) -> None:
        p = self.pulses()
        names = TRI_LANE_NAMES if self.output.currentData() == "triphase" else AB_LANE_NAMES
        legend = self.plot.plotItem.legend
        for curve, name in zip(self.curves, names):
            label = legend.getLabel(curve) if legend is not None else None
            if label is not None:
                label.setText(name)
        if p is None or not len(p):
            for c in self.curves:
                c.setData([], [])
            return
        t = np.arange(len(p)) / FS
        for i, c in enumerate(self.curves):
            c.setData(t, p[:, i], skipFiniteCheck=True)

    # ── output ──────────────────────────────────────────────────────────────
    def result_data(self) -> dict:
        return {"path": self.path, "pulses": self.pulses(),
                "triphase": self.output.currentData() == "triphase", "start_s": self.start_s}

    def create_clip(self):
        if self.analysis is None:
            return None
        return self.main_window.add_triphase_audio(self.result_data())
