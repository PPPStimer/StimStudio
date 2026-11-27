"""Output monitor for the audio outputs: oscilloscope, spectrum and level meters.

Reads the sink's :class:`~stimstudio.outputs.audio.ScopeTap` — the samples the
sound card actually received, after power, mute, soft start and fades — on a
30 Hz timer that only runs while the monitor is on screen. Drawn with
pyqtgraph (OpenGL, configured globally by ``gui.gl``) in one layout widget.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize, QTimer
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...audio.monitor import (FLOOR_DB, dominant_hz, find_trigger, levels_db, phase_between,
                              spectrum_db)
from .. import theme
from ..settings import app_settings

TICK_MS = 33
WINDOWS_MS = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
TRIGGERS = (("Off", None), ("A ↑", 0), ("B ↑", 1))
VIEWS = ("Scope", "Spectrum", "XY")
XY_COLOR = "#9cdcfe"
NFFT = 4096
RMS_S = 0.3
PEAK_HOLD_S = 1.5
METER_MIN_DB = -60.0
SPECTRUM_SMOOTH = 0.5            # weight of the previous frame
COLORS = (theme.LANE_COLORS[0], theme.LANE_COLORS[1])   # A, B as on the lanes


def _style(plot: pg.PlotItem) -> None:
    axis_pen = pg.mkPen(theme.BORDER, width=1)
    for side in ("left", "bottom"):
        plot.getAxis(side).setPen(axis_pen)
        plot.getAxis(side).setTextPen(pg.mkPen(theme.TEXT_DIM))
    plot.setMouseEnabled(x=False, y=False)
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.showGrid(x=True, y=True, alpha=0.12)


class OutputMonitor(QWidget):
    def __init__(self, playback, parent=None):
        super().__init__(parent)
        self.pb = playback
        self.settings = app_settings()
        s = self.settings

        # controls
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(4)
        self.view = QComboBox()
        self.view.addItems(VIEWS)
        self.view.setCurrentText(str(s.value("monitor/view", VIEWS[0])))
        self.view.setToolTip("Oscilloscope, spectrum or XY (A across, B up: the Lissajous "
                             "figure shows the phase between the channels) of the signal "
                             "sent to the sound card")
        bar.addWidget(self.view)
        self.window_ms = QComboBox()
        for ms in WINDOWS_MS:
            self.window_ms.addItem(f"{ms} ms" if ms < 1000 else f"{ms // 1000} s", ms)
        idx = self.window_ms.findData(int(s.value("monitor/window_ms", 10)))
        self.window_ms.setCurrentIndex(max(0, idx))
        self.window_ms.setToolTip("Time span shown by the oscilloscope")
        bar.addWidget(self.window_ms)
        self.trigger = QComboBox()
        for label, ch in TRIGGERS:
            self.trigger.addItem(label, ch)
        self.trigger.setCurrentIndex(max(0, min(len(TRIGGERS) - 1,
                                                int(s.value("monitor/trigger", 1)))))
        self.trigger.setToolTip("Start the trace on a rising zero crossing of this channel")
        bar.addWidget(self.trigger)
        self.hold = QCheckBox("Hold")
        self.hold.setToolTip("Freeze the trace (the meters keep running)")
        bar.addWidget(self.hold)
        bar.addStretch()

        self.readout = QLabel()
        self.readout.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.readout.setWordWrap(True)

        # plots
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setMinimumHeight(190)
        self.scope_plots: list[pg.PlotItem] = []
        self.scope_curves: list[pg.PlotDataItem] = []
        for ch, name in enumerate("AB"):
            p = pg.PlotItem()
            _style(p)
            p.setYRange(-1.05, 1.05, padding=0)
            p.getAxis("left").setTicks([[(-1.0, "-1"), (0.0, "0"), (1.0, "1")]])
            p.getAxis("left").setWidth(30)
            p.setLabel("left", name, color=COLORS[ch], size="8pt")
            if ch == 0:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "ms", color=theme.TEXT_DIM, size="8pt")
            curve = p.plot(pen=pg.mkPen(COLORS[ch], width=1.2))
            # a 1 s window is 48 000+ samples: draw per-pixel peaks, not every sample
            curve.setDownsampling(auto=True, method="peak")
            self.scope_curves.append(curve)
            self.scope_plots.append(p)
        self.scope_plots[1].setXLink(self.scope_plots[0])

        self.spec_plot = pg.PlotItem()
        _style(self.spec_plot)
        self.spec_plot.setLogMode(x=True, y=False)
        self.spec_plot.setYRange(-100.0, 3.0, padding=0)
        self.spec_plot.getAxis("left").setWidth(30)
        self.spec_plot.setLabel("bottom", "Hz", color=theme.TEXT_DIM, size="8pt")
        self.spec_curves = [self.spec_plot.plot(pen=pg.mkPen(c, width=1.1)) for c in COLORS]

        self.xy_plot = pg.PlotItem()
        _style(self.xy_plot)
        self.xy_plot.setXRange(-1.05, 1.05, padding=0)
        self.xy_plot.setYRange(-1.05, 1.05, padding=0)
        self.xy_plot.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.xy_plot.getAxis(side).setTicks([[(-1.0, "-1"), (0.0, "0"), (1.0, "1")]])
        self.xy_plot.getAxis("left").setWidth(30)
        self.xy_plot.setLabel("bottom", "A", color=COLORS[0], size="8pt")
        self.xy_plot.setLabel("left", "B", color=COLORS[1], size="8pt")
        self.xy_curve = self.xy_plot.plot(pen=pg.mkPen(XY_COLOR, width=1.1))

        self.meter_plot = pg.PlotItem()
        _style(self.meter_plot)
        self.meter_plot.showGrid(x=False, y=True, alpha=0.12)
        self.meter_plot.setXRange(-0.6, 1.6, padding=0)
        self.meter_plot.setYRange(METER_MIN_DB, 0.0, padding=0)
        self.meter_plot.getAxis("left").setWidth(28)
        self.meter_plot.getAxis("left").setTicks(
            [[(v, str(v)) for v in (0, -6, -12, -24, -36, -48, -60)]])
        self.meter_plot.getAxis("bottom").setTicks([[(0, "A"), (1, "B")]])
        self.meter_plot.setMaximumWidth(78)
        self.meter_bars = pg.BarGraphItem(x=[0, 1], height=[0, 0], y0=METER_MIN_DB, width=0.7,
                                          brushes=[pg.mkBrush(c) for c in COLORS],
                                          pens=[pg.mkPen(None)] * 2)
        self.meter_plot.addItem(self.meter_bars)
        self.peak_ticks = pg.BarGraphItem(x=[0, 1], height=[0.8, 0.8],
                                          y0=[METER_MIN_DB, METER_MIN_DB], width=0.7,
                                          brush=pg.mkBrush(theme.TEXT), pen=pg.mkPen(None))
        self.meter_plot.addItem(self.peak_ticks)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addLayout(bar)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.readout)

        self._spec_avg: list[np.ndarray | None] = [None, None]
        self._peak_hold = [[FLOOR_DB, 0.0], [FLOOR_DB, 0.0]]   # (dB, time) per channel
        self._layout_view()

        self.view.currentIndexChanged.connect(self._on_view)
        self.window_ms.currentIndexChanged.connect(self._on_options)
        self.trigger.currentIndexChanged.connect(self._on_options)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.tick)

    def sizeHint(self) -> QSize:
        return QSize(300, 260)

    # ── layout / options ────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return self.view.currentText()

    def _layout_view(self) -> None:
        ci = self.glw.ci
        for item in list(ci.items):
            ci.removeItem(item)
        if self.mode == "Spectrum":
            ci.addItem(self.spec_plot, row=0, col=0, rowspan=2)
        elif self.mode == "XY":
            ci.addItem(self.xy_plot, row=0, col=0, rowspan=2)
        else:
            ci.addItem(self.scope_plots[0], row=0, col=0)
            ci.addItem(self.scope_plots[1], row=1, col=0)
        ci.addItem(self.meter_plot, row=0, col=1, rowspan=2)
        self.window_ms.setEnabled(self.mode in ("Scope", "XY"))
        self.trigger.setEnabled(self.mode == "Scope")

    def _on_view(self, *_):
        self.settings.setValue("monitor/view", self.mode)
        self._spec_avg = [None, None]
        self._layout_view()

    def _on_options(self, *_):
        self.settings.setValue("monitor/window_ms", self.window_ms.currentData())
        self.settings.setValue("monitor/trigger", self.trigger.currentIndex())

    # ── timer ───────────────────────────────────────────────────────────────
    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self.tick()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def tap(self):
        sink = getattr(self.pb, "sink", None)
        return getattr(sink, "scope", None)

    def tick(self) -> None:
        tap = self.tap()
        if tap is None:
            sr = 48000
            need = max(NFFT, int(sr * RMS_S))
            data = np.zeros((need, 2), dtype=np.float32)
        else:
            sr = tap.sr
            win_n = int(sr * self.window_ms.currentData() / 1000.0)
            need = max(NFFT, int(sr * RMS_S), win_n * 3)
            data = tap.latest(need)
        if not self.hold.isChecked():
            if self.mode == "Spectrum":
                self._update_spectrum(data, sr)
            elif self.mode == "XY":
                self._update_xy(data, sr)
            else:
                self._update_scope(data, sr)
        self._update_meters(data[-int(sr * RMS_S):])      # meters run even while held

    # ── views ───────────────────────────────────────────────────────────────
    def _update_scope(self, data: np.ndarray, sr: int) -> None:
        ms = self.window_ms.currentData()
        win_n = max(2, int(sr * ms / 1000.0))
        start = len(data) - win_n
        ch = self.trigger.currentData()
        if ch is not None:
            # search the newest 2 windows so the trace moves as little as possible
            search = data[-(win_n * 2):, ch]
            found = find_trigger(search, win_n)
            if found is not None:
                start = len(data) - len(search) + found
        block = data[start:start + win_n]
        t = np.arange(len(block), dtype=np.float64) * (1000.0 / sr)
        for c, curve in enumerate(self.scope_curves):
            curve.setData(t, block[:, c], skipFiniteCheck=True)
        self.scope_plots[0].setXRange(0.0, ms, padding=0)

    def _update_spectrum(self, data: np.ndarray, sr: int) -> None:
        peaks = []
        for c, curve in enumerate(self.spec_curves):
            freqs, db = spectrum_db(data[:, c], sr, NFFT)
            prev = self._spec_avg[c]
            if prev is not None and len(prev) == len(db):
                db = SPECTRUM_SMOOTH * prev + (1.0 - SPECTRUM_SMOOTH) * db
            self._spec_avg[c] = db
            curve.setData(freqs, db, skipFiniteCheck=True)
            peaks.append(dominant_hz(freqs, db))
        self.spec_plot.setXRange(math.log10(20.0), math.log10(sr / 2.0), padding=0)
        self._spectrum_peaks = peaks

    def _update_xy(self, data: np.ndarray, sr: int) -> None:
        """A on X, B on Y over the window: a line = in phase, a circle = 90°, the other
        diagonal = opposite phase; an ellipse's tilt and width tell the phase between."""
        win_n = max(2, int(sr * self.window_ms.currentData() / 1000.0))
        block = data[-win_n:]
        self.xy_curve.setData(block[:, 0], block[:, 1], skipFiniteCheck=True)
        self._xy_phase = phase_between(data[:, 0], data[:, 1], sr, NFFT)

    def _update_meters(self, block: np.ndarray) -> None:
        rms, peak = levels_db(block)
        now = time.monotonic()
        for c in range(2):
            held, at = self._peak_hold[c]
            if peak[c] >= held or now - at > PEAK_HOLD_S:
                self._peak_hold[c] = [float(peak[c]), now]
        heights = [max(0.0, float(r) - METER_MIN_DB) for r in rms]
        self.meter_bars.setOpts(height=heights)
        ticks = [max(METER_MIN_DB, min(0.0, h[0])) - 0.4 for h in self._peak_hold]
        self.peak_ticks.setOpts(y0=ticks)
        text = "   ".join(f"{n}  RMS {self._fmt(rms[c])}  peak {self._fmt(peak[c])}"
                           for c, n in enumerate("AB"))
        if self.mode == "Spectrum":
            hz = getattr(self, "_spectrum_peaks", [None, None])
            text += "   ·   " + " · ".join(
                f"{n} {h:.0f} Hz" if h else f"{n} —" for n, h in zip("AB", hz))
        elif self.mode == "XY":
            ph = getattr(self, "_xy_phase", None)
            text += ("   ·   B vs A: —" if ph is None else
                     f"   ·   B vs A at {ph[0]:.0f} Hz: {ph[1]:+.0f}°".replace("-", "−"))
        self.readout.setText(text)

    @staticmethod
    def _fmt(db: float) -> str:
        return "−∞" if db <= FLOOR_DB + 0.01 else f"{db:.1f} dB".replace("-", "−")
