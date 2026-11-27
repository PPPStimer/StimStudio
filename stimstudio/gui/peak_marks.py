"""Beat marks from a clip's peaks: pick a level on the clip's loudness curve.

The plot shows the clip's level (max of L/R amplitude, or the triphase volume)
with the level line (drag it) and the beats found, live.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QHBoxLayout, QLabel, QVBoxLayout)

from ..core.beats import PEAK_LABELS, PEAK_METHODS, detect_peaks, marks_from_peaks
from ..core.pulse import FS
from . import metrics, theme
from .pattern.curves import Y_TICKS

DEFAULTS = {"method": "cross", "level": 0.5, "min_gap_s": 0.1, "hysteresis": 0.05,
            "replace": False}


def peak_marks(signal, options: dict):
    """The marks *options* (see ``DEFAULTS``) find in *signal*."""
    o = {**DEFAULTS, **options}
    idx, heights = detect_peaks(signal, o["level"], o["method"], o["min_gap_s"],
                                o["hysteresis"])
    return marks_from_peaks(idx, heights)


class PeakMarksDialog(QDialog):
    def __init__(self, signal: np.ndarray, title: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Beat marks from peaks")
        metrics.fit_to_screen(self, 820, 420)
        self.signal = np.asarray(signal, dtype=np.float64)
        top = float(self.signal.max()) if len(self.signal) else 1.0
        lay = QVBoxLayout(metrics.scroll_body(self))
        if title:
            head = QLabel(title)
            head.setStyleSheet(f"color: {theme.TEXT_DIM};")
            lay.addWidget(head)

        self.plot = pg.PlotWidget(background=theme.BG)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setYRange(0.0, 1.02, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getAxis("left").setTicks([Y_TICKS])
        self.plot.getAxis("left").setWidth(36)
        t = np.arange(len(self.signal)) / FS
        self.curve = self.plot.plot(t, self.signal, pen=pg.mkPen(theme.LANE_COLORS[0], width=1.2),
                                    fillLevel=0.0, brush=pg.mkBrush(78, 201, 176, 40))
        self.ticks = self.plot.plot(pen=pg.mkPen("#ffffff", width=1.5), connect="pairs")
        self.level_line = pg.InfiniteLine(pos=round(top * 0.5, 3), angle=0, movable=True,
                                          pen=pg.mkPen(theme.ACCENT, width=1.5),
                                          hoverPen=pg.mkPen(theme.ACCENT, width=3))
        self.level_line.setBounds((0.0, 1.0))
        self.plot.addItem(self.level_line)
        self.hyst_line = pg.InfiniteLine(angle=0, movable=False,
                                         pen=pg.mkPen(theme.ACCENT, width=1,
                                                      style=Qt.PenStyle.DotLine))
        self.plot.addItem(self.hyst_line)
        lay.addWidget(self.plot, 1)

        row = QHBoxLayout()
        form = QFormLayout()
        self.method = QComboBox()
        for m in PEAK_METHODS:
            self.method.addItem(PEAK_LABELS[m], m)
        form.addRow("Method", self.method)
        self.level = QDoubleSpinBox()
        self.level.setRange(0.0, 1.0)
        self.level.setDecimals(3)
        self.level.setSingleStep(0.01)
        self.level.setValue(self.level_line.value())
        form.addRow("Level", self.level)
        row.addLayout(form)
        form2 = QFormLayout()
        self.min_gap = QDoubleSpinBox()
        self.min_gap.setRange(0.0, 10.0)
        self.min_gap.setDecimals(3)
        self.min_gap.setSingleStep(0.025)
        self.min_gap.setSuffix(" s")
        self.min_gap.setValue(DEFAULTS["min_gap_s"])
        form2.addRow("Min. interval", self.min_gap)
        self.hyst = QDoubleSpinBox()
        self.hyst.setRange(0.0, 1.0)
        self.hyst.setDecimals(3)
        self.hyst.setSingleStep(0.01)
        self.hyst.setValue(DEFAULTS["hysteresis"])
        self.hyst.setToolTip("Level crossing: the signal must fall this far below the level "
                             "before the next beat can start")
        form2.addRow("Hysteresis", self.hyst)
        row.addLayout(form2)
        col = QVBoxLayout()
        self.count = QLabel()
        self.replace = QCheckBox("Replace the clip (otherwise: new track below)")
        col.addWidget(self.count)
        col.addWidget(self.replace)
        row.addLayout(col, 1)
        lay.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.level_line.sigPositionChanged.connect(
            lambda: self.level.setValue(round(self.level_line.value(), 3)))
        self.level.valueChanged.connect(self._level_spin)
        for w in (self.method.currentIndexChanged, self.min_gap.valueChanged,
                  self.hyst.valueChanged):
            w.connect(self._update)
        self._update()

    def _level_spin(self, v: float) -> None:
        if abs(self.level_line.value() - v) > 1e-6:
            self.level_line.blockSignals(True)
            self.level_line.setValue(v)
            self.level_line.blockSignals(False)
        self._update()

    def options(self) -> dict:
        return {"method": self.method.currentData(), "level": self.level.value(),
                "min_gap_s": self.min_gap.value(), "hysteresis": self.hyst.value(),
                "replace": self.replace.isChecked()}

    def marks(self):
        return peak_marks(self.signal, self.options())

    def _update(self, *_):
        o = self.options()
        cross = o["method"] == "cross"
        self.hyst.setEnabled(cross)
        self.hyst_line.setVisible(cross)
        self.hyst_line.setValue(max(0.0, o["level"] - o["hysteresis"]))
        marks = self.marks()
        x = np.repeat([m.t_s for m in marks], 2)
        y = np.ravel([[0.0, m.velocity] for m in marks])
        self.ticks.setData(x, y)
        self.count.setText(f"<b>{len(marks)}</b> beat marks")
