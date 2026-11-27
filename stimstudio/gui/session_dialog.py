"""Session builder dialog: choose length, peak and style, preview the arc, create."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout)

from ..core import session as ss
from . import metrics, theme

PHASE_COLORS = {"warmup": (78, 201, 176, 40), "build": (86, 156, 214, 34),
                "peak": (241, 76, 76, 46), "cooldown": (197, 134, 192, 40)}


class SessionDialog(QDialog):
    def __init__(self, parent=None, bpm: float = 120.0, default_duration_s: float = 1200.0,
                 start_s: float = 0.0, markers=()):
        super().__init__(parent)
        self.setWindowTitle("Session builder")
        self.bpm = bpm
        self.markers = sorted(float(m) for m in markers)
        self.plan: ss.SessionPlan | None = None
        lay = QVBoxLayout(metrics.scroll_body(self))

        form = QFormLayout()
        self.minutes = QDoubleSpinBox()
        self.minutes.setRange(1.0, 240.0)
        self.minutes.setDecimals(1)
        self.minutes.setSuffix(" min")
        self.minutes.setValue(max(1.0, round(default_duration_s / 60.0, 1)))
        form.addRow("Length", self.minutes)
        self.start = QDoubleSpinBox()
        self.start.setRange(0.0, 36000.0)
        self.start.setDecimals(1)
        self.start.setSuffix(" s")
        self.start.setValue(start_s)
        form.addRow("Starts at", self.start)
        self.style = QComboBox()
        for key in ss.STYLES:
            self.style.addItem(ss.STYLE_LABELS[key], key)
        form.addRow("Style", self.style)
        self.peak = QDoubleSpinBox()
        self.peak.setRange(0.1, 1.0)
        self.peak.setSingleStep(0.05)
        self.peak.setValue(0.85)
        self.peak.setToolTip("Highest level of the arc (0..1, before output power)")
        form.addRow("Peak level", self.peak)
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 12)
        self.cycles.setValue(3)
        form.addRow("Build-up waves", self.cycles)
        frac = QHBoxLayout()
        self.warm = QSpinBox()
        self.cool = QSpinBox()
        for w, v in ((self.warm, 12), (self.cool, 12)):
            w.setRange(3, 40)
            w.setSuffix(" %")
            w.setValue(v)
        frac.addWidget(QLabel("warm-up"))
        frac.addWidget(self.warm)
        frac.addWidget(QLabel("cool-down"))
        frac.addWidget(self.cool)
        form.addRow("Phases", frac)
        self.triphase = QCheckBox("Triphase tracks (three electrodes)")
        form.addRow(self.triphase)
        self.bpm_sync = QCheckBox(f"Clip lengths in whole bars ({bpm:g} BPM)")
        self.bpm_sync.setChecked(True)
        form.addRow(self.bpm_sync)
        self.align = QCheckBox(f"Snap phases to markers ({len(self.markers)} in the project)")
        self.align.setEnabled(bool(self.markers))
        self.align.setChecked(bool(self.markers))
        form.addRow(self.align)
        seed_row = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        reroll = QPushButton("Re-roll")
        reroll.setToolTip("Another variation with the same settings")
        reroll.clicked.connect(lambda: self.seed.setValue((self.seed.value() * 7919 + 13)
                                                          % 1000000))
        seed_row.addWidget(self.seed)
        seed_row.addWidget(reroll)
        form.addRow("Variation", seed_row)
        lay.addLayout(form)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(170)
        self.plot.setBackground(theme.BG)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setYRange(0, 1.05)
        self.plot.getAxis("bottom").setLabel("minutes")
        self.arc_curve = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self._extras: list = []
        lay.addWidget(self.plot)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        for w in (self.minutes, self.peak, self.start):
            w.valueChanged.connect(self.update_plan)
        for w in (self.cycles, self.warm, self.cool, self.seed):
            w.valueChanged.connect(self.update_plan)
        self.style.currentIndexChanged.connect(self.update_plan)
        for w in (self.triphase, self.bpm_sync, self.align):
            w.toggled.connect(self.update_plan)
        metrics.fit_to_screen(self, 560, 640)
        self.update_plan()

    def spec(self) -> ss.SessionSpec:
        return ss.SessionSpec(
            duration_s=self.minutes.value() * 60.0, peak=self.peak.value(),
            style=self.style.currentData(), cycles=self.cycles.value(),
            warmup_frac=self.warm.value() / 100.0, cooldown_frac=self.cool.value() / 100.0,
            triphase=self.triphase.isChecked(), seed=self.seed.value(),
            align_to_markers=self.align.isChecked(), bpm_sync=self.bpm_sync.isChecked())

    def update_plan(self) -> None:
        start = self.start.value()
        rel = [m - start for m in self.markers]
        self.plan = ss.build_session(self.spec(), self.bpm, rel)
        for item in self._extras:
            self.plot.removeItem(item)
        self._extras = []
        d = self.plan.spec.duration_s
        t = np.linspace(0.0, d, 600)
        self.arc_curve.setData(t / 60.0, self.plan.arc_at(t))
        for ph in self.plan.phases:
            region = pg.LinearRegionItem((ph.t0 / 60.0, ph.t1 / 60.0), movable=False,
                                         brush=pg.mkBrush(*PHASE_COLORS[ph.kind]),
                                         pen=pg.mkPen(None))
            region.setZValue(-10)
            self.plot.addItem(region)
            label = pg.TextItem(ph.name, color=theme.TEXT_DIM, anchor=(0, 0))
            label.setPos(ph.t0 / 60.0, 1.04)
            self.plot.addItem(label)
            self._extras += [region, label]
        for c in self.plan.clips[1:]:
            line = pg.InfiniteLine(pos=c.start_s / 60.0, angle=90,
                                   pen=pg.mkPen(theme.TEXT_DIM, width=1,
                                                style=Qt.PenStyle.DotLine))
            self.plot.addItem(line)
            self._extras.append(line)
        self.plot.setXRange(0, d / 60.0, padding=0.01)
        self.summary.setText(ss.describe(self.plan) + ". Two tracks with crossfades, an "
                             "intensity envelope, a slew limiter and phase markers.")
