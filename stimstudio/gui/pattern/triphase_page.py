"""Pattern-editor page for triphase clips: Position Pad + volume/position/freq plots.

The pad shows the clip's trajectory and the position at the edit cursor (or the
playhead while playing). For a *Fixed position* clip the point can be dragged.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core.pulse import FS
from ...core.triphase import TRI_LANE_NAMES, Calibration, unit_to_xy
from .. import theme
from ..widgets.master_preview import TimeAxisItem
from ..widgets.position_pad import PositionPad
from .curves import Y_TICKS

TRI_COLORS = ("#4ec9b0", "#c586c0", "#9cdcfe", "#ce9178")


class TriphasePage(QWidget):
    editRequested = pyqtSignal(dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clip = None
        self.rendered: np.ndarray | None = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        left = QVBoxLayout()
        self.pad = PositionPad(editable=True)
        self.pad.dragFinished.connect(self._point_dropped)
        self.hint = QLabel()
        self.hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.hint.setWordWrap(True)
        left.addWidget(self.pad, 1)
        left.addWidget(self.hint)
        lay.addLayout(left, 0)

        self.plots_widget = pg.GraphicsLayoutWidget()
        lay.addWidget(self.plots_widget, 1)
        self.curves, self.plots = [], []
        for row, name in enumerate(TRI_LANE_NAMES):
            p = self.plots_widget.addPlot(row=row, col=0,
                                          axisItems={"bottom": TimeAxisItem(orientation="bottom")})
            p.setYRange(0, 1.02, padding=0)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", name.split()[-1], color=theme.TEXT_DIM, size="8pt")
            p.getAxis("left").setWidth(44)
            p.getAxis("left").setTicks([Y_TICKS])
            if row < 3:
                p.getAxis("bottom").setStyle(showValues=False)
            if row:
                p.setXLink(self.plots[0])
            self.curves.append(p.plot(pen=pg.mkPen(TRI_COLORS[row], width=1.3)))
            self.plots.append(p)
        self._time_lines = []
        for p in self.plots:
            line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.DANGER, width=1))
            line.setVisible(False)
            p.addItem(line, ignoreBounds=True)
            self._time_lines.append(line)

    def show_clip(self, clip, rendered: np.ndarray, cal: Calibration) -> None:
        self.clip, self.rendered = clip, rendered
        self.pad.set_calibration(cal)
        t = clip.trim_s + np.arange(len(rendered)) / FS
        for col, curve in enumerate(self.curves):
            curve.setData(t, rendered[:, col], skipFiniteCheck=True)
        self.plots[0].setXRange(clip.trim_s, clip.trim_s + max(0.1, clip.length_s), padding=0)
        xy = np.stack([unit_to_xy(rendered[:, 1]), unit_to_xy(rendered[:, 2])], 1) \
            if len(rendered) else None
        self.pad.set_trajectory(xy)
        src = clip.source
        self.pad.editable = getattr(src, "type_id", None) == "tri_point"
        self.hint.setText("Drag the point to place it." if self.pad.editable
                          else "Trajectory of the clip; the dot follows the cursor / playhead.")
        if self.pad.editable:
            self.pad.set_point(float(src.params.get("x", 0.0)), float(src.params.get("y", 0.0)))
        elif len(rendered):
            self._show_index(0)

    def set_time(self, seconds: float) -> None:
        """Move the dot to timeline time *seconds* (if inside the clip)."""
        clip, data = self.clip, self.rendered
        if clip is None or data is None or not len(data):
            return
        i = round((seconds - clip.start_s) * FS)
        inside = 0 <= i < len(data)
        for line in self._time_lines:
            line.setVisible(inside)
            if inside:
                line.setPos(clip.trim_s + i / FS)
        if inside and not self.pad.editable:
            self._show_index(i)

    def _show_index(self, i: int) -> None:
        row = self.rendered[i]
        self.pad.set_point(float(unit_to_xy(row[1])), float(unit_to_xy(row[2])),
                           volume=float(row[0]))

    def _point_dropped(self, x: float, y: float) -> None:
        clip = self.clip
        if clip is None or getattr(clip.source, "type_id", None) != "tri_point":
            return
        src = clip.source.clone()
        src.params = {**src.params, "x": round(x, 3), "y": round(y, 3)}
        self.editRequested.emit({"source": src}, "Move triphase point")
