"""Master output preview (pyqtgraph, OpenGL): amplitude and frequency, L/R overlaid.

Follows the song editor's visible time window; shows the edit cursor.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSize

from ...core.pulse import FS
from .. import theme


class TimeAxisItem(pg.AxisItem):
    """Bottom axis labelled mm:ss.mmm (ported from HWL Viewer)."""

    STEPS = (0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900,
             1800, 3600)

    def tickSpacing(self, minVal, maxVal, size):
        span = abs(maxVal - minVal)
        if span <= 0:
            return super().tickSpacing(minVal, maxVal, size)
        wanted = max(3, int(size / 90))
        major = self.STEPS[-1]
        for step in self.STEPS:
            if span / step <= wanted:
                major = step
                break
        while span / major > wanted:
            major *= 2
        idx = self.STEPS.index(major) if major in self.STEPS else 0
        minor = self.STEPS[idx - 1] if idx > 0 else major / 2
        return [(major, 0), (minor, 0)]

    def tickStrings(self, values, scale, spacing):
        decimals = 0
        while decimals < 3 and round(spacing, decimals) != round(spacing, 3):
            decimals += 1
        return [theme.format_time(v, decimals) for v in values]


class MasterPreview(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._plots: list[pg.PlotItem] = []
        self._curves: list[pg.PlotDataItem] = []
        self._cursors: list[pg.InfiniteLine] = []
        self._playheads: list[pg.InfiniteLine] = []
        axis_pen = pg.mkPen(theme.BORDER, width=1)
        for row, (label, cols) in enumerate((("Amp", (0, 1)), ("Freq", (2, 3)))):
            p = self.addPlot(row=row, col=0,
                             axisItems={"bottom": TimeAxisItem(orientation="bottom")})
            for side in ("left", "bottom"):
                p.getAxis(side).setPen(axis_pen)
                p.getAxis(side).setTextPen(pg.mkPen(theme.TEXT_DIM))
            p.getAxis("left").setWidth(44)
            p.getAxis("left").setTicks([[(0.0, "0"), (0.5, ".5"), (1.0, "1")]])
            p.setLabel("left", label, color=theme.TEXT_DIM, size="8pt")
            p.setYRange(0.0, 1.05, padding=0)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.1)
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if row == 0:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setXLink(self._plots[0])
            for col in cols:
                self._curves.append(p.plot(pen=pg.mkPen(theme.LANE_COLORS[col], width=1.1)))
            cur = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.CURSOR, width=1))
            p.addItem(cur, ignoreBounds=True)
            self._cursors.append(cur)
            ph = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(theme.DANGER, width=1.5))
            ph.setVisible(False)
            ph.setZValue(20)
            p.addItem(ph, ignoreBounds=True)
            self._playheads.append(ph)
            self._plots.append(p)

    def sizeHint(self) -> QSize:
        return QSize(800, 200)

    def set_pulses(self, master: np.ndarray) -> None:
        t = np.arange(len(master), dtype=np.float64) / FS
        for col, curve in enumerate(self._curves):
            curve.setData(t, np.ascontiguousarray(master[:, col]), skipFiniteCheck=True)

    def set_view_range(self, t0: float, t1: float) -> None:
        self._plots[0].setXRange(t0, max(t1, t0 + 0.1), padding=0)

    def set_cursor(self, t: float) -> None:
        for c in self._cursors:
            c.setPos(t)

    def set_playhead(self, t: float) -> None:
        for ph in self._playheads:
            ph.setVisible(t >= 0)
            if t >= 0:
                ph.setPos(t)
