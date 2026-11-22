"""Triphase Position Pad: the electrode triangle, a trajectory and the current
position, with live electrode levels. Hardware-rendered (QOpenGLWidget) like
every other plot; optionally the point can be dragged.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QSurfaceFormat
from PyQt6.QtWidgets import QWidget

from ...core.triphase import (DEFAULT_CAL, ELECTRODE_ANGLES_DEG, ELECTRODES, Calibration,
                              electrode_levels)
from .. import theme
from ..gl import MSAA_SAMPLES, gl_enabled

if gl_enabled():
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget as _Base
else:
    _Base = QWidget

ELECTRODE_COLORS = {"N": "#dcdcaa", "L": "#4ec9b0", "R": "#569cd6"}


class PositionPad(_Base):
    pointDragged = pyqtSignal(float, float)    # x, y (−1..1) while dragging
    dragFinished = pyqtSignal(float, float)

    def __init__(self, parent=None, editable: bool = False):
        super().__init__(parent)
        if gl_enabled():
            fmt = QSurfaceFormat.defaultFormat()
            fmt.setSamples(MSAA_SAMPLES)
            self.setFormat(fmt)
        self.editable = editable
        self.cal: Calibration = DEFAULT_CAL
        self.trajectory: np.ndarray | None = None   # (n, 2) x, y
        self.point: tuple[float, float] | None = None
        self.volume = 1.0
        self.show_levels = True
        self._dragging = False
        self.setMinimumSize(180, 180)
        self.setMouseTracking(False)

    def sizeHint(self) -> QSize:
        return QSize(240, 240)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return w

    # ── data ────────────────────────────────────────────────────────────────
    def set_calibration(self, cal: Calibration) -> None:
        self.cal = cal
        self.update()

    def set_trajectory(self, xy: np.ndarray | None) -> None:
        self.trajectory = xy
        self.update()

    def set_point(self, x: float | None, y: float | None = None, volume: float = 1.0) -> None:
        self.point = None if x is None else (float(x), float(y))
        self.volume = float(volume)
        self.update()

    # ── geometry ────────────────────────────────────────────────────────────
    def _frame(self) -> tuple[QPointF, float]:
        side = min(self.width(), self.height())
        return QPointF(self.width() / 2, self.height() / 2), side / 2 - 26

    def _to_screen(self, x: float, y: float) -> QPointF:
        c, r = self._frame()
        return QPointF(c.x() + x * r, c.y() - y * r)

    def _from_screen(self, p: QPointF) -> tuple[float, float]:
        c, r = self._frame()
        x, y = (p.x() - c.x()) / r, (c.y() - p.y()) / r
        d = math.hypot(x, y)
        if d > 1.0:
            x, y = x / d, y / d
        return x, y

    # ── painting ────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(theme.BG))
        c, r = self._frame()
        # disk and electrode triangle
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QBrush(QColor("#202022")))
        p.drawEllipse(c, r, r)
        tri = [self._to_screen(math.cos(math.radians(a)), math.sin(math.radians(a)))
               for a in ELECTRODE_ANGLES_DEG.values()]
        p.setPen(QPen(QColor("#3a3a3e"), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(3):
            p.drawLine(tri[i], tri[(i + 1) % 3])
        p.drawLine(QPointF(c.x() - 6, c.y()), QPointF(c.x() + 6, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 6), QPointF(c.x(), c.y() + 6))
        if self.cal.center_x or self.cal.center_y:
            cc = self._to_screen(self.cal.center_x, self.cal.center_y)
            p.setPen(QPen(QColor(theme.ACCENT), 1))
            p.drawEllipse(cc, 4, 4)

        # trajectory
        if self.trajectory is not None and len(self.trajectory) > 1:
            path = QPainterPath(self._to_screen(*self.trajectory[0]))
            step = max(1, len(self.trajectory) // 2000)
            for x, y in self.trajectory[::step]:
                path.lineTo(self._to_screen(x, y))
            p.setPen(QPen(QColor(197, 134, 192, 150), 1.4))
            p.drawPath(path)

        # electrodes, sized by their current at the point
        levels = None
        if self.point is not None and self.show_levels:
            lv = electrode_levels(self.volume, self.point[0], self.point[1], self.cal)
            levels = {k: float(lv[k]) for k in ELECTRODES}
        font = QFont(self.font())
        font.setPixelSize(12)
        font.setBold(True)
        p.setFont(font)
        for k, pt in zip(ELECTRODES, tri):
            col = QColor(ELECTRODE_COLORS[k])
            if levels is not None:
                glow = QColor(col)
                glow.setAlpha(110)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(glow))
                rad = 6 + 16 * min(1.5, levels[k]) / 1.5
                p.drawEllipse(pt, rad, rad)
            p.setPen(QPen(QColor("#111111"), 1))
            p.setBrush(QBrush(col))
            p.drawEllipse(pt, 6, 6)
            p.setPen(QColor(theme.TEXT))
            off = QPointF((pt.x() - c.x()) / r * 16, -(c.y() - pt.y()) / r * 16)
            p.drawText(QRectF(pt.x() + off.x() - 12, pt.y() + off.y() - 9, 24, 18),
                       Qt.AlignmentFlag.AlignCenter, k)

        # current point
        if self.point is not None:
            pt = self._to_screen(*self.point)
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.setBrush(QBrush(QColor(theme.DANGER if self._dragging else "#f14c4c")))
            p.drawEllipse(pt, 6, 6)
        p.end()

    # ── dragging ────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if self.editable and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_to(event.position())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_to(event.position())

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            x, y = self._from_screen(event.position())
            self.point = (x, y)
            self.update()
            self.dragFinished.emit(x, y)

    def _drag_to(self, pos) -> None:
        x, y = self._from_screen(pos)
        self.point = (x, y)
        self.update()
        self.pointDragged.emit(x, y)
