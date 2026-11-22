"""Intensity map: the colour scale, the cached strip image, and the project
overview bar above the song editor (whole project, view rectangle, issues).

The strip is one pixel per profile bin (``core.safety.level_profile``), built
once per render and scaled when painted, so painting stays cheap at any zoom.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QFrame, QGraphicsScene, QGraphicsView

from .. import theme
from ..gl import GLGraphicsView

OVERVIEW_H = 20

#: (level, colour) stops of the heat scale
HEAT_STOPS = ((0.0, "#252526"), (0.04, "#1b2b40"), (0.3, "#2f81f7"), (0.6, "#d7ba7d"),
              (0.82, "#f0883e"), (1.0, "#f14c4c"))


def _lut() -> np.ndarray:
    xs = np.array([s for s, _ in HEAT_STOPS])
    cols = np.array([[QColor(c).red(), QColor(c).green(), QColor(c).blue()]
                     for _, c in HEAT_STOPS], np.float64)
    grid = np.linspace(0.0, 1.0, 256)
    rgb = np.stack([np.interp(grid, xs, cols[:, k]) for k in range(3)], axis=1)
    rgb = rgb.round().astype(np.uint32)
    return (0xFF000000 | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32)


HEAT_LUT = _lut()


def heat_color(value: float) -> QColor:
    return QColor.fromRgba(int(HEAT_LUT[int(np.clip(value, 0.0, 1.0) * 255)]))


def heat_image(profile: np.ndarray) -> QImage | None:
    """A 1-pixel-high image, one pixel per profile bin."""
    if profile is None or not len(profile):
        return None
    idx = (np.clip(profile, 0.0, 1.0) * 255).astype(np.int64)
    argb = np.ascontiguousarray(HEAT_LUT[idx])
    img = QImage(argb.tobytes(), len(argb), 1, 4 * len(argb), QImage.Format.Format_ARGB32)
    return img.copy()                      # own the pixels


def paint_heat(p: QPainter, img: QImage | None, bin_s: float, t0: float, t1: float,
               target: QRectF) -> None:
    """Paint the part of *img* covering ``[t0, t1]`` seconds into *target*."""
    p.fillRect(target, QColor(theme.PANEL_BG))
    if img is None or t1 <= t0:
        return
    b0, b1 = t0 / bin_s, t1 / bin_s
    if b1 <= 0 or b0 >= img.width():
        return
    c0, c1 = max(0.0, b0), min(float(img.width()), b1)
    x0 = target.left() + (c0 - b0) / (b1 - b0) * target.width()
    x1 = target.left() + (c1 - b0) / (b1 - b0) * target.width()
    p.drawImage(QRectF(x0, target.top(), x1 - x0, target.height()), img,
                QRectF(c0, 0, c1 - c0, 1))


def paint_issue_marks(p: QPainter, issues, t_to_x, top: float, height: float) -> None:
    from .. import safety as gs
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    for issue in issues:
        if issue.kind == "dose":
            continue
        x = t_to_x(issue.t0)
        p.setBrush(QColor(gs.SEVERITY_COLORS[issue.severity]))
        h = min(height, 6.0)
        p.drawPolygon(QPolygonF([QPointF(x - 3.5, top + height), QPointF(x + 3.5, top + height),
                                 QPointF(x, top + height - h)]))
    p.restore()


class Overview(GLGraphicsView):
    """The whole project at a glance; click / drag to scroll, wheel to zoom."""

    scrollRequested = pyqtSignal(float)          # timeline seconds for the view centre

    def __init__(self, editor, parent=None):
        super().__init__(QGraphicsScene(), parent)
        self.editor = editor
        self.setFixedHeight(OVERVIEW_H)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Project overview (intensity): click or drag to scroll · "
                        "wheel to zoom")
        self.setMouseTracking(False)

    def total_s(self) -> float:
        ed = self.editor
        t0, t1 = ed.visible_range()
        return max(ed.project.duration_s(), t1, 1.0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scene().setSceneRect(QRectF(0, 0, self.viewport().width(),
                                         self.viewport().height()))
        self.resetTransform()

    def _x(self, t: float) -> float:
        return t / self.total_s() * self.viewport().width()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        ed = self.editor
        w, h = self.viewport().width(), self.viewport().height()
        total = self.total_s()
        paint_heat(painter, ed.heat_img, ed.heat_bin_s, 0.0, total, QRectF(0, 3, w, h - 6))
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawLine(QPointF(0, h - 0.5), QPointF(w, h - 0.5))
        for m in ed.project.markers:
            x = self._x(m.t_s)
            painter.setPen(QPen(QColor(m.color), 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, 4))
        if ed.safety_report is not None:
            paint_issue_marks(painter, ed.safety_report.issues, self._x, 0, h)
        t0, t1 = ed.visible_range()
        view = QRectF(self._x(t0), 0.5, max(3.0, self._x(t1) - self._x(t0)), h - 1.5)
        painter.setPen(QPen(QColor(255, 255, 255, 170), 1))
        painter.setBrush(QColor(255, 255, 255, 28))
        painter.drawRect(view)
        if ed.playhead_s >= 0:
            painter.setPen(QPen(QColor(theme.DANGER), 1))
            x = self._x(ed.playhead_s)
            painter.drawLine(QPointF(x, 0), QPointF(x, h))
        x = self._x(ed.doc.cursor_s)
        painter.setPen(QPen(QColor(theme.CURSOR), 1))
        painter.drawLine(QPointF(x, 0), QPointF(x, h))

    def _time_at(self, x: float) -> float:
        return max(0.0, x / max(1, self.viewport().width()) * self.total_s())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.scrollRequested.emit(self._time_at(event.position().x()))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.scrollRequested.emit(self._time_at(event.position().x()))

    def mouseDoubleClickEvent(self, event):
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        self.editor.zoom(1.25 if delta > 0 else 0.8)      # around the visible centre
        event.accept()
