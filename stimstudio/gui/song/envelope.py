"""Automation envelope drawn and edited on a lane of the song editor.

* click on empty lane space — add a point (and keep dragging it);
* drag a point — move it (time snaps to the grid, hold Alt for free movement);
* right-click a point — delete it.

A gesture edits a local copy; the release hands the new point list to the
editor, which turns it into one undoable command.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem

from ...core.automation import evaluate
from ...core.model import Envelope
from .layout import LANE_H

PAD = 3          # vertical padding inside the lane
HIT_PX = 6       # point grab radius
SAMPLE_PX = 3    # curve sampling step


class EnvelopeItem(QGraphicsItem):
    def __init__(self, editor, track_id: str, ref: tuple, color: str):
        super().__init__()
        self.editor = editor
        self.track_id = track_id
        self.ref = ref                     # ("lane", key) or ("fx", index, param)
        self.color = QColor(color)
        self.points: list[tuple[float, float]] = []
        self.interp = "linear"
        self.enabled = True
        self._width = 1.0
        self._path = QPainterPath()
        self._window = (0.0, 1.0)
        self._drag: int | None = None
        self._moved = False
        self.setZValue(8)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)

    # ── data ────────────────────────────────────────────────────────────────
    @property
    def _pps(self) -> float:
        return self.editor.pps   # always the editor's current zoom

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    def set_envelope(self, points, interp: str, enabled: bool) -> None:
        if self.dragging:
            return
        self.points = sorted((float(t), float(v)) for t, v in points)
        self.interp, self.enabled = interp, enabled
        self._rebuild_path()

    def set_width(self, width: float) -> None:
        if width != self._width:
            self.prepareGeometryChange()
            self._width = width

    def rebuild(self, t0: float, t1: float, _pps: float | None = None) -> None:
        self._window = (t0, t1)
        self._rebuild_path()

    def _y(self, v: float) -> float:
        return PAD + (LANE_H - 2 * PAD) * (1.0 - v)

    def _v(self, y: float) -> float:
        return float(np.clip(1.0 - (y - PAD) / (LANE_H - 2 * PAD), 0.0, 1.0))

    def _rebuild_path(self) -> None:
        t0, t1 = self._window
        self._path = QPainterPath()
        if self.points and t1 > t0:
            n = max(2, int((t1 - t0) * self._pps / SAMPLE_PX) + 2)
            times = np.linspace(t0, t1, n)
            # include the points themselves so corners stay sharp
            inside = [t for t, _ in self.points if t0 < t < t1]
            if inside:
                times = np.unique(np.concatenate([times, inside, np.nextafter(inside, -np.inf)]))
            vals = evaluate(Envelope(self.points, interp=self.interp), times)
            self._path.moveTo(times[0] * self._pps, self._y(vals[0]))
            for t, v in zip(times[1:], vals[1:]):
                self._path.lineTo(t * self._pps, self._y(v))
        self.update()

    # ── painting ────────────────────────────────────────────────────────────
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, LANE_H)

    def paint(self, painter, option, widget=None):
        color = QColor(self.color if self.enabled else QColor("#777777"))
        painter.setPen(QPen(color, 1.8 if self.enabled else 1.0,
                            Qt.PenStyle.SolidLine if self.enabled else Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(color))
        t0, t1 = self._window
        for i, (t, v) in enumerate(self.points):
            if t0 - 1 <= t <= t1 + 1:
                r = 4.5 if i == self._drag else 3.5
                painter.drawRect(QRectF(t * self._pps - r, self._y(v) - r, 2 * r, 2 * r))

    # ── editing ─────────────────────────────────────────────────────────────
    def _hit(self, pos: QPointF) -> int | None:
        for i, (t, v) in enumerate(self.points):
            if abs(t * self._pps - pos.x()) <= HIT_PX and abs(self._y(v) - pos.y()) <= HIT_PX:
                return i
        return None

    def _time(self, x: float, modifiers) -> float:
        t = max(0.0, x / self._pps)
        if not modifiers & Qt.KeyboardModifier.AltModifier:
            t = self.editor.snap_time(t)
        return t

    def hoverMoveEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeAllCursor if self._hit(event.pos()) is not None
                       else Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        hit = self._hit(event.pos())
        if event.button() == Qt.MouseButton.RightButton:
            if hit is not None:
                del self.points[hit]
                self._rebuild_path()
                self.editor.commit_envelope(self.track_id, self.ref, list(self.points))
            event.accept()
            return
        if hit is None:  # add a point and keep dragging it
            self.points.append((self._time(event.pos().x(), event.modifiers()),
                                self._v(event.pos().y())))
            self.points.sort()
            hit = self._hit(event.pos())
            if hit is None:  # snapped away from the cursor: take the closest
                ts = np.array([t for t, _ in self.points])
                hit = int(np.argmin(np.abs(ts - event.pos().x() / self._pps)))
            self._moved = True
        else:
            self._moved = False
        self._drag = hit
        self._rebuild_path()
        event.accept()

    def mouseMoveEvent(self, event):
        i = self._drag
        if i is None:
            return
        t = self._time(event.pos().x(), event.modifiers())
        lo = self.points[i - 1][0] if i > 0 else 0.0
        hi = self.points[i + 1][0] if i + 1 < len(self.points) else float("inf")
        t = min(max(t, lo), hi)                      # points never cross
        self.points[i] = (t, self._v(event.pos().y()))
        self._moved = True
        self._rebuild_path()
        self.editor.statusMessage.emit(f"{t:.3f} s  ·  {self.points[i][1]:.2f}")
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        self._drag = None
        if self._moved:
            self.editor.commit_envelope(self.track_id, self.ref, list(self.points))
        self._rebuild_path()
        event.accept()
