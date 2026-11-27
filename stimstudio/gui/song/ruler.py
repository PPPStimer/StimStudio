"""Time ruler above the song view: markers, time labels, bar numbers, cursor scrubbing."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from .. import theme
from .layout import time_step_for

MARK_H = 14                    # marker strip at the top
SCALE_H = 30                   # time labels, bars, loop band
HEAT_H = 6                     # intensity strip at the bottom
RULER_H = MARK_H + SCALE_H + HEAT_H
FLAG_HIT_PX = 6


class Ruler(QWidget):
    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.offset = 0.0              # scene x at the widget's left edge
        self._loop_anchor: float | None = None
        self._loop_moved = False
        self._marker_drag: str | None = None     # id of the marker being dragged
        self._marker_moved = False
        self._drag_serial = 0
        self.setFixedHeight(RULER_H)
        self.setToolTip("Click / drag: move the cursor · right-drag: set the loop region · "
                        "right-click: clear it · double-click: add a marker\n"
                        "Marker: drag to move · double-click to rename · right-click for menu")
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.IBeamCursor)

    def set_offset(self, x: float) -> None:
        self.offset = x
        self.update()

    def _time_at(self, x: float) -> float:
        return max(0.0, (x + self.offset) / self.editor.pps)

    def _x_of(self, t: float) -> float:
        return t * self.editor.pps - self.offset

    def marker_at(self, x: float, y: float):
        """The marker whose flag (or label) is under widget point (*x*, *y*)."""
        if y > MARK_H + 2:
            return None
        markers = self.editor.project.markers
        best, best_d = None, FLAG_HIT_PX + 1.0
        for m in markers:
            d = abs(self._x_of(m.t_s) - x)
            if d < best_d:
                best, best_d = m, d
        if best is not None:
            return best
        fm = QFontMetrics(self._marker_font())
        for m in reversed(markers):           # label to the right of the flag
            mx = self._x_of(m.t_s)
            if mx <= x <= mx + 10 + fm.horizontalAdvance(m.name):
                return m
        return None

    def _marker_font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(10)
        return font

    # ── painting ────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        ed = self.editor
        pps = ed.pps
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.fillRect(self.rect(), QColor(theme.PANEL_BG))
        w = self.width()
        t0, t1 = self._time_at(0), self._time_at(w)
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        top = MARK_H                       # the scale sits under the marker strip
        bottom = top + SCALE_H

        # loop region
        if ed.loop_region is not None:
            a, b = ed.loop_region
            xa, xb = self._x_of(a), self._x_of(b)
            band = QColor(0, 122, 204, 120 if ed.loop_enabled else 50)
            p.fillRect(QRectF(xa, bottom - 11, xb - xa, 11), band)

        # bars (lower half)
        beat = 60.0 / max(1e-6, ed.project.bpm)
        bar = beat * 4
        bar_every = max(1, math.ceil(40 / max(1e-6, bar * pps)))
        k0, k1 = int(t0 / bar), int(t1 / bar) + 1
        p.setPen(QPen(QColor(theme.TEXT_DIM), 1))
        if k1 - k0 < 3000:
            for k in range(k0, k1 + 1):
                if k % bar_every:
                    continue
                x = k * bar * pps - self.offset
                p.drawLine(QPointF(x, bottom - 9), QPointF(x, bottom))
                p.drawText(QPointF(x + 3, bottom - 2), str(k + 1))

        # time labels (upper half)
        step = time_step_for(pps, 80)
        decimals = 0 if step >= 1 else (1 if step >= 0.1 else 3)
        k0, k1 = int(t0 / step), int(t1 / step) + 1
        p.setPen(QPen(QColor(theme.TEXT), 1))
        for k in range(k0, k1 + 1):
            t = k * step
            x = t * pps - self.offset
            p.drawLine(QPointF(x, top), QPointF(x, top + 6))
            p.drawText(QPointF(x + 3, top + 12), theme.format_time(t, decimals))

        self._paint_markers(p, t0, t1)

        # intensity strip + safety issues
        from .overview import paint_heat, paint_issue_marks
        heat = QRectF(0, bottom, w, HEAT_H)
        paint_heat(p, ed.heat_img, ed.heat_bin_s, t0, t1, heat)
        if ed.safety_report is not None:
            paint_issue_marks(p, ed.safety_report.issues, self._x_of, bottom - 2, HEAT_H + 2)

        # playhead
        if ed.playhead_s >= 0:
            px = self._x_of(ed.playhead_s)
            p.setPen(QPen(QColor(theme.DANGER), 2))
            p.drawLine(QPointF(px, 0), QPointF(px, bottom))

        # cursor marker
        cx = self._x_of(ed.doc.cursor_s)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.CURSOR))
        p.drawPolygon(QPolygonF([QPointF(cx - 5, bottom - 8), QPointF(cx + 5, bottom - 8),
                                 QPointF(cx, bottom)]))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(QPointF(0, RULER_H - 0.5), QPointF(w, RULER_H - 0.5))
        p.end()

    def _paint_markers(self, p: QPainter, t0: float, t1: float) -> None:
        markers = self.editor.project.markers
        if not markers:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = self._marker_font()
        p.setFont(font)
        fm = QFontMetrics(font)
        visible = [m for m in markers if t0 - 60 <= m.t_s <= t1]
        for i, m in enumerate(visible):
            x = self._x_of(m.t_s)
            color = QColor(m.color)
            p.setPen(QPen(color, 1))
            p.drawLine(QPointF(x, 0), QPointF(x, RULER_H))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawPolygon(QPolygonF([QPointF(x, 1), QPointF(x + 8, 4.5), QPointF(x, 8)]))
            # the label is elided before the next marker's flag
            room = (self._x_of(visible[i + 1].t_s) - x - 14) if i + 1 < len(visible) else 400
            if room > 12 and m.name:
                p.setPen(QPen(color, 1))
                p.drawText(QPointF(x + 10, 10), fm.elidedText(m.name, Qt.TextElideMode.ElideRight,
                                                              int(room)))
        p.restore()

    # ── mouse ───────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        pos = event.position()
        marker = self.marker_at(pos.x(), pos.y())
        if event.button() == Qt.MouseButton.LeftButton:
            if marker is not None:
                self._marker_drag = marker.id
                self._marker_moved = False
                self._drag_serial += 1
                return
            self._scrub(event)
        elif event.button() == Qt.MouseButton.RightButton:
            if marker is not None:
                self.editor.marker_menu(marker.id, event.globalPosition().toPoint())
                return
            self._loop_anchor = self._snapped(event)
            self._loop_moved = False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self._marker_drag is not None:
            t = self._snapped(event, exclude=self._marker_drag)
            m = self.editor.doc.find_marker(self._marker_drag)
            if m is not None and abs(m.t_s - t) > 1e-9:
                self._marker_moved = True
                self.editor.doc.set_marker_attrs(
                    self._marker_drag, merge_key=("marker-drag", self._drag_serial),
                    text="Move marker", t_s=t)
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._scrub(event)
        elif event.buttons() & Qt.MouseButton.RightButton and self._loop_anchor is not None:
            t = self._snapped(event)
            if abs(t - self._loop_anchor) > 0:
                self._loop_moved = True
                self.editor.loopRegionEdited.emit((self._loop_anchor, t))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._marker_drag is not None:
            m = self.editor.doc.find_marker(self._marker_drag)
            if m is not None and not self._marker_moved:
                self.editor.doc.set_cursor(m.t_s)        # a click jumps to the marker
            self._marker_drag = None
            return
        if event.button() == Qt.MouseButton.RightButton and self._loop_anchor is not None:
            if not self._loop_moved:
                self.editor.loopRegionEdited.emit(None)
            self._loop_anchor = None

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        marker = self.marker_at(pos.x(), pos.y())
        if marker is not None:
            self._marker_drag = None
            self.editor.edit_marker(marker.id)
        else:
            self.editor.create_marker(self._snapped(event))

    def _snapped(self, event, exclude: str | None = None) -> float:
        t = self._time_at(event.position().x())
        snap_on = not event.modifiers() & Qt.KeyboardModifier.AltModifier
        return max(0.0, self.editor.snap_time(t, snap_on, exclude_marker=exclude))

    def _scrub(self, event):
        self.editor.doc.set_cursor(self._snapped(event))
