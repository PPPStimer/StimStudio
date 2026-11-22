"""The song editor's hardware-rendered (OpenGL) graphics view."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsView

from .. import theme
from ..gl import GLGraphicsView
from .layout import CLIP_ROW_H, LANE_H, N_LANES, time_step_for

GENERATOR_MIME = "application/x-stimstudio-generator"
PRESET_MIME = "application/x-stimstudio-preset"
_DROP_MIMES = (GENERATOR_MIME, PRESET_MIME)


class SongView(GLGraphicsView):
    """Draws the grid and cursor; forwards editing gestures to the editor."""

    zoomRequested = pyqtSignal(float, float)       # factor, anchor time (s)

    def __init__(self, editor, scene):
        super().__init__(scene)
        self.editor = editor
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    # ── painting ────────────────────────────────────────────────────────────
    def drawBackground(self, painter, rect: QRectF):
        ed = self.editor
        painter.fillRect(rect, QColor(theme.BG))
        layout = ed.layout
        if layout is None:
            return
        # track rows
        for row in layout.rows:
            if row.bottom < rect.top() or row.y > rect.bottom():
                continue
            band = QRectF(rect.left(), row.y, rect.width(), CLIP_ROW_H)
            painter.fillRect(band, QColor(theme.BG_ALT if row.index % 2 else theme.BG))
            if row.track_id == ed.current_track_id:
                painter.fillRect(band, QColor(0, 122, 204, 18))
            if row.n_lanes:
                painter.fillRect(QRectF(rect.left(), row.y + CLIP_ROW_H, rect.width(),
                                        min(row.n_lanes, N_LANES if row.kind != "audio_mod"
                                            else row.n_lanes) * LANE_H), QColor("#19191a"))
                if row.auto_lanes:
                    painter.fillRect(QRectF(rect.left(), row.lane_y(N_LANES), rect.width(),
                                            len(row.auto_lanes) * LANE_H), QColor("#1d1a1f"))
                painter.setPen(QPen(QColor(theme.GRID), 1))
                for lane in range(row.n_lanes):
                    y = row.lane_y(lane)
                    painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QPen(QColor("#141414"), 1))
            painter.drawLine(QPointF(rect.left(), row.bottom), QPointF(rect.right(), row.bottom))

        # vertical grid
        self._draw_grid(painter, rect, min(rect.bottom(), max(layout.height, rect.bottom())))

    def _draw_grid(self, painter, rect: QRectF, bottom: float):
        ed = self.editor
        pps = ed.pps
        t0, t1 = max(0.0, rect.left() / pps), rect.right() / pps
        beat = 60.0 / max(1e-6, ed.project.bpm)
        if ed.snap.kind == "beats" and beat * pps >= 6:
            fine, strong = beat, beat * 4
            if beat * pps < 14:
                fine = strong          # too dense: bars only
        else:
            fine = time_step_for(pps, 24)
            strong = time_step_for(pps, 110)
        pen_fine = QPen(QColor(theme.GRID), 1)
        pen_strong = QPen(QColor(theme.GRID_STRONG), 1)
        k0, k1 = int(math.floor(t0 / fine)), int(math.ceil(t1 / fine))
        if k1 - k0 > 4000:
            return
        for k in range(k0, k1 + 1):
            t = k * fine
            is_strong = abs(t / strong - round(t / strong)) < 1e-6
            painter.setPen(pen_strong if is_strong else pen_fine)
            x = t * pps
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, bottom))

    def drawForeground(self, painter, rect: QRectF):
        ed = self.editor
        if ed.loop_region is not None:
            a, b = ed.loop_region
            band = QRectF(a * ed.pps, rect.top(), (b - a) * ed.pps, rect.height())
            painter.fillRect(band, QColor(0, 122, 204, 26 if ed.loop_enabled else 10))
            painter.setPen(QPen(QColor(0, 122, 204, 160 if ed.loop_enabled else 70), 1))
            for x in (band.left(), band.right()):
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        if ed.project.markers:
            t0, t1 = rect.left() / ed.pps, rect.right() / ed.pps
            for m in ed.project.markers:
                if t0 - 1 <= m.t_s <= t1 + 1:
                    color = QColor(m.color)
                    color.setAlpha(110)
                    pen = QPen(color, 1, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    x = m.t_s * ed.pps
                    painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        if ed.playhead_s >= 0:
            px = ed.playhead_s * ed.pps
            if rect.left() - 2 <= px <= rect.right() + 2:
                painter.setPen(QPen(QColor(theme.DANGER), 1.5))
                painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
        x = ed.doc.cursor_s * ed.pps
        if rect.left() - 2 <= x <= rect.right() + 2:
            painter.setPen(QPen(QColor(theme.CURSOR), 1))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        if ed.drop_marker is not None:
            mx, my = ed.drop_marker
            painter.setPen(QPen(QColor(theme.ACCENT), 2))
            painter.drawLine(QPointF(mx, my), QPointF(mx, my + CLIP_ROW_H))

    # ── mouse ───────────────────────────────────────────────────────────────
    def wheelEvent(self, event):
        mods = event.modifiers()
        delta = event.angleDelta().y() or event.angleDelta().x()
        if mods & Qt.KeyboardModifier.ControlModifier:
            anchor_t = self.mapToScene(event.position().toPoint()).x() / self.editor.pps
            self.zoomRequested.emit(1.25 if delta > 0 else 0.8, anchor_t)
            event.accept()
            return
        if mods & Qt.KeyboardModifier.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.itemAt(event.pos()) is None:
            self.editor.empty_clicked(self.mapToScene(event.pos()), event.modifiers())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.itemAt(event.pos()) is None:
            self.editor.empty_double_clicked(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        self.editor.context_menu(self.mapToScene(event.pos()), self.itemAt(event.pos()),
                                 event.globalPos())

    # ── drag & drop from the generator browser ──────────────────────────────
    def _audio_paths(self, mime) -> list[str]:
        from ...audio.loader import is_audio_file
        from ...io.video import is_video_file
        if not mime.hasUrls():
            return []
        return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()
                and (is_audio_file(u.toLocalFile()) or is_video_file(u.toLocalFile())
                     or u.toLocalFile().lower().endswith((".funscript", ".csv")))]

    def dragEnterEvent(self, event):
        if self._audio_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        if any(event.mimeData().hasFormat(m) for m in _DROP_MIMES):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._audio_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        if any(event.mimeData().hasFormat(m) for m in _DROP_MIMES):
            self.editor.drop_hover(self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.editor.drop_hover(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        paths = self._audio_paths(mime)
        if paths:
            pos = self.mapToScene(event.position().toPoint())
            from ...io.video import is_video_file
            scripts = [p for p in paths if p.lower().endswith(".funscript")]
            beats = [p for p in paths if p.lower().endswith(".csv")]
            videos = [p for p in paths if is_video_file(p)]
            if videos:
                self.editor.videoDropped.emit(videos[0])
            elif beats:
                self.editor.drop_beats(beats, pos)
            elif scripts:
                self.editor.drop_funscripts(scripts, pos)
            else:
                self.editor.drop_audio(paths[0], pos)
            event.acceptProposedAction()
            return
        if mime.hasFormat(GENERATOR_MIME):
            type_id = bytes(mime.data(GENERATOR_MIME)).decode("utf-8")
            self.editor.drop_generator(type_id, self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
        elif mime.hasFormat(PRESET_MIME):
            path = bytes(mime.data(PRESET_MIME)).decode("utf-8")
            self.editor.drop_preset(path, self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
