"""Graphics items of the song editor: clips and rendered lane curves."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem

from ...core.pulse import FS
from .. import metrics, theme
from .layout import CLIP_ROW_H, LANE_H
from .paths import curve_path

#: audio_mod signal lane colours: envelope, onsets, low, mid, high
SIGNAL_COLORS = ("#c586c0", "#f14c4c", "#4fc1ff", "#b5cea8", "#dcdcaa")

CLIP_PAD = 2            # vertical gap between the row edge and the clip body
CLIP_H = CLIP_ROW_H - 2 * CLIP_PAD
TITLE_H = 15
EDGE_PX = 6             # resize grip width
FADE_GRIP = 5           # half-size of a fade handle
XFADE_COLOR = QColor(220, 220, 170)     # crossfade ramps drawn on the upper clip
#: pattern pool entries A, B, C, … (cycled)
KEY_COLORS = ("#4fc1ff", "#f14c4c", "#b5cea8", "#dcdcaa", "#c586c0", "#ce9178", "#4ec9b0",
              "#d7ba7d")


def key_color(key: str | None) -> QColor:
    """Colour of a pattern pool entry (grey for a rest)."""
    if not key:
        return QColor(theme.TEXT_DIM)
    i = sum((ord(c) - 64) * 26 ** k for k, c in enumerate(reversed(key))) - 1
    return QColor(KEY_COLORS[i % len(KEY_COLORS)])


def clip_marks(clip) -> list[tuple[float, float, str | None]]:
    """``(time from the clip start, velocity, pool key)`` of the marks a marks /
    pattern clip shows (those inside its trimmed span)."""
    src = clip.source
    if src.kind == "pattern":
        from ...core.pattern import resolve
        pairs = [(m, e.key if e else None) for m, e in resolve(src)]
    elif src.kind == "marks":
        pairs = [(m, None) for m in src.marks]
    else:
        return []
    out = []
    for m, key in pairs:
        t = m.t_s - clip.trim_s
        if 0.0 <= t < clip.length_s:
            out.append((t, m.velocity, key))
    return out


class ClipItem(QGraphicsItem):
    """One clip on a track's main row. Dragging is driven by the editor."""

    def __init__(self, editor, clip_id: str):
        super().__init__()
        self.editor = editor
        self.clip_id = clip_id
        self.track_id = ""
        self.start_s = 0.0
        self.length_s = 1.0
        self.fade_in_s = 0.0
        self.fade_out_s = 0.0
        self.name = ""
        self.color = QColor(theme.LANE_COLORS[0])
        self.muted = False
        self.dragging = False
        self._pps = 1.0
        #: (m, 4) slice of the track render; row 0 is at the clip start, so the
        #: preview travels with the item while it is dragged
        self._preview: np.ndarray | None = None
        self._paths: list[QPainterPath] = []
        self.video = False
        self.kind = ""
        #: marks / pattern clips: (time from clip start, velocity, pool key)
        self.marks: list[tuple[float, float, str | None]] = []
        self.show_thumbs = True          # video: filmstrip on the clip body
        self.audio_lane = False          # video: waveform lane below the clip row
        self.video_path = ""
        self.video_duration = 0.0
        self.missing = False
        #: crossfade layout on a "crossfade" track (core.joins.Layer), else None
        self.layer = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    # ── state ───────────────────────────────────────────────────────────────
    def sync(self, clip, track, row_y: float, pps: float) -> None:
        """Copy model state; geometry is left alone while the item is dragged."""
        self.track_id = track.id
        self.kind = clip.source.kind
        self.marks = clip_marks(clip)
        self.audio = clip.source.kind == "audio"
        self.video = clip.source.kind == "video"
        if self.video:
            self.video_path = clip.source.path
            self.video_duration = clip.source.duration_s or clip.length_s
            self.missing = not os.path.isfile(self.video_path)
        self.name = clip.name or {"generator": getattr(clip.source, "type_id", ""),
                                  "audio": "audio", "video": "video", "marks": "Beat marks",
                                  "pattern": "Pattern"}.get(clip.source.kind, "pulses")
        if self.video and self.missing:
            self.name = f"{self.name} — file not found"
        self.color = QColor(track.color)
        self.muted = clip.muted
        if self.dragging:
            return
        self.set_geometry(clip.start_s, clip.length_s, row_y, pps)
        self.fade_in_s, self.fade_out_s = clip.fade_in_s, clip.fade_out_s
        self.update()

    def set_layer(self, layer) -> None:
        """Crossfade ramps and stacking (the clip that starts later lies on top)."""
        self.layer = layer
        self.setZValue(10 + (0.001 * layer.order if layer is not None else 0))
        self.update()

    @property
    def covered_in(self) -> bool:
        return self.layer is not None and self.layer.covered_in

    @property
    def covered_out(self) -> bool:
        return self.layer is not None and self.layer.covered_out

    def set_geometry(self, start_s: float, length_s: float, row_y: float, pps: float) -> None:
        if length_s != self.length_s or pps != self._pps:
            self.prepareGeometryChange()
        self.start_s, self.length_s, self._pps = start_s, length_s, pps
        self.setPos(start_s * pps, row_y + CLIP_PAD)

    def set_preview(self, data: np.ndarray | None) -> None:
        self._preview = data

    def rebuild_preview(self, t0: float, t1: float) -> None:
        """Rebuild the amplitude preview for the visible time window."""
        self._paths = []
        if self._preview is not None and len(self._preview):
            h = CLIP_H - TITLE_H - 3
            lo, hi = t0 - self.start_s, t1 - self.start_s
            if hi > 0 and lo < self.length_s:
                for col in (0, 1):
                    p = curve_path(self._preview[:, col], 0.0, self._pps, max(0.0, lo),
                                   min(self.length_s, hi), h)
                    p.translate(0, TITLE_H + 2)
                    self._paths.append(p)
        self.update()

    # ── geometry / painting ─────────────────────────────────────────────────
    @property
    def width(self) -> float:
        return max(2.0, self.length_s * self._pps)

    @property
    def height(self) -> float:
        return CLIP_H + (CLIP_PAD + LANE_H if self.video and self.audio_lane else 0)

    def set_video_view(self, thumbs: bool, audio_lane: bool) -> None:
        if audio_lane != self.audio_lane:
            self.prepareGeometryChange()
        self.show_thumbs, self.audio_lane = thumbs, audio_lane
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        w = self.width
        rect = QRectF(0, 0, w, CLIP_H)
        base = QColor(theme.TEXT_DIM) if self.muted else QColor(self.color)
        body = QColor(base)
        body.setAlpha(55)
        title = QColor(base)
        title.setAlpha(150 if not self.muted else 90)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(body))
        painter.drawRoundedRect(rect, 3, 3)
        if self.kind == "marks":        # a blank: hatched, it does not sound
            hatch = QColor(base)
            hatch.setAlpha(70)
            painter.setBrush(QBrush(hatch, Qt.BrushStyle.BDiagPattern))
            painter.drawRoundedRect(rect, 3, 3)
        painter.setBrush(QBrush(title))
        painter.drawRoundedRect(QRectF(0, 0, w, TITLE_H), 3, 3)

        # amplitude preview (L / R)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setClipRect(rect)
        colors = (SIGNAL_COLORS[:2] if getattr(self, "audio", False) else theme.LANE_COLORS[:2])
        for path, color in zip(self._paths, colors):
            painter.setPen(QPen(QColor(color), 1.0))
            painter.drawPath(path)

        if self.marks:
            self._paint_marks(painter, option)
        if self.video:
            if self.show_thumbs:
                self._paint_filmstrip(painter, option, w)
            if self.audio_lane:
                painter.setClipping(False)
                self._paint_waveform(painter, option, w, base)
                painter.setClipRect(rect)
        # crossfades (on the upper clip), then the manual fades of free edges
        if self.layer is not None and not self.dragging:
            self._paint_crossfades(painter, w)
        fi, fo = self.fade_in_s * self._pps, self.fade_out_s * self._pps
        fade_pen = QPen(QColor(255, 255, 255, 150), 1.0)
        painter.setPen(fade_pen)
        if fi > 0 and not self.covered_in:
            painter.drawLine(QPointF(0, CLIP_H), QPointF(fi, TITLE_H))
        if fo > 0 and not self.covered_out:
            painter.drawLine(QPointF(w - fo, TITLE_H), QPointF(w, CLIP_H))
        if (self.isSelected() or self.isUnderMouse()) and not self.video:
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(Qt.PenStyle.NoPen)
            for x, covered in ((fi, self.covered_in), (w - fo, self.covered_out)):
                if not covered:
                    painter.drawRect(QRectF(x - FADE_GRIP + 1, TITLE_H - FADE_GRIP + 1,
                                            2 * FADE_GRIP - 2, 2 * FADE_GRIP - 2))

        # title text
        painter.setPen(QColor("#ffffff"))
        font = QFont(painter.font())
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(QRectF(4, 0, max(0.0, w - 8), TITLE_H),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         self.name)
        painter.setClipping(False)

        # outline
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self.isSelected():
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
        else:
            painter.setPen(QPen(base.darker(140), 1.0))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)

    def _paint_marks(self, painter, option) -> None:
        """Beat marks as ticks (height = velocity); pattern marks take the
        colour of the pool entry that plays there, with its key when room."""
        exposed = option.exposedRect
        top, bottom = TITLE_H + 2, CLIP_H - 2
        span = bottom - top
        pattern = self.kind == "pattern"
        xs = [t * self._pps for t, _v, _k in self.marks]
        roomy = len(xs) < 2 or min(b - a for a, b in zip(xs, xs[1:])) > 12
        painter.setFont(metrics.paint_font(9, base=painter.font()))
        for x, (_t, v, key) in zip(xs, self.marks):
            if x < exposed.left() - 8 or x > exposed.right() + 8:
                continue
            color = key_color(key) if pattern else QColor("#ffffff")
            if pattern and key is None:
                pen = QPen(color, 1.0, Qt.PenStyle.DotLine)
            else:
                color.setAlpha(220 if pattern else 170)
                pen = QPen(color, 1.5)
            painter.setPen(pen)
            h = max(3.0, span * min(1.0, max(0.0, v)))
            painter.drawLine(QPointF(x, bottom), QPointF(x, bottom - h))
            if pattern and key and roomy:
                painter.drawText(QRectF(x + 2, top, 20, 10),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, key)

    def _paint_crossfades(self, painter, w: float) -> None:
        """Each opacity ramp as an X: this clip's curve solid, the clip below dashed."""
        top, bottom = TITLE_H + 1, CLIP_H - 1
        for k, rising in ((self.layer.ramp_in, True), (self.layer.ramp_out, False)):
            if not k:
                continue
            width = k / FS * self._pps
            x0 = 0.0 if rising else w - width
            tint = QColor(XFADE_COLOR)
            tint.setAlpha(28)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(tint))
            painter.drawRect(QRectF(x0, top, width, bottom - top))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            u = np.linspace(0.0, 1.0, max(2, min(48, int(width / 3) + 2)))
            s_curve = u * u * (3 - 2 * u)
            for solid in (True, False):
                level = s_curve if rising == solid else 1.0 - s_curve
                path = QPainterPath(QPointF(x0, bottom - level[0] * (bottom - top)))
                for uu, lv in zip(u[1:], level[1:]):
                    path.lineTo(x0 + uu * width, bottom - lv * (bottom - top))
                color = QColor(XFADE_COLOR)
                color.setAlpha(230 if solid else 140)
                pen = QPen(color, 1.3 if solid else 1.0)
                if not solid:
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawPath(path)

    def _paint_waveform(self, painter, option, w: float, base: QColor) -> None:
        """The video's sound as a mirrored peak envelope in the lane below the clip."""
        from ..video.media import waveform_cache
        top = CLIP_ROW_H - CLIP_PAD                  # lane top in item coordinates
        lane = QRectF(0, top + 1, w, LANE_H - 2)
        fill = QColor(base)
        fill.setAlpha(28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill))
        painter.drawRect(lane)
        wave = None if self.missing else waveform_cache().waveform(self.video_path,
                                                                   self.video_duration)
        painter.setPen(QColor(theme.TEXT_DIM))
        font = QFont(painter.font())
        font.setPixelSize(10)
        painter.setFont(font)
        if wave is None or (wave.finished and not wave.has_audio):
            text = ("no sound" if wave is not None else
                    "waveform needs PyAV (pip install av)" if not self.missing else "")
            painter.drawText(lane.adjusted(6, 0, 0, 0),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            return
        exposed = option.exposedRect if option is not None else QRectF(0, 0, w, self.height)
        x0, x1 = max(0, int(exposed.left())), min(int(np.ceil(w)), int(exposed.right()) + 1)
        if x1 <= x0:
            return
        peaks = wave.column_peaks(x0 / self._pps, 1.0 / self._pps, x1 - x0)
        ready_x = wave.ready_bins / wave.rate * self._pps
        scale = 1.0 / max(wave.top, 1e-3)
        mid, half = lane.center().y(), lane.height() / 2 - 1
        path = QPainterPath()
        xs = np.arange(x0, x1, dtype=float)
        hs = np.clip(peaks * scale, 0, 1) * half
        path.moveTo(xs[0], mid - hs[0])
        for x, hgt in zip(xs[1:], hs[1:]):
            path.lineTo(x, mid - hgt)
        for x, hgt in zip(xs[::-1], hs[::-1]):
            path.lineTo(x, mid + hgt)
        path.closeSubpath()
        wave_color = QColor(SIGNAL_COLORS[0])
        wave_color.setAlpha(200)
        painter.setBrush(QBrush(wave_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.save()
        painter.setClipRect(QRectF(0, lane.top(), min(w, ready_x), lane.height()))
        painter.drawPath(path)
        painter.restore()
        if not wave.finished and ready_x < w:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(QRectF(max(ready_x, 0) + 6, lane.top(), 200, lane.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             "reading sound…")

    def _paint_filmstrip(self, painter, option, w: float) -> None:
        """Thumbnails along the clip body, one cell per thumbnail width at this zoom."""
        from ..video.media import thumbnail_cache
        if self.missing:
            return
        strip = thumbnail_cache().strip(self.video_path, self.video_duration)
        if strip is None:
            return
        sample = next((img for img in strip.images if img is not None), None)
        if sample is None:
            return
        top, h = TITLE_H, CLIP_H - TITLE_H
        cell = max(8.0, sample.width() * h / max(1, sample.height()))
        exposed = option.exposedRect if option is not None else QRectF(0, 0, w, CLIP_H)
        first = max(0, int(exposed.left() // cell))
        last = min(int(np.ceil(w / cell)), int(exposed.right() // cell) + 1)
        for k in range(first, last):
            x = k * cell
            img = strip.image_at((x + cell / 2) / self._pps)
            if img is not None:
                part = min(1.0, (w - x) / cell)
                painter.drawImage(QRectF(x, top, cell * part, h), img,
                                  QRectF(0, 0, img.width() * part, img.height()))

    # ── interaction zones ───────────────────────────────────────────────────
    def zone_at(self, pos: QPointF) -> str:
        if self.video:
            return "move"                # a video clip can only be moved
        x, y = pos.x(), pos.y()
        w = self.width
        fi, fo = self.fade_in_s * self._pps, w - self.fade_out_s * self._pps
        if y <= TITLE_H + FADE_GRIP:
            if abs(x - fi) <= FADE_GRIP + 1 and not self.covered_in:
                return "fade_in"
            if abs(x - fo) <= FADE_GRIP + 1 and not self.covered_out:
                return "fade_out"
        if x <= min(EDGE_PX, w / 3):
            return "left"
        if x >= w - min(EDGE_PX, w / 3):
            return "right"
        return "move"

    def hoverMoveEvent(self, event):
        zone = self.zone_at(event.pos())
        cursor = {"left": Qt.CursorShape.SizeHorCursor, "right": Qt.CursorShape.SizeHorCursor,
                  "fade_in": Qt.CursorShape.PointingHandCursor,
                  "fade_out": Qt.CursorShape.PointingHandCursor}.get(zone, Qt.CursorShape.SizeAllCursor)
        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def hoverEnterEvent(self, event):
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        additive = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.ShiftModifier))
        if additive and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.setSelected(not self.isSelected())
        elif not self.isSelected():
            if not additive:
                self.scene().clearSelection()
            self.setSelected(True)
        self.editor.begin_drag(self, self.zone_at(event.pos()), event.scenePos())
        event.accept()

    def mouseMoveEvent(self, event):
        self.editor.update_drag(event.scenePos(), event.modifiers())
        event.accept()

    def mouseReleaseEvent(self, event):
        self.editor.end_drag(event.modifiers())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.editor.clip_double_clicked(self.clip_id)
        event.accept()


class LaneCurveItem(QGraphicsItem):
    """The rendered output of one track lane (after track FX and automation)."""

    def __init__(self, track_id: str, lane: int, color: str | None = None,
                 fill: bool | None = None, mask_inactive: bool | None = None):
        super().__init__()
        self.track_id = track_id
        self.lane = lane
        self.color = color or theme.LANE_COLORS[lane % 4]
        self.fill = lane < 2 if fill is None else fill
        #: frequency is meaningless where the track is silent: draw it at 0
        self.mask_inactive = lane >= 2 if mask_inactive is None else mask_inactive
        self.values: np.ndarray | None = None
        self.active: np.ndarray | None = None
        self._width = 1.0
        self._path = QPainterPath()
        self._pps = 1.0
        self.setZValue(5)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_data(self, values: np.ndarray | None, active: np.ndarray | None) -> None:
        self.values, self.active = values, active

    def set_width(self, width: float) -> None:
        if width != self._width:
            self.prepareGeometryChange()
            self._width = width

    def rebuild(self, t0: float, t1: float, pps: float) -> None:
        self._pps = pps
        if self.values is None or not len(self.values):
            self._path = QPainterPath()
        else:
            vals = self.values
            if self.mask_inactive and self.active is not None:
                vals = np.where(self.active, vals, 0.0)
            self._path = curve_path(vals, 0.0, pps, t0, t1, LANE_H - 6, FS, fill=self.fill)
            self._path.translate(0, 3)
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, LANE_H)

    def paint(self, painter, option, widget=None):
        color = QColor(self.color)
        if self.fill:
            fill = QColor(color)
            fill.setAlpha(60)
            painter.setBrush(QBrush(fill))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 1.0))
        painter.drawPath(self._path)
