"""Record dock: play the project and perform a take live with the mouse / keys.

* standard track — an X/Y pad: up = level, right = pulse frequency;
* triphase track — the electrode triangle (position), wheel / keys = volume.

Hold the left button to play (release springs back to 0 unless *Hold* is on),
keys 1–9 set 10–90 % and 0 silences, the wheel nudges frequency (or volume).
With *Monitor* on the output plays exactly what you perform (power, soft
start, mute and PANIC still apply). Stop (or R again) turns the take into a
pulse clip; earlier takes underneath are muted, not deleted.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QSurfaceFormat
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout,
                             QWidget)

from ...core import record as rec
from ...core.model import Clip, PulseDataSource
from ...core.pulse import FS
from ...core.triphase import Calibration, tri_to_pulses, xy_to_unit
from ...outputs import LiveInput
from .. import theme
from ..gl import MSAA_SAMPLES, gl_enabled
from .position_pad import PositionPad

if gl_enabled():
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget as _Base
else:
    _Base = QWidget

TICK_MS = 25                 # sampling / monitoring rate (40 Hz)
WHEEL_STEP = 0.05


def _digit(event) -> int | None:
    key = event.key()
    if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        return key - Qt.Key.Key_0
    return None


class RecordPad(_Base):
    """X/Y pad: Y = level (0 bottom … 1 top), X = frequency (0 left … 1 right)."""

    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        if gl_enabled():
            fmt = QSurfaceFormat.defaultFormat()
            fmt.setSamples(MSAA_SAMPLES)
            self.setFormat(fmt)
        self.amp = 0.0
        self.freq = 0.5
        self.pressed = False
        self.spring = True
        self.setMinimumSize(200, 160)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def sizeHint(self) -> QSize:
        return QSize(260, 220)

    def set_value(self, amp: float | None = None, freq: float | None = None) -> None:
        if amp is not None:
            self.amp = float(np.clip(amp, 0.0, 1.0))
        if freq is not None:
            self.freq = float(np.clip(freq, 0.0, 1.0))
        self.update()
        self.valueChanged.emit()

    def _area(self) -> QRectF:
        return QRectF(28, 8, max(10, self.width() - 36), max(10, self.height() - 30))

    def _from_pos(self, pos: QPointF) -> None:
        r = self._area()
        self.set_value(1.0 - (pos.y() - r.top()) / r.height(), (pos.x() - r.left()) / r.width())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.pressed = True
            self._from_pos(event.position())

    def mouseMoveEvent(self, event):
        if self.pressed:
            self._from_pos(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.pressed:
            self.pressed = False
            if self.spring:
                self.set_value(amp=0.0)
            else:
                self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        self.set_value(freq=self.freq + (WHEEL_STEP if delta > 0 else -WHEEL_STEP))
        event.accept()

    def keyPressEvent(self, event):
        d = _digit(event)
        if d is not None and not event.isAutoRepeat():
            self.set_value(amp=d / 10.0)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(theme.PANEL_BG))
        r = self._area()
        p.fillRect(r, QColor(theme.BG))
        p.setPen(QPen(QColor(theme.GRID), 1))
        for k in range(1, 10):
            y = r.bottom() - r.height() * k / 10
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        for k in range(1, 4):
            x = r.left() + r.width() * k / 4
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        # level bar
        bar = QRectF(4, r.top() + r.height() * (1 - self.amp), 14, r.height() * self.amp)
        p.fillRect(QRectF(4, r.top(), 14, r.height()), QColor(theme.BG))
        p.fillRect(bar, QColor(theme.DANGER if self.amp > 0.8 else theme.ACCENT))
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QPen(QColor(theme.TEXT_DIM), 1))
        p.drawText(QPointF(r.left(), self.height() - 6),
                   f"level {self.amp:.2f}   freq {self.freq:.2f}"
                   + ("   ● held" if self.pressed else ""))
        pt = QPointF(r.left() + r.width() * self.freq, r.bottom() - r.height() * self.amp)
        p.setPen(QPen(QColor(theme.TEXT), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(r.left(), pt.y()), QPointF(r.right(), pt.y()))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.DANGER if self.pressed else theme.ACCENT))
        p.drawEllipse(pt, 7, 7)
        p.end()


class TriRecordPad(PositionPad):
    """The electrode triangle: drag for position, wheel / keys for volume."""

    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, editable=True)
        self.point = (0.0, 0.0)
        self.volume = 0.0
        self.spring = True
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pointDragged.connect(lambda *_: self.valueChanged.emit())

    def set_volume(self, v: float) -> None:
        self.volume = float(np.clip(v, 0.0, 1.0))
        self.update()
        self.valueChanged.emit()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton and self.volume <= 0.0:
            self.set_volume(0.6)                 # pressing plays; wheel adjusts
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.spring:
            self.set_volume(0.0)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        self.set_volume(self.volume + (WHEEL_STEP if delta > 0 else -WHEEL_STEP))
        event.accept()

    def keyPressEvent(self, event):
        d = _digit(event)
        if d is not None and not event.isAutoRepeat():
            self.set_volume(d / 10.0)
            event.accept()
            return
        super().keyPressEvent(event)


class RecordPanel(QWidget):
    recordingChanged = pyqtSignal(bool)
    takeRecorded = pyqtSignal(str)               # clip id

    def __init__(self, doc, playback, editor, parent=None):
        super().__init__(parent)
        self.doc, self.playback, self.editor = doc, playback, editor
        self.recorder: rec.TakeRecorder | None = None
        self.track_id: str | None = None
        self.track_type = ""
        self._return_to = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.target = QLabel()
        self.target.setWordWrap(True)
        lay.addWidget(self.target)
        self.stack = QStackedWidget()
        self.pad = RecordPad()
        self.tri_pad = TriRecordPad()
        self.none_label = QLabel("Select a standard or triphase track to record on.")
        self.none_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.none_label.setWordWrap(True)
        for w in (self.pad, self.tri_pad, self.none_label):
            self.stack.addWidget(w)
        lay.addWidget(self.stack, 1)

        form = QFormLayout()
        self.channels = QComboBox()
        for label, key in (("A + B", "AB"), ("A only", "A"), ("B only", "B")):
            self.channels.addItem(label, key)
        form.addRow("Channels", self.channels)
        self.tri_freq = QDoubleSpinBox()
        self.tri_freq.setRange(0.0, 1.0)
        self.tri_freq.setSingleStep(0.05)
        self.tri_freq.setValue(0.5)
        form.addRow("Pulse freq", self.tri_freq)
        self.hold = QCheckBox("Hold level on release")
        self.hold.toggled.connect(self._sync_spring)
        form.addRow(self.hold)
        self.monitor = QCheckBox("Monitor — the output plays what you perform")
        self.monitor.setChecked(True)
        form.addRow(self.monitor)
        self.punch = QCheckBox("Punch in: record only inside the loop region")
        form.addRow(self.punch)
        self.preroll = QComboBox()
        for s in (0, 2, 4):
            self.preroll.addItem(f"{s} s", float(s))
        self.preroll.setCurrentIndex(1)
        form.addRow("Pre-roll", self.preroll)
        self.smooth = QSpinBox()
        self.smooth.setRange(0, 500)
        self.smooth.setSingleStep(10)
        self.smooth.setValue(40)
        self.smooth.setSuffix(" ms")
        form.addRow("Smoothing", self.smooth)
        lay.addLayout(form)

        row = QHBoxLayout()
        self.rec_btn = QPushButton("●  Record  (R)")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setStyleSheet(f"QPushButton:checked {{ background: {theme.DANGER}; "
                                   "color: white; font-weight: bold; }")
        self.rec_btn.clicked.connect(lambda _c: self.toggle_record())
        row.addWidget(self.rec_btn)
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        row.addWidget(self.status, 1)
        lay.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        editor.currentTrackChanged.connect(lambda _t: self.refresh())
        doc.modelChanged.connect(self.refresh)
        playback.playingChanged.connect(self._on_playing)
        self.refresh()

    # ── target ──────────────────────────────────────────────────────────────
    @property
    def recording(self) -> bool:
        return self.recorder is not None

    def _sync_spring(self) -> None:
        self.pad.spring = self.tri_pad.spring = not self.hold.isChecked()

    def refresh(self) -> None:
        if self.recording:
            return
        tid = self.editor.current_track_id
        track = None
        if tid:
            try:
                track = self.doc.project.track_by_id(tid)
            except KeyError:
                track = None
        kind = track.type if track is not None else ""
        if kind == "standard":
            self.stack.setCurrentWidget(self.pad)
            self.target.setText(f"Recording onto <b>{track.name}</b>")
        elif kind == "triphase":
            self.stack.setCurrentWidget(self.tri_pad)
            self.tri_pad.set_calibration(Calibration.from_dict(self.doc.project.triphase_cal))
            self.target.setText(f"Recording onto <b>{track.name}</b> (triphase)")
        else:
            self.stack.setCurrentWidget(self.none_label)
            self.target.setText("No track to record on")
        self.channels.setEnabled(kind == "standard")
        self.tri_freq.setEnabled(kind == "triphase")
        self.rec_btn.setEnabled(kind in ("standard", "triphase"))
        self.track_id = track.id if kind in ("standard", "triphase") else None
        self.track_type = kind

    # ── the current input ───────────────────────────────────────────────────
    def current_row(self) -> tuple[float, float, float, float]:
        """The clip row being performed right now."""
        if self.track_type == "triphase":
            ux, uy = xy_to_unit(np.array(self.tri_pad.point or (0.0, 0.0)))
            return (self.tri_pad.volume, float(ux), float(uy), self.tri_freq.value())
        return rec.pad_row(self.pad.amp, self.pad.freq, self.channels.currentData())

    def live_input(self) -> LiveInput:
        row = self.current_row()
        if self.track_type == "triphase":
            cal = self.tri_pad.cal
            ab = tri_to_pulses(np.array([row], np.float32), cal)[0]
            return LiveInput(float(ab[0]), float(ab[1]), float(ab[2]), float(ab[3]))
        return LiveInput(*row)

    # ── recording ───────────────────────────────────────────────────────────
    def toggle_record(self) -> None:
        if self.recording:
            self.stop()
        else:
            self.start()
        self.rec_btn.setChecked(self.recording)

    def start(self) -> bool:
        pb = self.playback
        self.refresh()
        if self.track_id is None:
            self.status.setText("Select a standard or triphase track first.")
            return False
        if pb.sink is None:
            self.status.setText("Choose an output first — Dry run records without a device.")
            return False
        region = pb.loop_region if self.punch.isChecked() else None
        if self.punch.isChecked() and region is None:
            self.status.setText("Punch in needs a loop region (right-drag on the ruler).")
            return False
        if pb.playing:
            start = pb.position_s()
        else:
            start = region[0] if region else self.doc.cursor_s
        self.recorder = rec.TakeRecorder(start, region[1] if region else None,
                                         self.smooth.value())
        self._return_to = start
        if not pb.playing:
            self.doc.set_cursor(max(0.0, start - self.preroll.currentData()))
            if not pb.play():
                self.recorder = None
                self.doc.set_cursor(start)
                self.status.setText("Could not start playback.")
                return False
        note = ""
        if self.monitor.isChecked() and not getattr(pb.sink, "supports_live", True):
            note = " (this output cannot monitor live)"
        self.status.setText("● recording" + note)
        self._timer.start()
        self.recordingChanged.emit(True)
        self.rec_btn.setChecked(True)
        return True

    def _tick(self) -> None:
        if self.recorder is None:
            return
        t = self.playback.position_s()
        if t < 0:
            return
        r = self.recorder
        inside = r._inside(t)
        if inside:
            r.add(t, self.current_row())
        monitoring = self.monitor.isChecked() and inside
        self.playback.source.set_live(self.live_input() if monitoring else None)
        self.status.setText(("● REC " if inside else "pre-roll ") + theme.format_time(t, 1))

    def _on_playing(self, playing: bool) -> None:
        if not playing and self.recording:
            self.stop()

    def stop(self, commit: bool = True) -> str | None:
        """End the take; returns the new clip id (or None)."""
        recorder, self.recorder = self.recorder, None
        self._timer.stop()
        self.playback.source.set_live(None)
        self.rec_btn.setChecked(False)
        if recorder is None:
            return None
        if self.playback.playing:
            self.playback.stop()
        self.recordingChanged.emit(False)
        clip_id = self._commit(recorder) if commit else None
        self.doc.set_cursor(self._return_to)
        self.refresh()
        return clip_id

    def _commit(self, recorder: rec.TakeRecorder) -> str | None:
        res = recorder.result()
        if res is None or len(res[1]) < FS // 4:
            self.status.setText("Take too short — nothing recorded.")
            return None
        start, data = res
        try:
            track = self.doc.project.track_by_id(self.track_id)
        except KeyError:
            return None
        n = rec.next_take_number(self.doc.project)
        end = start + len(data) / FS
        self.doc.undo.beginMacro(f"Record take {n}")
        try:
            for old in rec.overlapping_takes(track, start, end):
                self.doc.set_clip_attrs(old.id, text="Mute earlier take", muted=True)
            clip = Clip(PulseDataSource(data, origin=f"take {n}"), start_s=start,
                        length_s=len(data) / FS, name=f"Take {n}")
            self.doc.add_clip(track.id, clip)
        finally:
            self.doc.undo.endMacro()
        self.status.setText(f"Take {n}: {theme.format_time(len(data) / FS, 1)} recorded.")
        self.editor.select_clips([clip.id])
        self.takeRecorded.emit(clip.id)
        return clip.id
