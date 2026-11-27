"""Video window: the project's video, kept in sync with the timeline.

The picture is a ``QVideoWidget`` (Qt Multimedia renders through the GPU and
decodes in hardware where the platform allows). The output sink is the clock:
see :class:`~.sync.VideoSync`.

Full screen uses a separate frameless window placed on the monitor that shows
the video dock; the player just switches its video output to it (the docked
widget is never reparented — ``QVideoWidget.setFullScreen`` would open on the
primary monitor as a stray top-level window).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QSlider, QSpinBox, QStackedLayout, QToolButton,
                             QVBoxLayout, QWidget)

from .. import theme
from ..settings import app_settings
from .sync import AFTER, BEFORE, INSIDE, NONE, VideoSync


def _video_widget_class():
    from PyQt6.QtMultimediaWidgets import QVideoWidget

    class VideoView(QVideoWidget):
        """Double-click / F11 toggle full screen; Esc leaves it (and PANIC still fires)."""

        def __init__(self, panel):
            super().__init__()
            self.panel = panel

        def mouseDoubleClickEvent(self, event):
            self.panel.toggle_fullscreen()
            event.accept()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_F11:
                self.panel.toggle_fullscreen()
                event.accept()
            elif event.key() == Qt.Key.Key_Escape:
                self.panel.set_fullscreen(False)
                if self.panel.panic:
                    self.panel.panic()
                event.accept()
            elif event.key() == Qt.Key.Key_Space and self.panel.toggle_play:
                self.panel.toggle_play()
                event.accept()
            else:
                super().keyPressEvent(event)

    return VideoView


class FullscreenVideo(QWidget):
    """Frameless black window covering one monitor, holding a second video view."""

    def __init__(self, panel):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("StimStudio — video")
        self.setStyleSheet("background: #000;")
        self.setCursor(Qt.CursorShape.BlankCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = _video_widget_class()(panel)
        self.view.setCursor(Qt.CursorShape.BlankCursor)
        lay.addWidget(self.view)

    def show_on(self, screen) -> None:
        geo = screen.geometry()
        self.setGeometry(geo)
        self.show()                                   # create the native window …
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(screen)                  # … on the dock's monitor
        self.setGeometry(geo)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.view.setFocus()


class VideoPanel(QWidget):
    openRequested = pyqtSignal()

    def __init__(self, doc, playback, parent=None):
        super().__init__(parent)
        self.doc, self.playback = doc, playback
        self.panic = playback.panic if playback is not None else None
        self.toggle_play = playback.toggle_play if playback is not None else None
        self.settings = app_settings()
        self.path = ""
        self.track_muted = False
        self._shuttle_muted = False     # fast-forward mutes the video's sound

        # Qt Multimedia objects are created when the project first gets a video
        # (see _ensure_media): a window without video never starts the backend.
        self.player = self.audio = self.view = None
        self.fullscreen: FullscreenVideo | None = None
        self.sync = VideoSync(None, self._on_state)
        self.sync.offset_ms = float(self.settings.value("video/offset_ms", 0.0, float))

        # toolbar
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        self.open_btn = QToolButton(text="Open video…")
        self.open_btn.clicked.connect(self.openRequested.emit)
        bar.addWidget(self.open_btn)
        self.name_label = QLabel("—")
        self.name_label.setMinimumWidth(40)
        bar.addWidget(self.name_label, 1)
        self.time_label = QLabel("")
        self.time_label.setObjectName("timeLabel")
        bar.addWidget(self.time_label)
        self.mute_btn = QToolButton(text="🔊", checkable=True)
        self.mute_btn.setToolTip("Mute the video's sound")
        self.mute_btn.toggled.connect(self._apply_audio)
        bar.addWidget(self.mute_btn)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setFixedWidth(80)
        self.volume.setToolTip("Video volume")
        self.volume.valueChanged.connect(self._apply_audio)
        bar.addWidget(self.volume)
        bar.addWidget(QLabel("Offset"))
        self.offset = QSpinBox()
        self.offset.setRange(-5000, 5000)
        self.offset.setSingleStep(10)
        self.offset.setSuffix(" ms")
        self.offset.setToolTip("Shift the picture against the timeline — positive shows "
                               "later frames (compensates output latency)")
        self.offset.setValue(int(self.sync.offset_ms))
        self.offset.valueChanged.connect(self._set_offset)
        bar.addWidget(self.offset)
        self.full_btn = QToolButton(text="⛶")
        self.full_btn.setToolTip("Full screen on this monitor (F11 / double-click; Esc leaves)")
        self.full_btn.clicked.connect(self.toggle_fullscreen)
        bar.addWidget(self.full_btn)

        # fast-forward row (a second row keeps the dock as narrow as before)
        from .shuttle import SPEEDS, Shuttle
        self.shuttle = Shuttle(doc, playback, end_s=self._content_end, parent=self)
        self.shuttle.speedChanged.connect(self._on_shuttle_speed)
        self.shuttle.positionChanged.connect(self._on_shuttle_position)
        ff_bar = QHBoxLayout()
        ff_bar.setContentsMargins(4, 0, 4, 2)
        ff_bar.addWidget(QLabel("Fast-forward ▶▶"))
        self.ff_buttons: dict[int, QToolButton] = {}
        for speed in SPEEDS:
            b = QToolButton(text=f"x{speed}", checkable=True)
            b.setToolTip(f"Fast-forward ×{speed}: the cursor and the video run {speed}× faster. "
                         "A playing output pauses (short fade) and resumes from the new spot "
                         "when you press the button again. Esc cancels the resume.")
            b.clicked.connect(lambda _c=False, s=speed: self.shuttle.toggle(s))
            ff_bar.addWidget(b)
            self.ff_buttons[speed] = b
        ff_bar.addStretch()

        self.placeholder = QLabel()
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet(f"background: #000; color: {theme.TEXT_DIM};")
        self.stack = QStackedLayout()
        self.stack.addWidget(self.placeholder)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(bar)
        lay.addLayout(ff_bar)
        lay.addLayout(self.stack, 1)

        self.volume.setValue(int(self.settings.value("video/volume", 80, int)))
        self.mute_btn.setChecked(self.settings.value("video/muted", False, bool))
        self._apply_audio()

        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(30)
        self._scrub_timer.timeout.connect(self._scrub_to_cursor)   # bound method, not a lambda

        doc.modelChanged.connect(self.refresh)
        doc.cursorChanged.connect(self._on_cursor)
        if playback is not None:
            playback.playheadChanged.connect(self._on_playhead)
        self.refresh()

    def _ensure_media(self) -> None:
        if self.player is not None:
            return
        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.view = _video_widget_class()(self)
        self.player.setVideoOutput(self.view)
        self.player.errorOccurred.connect(self._on_error)
        self.stack.addWidget(self.view)
        self.sync.player = self.player
        self._apply_audio()

    # ── model ───────────────────────────────────────────────────────────────
    def video_clip(self):
        track = self.doc.project.video_track()
        if track is None or not track.clips:
            return None, None
        return track, track.clips[0]

    def refresh(self) -> None:
        track, clip = self.video_clip()
        if clip is None:
            if self.path and self.player is not None:
                self.player.stop()
                self.player.setSource(QUrl())
            self.path = ""
            self.sync.clear()
            self.name_label.setText("—")
            self._on_state(NONE)
            return
        self._ensure_media()
        path = clip.source.path
        if path != self.path:
            self.path = path
            self.sync.clear()
            if Path(path).is_file():
                self.player.setSource(QUrl.fromLocalFile(path))
            else:
                self.player.setSource(QUrl())
            self.name_label.setText(Path(path).name)
            self.name_label.setToolTip(path)
        self.track_muted = track.muted
        self._apply_audio()
        self.sync.set_clip(clip.start_s, clip.source.duration_s or clip.length_s)
        if not self.sync.running:
            self.sync.scrub(self.doc.cursor_s)

    # ── timeline events ─────────────────────────────────────────────────────
    def _on_playhead(self, seconds: float) -> None:
        if self.shuttle.busy:
            return            # the fading output's last playheads: fast-forward drives now
        if seconds < 0:
            self.sync.stopped(self.doc.cursor_s)
        else:
            self.sync.playhead(seconds)
        self._update_time(seconds if seconds >= 0 else self.doc.cursor_s)

    def _on_cursor(self, seconds: float) -> None:
        if self.shuttle.active:
            return
        if not self.sync.running:
            self._scrub_timer.start()
            self._update_time(seconds)

    def _scrub_to_cursor(self) -> None:
        self.sync.scrub(self.doc.cursor_s)

    def _update_time(self, t: float) -> None:
        if not self.sync.has_clip:
            self.time_label.setText("")
            return
        v = max(0.0, self.sync.expected_ms(t) / 1000.0)
        self.time_label.setText(f"{theme.format_time(t)}  ▸  {theme.format_time(v)}")

    def _on_state(self, state: str) -> None:
        missing = bool(self.path) and not Path(self.path).is_file()
        if state == INSIDE and not missing and not self.track_muted and self.view is not None:
            self.stack.setCurrentWidget(self.view)
            return
        if state == NONE:
            text = "No video\n\nFile → Open video…, or drop a video file on the timeline"
        elif missing:
            text = f"Video file not found:\n{self.path}\n\nUse Replace / Locate in the Inspector"
        elif self.track_muted:
            text = "Video track is muted"
        elif state == BEFORE:
            text = f"Video starts at {theme.format_time(self.sync.start_s)}"
        elif state == AFTER:
            text = "Video ended"
        else:
            text = ""
        self.placeholder.setText(text)
        self.stack.setCurrentWidget(self.placeholder)

    # ── fast-forward ────────────────────────────────────────────────────────
    def _content_end(self) -> float:
        """Where fast-forward stops: the end of the stim or of the video, whichever is later."""
        end = self.doc.project.duration_s()
        if self.sync.has_clip:
            end = max(end, self.sync.start_s + self.sync.duration_s
                      - self.sync.offset_ms / 1000.0)
        return max(0.0, end)

    def _on_shuttle_speed(self, speed: int) -> None:
        for s, b in self.ff_buttons.items():
            b.setChecked(s == speed)
        if speed:
            self._shuttle_muted = True
            self._apply_audio()
            return
        self._shuttle_muted = False
        self._apply_audio()
        # pause on the cursor's frame; a resumed output restarts the video at 1x
        self.sync.stopped(self.doc.cursor_s)
        self._update_time(self.doc.cursor_s)

    def _on_shuttle_position(self, seconds: float) -> None:
        if self.player is not None:
            self.sync.playhead(seconds, rate=float(self.shuttle.speed or 1))
        self._update_time(seconds)

    def _on_error(self, *_args) -> None:
        if self.path and self.player is not None:
            self.name_label.setText(f"{Path(self.path).name} — {self.player.errorString()}")

    # ── controls ────────────────────────────────────────────────────────────
    def _apply_audio(self, *_args) -> None:
        muted = self.mute_btn.isChecked() or self.track_muted or self._shuttle_muted
        if self.audio is not None:
            self.audio.setMuted(muted)
            self.audio.setVolume(self.volume.value() / 100.0)
        self.mute_btn.setText("🔇" if self.mute_btn.isChecked() else "🔊")
        self.settings.setValue("video/volume", self.volume.value())
        self.settings.setValue("video/muted", self.mute_btn.isChecked())
        if self.sync.state == INSIDE or self.track_muted:
            self._on_state(self.sync.state)

    def _set_offset(self, value: int) -> None:
        self.sync.offset_ms = float(value)
        self.settings.setValue("video/offset_ms", value)
        if not self.sync.running:
            self.sync.scrub(self.doc.cursor_s)

    def is_fullscreen(self) -> bool:
        return self.fullscreen is not None and self.fullscreen.isVisible()

    def target_screen(self):
        """The monitor showing the video dock (or the main window as a fallback)."""
        from PyQt6.QtGui import QGuiApplication
        from PyQt6.QtCore import QPoint
        centre = self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2))
        return (QGuiApplication.screenAt(centre) or self.screen()
                or QGuiApplication.primaryScreen())

    def set_fullscreen(self, on: bool) -> None:
        if self.player is None or on == self.is_fullscreen():
            return
        position = self.player.position()
        if on:
            if self.fullscreen is None:
                self.fullscreen = FullscreenVideo(self)
            self.player.setVideoOutput(self.fullscreen.view)
            self.fullscreen.show_on(self.target_screen())
        else:
            self.player.setVideoOutput(self.view)
            self.fullscreen.hide()
            self.window().activateWindow()
        if not self.sync.running:                     # paused: show the frame again
            self.player.setPosition(position)

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.is_fullscreen())

    def shutdown(self) -> None:
        """Stop and release the media backend before the window goes away."""
        self.shuttle.cancel()
        if self.player is None:
            return
        self.set_fullscreen(False)
        if self.fullscreen is not None:
            self.fullscreen.deleteLater()
            self.fullscreen = None
        self.sync.clear()                 # late timeline events must not reach the player
        self.path = ""
        self.player.stop()
        self.player.setVideoOutput(None)
        self.player.setAudioOutput(None)
        self.player.setSource(QUrl())
        self.sync.player = None
        for obj in (self.player, self.audio):
            obj.deleteLater()
        self.player = self.audio = None
