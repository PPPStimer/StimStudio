"""Fast-forward (shuttle): run the edit cursor and the video at 2×, 5× or 10×.

Stim is never played time-compressed. Starting fast-forward while the output
plays stops it with the usual short fade and remembers to resume; turning
fast-forward off then starts the output again from the new position, with the
normal soft start. PANIC cancels the resume, and so does running off the end.

The shuttle owns its own clock (a monotonic timer scaled by the speed) and
moves ``doc.cursor_s``; the video panel feeds the same position to
:class:`~.sync.VideoSync` with the speed as the nominal playback rate.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

SPEEDS = (2, 5, 10)
TICK_MS = 33


class Shuttle(QObject):
    speedChanged = pyqtSignal(int)          # 0 = off
    positionChanged = pyqtSignal(float)     # timeline seconds while running

    def __init__(self, doc, playback, end_s=None, clock=time.monotonic, parent=None):
        super().__init__(parent)
        self.doc, self.playback = doc, playback
        self.end_s = end_s or (lambda: float("inf"))
        self.clock = clock
        #: called to restart the output after fast-forward (MainWindow puts the
        #: safety prompt in front of it); None → ``playback.play``
        self.play_fn = None
        self.speed = 0
        self.resume = False
        self._waiting = False        # off, but the output is still fading out
        self._t0 = 0.0
        self._anchor = 0.0
        self._last_set: float | None = None

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.tick)
        doc.cursorChanged.connect(self._on_cursor)
        if playback is not None:
            playback.playingChanged.connect(self._on_playing)
            if hasattr(playback, "panicked"):
                playback.panicked.connect(self.cancel)

    @property
    def active(self) -> bool:
        return self.speed > 0

    @property
    def busy(self) -> bool:
        """Running, or off but waiting for the output's fade to end before resuming."""
        return self.speed > 0 or self._waiting

    # ── control ─────────────────────────────────────────────────────────────
    def toggle(self, speed: int) -> None:
        """A speed button: start / change speed, or off when it is the current one."""
        self.set_speed(0 if speed == self.speed else speed)

    def set_speed(self, speed: int) -> None:
        speed = max(0, int(speed))
        if speed == self.speed:
            return
        if speed == 0:
            self._finish(resume=self.resume)
            return
        if not self.active:
            self._waiting = False
            pb = self.playback
            playing = pb is not None and pb.playing
            t0 = pb.position_s() if playing else self.doc.cursor_s
            self.resume = playing
            self.speed = speed           # before stop(): its signals must see us active
            if playing:
                pb.stop()                # the usual short fade, on the device thread
            self._anchor_at(max(0.0, t0))
            self.timer.start()
        else:
            self._anchor_at(self.position())
            self.speed = speed
        self.speedChanged.emit(self.speed)

    def cancel(self) -> None:
        """Off, and never resume (PANIC, Stop, end of the content)."""
        self._waiting = False
        self.resume = False
        if self.active:
            self._finish(resume=False)
        else:
            self.timer.stop()

    # ── clock ───────────────────────────────────────────────────────────────
    def position(self) -> float:
        return self._t0 + (self.clock() - self._anchor) * self.speed

    def _anchor_at(self, t: float) -> None:
        self._t0, self._anchor = float(t), self.clock()
        self._set_cursor(t)

    def _set_cursor(self, t: float) -> None:
        self._last_set = t
        self.doc.set_cursor(t)
        self._last_set = self.doc.cursor_s     # set_cursor rounds to the pulse grid

    def tick(self) -> None:
        if self._waiting:
            self._try_resume()
            return
        if not self.active:
            self.timer.stop()
            return
        t = self.position()
        end = self.end_s()
        if t >= end:
            self._set_cursor(end)
            self.positionChanged.emit(end)
            self.cancel()
            return
        self._set_cursor(t)
        self.positionChanged.emit(t)

    # ── ending ──────────────────────────────────────────────────────────────
    def _finish(self, resume: bool) -> None:
        self.speed = 0
        self.resume = False
        self.speedChanged.emit(0)
        if resume:
            self._waiting = True
            self._try_resume()
        else:
            self._waiting = False
            self.timer.stop()

    def _try_resume(self) -> None:
        pb = self.playback
        if pb is not None and pb.playing:
            if not self.timer.isActive():   # the fade is still running: wait for it
                self.timer.start()
            return
        self._waiting = False
        self.timer.stop()
        play = self.play_fn or (pb.play if pb is not None else None)
        if play is not None:
            play()

    # ── outside events ──────────────────────────────────────────────────────
    def _on_cursor(self, seconds: float) -> None:
        if self.active and self._last_set is not None and abs(seconds - self._last_set) > 1e-9:
            self._t0, self._anchor = float(seconds), self.clock()   # the user clicked
            self._last_set = seconds

    def _on_playing(self, playing: bool) -> None:
        # Space, Record or Play while fast-forwarding: the output is running
        # again from the cursor, so fast-forward simply ends.
        if playing and (self.active or self._waiting):
            self._waiting = False
            self.resume = False
            if self.active:
                self._finish(resume=False)
            self.timer.stop()
