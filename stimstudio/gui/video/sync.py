"""Keeps a media player in step with the timeline.

The output sink owns the clock (``PlaybackController.playheadChanged``, ~30 Hz).
The player runs freely and is corrected: a large drift seeks, a small one nudges
the playback rate until it converges. When stopped, the player shows the frame
under the edit cursor.

The player is duck-typed (``position, setPosition, play, pause,
setPlaybackRate``) so the logic is testable without Qt Multimedia.
"""

from __future__ import annotations

import math
import time

SEEK_MS = 150.0       # drift above this → seek
NUDGE_MS = 40.0       # drift above this → speed up / slow down slightly
NUDGE = 0.05          # rate change while nudging
SEEK_SETTLE_S = 0.35  # ignore drift right after a seek (the decoder catches up)

BEFORE, INSIDE, AFTER, NONE = "before", "inside", "after", "none"


class VideoSync:
    def __init__(self, player, on_state=None, clock=time.monotonic):
        self.player = player
        self.on_state = on_state
        self.clock = clock
        self.start_s = 0.0
        self.duration_s = 0.0
        self.offset_ms = 0.0
        self.has_clip = False
        self.running = False          # player started by us during playback
        self.rate = 1.0
        self.state = NONE
        self._settle_until = 0.0

    # ── configuration ───────────────────────────────────────────────────────
    def set_clip(self, start_s: float, duration_s: float) -> None:
        self.start_s, self.duration_s = float(start_s), float(duration_s)
        self.has_clip = duration_s > 0

    def clear(self) -> None:
        self.has_clip = False
        self._halt()
        self._set_state(NONE)

    def expected_ms(self, t: float) -> float:
        return (t - self.start_s) * 1000.0 + self.offset_ms

    def where(self, t: float) -> str:
        if not self.has_clip or self.player is None:
            return NONE
        e = self.expected_ms(t)
        if e < 0:
            return BEFORE
        if e >= self.duration_s * 1000.0:
            return AFTER
        return INSIDE

    # ── timeline events ─────────────────────────────────────────────────────
    def playhead(self, t: float) -> None:
        """Running output at timeline second *t*."""
        state = self.where(t)
        self._set_state(state)
        if state != INSIDE:
            self._halt()
            return
        e = self.expected_ms(t)
        now = self.clock()
        if not self.running:
            self._seek(e, now)
            self._set_rate(1.0)
            self.player.play()
            self.running = True
            return
        if now < self._settle_until:
            return
        drift = float(self.player.position()) - e
        if abs(drift) > SEEK_MS:
            self._seek(e, now)
            self._set_rate(1.0)
        elif abs(drift) > NUDGE_MS:
            self._set_rate(1.0 - math.copysign(NUDGE, drift))   # ahead → slow down
        else:
            self._set_rate(1.0)

    def stopped(self, cursor_s: float) -> None:
        self._halt()
        self.scrub(cursor_s)

    def scrub(self, t: float) -> None:
        """Show the frame at timeline second *t* (not playing)."""
        if self.running:
            return
        state = self.where(t)
        self._set_state(state)
        if state == NONE:
            return
        e = min(max(self.expected_ms(t), 0.0), max(0.0, self.duration_s * 1000.0 - 1.0))
        self.player.setPosition(int(round(e)))

    # ── internals ───────────────────────────────────────────────────────────
    def _halt(self) -> None:
        if self.running:
            self.player.pause()
        self.running = False
        self._set_rate(1.0)

    def _seek(self, e: float, now: float) -> None:
        self.player.setPosition(int(round(e)))
        self._settle_until = now + SEEK_SETTLE_S

    def _set_rate(self, rate: float) -> None:
        if rate != self.rate:
            self.rate = rate
            self.player.setPlaybackRate(rate)

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            if self.on_state:
                self.on_state(state)
