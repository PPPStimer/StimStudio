"""A sink with no hardware behind it.

Its job is to make the whole output path — panel, playhead, transport, ramp,
mute, panic — exercisable on a machine with nothing plugged in, and to give the
tests something concrete to assert against.
"""

from __future__ import annotations

import time

from .base import (EV_FRAME, ClockedSink, Controls, FrequencyMap, OutputPulse,
                   format_frame)


class DryRunSink(ClockedSink):
    """Records every frame it is handed and logs a readable sample of them."""

    label    = "Dry run (no device)"
    requires = ()

    frame_interval   = 0.1       # same cadence as a Coyote 3, for realistic timing
    pulse_batch_size = 4
    default_freq_map = FrequencyMap(1, 200, 0.05, 0.5)

    LOG_INTERVAL = 0.25          # seconds between log lines, so the panel stays readable
    MAX_FRAMES   = 4096          # ring cap: an hour at 10 fps would be 36k frames

    def __init__(self, source, events):
        super().__init__(source, events)
        self.frames: list[list[OutputPulse]] = []
        self._last_log = 0.0

    def _emit(self, frame: list[OutputPulse], controls: Controls):
        self.frames.append(frame)
        if len(self.frames) > self.MAX_FRAMES:
            del self.frames[:len(self.frames) // 2]

        now = time.monotonic()
        if now - self._last_log >= self.LOG_INTERVAL:
            self._last_log = now
            self._post(EV_FRAME, format_frame(self.position(), frame[0]))
