"""PlaybackController: transport state and the active output sink.

Owns the ``PulseSource`` the sinks read from, keeps it fed with the latest
master render (so edits reach a running output within one render), maps the
edit cursor and loop region to the sink's window, and drains the sinks' event
queue on a timer. Nothing here blocks on hardware: sinks run their own threads.
"""

from __future__ import annotations

import queue

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .. import outputs
from ..core.pulse import FS
from ..outputs import (EV_BATTERY, EV_DEVICES, EV_ERROR, EV_FRAME, EV_POWER, EV_STATUS,
                       EV_STOPPED, STATUS_CONNECTED, STATUS_DISCONNECTED)

POLL_MS = 33   # ~30 Hz playhead


class PlaybackController(QObject):
    playheadChanged = pyqtSignal(float)        # absolute seconds; -1 when stopped
    playingChanged = pyqtSignal(bool)
    statusChanged = pyqtSignal(str)            # connection status of the sink
    outputChanged = pyqtSignal(str)            # output key
    loopChanged = pyqtSignal()
    message = pyqtSignal(str)
    powerReported = pyqtSignal(int, int)       # strength reported by the device
    batteryReported = pyqtSignal(int)
    logLine = pyqtSignal(str)
    devicesReported = pyqtSignal(list)         # Intiface device dicts

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.source = outputs.PulseSource()
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.sink: outputs.OutputSink | None = None
        self.output_key: str | None = None
        self.loop_enabled = False
        self.loop_region: tuple[float, float] | None = None
        self._offset = 0          # window start, in pulses of the master
        self._was_playing = False
        self._settings: dict = {}
        self._freq_hz: tuple[float, float] | None = None
        self._applied_cal: dict | None = None

        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._on_poll)
        self._poll.start()
        doc.renderReady.connect(self._on_render)
        doc.cursorChanged.connect(self._on_cursor)

    # ── output selection / configuration ────────────────────────────────────
    @property
    def info(self) -> outputs.OutputInfo | None:
        return outputs.info(self.output_key) if self.output_key else None

    @property
    def status(self) -> str:
        return self.sink.status if self.sink is not None else STATUS_DISCONNECTED

    def select_output(self, key: str) -> None:
        if key == self.output_key and self.sink is not None:
            return
        self._drop_sink()
        self.output_key = key
        try:
            self.sink = outputs.make(key, self.source, self.events)
        except Exception as exc:  # noqa: BLE001
            self.sink = None
            self.message.emit(f"Output unavailable: {exc}")
        self._apply_settings()
        if self.doc.last_render is not None:
            self._publish_window()
        self.outputChanged.emit(key)
        self.statusChanged.emit(self.status)

    def configure(self, settings: dict, freq_hz: tuple[float, float] | None = None) -> None:
        """Device settings from the panel; safe to call while playing."""
        self._settings = dict(settings)
        self._freq_hz = freq_hz
        self._apply_settings()

    def _apply_settings(self) -> None:
        sink = self.sink
        if sink is None:
            return
        if self._freq_hz is not None:
            lo, hi = sorted(self._freq_hz)
            sink.freq_map = sink.default_freq_map.with_hz(lo, hi)
        self._applied_cal = dict(self.doc.project.triphase_cal)
        sink.configure({**self._settings, "triphase_cal": self._applied_cal})

    def set_controls(self, **changes) -> None:
        self.source.set_controls(**changes)

    def connect_output(self) -> None:
        if self.sink is not None and self.sink.status == STATUS_DISCONNECTED:
            self._apply_settings()
            self.sink.connect()
            self.statusChanged.emit(self.sink.status)

    def disconnect_output(self) -> None:
        if self.sink is not None:
            self.sink.stop(panic=True)
            self.sink.disconnect()
            self.statusChanged.emit(self.sink.status)
            self._stopped()

    def _drop_sink(self) -> None:
        if self.sink is not None:
            self.sink.stop(panic=True)
            self.sink.disconnect()
            self.sink = None
            self._stopped()

    # ── transport ───────────────────────────────────────────────────────────
    @property
    def playing(self) -> bool:
        return self.sink is not None and self.sink.playing

    def play(self) -> bool:
        sink = self.sink
        if sink is None:
            self.message.emit("Choose an output first.")
            return False
        if sink.playing:
            return True
        if sink.status != STATUS_CONNECTED:
            if self.info.explicit_connect:
                self.message.emit(f"Connect {self.info.label} first.")
                return False
            self.connect_output()          # local outputs connect on demand
            if sink.status != STATUS_CONNECTED:
                return False
        self._publish_window(current=True)
        window = self.source.window
        if not window.n:
            self.message.emit("Nothing to play — the project is empty.")
            return False
        sink.start(self._local_pulse(self.doc.cursor_s))
        controls = self.source.controls
        if not controls.muted and controls.effective_a == 0 and controls.effective_b == 0:
            self.message.emit("Playing at power 0 — raise power A/B in the Output panel.")
        self._was_playing = True
        self.playingChanged.emit(True)
        return True

    def stop(self) -> None:
        """Graceful stop (short fade)."""
        if self.sink is not None:
            self.sink.stop()

    def toggle_play(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def panic(self) -> None:
        """Instant silence, no fade."""
        if self.sink is not None:
            self.sink.stop(panic=True)
        self._stopped()
        self.message.emit("Output stopped (panic).")

    def _stopped(self) -> None:
        self.source.set_live(None)        # never leave a live monitor behind
        self._was_playing = False
        self.playheadChanged.emit(-1.0)
        self.playingChanged.emit(False)

    def position_s(self) -> float:
        """Absolute timeline position of the running output."""
        if not self.playing:
            return -1.0
        return self._offset / FS + self.sink.position()

    # ── loop ────────────────────────────────────────────────────────────────
    def set_loop_enabled(self, on: bool) -> None:
        self.loop_enabled = bool(on)
        self._republish_keeping_position()
        self.loopChanged.emit()

    def set_loop_region(self, region: tuple[float, float] | None) -> None:
        if region is not None:
            a, b = sorted(region)
            region = (max(0.0, a), b) if b - a >= 1.0 / FS else None
        self.loop_region = region
        self._republish_keeping_position()
        self.loopChanged.emit()

    # ── window plumbing ─────────────────────────────────────────────────────
    def _render(self, current: bool):
        doc = self.doc
        if doc.last_render is not None and (doc.render_is_current or not current):
            return doc.last_render
        return doc.render_now()

    def _publish_window(self, current: bool = False, render=None) -> None:
        render = self._render(current) if render is None else render
        tri = None
        if (self.sink is not None and self.sink.supports_triphase
                and render.triphase is not None and render.master_audio is not None):
            master, tri = render.master_audio, render.triphase   # phase-true triphase
        else:
            master = render.master                                # triphase as A/B
        n = len(master)
        if self.loop_enabled and self.loop_region is not None:
            a = min(n, round(self.loop_region[0] * FS))
            b = min(n, round(self.loop_region[1] * FS))
            if b > a:
                self._offset = a
                self.source.set_window(master[a:b], True, a,
                                       tri[a:b] if tri is not None else None)
                return
        self._offset = 0
        self.source.set_window(master, self.loop_enabled, 0, tri)

    def _republish_keeping_position(self) -> None:
        if self.doc.last_render is None and not self.playing:
            return
        pos = self.position_s()
        self._publish_window()
        if pos >= 0:
            self.sink.seek(self._local_pulse(pos))

    def _local_pulse(self, seconds: float) -> float:
        """Timeline seconds → pulse index within the current window (0 if outside)."""
        n = self.source.window.n
        local = round(seconds * FS) - self._offset
        return float(local) if 0 <= local < n else 0.0

    def _on_render(self, render, _seconds) -> None:
        # keep the running output on the latest edit; the window offset is
        # unchanged, so the sink's cursor stays valid
        self._publish_window(render=render)
        cal = dict(self.doc.project.triphase_cal)
        if self.sink is not None and cal != self._applied_cal:
            self._apply_settings()    # reconfigure only on a real change (wavelet resets)

    def _on_cursor(self, seconds: float) -> None:
        if self.playing:
            self.sink.seek(self._local_pulse(seconds))

    # ── events from the sink threads ────────────────────────────────────────
    def _on_poll(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == EV_STATUS:
                self.statusChanged.emit(str(payload))
            elif kind == EV_POWER:
                try:
                    a, b = payload
                    self.powerReported.emit(int(a), int(b))
                except (TypeError, ValueError):
                    pass
            elif kind == EV_BATTERY:
                self.batteryReported.emit(int(payload))
            elif kind == EV_DEVICES:
                self.devicesReported.emit(list(payload or []))
            elif kind == EV_FRAME:
                self.logLine.emit(str(payload))
            elif kind == EV_ERROR:
                self.message.emit(f"Output error: {payload}")
            elif kind == EV_STOPPED:
                if payload not in ("stopped", "panic"):
                    self.message.emit(f"Output stopped: {payload}")
        playing = self.playing
        if playing:
            self.playheadChanged.emit(self.position_s())
        elif self._was_playing:
            self._stopped()  # played out, faded out, or stopped by the device

    def shutdown(self) -> None:
        """Silence the device before the window goes away."""
        self._poll.stop()
        self._drop_sink()
