"""Device output core: the loop window, the control state, the sink contract.

Deliberately Qt-free and hardware-free — everything in this module runs headless
and is covered by `tests/test_outputs.py`.

Ported from HWL Viewer's ``outputs`` package. The conversion formulas are ported
one-for-one from Howl
(github.com/Amethyst-Sysadmin/Howl, `OutputBase.kt`) so that a file played from
here sounds the same as the same file played from the Android app.

Threading contract
------------------
`PulseSource` is written from the GUI thread and read from a BLE worker thread or
from the PortAudio callback. Both readers must never block, so the shared state
is kept in frozen dataclasses that are swapped by rebinding one attribute — an
atomic operation under the GIL. A reader grabs the reference once at the top of
its callback and works from that consistent snapshot for the whole frame.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.pulse import FS, L_AMP, L_FREQ, R_AMP, R_FREQ  # noqa: F401  (re-exported)
from ..io.hwl import encode_hwl  # noqa: F401  (re-exported for howl_remote)

PULSE_RATE = FS
PULSE_INTERVAL = 1.0 / PULSE_RATE

# Howl's channel strength scale — the same 0..200 the Coyote firmware uses.
POWER_MAX = 200

# Nothing plays longer than this without the user pressing Play again.
MAX_PLAY_SECONDS = 2 * 60 * 60

STOP_FADE_S = 0.3        # graceful stop; panic skips it entirely

PulseArray = np.ndarray  # shape (N, 4), float32, values in 0..1

_EMPTY = np.zeros((0, 4), dtype=np.float32)
_EMPTY.setflags(write=False)     # shared by every default Window; nobody may touch it


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


# ── shared state ────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Window:
    """The stretch of pulses being played, already detached from the editor.

    `pulses` is always a private contiguous copy, so nothing the editor does
    afterwards can change what a running transport reads.
    """

    pulses: PulseArray = field(default_factory=lambda: _EMPTY)
    loop: bool = False
    offset: int = 0          # index of pulses[0] within the file, for the playhead
    #: triphase data (volume, x, y, freq) parallel to ``pulses``, for sinks that
    #: synthesise real phase (continuous audio); ``pulses`` then excludes it
    triphase: PulseArray | None = None

    @property
    def n(self) -> int:
        return len(self.pulses)

    @property
    def duration(self) -> float:
        return len(self.pulses) / PULSE_RATE


@dataclass(frozen=True, slots=True)
class Controls:
    """Live knobs. Replaced wholesale rather than mutated field by field."""

    power_a:    int   = 0
    power_b:    int   = 0
    limit_a:    int   = POWER_MAX
    limit_b:    int   = POWER_MAX
    muted:      bool  = False
    soft_start: float = 2.0

    @property
    def effective_a(self) -> int:
        return int(clamp(self.power_a, 0, self.limit_a))

    @property
    def effective_b(self) -> int:
        return int(clamp(self.power_b, 0, self.limit_b))


@dataclass(frozen=True, slots=True)
class LiveInput:
    """One live pulse row (amp A, amp B, freq A, freq B; 0..1) played instead of
    the window while recording with monitoring on."""

    amp_a: float = 0.0
    amp_b: float = 0.0
    freq_a: float = 0.5
    freq_b: float = 0.5

    def row(self) -> np.ndarray:
        return np.array([self.amp_a, self.amp_b, self.freq_a, self.freq_b], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class OutputPulse:
    """One pulse after conversion: amplitude still 0..1, frequency now in Hz.

    Power rides along because the Coyotes carry it in the same packet, while the
    audio sinks fold it into the sample amplitude instead.
    """

    amp_a: float = 0.0
    amp_b: float = 0.0
    freq_a_hz: float = 0.0
    freq_b_hz: float = 0.0
    power_a: int = 0
    power_b: int = 0


def format_frame(position: float, pulse: OutputPulse) -> str:
    """One compact line describing a frame, for the dry-run log."""
    return (f"{position:7.3f}s  "
            f"A {pulse.amp_a:.2f} @{pulse.freq_a_hz:5.1f}Hz  "
            f"B {pulse.amp_b:.2f} @{pulse.freq_b_hz:5.1f}Hz  "
            f"pwr {pulse.power_a}/{pulse.power_b}")


class PulseSource:
    """The handoff point between the editor and whichever transport is running.

    Both attributes are rebound, never mutated; see the module docstring.
    """

    def __init__(self):
        self.window: Window = Window()
        self.controls: Controls = Controls()
        #: while set, sinks that can monitor play this instead of the window
        self.live: LiveInput | None = None

    def set_window(self, pulses: PulseArray, loop: bool, offset: int = 0,
                   triphase: PulseArray | None = None):
        """Publish a fresh snapshot. Safe to call while a transport is running."""
        data = np.ascontiguousarray(pulses, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(f"expected an (N, 4) pulse array, got {data.shape}")
        tri = None
        if triphase is not None:
            tri = np.ascontiguousarray(triphase, dtype=np.float32).copy()
            if tri.shape != data.shape:
                raise ValueError("triphase data must match the pulse window")
        self.window = Window(data.copy(), loop, offset, tri)

    def set_controls(self, **changes):
        self.controls = replace(self.controls, **changes)

    def set_live(self, live: LiveInput | None) -> None:
        """Monitor a live input (None returns to the window). Rebinding is atomic."""
        self.live = live


# ── frequency mapping ───────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FrequencyMap:
    """Normalised 0..1 frequency to device Hz, exactly as Howl does it.

    A device declares a hard range (`min_limit`..`max_limit`); the user picks a
    subset of that range, and the pulse value spans the subset. The integer
    truncation of the subset bounds is Howl's, kept for bit-identical output.
    """

    min_limit: int = 1
    max_limit: int = 200
    subset_start: float = 0.05
    subset_end: float = 0.5

    @property
    def min_hz(self) -> int:
        return self.min_limit + int((self.max_limit - self.min_limit) * self.subset_start)

    @property
    def max_hz(self) -> int:
        return self.min_limit + int((self.max_limit - self.min_limit) * self.subset_end)

    def to_hz(self, norm: float) -> float:
        span = self.max_hz - self.min_hz
        return clamp(self.min_hz + norm * span, self.min_limit, self.max_limit)

    def to_hz_array(self, norm: np.ndarray) -> np.ndarray:
        span = self.max_hz - self.min_hz
        return np.clip(self.min_hz + norm * span, self.min_limit, self.max_limit)

    def to_hz_into(self, norm: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Allocation-free variant, for the audio callback."""
        np.multiply(norm, self.max_hz - self.min_hz, out=out)
        np.add(out, self.min_hz, out=out)
        return np.clip(out, self.min_limit, self.max_limit, out=out)

    def with_hz(self, min_hz: float, max_hz: float) -> "FrequencyMap":
        """Rebuild from user-facing Hz bounds, inverting the subset arithmetic."""
        span = self.max_limit - self.min_limit
        # a hair above the exact fraction: min_hz / max_hz truncate span × subset,
        # and a float a hair below the integer would lose a whole Hz (1000 → 999)
        lo = clamp((min_hz - self.min_limit + 1e-6) / span, 0.0, 1.0)
        hi = clamp((max_hz - self.min_limit + 1e-6) / span, 0.0, 1.0)
        return replace(self, subset_start=min(lo, hi), subset_end=max(lo, hi))


# ── events back to the GUI ──────────────────────────────────────────────────
# Sinks run on their own threads (and, for audio, inside the PortAudio callback),
# so they never touch Qt. They post (kind, payload) tuples here instead and the
# panel drains the queue on a timer.
EV_STATUS  = "status"    # payload: one of the STATUS_* strings below
EV_BATTERY = "battery"   # payload: int percent
EV_POWER   = "power"     # payload: (power_a, power_b) reported by the device
EV_ERROR   = "error"     # payload: str
EV_STOPPED = "stopped"   # payload: str reason
EV_FRAME   = "frame"     # payload: str — dry-run / debug only
EV_DEVICES = "devices"   # payload: list of device dicts (Intiface)

STATUS_DISCONNECTED = "disconnected"
STATUS_CONNECTING   = "connecting"
STATUS_CONNECTED    = "connected"


# ── the sink contract ───────────────────────────────────────────────────────
class OutputSink:
    """One device output: owns its transport, its clock and its ramp.

    Subclasses implement `_open` / `_close` (connection lifecycle) and
    `_emit` (hand one frame of already-converted pulses to the hardware).
    The base class does everything that is the same everywhere: the soft-start
    ramp, mute, the power ceiling and the play cursor.
    """

    label      = "?"
    requires   = ()          # importable module names this sink needs
    verified   = True        # False shows a "not tested on hardware" note
    needs_link = False       # True for anything that must Connect before Play

    #: True if the sink renders ``Window.triphase`` with phase (else it is given
    #: the A/B-converted mix)
    supports_triphase = False
    #: False for sinks that cannot follow a live input (Howl plays whole files)
    supports_live = True
    pulse_divider    = 1     # 1 -> 40 Hz, 2 -> 20 Hz
    pulse_batch_size = 1
    default_freq_map = FrequencyMap()

    def __init__(self, source: PulseSource, events: "queue.Queue[tuple[str, object]]"):
        self.source = source
        self.events = events
        self.freq_map = self.default_freq_map
        self.status = STATUS_DISCONNECTED
        self._playing = False
        self._started_at = 0.0
        self._stopping_at = 0.0   # non-zero while the graceful fade runs
        self._cursor = 0.0        # pulses elapsed since Play, before wrapping

    # ── lifecycle, overridden by subclasses ─────────────────────────────
    def configure(self, settings: dict):
        """Apply the panel's device-specific settings. Ignored by default.

        Called whenever a spin box moves, so it must tolerate being invoked
        mid-playback and must never block.
        """

    def connect(self):
        """Bring the transport up. Non-blocking: progress arrives as events."""
        self._set_status(STATUS_CONNECTED)

    def disconnect(self):
        self.stop(panic=True)
        self._set_status(STATUS_DISCONNECTED)

    def start(self, at_pulse: float = 0.0):
        if self._playing:
            return
        self._cursor = float(at_pulse)
        self._started_at = time.monotonic()
        self._stopping_at = 0.0
        self._playing = True

    def stop(self, *, panic: bool = False):
        if not self._playing:
            return
        if panic:
            self._playing = False
            self._stopping_at = 0.0
            self._cursor = 0.0
        elif not self._stopping_at:
            self._stopping_at = time.monotonic()

    def seek(self, at_pulse: float):
        """Jump the play cursor without touching `playing` or the ramp state —
        used when the user clicks a new position while already playing.
        `AudioSink` overrides this: its `_cursor` is derived each callback
        rather than authoritative, so it has its own bookkeeping to update.
        """
        if not self._playing:
            return
        self._cursor = float(at_pulse)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def stopping(self) -> bool:
        """Still playing, but inside the graceful stop's fade-out."""
        return self._playing and bool(self._stopping_at)

    def position(self) -> float:
        """Seconds into the window, wrapped when looping."""
        window = self.source.window
        if not window.n:
            return 0.0
        pos = self._cursor % window.n if window.loop else min(self._cursor, window.n)
        return pos / PULSE_RATE

    # ── the ramp ────────────────────────────────────────────────────────
    def gain(self, now: float | None = None) -> float:
        """0..1 envelope: eased in on Play, faded out on a graceful Stop.

        A hard ceiling is not enough on its own — jumping straight to full
        amplitude is the unpleasant part, so nothing starts at full scale.
        """
        if not self._playing:
            return 0.0
        now = time.monotonic() if now is None else now
        if self._stopping_at:
            done = (now - self._stopping_at) / STOP_FADE_S
            if done >= 1.0:
                self._playing = False
                self._stopping_at = 0.0
                return 0.0
            return 1.0 - done
        ramp = self.source.controls.soft_start
        if ramp <= 0.0:
            return 1.0
        return clamp((now - self._started_at) / ramp, 0.0, 1.0)

    # ── reading the window ──────────────────────────────────────────────
    def pulse_at(self, window: Window, index: float) -> np.ndarray:
        """One (4,) row, wrapping when looping and clamping when not."""
        if not window.n:
            return np.zeros(4, dtype=np.float32)
        i = int(index)
        i = i % window.n if window.loop else min(i, window.n - 1)
        return window.pulses[i]

    def finished(self, window: Window) -> bool:
        """True once a non-looping window has played out."""
        return not window.loop and self._cursor >= window.n

    def over_time_limit(self) -> bool:
        return time.monotonic() - self._started_at > MAX_PLAY_SECONDS

    # ── helpers for subclasses ──────────────────────────────────────────
    def _set_status(self, status: str):
        self.status = status
        self.events.put((EV_STATUS, status))

    def _post(self, kind: str, payload: object = None):
        self.events.put((kind, payload))


class ClockedSink(OutputSink):
    """A sink that drives itself: one frame of `pulse_batch_size` pulses per tick.

    Covers both Coyotes and the dry run. The audio sinks do *not* use this — the
    PortAudio callback is already a clock, and running a second one alongside it
    would only add a source of drift.
    """

    frame_interval = 0.1     # seconds of signal carried by one transmission

    def __init__(self, source, events):
        super().__init__(source, events)
        self._clock = PulseClock(self.frame_interval, self._tick,
                                 name=f"{type(self).__name__}-clock",
                                 on_error=self._on_clock_error)

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self, at_pulse: float = 0.0):
        if self._playing:
            return
        super().start(at_pulse)
        self._clock.start()

    def stop(self, *, panic: bool = False):
        was_playing = self._playing
        super().stop(panic=panic)
        if panic and was_playing:
            self._clock.stop()
            self._silence()
            self._post(EV_STOPPED, "panic")

    def disconnect(self):
        self._clock.stop()
        super().disconnect()

    # ── the frame ───────────────────────────────────────────────────────
    def _tick(self, skipped: int):
        gain = self.gain()
        if not self._playing:            # the graceful fade ran out inside gain()
            self._end("stopped")
            return

        window = self.source.window
        controls = self.source.controls
        self._emit(self._collect(window, controls, gain), controls)
        self._cursor += (1 + skipped) * self.frame_interval * PULSE_RATE

        if self.finished(window):
            self.stop()
        elif self.over_time_limit():
            self.stop(panic=True)
            self._post(EV_STOPPED, "playback time limit reached")

    def _collect(self, window: Window, controls: Controls, gain: float) -> list[OutputPulse]:
        live = self.source.live
        if live is not None:
            pulse = self._convert(live.row(), controls, gain)
            return [pulse] * self.pulse_batch_size
        step = self.frame_interval * PULSE_RATE / self.pulse_batch_size
        return [self._convert(self.pulse_at(window, self._cursor + k * step), controls, gain)
                for k in range(self.pulse_batch_size)]

    def _convert(self, row: np.ndarray, controls: Controls, gain: float) -> OutputPulse:
        # Mute zeroes the amplitude rather than halting the stream: the device
        # keeps its cadence and resumes instantly, which is how Howl does it.
        scale = 0.0 if controls.muted else gain
        return OutputPulse(
            amp_a=clamp(float(row[L_AMP]) * scale, 0.0, 1.0),
            amp_b=clamp(float(row[R_AMP]) * scale, 0.0, 1.0),
            freq_a_hz=self.freq_map.to_hz(float(row[L_FREQ])),
            freq_b_hz=self.freq_map.to_hz(float(row[R_FREQ])),
            power_a=controls.effective_a,
            power_b=controls.effective_b,
        )

    def _silence(self):
        """One frame of zero-amplitude pulses — the last thing sent on a stop."""
        quiet = OutputPulse(freq_a_hz=self.freq_map.min_hz,
                            freq_b_hz=self.freq_map.min_hz)
        try:
            self._emit([quiet] * self.pulse_batch_size, Controls())
        except Exception as exc:                          # noqa: BLE001
            self._post(EV_ERROR, f"could not send the stop frame: {exc}")

    def _end(self, reason: str):
        self._clock.stop()
        self._silence()
        self._post(EV_STOPPED, reason)

    def _on_clock_error(self, exc: BaseException):
        self._playing = False
        self._post(EV_ERROR, f"{type(self).__name__}: {exc}")
        self._silence()

    # ── implemented by the device ───────────────────────────────────────
    def _emit(self, frame: list[OutputPulse], controls: Controls):
        raise NotImplementedError


# ── a clock for the transports that need one ────────────────────────────────
class PulseClock:
    """Absolute-deadline tick thread.

    Deadlines are absolute so jitter never accumulates. Falling more than one
    interval behind snaps the schedule forward instead of firing a catch-up
    burst: the Coyote consumes exactly one packet per frame, so a burst would
    pile up inside the device and show as growing latency, not as a fast
    passage. Howl's main loop guards the same way.
    """

    def __init__(self, interval: float, tick, name: str = "pulse-clock", on_error=None):
        self.interval = interval
        self._tick = tick
        self._name = name
        self._on_error = on_error
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            skipped = 0
            next_tick += self.interval
            now = time.monotonic()
            if next_tick - now < -self.interval:
                skipped = int((now - next_tick) / self.interval)
                next_tick = now
            try:
                self._tick(skipped)
            except Exception as exc:                      # noqa: BLE001
                # A raise here would die silently in a daemon thread and leave
                # the device transmitting whatever it last received.
                if self._on_error is not None:
                    self._on_error(exc)
                return
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
