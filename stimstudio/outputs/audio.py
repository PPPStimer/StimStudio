"""Audio output — the two experimental modes Howl offers for line-in estim boxes.

`ContinuousSink` emits an unbroken tone whose pitch follows the frequency track;
`WaveletSink` emits short bursts of a fixed carrier, repeated at the frequency
track's rate, which is what most audio-driven boxes actually expect.

There is no tick thread here. The PortAudio callback already runs on the sound
card's clock, so it derives the pulse index from its own sample position — one
clock instead of two, and no drift between them.

`sounddevice` is imported lazily so this module stays importable without it.
"""

from __future__ import annotations

import math
import queue

import numpy as np

from .base import (EV_ERROR, EV_STOPPED, L_AMP, L_FREQ, POWER_MAX, PULSE_RATE,
                   R_AMP, R_FREQ, STATUS_CONNECTED, STATUS_DISCONNECTED,
                   FrequencyMap, OutputSink, PulseSource, Window)

BLOCK_FRAMES = 512      # ~11 ms at 48 kHz: low latency, and a fixed size we can
                        # preallocate scratch buffers for
TWO_PI = 2.0 * math.pi


def wave_shape(name: str, phase: np.ndarray, out: np.ndarray, duty: float = 0.25):
    """Write one cycle-shape into `out` for the given phase in radians, in [-1, 1]."""
    if name == "sine":
        np.sin(phase, out=out)
        return
    t = np.mod(phase / TWO_PI, 1.0)
    if name == "square":
        np.subtract(1.0, 2.0 * (t >= 0.5), out=out)
    elif name == "triangle":
        np.subtract(1.0, 4.0 * np.abs(t - 0.5), out=out)
    elif name == "trapezoid":
        # A triangle with the slopes steepened until `duty` of the period is ramp
        # on each side and the rest is flat. Howl's exact slope constant is not
        # recoverable from its source; this reproduces the shape and the default.
        np.clip((1.0 - 4.0 * np.abs(t - 0.5)) / max(2.0 * duty, 1e-6), -1.0, 1.0, out=out)
    else:
        np.sin(phase, out=out)


class ScopeTap:
    """The last seconds of what the sound card received, for the output monitor
    (3 s: the longest scope window, 1 s, plus room to find its trigger).

    The audio callback writes, the GUI reads. There is no lock: `write` copies
    into a preallocated ring and bumps `written` last, so a reader racing the
    writer can at worst see one block that is half old, half new — fine for a
    display, and the callback never waits or allocates.
    """

    def __init__(self, sr: int, seconds: float = 3.0):
        self.sr = int(sr)
        self.cap = max(BLOCK_FRAMES * 2, int(self.sr * seconds))
        self.buf = np.zeros((self.cap, 2), dtype=np.float32)
        self.written = 0

    def write(self, left: np.ndarray, right: np.ndarray) -> None:
        n = len(left)
        if n <= 0:
            return
        pos = self.written % self.cap
        first = min(n, self.cap - pos)
        self.buf[pos:pos + first, 0] = left[:first]
        self.buf[pos:pos + first, 1] = right[:first]
        if first < n:
            rest = n - first
            self.buf[:rest, 0] = left[first:]
            self.buf[:rest, 1] = right[first:]
        self.written += n

    def latest(self, n: int) -> np.ndarray:
        """A copy of the newest *n* frames, shape (n, 2); zeros before the first write."""
        n = max(1, min(int(n), self.cap))
        end = self.written % self.cap
        if end >= n:
            return self.buf[end - n:end].copy()
        return np.concatenate((self.buf[self.cap - (n - end):], self.buf[:end]))


class AudioSink(OutputSink):
    """Shared PortAudio plumbing: the stream, the block clock, the pulse lerp."""

    requires   = ("sounddevice",)
    needs_link = True
    shape      = "sine"

    def __init__(self, source, events):
        super().__init__(source, events)
        self._stream = None
        self._device = None
        self._sr = 48000
        self._samples = 0            # samples emitted since Play
        self._start_pulse = 0.0
        self._scratch: dict[str, np.ndarray] = {}
        self.scope: ScopeTap | None = None

    # ── settings ────────────────────────────────────────────────────────
    def configure(self, settings: dict):
        self.shape = settings.get("shape", "sine")
        device = settings.get("sound_device")
        if device == self._device:
            return
        # A different card means a different stream. Swap it only between takes:
        # tearing the stream down mid-playback would click through the box.
        if self.status == STATUS_CONNECTED and not self._playing:
            self.disconnect()
            self._device = device
            self.connect()
        else:
            self._device = device

    # ── lifecycle ───────────────────────────────────────────────────────
    def connect(self):
        """Claim the sound card and start the stream, silent until Play.

        Keeping the stream open between takes avoids the click a stop/start
        would put through the box on every Play.
        """
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
            info = sd.query_devices(self._device, "output")
            self._sr = int(info["default_samplerate"])
            self._alloc()
            self._stream = sd.OutputStream(
                samplerate=self._sr, device=self._device, channels=2,
                dtype="float32", blocksize=BLOCK_FRAMES, latency="low",
                callback=self._callback, finished_callback=self._on_finished)
            self._stream.start()
        except Exception as exc:                          # noqa: BLE001
            self._stream = None
            self._post(EV_ERROR, f"could not open the sound device: {exc}")
            self._set_status(STATUS_DISCONNECTED)
            return
        self._set_status(STATUS_CONNECTED)

    def disconnect(self):
        self._playing = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.abort()
                stream.close()
            except Exception:                             # noqa: BLE001
                pass
        self._set_status(STATUS_DISCONNECTED)

    def start(self, at_pulse: float = 0.0):
        if self._playing or self._stream is None:
            return
        self._samples = 0
        self._start_pulse = float(at_pulse)
        self._reset_voices()
        super().start(at_pulse)

    def stop(self, *, panic: bool = False):
        was_playing = self._playing
        super().stop(panic=panic)
        if panic and was_playing:
            self._post(EV_STOPPED, "panic")

    def seek(self, at_pulse: float):
        """`_cursor` is recomputed from `_start_pulse` + samples every callback
        (see `_callback` below), so a plain `self._cursor = ...` would just be
        overwritten on the next block — reset the sample count instead."""
        if not self._playing:
            return
        self._samples = 0
        self._start_pulse = float(at_pulse)

    def _on_finished(self):
        """PortAudio dropped the stream out from under us — never stay 'connected'."""
        if self._stream is not None:
            self._playing = False
            self._stream = None
            self._post(EV_ERROR, "the audio stream stopped unexpectedly")
            self._set_status(STATUS_DISCONNECTED)

    # ── the callback ────────────────────────────────────────────────────
    def _callback(self, outdata, frames, _time_info, _status):
        # One consistent snapshot for the whole block; both are swapped
        # atomically by the GUI thread, never mutated in place.
        window = self.source.window
        controls = self.source.controls
        gain = self.gain()

        if not self._playing or not window.n or gain <= 0.0:
            outdata.fill(0.0)
            if not self._playing:
                self._samples = 0
            if self.scope is not None:
                silent = self._scratch["silence"][:frames]
                self.scope.write(silent, silent)
            return

        live = self.source.live
        if live is None:
            rows = self._interpolate(window, frames)
        else:                                   # monitoring a live input
            rows = self._scratch["rows"][:frames]
            rows[:] = (live.amp_a, live.amp_b, live.freq_a, live.freq_b)
        scale = 0.0 if controls.muted else gain
        buf = self._scratch
        amp_a = np.multiply(rows[:, L_AMP], scale * controls.effective_a / POWER_MAX,
                            out=buf["amp_a"][:frames])
        amp_b = np.multiply(rows[:, R_AMP], scale * controls.effective_b / POWER_MAX,
                            out=buf["amp_b"][:frames])
        hz_a = self.freq_map.to_hz_into(rows[:, L_FREQ], buf["hz_a"][:frames])
        hz_b = self.freq_map.to_hz_into(rows[:, R_FREQ], buf["hz_b"][:frames])

        left, right = buf["left"][:frames], buf["right"][:frames]
        self._render("a", left, amp_a, hz_a)
        self._render("b", right, amp_b, hz_b)
        if window.triphase is not None and self.supports_triphase and live is None:
            self._add_triphase(window, frames, scale * controls.effective_a / POWER_MAX,
                               scale * controls.effective_b / POWER_MAX, left, right)
        np.clip(left, -1.0, 1.0, out=left)
        np.clip(right, -1.0, 1.0, out=right)
        outdata[:, 0] = left       # channel A is the left channel, as in Howl
        outdata[:, 1] = right
        if self.scope is not None:
            self.scope.write(left, right)

        self._samples += frames
        self._cursor = self._start_pulse + self._samples * PULSE_RATE / self._sr
        if self.finished(window) or self.over_time_limit():
            self.stop()

    def _interpolate(self, window: Window, frames: int) -> np.ndarray:
        """The (frames, 4) pulse rows under this block, linearly interpolated.

        A pulse lasts 25 ms and a block is ~11 ms, so without interpolation the
        amplitude would step audibly; Howl lerps between pulses for the same reason.
        """
        buf = self._scratch
        position = np.multiply(buf["ramp"][:frames], PULSE_RATE / self._sr,
                               out=buf["position"][:frames])
        position += self._cursor
        base = np.floor(position, out=buf["base"][:frames])
        frac = np.subtract(position, base, out=buf["frac"][:frames])[:, None]
        i0, i1 = buf["i0"][:frames], buf["i1"][:frames]
        np.copyto(i0, base, casting="unsafe")   # truncates toward zero, as floor already did
        if window.loop:
            np.mod(i0, window.n, out=i0)
            np.mod(np.add(i0, 1, out=i1), window.n, out=i1)
        else:
            np.clip(i0, 0, window.n - 1, out=i0)
            np.clip(np.add(i0, 1, out=i1), 0, window.n - 1, out=i1)

        low, high, rows = (buf["low"][:frames], buf["high"][:frames], buf["rows"][:frames])
        np.take(window.pulses, i0, axis=0, out=low)
        np.take(window.pulses, i1, axis=0, out=high)
        np.subtract(high, low, out=rows)
        np.multiply(rows, frac, out=rows)
        return np.add(rows, low, out=rows)

    def _alloc(self):
        """Preallocate everything the callback touches — it must not allocate."""
        n = BLOCK_FRAMES
        self._scratch = {
            "left":  np.zeros(n, dtype=np.float32),
            "right": np.zeros(n, dtype=np.float32),
            "shape": np.zeros(n, dtype=np.float32),
            "amp_a": np.zeros(n, dtype=np.float32),
            "amp_b": np.zeros(n, dtype=np.float32),
            "hz_a":  np.zeros(n, dtype=np.float32),
            "hz_b":  np.zeros(n, dtype=np.float32),
            "phase": np.zeros(n, dtype=np.float64),
            "position": np.zeros(n, dtype=np.float64),
            "base": np.zeros(n, dtype=np.float64),
            "frac": np.zeros(n, dtype=np.float32),
            "ramp": np.arange(n, dtype=np.float64),
            "i0": np.zeros(n, dtype=np.int64),
            "i1": np.zeros(n, dtype=np.int64),
            "low":  np.zeros((n, 4), dtype=np.float32),
            "high": np.zeros((n, 4), dtype=np.float32),
            "rows": np.zeros((n, 4), dtype=np.float32),
            "silence": np.zeros(n, dtype=np.float32),
        }
        self.scope = ScopeTap(self._sr)

    # ── implemented per mode ────────────────────────────────────────────
    def _reset_voices(self):
        raise NotImplementedError

    def _render(self, side: str, out: np.ndarray, amp: np.ndarray, hz: np.ndarray):
        raise NotImplementedError


class ContinuousSink(AudioSink):
    """An unbroken tone per channel whose pitch tracks the frequency curve.

    Triphase data in the window is added on top as a real three-phase signal:
    both channels share one carrier (pitch from the triphase frequency) and the
    phasors from ``core.triphase`` set each channel's level and phase.
    """

    label = "Audio (continuous)"
    supports_triphase = True
    # The frequency track spans the tone's pitch range directly (500..1000 Hz by
    # default); the range itself can be set anywhere in the audible 20 Hz..20 kHz.
    default_freq_map = FrequencyMap(20, 20000).with_hz(500, 1000)

    def __init__(self, source, events):
        super().__init__(source, events)
        self._phase = {"a": 0.0, "b": 0.0}
        self._tri_phase = 0.0
        from ..core.triphase import Calibration
        self.tri_cal = Calibration()

    def configure(self, settings: dict):
        super().configure(settings)
        from ..core.triphase import Calibration
        self.tri_cal = Calibration.from_dict(settings.get("triphase_cal"))

    def _reset_voices(self):
        self._phase = {"a": 0.0, "b": 0.0}
        self._tri_phase = 0.0

    def _add_triphase(self, window, frames, scale_a, scale_b, left, right):
        """Mix the window's triphase data into the block (allocates a little)."""
        from ..core.triphase import output_phasors, unit_to_xy
        pos = self._cursor + np.arange(frames) * (PULSE_RATE / self._sr)
        i0 = np.floor(pos).astype(np.int64)
        frac = (pos - i0)[:, None]
        n = window.n
        if window.loop:
            i0, i1 = i0 % n, (i0 + 1) % n
        else:
            i0, i1 = np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1)
        tri = window.triphase
        rows = tri[i0] + (tri[i1] - tri[i0]) * frac
        a, b = output_phasors(rows[:, 0], unit_to_xy(rows[:, 1]), unit_to_xy(rows[:, 2]),
                              self.tri_cal)
        hz = self.freq_map.to_hz_array(rows[:, 3])
        phase = self._tri_phase + np.cumsum(hz) * (TWO_PI / self._sr)
        carrier = np.exp(1j * phase)
        left += (scale_a * np.real(a * carrier)).astype(np.float32)
        right += (scale_b * np.real(b * carrier)).astype(np.float32)
        self._tri_phase = float(phase[-1] % TWO_PI)

    def _render(self, side: str, out: np.ndarray, amp: np.ndarray, hz: np.ndarray):
        frames = len(out)
        phase = self._scratch["phase"][:frames]
        np.cumsum(hz, out=phase)
        phase *= TWO_PI / self._sr
        phase += self._phase[side]
        shape = self._scratch["shape"][:frames]
        wave_shape(self.shape, phase, shape)
        np.multiply(shape, amp, out=out)
        self._phase[side] = float(phase[-1] % TWO_PI)


class _Voice:
    """Where one channel is inside its burst / rest cycle, across blocks."""

    __slots__ = ("burst_pos", "burst_amp", "rest")

    def __init__(self, length: int):
        self.burst_pos = length     # length means "idle"
        self.burst_amp = 0.0
        self.rest = 0


class WaveletSink(AudioSink):
    """Short windowed bursts of a fixed carrier, repeated at the pulse frequency.

    The burst shape never changes, so it is built once and copied in — the
    callback only decides *when* each burst starts and how loud it is.
    """

    label = "Audio (wavelet)"
    default_freq_map = FrequencyMap(1, 200, 0.05, 0.5)

    MIN_REST = 100      # samples; Howl's floor, keeps bursts from merging

    def __init__(self, source, events):
        super().__init__(source, events)
        self.carrier = 1000
        self.width = 5
        self.fade = 0.5
        self._wavelet = np.zeros(1, dtype=np.float32)
        self._voices = {"a": _Voice(1), "b": _Voice(1)}

    def configure(self, settings: dict):
        super().configure(settings)
        self.carrier = int(settings.get("carrier", 1000))
        self.width = int(settings.get("wavelet_width", 5))
        self.fade = float(settings.get("wavelet_fade", 0.5))
        self._build_wavelet()

    def connect(self):
        super().connect()
        if self.status == STATUS_CONNECTED:
            self._build_wavelet()

    def _build_wavelet(self):
        """One burst: `width` carrier cycles under a raised-cosine envelope."""
        length = max(1, int(self.width * self._sr / max(self.carrier, 1)))
        n = np.arange(length, dtype=np.float64)
        carrier = np.empty(length, dtype=np.float32)
        wave_shape(self.shape, n * (TWO_PI * self.carrier / self._sr), carrier)

        fade_len = int(length * 0.5 * self.fade)
        envelope = np.ones(length, dtype=np.float32)
        if fade_len > 0:
            rise = np.arange(1, fade_len + 1, dtype=np.float64)
            envelope[:fade_len] = 0.5 * (1.0 - np.cos(math.pi * rise / fade_len))
            envelope[length - fade_len:] = envelope[fade_len - 1::-1]
        self._wavelet = (carrier * envelope).astype(np.float32)
        for voice in self._voices.values():
            voice.burst_pos = len(self._wavelet)
            voice.rest = 0

    def _reset_voices(self):
        length = len(self._wavelet)
        self._voices = {"a": _Voice(length), "b": _Voice(length)}

    def _render(self, side: str, out: np.ndarray, amp: np.ndarray, hz: np.ndarray):
        out.fill(0.0)
        wavelet = self._wavelet
        length = len(wavelet)
        voice = self._voices[side]
        frames = len(out)

        # A burst normally keeps the amplitude it was fired with, but mute and a
        # zeroed strength are safety controls: cut the burst in flight instead of
        # letting up to 5 ms of it out.
        if not amp.any():
            voice.burst_pos = length
            voice.rest = 0
            return

        i = 0
        # Walks segment by segment, not sample by sample: at these rates a block
        # holds only a handful of burst/rest boundaries.
        while i < frames:
            if voice.burst_pos < length:
                take = min(length - voice.burst_pos, frames - i)
                np.multiply(wavelet[voice.burst_pos:voice.burst_pos + take],
                            voice.burst_amp, out=out[i:i + take])
                voice.burst_pos += take
                i += take
                if voice.burst_pos >= length:
                    freq = max(float(hz[min(i, frames - 1)]), 1e-3)
                    voice.rest = max(self.MIN_REST, int(self._sr / freq) - length)
            elif voice.rest > 0:
                take = min(voice.rest, frames - i)
                voice.rest -= take
                i += take
            else:
                voice.burst_amp = float(amp[i])
                voice.burst_pos = 0


# ── offline rendering (export) ──────────────────────────────────────────────
AUDIO_MODES = {"continuous": ContinuousSink, "wavelet": WaveletSink}


def render_offline(pulses: np.ndarray, mode: str = "continuous", sr: int = 48000,
                   settings: dict | None = None, freq_hz: tuple[float, float] | None = None,
                   power: tuple[int, int] = (POWER_MAX, POWER_MAX),
                   triphase: np.ndarray | None = None) -> np.ndarray:
    """Render a whole pulse array to a (samples, 2) float32 signal.

    Drives the very callback the live sink uses, block by block, so an exported
    file is sample-for-sample what the sound card would have played (minus the
    soft start, which is disabled here).
    """
    source = PulseSource()
    source.set_window(pulses, loop=False,
                      triphase=triphase if AUDIO_MODES[mode].supports_triphase else None)
    source.set_controls(power_a=int(power[0]), power_b=int(power[1]), soft_start=0.0)
    sink = AUDIO_MODES[mode](source, queue.Queue())
    sink._sr = int(sr)
    sink._alloc()
    sink.configure({"sound_device": None, **(settings or {})})
    if freq_hz is not None:
        sink.freq_map = sink.default_freq_map.with_hz(*freq_hz)
    sink._stream = object()          # stand-in: start() refuses without a stream
    sink.start(0.0)

    total = int(math.ceil(len(pulses) / PULSE_RATE * sink._sr))
    out = np.zeros((total, 2), dtype=np.float32)
    block = np.zeros((BLOCK_FRAMES, 2), dtype=np.float32)
    pos = 0
    while pos < total:
        frames = min(BLOCK_FRAMES, total - pos)
        sink._callback(block[:frames], frames, None, None)
        out[pos:pos + frames] = block[:frames]
        pos += frames
    sink._stream = None
    return out
