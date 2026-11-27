"""Output monitor: the sink's scope tap, the measurements, the widget in the Output panel."""

import numpy as np
import pytest

from stimstudio.audio.monitor import (FLOOR_DB, dominant_hz, find_trigger, levels_db,
                                      spectrum_db)
from stimstudio.outputs.audio import BLOCK_FRAMES, ScopeTap, WaveletSink, render_offline

SR = 48000


# ── the tap ─────────────────────────────────────────────────────────────────
def test_scope_tap_ring_and_latest():
    tap = ScopeTap(SR, seconds=0.05)                     # cap = 2400 frames
    assert tap.latest(10).shape == (10, 2) and not tap.latest(10).any()
    total = 0
    for _ in range(7):                                    # wraps around a few times
        block = np.arange(total, total + 1000, dtype=np.float32)
        tap.write(block, -block)
        total += 1000
    out = tap.latest(1500)
    assert np.array_equal(out[:, 0], np.arange(total - 1500, total, dtype=np.float32))
    assert np.array_equal(out[:, 1], -out[:, 0])
    assert tap.latest(10 ** 6).shape == (tap.cap, 2)      # clipped to the capacity
    assert tap.written == total


def test_tap_holds_what_the_card_got():
    pulses = np.zeros((40, 4), dtype=np.float32)
    pulses[:, 0] = 1.0
    pulses[:, 1] = 0.5
    pulses[:, 2:] = 0.5
    from stimstudio.outputs.base import PulseSource
    from queue import Queue
    from stimstudio.outputs.audio import ContinuousSink
    # render_offline drives the live callback; reproduce it to keep the sink around
    out = render_offline(pulses, "continuous", SR)
    source = PulseSource()
    source.set_window(pulses, loop=False)
    source.set_controls(power_a=200, power_b=200, soft_start=0.0)
    sink = ContinuousSink(source, Queue())
    sink._sr = SR
    sink._alloc()
    sink.configure({"sound_device": None})
    sink._stream = object()
    sink.start(0.0)
    block = np.zeros((BLOCK_FRAMES, 2), dtype=np.float32)
    for _ in range(6):
        sink._callback(block, BLOCK_FRAMES, None, None)
    got = sink.scope.latest(BLOCK_FRAMES)
    assert np.allclose(got, block)
    assert np.allclose(sink.scope.latest(6 * BLOCK_FRAMES), out[:6 * BLOCK_FRAMES], atol=1e-6)
    sink.stop(panic=True)
    sink._callback(block, BLOCK_FRAMES, None, None)       # silent blocks reach the tap too
    assert not sink.scope.latest(BLOCK_FRAMES).any()


# ── measurements ────────────────────────────────────────────────────────────
def sine(hz, seconds=0.2, amp=1.0, phase=0.0):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * hz * t + phase)).astype(np.float32)


def test_trigger_on_rising_crossing():
    x = sine(1000, 0.05, phase=1.0)
    win = 96                                              # 2 ms = two periods
    i = find_trigger(x, win)
    assert i is not None and i <= len(x) - win
    assert x[i - 1] <= 0.0 < x[i]
    assert find_trigger(np.zeros(1000, dtype=np.float32), 100) is None
    assert find_trigger(np.full(1000, 0.5, dtype=np.float32), 100) is None
    # noise around zero alone does not trigger (hysteresis)
    wiggle = np.tile(np.array([0.005, -0.005], dtype=np.float32), 500)
    assert find_trigger(wiggle, 100) is None


def test_trigger_on_wavelet_bursts():
    pulses = np.zeros((20, 4), dtype=np.float32)
    pulses[:, 0] = pulses[:, 1] = 1.0
    pulses[:, 2:] = 0.3
    x = render_offline(pulses, "wavelet", SR)[:, 0]
    i = find_trigger(x, 480)
    assert i is not None and x[i] > 0.0 and x[i - 1] <= 0.0


def test_spectrum_peak_and_level():
    x = sine(1000, 0.2)
    freqs, db = spectrum_db(x, SR)
    assert dominant_hz(freqs, db) == pytest.approx(1000, abs=3)
    # full-scale sine reads ≈ 0 dBFS (Hann scalloping costs up to ~1.4 dB between bins)
    assert db.max() == pytest.approx(0.0, abs=1.5)
    assert dominant_hz(*spectrum_db(np.zeros(4096), SR)) is None
    rms, peak = levels_db(np.stack([x, 0.5 * x], axis=1))
    assert rms[0] == pytest.approx(-3.01, abs=0.05) and peak[0] == pytest.approx(0.0, abs=0.01)
    assert peak[1] == pytest.approx(-6.02, abs=0.05)
    rms, peak = levels_db(np.zeros((100, 2)))
    assert (rms == FLOOR_DB).all() and (peak == FLOOR_DB).all()


def test_phase_between_channels():
    from stimstudio.audio.monitor import phase_between
    a = sine(500, 0.2)
    for lag in (0.0, 90.0, 120.0, 180.0):
        b = sine(500, 0.2, phase=-np.radians(lag))          # B lags A by `lag`
        hz, deg = phase_between(a, b, SR)
        assert hz == pytest.approx(500, abs=12)
        assert ((deg - lag + 180) % 360) - 180 == pytest.approx(0, abs=1.0)
    assert phase_between(a, np.zeros_like(a), SR) is None


# ── the widget in the Output panel ──────────────────────────────────────────
class StubSink:
    def __init__(self):
        self.scope = ScopeTap(SR)
        x = sine(700, 1.0)
        self.scope.write(x, 0.5 * x)


def test_scope_window_up_to_one_second(window, monkeypatch):
    """Windows up to 1 s; the tap keeps 3 s, room to find the trigger of a 1 s window."""
    mon = window.device_panel.monitor
    assert mon.window_ms.itemData(mon.window_ms.count() - 1) == 1000
    assert mon.window_ms.itemText(mon.window_ms.count() - 1) == "1 s"
    tap = ScopeTap(SR)
    assert tap.cap == 3 * SR
    sink = StubSink()
    sink.scope = tap
    x = sine(700, 3.0)
    tap.write(x, 0.5 * x)
    monkeypatch.setattr(window.playback, "sink", sink)
    mon.view.setCurrentText("Scope")
    mon.trigger.setCurrentIndex(1)                        # A ↑
    before = mon.window_ms.currentIndex()
    mon.window_ms.setCurrentIndex(mon.window_ms.count() - 1)
    try:
        mon.tick()
        xs, _ys = mon.scope_curves[0].getData()               # peak-downsampled: fewer points
        assert xs[-1] == pytest.approx(1000.0, abs=1.0)       # ms: the whole second is drawn
    finally:
        mon.window_ms.setCurrentIndex(before)                 # the choice is a saved setting


def test_monitor_only_for_audio_outputs(window):
    panel = window.device_panel
    row = panel.rows["monitor"]
    for key, visible in (("continuous", True), ("wavelet", True), ("howl", False),
                         ("dryrun", False)):
        idx = panel.device.findData(key)
        if idx < 0 or not panel.device.model().item(idx).isEnabled():
            continue
        panel.device.setCurrentIndex(idx)
        assert row.isVisibleTo(panel) == visible, key


def test_monitor_draws_every_view(window, monkeypatch):
    mon = window.device_panel.monitor
    monkeypatch.setattr(window.playback, "sink", StubSink())
    mon.trigger.setCurrentIndex(1)                        # A ↑
    mon.view.setCurrentText("Scope")
    mon.tick()
    xs, ys = mon.scope_curves[0].getData()
    assert len(xs) == int(SR * mon.window_ms.currentData() / 1000)
    assert ys[0] >= -0.05 and ys[1] > ys[0]               # starts on a rising crossing
    mon.view.setCurrentText("Spectrum")
    mon.tick()
    assert mon._spectrum_peaks[0] == pytest.approx(700, abs=5)
    assert "700 Hz" in mon.readout.text() or "699 Hz" in mon.readout.text()
    heights = mon.meter_bars.opts["height"]
    assert heights[0] > heights[1] > 0                    # A louder than B
    mon.hold.setChecked(True)
    before = mon.spec_curves[0].getData()[1].copy()
    mon.view.setCurrentText("Scope")
    mon.tick()                                            # held: the scope trace is not redrawn
    assert np.array_equal(mon.spec_curves[0].getData()[1], before)
    mon.hold.setChecked(False)
    monkeypatch.setattr(window.playback, "sink", None)    # no tap: flat, no error
    mon.tick()
    assert not mon.scope_curves[0].getData()[1].any()


class QuadratureSink:
    """A on the cosine, B on the sine: a circle in XY, B 90° behind A."""

    def __init__(self):
        self.scope = ScopeTap(SR)
        self.scope.write(sine(400, 1.0, phase=np.pi / 2), sine(400, 1.0))


def test_monitor_xy_view(window, monkeypatch):
    mon = window.device_panel.monitor
    monkeypatch.setattr(window.playback, "sink", QuadratureSink())
    mon.view.setCurrentText("XY")
    assert mon.window_ms.isEnabled() and not mon.trigger.isEnabled()
    mon.tick()
    xs, ys = mon.xy_curve.getData()
    assert len(xs) == int(SR * mon.window_ms.currentData() / 1000)
    radius = np.hypot(xs, ys)
    assert radius.min() > 0.95 and radius.max() < 1.05     # a circle
    hz, deg = mon._xy_phase
    assert hz == pytest.approx(400, abs=12) and deg == pytest.approx(90, abs=1.5)
    assert "+90°" in mon.readout.text()
    mon.view.setCurrentText("Scope")
