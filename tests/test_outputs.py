"""Output sinks (ported from HWL Viewer's tests/test_outputs.py, Coyote parts dropped).

No hardware is touched: audio sinks are driven by calling the PortAudio callback
directly, Howl remote goes through a fake HTTP function.
"""

import base64
import queue
import threading
import time

import numpy as np
import pytest

from stimstudio import outputs
from stimstudio.core.pulse import L_AMP, L_FREQ, R_AMP, R_FREQ
from stimstudio.io.hwl import HWL_MAGIC
from stimstudio.outputs import Controls, FrequencyMap, PulseSource
from stimstudio.outputs import howl_remote
from stimstudio.outputs.audio import (BLOCK_FRAMES, ContinuousSink, WaveletSink, render_offline,
                                      wave_shape)
from stimstudio.outputs.dryrun import DryRunSink
from stimstudio.outputs.howl_remote import HowlLink, HowlRemoteSink

SR = 48000


def window(n=200, amp_a=1.0, amp_b=0.5, freq_a=0.5, freq_b=0.5):
    p = np.zeros((n, 4), dtype=np.float32)
    p[:, L_AMP], p[:, R_AMP] = amp_a, amp_b
    p[:, L_FREQ], p[:, R_FREQ] = freq_a, freq_b
    return p


def audio_sink(cls, source, **settings):
    sink = cls(source, queue.Queue())
    sink._sr = SR
    sink._alloc()
    sink.configure({"shape": "sine", "carrier": 1000, "wavelet_width": 5,
                    "wavelet_fade": 0.5, "sound_device": None, **settings})
    sink._stream = object()
    return sink


def render(sink, blocks=8):
    out = np.zeros((BLOCK_FRAMES, 2), dtype=np.float32)
    chunks = []
    for _ in range(blocks):
        sink._callback(out, BLOCK_FRAMES, None, None)
        chunks.append(out.copy())
    return np.concatenate(chunks)


def wait(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_registry():
    assert [i.key for i in outputs.OUTPUTS] == ["howl", "continuous", "wavelet", "buttplug",
                                               "dryrun"]
    assert outputs.info("howl").explicit_connect
    assert outputs.info("buttplug").explicit_connect and not outputs.info("buttplug").verified
    assert not outputs.info("dryrun").missing


# ── frequency mapping ───────────────────────────────────────────────────────
def test_frequency_map():
    fmap = FrequencyMap(1, 200, 0.05, 0.5)
    assert (fmap.min_hz, fmap.max_hz) == (10, 100)
    assert fmap.to_hz(0.0) == 10.0 and fmap.to_hz(1.0) == 100.0 and fmap.to_hz(0.5) == 55.0
    audio = FrequencyMap(500, 1000, 0.0, 1.0)
    assert (audio.min_hz, audio.max_hz) == (500, 1000)
    rt = fmap.with_hz(10, 100)
    assert (rt.min_hz, rt.max_hz) == (10, 100)
    flipped = fmap.with_hz(100, 10)
    assert flipped.subset_start < flipped.subset_end
    buf = np.zeros(3, dtype=np.float32)
    assert np.allclose(fmap.to_hz_into(np.array([0.0, 0.5, 1.0], np.float32), buf), [10, 55, 100])


def test_frequency_range_round_trip():
    """The Hz typed in the Output panel come back exactly (a float a hair under
    the integer used to lose 1 Hz: 50 → 49, 1000 → 999)."""
    for lo, hi in ((1, 200), (20, 20000)):
        base = FrequencyMap(lo, hi)
        for a in list(range(lo, hi + 1, 7)) + [hi]:
            for b in (a, min(hi, a + 1), min(hi, a + 333), hi):
                m = base.with_hz(a, b)
                assert (m.min_hz, m.max_hz) == (a, b), (lo, hi, a, b)


def test_continuous_pitch_range_up_to_20_khz():
    fmap = ContinuousSink.default_freq_map
    assert (fmap.min_limit, fmap.max_limit) == (20, 20000)
    assert (fmap.min_hz, fmap.max_hz) == (500, 1000)                 # the default pitch
    high = fmap.with_hz(15000, 20000)
    assert high.to_hz(0.0) == 15000 and high.to_hz(1.0) == 20000


# ── pulse source / safety ───────────────────────────────────────────────────
def test_pulse_source_snapshot_and_wrap():
    source = PulseSource()
    live = window(10)
    source.set_window(live, loop=False)
    live[0, L_AMP] = 0.0
    assert source.window.pulses[0, L_AMP] == 1.0
    with pytest.raises(ValueError):
        source.set_window(np.zeros((4, 3), np.float32), loop=False)
    sink = DryRunSink(source, queue.Queue())
    source.set_window(window(10), loop=True)
    assert np.array_equal(sink.pulse_at(source.window, 12), sink.pulse_at(source.window, 2))
    source.set_window(window(10), loop=False)
    assert np.array_equal(sink.pulse_at(source.window, 99), sink.pulse_at(source.window, 9))
    assert sink.pulse_at(outputs.Window(), 0).tolist() == [0, 0, 0, 0]


def test_safety_controls_ramp_and_panic():
    c = Controls(power_a=180, power_b=50, limit_a=100, limit_b=200)
    assert (c.effective_a, c.effective_b) == (100, 50)

    source = PulseSource()
    source.set_window(window(40), loop=True)
    source.set_controls(power_a=200, power_b=200, soft_start=0.0)
    sink = DryRunSink(source, queue.Queue())
    sink.connect()
    assert sink.gain() == 0.0
    sink.start()
    frame = sink._collect(source.window, source.controls, 1.0)
    assert frame[0].amp_a == 1.0 and frame[0].power_a == 200
    source.set_controls(muted=True)
    frame = sink._collect(source.window, source.controls, 1.0)
    assert frame[0].amp_a == 0.0 and frame[0].amp_b == 0.0 and frame[0].power_a == 200
    source.set_controls(muted=False)

    ramped = DryRunSink(source, queue.Queue())
    source.set_controls(soft_start=4.0)
    ramped.start()
    ramped._started_at -= 1.0
    assert abs(ramped.gain() - 0.25) < 0.02
    ramped._started_at -= 10.0
    assert ramped.gain() == 1.0
    ramped.stop(panic=True)
    assert ramped.gain() == 0.0 and not ramped.playing

    sink.seek(77)
    assert sink._cursor == 77.0 and sink.playing
    sink.stop(panic=True)
    sink.seek(5)
    assert sink._cursor == 0.0
    sink.disconnect()


# ── audio ───────────────────────────────────────────────────────────────────
def test_continuous_audio():
    source = PulseSource()
    source.set_window(window(400, amp_a=0.8, amp_b=0.4), loop=True)
    source.set_controls(power_a=200, power_b=100, soft_start=0.0)
    cont = audio_sink(ContinuousSink, source)
    assert (cont.freq_map.min_hz, cont.freq_map.max_hz) == (500, 1000)
    cont.start()
    sig = render(cont)
    assert np.isfinite(sig).all() and sig.shape == (BLOCK_FRAMES * 8, 2)
    assert abs(float(np.abs(sig[:, 0]).max()) - 0.8) < 1e-3
    assert abs(float(np.abs(sig[:, 1]).max()) - 0.2) < 1e-3
    crossings = int(np.sum(np.diff(np.signbit(sig[:, 0])) != 0))
    assert abs(crossings / 2 / (len(sig) / SR) - 750) < 15

    phase_before = cont._phase["a"]
    more = render(cont, blocks=1)
    assert abs(float(more[0, 0]) - float(np.sin(phase_before + 2 * np.pi * 750 / SR) * 0.8)) < 0.02

    source.set_controls(power_a=0, power_b=0)
    assert np.all(render(cont, blocks=1) == 0.0)
    source.set_controls(power_a=200, power_b=100, muted=True)
    assert np.all(render(cont, blocks=1) == 0.0)
    source.set_controls(muted=False)
    cont.seek(123)
    assert cont._start_pulse == 123.0 and cont._samples == 0
    cont.stop(panic=True)
    cont.seek(9)
    assert cont._start_pulse == 123.0


def test_wavelet_audio():
    source = PulseSource()
    source.set_window(window(400, amp_a=0.8, amp_b=0.4), loop=True)
    source.set_controls(power_a=200, power_b=100, soft_start=0.0)
    wl = audio_sink(WaveletSink, source)
    assert len(wl._wavelet) == 5 * SR // 1000
    assert abs(float(wl._wavelet[0])) < 1e-6
    assert abs(float(np.abs(wl._wavelet).max()) - 1.0) < 1e-3
    wl.start()
    sig = render(wl, blocks=24)
    assert np.isfinite(sig).all() and float(np.abs(sig).max()) <= 1.0

    silent = np.abs(sig[:, 0]) < 1e-5
    rests, i = [], 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            if j - i > 30:
                rests.append((i, j - i))
            i = j
        else:
            i += 1
    expected_period = SR / 55.0
    expected_rest = max(100, int(expected_period) - len(wl._wavelet))
    assert len(rests) > 5
    assert all(abs(length - expected_rest) <= 1 for _, length in rests)
    periods = np.diff([start for start, _ in rests])
    assert np.all(np.abs(periods - expected_period) <= 2)
    source.set_controls(power_a=0, power_b=0)
    assert np.all(render(wl, blocks=1) == 0.0)
    wl.stop(panic=True)


@pytest.mark.parametrize("name", ["sine", "square", "triangle", "trapezoid"])
def test_wave_shapes(name):
    phase = np.linspace(0, 2 * np.pi, 9, dtype=np.float64)[:-1]
    out = np.zeros(len(phase), dtype=np.float32)
    wave_shape(name, phase, out)
    assert np.isfinite(out).all() and float(np.abs(out).max()) <= 1.0 + 1e-6


# ── offline render (audio export) ───────────────────────────────────────────
@pytest.mark.parametrize("mode", ["continuous", "wavelet"])
def test_offline_render_matches_live_callback(mode):
    pulses = np.random.default_rng(0).random((90, 4)).astype(np.float32)
    off = render_offline(pulses, mode, SR)
    assert off.shape == (int(np.ceil(90 / 40 * SR)), 2) and off.dtype == np.float32

    source = PulseSource()
    source.set_window(pulses, loop=False)
    source.set_controls(power_a=200, power_b=200, soft_start=0.0)
    sink = audio_sink(ContinuousSink if mode == "continuous" else WaveletSink, source)
    sink.start(0.0)
    live = render(sink, blocks=len(off) // BLOCK_FRAMES)
    np.testing.assert_array_equal(off[:len(live)], live)


def test_offline_render_freq_range_and_power():
    pulses = window(80, amp_a=1.0, amp_b=1.0, freq_a=0.5, freq_b=0.5)
    sig = render_offline(pulses, "continuous", SR, freq_hz=(600, 800), power=(200, 100))
    crossings = int(np.sum(np.diff(np.signbit(sig[:, 0])) != 0))
    assert abs(crossings / 2 / (len(sig) / SR) - 700) < 15
    assert abs(float(np.abs(sig[:, 1]).max()) - 0.5) < 1e-3
    assert np.all(render_offline(pulses, "continuous", SR, power=(0, 0)) == 0)


# ── Howl remote ─────────────────────────────────────────────────────────────
@pytest.fixture
def fake_http(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []
    responses: dict[str, object] = {}

    def fake(url, body, headers, timeout):
        calls.append((url, body, headers))
        for path, resp in responses.items():
            if url.endswith(path):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return {}

    monkeypatch.setattr(howl_remote, "_http_post", fake)
    return calls, responses


def _first(calls, suffix):
    return next(c for c in list(calls) if c[0].endswith(suffix))


def _has(calls, suffix):
    return any(u.endswith(suffix) for u, _, _ in list(calls))


def test_encode_hwl_for_howl():
    assert outputs.encode_hwl(np.zeros((0, 4), np.float32)) == HWL_MAGIC
    sample = window(3, amp_a=1.0, amp_b=0.25, freq_a=0.5, freq_b=0.75)
    assert outputs.encode_hwl(sample) == HWL_MAGIC + sample.astype("<f4").tobytes()


def test_howl_link(fake_http, monkeypatch):
    calls, responses = fake_http
    errors = []
    link = HowlLink(errors.append)
    link.configure("192.0.2.1", 4695, "secret")
    got = {}
    link.call("/status", {}, callback=lambda r: got.setdefault("status", r))
    assert wait(lambda: "status" in got)
    url, body, headers = calls[-1]
    assert url == "http://192.0.2.1:4695/status"
    assert headers["Authorization"] == "Bearer secret" and body == {}

    block = threading.Event()

    def slow(url, body, headers, timeout):
        calls.append((url, body, headers))
        block.wait(2.0)
        return {}

    monkeypatch.setattr(howl_remote, "_http_post", slow)
    assert link.call("/status", {}, important=True)
    assert wait(lambda: link._busy)
    assert link.call("/status", {}, important=False) is False
    block.set()
    assert wait(lambda: not link._busy)

    def boom(url, body, headers, timeout):
        raise RuntimeError("network is down")

    monkeypatch.setattr(howl_remote, "_http_post", boom)
    failed = {}
    link.call("/boom", {}, callback=lambda r: failed.setdefault("result", r))
    assert wait(lambda: "result" in failed) and failed["result"] is None
    assert any("network is down" in m for m in errors)
    link.close()


def test_howl_sink_end_to_end(fake_http):
    calls, responses = fake_http
    responses["/status"] = {"options": {"power_a": 10, "power_b": 20},
                            "player": {"position": 0.0, "duration": 1.0, "playing": True}}
    source = PulseSource()
    source.set_window(window(40), loop=True)
    source.set_controls(power_a=80, power_b=40)
    sink = HowlRemoteSink(source, queue.Queue())
    sink.configure({"remote_host": "192.0.2.2", "remote_port": 4695, "remote_key": "k"})
    sink.connect()
    assert wait(lambda: sink.status == outputs.STATUS_CONNECTED)
    assert wait(lambda: _has(calls, "/set_freq_range"))
    assert _first(calls, "/set_freq_range")[1] == {"min": 10, "max": 100}
    assert wait(lambda: _has(calls, "/set_swap_channels"))

    calls.clear()
    sink.configure({"remote_host": "192.0.2.2", "remote_port": 4695, "remote_key": "k",
                    "freq_min": 100, "freq_max": 20, "swap_channels": True})
    assert wait(lambda: _has(calls, "/set_freq_range"))
    assert _first(calls, "/set_freq_range")[1] == {"min": 20, "max": 100}

    sink.start(0.0)
    assert wait(lambda: _has(calls, "/load_hwl"))
    load = _first(calls, "/load_hwl")
    assert load[1]["loop"] is True and load[1]["title"] == "StimStudio"
    assert base64.b64decode(load[1]["hwl"]) == outputs.encode_hwl(source.window.pulses)
    assert wait(lambda: _has(calls, "/start_player")) and sink.playing
    assert wait(lambda: _has(calls, "/set_power"))
    assert _first(calls, "/set_power")[1] == {"power_a": 80, "power_b": 40}

    calls.clear()
    sink.seek(20)
    assert wait(lambda: _has(calls, "/seek"))
    assert _first(calls, "/seek")[1] == {"position": 0.5} and sink.position() == 0.5
    assert not _has(calls, "/load_hwl")

    calls.clear()
    source.set_window(window(50), loop=True)   # a new render
    assert wait(lambda: _has(calls, "/load_hwl"))
    assert base64.b64decode(_first(calls, "/load_hwl")[1]["hwl"]) == \
        outputs.encode_hwl(source.window.pulses)
    assert wait(lambda: _has(calls, "/start_player"))
    assert _first(calls, "/start_player")[1] == {"from": 0.5}

    sink.stop(panic=True)
    assert not sink.playing and wait(lambda: _has(calls, "/stop_player"))
    sink.disconnect()
    assert sink.status == outputs.STATUS_DISCONNECTED


def test_howl_scan(monkeypatch):
    import socket as socket_module
    reachable = {"10.0.0.5", "10.0.0.9"}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_create_connection(addr, timeout=None):
        if addr[0] in reachable:
            return _Conn()
        raise OSError("refused")

    monkeypatch.setattr(socket_module, "create_connection", fake_create_connection)
    found = howl_remote.scan_for_howl(port=4695, timeout=0.05,
                                      hosts=[f"10.0.0.{i}" for i in range(1, 11)])
    assert found == sorted(reachable)
    assert howl_remote.scan_for_howl(port=4695, hosts=[]) == []
    prefixes = howl_remote.local_subnets()
    assert all(p.count(".") == 2 for p in prefixes) and len(prefixes) == len(set(prefixes))
