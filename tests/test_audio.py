"""M6: audio → lanes, audio analysis, modulator tracks and the modulation matrix."""

import contextlib
import importlib.util
import io

import numpy as np
import pytest
import soundfile as sf
from PyQt6.QtCore import QPointF, Qt

from stimstudio.audio.analysis import analyze_file, analyze_samples, cached_analysis
from stimstudio.audio.convert import Cancelled, audio_to_pulses
from stimstudio.core.model import (AudioFileSource, Clip, GeneratorSource, ModRoute, Project,
                                   Track)
from stimstudio.core.pulse import FS
from stimstudio.core.render import render_project, render_track
from stimstudio.effects import get_effect
from stimstudio.io.project_file import load_project, save_project

SR = 44100


@pytest.fixture(scope="module")
def stereo_wav(tmp_path_factory):
    """6 s: L = 300 Hz tone ramping up in level, R = 3 kHz tone, gated on/off each second."""
    t = np.arange(int(SR * 6.0)) / SR
    left = np.sin(2 * np.pi * 300 * t) * np.linspace(0.05, 0.9, len(t))
    right = 0.5 * np.sin(2 * np.pi * 3000 * t) * (t % 1.0 < 0.5)
    path = tmp_path_factory.mktemp("audio") / "stereo.wav"
    sf.write(path, np.stack([left, right], 1).astype(np.float32), SR)
    return path


@pytest.fixture(scope="module")
def clicks_wav(tmp_path_factory):
    """8 s mono: a click every 0.5 s (from 0.25 s), bass first half, treble second half."""
    t = np.arange(int(SR * 8.0)) / SR
    y = 0.3 * np.sin(2 * np.pi * 60 * t) * (t < 4) + 0.2 * np.sin(2 * np.pi * 5000 * t) * (t >= 4)
    for k in range(16):
        s = int((0.25 + 0.5 * k) * SR)
        y[s:s + 400] += np.hanning(400) * 0.9
    path = tmp_path_factory.mktemp("audio") / "clicks.wav"
    sf.write(path, y.astype(np.float32), SR)
    return path


# ── audio → lanes ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["rolloff", "centroid", "yin"])
def test_audio_to_pulses(stereo_wav, method):
    stages = []
    p = audio_to_pulses(stereo_wav, method, lambda s, f: stages.append(s))
    assert p.shape == (240, 4) and p.dtype == np.float32
    assert p.min() >= 0 and p.max() <= 1
    assert "Loading audio" in stages and any("left" in s for s in stages)
    assert p[220:235, 0].mean() > 3 * p[5:20, 0].mean()            # L level ramps up
    on, off = p[4:16, 1], p[24:36, 1]                                # R gated each second
    assert on.mean() > 0.5 and off.mean() < 0.05
    assert np.median(p[40:200, 3]) > np.median(p[40:200, 2])        # R (3 kHz) higher than L


def test_audio_to_pulses_cancel(stereo_wav):
    with pytest.raises(Cancelled):
        audio_to_pulses(stereo_wav, "rolloff", cancel=lambda: True)


def test_mono_file_duplicates_channels(clicks_wav):
    p = audio_to_pulses(clicks_wav, "centroid")
    np.testing.assert_array_equal(p[:, 0], p[:, 1])
    np.testing.assert_array_equal(p[:, 2], p[:, 3])


# Reference: the previous column-major, whole-signal estimators (= hwlfromaudio's maths)
def _ref_frames(y):
    from stimstudio.audio.loader import frames
    from stimstudio.audio.convert import HOP, WINDOW
    return frames(y, WINDOW, HOP)


def _ref_gate(fr):
    return np.sqrt(np.mean(fr.astype(np.float32) ** 2, axis=0)) > 10.0 ** (-50.0 / 20.0)


def _ref_rolloff(y, sr, fmin, fmax):
    fr = _ref_frames(y)
    x = np.ascontiguousarray(fr, dtype=np.float32) * np.hanning(4096).astype(np.float32)[:, None]
    power = np.abs(np.fft.rfft(x, n=4096, axis=0)) ** 2
    freqs = np.fft.rfftfreq(4096, d=1.0 / sr).astype(np.float32)
    valid = (freqs >= fmin) & (freqs <= fmax)
    cum = np.cumsum(power[valid], axis=0)
    idx = np.argmax(cum >= 0.5 * cum[-1][None, :], axis=0)
    return np.where(_ref_gate(fr), freqs[valid][idx], 0.0).astype(np.float32)


def _ref_centroid(y, sr, fmin, fmax):
    fr = _ref_frames(y)
    x = np.ascontiguousarray(fr, dtype=np.float32) * np.hanning(4096).astype(np.float32)[:, None]
    power = np.abs(np.fft.rfft(x, n=4096, axis=0)) ** 2
    freqs = np.fft.rfftfreq(4096, d=1.0 / sr)
    valid = (freqs >= fmin) & (freqs <= fmax)
    pv, logf = power[valid], np.log(freqs[valid])
    total = pv.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lc = np.where(total > 0, (logf[:, None] * pv).sum(axis=0) / total, np.log(fmin))
    return np.where(_ref_gate(fr), np.exp(lc), 0.0).astype(np.float32)


def _ref_yin(y, sr, fmin, fmax, threshold=0.12):
    W = 4096
    fr = _ref_frames(y)
    x = np.ascontiguousarray(fr, dtype=np.float32)
    F = np.fft.rfft(x, n=2 * W, axis=0)
    acf = np.fft.irfft(np.abs(F) ** 2, n=2 * W, axis=0)[:W]
    d = 2.0 * acf[0] - 2.0 * acf
    d[0] = 0.0
    cumsum = np.cumsum(d[1:], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmndf = np.where(cumsum > 0, d[1:] * np.arange(1, W, dtype=np.float32)[:, None] / cumsum,
                         1.0)
    tau_min = max(1, int(np.ceil(sr / fmax)))
    tau_max = min(W - 2, int(np.floor(sr / fmin)))
    search = cmndf[tau_min - 1:tau_max]
    n_lags, n_frames = search.shape
    below = search < threshold
    has_below = below.any(axis=0)
    tau_idx = np.where(has_below, np.argmax(below, axis=0), np.argmin(search, axis=0))
    fi = np.arange(n_frames)
    for _ in range(1, 8):
        nxt = np.clip(tau_idx + 1, 0, n_lags - 1)
        lower = search[nxt, fi] < search[tau_idx, fi]
        tau_idx = np.where(has_below & lower & (search[nxt, fi] < threshold), nxt, tau_idx)
    s0 = search[np.clip(tau_idx - 1, 0, n_lags - 1), fi]
    s1 = search[tau_idx, fi]
    s2 = search[np.clip(tau_idx + 1, 0, n_lags - 1), fi]
    denom = s0 - 2.0 * s1 + s2
    with np.errstate(invalid="ignore", divide="ignore"):
        interp = np.clip(np.where(np.abs(denom) > 1e-6, 0.5 * (s0 - s2) / denom, 0.0), -1.0, 1.0)
    f0 = (sr / ((tau_idx + tau_min) + interp)).astype(np.float32)
    voiced = _ref_gate(fr) & (s1 < 0.45) & (f0 >= fmin) & (f0 <= fmax)
    return np.where(voiced, f0, 0.0).astype(np.float32)


@pytest.mark.parametrize("method,ref", [("rolloff", _ref_rolloff), ("centroid", _ref_centroid),
                                        ("yin", _ref_yin)])
def test_block_estimators_match_reference(method, ref):
    """Threaded row blocks give the same per-hop estimates as one whole-signal pass."""
    from stimstudio.audio.convert import BLOCK, FMIN, HOP, SAMPLE_RATE, detect_frequency
    sr = SAMPLE_RATE
    t = np.arange(int(sr * 4.5)) / sr                       # several blocks + a partial one
    y = 0.3 * np.sin(2 * np.pi * (180 + 60 * t) * t) * (np.abs(t - 2.0) > 0.4)
    y += 0.02 * np.random.default_rng(3).standard_normal(len(t))
    y[int(1.6 * sr):int(2.4 * sr)] = 0.0                     # silence → gate
    y = y.astype(np.float32)
    ours = detect_frequency(y, sr, method)
    theirs = ref(y, sr, FMIN, sr / 2.0)
    assert len(ours) == len(theirs) > 3 * BLOCK
    assert len(ours) == (len(y) - 4096) // HOP + 1
    assert np.mean(np.abs(ours - theirs) > 0.5) <= 1e-3


def test_frequency_lanes_stay_aligned_on_long_files(tmp_path):
    """A pitch change lands on the same pulse as the level change, even after 70 s."""
    sr = 40960
    t = np.arange(sr * 75) / sr
    switch = 70.0
    y = np.where(t < switch, 0.1 * np.sin(2 * np.pi * 200 * t), 0.8 * np.sin(2 * np.pi * 2000 * t))
    path = tmp_path / "long.wav"
    sf.write(path, y.astype(np.float32), sr)
    p = audio_to_pulses(path, "rolloff")
    k = int(switch * FS)
    assert p[k - 6, 0] < 0.3 and p[k + 6, 0] > 0.9           # amplitude switches at k
    assert p[k - 6, 2] < 0.2 and p[k + 6, 2] > 0.5           # … and so does frequency


def test_mono_detects_once_and_results_are_cached(clicks_wav, monkeypatch):
    import os
    import shutil
    from stimstudio.audio import convert
    calls = []
    real = convert.detect_frequencies
    monkeypatch.setattr(convert, "detect_frequencies",
                        lambda ch, *a, **k: (calls.append(len(ch)), real(ch, *a, **k))[1])
    loads = []
    real_load = convert.load_audio
    monkeypatch.setattr(convert, "load_audio", lambda *a: (loads.append(1), real_load(*a))[1])
    path = clicks_wav.parent / "cache_me.wav"
    shutil.copy(clicks_wav, path)
    a = audio_to_pulses(path, "yin")
    assert calls == [1] and len(loads) == 1                  # mono: one channel analysed
    a[:] = 0                                                 # callers get a private copy
    b = audio_to_pulses(path, "yin")
    assert len(loads) == 1 and b.any()                       # cache hit
    audio_to_pulses(path, "rolloff")
    assert len(loads) == 2                                   # per method
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    audio_to_pulses(path, "yin")
    assert len(loads) == 3                                   # file changed → re-read


@pytest.mark.skipif(importlib.util.find_spec("librosa") is None, reason="librosa missing")
def test_parity_with_hwl_viewer_frequencies(stereo_wav, tmp_path):
    """Frequencies match HWL Viewer's hwlfromaudio on every pulse both produce."""
    import shutil
    import sys
    from conftest import EXTERNAL
    folder = EXTERNAL / "HWL_viewer"
    if not folder.is_dir():
        pytest.skip("External Data/HWL_viewer not present")
    sys.path.insert(0, str(folder))
    import hwlfromaudio
    wav = tmp_path / "x.wav"
    shutil.copy(stereo_wav, wav)
    with contextlib.redirect_stdout(io.StringIO()):
        hwlfromaudio.convert_audio_file(wav, 40, "centroid")
    theirs = np.frombuffer(wav.with_suffix(".hwl").read_bytes()[8:], "<f4").reshape(-1, 4)
    ours = audio_to_pulses(wav, "centroid")
    m = min(len(ours), len(theirs)) - 1
    np.testing.assert_allclose(ours[:m, 2:], theirs[:m, 2:], atol=5e-4)


# ── analysis ────────────────────────────────────────────────────────────────
def test_analysis_signals(clicks_wav):
    sig = analyze_file(clicks_wav)
    assert set(sig) == {"envelope", "onsets", "low", "mid", "high"}
    assert all(len(v) == 320 for v in sig.values())
    peaks = np.flatnonzero(sig["onsets"] > 0.99) / FS
    clicks = 0.25 + 0.5 * np.arange(16)
    assert all(np.min(np.abs(peaks - c)) <= 0.05 for c in clicks)    # every click found
    assert sig["low"][:150].mean() > 2 * sig["low"][170:].mean()
    assert sig["high"][170:].mean() > 2 * sig["high"][:150].mean() + 0.1
    assert cached_analysis(clicks_wav) is sig                           # cached per file
    assert not sig["envelope"].flags.writeable


def test_analysis_of_silence_is_zero():
    sig = analyze_samples(np.zeros(SR * 2, np.float32), SR)
    assert all(np.all(v == 0) for v in sig.values())


# ── render: modulator tracks and routes ─────────────────────────────────────
FLAT = GeneratorSource("waveform", {"amp_speed": 0, "amp_min": 0.8, "amp_max": 0.8,
                                    "freq_speed": 0, "freq_min": 0.5, "freq_max": 0.5})


def mod_project(path, **route_kw):
    stim = Track(name="stim", clips=[Clip(FLAT, length_s=8.0)])
    mod = Track(name="mod", type="audio_mod",
                clips=[Clip(AudioFileSource(str(path)), length_s=8.0)])
    routes = [ModRoute(mod.id, target_track=stim.id, **route_kw)] if route_kw else []
    return Project(tracks=[stim, mod], mod_routes=routes), stim, mod


def test_modulator_track_is_silent_but_has_signals(clicks_wav):
    p, stim, mod = mod_project(clicks_wav)
    r = render_project(p)
    assert np.all(r.tracks[mod.id].pulses == 0)
    np.testing.assert_allclose(r.master[:, 0], 0.8, atol=1e-6)
    np.testing.assert_array_equal(r.signals[mod.id]["onsets"], analyze_file(clicks_wav)["onsets"])


@pytest.mark.parametrize("mode,depth", [("multiply", 1.0), ("multiply", 0.5), ("add", -0.5),
                                        ("replace", 1.0)])
def test_lane_route_modes(clicks_wav, mode, depth):
    p, stim, mod = mod_project(clicks_wav, signal="onsets", target="lane:amp_l", mode=mode,
                               depth=depth)
    s = analyze_file(clicks_wav)["onsets"].astype(np.float64)
    x = 0.8
    expected = {"multiply": x * (1 - depth + depth * s), "add": x + depth * s,
                "replace": x * (1 - depth) + depth * s}[mode]
    m = render_project(p).master
    np.testing.assert_allclose(m[:, 0], np.clip(expected, 0, 1), atol=1e-5)
    np.testing.assert_allclose(m[:, 1], 0.8, atol=1e-6)               # other lanes untouched


def test_route_invert_smooth_disable_and_mute(clicks_wav):
    p, stim, mod = mod_project(clicks_wav, signal="onsets", target="lane:amp_l", invert=True)
    m = render_project(p).master
    s = analyze_file(clicks_wav)["onsets"]
    assert m[np.argmax(s), 0] == pytest.approx(0.0, abs=1e-5)         # inverted: peak → 0
    p.mod_routes[0].invert = False
    raw = render_project(p).master[:, 0]
    p.mod_routes[0].smooth_s = 0.5
    smooth = render_project(p).master[:, 0]
    assert np.abs(np.diff(smooth)).max() < np.abs(np.diff(raw)).max() / 3
    p.mod_routes[0].enabled = False
    np.testing.assert_allclose(render_project(p).master[:, 0], 0.8, atol=1e-6)
    p.mod_routes[0].enabled = True
    mod.muted = True                                                   # muted modulator: no effect
    np.testing.assert_allclose(render_project(p).master[:, 0], 0.8, atol=1e-6)


def test_fx_parameter_route(clicks_wav):
    p, stim, mod = mod_project(clicks_wav, signal="envelope", target="fx:0:value",
                               mode="replace", depth=1.0)
    stim.effects = [get_effect("scale_amplitude").make()]
    env = analyze_file(clicks_wav)["envelope"].astype(np.float64)
    m = render_project(p).master
    np.testing.assert_allclose(m[:, 0], np.minimum(0.8 * env * 4.0, 1.0), atol=1e-4)


def test_missing_audio_file_renders_silent(tmp_path):
    p, stim, mod = mod_project(tmp_path / "gone.wav", signal="envelope", target="lane:amp_l")
    r = render_project(p)
    assert np.all(r.signals[mod.id]["envelope"] == 0)
    assert np.all(r.master[:, 0] == 0)                               # multiply by a silent signal


def test_audio_clip_on_standard_track_is_ignored(clicks_wav):
    t = Track(clips=[Clip(AudioFileSource(str(clicks_wav)), length_s=2.0)])
    tr = render_track(t, 80)
    assert np.all(tr.pulses[:, :2] == 0) and not tr.active.any()


def test_routes_and_audio_clips_survive_save(clicks_wav, tmp_path):
    p, stim, mod = mod_project(clicks_wav, signal="low", target="fx:0:depth", mode="add",
                               depth=-0.3, smooth_s=0.2, invert=True)
    save_project(p, tmp_path / "m.stimproj")
    q = load_project(tmp_path / "m.stimproj")
    assert q.to_dict() == p.to_dict()
    assert isinstance(q.tracks[1].clips[0].source, AudioFileSource)
    assert q.clone().mod_routes[0].to_dict() == p.mod_routes[0].to_dict()


# ── GUI ─────────────────────────────────────────────────────────────────────
def test_import_audio_as_lanes(window, stereo_wav):
    doc = window.doc
    track = doc.project.tracks[1]
    clip = window.import_audio(str(stereo_wav), start_s=60.0, track_id=track.id, mode="convert",
                               method="centroid")
    assert clip in track.clips and clip.start_s == 60.0 and clip.length_s == pytest.approx(6.0)
    assert clip.source.kind == "pulses" and clip.source.data.shape == (240, 4)
    assert doc.wait_for_render() and doc.last_render.n == 66 * FS
    doc.undo.undo()
    assert clip not in track.clips


def test_import_audio_as_modulator_and_route(window, clicks_wav):
    doc, ed = window.doc, window.editor
    n_tracks = len(doc.project.tracks)
    clip = window.import_audio(str(clicks_wav), start_s=0.0, mode="analyze")
    assert len(doc.project.tracks) == n_tracks + 1
    mod = doc.project.tracks[-1]
    assert mod.type == "audio_mod" and clip in mod.clips and clip.length_s == pytest.approx(8.0)
    assert ed.headers.headers[mod.id].badge.text() == "♪"
    assert doc.wait_for_render()
    ed.refresh()
    assert ed.clip_items[clip.id]._preview is not None
    ed.toggle_expanded(mod.id)
    signal_lanes = [it for (tid, _), it in ed.lane_items.items() if tid == mod.id]
    assert len(signal_lanes) == 5 and all(it.values is not None for it in signal_lanes)

    panel = window.modulation
    ed.set_current_track(doc.project.tracks[0].id)
    panel.add()
    route = doc.project.mod_routes[0]
    assert route.source_track == mod.id and route.target_track == doc.project.tracks[0].id
    row = panel.rows[route.id]
    n = doc.undo.count()
    row.depth.setValue(0.5)
    row.depth.setValue(0.25)
    assert doc.project.mod_routes[0].depth == 0.25 and doc.undo.count() == n + 1
    row.source.setCurrentIndex(row.source.findData(f"{mod.id}|onsets"))
    row._source_changed(0)
    assert doc.project.mod_routes[0].signal == "onsets"
    assert doc.wait_for_render()
    tid = doc.project.tracks[0].id
    modulated = doc.last_render.tracks[tid].pulses[:, 0].astype(np.float64)
    row.enabled.setChecked(False)
    assert doc.wait_for_render()
    plain = doc.last_render.tracks[tid].pulses[:, 0].astype(np.float64)
    s = np.zeros(len(plain))
    onsets = analyze_file(clicks_wav)["onsets"]
    s[:len(onsets)] = onsets[:len(plain)]
    np.testing.assert_allclose(modulated, np.clip(plain * (0.75 + 0.25 * s), 0, 1), atol=1e-5)
    panel.remove(route.id)
    assert doc.project.mod_routes == []


def test_audio_clips_stay_on_modulator_tracks(window, clicks_wav):
    doc, ed = window.doc, window.editor
    clip = window.import_audio(str(clicks_wav), start_s=0.0, mode="analyze")
    mod = doc.project.tracks[-1]
    ed.select_clips([clip.id])
    item = ed.clip_items[clip.id]
    dy = ed.layout.row(doc.project.tracks[0].id).y - ed.layout.row(mod.id).y
    origin = item.scenePos() + QPointF(5, 5)
    ed.begin_drag(item, "move", origin)
    ed.update_drag(origin + QPointF(0, dy), Qt.KeyboardModifier.NoModifier)
    ed.end_drag(Qt.KeyboardModifier.NoModifier)
    assert clip in mod.clips
    assert not window.a_save_preset.isEnabled() and not window.a_export_part.isEnabled()


def test_drop_audio_file_opens_import(window, clicks_wav, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "import_audio", lambda *a, **k: calls.append((a, k)))
    window.editor.audioDropped.disconnect()
    window.editor.audioDropped.connect(
        lambda path, t, tid: window.import_audio(path, start_s=t, track_id=tid or None))
    ed = window.editor
    row = ed.layout.row(window.doc.project.tracks[1].id)
    ed.drop_audio(str(clicks_wav), QPointF(10.0 * ed.pps, row.y + 5))
    (args, kw), = calls
    assert args[0] == str(clicks_wav) and kw["track_id"] == row.track_id
    assert kw["start_s"] == pytest.approx(ed.snap_time(10.0))   # grid + magnets
