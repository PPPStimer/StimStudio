"""Tools → Triphase audio: phase-coded three-phase stereo → pulse lanes."""

import numpy as np
import pytest
import soundfile as sf

from stimstudio.audio import triphase_decode as td
from stimstudio.core.pulse import FS
from stimstudio.core.triphase import (Calibration, output_phasors, phasors_to_tri,
                                      tri_to_pulses, unit_to_xy, xy_to_unit)
from stimstudio.outputs.audio import render_offline

SR = 44100
DUR = 6.0


def _path(sr=SR, dur=DUR):
    t = np.arange(int(sr * dur)) / sr
    vol = 0.4 + 0.4 * np.sin(2 * np.pi * 0.3 * t) ** 2
    x = 0.6 * np.cos(2 * np.pi * 0.4 * t)
    y = 0.6 * np.sin(2 * np.pi * 0.27 * t)
    return t, vol, x, y


def _continuous(sr=SR, dur=DUR, carrier=700.0):
    t, vol, x, y = _path(sr, dur)
    a, b = output_phasors(vol, x, y)
    c = np.exp(2j * np.pi * carrier * t)
    return np.stack([np.real(a * c), np.real(b * c)], 1).astype(np.float32), (vol, x, y)


def _at_pulses(arrs, n, sr=SR):
    idx = ((np.arange(n) + 0.5) / FS * sr).astype(int)
    return [a[idx] for a in arrs]


def _bursts(rate=25.0, carrier=1000.0, burst_s=0.004):
    t, vol, x, y = _path()
    env = np.zeros_like(t)
    w = np.hanning(int(SR * burst_s))
    for s in np.arange(0.0, DUR, 1.0 / rate):
        i = int(s * SR)
        env[i:i + len(w)] = w[:len(env) - i]
    a, b = output_phasors(vol, x, y)
    c = env * np.exp(2j * np.pi * carrier * t)
    return np.stack([np.real(a * c), np.real(b * c)], 1).astype(np.float32), (vol, x, y)


# ── inverse model ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("cal", [Calibration(), Calibration(center_x=0.1, center_y=-0.05,
                                                              gain_n=1.1, gain_l=0.95,
                                                              gain_r=1.05, reach=0.4)])
def test_phasors_to_tri_inverts_output_phasors(cal):
    rng = np.random.default_rng(3)
    r, th = np.sqrt(rng.random(500)) * 0.7, rng.random(500) * 2 * np.pi
    x, y, vol = r * np.cos(th), r * np.sin(th), 0.05 + 0.95 * rng.random(500)
    v2, x2, y2 = phasors_to_tri(*output_phasors(vol, x, y, cal), cal)
    np.testing.assert_allclose(v2, vol, atol=1e-6)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)


def test_phasors_to_tri_silence_is_centre():
    v, x, y = phasors_to_tri(np.zeros(3), np.zeros(3))
    assert np.all(v == 0) and np.all(x == 0) and np.all(y == 0)


# ── continuous carrier ──────────────────────────────────────────────────────
def test_continuous_round_trip():
    data, (vol, x, y) = _continuous()
    an = td.analyse(data, SR)
    assert an.n == int(np.ceil(DUR * FS)) and an.detected_mode == "continuous"
    assert an.carrier_median == pytest.approx(700.0, abs=0.5)
    assert an.mean_coherence > 0.99 and an.duty > 0.95
    vol_p, x_p, y_p = _at_pulses((vol, x, y), an.n)
    tri = td.to_pulses(an, output="triphase", normalize=False)
    inner = slice(2, -2)                                         # FFT edges
    np.testing.assert_allclose(tri[inner, 0], vol_p[inner], atol=0.01)
    np.testing.assert_allclose(unit_to_xy(tri[inner, 1]), x_p[inner], atol=0.02)
    np.testing.assert_allclose(unit_to_xy(tri[inner, 2]), y_p[inner], atol=0.02)
    ab = td.to_pulses(an, output="channels", normalize=False)
    ref = tri_to_pulses(np.stack([vol_p, xy_to_unit(x_p), xy_to_unit(y_p), vol_p], 1))
    np.testing.assert_allclose(ab[inner, :2], ref[inner, :2], atol=0.01)
    assert ab[10, 2] == pytest.approx((700 - 500) / 500, abs=0.01)  # carrier, 500..1000 Hz
    assert np.array_equal(ab[:, 2], ab[:, 3])


def test_electrodes_follow_the_currents():
    data, (vol, x, y) = _continuous()
    an = td.analyse(data, SR)
    el = td.to_pulses(an, output="electrodes", normalize=False)
    a, b = output_phasors(*_at_pulses((vol, x, y), an.n))
    full = td._electrode_full(Calibration())
    np.testing.assert_allclose(el[2:-2, 0], (np.abs(2 * a - b) / full)[2:-2], atol=0.01)
    np.testing.assert_allclose(el[2:-2, 1], (np.abs(2 * b - a) / full)[2:-2], atol=0.01)
    norm = td.to_pulses(an, output="electrodes")
    assert norm[:, :2].max() == pytest.approx(1.0, abs=1e-6)


def test_frequency_sources_and_options():
    data, _ = _continuous(carrier=800.0)
    an = td.analyse(data, SR)
    const = td.to_pulses(an, freq_source="constant", hz_range=(10, 110), const_hz=60)
    assert np.allclose(const[:, 2], 0.5)
    car = td.to_pulses(an, freq_source="carrier", hz_range=(600, 1000))
    assert np.allclose(car[5:-5, 2], 0.5, atol=0.01)
    bursts = td.to_pulses(an, freq_source="burst")                  # none: middle of range
    assert np.allclose(bursts[:, 2], 0.5)
    smooth = td.to_pulses(an, smooth_s=0.5)
    assert smooth.shape == (an.n, 4) and smooth.dtype == np.float32
    with pytest.raises(ValueError):
        td.to_pulses(an, output="nope")


def _sweep(lo=600.0, hi=900.0, dur=DUR):
    """Carrier gliding lo → hi → lo (one cosine period over the file)."""
    t = np.arange(int(SR * dur)) / SR
    hz = lo + (hi - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * t / dur))
    c = np.exp(1j * np.cumsum(2 * np.pi * hz / SR))
    a, b = output_phasors(0.8, 0.3, -0.2)
    return np.stack([np.real(a * c), np.real(b * c)], 1).astype(np.float32)


def test_fitted_range_spans_the_carrier():
    an = td.analyse(_sweep(), SR)
    lo, hi = td.fitted_hz_range(an, "carrier")
    assert lo == pytest.approx(600.0 - 37.5, abs=3.0) and hi == pytest.approx(900.0 + 37.5, abs=3.0)
    freq = td.to_pulses(an, freq_source="carrier", hz_range=(lo, hi))[:, 2]
    assert freq.min() == pytest.approx(0.1, abs=1e-4) and freq.max() == pytest.approx(0.9, abs=1e-4)
    steady = td.analyse(_continuous()[0], SR)                        # steady: still 0.1..0.9
    lo, hi = td.fitted_hz_range(steady, "carrier")
    assert lo < 700.0 < hi and hi - lo < 10.0
    freq = td.to_pulses(steady, freq_source="carrier", hz_range=(lo, hi))[:, 2]
    assert freq.min() == pytest.approx(0.1, abs=1e-4) and freq.max() == pytest.approx(0.9, abs=1e-4)
    flat = td.TriAudioAnalysis(**{**steady.__dict__,
                                  "carrier_hz": np.full(steady.n, 700.0, np.float32)})
    assert td.fitted_hz_range(flat, "carrier") == pytest.approx((699.375, 700.625))  # ≥ 1 Hz
    assert td.fitted_hz_range(steady, "burst") is None              # no bursts
    assert td.fitted_hz_range(steady, "constant") is None


def test_window_fits_the_carrier(window):
    win = window.show_tool("triphase_audio")
    win.freq_source.setCurrentIndex(win.freq_source.findData("carrier"))
    win.set_analysis("sweep.wav", td.analyse(_sweep(), SR))
    lo, hi = win.hz_lo.value(), win.hz_hi.value()
    assert lo == pytest.approx(562.5, abs=3.0) and hi == pytest.approx(937.5, abs=3.0)
    freq = win.pulses()[:, 2]
    assert freq.min() == pytest.approx(0.1, abs=1e-3) and freq.max() == pytest.approx(0.9, abs=1e-3)
    win.close()


def test_rendered_triphase_decodes_back():
    """Our own continuous-audio render of a triphase clip decodes to the same clip."""
    n = 4 * FS
    k = np.arange(n) / FS
    tri = np.zeros((n, 4), np.float32)
    tri[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * k)
    tri[:, 1] = xy_to_unit(0.5 * np.cos(2 * np.pi * 0.3 * k))
    tri[:, 2] = xy_to_unit(0.5 * np.sin(2 * np.pi * 0.3 * k))
    tri[:, 3] = 0.5                                                  # 750 Hz carrier
    sig = render_offline(np.zeros((n, 4), np.float32), "continuous", 48000, triphase=tri)
    an = td.analyse(sig, 48000)
    assert an.carrier_median == pytest.approx(750.0, abs=1.0)
    out = td.to_pulses(an, output="triphase", normalize=False)
    inner = slice(4, -4)
    np.testing.assert_allclose(out[inner, 1:3], tri[inner, 1:3], atol=0.02)
    ratio = out[inner, 0] / tri[inner, 0]                           # output power scaling
    assert ratio.std() < 0.02
    np.testing.assert_allclose(out[inner, 3], 0.5, atol=0.01)


# ── bursts ──────────────────────────────────────────────────────────────────
def test_bursts_detected_and_held():
    data, (vol, x, y) = _bursts(rate=25.0)
    an = td.analyse(data, SR)
    assert an.detected_mode == "pulsed" and an.duty < 0.3
    assert an.burst_median == pytest.approx(25.0, abs=0.5)
    assert an.carrier_median == pytest.approx(1000.0, abs=2.0)
    vol_p, x_p, _ = _at_pulses((vol, x, y), an.n)
    tri = td.to_pulses(an, output="triphase", normalize=False)
    inner = slice(4, -4)
    # no gaps between bursts, level = burst peak (not lowered by the duty cycle)
    np.testing.assert_allclose(tri[inner, 0], vol_p[inner], atol=0.03)
    # held from burst to burst: x moves up to 0.6·2π·0.4 ≈ 1.5 / s, ≤ 0.06 per burst
    np.testing.assert_allclose(unit_to_xy(tri[inner, 1]), x_p[inner], atol=0.07)
    rate = td.to_pulses(an, freq_source="burst", hz_range=(10, 40))
    assert np.allclose(rate[inner, 2], 0.5, atol=0.02)
    unbroken = td.to_pulses(an, mode="continuous", output="triphase", normalize=False)
    assert unbroken[inner, 0].mean() < 0.5 * tri[inner, 0].mean()  # RMS over the gaps


# ── inputs ──────────────────────────────────────────────────────────────────
def test_noise_is_not_coherent():
    rng = np.random.default_rng(1)
    an = td.analyse((rng.standard_normal((SR * 3, 2)) * 0.3).astype(np.float32), SR)
    assert an.mean_coherence < 0.2


def test_mono_and_empty_rejected():
    with pytest.raises(ValueError):
        td.analyse(np.zeros((1000, 1), np.float32), SR)
    with pytest.raises(ValueError):
        td.analyse(np.zeros((0, 2), np.float32), SR)


def test_silence_is_quiet_centre():
    an = td.analyse(np.zeros((SR * 2, 2), np.float32), SR)
    out = td.to_pulses(an, output="triphase")
    assert np.all(out[:, 0] == 0) and np.allclose(out[:, 1:3], 0.5)


@pytest.mark.parametrize("sr", [44100, 48000, 22050])
def test_pulse_count_independent_of_rate(sr):
    data, _ = _continuous(sr=sr, dur=2.5)
    assert td.analyse(data, sr).n == 100


def test_analyse_file_cached(tmp_path):
    data, _ = _continuous(dur=2.0)
    path = tmp_path / "tri.wav"
    sf.write(str(path), data, SR, subtype="FLOAT")
    stages = []
    an = td.analyse_file(str(path), lambda s, f: stages.append(s))
    assert "Demodulating" in stages and an.n == 80
    assert td.analyse_file(str(path)) is an
    with pytest.raises(td.Cancelled):
        td.analyse(data, SR, cancel=lambda: True)


# ── GUI ─────────────────────────────────────────────────────────────────────
def test_tool_window(window, tmp_path):
    data, _ = _continuous(dur=3.0)
    path = tmp_path / "three.wav"
    sf.write(str(path), data, SR, subtype="FLOAT")
    titles = {a.text().replace("&", ""): a for a in window.menuBar().actions()}
    texts = [a.text().replace("&", "") for a in titles["Tools"].menu().actions()]
    assert any(t.startswith("Triphase audio") for t in texts)
    win = window.show_tool("triphase_audio")
    assert win.analysis is None and not win.create_btn.isEnabled()
    assert win.open_file(str(path))
    assert win.create_btn.isEnabled() and "carrier ≈ 700 Hz" in win.info.text()
    assert "Auto (unbroken carrier)" in win.signal.currentText()
    assert win.freq_source.currentData() == "carrier"
    fitted = (win.hz_lo.value(), win.hz_hi.value())
    assert fitted[0] < 700.0 < fitted[1]
    freq = win.pulses()[:, 2]
    assert freq.min() == pytest.approx(0.1, abs=0.01) and freq.max() == pytest.approx(0.9, abs=0.01)

    doc = window.doc
    doc.set_cursor(1.5)
    n_tracks = len(doc.project.tracks)
    win.output.setCurrentIndex(win.output.findData("channels"))
    clip = win.create_clip()
    track, found = doc.project.find_clip(clip.id)
    assert track.type == "standard" and found.start_s == 1.5
    assert found.length_s == pytest.approx(3.0) and found.source.origin == "3φ audio: three.wav"
    assert len(doc.project.tracks) == n_tracks + 1

    win.output.setCurrentIndex(win.output.findData("triphase"))
    assert win.plot.plotItem.legend.getLabel(win.curves[1]).text == "Position X"
    clip = win.create_clip()
    track, _ = doc.project.find_clip(clip.id)
    assert track.type == "triphase" and track.name == "three (3φ)"
    doc.undo.undo()                                                  # one step per clip
    assert len(doc.project.tracks) == n_tracks + 1

    win.freq_source.setCurrentIndex(win.freq_source.findData("burst"))
    assert (win.hz_lo.value(), win.hz_hi.value()) == td.default_hz_range("burst")
    assert win.const_hz.isEnabled() is False
    win.freq_source.setCurrentIndex(win.freq_source.findData("carrier"))
    assert (win.hz_lo.value(), win.hz_hi.value()) == fitted
    win.freq_source.setCurrentIndex(win.freq_source.findData("burst"))
    win.close()
    again = window.show_tool("triphase_audio")
    assert again is win and win.freq_source.currentData() == "burst"
    from stimstudio.gui.tools.triphase_audio import TriphaseAudioWindow
    fresh = TriphaseAudioWindow(window)                              # options remembered
    assert fresh.output.currentData() == "triphase" and fresh.freq_source.currentData() == "burst"
    fresh.close()


def test_mono_file_reports_error(window, tmp_path, monkeypatch):
    path = tmp_path / "mono.wav"
    sf.write(str(path), np.zeros(SR, np.float32), SR)
    shown = []
    from stimstudio.gui.tools import triphase_audio as ta
    monkeypatch.setattr(ta.QMessageBox, "warning", lambda *a: shown.append(a[2]))
    win = window.show_tool("triphase_audio")
    assert not win.open_file(str(path))
    assert shown and "stereo" in shown[0] and win.analysis is None
    win.close()


def test_add_triphase_audio_ignores_empty(window):
    assert window.add_triphase_audio({"pulses": None}) is None
    assert window.add_triphase_audio({"pulses": np.zeros((0, 4))}) is None
