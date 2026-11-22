"""File → Import audio: three excerpts → three-phase or two-channel, then the choice."""

import random

import numpy as np
import pytest
import soundfile as sf

from stimstudio.audio import triphase_probe as tp
from stimstudio.core.triphase import output_phasors

SR = 8000


def _tri(dur, carrier=700.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    x, y = 0.5 * np.cos(2 * np.pi * 0.2 * t), 0.5 * np.sin(2 * np.pi * 0.13 * t)
    a, b = output_phasors(0.8, x, y)
    c = np.exp(2j * np.pi * carrier * t)
    return np.stack([np.real(a * c), np.real(b * c)], 1).astype(np.float32)


def _tone(dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return (np.sin(2 * np.pi * 700 * t) * (0.5 + 0.4 * np.sin(2 * np.pi * 0.5 * t))) \
        .astype(np.float32)


def _probe(data):
    return tp.probe([data], SR, len(data) / SR, [(0.0, len(data) / SR)])


# ── excerpts ────────────────────────────────────────────────────────────────
def test_pick_segments():
    segs = tp.pick_segments(600.0, rng=random.Random(4))
    assert len(segs) == 3 and all(n == 30.0 for _, n in segs)
    for i, (s, n) in enumerate(segs):                               # one in each third
        assert 200.0 * i <= s and s + n <= 200.0 * (i + 1)
    assert tp.pick_segments(80.0) == [(0.0, 80.0)]                    # short: whole file
    assert tp.pick_segments(600.0, rng=random.Random(1)) != tp.pick_segments(
        600.0, rng=random.Random(2))                                  # random places


def test_probe_file_reads_only_excerpts(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(str(path), _tri(200.0), SR, subtype="FLOAT")
    stages = []
    res = tp.probe_file(str(path), lambda s, f: stages.append(s), rng=random.Random(0),
                        length_s=20.0)
    assert res.is_triphase and res.duration_s == pytest.approx(200.0)
    assert len(res.segments) == 3 and "Reading excerpts" in stages
    assert res.carrier_hz == pytest.approx(700.0, abs=1.0)


# ── verdicts ────────────────────────────────────────────────────────────────
def test_three_phase_recognised():
    res = _probe(_tri(20.0))
    assert res.kind == "triphase" and res.coherence > 0.99 and res.off_axis > 0.9
    assert 15 < res.phase_deg < 165
    text = " ".join(res.reasons)
    assert "one signal" in text and "third electrode" in text and "700 Hz" in text


@pytest.mark.parametrize("right,word", [(1.0, "in phase"), (-0.7, "in antiphase")])
def test_same_signal_both_sides_is_two_channel(right, word):
    s = _tone(20.0)
    res = _probe(np.stack([s, right * s], 1))
    assert res.kind == "stereo" and res.coherence > 0.99
    assert word in " ".join(res.reasons)


def test_unrelated_channels_are_two_channel():
    t = np.arange(SR * 20) / SR
    res = _probe(np.stack([np.sin(2 * np.pi * 700 * t),
                           np.sin(2 * np.pi * 820 * t)], 1).astype(np.float32))
    assert res.kind == "stereo" and res.coherence < 0.5
    assert "different signals" in " ".join(res.reasons)
    assert not any("Carrier" in r for r in res.reasons)          # no carrier to speak of


def test_silent_channel_and_silence_and_mono():
    s = _tone(10.0)
    one = _probe(np.stack([s, np.zeros_like(s)], 1))
    assert one.kind == "stereo" and "right channel is (nearly) silent" in " ".join(one.reasons)
    quiet = _probe(np.zeros((SR * 5, 2), np.float32))
    assert quiet.kind == "stereo" and quiet.silent
    mono = _probe(s[:, None])
    assert mono.kind == "stereo" and mono.channels == 1 and "Mono" in mono.reasons[0]


# ── dialog ──────────────────────────────────────────────────────────────────
def test_dialog_shows_verdict_and_two_equal_buttons(qapp):
    from stimstudio.gui.audio_import import AudioImportDialog
    res = _probe(_tri(20.0))
    res.segments = [(10.0, 30.0), (200.0, 30.0), (400.0, 30.0)]
    dlg = AudioImportDialog("x.wav", res)
    dlg.show()
    qapp.processEvents()
    assert "three-phase" in dlg.verdict.text() and "third electrode" in dlg.reasons.text()
    assert dlg.tri_btn.isDefault() and not dlg.two_btn.isDefault()
    assert dlg.tri_btn.width() == dlg.two_btn.width()
    assert dlg.tri_btn.height() == dlg.two_btn.height()
    dlg.two_btn.click()
    assert dlg.result() == dlg.DialogCode.Accepted and dlg.mode == "convert"

    stereo = AudioImportDialog("y.wav", _probe(np.stack([_tone(5.0)] * 2, 1)))
    assert "two-channel" in stereo.verdict.text() and stereo.two_btn.isDefault()
    stereo.mod_btn.click()
    assert stereo.mode == "analyze"
    mono = AudioImportDialog("m.wav", _probe(_tone(5.0)[:, None]))
    assert not mono.tri_btn.isEnabled()
    for d in (dlg, stereo, mono):
        d.close()


# ── File → Import audio ─────────────────────────────────────────────────────
def _answer(monkeypatch, button):
    """Make the import dialog click *button* instead of waiting for the user."""
    from stimstudio.gui import audio_import
    seen = []

    def exec_(self):
        seen.append(self)
        getattr(self, button).click()
        return self.result()
    monkeypatch.setattr(audio_import.AudioImportDialog, "exec", exec_)
    return seen


def test_import_three_phase_opens_the_tool(window, tmp_path, monkeypatch):
    path = tmp_path / "three.wav"
    sf.write(str(path), _tri(4.0), SR, subtype="FLOAT")
    seen = _answer(monkeypatch, "tri_btn")
    n = len(window.doc.project.tracks)
    assert window.import_audio(str(path), start_s=12.0) is None
    assert seen and seen[0].probe.is_triphase
    win = window._tool_windows["triphase_audio"]
    assert win.isVisible() and win.path == str(path) and win.start_s == 12.0
    assert len(window.doc.project.tracks) == n                       # nothing yet
    clip = win.create_clip()
    assert window.doc.project.find_clip(clip.id)[1].start_s == 12.0   # at the drop point
    win.close()


def test_import_two_channel_converts(window, tmp_path, monkeypatch):
    path = tmp_path / "two.wav"
    s = _tone(4.0)
    sf.write(str(path), np.stack([s, s], 1), SR)
    seen = _answer(monkeypatch, "two_btn")
    clip = window.import_audio(str(path), start_s=3.0)
    assert not seen[0].probe.is_triphase
    assert clip is not None and clip.start_s == 3.0 and clip.source.kind == "pulses"


def test_drop_on_modulator_track_skips_the_dialog(window, tmp_path, monkeypatch):
    path = tmp_path / "mod.wav"
    sf.write(str(path), np.stack([_tone(3.0)] * 2, 1), SR)
    seen = _answer(monkeypatch, "two_btn")
    first = window.import_audio(str(path), start_s=0.0, mode="analyze")
    track = window.doc.project.find_clip(first.id)[0]
    clip = window.import_audio(str(path), start_s=5.0, track_id=track.id)
    assert not seen and window.doc.project.find_clip(clip.id)[0] is track


def test_import_cancelled_in_the_dialog(window, tmp_path, monkeypatch):
    path = tmp_path / "c.wav"
    sf.write(str(path), np.stack([_tone(2.0)] * 2, 1), SR)
    from stimstudio.gui import audio_import
    monkeypatch.setattr(audio_import.AudioImportDialog, "exec", lambda self: 0)
    n = len(window.doc.project.tracks)
    assert window.import_audio(str(path)) is None
    assert len(window.doc.project.tracks) == n
